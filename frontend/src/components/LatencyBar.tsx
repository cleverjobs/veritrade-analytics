import type { Timings } from "@/lib/api";
import { formatMs } from "@/lib/format";
import type { LaneStatus } from "@/lib/run-state";

import styles from "./LatencyBar.module.css";

export type StageKey =
  "fetch" | "consent" | "parse" | "extract" | "load" | "prompt" | "generation" | "inference" | "other";

export const STAGE_LABELS: Record<StageKey, string> = {
  fetch: "fetch",
  consent: "consent",
  parse: "parse",
  extract: "extract",
  load: "model load",
  prompt: "prompt eval",
  generation: "generation",
  inference: "inference",
  other: "network + overhead",
};

export interface Segment {
  key: StageKey;
  ms: number;
  startMs: number;
}

// Server stages laid end to end; whatever the client waited beyond them is "other". Decision
// models report only wall time, so they get a single inference segment.
export function stageSegments(timings: Timings, clientTotalMs: number | null): Segment[] {
  const segments: Segment[] = [];
  let cursor = 0;
  const push = (key: StageKey, ms: number | null | undefined) => {
    if (typeof ms !== "number" || ms <= 0) return;
    segments.push({ key, ms, startMs: cursor });
    cursor += ms;
  };
  push("fetch", timings.fetch_ms);
  push("consent", timings.consent_ms);
  push("parse", timings.parse_ms);
  push("extract", timings.extract_ms);
  const inference = timings.inference;
  if (inference) {
    const { load_ms, prompt_eval_ms, eval_ms } = inference;
    const split = [load_ms, prompt_eval_ms, eval_ms].some((ms) => typeof ms === "number");
    if (split) {
      push("load", load_ms);
      push("prompt", prompt_eval_ms);
      push("generation", eval_ms);
    } else {
      push("inference", inference.wall_ms);
    }
  }
  push("other", (clientTotalMs ?? timings.total_ms) - cursor);
  return segments;
}

export function niceStep(raw: number): number {
  if (raw <= 0) return 1;
  const pow = 10 ** Math.floor(Math.log10(raw));
  const n = raw / pow;
  const mult = n <= 1 ? 1 : n <= 2 ? 2 : n <= 2.5 ? 2.5 : n <= 5 ? 5 : 10;
  return mult * pow;
}

export interface LaneMark {
  kind: "signal" | "token";
  label: string;
  atMs: number;
}

export interface LatencyLane {
  id: string;
  label: string;
  status: LaneStatus;
  timings: Timings | null;
  /** Client-observed completion, ms since submit. */
  endMs: number | null;
  marks: LaneMark[];
}

function laneExtent(lane: LatencyLane): number {
  const markMax = Math.max(0, ...lane.marks.map((m) => m.atMs));
  return Math.max(lane.endMs ?? lane.timings?.total_ms ?? 0, markMax);
}

const pct = (ms: number, axis: number) => `${Math.min(100, (ms / axis) * 100)}%`;

export function LatencyBar({ lanes }: { lanes: LatencyLane[] }) {
  const axis = Math.max(1, ...lanes.map(laneExtent));
  const step = niceStep(axis / 4);
  const ticks = Array.from({ length: Math.floor(axis / step) + 1 }, (_, i) => i * step);
  const laneSegments = lanes.map((lane) => (lane.timings ? stageSegments(lane.timings, lane.endMs) : []));
  const usedKeys = new Set(laneSegments.flat().map((s) => s.key));

  return (
    <section className={styles.wrap} aria-labelledby="latency-heading" data-testid="latency">
      <h2 id="latency-heading" className={styles.heading}>
        Latency
      </h2>
      <p className={styles.caption}>
        Server stage timings placed on one client time axis, measured from submit with performance.now().
      </p>
      <div className={styles.chart}>
        {lanes.map((lane, i) => (
          <div key={lane.id} className={styles.lane} data-testid={`lane-${lane.id}`}>
            <div className={styles.laneLabel}>{lane.label}</div>
            <div className={styles.track} aria-hidden="true">
              {laneSegments[i].map((segment) => (
                <span
                  key={segment.key}
                  className={styles.segment}
                  data-stage={segment.key}
                  data-testid={`segment-${lane.id}-${segment.key}`}
                  style={{ left: pct(segment.startMs, axis), width: pct(segment.ms, axis) }}
                  title={`${STAGE_LABELS[segment.key]} ${formatMs(segment.ms)}`}
                />
              ))}
              {lane.marks.map((mark) => (
                <span
                  key={mark.kind}
                  className={styles.mark}
                  data-kind={mark.kind}
                  data-testid={`mark-${lane.id}-${mark.kind}`}
                  style={{ left: pct(mark.atMs, axis) }}
                  title={`${mark.label} ${formatMs(mark.atMs)}`}
                />
              ))}
              {!lane.timings && (lane.status === "pending" || lane.status === "streaming") ? (
                <span className={styles.running}>running...</span>
              ) : null}
            </div>
          </div>
        ))}
        <div className={styles.axis} aria-hidden="true">
          <div />
          <div className={styles.ticks}>
            {ticks.map((t) => (
              <span key={t} className={styles.tick} style={{ left: pct(t, axis) }}>
                {formatMs(t)}
              </span>
            ))}
          </div>
        </div>
      </div>

      <ul className={styles.legend} aria-hidden="true">
        {[...usedKeys].map((key) => (
          <li key={key}>
            <span className={styles.swatch} data-stage={key} />
            {STAGE_LABELS[key]}
          </li>
        ))}
        {lanes.some((l) => l.marks.some((m) => m.kind === "signal")) ? (
          <li>
            <span className={styles.markSwatch} data-kind="signal" />
            signal arrived
          </li>
        ) : null}
        {lanes.some((l) => l.marks.some((m) => m.kind === "token")) ? (
          <li>
            <span className={styles.markSwatch} data-kind="token" />
            first token
          </li>
        ) : null}
      </ul>

      <div className={styles.details}>
        {lanes.map((lane, i) => (
          <div key={lane.id} className={styles.detail}>
            <h3 className={styles.detailHeading}>{lane.label}</h3>
            <dl className={styles.values}>
              {lane.marks.map((mark) => (
                <div key={mark.kind}>
                  <dt>{mark.label}</dt>
                  <dd>{formatMs(mark.atMs)}</dd>
                </div>
              ))}
              {lane.endMs !== null ? (
                <div>
                  <dt>complete (client)</dt>
                  <dd>{formatMs(lane.endMs)}</dd>
                </div>
              ) : null}
              {lane.timings ? (
                <div>
                  <dt>server total</dt>
                  <dd>{formatMs(lane.timings.total_ms)}</dd>
                </div>
              ) : null}
              {typeof lane.timings?.inference?.ttft_ms === "number" ? (
                <div>
                  <dt>TTFT (server)</dt>
                  <dd>{formatMs(lane.timings.inference.ttft_ms)}</dd>
                </div>
              ) : null}
              {typeof lane.timings?.inference?.cold_start === "boolean" ? (
                <div>
                  <dt>cold start</dt>
                  <dd>{lane.timings.inference.cold_start ? "yes, model loaded" : "no, already loaded"}</dd>
                </div>
              ) : null}
              {laneSegments[i].map((segment) => (
                <div key={segment.key}>
                  <dt>{STAGE_LABELS[segment.key]}</dt>
                  <dd>{formatMs(segment.ms)}</dd>
                </div>
              ))}
              {lane.timings ? (
                <div>
                  <dt>cache</dt>
                  <dd>
                    fetch {lane.timings.cache.fetch}, result {lane.timings.cache.result}
                  </dd>
                </div>
              ) : null}
            </dl>
          </div>
        ))}
      </div>
    </section>
  );
}
