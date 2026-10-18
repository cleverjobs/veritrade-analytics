"use client";

import { useAnalysis } from "@/hooks/useAnalysis";
import { useModels } from "@/hooks/useModels";
import { capitalize } from "@/lib/format";
import {
  articleMeta,
  distinctErrors,
  laneById,
  type Lane,
  type LaneId,
  type RunState,
} from "@/lib/run-state";

import { AnalyzeForm } from "./AnalyzeForm";
import styles from "./AnalyzeApp.module.css";
import { ArticleMetaLine } from "./ArticleMetaLine";
import { ErrorBanner } from "./ErrorBanner";
import { LatencyBar, type LaneMark, type LatencyLane } from "./LatencyBar";
import { SignalCard } from "./SignalCard";
import { SummaryPanel } from "./SummaryPanel";

const LANE_LABELS: Record<LaneId, string> = { signal: "Signal", summary: "Summary", analyze: "Analyze" };

function toLatencyLane(lane: Lane): LatencyLane {
  const marks: LaneMark[] = [];
  if (lane.signalAtMs !== null) {
    marks.push({ kind: "signal", label: "time to signal", atMs: lane.signalAtMs });
  }
  if (lane.firstTokenAtMs !== null) {
    marks.push({ kind: "token", label: "time to first token", atMs: lane.firstTokenAtMs });
  }
  return {
    id: lane.id,
    label: LANE_LABELS[lane.id],
    status: lane.status,
    timings: lane.timings,
    endMs: lane.endAtMs,
    marks,
  };
}

function announcement(state: RunState): string {
  if (state.running && !state.signal) return "Analyzing...";
  const parts: string[] = [];
  if (state.signal) {
    parts.push(`Signal ${capitalize(state.signal.sentiment)}${state.signal.abstained ? ", abstained" : ""}.`);
  }
  if (!state.running && state.summary) parts.push("Summary complete.");
  return parts.join(" ");
}

export function AnalyzeApp() {
  const models = useModels();
  const { state, start, cancel } = useAnalysis();

  const split = state.mode === "split";
  const signalLane = laneById(state, split ? "signal" : "analyze");
  const summaryLane = laneById(state, split ? "summary" : "analyze");
  const meta = articleMeta(state);
  const errors = distinctErrors(state);
  const showLatency = state.lanes.some(
    (l) => l.timings || l.signalAtMs !== null || l.firstTokenAtMs !== null,
  );

  return (
    <>
      <AnalyzeForm models={models} running={state.running} onSubmit={start} onCancel={cancel} />

      <p role="status" className="visually-hidden">
        {announcement(state)}
      </p>

      {models.status === "error" ? <ErrorBanner error={models.error} context="Model list" /> : null}
      {errors.map(({ lanes, error }) => (
        <ErrorBanner
          key={`${error.code}-${lanes.join("-")}`}
          error={error}
          context={lanes.map((id) => LANE_LABELS[id]).join(" and ")}
        />
      ))}

      {state.lanes.length ? (
        <div className={styles.results}>
          {meta ? <ArticleMetaLine meta={meta} /> : null}
          <div className={styles.cards}>
            {signalLane ? (
              <SignalCard
                status={signalLane.status}
                signal={state.signal}
                entity={state.entity}
                model={signalLane.meta?.model}
                timeToSignalMs={signalLane.signalAtMs}
                serverMs={state.serverSignalMs}
              />
            ) : null}
            {summaryLane ? (
              <SummaryPanel
                status={summaryLane.status}
                text={state.summary}
                truncated={state.summaryTruncated}
                model={summaryLane.meta?.model}
                streaming={state.streaming}
                firstTokenMs={summaryLane.firstTokenAtMs}
              />
            ) : null}
          </div>
          {showLatency ? <LatencyBar lanes={state.lanes.map(toLatencyLane)} /> : null}
        </div>
      ) : (
        <p className={styles.empty}>
          Results appear here. Split mode shows the signal as soon as /api/signal answers while the summary
          keeps streaming; combined mode gets both from one generation.
        </p>
      )}
    </>
  );
}
