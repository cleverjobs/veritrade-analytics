import type { PolicyName, Sentiment, SignalResult } from "@/lib/api";
import { capitalize, formatMs, formatPct } from "@/lib/format";
import type { LaneStatus } from "@/lib/run-state";

import styles from "./SignalCard.module.css";

const POLICY_NOTES: Record<PolicyName, string> = {
  probability: "threshold on the decision model's class probability",
  "label-logprob": "threshold on the label token probability",
  ordinal: "threshold on |score| of the -2..+2 scale",
};

const ORDER: readonly Sentiment[] = ["bullish", "bearish", "neutral"];

/** Direction the model leaned before the threshold turned it into an abstention. */
export function leaning(signal: SignalResult): Sentiment {
  if (signal.probabilities) {
    let best: Sentiment = signal.sentiment;
    for (const s of ORDER) {
      if ((signal.probabilities[s] ?? 0) > (signal.probabilities[best] ?? 0)) best = s;
    }
    return best;
  }
  if (typeof signal.score === "number" && signal.score !== 0) return signal.score > 0 ? "bullish" : "bearish";
  return signal.sentiment;
}

function formatScore(score: number): string {
  return score > 0 ? `+${score}` : String(score);
}

interface SignalCardProps {
  status: LaneStatus;
  signal: SignalResult | null;
  entity?: string | null;
  model?: string | null;
  /** Client-measured time from submit to the signal arriving. */
  timeToSignalMs?: number | null;
  /** Server-reported time to sentiment on a combined stream. */
  serverMs?: number | null;
}

export function SignalCard({ status, signal, entity, model, timeToSignalMs, serverMs }: SignalCardProps) {
  const busy = !signal && (status === "pending" || status === "streaming");
  return (
    <section className={styles.card} aria-labelledby="signal-heading" aria-busy={busy}>
      <h2 id="signal-heading" className={styles.heading}>
        Signal
      </h2>
      {signal ? (
        <SignalBody
          signal={signal}
          entity={entity}
          model={model}
          timeToSignalMs={timeToSignalMs}
          serverMs={serverMs}
        />
      ) : (
        <p className={styles.placeholder}>
          {busy
            ? "Waiting for the signal..."
            : status === "cancelled"
              ? "Cancelled."
              : "No signal: the request failed."}
        </p>
      )}
    </section>
  );
}

type SignalBodyProps = Omit<SignalCardProps, "status" | "signal"> & { signal: SignalResult };

function SignalBody({ signal, entity, model, timeToSignalMs, serverMs }: SignalBodyProps) {
  const lean = leaning(signal);
  return (
    <>
      <div className={styles.row}>
        <span
          className={styles.badge}
          data-sentiment={signal.sentiment}
          data-abstained={signal.abstained || undefined}
          data-testid="signal-badge"
        >
          {capitalize(signal.sentiment)}
        </span>
        {signal.abstained ? <span className={styles.abstainTag}>Abstained</span> : null}
        {entity ? <span className={styles.entity}>{entity}</span> : null}
      </div>
      {signal.abstained ? (
        <p className={styles.abstainNote}>
          Below confidence threshold: the call is not confident enough to trade on, so treat it as no
          directional signal.
          {lean !== signal.sentiment ? ` The model leaned ${capitalize(lean)}.` : null}
        </p>
      ) : null}
      <dl className={styles.stats}>
        {typeof signal.confidence === "number" ? (
          <div>
            <dt>Confidence</dt>
            <dd>{formatPct(signal.confidence)}</dd>
          </div>
        ) : null}
        {typeof signal.score === "number" ? (
          <div>
            <dt>Score</dt>
            <dd>{formatScore(signal.score)} on -2..+2</dd>
          </div>
        ) : null}
        <div>
          <dt>Policy</dt>
          <dd>
            <code>{signal.policy}</code> <span className={styles.note}>({POLICY_NOTES[signal.policy]})</span>
          </dd>
        </div>
        {model ? (
          <div>
            <dt>Model</dt>
            <dd>
              <code>{model}</code>
            </dd>
          </div>
        ) : null}
        {typeof timeToSignalMs === "number" ? (
          <div>
            <dt>Time to signal</dt>
            <dd>
              {formatMs(timeToSignalMs)}
              {typeof serverMs === "number" ? (
                <span className={styles.note}> (server {formatMs(serverMs)})</span>
              ) : null}
            </dd>
          </div>
        ) : null}
      </dl>
      {signal.probabilities ? (
        <ul className={styles.probs} aria-label="Class probabilities">
          {ORDER.map((s) => {
            const p = signal.probabilities?.[s] ?? 0;
            return (
              <li key={s} className={styles.prob}>
                <span className={styles.probLabel}>{capitalize(s)}</span>
                <span className={styles.probTrack} aria-hidden="true">
                  <span
                    className={styles.probFill}
                    data-sentiment={s}
                    style={{ width: `${Math.max(0, Math.min(1, p)) * 100}%` }}
                  />
                </span>
                <span className={styles.probValue}>{formatPct(p)}</span>
              </li>
            );
          })}
        </ul>
      ) : null}
    </>
  );
}
