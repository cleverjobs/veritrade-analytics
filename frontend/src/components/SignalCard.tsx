import type { PolicyName, Sentiment, SignalResult } from "@/lib/api";
import { capitalize, coldStartNote, formatMs, formatPct, missDigits } from "@/lib/format";
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

/** Decimals for the card's percentages: extra ones only when a missed threshold or tau would print as met. */
export function pctDigits(signal: SignalResult): number {
  const probs = signal.probabilities;
  if (!probs) return 1;
  const lean = leaning(signal);
  let digits = 1;
  if (signal.abstained && typeof signal.threshold === "number") {
    digits = missDigits(probs[lean] ?? 0, signal.threshold);
  }
  if (lean === "neutral" && signal.sentiment === "neutral" && typeof signal.directional_tau === "number") {
    const side = Math.max(probs.bullish ?? 0, probs.bearish ?? 0);
    digits = Math.max(digits, missDigits(side, signal.directional_tau));
  }
  return digits;
}

// Non-breaking, so a wrapped line never strands the sign away from its number.
const AT_LEAST = "≥ ";

/** The frozen rule this call was held to, in words; null for payloads that predate thresholds. */
export function thresholdText(signal: SignalResult): string | null {
  const { threshold, directional_tau: tau } = signal;
  if (typeof threshold !== "number") return null;
  if (signal.policy === "ordinal") return `|score| ${AT_LEAST}${threshold}`;
  const hasTau = typeof tau === "number";
  if (threshold <= 0 && !hasTau) return "none (top class wins)";
  const digits = pctDigits(signal);
  const base = threshold > 0 ? `call needs ${AT_LEAST}${formatPct(threshold, digits)}` : "top class wins";
  return hasTau
    ? `${base}; bullish/bearish called at ${AT_LEAST}${formatPct(tau, digits)} even when neutral leads`
    : base;
}

/** Why the call abstained, with the numbers when the payload carries the threshold. */
export function abstainReason(signal: SignalResult): string {
  const { threshold, directional_tau: tau } = signal;
  const lean = leaning(signal);
  if (typeof threshold === "number") {
    if (signal.policy === "ordinal" && typeof signal.score === "number") {
      return `|score| ${Math.abs(signal.score)} (leaning ${capitalize(lean)}) is below this model's threshold of ${threshold}`;
    }
    const p = signal.probabilities?.[lean];
    if (typeof p === "number") {
      const pct = (value: number) => formatPct(value, pctDigits(signal));
      const reason = `the top class (${capitalize(lean)}, ${pct(p)}) is below this model's ${pct(threshold)} threshold`;
      // tau only rescues a neutral lead, so it is part of the reason only then.
      return lean === "neutral" && typeof tau === "number"
        ? `${reason}, and neither Bullish nor Bearish reached ${pct(tau)}`
        : reason;
    }
  }
  const leaned = lean === "neutral" ? "" : ` (it leaned ${capitalize(lean)})`;
  return `the call is below this model's confidence threshold${leaned}`;
}

function formatScore(score: number): string {
  return score > 0 ? `+${score}` : String(score);
}

// Rounded to 0.1% so float noise (0.28 * 100) never reaches the style attribute.
const share = (value: number) => `${Math.round(Math.max(0, Math.min(1, value)) * 1000) / 10}%`;

interface SignalCardProps {
  status: LaneStatus;
  signal: SignalResult | null;
  entity?: string | null;
  model?: string | null;
  /** Client-measured time from submit to the signal arriving. */
  timeToSignalMs?: number | null;
  /** Server-reported time to sentiment on a combined stream. */
  serverMs?: number | null;
  /** The model had to be loaded for this request. */
  coldStart?: boolean | null;
  /** Ollama's model load time; decision models report none. */
  loadMs?: number | null;
}

export function SignalCard({ status, signal, coldStart, ...rest }: SignalCardProps) {
  const busy = !signal && (status === "pending" || status === "streaming");
  return (
    <section className={styles.card} aria-labelledby="signal-heading" aria-busy={busy}>
      <h2 id="signal-heading" className={styles.heading}>
        Signal
      </h2>
      {signal ? (
        <SignalBody signal={signal} coldStart={coldStart} {...rest} />
      ) : (
        <p className={styles.placeholder}>
          {busy
            ? coldStart
              ? "Waiting for the signal while the model loads into memory (cold start)..."
              : "Waiting for the signal..."
            : status === "cancelled"
              ? "Cancelled."
              : "No signal: the request failed."}
        </p>
      )}
    </section>
  );
}

type SignalBodyProps = Omit<SignalCardProps, "status" | "signal"> & { signal: SignalResult };

function SignalBody({ signal, entity, model, timeToSignalMs, serverMs, coldStart, loadMs }: SignalBodyProps) {
  const rule = thresholdText(signal);
  const digits = pctDigits(signal);
  const thresholdMark =
    signal.policy !== "ordinal" && typeof signal.threshold === "number" && signal.threshold > 0
      ? signal.threshold
      : null;
  const tauMark = typeof signal.directional_tau === "number" ? signal.directional_tau : null;
  return (
    <>
      <div className={styles.row}>
        <span
          className={styles.badge}
          data-sentiment={signal.sentiment}
          data-abstained={signal.abstained || undefined}
          data-testid="signal-badge"
        >
          {signal.abstained ? "No signal" : capitalize(signal.sentiment)}
        </span>
        {entity ? <span className={styles.entity}>{entity}</span> : null}
      </div>
      {signal.abstained ? (
        <p className={styles.abstainNote} data-testid="abstain-note">
          Abstained: {abstainReason(signal)}, so there is no directional call.
        </p>
      ) : null}
      <dl className={styles.stats}>
        {typeof signal.confidence === "number" ? (
          <div>
            <dt>Confidence</dt>
            <dd>{formatPct(signal.confidence, digits)}</dd>
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
        {rule ? (
          <div>
            <dt>Threshold</dt>
            <dd data-testid="signal-threshold">{rule}</dd>
          </div>
        ) : null}
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
              {coldStart ? (
                <span className={styles.cold} data-testid="signal-cold-start">
                  <span className={styles.tag}>Cold start</span> {coldStartNote(loadMs)}
                </span>
              ) : null}
            </dd>
          </div>
        ) : null}
      </dl>
      {signal.probabilities ? (
        <>
          <ul className={styles.probs} aria-label="Class probabilities">
            {ORDER.map((s) => {
              const p = signal.probabilities?.[s] ?? 0;
              return (
                <li key={s} className={styles.prob}>
                  <span className={styles.probLabel}>{capitalize(s)}</span>
                  <span className={styles.probTrack} aria-hidden="true">
                    <span className={styles.probFill} data-sentiment={s} style={{ width: share(p) }} />
                    {thresholdMark !== null ? (
                      <span
                        className={styles.probMark}
                        data-kind="threshold"
                        data-testid={`mark-threshold-${s}`}
                        style={{ left: share(thresholdMark) }}
                      />
                    ) : null}
                    {tauMark !== null && s !== "neutral" ? (
                      <span
                        className={styles.probMark}
                        data-kind="tau"
                        data-testid={`mark-tau-${s}`}
                        style={{ left: share(tauMark) }}
                      />
                    ) : null}
                  </span>
                  <span className={styles.probValue}>{formatPct(p, digits)}</span>
                </li>
              );
            })}
          </ul>
          {thresholdMark !== null || tauMark !== null ? (
            <p className={styles.markLegend} aria-hidden="true">
              {thresholdMark !== null ? (
                <span>
                  <span className={styles.markSwatch} data-kind="threshold" />
                  threshold {formatPct(thresholdMark, digits)}
                </span>
              ) : null}
              {tauMark !== null ? (
                <span>
                  <span className={styles.markSwatch} data-kind="tau" />
                  tau {formatPct(tauMark, digits)} (bullish/bearish)
                </span>
              ) : null}
            </p>
          ) : null}
        </>
      ) : null}
    </>
  );
}
