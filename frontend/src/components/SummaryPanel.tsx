import { Fragment, type ReactNode } from "react";

import { coldStartNote, formatMs } from "@/lib/format";
import type { LaneStatus } from "@/lib/run-state";

import styles from "./SummaryPanel.module.css";

interface SummaryPanelProps {
  status: LaneStatus;
  text: string;
  /** The server cut the summary at the model's output limit. */
  truncated?: boolean;
  model?: string | null;
  streaming: boolean;
  /** Client-measured time from submit to the first summary token. */
  firstTokenMs?: number | null;
  /** The model had to be loaded for this request (known from the stream's meta event or final timings). */
  coldStart?: boolean | null;
  /** Ollama's model load time, once the final timings arrive. */
  loadMs?: number | null;
  /** The same generation already returned the signal, so the model is loaded (combined mode). */
  modelAnswered?: boolean;
}

function statusLabel(status: LaneStatus, text: string, streaming: boolean, loading: boolean): string {
  const waiting = loading
    ? "Loading the model into memory (cold start)..."
    : "Waiting for the first token...";
  switch (status) {
    case "pending":
      return streaming ? waiting : "Waiting for the summary...";
    case "streaming":
      return text ? "Streaming..." : waiting;
    case "done":
      return "Complete";
    case "error":
      return "Failed";
    case "cancelled":
      return "Cancelled";
  }
}

export function SummaryPanel({
  status,
  text,
  truncated,
  model,
  streaming,
  firstTokenMs,
  coldStart,
  loadMs,
  modelAnswered = false,
}: SummaryPanelProps) {
  const active = status === "pending" || status === "streaming";
  const loading = coldStart === true && !modelAnswered;
  const footer: ReactNode[] = [];
  if (model) {
    footer.push(
      <>
        Model <code>{model}</code>
      </>,
    );
  }
  if (typeof firstTokenMs === "number") footer.push(`first token ${formatMs(firstTokenMs)}`);
  // While the status line says the model is loading, the footer does not repeat it.
  if (coldStart && (text || !active || modelAnswered)) {
    footer.push(<span data-testid="summary-cold-start">cold start: {coldStartNote(loadMs)}</span>);
  }
  return (
    <section className={styles.panel} aria-labelledby="summary-heading" aria-busy={active}>
      <div className={styles.header}>
        <h2 id="summary-heading" className={styles.heading}>
          Summary
        </h2>
        <span className={styles.status} data-status={status} data-testid="summary-status">
          {statusLabel(status, text, streaming, loading)}
        </span>
      </div>
      {text ? (
        <p className={styles.text} data-testid="summary-text">
          {text}
          {active ? <span className={styles.caret} aria-hidden="true" /> : null}
        </p>
      ) : status === "done" ? (
        <p className={styles.empty}>The model returned an empty summary.</p>
      ) : null}
      {truncated && text ? (
        <p className={styles.note} data-testid="summary-truncated">
          The model hit its output limit, so this summary was cut short.
        </p>
      ) : null}
      {footer.length ? (
        <p className={styles.footer}>
          {footer.map((part, i) => (
            <Fragment key={i}>
              {i ? " · " : null}
              {part}
            </Fragment>
          ))}
        </p>
      ) : null}
    </section>
  );
}
