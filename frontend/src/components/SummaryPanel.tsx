import { formatMs } from "@/lib/format";
import type { LaneStatus } from "@/lib/run-state";

import styles from "./SummaryPanel.module.css";

interface SummaryPanelProps {
  status: LaneStatus;
  text: string;
  model?: string | null;
  streaming: boolean;
  /** Client-measured time from submit to the first summary token. */
  firstTokenMs?: number | null;
}

function statusLabel(status: LaneStatus, text: string, streaming: boolean): string {
  switch (status) {
    case "pending":
      return streaming ? "Waiting for the first token..." : "Waiting for the summary...";
    case "streaming":
      return text ? "Streaming..." : "Waiting for the first token...";
    case "done":
      return "Complete";
    case "error":
      return "Failed";
    case "cancelled":
      return "Cancelled";
  }
}

export function SummaryPanel({ status, text, model, streaming, firstTokenMs }: SummaryPanelProps) {
  const active = status === "pending" || status === "streaming";
  return (
    <section className={styles.panel} aria-labelledby="summary-heading" aria-busy={active}>
      <div className={styles.header}>
        <h2 id="summary-heading" className={styles.heading}>
          Summary
        </h2>
        <span className={styles.status} data-status={status} data-testid="summary-status">
          {statusLabel(status, text, streaming)}
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
      {model || typeof firstTokenMs === "number" ? (
        <p className={styles.footer}>
          {model ? (
            <>
              Model <code>{model}</code>
            </>
          ) : null}
          {model && typeof firstTokenMs === "number" ? " · " : null}
          {typeof firstTokenMs === "number" ? `first token ${formatMs(firstTokenMs)}` : null}
        </p>
      ) : null}
    </section>
  );
}
