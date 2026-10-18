import {
  LATENCY_BUDGETS,
  withinBudget,
  type BenchmarksV1,
  type ExtractionBenchmark,
  type LatencyBenchmark,
  type SelectionEndpoint,
  type SignalBenchmark,
  type SummaryBenchmark,
} from "@/lib/benchmarks";
import { formatInt, formatMs, formatNum, formatPct } from "@/lib/format";
import { sourceName } from "@/lib/sources";

import styles from "./Benchmarks.module.css";
import { DataTable, type Column } from "./DataTable";

const SELECTION_ORDER: readonly SelectionEndpoint[] = ["signal", "summary", "analyze"];

function Selected() {
  return <span className={styles.selected}>selected</span>;
}

function ModelCell({ model, selected, note }: { model: string; selected: boolean; note?: string | null }) {
  return (
    <>
      <code>{model}</code>
      {selected ? <Selected /> : null}
      {note ? <span className={styles.sub}>{note}</span> : null}
    </>
  );
}

function isSelected(data: BenchmarksV1, endpoint: string, model: string): boolean {
  return (
    (endpoint === "signal" || endpoint === "summary" || endpoint === "analyze") &&
    data.selection[endpoint]?.model === model
  );
}

export function BenchmarksView({ data }: { data: BenchmarksV1 }) {
  const signalColumns: Column<SignalBenchmark>[] = [
    {
      header: "Model",
      cell: (r) => (
        <ModelCell model={r.model} selected={isSelected(data, r.endpoint, r.model)} note={r.kind} />
      ),
    },
    {
      header: "Endpoint",
      cell: (r) => (
        <>
          {r.endpoint}
          {r.fieldOrder ? <span className={styles.sub}>{r.fieldOrder}</span> : null}
        </>
      ),
    },
    { header: "Policy", cell: (r) => r.policy },
    { header: "Scope", cell: (r) => r.scope },
    {
      header: "Dataset",
      cell: (r) => (
        <>
          {r.dataset} / {r.split}
          <span className={styles.sub}>prompt {r.promptVersion}</span>
        </>
      ),
    },
    { header: "n", numeric: true, cell: (r) => formatInt(r.n) },
    {
      header: "F0.5",
      title: "F0.5 macro over bullish and bearish (selection metric)",
      numeric: true,
      cell: (r) => (
        <>
          <strong>{formatNum(r.f05Macro)}</strong>
          {r.ciLow !== null && r.ciHigh !== null ? (
            <span className={styles.sub}>
              {formatNum(r.ciLow, 2)} to {formatNum(r.ciHigh, 2)}
            </span>
          ) : null}
        </>
      ),
    },
    {
      header: "P bull",
      title: "Precision, bullish",
      numeric: true,
      cell: (r) => formatNum(r.precisionBullish),
    },
    {
      header: "P bear",
      title: "Precision, bearish",
      numeric: true,
      cell: (r) => formatNum(r.precisionBearish),
    },
    { header: "R bull", title: "Recall, bullish", numeric: true, cell: (r) => formatNum(r.recallBullish) },
    { header: "R bear", title: "Recall, bearish", numeric: true, cell: (r) => formatNum(r.recallBearish) },
    {
      header: "Inversion",
      title: "Bullish called bearish or the reverse (cap 2%)",
      numeric: true,
      cell: (r) => (
        <span data-over={r.inversionRate > 0.02 || undefined} className={styles.flag}>
          {formatPct(r.inversionRate)}
        </span>
      ),
    },
    {
      header: "Coverage",
      title: "Share of rows not abstained",
      numeric: true,
      cell: (r) => formatPct(r.coverage),
    },
  ];

  const latencyColumns: Column<LatencyBenchmark>[] = [
    { header: "Endpoint", cell: (r) => r.endpoint },
    {
      header: "Model",
      cell: (r) => <ModelCell model={r.model} selected={isSelected(data, r.endpoint, r.model)} />,
    },
    { header: "Scope", cell: (r) => r.scope },
    { header: "Streaming", cell: (r) => (r.streaming ? "SSE" : "JSON") },
    { header: "Backend", cell: (r) => r.backend },
    { header: "n", numeric: true, cell: (r) => formatInt(r.n) },
    { header: "p50", numeric: true, cell: (r) => formatMs(r.p50Ms) },
    { header: "p95", numeric: true, cell: (r) => formatMs(r.p95Ms) },
    {
      header: "TTFT",
      title: "Time to first summary token, p50",
      numeric: true,
      cell: (r) => formatMs(r.ttftP50Ms),
    },
    {
      header: "To signal",
      title: "Time to sentiment, p50",
      numeric: true,
      cell: (r) => formatMs(r.timeToSignalP50Ms),
    },
    {
      header: "Budget",
      cell: (r) => {
        const ok = withinBudget(r);
        if (ok === null) return <span className={styles.sub}>n/a</span>;
        return (
          <span className={styles.flag} data-over={!ok || undefined}>
            {ok ? "within" : "over"}
          </span>
        );
      },
    },
  ];

  const summaryColumns: Column<SummaryBenchmark>[] = [
    { header: "Endpoint", cell: (r) => r.endpoint },
    {
      header: "Model",
      cell: (r) => <ModelCell model={r.model} selected={isSelected(data, r.endpoint, r.model)} />,
    },
    { header: "n", numeric: true, cell: (r) => formatInt(r.n) },
    {
      header: "Checks",
      title: "Deterministic pass rate: numbers grounded, entity named, length",
      numeric: true,
      cell: (r) => formatPct(r.deterministicPassRate),
    },
    {
      header: "Key facts",
      title: "Key-fact coverage by embedding similarity",
      numeric: true,
      cell: (r) => formatPct(r.keyFactCoverage),
    },
    {
      header: "Faithful",
      title: "Judge faithfulness, 1 to 5",
      numeric: true,
      cell: (r) => formatNum(r.judgeFaithfulness, 1),
    },
    {
      header: "Cover",
      title: "Judge coverage, 1 to 5",
      numeric: true,
      cell: (r) => formatNum(r.judgeCoverage, 1),
    },
    {
      header: "Halluc.",
      title: "Share of summaries with a hallucination flag",
      numeric: true,
      cell: (r) => formatPct(r.hallucinationRate),
    },
  ];

  const extractionColumns: Column<ExtractionBenchmark>[] = [
    { header: "Parser", cell: (r) => r.parser ?? <span className={styles.sub}>built-in</span> },
    { header: "Extractor", cell: (r) => <code>{r.extractor}</code> },
    { header: "n", numeric: true, cell: (r) => formatInt(r.n) },
    { header: "p50", numeric: true, cell: (r) => `${formatNum(r.p50Ms, 1)} ms` },
    { header: "p95", numeric: true, cell: (r) => `${formatNum(r.p95Ms, 1)} ms` },
    {
      header: "P",
      title: "Token precision vs gold body",
      numeric: true,
      cell: (r) => formatNum(r.precision, 2),
    },
    { header: "R", title: "Token recall vs gold body", numeric: true, cell: (r) => formatNum(r.recall, 2) },
    { header: "F1", numeric: true, cell: (r) => <strong>{formatNum(r.f1, 2)}</strong> },
    {
      header: "Leakage",
      title: "Share of output tokens that are boilerplate",
      numeric: true,
      cell: (r) => formatPct(r.leakage),
    },
    {
      header: "Tokens",
      title: "Tokens sent to the LLM",
      numeric: true,
      cell: (r) => formatInt(r.tokensToLlm),
    },
    { header: "Fail", title: "Failure rate", numeric: true, cell: (r) => formatPct(r.failureRate) },
    {
      header: "Install",
      title: "Install size in MB",
      numeric: true,
      cell: (r) => (r.installMb === null ? "-" : `${formatNum(r.installMb, 1)} MB`),
    },
  ];

  const sources = Object.entries(data.articles.perSource);
  const env = data.environment;

  return (
    <div className={styles.page}>
      <h1>Benchmarks</h1>
      {data.sample ? (
        <p className={styles.sampleBanner} data-testid="sample-banner">
          <strong>Sample data.</strong> These numbers are illustrative placeholders, not measurements. They
          are replaced by <code>evals/results/benchmarks.json</code> once the eval suite has run.
        </p>
      ) : null}
      <p className={styles.env}>
        {env.chip}, {env.ramGb} GB RAM, Ollama {env.ollamaVersion}, backend <code>{env.backend}</code>
        {data.generatedAt ? `, generated ${data.generatedAt}` : ""}
        {env.gitSha ? (
          <>
            {" "}
            at <code>{env.gitSha.slice(0, 10)}</code>
          </>
        ) : null}
        .
      </p>

      <section aria-labelledby="selection-heading" className={styles.section}>
        <h2 id="selection-heading">Selected models</h2>
        <ul className={styles.selection}>
          {SELECTION_ORDER.map((endpoint) => {
            const pick = data.selection[endpoint];
            return (
              <li key={endpoint} className={styles.pick}>
                <span className={styles.pickEndpoint}>/api/{endpoint}</span>
                <code className={styles.pickModel}>{pick?.model ?? "-"}</code>
                <span className={styles.pickReason}>{pick?.reason ?? "Not selected yet."}</span>
              </li>
            );
          })}
        </ul>
      </section>

      <section aria-labelledby="signal-heading" className={styles.section}>
        <h2 id="signal-heading">Signal accuracy</h2>
        <p className={styles.lead}>
          Winner = best F0.5 macro over the directional classes within the latency budget, with
          bullish/bearish inversions capped at 2%. Neutral is the safe default, so precision is weighted over
          recall.
        </p>
        <DataTable
          caption="Signal accuracy by model, policy, scope and dataset"
          columns={signalColumns}
          rows={data.signal}
          rowKey={(r, i) => `${r.model}-${r.endpoint}-${r.policy}-${r.scope}-${r.dataset}-${r.split}-${i}`}
          highlight={(r) => isSelected(data, r.endpoint, r.model)}
        />
      </section>

      <section aria-labelledby="latency-heading" className={styles.section}>
        <h2 id="latency-heading">Latency</h2>
        <p className={styles.lead}>
          Budgets (native, warm, excluding fetch): /signal lede p50 at most{" "}
          {formatMs(LATENCY_BUDGETS.signalLedeP50Ms)}; /summary and /analyze time to first token at most{" "}
          {formatMs(LATENCY_BUDGETS.ttftMs)} and complete p50 at most{" "}
          {formatMs(LATENCY_BUDGETS.completeP50Ms)}.
        </p>
        <DataTable
          caption="Latency by endpoint, model, scope, transport and backend"
          columns={latencyColumns}
          rows={data.latency}
          rowKey={(r, i) => `${r.endpoint}-${r.model}-${r.scope}-${r.streaming}-${r.backend}-${i}`}
          highlight={(r) => isSelected(data, r.endpoint, r.model)}
        />
      </section>

      <section aria-labelledby="summary-heading" className={styles.section}>
        <h2 id="summary-heading">Summary quality</h2>
        <DataTable
          caption="Summary quality by endpoint and model"
          columns={summaryColumns}
          rows={data.summary}
          rowKey={(r, i) => `${r.endpoint}-${r.model}-${i}`}
          highlight={(r) => isSelected(data, r.endpoint, r.model)}
        />
      </section>

      <section aria-labelledby="extraction-heading" className={styles.section}>
        <h2 id="extraction-heading">Extraction matrix</h2>
        <p className={styles.lead}>
          Our strategies (cascade, jsonld, profile, density) run on each parser backend; third-party
          extractors use their built-in parser.
        </p>
        <DataTable
          caption="Extraction speed and quality by parser and extractor"
          columns={extractionColumns}
          rows={data.extraction}
          rowKey={(r, i) => `${r.parser ?? "built-in"}-${r.extractor}-${i}`}
        />
      </section>

      <section aria-labelledby="articles-heading" className={styles.section}>
        <h2 id="articles-heading">Article set</h2>
        <p className={styles.lead}>
          {formatInt(data.articles.n)} labelled articles; labeller agreement{" "}
          {formatPct(data.articles.agreement)}, Cohen&apos;s kappa {formatNum(data.articles.kappa, 2)}.
        </p>
        <DataTable
          caption="Articles per source and label"
          columns={[
            { header: "Source", cell: ([key]) => sourceName(key) ?? key },
            { header: "Bullish", numeric: true, cell: ([, c]) => formatInt(c.bullish) },
            { header: "Bearish", numeric: true, cell: ([, c]) => formatInt(c.bearish) },
            { header: "Neutral", numeric: true, cell: ([, c]) => formatInt(c.neutral) },
            { header: "Total", numeric: true, cell: ([, c]) => formatInt(c.bullish + c.bearish + c.neutral) },
          ]}
          rows={sources}
          rowKey={([key]) => key}
        />
      </section>
    </div>
  );
}
