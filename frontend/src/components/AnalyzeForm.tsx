"use client";

import { useId, useState, type SubmitEvent } from "react";

import type { ModelsState } from "@/hooks/useModels";
import type { RunParams } from "@/hooks/useAnalysis";
import {
  EXTRACTORS,
  PARSERS,
  SCOPES,
  type Endpoint,
  type ExtractorName,
  type ModelInfo,
  type ParserName,
  type Scope,
} from "@/lib/api";
import type { Mode } from "@/lib/run-state";
import { SOURCES } from "@/lib/sources";

import styles from "./AnalyzeForm.module.css";

// Third-party extractors bring their own HTML parser, so the parser picker does not apply.
const PARSER_AWARE: ReadonlySet<ExtractorName> = new Set(["cascade", "jsonld", "profile", "density"]);

export function isHttpUrl(value: string): boolean {
  try {
    const url = new URL(value);
    return url.protocol === "http:" || url.protocol === "https:";
  } catch {
    return false;
  }
}

function modelLabel(model: ModelInfo, isDefault: boolean): string {
  const notes = [
    isDefault ? "default" : null,
    model.installed ? null : "not installed",
    model.evaluated ? null : "CI plumbing, unevaluated",
  ].filter(Boolean);
  return notes.length ? `${model.name} (${notes.join(", ")})` : model.name;
}

interface ModelSelectProps {
  label: string;
  endpoint: Endpoint;
  value: string;
  onChange: (value: string) => void;
  models: ModelsState;
}

function ModelSelect({ label, endpoint, value, onChange, models }: ModelSelectProps) {
  const id = useId();
  const ready = models.status === "ready" ? models.data : null;
  const fallback = ready?.defaults[endpoint];
  const options = ready ? ready.models.filter((m) => m.endpoints.includes(endpoint)) : [];
  return (
    <div className={styles.field}>
      <label htmlFor={id}>{label}</label>
      <select id={id} value={value} onChange={(e) => onChange(e.target.value)}>
        <option value="">{fallback ? `Server default (${fallback})` : "Server default"}</option>
        {options.map((m) => (
          <option key={m.name} value={m.name} disabled={!m.installed}>
            {modelLabel(m, m.name === fallback)}
          </option>
        ))}
      </select>
    </div>
  );
}

interface AnalyzeFormProps {
  models: ModelsState;
  running: boolean;
  onSubmit: (params: RunParams) => void;
  onCancel: () => void;
}

export function AnalyzeForm({ models, running, onSubmit, onCancel }: AnalyzeFormProps) {
  const [url, setUrl] = useState("");
  const [urlError, setUrlError] = useState<string | null>(null);
  const [mode, setMode] = useState<Mode>("split");
  const [streaming, setStreaming] = useState(true);
  const [signalModel, setSignalModel] = useState("");
  const [summaryModel, setSummaryModel] = useState("");
  const [analyzeModel, setAnalyzeModel] = useState("");
  const [parser, setParser] = useState<ParserName>("selectolax");
  const [extractor, setExtractor] = useState<ExtractorName>("cascade");
  const [scope, setScope] = useState<Scope>("lede");

  const ids = { url: useId(), urlHelp: useId(), parser: useId(), extractor: useId(), scope: useId() };
  const ready = models.status === "ready" ? models.data : null;
  const parsers = ready?.parsers.length ? ready.parsers : PARSERS;
  const extractors = ready?.extractors.length ? ready.extractors : EXTRACTORS;
  const scopes = ready?.scopes.length ? ready.scopes : SCOPES;
  const parserApplies = PARSER_AWARE.has(extractor);

  function handleSubmit(event: SubmitEvent<HTMLFormElement>) {
    event.preventDefault();
    const trimmed = url.trim();
    if (!isHttpUrl(trimmed)) {
      setUrlError("Enter a full http(s) URL, for example https://www.cnbc.com/2026/...");
      return;
    }
    setUrlError(null);
    onSubmit({
      url: trimmed,
      mode,
      streaming,
      parser,
      extractor,
      scope,
      signalModel,
      summaryModel,
      analyzeModel,
    });
  }

  const examples = SOURCES.filter((s) => s.exampleUrl);

  return (
    <form className={styles.form} onSubmit={handleSubmit} noValidate aria-label="Analyze an article">
      <div className={styles.urlRow}>
        <div className={styles.urlField}>
          <label htmlFor={ids.url}>Article URL</label>
          <input
            id={ids.url}
            name="url"
            type="url"
            inputMode="url"
            autoComplete="off"
            spellCheck={false}
            placeholder="https://..."
            value={url}
            onChange={(e) => setUrl(e.target.value)}
            aria-invalid={urlError ? true : undefined}
            aria-describedby={ids.urlHelp}
            required
          />
        </div>
        <div className={styles.actions}>
          <button type="submit" className={styles.primary}>
            {running ? "Restart" : "Analyze"}
          </button>
          {running ? (
            <button type="button" className={styles.secondary} onClick={onCancel}>
              Cancel
            </button>
          ) : null}
        </div>
      </div>
      <p id={ids.urlHelp} className={urlError ? styles.error : styles.help}>
        {urlError ?? "A public news article, press release or SEC 8-K exhibit."}
      </p>

      <div className={styles.sources}>
        <span className={styles.sourcesLabel}>Tested sources:</span>
        <ul className={styles.chips}>
          {SOURCES.map((s) => (
            <li key={s.key} className={styles.chip}>
              {s.name}
            </li>
          ))}
        </ul>
        {examples.length ? (
          <div className={styles.examples}>
            {examples.map((s) => (
              <button
                key={s.key}
                type="button"
                className={styles.secondary}
                onClick={() => setUrl(s.exampleUrl ?? "")}
              >
                Example: {s.name}
              </button>
            ))}
          </div>
        ) : null}
      </div>

      <div className={styles.groups}>
        <fieldset className={styles.group}>
          <legend>Mode</legend>
          <label className={styles.choice}>
            <input
              type="radio"
              name="mode"
              value="split"
              checked={mode === "split"}
              onChange={() => setMode("split")}
            />
            <span>
              Split
              <small>/api/signal and /api/summary fire together; the signal lands first.</small>
            </span>
          </label>
          <label className={styles.choice}>
            <input
              type="radio"
              name="mode"
              value="combined"
              checked={mode === "combined"}
              onChange={() => setMode("combined")}
            />
            <span>
              Combined
              <small>One /api/analyze call generates sentiment and summary together.</small>
            </span>
          </label>
          <label className={styles.choice}>
            <input type="checkbox" checked={streaming} onChange={(e) => setStreaming(e.target.checked)} />
            <span>
              Streaming
              <small>Server-sent events; off uses the JSON endpoints.</small>
            </span>
          </label>
        </fieldset>

        <fieldset className={styles.group}>
          <legend>Models</legend>
          {mode === "split" ? (
            <>
              <ModelSelect
                label="Signal model"
                endpoint="signal"
                value={signalModel}
                onChange={setSignalModel}
                models={models}
              />
              <ModelSelect
                label="Summary model"
                endpoint="summary"
                value={summaryModel}
                onChange={setSummaryModel}
                models={models}
              />
            </>
          ) : (
            <ModelSelect
              label="Analyze model"
              endpoint="analyze"
              value={analyzeModel}
              onChange={setAnalyzeModel}
              models={models}
            />
          )}
          {models.status === "loading" ? <p className={styles.help}>Loading models...</p> : null}
        </fieldset>

        <fieldset className={styles.group}>
          <legend>Extraction</legend>
          <div className={styles.field}>
            <label htmlFor={ids.extractor}>Extractor</label>
            <select
              id={ids.extractor}
              value={extractor}
              onChange={(e) => setExtractor(e.target.value as ExtractorName)}
            >
              {extractors.map((name) => (
                <option key={name} value={name}>
                  {name}
                </option>
              ))}
            </select>
          </div>
          <div className={styles.field}>
            <label htmlFor={ids.parser}>Parser</label>
            <select
              id={ids.parser}
              value={parser}
              onChange={(e) => setParser(e.target.value as ParserName)}
              disabled={!parserApplies}
            >
              {parsers.map((name) => (
                <option key={name} value={name}>
                  {name}
                </option>
              ))}
            </select>
            {!parserApplies ? <small className={styles.help}>{extractor} uses its own parser.</small> : null}
          </div>
          {mode === "split" ? (
            <div className={styles.field}>
              <label htmlFor={ids.scope}>Signal scope</label>
              <select id={ids.scope} value={scope} onChange={(e) => setScope(e.target.value as Scope)}>
                {scopes.map((name) => (
                  <option key={name} value={name}>
                    {name === "lede" ? "lede (headline + opening paragraphs)" : "full article"}
                  </option>
                ))}
              </select>
            </div>
          ) : null}
        </fieldset>
      </div>
    </form>
  );
}
