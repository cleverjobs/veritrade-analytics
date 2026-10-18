// Mirrors the primary sources in backend/src/veritrade/data/sources.toml, in rank order.
export interface SourceInfo {
  key: string;
  name: string;
  /** A stable, freely fetchable article for the quick-pick buttons. */
  exampleUrl?: string;
}

export const SOURCES: readonly SourceInfo[] = [
  { key: "yahoo", name: "Yahoo Finance" },
  { key: "cnbc", name: "CNBC" },
  { key: "fool", name: "The Motley Fool" },
  { key: "prnewswire", name: "PR Newswire" },
  { key: "globenewswire", name: "GlobeNewswire" },
  { key: "sec", name: "SEC EDGAR 8-K EX-99.1" },
  { key: "cnn", name: "CNN Business" },
  { key: "fortune", name: "Fortune" },
  { key: "guardian", name: "The Guardian Business" },
  { key: "bbc", name: "BBC Business" },
];

export function sourceName(key: string | null | undefined): string | null {
  if (!key) return null;
  return SOURCES.find((s) => s.key === key)?.name ?? key;
}
