export function formatMs(ms: number | null | undefined): string {
  if (ms === null || ms === undefined || !Number.isFinite(ms)) return "-";
  if (ms >= 10_000) return `${(ms / 1000).toFixed(1)} s`;
  if (ms >= 1000) return `${(ms / 1000).toFixed(2)} s`;
  return `${Math.round(ms)} ms`;
}

export function formatPct(value: number | null | undefined, digits = 1): string {
  if (value === null || value === undefined || !Number.isFinite(value)) return "-";
  return `${(value * 100).toFixed(digits)}%`;
}

/** Fewest decimals (1 to 3) at which `value` stops printing as `limit`, so a near miss never reads as a hit. */
export function missDigits(value: number, limit: number): number {
  let digits = 1;
  while (digits < 3 && formatPct(value, digits) === formatPct(limit, digits)) digits += 1;
  return digits;
}

export function formatNum(value: number | null | undefined, digits = 3): string {
  if (value === null || value === undefined || !Number.isFinite(value)) return "-";
  return value.toFixed(digits);
}

export function formatInt(value: number | null | undefined): string {
  if (value === null || value === undefined || !Number.isFinite(value)) return "-";
  return Math.round(value).toLocaleString("en-US");
}

/** Why a cold request was slow; decision models report no separate load time. */
export function coldStartNote(loadMs: number | null | undefined): string {
  return typeof loadMs === "number" && loadMs > 0
    ? `model loaded for this request (${formatMs(loadMs)})`
    : "includes loading the model into memory";
}

export function capitalize(value: string): string {
  return value ? value[0].toUpperCase() + value.slice(1) : value;
}
