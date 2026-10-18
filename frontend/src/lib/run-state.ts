import type { ApiError, ArticleMeta, SignalResult, Timings } from "./api";

export type Mode = "split" | "combined";
export type LaneId = "signal" | "summary" | "analyze";
export type LaneStatus = "pending" | "streaming" | "done" | "error" | "cancelled";

export interface Lane {
  id: LaneId;
  status: LaneStatus;
  meta: ArticleMeta | null;
  timings: Timings | null;
  error: ApiError | null;
  // Client clock: milliseconds since the run started, from performance.now().
  signalAtMs: number | null;
  firstTokenAtMs: number | null;
  endAtMs: number | null;
}

export interface RunState {
  runId: number;
  mode: Mode;
  streaming: boolean;
  running: boolean;
  lanes: Lane[];
  signal: SignalResult | null;
  entity: string | null;
  /** Server-reported time to sentiment on the combined stream. */
  serverSignalMs: number | null;
  summary: string;
}

interface Tagged {
  runId: number;
  lane: LaneId;
  atMs: number;
}

export type RunAction =
  | { type: "start"; runId: number; mode: Mode; streaming: boolean; lanes: LaneId[] }
  | ({ type: "meta"; meta: ArticleMeta } & Tagged)
  | ({ type: "signal"; signal: SignalResult; entity: string | null; serverMs: number | null } & Tagged)
  | ({ type: "delta"; text: string } & Tagged)
  | ({
      type: "done";
      meta: ArticleMeta;
      timings: Timings;
      summary?: string;
      signal?: SignalResult;
      entity?: string | null;
    } & Tagged)
  | ({ type: "error"; error: ApiError } & Tagged)
  | { type: "cancel"; runId: number; atMs: number };

export const initialRunState: RunState = {
  runId: 0,
  mode: "split",
  streaming: true,
  running: false,
  lanes: [],
  signal: null,
  entity: null,
  serverSignalMs: null,
  summary: "",
};

function newLane(id: LaneId): Lane {
  return {
    id,
    status: "pending",
    meta: null,
    timings: null,
    error: null,
    signalAtMs: null,
    firstTokenAtMs: null,
    endAtMs: null,
  };
}

const isActive = (lane: Lane) => lane.status === "pending" || lane.status === "streaming";

function patchLane(state: RunState, id: LaneId, patch: (lane: Lane) => Partial<Lane>): RunState {
  const lanes = state.lanes.map((lane) => (lane.id === id ? { ...lane, ...patch(lane) } : lane));
  return { ...state, lanes, running: lanes.some(isActive) };
}

export function runReducer(state: RunState, action: RunAction): RunState {
  if (action.type === "start") {
    return {
      ...initialRunState,
      runId: action.runId,
      mode: action.mode,
      streaming: action.streaming,
      running: true,
      lanes: action.lanes.map(newLane),
    };
  }
  // Late events from a superseded run must not leak into the current one.
  if (action.runId !== state.runId) return state;

  switch (action.type) {
    case "meta":
      return patchLane(state, action.lane, (lane) => ({
        meta: lane.meta ?? action.meta,
        status: lane.status === "pending" ? "streaming" : lane.status,
      }));
    case "signal":
      return {
        ...patchLane(state, action.lane, (lane) => ({ signalAtMs: lane.signalAtMs ?? action.atMs })),
        signal: action.signal,
        entity: action.entity,
        serverSignalMs: action.serverMs,
      };
    case "delta":
      return {
        ...patchLane(state, action.lane, (lane) => ({
          firstTokenAtMs: lane.firstTokenAtMs ?? action.atMs,
          status: "streaming",
        })),
        summary: state.summary + action.text,
      };
    case "done": {
      const next = patchLane(state, action.lane, (lane) => ({
        status: "done",
        meta: action.meta,
        timings: action.timings,
        endAtMs: action.atMs,
        signalAtMs: action.signal ? (lane.signalAtMs ?? action.atMs) : lane.signalAtMs,
      }));
      return {
        ...next,
        // The final payload is authoritative over the concatenated deltas.
        summary: action.summary ?? next.summary,
        signal: action.signal ?? next.signal,
        entity: action.entity !== undefined ? action.entity : next.entity,
      };
    }
    case "error":
      return patchLane(state, action.lane, () => ({
        status: "error",
        error: action.error,
        endAtMs: action.atMs,
      }));
    case "cancel": {
      const lanes = state.lanes.map((lane) =>
        isActive(lane) ? { ...lane, status: "cancelled" as const, endAtMs: action.atMs } : lane,
      );
      return { ...state, lanes, running: false };
    }
  }
}

export function laneById(state: RunState, id: LaneId): Lane | undefined {
  return state.lanes.find((lane) => lane.id === id);
}

/** Article-level meta from whichever lane reported first. */
export function articleMeta(state: RunState): ArticleMeta | null {
  return state.lanes.find((lane) => lane.meta)?.meta ?? null;
}

/** One banner per distinct failure; split lanes often fail identically on a shared fetch. */
export function distinctErrors(state: RunState): { lanes: LaneId[]; error: ApiError }[] {
  const out: { lanes: LaneId[]; error: ApiError }[] = [];
  for (const lane of state.lanes) {
    if (!lane.error) continue;
    const same = out.find((e) => e.error.code === lane.error?.code && e.error.message === lane.error.message);
    if (same) same.lanes.push(lane.id);
    else out.push({ lanes: [lane.id], error: lane.error });
  }
  return out;
}
