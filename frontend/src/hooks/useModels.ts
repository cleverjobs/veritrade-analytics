import { useEffect, useState } from "react";

import { getModels, isAbortError, toApiError, type ApiError, type ModelsResponse } from "@/lib/api";

export type ModelsState =
  { status: "loading" } | { status: "ready"; data: ModelsResponse } | { status: "error"; error: ApiError };

export function useModels(): ModelsState {
  const [state, setState] = useState<ModelsState>({ status: "loading" });

  useEffect(() => {
    const controller = new AbortController();
    getModels(controller.signal).then(
      (data) => setState({ status: "ready", data }),
      (err: unknown) => {
        if (!isAbortError(err)) setState({ status: "error", error: toApiError(err) });
      },
    );
    return () => controller.abort();
  }, []);

  return state;
}
