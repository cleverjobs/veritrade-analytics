# Frontend

Next.js 16 (App Router, strict TypeScript, CSS modules) built as a **static export**. In production FastAPI
serves `out/` from the same origin as `/api`, so no Node process runs and the browser talks to the API with no
proxy hop.

## Pages

- `/` (Analyze): URL input, Split vs Combined mode, streaming on/off, model, parser, extractor and scope pickers
  filled from `GET /api/models`. Split mode fires `/api/signal` and `/api/summary/stream` together so the
  signal card renders as soon as it lands; Combined mode calls `/api/analyze` (or `/api/analyze/stream`). The
  latency bar puts the server stage timings of each request on one client time axis (`performance.now()`
  from submit), with time-to-signal and time-to-first-token marks. Resubmitting aborts the in-flight run.
- `/benchmarks/`: renders `src/data/benchmarks.json`, which `scripts/copy-benchmarks.mjs` copies from
  `evals/results/benchmarks.json`, falling back to the committed `benchmarks.sample.json` (shown with a
  "sample data" banner).

## Layout

| Path                                 | Role                                                                                                                  |
| ------------------------------------ | --------------------------------------------------------------------------------------------------------------------- |
| `src/lib/api.ts`                     | Typed fetch helpers over the generated `api-schema.d.ts`; error envelope to `ApiError`                                |
| `src/lib/sse.ts`                     | POST + `TextDecoderStream` SSE parser yielding typed `meta` / `sentiment` / `summary_delta` / `done` / `error` events |
| `src/lib/run-state.ts`, `src/hooks/` | Run reducer (per-lane status and client timing marks) and the hooks that drive it                                     |
| `src/components/`                    | `AnalyzeForm`, `SignalCard`, `SummaryPanel`, `ArticleMetaLine`, `LatencyBar`, `ErrorBanner`, benchmark tables         |
| `src/mocks/`                         | MSW handlers and synthetic fixtures, shared by tests and any mocked dev run                                           |
| `tests/unit`, `tests/integration`    | Vitest + Testing Library; integration tests run the real components against MSW                                       |
| `e2e/`                               | Playwright specs for the docker compose stack                                                                         |

## Commands

```bash
pnpm install
pnpm dev            # http://localhost:3000; set NEXT_PUBLIC_API_BASE=http://localhost:8000 for a local API
pnpm lint && pnpm format:check && pnpm typecheck
pnpm test           # unit + MSW integration tests, no backend needed
pnpm build          # static export to out/
pnpm gen:api        # regenerate src/lib/api-schema.d.ts from backend/openapi.json
E2E_BASE_URL=http://localhost:8000 E2E_ARTICLE_URL=... pnpm e2e   # against a running stack
```

`NEXT_PUBLIC_API_BASE` is inlined at build time; leave it unset for the production build so requests stay
same-origin. In development the backend must allow the dev origin through `VT_CORS_ORIGINS`.
