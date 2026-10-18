# VeriTrade Analytics - real-time financial news analysis (PoC)

Paste a public financial-news URL, get a concise summary and a Bullish / Bearish / Neutral signal, fast.

> Work in progress. Design decisions: [`.context/docs/design/01-decisions.md`](.context/docs/design/01-decisions.md).
> The full README (run instructions, latency trade-offs, eval results, AI tooling) lands with the final milestone.

## Layout

| Path | What |
|---|---|
| `backend/` | FastAPI service (`veritrade` package): fetch, extract, inference, API |
| `evals/` | Evaluation and benchmarking harness (`veritrade_evals` package) |
| `frontend/` | Next.js 16 static export, served by FastAPI |
| `scripts/` | Setup, quality gate, probes, exports |
| `.context/` | Brief, research, design docs, AI conversation transcript |

## License

MIT
