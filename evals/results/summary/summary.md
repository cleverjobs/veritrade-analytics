# Summary evaluation

Decision R6: deterministic checks, key-fact coverage by a local embedding model, and Claude Opus as a
blind judge. Summaries come from the serving adapters (`/api/summary`, `/api/analyze`) with the default
`selectolax` + `cascade` extraction and `full` scope, run on the stored HTML of the Opus article set.
Written by `uv run python -m veritrade_evals.summary_eval report`; no article or summary text is committed.
Every check scores the served summary: serving caps it at 3 sentences and 60 words
(`cap_summary` in `veritrade.inference.adapters`), and `score` applies the same cap to cached generations.

- **Halluc.** (`hallucinationRate`): share of summaries with any unsupported detail, minor slips included.
- **Material** (`materialErrorRate`): share at faithfulness <= 3: a material error.

## dev split (n=30 articles)

| Endpoint | Model | Pass | <=60 words | <=3 sent. | Numbers | Entity | Truncated | Capped | Errors | Median words | Key facts | Key facts (strict) | Faithful | Cover | Halluc. | Material | Leads |
|---|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| summary | `qwen3.5:2b-q4_K_M` | 90% | 100% | 100% | 97% | 93% | 3% | 37% | 0% | 49.0 | 52% | 24% | 3.97 | 4.30 | 67% | 30% | 90% |
| summary | `qwen3.5:4b-q4_K_M` | 90% | 100% | 100% | 93% | 97% | 0% | 3% | 0% | 40.0 | 47% | 23% | 4.40 | 4.57 | 50% | 10% | 100% |
| analyze | `qwen3.5:4b-q4_K_M` | 93% | 100% | 100% | 97% | 97% | 0% | 33% | 0% | 48.0 | 57% | 29% | 4.47 | 4.63 | 40% | 10% | 100% |
| analyze | `qwen3:4b-instruct-2507-q4_K_M` | 90% | 100% | 100% | 90% | 100% | 0% | 7% | 0% | 40.5 | 56% | 25% | 4.73 | 4.43 | 20% | 7% | 83% |
| analyze | `ministral-3:3b` | 77% | 100% | 100% | 80% | 97% | 0% | 0% | 0% | 37.0 | 45% | 19% | 3.77 | 4.27 | 63% | 40% | 80% |

Unsupported numbers (distinct per summary, summed) and how many of them the model input had:

- summary `qwen3.5:2b-q4_K_M`: 1 unsupported, 1 present in the extracted input but not in the gold article
- summary `qwen3.5:4b-q4_K_M`: 2 unsupported, 0 present in the extracted input but not in the gold article
- analyze `qwen3.5:4b-q4_K_M`: 1 unsupported, 0 present in the extracted input but not in the gold article
- analyze `qwen3:4b-instruct-2507-q4_K_M`: 3 unsupported, 0 present in the extracted input but not in the gold article
- analyze `ministral-3:3b`: 8 unsupported, 1 present in the extracted input but not in the gold article

Key-fact coverage at other thresholds (main threshold 0.60):

- summary `qwen3.5:2b-q4_K_M`: 0.65: 46%, 0.70: 33%, 0.80: 17%
- summary `qwen3.5:4b-q4_K_M`: 0.65: 39%, 0.70: 32%, 0.80: 16%
- analyze `qwen3.5:4b-q4_K_M`: 0.65: 48%, 0.70: 40%, 0.80: 23%
- analyze `qwen3:4b-instruct-2507-q4_K_M`: 0.65: 48%, 0.70: 32%, 0.80: 17%
- analyze `ministral-3:3b`: 0.65: 36%, 0.70: 26%, 0.80: 14%

Quality order (deterministic pass rate >= 0.8, then judge faithfulness, then judge
coverage; latency budgets are applied later by `select`):

- /summary: `qwen3.5:4b-q4_K_M`, `qwen3.5:2b-q4_K_M`
- /analyze: `qwen3:4b-instruct-2507-q4_K_M`, `qwen3.5:4b-q4_K_M`

Judge means with 95% bootstrap CIs (articles resampled within their sentiment label):

- summary `qwen3.5:2b-q4_K_M`: faithfulness 3.97 [3.67, 4.23], coverage 4.30 [4.03, 4.57], hallucination rate 67% [53%, 80%], material error rate 30% [17%, 47%] (n=30)
- summary `qwen3.5:4b-q4_K_M`: faithfulness 4.40 [4.17, 4.60], coverage 4.57 [4.40, 4.73], hallucination rate 50% [33%, 67%], material error rate 10% [0%, 20%] (n=30)
- analyze `qwen3.5:4b-q4_K_M`: faithfulness 4.47 [4.20, 4.70], coverage 4.63 [4.47, 4.80], hallucination rate 40% [27%, 57%], material error rate 10% [0%, 20%] (n=30)
- analyze `qwen3:4b-instruct-2507-q4_K_M`: faithfulness 4.73 [4.53, 4.93], coverage 4.43 [4.17, 4.67], hallucination rate 20% [7%, 33%], material error rate 7% [0%, 17%] (n=30)
- analyze `ministral-3:3b`: faithfulness 3.77 [3.37, 4.20], coverage 4.27 [3.97, 4.60], hallucination rate 63% [47%, 80%], material error rate 40% [23%, 57%] (n=30)

Leader minus runner-up, paired by article:

- /summary: `qwen3.5:4b-q4_K_M` vs `qwen3.5:2b-q4_K_M`: faithfulness 0.43 [0.07, 0.77], coverage 0.27 [0.03, 0.53]
- /analyze: `qwen3:4b-instruct-2507-q4_K_M` vs `qwen3.5:4b-q4_K_M`: faithfulness 0.27 [-0.07, 0.60], coverage -0.20 [-0.53, 0.10]

Coverage threshold spot check:

- 10 summaries, 45 key facts, 29 of them stated in the summary by hand label (read in the private files)
- at 0.60: 33 of 45 flags right, 5 wrongly covered, 7 wrongly not covered
- at 0.75: 25 of 45 flags right, 0 wrongly covered, 20 wrongly not covered
- Hand labels from reading the summaries as generated before the serving cap; the flags shown in the review file used 0.75.

## Definitions

- **Pass**: <= 60 words, <= 3 sentences (serving sentence splitter), every number
  grounded, entity named, not truncated. Rates are over all articles; a candidate that produced no summary
  fails every check. Length holds by construction for served summaries; **Capped** is the share the cap
  had to shorten (whole leading sentences kept; an overlong first sentence ends in an ellipsis).
- **Numbers**: every number in the summary appears in the gold title or body. Thousands separators, `%`,
  currency symbols and units are ignored and 1.50 equals 1.5; number words up to twenty in the article
  ground their digits; Q1-Q4 / H1-H2 match the same token, `third quarter` / `first half`, `September
  quarter` or `July to September` (calendar quarters); `FY27` matches 2027; `80K` matches 80,000 (K/M/B
  abbreviations); a clock time on the hour (`5:00 p.m.`) matches the bare hour (`5PM`). Derived figures
  (a percentage change the model computed) count as unsupported: the prompt asks for verbatim numbers.
- **Entity**: the gold entity appears (case-insensitive, whole words); or its short name, without legal
  suffixes (Inc, Corp, Ltd, plc, Holdings, Group, ...) and trailing descriptors (Entertainment,
  Technologies, Industries, ...), as a phrase or as all its words anywhere (`UK ... economy`); or its first
  word when the name has several words and that word has 4+ letters and is not generic (American, United,
  Bank, ...); or the gold ticker (case-sensitive, also without an exchange suffix such as `.HK`).
- **Truncated**: the output hit `num_predict` and the serving path cut it to its last full sentence.
- **Key facts**: each of the 3-5 Opus key facts and each summary sentence is embedded with
  `qwen3-embedding:0.6b`; a fact is covered when its best cosine is >= 0.60, the threshold
  that agreed best with hand labels in the spot check. Mean share per article. **Key facts (strict)** uses
  0.75, where every flagged fact was really stated: a conservative lower bound.
- **Faithful / Cover**: Opus judge scores 1-5 (`evals/opus_prompts/50-judge.md`), candidates anonymised and
  shuffled per article. **Halluc.** and **Material** are defined at the top: a minor slip such as a
  loosely worded detail counts towards the first only. **Leads**: share the judge found leading with the
  market-moving fact.
