# VeriTrade Analytics — Financial News AI Architecture & Evaluation Plan

**Technical Design and Benchmarking Strategy**  
**Date:** October 16, 2026  
**Status:** Proposed architecture — pending empirical benchmarking

---

## 1. Executive Summary

The goal is to develop a low-latency financial-news analysis service that accepts a publicly accessible article URL and returns a concise factual summary and a financial sentiment classification.

Three architectures were considered:

1. **Single generative LLM:** One lightweight model produces the summary and financial sentiment as structured JSON.
2. **Hybrid decision model + LLM:** Clef-Flash or Jev performs sentiment classification while a separate generative LLM summarizes the article.
3. **Dedicated financial classifier + LLM:** A specialized model such as FinBERT handles sentiment while a generative model produces summaries.

### Recommended approach

**Start with a single locally hosted Qwen3.5 9B model through Ollama**, using structured JSON output and Pydantic validation.

Benchmark Qwen3.5 4B as a faster alternative and compare the results against a Clef-Flash-based hybrid architecture.

Introduce a dedicated classifier only if evaluation demonstrates a meaningful improvement in accuracy, reliability, latency, or throughput.

**Guiding principle:** Select the simplest architecture that satisfies measured performance requirements rather than assuming a more complex architecture is inherently superior.

---

## 2. Project Requirements

### Functional requirements

- Accept a public financial-news article URL via an API.
- Retrieve and extract the article's main text.
- Generate a concise, factual summary.
- Classify financial sentiment as Bullish, Bearish, or Neutral.
- Return structured JSON suitable for programmatic consumption.
- Provide a minimal web frontend.

### Non-functional requirements

- Prioritize end-to-end latency.
- Keep deployment and operation straightforward.
- Ensure predictable output formatting.
- Handle scraping failures and invalid URLs safely.
- Make model behavior reproducible and measurable.
- Allow straightforward replacement of the inference backend.

### Prescreen constraints

- Python backend, preferably FastAPI.
- Simple frontend, preferably React + Vite.
- Public GitHub repository.
- README documenting architectural decisions, latency trade-offs, run instructions, and testing.
- AI assistant conversation log committed to the repository.
- Approximately eight hours of planned development within the ten-day submission window.

---

## 3. Architecture Options

### Option A — Single Generative LLM

```text
Public article URL
       |
       v
FastAPI backend
       |
       v
URL validation + HTTP fetch
       |
       v
Article extraction + normalization
       |
       v
Ollama: Qwen3.5 9B
       |
       |-- Summary
       |-- Sentiment
       |-- Impact score
       |-- Supporting evidence
       |
       v
Pydantic validation
       |
       v
Structured JSON API
       |
       v
React frontend
```

**Advantages**

- One model and one inference request.
- One interpretation of the article for both summary and sentiment.
- Simpler infrastructure and lower maintenance overhead.
- No need to synchronize or reconcile outputs from multiple models.
- Strong starting point for a constrained prototype.

**Disadvantages**

- Sentiment classification involves autoregressive generation.
- An LLM-generated confidence score is not automatically calibrated.
- Model reasoning may be inconsistent across subtle financial scenarios.
- A single inference failure affects both outputs.

### Option B — Clef-Flash/Jev + Generative LLM

```text
Extracted article
       |
       +-------------------------+
       |                         |
       v                         v
Clef-Flash / Jev          Generative LLM
       |                         |
       v                         v
Sentiment probabilities       Summary
       |                         |
       +------------+------------+
                    |
                    v
             Response assembly
                    |
                    v
               API response
```

**Advantages**

- Specialized decision models produce distributions over predefined classes.
- Decision-model inference is non-autoregressive.
- Classification policies and thresholds can be evaluated independently.
- The classifier can potentially deliver a preliminary signal before summarization finishes.

**Disadvantages**

- Two inference requests and potentially two models loaded in memory.
- Greater operational complexity.
- Models may interpret the same source differently.
- Parallel inference is only beneficial if sufficient compute resources are available.

**Important:** Two model calls do not necessarily double latency. With independent compute resources, overall latency can approach the slower operation's latency. On a shared GPU, concurrent inference may introduce contention and increase response time.

### Option C — FinBERT + Generative LLM

A dedicated financial sentiment encoder is an important baseline.

FinBERT is designed for finance-domain sentiment classification and may offer excellent latency and classification accuracy relative to a larger general-purpose model.

However, FinBERT cannot produce summaries, and its short input context introduces complications when processing complete articles.

Additionally, the widely used ProsusAI/FinBERT checkpoint has a noncommercial Creative Commons license, so commercial deployment would require further licensing review.

### Comparison

| Criterion | Single LLM | Decision + LLM | FinBERT + LLM |
|---|---|---|---|
| Summary | Yes | Yes | Yes |
| Sentiment | Generated | Specialized | Specialized |
| Native class distribution | Not necessarily | Yes | Yes |
| Model deployments | 1 | 2 | 2 |
| Complexity | Low | Medium | Medium |
| Classification-only speed | Moderate | Potentially high | Potentially very high |
| Full-response speed | Must benchmark | Must benchmark | Must benchmark |
| Probability calibration | More involved | Natural postprocessing | Natural postprocessing |

**Initial selection: Option A.**

---

## 4. Local Model Candidates

Candidate models available through Ollama:

| Model | Approximate download size | Intended role |
|---|---:|---|
| Qwen3.5 4B | 3.3–4.0 GB | Speed-oriented generative model |
| Qwen3.5 9B | 6.6–7.6 GB | Primary generative candidate |
| Gemma 4 12B | 7.7–8.0 GB | Optional quality challenger |
| Ministral 3 3B | Around 3 GB | Lightweight alternative |
| Clef-Flash 9B | Depends on distribution | Dedicated decision model |
| Clef 27B | Larger model | Optional accuracy-oriented decision baseline |

Download sizes are not equivalent to runtime memory consumption. Actual requirements depend on quantization, context length, KV cache, and backend.

The shortlist does not imply that these models have been benchmarked on VeriTrade's workload.

### Why Qwen3.5 9B?

- Relatively compact.
- Suitable for local inference.
- Capable of generating summaries and structured responses.
- Large enough to be a reasonable candidate for nuanced financial language.
- Supported through Ollama.

### Why evaluate Qwen3.5 4B?

It may reduce inference time and memory requirements without materially harming classification or summarization quality.

This must be verified on financial articles, particularly examples involving conflicting financial indicators.

### Why evaluate Clef-Flash?

Clef-Flash is a 9B decision model developed by Cloudflare and supported through Ollama's `/v1/systemone` interface.

It evaluates predefined questions and decision criteria in a non-autoregressive forward pass.

Unlike a generative LLM, it cannot directly produce arbitrary prose summaries.

Its main potential advantages are fast classification, configurable decision criteria, and class-score distributions.

Ollama's documented requirement for Clef-Flash is version 0.35.1 or later.

---

## 5. Financial Sentiment Definition

The most important modeling decision is defining exactly what Bullish and Bearish mean.

**Financial sentiment should represent the assessed economic implications of the news for a specified entity, not the emotional tone of the article.**

Consider:

> Microsoft reports stronger cloud revenue, while Google experiences declining market share.

Potential interpretations:

- Microsoft: Bullish
- Google: Bearish
- Article overall: Mixed

This is why entity-aware sentiment analysis is preferable to generic article-level sentiment whenever the subject can be identified.

### Proposed ordinal impact scale

| Score | Meaning |
|---:|---|
| -2 | Strongly bearish |
| -1 | Bearish |
| 0 | Neutral |
| +1 | Bullish |
| +2 | Strongly bullish |

The three required sentiment classes can be mapped deterministically:

```python
sentiment = "bullish" if impact_score > 0 else "bearish" if impact_score < 0 else "neutral"
```

The impact score represents a qualitative assessment of the significance and direction of the reported information.

**It is not a calibrated probability, a forecast of future returns, or a measure of the probability that the stock price will rise or fall.**

### Additional metadata

An `evidence_status` field can indicate:

- `clear` — The source supports a reasonably clear direction.
- `mixed` — The source includes competing positive and negative implications.
- `insufficient` — The evidence does not support a reliable directional assessment.

This provides an explicit uncertainty indicator without generating misleading numerical confidence values.

Where no reliable judgment is possible, an internal `abstain` decision should be distinguished from genuine Neutral sentiment.

---

## 6. Structured Output Contract

Suggested JSON schema:

```json
{
  "summary": "Revenue exceeded expectations, but the company reduced forward guidance.",
  "target_entity": "Example Corp",
  "sentiment": "bearish",
  "impact_score": -1,
  "evidence_status": "mixed",
  "key_drivers": [
    "Revenue exceeded analyst expectations",
    "Forward earnings guidance was lowered"
  ],
  "model": "qwen3.5:9b",
  "latency_ms": 842
}
```

All values above are illustrative.

The backend should derive sentiment from the ordinal score rather than ask the model to independently generate both fields.

Use Pydantic to define and validate output contracts.

### Recommended model configuration

- Structured JSON output through Ollama's `format` parameter.
- Low-temperature or deterministic decoding where supported.
- Thinking disabled when supported and beneficial.
- Compact system instructions.
- Limited output tokens.
- Context size chosen for realistic article lengths.
- Keep the model resident for repeated requests.
- Pin model versions and record model digests for reproducibility.

Every response must undergo application-side validation, even when Ollama's structured-output mode is enabled.

### Minimal integration

```python
from typing import Literal

from ollama import Client
from pydantic import BaseModel


class ArticleAnalysis(BaseModel):
    summary: str
    impact_score: Literal[-2, -1, 0, 1, 2]
    evidence_status: Literal["clear", "mixed", "insufficient"]
    key_drivers: list[str]


client = Client(host="http://localhost:11434")

response = client.chat(
    model="qwen3.5:9b",
    messages=[
        {
            "role": "system",
            "content": (
                "Summarize financial news and evaluate "
                "economic implications for the target entity. "
                "Return factual information only."
            ),
        },
        {
            "role": "user",
            "content": article_text,
        },
    ],
    format=ArticleAnalysis.model_json_schema(),
    options={
        "temperature": 0,
        "num_ctx": 8192,
        "num_predict": 256,
    },
    think=False,
    keep_alive="30m",
)

analysis = ArticleAnalysis.model_validate_json(response.message.content)
```

This example is illustrative integration code, not a complete production endpoint. In particular, the real prompt must include an explicit scoring rubric and entity context.

---

## 7. Probability Scores vs. LLM-Generated Scores

A distinction must be maintained between three concepts.

### 7.1 Ordinal impact score

A model generates a category such as -2, -1, 0, +1, or +2.

This is an assessment of financial impact strength and direction.

It does not directly represent uncertainty.

### 7.2 Class probabilities

A classifier produces scores across the predefined categories.

Example:

```json
{
  "bullish": 0.76,
  "neutral": 0.18,
  "bearish": 0.06
}
```

These values may provide useful information about uncertainty, but are not automatically calibrated probabilities of correctness.

### 7.3 Calibrated probabilities

Probabilities are calibrated when predictions assigned approximately 80% confidence are correct approximately 80% of the time, under the evaluation distribution.

Calibration must be empirically measured.

For Clef-Flash, the `confidence` output specifically describes the concentration of the class distribution and should not be interpreted as the probability that the answer is correct.

### Recommended policy

For the initial generative-LLM architecture:

- Use an ordinal impact score.
- Use categorical evidence status.
- Avoid invented numerical confidence.
- Evaluate classification quality directly.

For the specialized decision-model experiments:

- Retain class distributions.
- Measure calibration.
- Optimize abstention thresholds.
- Compare accuracy against prediction coverage.

---

## 8. Public Financial Sentiment Datasets

### Financial PhraseBank

**Dataset:** https://huggingface.co/datasets/takala/financial_phrasebank

A finance-domain sentiment dataset containing sentences annotated as Positive, Negative, or Neutral.

It provides subsets based on annotator agreement:

| Agreement threshold | Examples |
|---|---:|
| At least 50% | 4,846 |
| At least 66% | 4,217 |
| At least 75% | 3,453 |
| Unanimous | 2,264 |

Best for:

- Baseline classification accuracy.
- Prompt comparison.
- Studying ambiguous financial statements.
- Assessing calibration on short-form financial language.

The subsets overlap and must not be treated as independent datasets.

**Licensing:** CC BY-NC-SA 3.0. Review noncommercial and redistribution restrictions.

### Twitter Financial News Sentiment

**Dataset:** https://huggingface.co/datasets/zeroshot/twitter-financial-news-sentiment

Contains approximately 11.9K financial short texts labeled:

- Bearish
- Bullish
- Neutral

Best for:

- Directional sentiment classification.
- Model comparison.
- Cross-domain evaluation.
- Testing prediction thresholds.

The dataset provides predefined training and validation partitions. Use these rather than introducing unnecessary leakage through ad hoc splitting.

The published license is MIT, although underlying social-media content rights may require separate consideration.

### SEntFiN

**Paper:** https://arxiv.org/abs/2305.12257

A dataset of approximately 10.7K financial headlines with entity-specific sentiment annotations.

Particularly useful for cases involving multiple companies with different financial implications.

Best for:

- Entity-specific sentiment.
- Multiple entities per article.
- Mixed sentiment scenarios.

### FinEntity

**Dataset:** https://huggingface.co/datasets/yixuantt/FinEntity

Financial-news annotations focused on sentiment toward individual entities.

Best for:

- Entity extraction and attribution.
- Financial context sensitivity.
- Company-specific sentiment evaluation.

### Custom full-article dataset

Public sentiment datasets generally emphasize sentences, headlines, or short text.

The production workflow requires evaluating complete articles.

Create a small manually reviewed benchmark of approximately 30–50 full articles, with labels for:

- Primary entity.
- Financial sentiment.
- Material positive and negative drivers.
- Contradictory evidence.
- Key numerical facts.
- Expected summary content.

This dataset should include earnings surprises, guidance changes, acquisitions, regulatory actions, layoffs, litigation, and mixed financial reports.

For reproducibility, record article URLs, retrieval timestamps, permitted source snapshots or hashes, and annotation criteria. Respect publisher usage restrictions.

---

## 9. Evaluation Methodology

### 9.1 Split the evaluation data

Use separate subsets for:

**Development set**

Optimize prompts, model selection, context handling, and decision criteria.

**Calibration set**

Tune abstention policies or probability calibration using frozen model predictions.

**Holdout set**

Evaluate the finalized configuration once without further tuning.

A possible custom-data allocation is 60% development, 20% calibration, and 20% holdout, but preserve official dataset splits where available.

Avoid placing duplicate or near-duplicate financial events across partitions.

For time-dependent market data, prefer chronological or event-grouped splits where feasible.

### 9.2 Classification metrics

| Metric | Purpose |
|---|---|
| Accuracy | Overall classification correctness |
| Macro-F1 | Balance across sentiment classes |
| Per-class precision | Reliability of individual signals |
| Per-class recall | Frequency of missed signals |
| Confusion matrix | Identify severe direction inversions |
| Directional precision | Reliability of Bullish/Bearish predictions |
| Brier score | Quality of probabilistic predictions |
| Log loss | Penalize confidently incorrect probabilities |
| Calibration error | Agreement between probability estimates and outcomes |
| Risk–coverage curve | Accuracy versus abstention rate |

Of particular concern are **Bullish ↔ Bearish inversions**, which may be significantly more costly than errors involving Neutral.

### 9.3 Summary evaluation

Measure:

- Factual correctness.
- Preservation of material numerical values.
- Identification of the correct entity.
- Important information omitted.
- Unsupported or hallucinated claims.
- Brevity and clarity.
- Consistency between the summary and sentiment classification.

A curated, manually assessed evaluation set is appropriate for the initial prototype.

### 9.4 Threshold optimization

For decision models with probability distributions:

1. Run each example through the model once.
2. Save the complete class probabilities.
3. Sweep possible thresholds over saved outputs.
4. Measure precision, coverage, and abstention rate.
5. Choose a threshold based on a predefined business objective.
6. Report final metrics on untouched holdout data.

Example decision policy:

```python
def decide(probabilities, threshold=0.80):
    label = max(probabilities, key=probabilities.get)

    if probabilities[label] < threshold:
        return "abstain"

    return label
```

The threshold of 0.80 is illustrative, not a recommended calibrated default.

For actual probability calibration, evaluate techniques such as temperature scaling or multinomial calibration on a dedicated calibration split.

For the generative model's ordinal scores, evaluate discrete threshold policies separately. Do not treat the ordinal score as a probability.

### 9.5 Model selection

Choose the architecture using a quality–latency Pareto comparison.

A model that is slightly more accurate but dramatically slower may be inappropriate for time-sensitive financial analysis.

Conversely, speed is not useful if it introduces unacceptable directional errors.

---

## 10. Latency Benchmarking

The primary business requirement is low latency.

Measure latency at both the individual-stage and end-to-end levels.

### Stages to measure

```text
Request received
    |
    +-- URL validation
    +-- HTTP fetch
    +-- HTML parsing
    +-- Article normalization
    +-- Input tokenization / prompt evaluation
    +-- Model inference
    +-- Output validation
    +-- Response serialization
    |
Response returned
```

### Metrics

- Median latency (p50).
- p95 latency.
- End-to-end request duration.
- Model load/cold-start time.
- Prompt evaluation duration.
- Generation duration.
- Output tokens per second.
- Peak memory consumption.
- Throughput under concurrent requests.
- Structured-output failure rate.

Distinguish cold starts from warm inference.

Also distinguish model inference latency from HTTP retrieval time, because fetching arbitrary public websites can dominate total response latency.

### Performance optimizations

1. Reuse HTTP connections.
2. Add bounded request timeouts.
3. Cache previously analyzed articles.
4. Cache extracted content separately from model results.
5. Avoid sending navigation, advertising, and unrelated page content to the model.
6. Keep prompts and generated summaries short.
7. Keep frequently used models loaded.
8. Avoid unnecessarily large context allocations.
9. Evaluate quantization choices.
10. Use asynchronous FastAPI endpoints for I/O operations.

For controlled benchmarks, reuse stored extracted article text so external network variability does not distort model comparisons.

For real-world benchmarks, run the entire URL-to-response workflow.

### Streaming considerations

Structured JSON generation may provide limited user value until a sufficient portion of the response is generated and validated.

If time-to-first-useful-result is important, consider a separate preliminary sentiment endpoint or staged response protocol.

---

## 11. Experimental Comparison

Benchmark these three complete pipelines:

| Pipeline | Sentiment | Summary |
|---|---|---|
| A | Qwen3.5 4B | Same model, same request |
| B | Qwen3.5 9B | Same model, same request |
| C | Clef-Flash 9B | Qwen3.5 4B |

Optional additional baseline: FinBERT for sentiment classification.

Ensure identical evaluation inputs, label definitions, article extraction, and hardware conditions.

For the hybrid system, measure sequential and concurrent inference where supported.

### Benchmark result template

| Metric | Qwen 4B | Qwen 9B | Clef + Qwen 4B |
|---|---|---|---|
| Macro-F1 | TBD | TBD | TBD |
| Directional precision | TBD | TBD | TBD |
| Summary factual accuracy | TBD | TBD | TBD |
| Valid JSON rate | TBD | TBD | TBD |
| Warm p50 latency | TBD | TBD | TBD |
| Warm p95 latency | TBD | TBD | TBD |
| Peak memory | TBD | TBD | TBD |

**Do not populate this table with estimated measurements.** Record actual benchmark results and their test conditions.

---

## 12. Reliability and Security

Because the backend accepts arbitrary public URLs, URL handling is a security-sensitive part of the design.

Minimum protections:

- Reject non-HTTP(S) URL schemes.
- Block loopback, private, link-local, and internal network destinations.
- Validate resolved IP addresses and redirects.
- Apply strict fetch timeouts and response-size limits.
- Restrict unsupported content types.
- Avoid unrestricted redirect chains.
- Apply rate limits where appropriate.
- Treat retrieved web content as untrusted data.
- Prevent article instructions from overriding the model's system instructions.
- Validate structured model responses with Pydantic.
- Return explicit errors for extraction and inference failures.
- Never expose local Ollama APIs publicly without suitable access controls.

Respect website access policies and content licensing.

For a production financial-news service, licensed data feeds would generally be preferable to arbitrary public-web scraping.

---

## 13. Future Low-Latency Architecture

If extremely fast preliminary classifications become necessary, introduce a two-stage pipeline:

```text
Incoming headline
       |
       v
Decision model
       |
       v
Preliminary sentiment
       |
       v
Frontend immediate update

Full article available
       |
       v
Generative LLM
       |
       v
Final summary + sentiment
       |
       v
Frontend final update
```

A specialized decision model could generate an early classification based on a headline, while the generative model later evaluates the full article.

Preliminary and final signals should be clearly distinguished.

This architecture would be justified if benchmarking demonstrates a meaningful reduction in time to the first useful market signal.

It is not necessary for the initial prototype.

---

## 14. Implementation Plan

### Recommended technology stack

**Backend:** Python, FastAPI, Pydantic, httpx, BeautifulSoup or a dedicated article extraction library.

**Inference:** Local Ollama with Qwen3.5 9B initially.

**Frontend:** React + Vite with a minimal form and results display.

**Testing:** pytest for backend behavior, a reproducible benchmark script for model evaluation.

**Dependency management:** uv for Python, npm for frontend development.

### Proposed repository structure

```text
veritrade-analytics/
├── backend/
│   ├── app/
│   │   ├── main.py
│   │   ├── schemas.py
│   │   ├── extraction.py
│   │   ├── inference.py
│   │   └── services.py
│   ├── tests/
│   └── pyproject.toml
├── frontend/
│   ├── src/
│   └── package.json
├── evals/
│   ├── benchmark.py
│   ├── metrics.py
│   ├── calibration.py
│   └── results/
├── docs/
│   └── ai-conversation-log.md
└── README.md
```

### Eight-hour development budget

| Task | Budget |
|---|---:|
| Backend API and article extraction | 1.5 h |
| Ollama inference and structured output | 1.5 h |
| Minimal frontend | 1 h |
| Evaluation harness and model comparison | 2 h |
| Tests, latency measurements, README, AI log | 2 h |
| **Total** | **8 h** |

If time becomes constrained, prioritize a reliable end-to-end implementation, a small reproducible evaluation, and excellent architectural documentation over additional model variants.

---

## 15. Final Architectural Decision

**Selected initial architecture**

Single generative model: Qwen3.5 9B through Ollama, with structured JSON and Pydantic validation.

**Primary optimization candidate**

Qwen3.5 4B for reduced latency and resource usage.

**Experimental alternative**

Clef-Flash 9B for dedicated sentiment classification, combined with a smaller generative summarizer.

**Optional reference classifier**

FinBERT for domain-specific classification benchmarking, subject to its licensing restrictions.

### Acceptance criteria

The prototype should demonstrate:

- A functioning URL-to-analysis workflow.
- Consistently valid structured responses.
- Financially meaningful sentiment definitions.
- Reproducible classification evaluation.
- A small collection of reviewed complete-article examples.
- Measured rather than assumed latency characteristics.
- Explicit acknowledgment of uncertainty and limitations.
- Documented engineering trade-offs.

### Final principle

> Start with the simplest complete system, evaluate its accuracy and latency empirically, and introduce specialized decision models only when the results justify the additional architectural complexity.

---

## 16. References

### Official Ollama documentation

- [Ollama — Structured Outputs](https://docs.ollama.com/capabilities/structured-outputs)
- [Qwen3.5 model library](https://ollama.com/library/qwen3.5)
- [Gemma 4 model library](https://ollama.com/library/gemma4)
- [Clef-Flash model library](https://ollama.com/library/clef-flash)
- [Clef model library](https://ollama.com/library/clef)

### Datasets and models

- [Financial PhraseBank](https://huggingface.co/datasets/takala/financial_phrasebank)
- [Twitter Financial News Sentiment](https://huggingface.co/datasets/zeroshot/twitter-financial-news-sentiment)
- [SEntFiN research paper](https://arxiv.org/abs/2305.12257)
- [FinEntity dataset](https://huggingface.co/datasets/yixuantt/FinEntity)
- [ProsusAI FinBERT](https://huggingface.co/ProsusAI/finbert)

### Evaluation methodology

- [Scikit-learn classification metrics](https://scikit-learn.org/stable/modules/model_evaluation.html)
- [Scikit-learn probability calibration](https://scikit-learn.org/stable/modules/calibration.html)

---

*This document summarizes the proposed design and evaluation strategy. Model rankings, prediction quality, inference latency, and optimal classification thresholds remain hypotheses until measured on the target hardware and benchmark data.*
