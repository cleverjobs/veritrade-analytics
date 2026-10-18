## Notes from the error-analysis read

Hand-written after reading the tables above; counts and my own descriptions only, no dataset text. These are
tweets, not articles: the thresholds and latencies here are a starting point for the article dev set, not a
substitute for it.

**A decoding fix came out of this grid.** The qwen3.5 Modelfiles ship `presence_penalty 1.5`, and the client
only pinned temperature and seed, so "greedy" qwen3.5 output was pushed away from any token in the recent
context. With four shots whose last answer is neutral (or score 0), the generated qwen3.5:0.8b label disagreed
with its own first-token argmax on 84/300 (v2) and 62/300 (v4) tweets, and v2-ordinal emitted -2 for 104/300.
`FIXED_OPTIONS` now neutralises the penalties; every number on this page comes from the greedy re-run. The
label-logprob cells were identical before and after (logprobs are reported before the penalty is applied), and
seven cells re-run to replace contended timings reproduced their predictions exactly, so the cache and decoding
are deterministic.

**Decision models lead on precision.** clef-flash v4 (plain question, bare labels) reaches F0.5 0.874 at
min_p 0.80 with no inversions, answering 73% of tweets and calling 33% directional against a 35% directional
prior. nimble over-calls: with no threshold it calls 65-74% of tweets directional, and it only becomes precise
at min_p 0.97-0.98, where it answers about a quarter of the tweets (F0.5 0.843 for v1). The served
`directional_tau` rule runs at min_p 0, so for these peaked models it only removes their abstention: it loses
to the plain rule by 0.046 (clef-flash) and 0.303 (nimble) F0.5, both significant, and is not served for them.

**qwen3.5:0.8b is neutral-leaning, which caps the served label policy.** Its first-token argmax is neutral on
most directional tweets (0.3-7% signal rate across label configs), so a min_p threshold can only remove calls:
the plain rule (v2-label, no abstention) misses 162 of 200 directional tweets at F0.5 0.512. The served
`directional_tau` rule calls the likelier directional class when neutral leads and that class reaches tau;
at tau 0.28 it gives F0.5 0.730 [0.651, 0.795] with precision 0.91 / 0.92 and no inversions (+0.218
[+0.141, +0.307] paired). It is the largest single lever found here, it is now the model's best config, and it
is frozen (min_p 0, directional_tau 0.28).

**Knob verdict for qwen3.5:0.8b.** As served (argmax + min_p on label tokens vs |score| >= t on the ordinal
schema), the best label config beats the best ordinal config by +0.059 F0.5 [-0.068, +0.187]: not significant
at n=300. With the `directional_tau` rule the logprob knob wins clearly. The ordinal schema cannot borrow that
rule: this tokenizer fuses the space and the minus sign (" -") while the bare-space alternative keeps about a
quarter of the mass, so the sign is unobservable for negative scores. `score_sign_probabilities` used to read
those as bearish 1.0 (and miss the " -" mass behind positive scores). It now returns None for every negative
score in these cells (48, 32, 17 and 8 of 300), and serving falls back to the ordinal policy there.

**Prompt dimensions.**
- Rubric A (definitions) beats rubric B (impact questions) for every generative model. B's "is the change
  clear and material" framing makes the small models answer neutral: qwen3.5:0.8b v3-label calls 0.3% of
  tweets directional, and ministral drops from 0.660 to 0.440.
- Four synthetic shots help qwen3.5:0.8b on the label schema (0.365 to 0.512) but hurt every /analyze model:
  qwen3.5:4b 0.752 to 0.655, ministral 0.660 to 0.494, and qwen3-instruct v2 breaks the inversion cap. Shots
  also roughly double qwen3.5:4b's time to score.
- For the decision models the spread across prompts is small (clef 0.83-0.87, nimble 0.81-0.84). Bare labels
  with the short question are also the fastest variant (clef 468 vs 710 ms, nimble 751 vs 975 ms).

**Where the calls go wrong.**
- Inversions are nearly absent: one bullish tweet called bearish across all six best configs.
- Wrong calls are mostly gold-neutral tweets called directional (1-8 per model). Relative to its size, the
  earnings and guidance bucket holds the most of them (13 across the six models in 46 tweets, against 14 in
  128 for the catch-all bucket), which fits mixed results (a beat alongside a cut outlook) being the hard case.
- The real loss is recall: 34 (clef-flash) to 162 (qwen3.5:0.8b) directional tweets missed.
- The biggest miss bucket for every generative model is the price-move tweet: 33-44 of 57 directional ones
  missed, against 10 for clef-flash. The rubrics ask for economic impact, not tone, and the small models read
  a reported share-price move as not material, while the dataset labels such moves as directional. Whether to
  close that gap is a product question: on articles the impact reading is the intended behaviour.
- Analyst actions (upgrades, price targets) are handled well by the larger models (qwen3.5:4b catches 21 of
  22, clef-flash 19 of 22, with no wrong calls).
- 20 directional tweets get no correct call from any model; they are the first place to look for label noise
  or genuinely ambiguous items.

**Follow-ups after the grid (analyze).** Two cells per model, each paired against v1 on the same tweets
and never eligible as the best config.
- v5 = v1 with the summary rule tightened after the summary eval (2 to 3 sentences, at most 50 words, entity
  first). Only the summary line changes, yet the score moves: ministral +0.025 [+0.002, +0.052], qwen3-instruct
  -0.011 [-0.047, +0.023], qwen3.5:4b -0.021 [-0.054, +0.007]. On the article dev set qwen3.5:4b loses more
  (0.800 vs 0.880, gap -0.080 [-0.178, -0.014]: two more bullish articles missed), so the freeze keeps v1 for
  qwen3.5:4b /analyze and serves v5 to ministral and qwen3-instruct.
- `v1-summary-first` (the field-order check) is never better: ministral -0.017 [-0.088, +0.059], qwen3.5:4b
  -0.061 [-0.122, +0.003], and its time to the score is 1.0-1.5 s against 0.15-0.6 s, because the whole
  summary decodes first. Every /analyze model stays sentiment-first.

**The one manual revision.** Few-shot is dropped for /analyze (every frozen /analyze prompt is zero-shot).
The price-move / analyst-action sentence was not added: on the article dev set the /analyze models miss 0-1
of the 5 directional price-move headlines and no headline falls in the analyst-action bucket, so the tweet
misses do not carry over (see `evals/results/articles_dev/summary.md`). qwen3.5:0.8b keeps its four shots: they
help the label schema (0.365 to 0.512 plain).

**Timing provenance.** Wall times are per call at client concurrency 1, warm. Concurrent load on the shared
machine (mostly the Docker Desktop VM saturating the CPU) inflated some tails, so clef-flash v2 and nimble v2/v3
were re-run at 19:28-19:47 and all eight qwen3.5:0.8b cells at 20:11-20:25, each with identical predictions.
The tables show the re-run timings. The qwen3.5:0.8b re-run still shared the machine (load about 30): on a
quiet machine earlier the same calls measured p50 155-200 ms with p95 near 210-300 ms, against 209-276 ms and
up to 1.06 s here. nimble v3's tail is also somewhat elevated. The analyze follow-ups (v5, summary-first)
ran later at concurrency 1 on an Ollama shared with two other eval runs, so their timings are indicative only.
