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
at min_p 0.97-0.98, where it answers about a quarter of the tweets (F0.5 0.843 for v1). Both models are so
peaked that the directional-mass rule changes nothing for them.

**qwen3.5:0.8b is neutral-leaning, which caps the served label policy.** Its first-token argmax is neutral on
most directional tweets (0.3-7% signal rate across label configs), so a min_p threshold can only remove calls:
the best served config (v2-label, no abstention) misses 162 of 200 directional tweets at F0.5 0.512. Calling
the likelier directional class whenever its probability reaches 0.28, even when neutral wins, gives F0.5 0.730
[0.651, 0.795] with precision 0.91 / 0.92 and no inversions (+0.218 paired). That is the largest single lever
found here and needs a serving policy before it can be frozen.

**Knob verdict for qwen3.5:0.8b.** As served (argmax + min_p on label tokens vs |score| >= t on the ordinal
schema), the best label config beats the best ordinal config by +0.059 F0.5 [-0.068, +0.187]: not significant
at n=300. With the directional-mass rule the logprob knob wins clearly. The ordinal schema cannot borrow that
rule: its score-sign probabilities are unreliable for this tokenizer, which fuses the space and the minus sign
into one token.

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

**Candidates for the one manual revision.**
1. qwen3.5:0.8b /signal: serve the directional-mass rule (see Requests in the grid report) rather than
   rewording the prompt; tune it again on the article dev set.
2. /analyze rubric A: one sentence saying a reported price move or analyst action is itself directional
   evidence, if the article dev set shows the same price-move misses.
3. Few-shot sets for /analyze: drop them (zero-shot wins for all three models) or rebalance them so the last
   shot is not neutral.

**Timing provenance.** Wall times are per call at client concurrency 1, warm. Concurrent load on the shared
machine inflated the tails of seven cells (mostly the Docker Desktop VM saturating the CPU from about 18:15 to
18:35): four qwen3.5:0.8b cells, clef-flash v2 and nimble v2/v3. Those cells were re-run at 19:28-19:47 with
identical predictions, and the tables show the re-run timings; nimble v3's tail is still somewhat elevated.
