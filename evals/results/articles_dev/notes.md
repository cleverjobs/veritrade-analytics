## Notes

Hand-written after reading the tables above; counts and my own descriptions only, no article text. With 10
articles per class, read these as a check that the dev300 operating points transfer, not as a ranking.

**Every served rule holds.** No served rule clearly fails (no inversion CI above the 2% cap, no coverage
collapse), so every dev300 threshold is frozen as tuned and nothing was re-picked on the articles.

**/signal.**
- clef-flash v4 at p >= 0.80 scores F0.5 0.848 on the lede and 0.907 on the full body, close to its dev300
  0.874. It makes one inversion in 20 calls on the lede and none on the full body.
- nimble v1 at p >= 0.97 answers far more articles than tweets (coverage 60-67% against 26% on dev300) at
  F0.5 0.88-0.90: articles state their news more plainly than tweets, so the peaked model is sure more often.
- qwen3.5:0.8b with `directional_tau` 0.28 scores 0.685 on the lede against 0.595 for the plain rule (a tie
  at n=30); on the full body the tau rule wins clearly (+0.229 [+0.022, +0.560]). It still misses about half
  the directional articles, so it remains the fast, low-recall option.
- Lede vs full is a tie for all three models; the lede (the /signal default) is two to three times faster.

**/analyze.**
- Articles are easier than tweets for every generative model: qwen3.5:4b v1 0.880, qwen3-instruct 0.798 and
  ministral 0.741, against 0.75 / 0.66 / 0.66 on dev300.
- The tightened summary rule (v5) changes no ministral call, costs qwen3-instruct one (a tie), and costs
  qwen3.5:4b two bullish and one bearish article plus one false alarm (-0.080 [-0.178, -0.014]). It therefore
  keeps v1 for /analyze; its /summary endpoint still gets the tightened summary prompt.
- Summary-first is a tie on the articles (qwen3.5:4b -0.013, ministral +0.058) and never wins on dev300,
  while its time to the score is two to three times longer. Every model stays sentiment-first.
- The wrong calls are mostly neutral articles called directional (3-5 of the 10 per model, mostly bullish).
  Price-move headlines are not the problem they were on tweets (0-1 of the 5 directional ones missed per
  model), so the price-move / analyst-action rubric sentence was not added.
