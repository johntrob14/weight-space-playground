# Lab notebook

## Step 1 (2026-11-10): the edit. Verified.
See notes/plan.md. Data: results/step1_diff.log, results/verify_step1.log.

## Step 2 (2026-11-10): do activation differences find the edited neurons? Partly.
Data: 200 unedited-model and 200 edited-model published MATH-500 generations, teacher-forced through the unedited
model in bf16 (3.48 M tokens; scripts/collect.py). Loop tokens 374,784; productive tokens 946,422 (looped texts),
170,375 (17 non-looped texts), 1,320,433 (edited-model texts). Analysis: scripts/analyze_step2.py -> results/step2.log.
Deviations: loop tails cut 2,048 tokens after onset; edited-model texts cut at 8,192 generated tokens.

Hits among the top 64 neurons (64 edited per layer, chance about 1.1), rankings fit on all problems:

| Ranking | L15 | L19 |
| --- | --- | --- |
| abs difference of means, loop vs productive tokens (within generation) | 28 | 18 |
| same, Cohen's d | 19 | 12 |
| same, AUROC | 21 | 13 |
| loop tokens vs edited-model productive tokens, abs difference | 27 | 17 |
| between prompts: looped vs non-looped generations, abs difference | 16 | 7 |
| between prompts, productive tokens only | 13 | 14 |
| activation difference x write along our own residual direction (signed) | 39 | 28 |
| activation difference x write along the published u (signed; the editor's criterion) | 47 | 37 |
| baseline: weights only, abs(u . W) with the published u | 31 | 18 |
| baseline: abs mean activation | 20 | 9 |
| baseline: activation variance | 16 | 14 |
| baseline: lowest index first | 15 | 13 |
| baseline: shuffled token labels (5 draws, from the 1% sample) | 5-14 | 8-13 |
| baseline: random | 1 | 1 |

- Independent recompute from the 1% raw-activation sample gives the same 28 and 18.
- Rankings are stable across problem halves (Spearman 0.99) and the discovery-half rankings score the same.
- Our difference-of-means residual direction has cosine 0.73 (after L15) and 0.75 (after L19) with the published u.
- Edited neurons are not uniformly more active in loops: 30/64 (L15) and 38/64 (L19) have higher mean activation
  on loop tokens; the rest are lower.
- Not yet checked: token-position confound (loop tokens come later in the text than productive tokens).

Open hypotheses for steps 3-4:
- H1: neurons outside the edited 64 that the activation ranking prefers work as well when edited (the 64 are not
  special). Test: same edit formula on the ranking's top 64; compare the loop proxy and generation loop rate.
- H2: the edited neurons are high-variance, low-index neurons with no single function. Test: step 3 labels vs
  random neurons matched on variance.
