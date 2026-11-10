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

## Queued by John (2026-11-10): difference-of-means baselines, after steps 3 and 4
Compare several difference-of-means directions as loop fixes, including one that does not assume loops are known:
incorrect vs correct rollouts. John's hypothesis: that contrast alone steers away from loops.
Candidate contrasts: (a) loop vs productive tokens (step 2 direction); (b) incorrect vs correct rollouts, all
generated tokens; (c) incorrect vs correct, last-N tokens; (d) hit-the-limit vs finished rollouts; (e) unedited-model
text vs edited-model text on the same problems (uses the published fix, reference only).
Interventions: additive steering at the residual stream (hook) and the rank-1 weight edit W - a u u^T W.
Controls: random direction of equal norm; capability check on productive text.
Data note: only 15 of the 200 published unedited MATH-500 first samples are correct; the logs repo also has first
samples for GSM8K and other benchmarks that may add correct rollouts without new generation.

## Step 4 proxy (2026-11-10): unverified by generation
24 held-out looped generations, 512 loop tokens each; change in log-prob per token vs the unedited model
(results/step4_proxy_table.csv; scripts/step4_proxy.py). Baseline log-prob: loop -0.060, productive -0.386.

| Edit (alpha 6 unless noted) | Weights changed | Loop | Productive |
| --- | --- | --- | --- |
| published edit | 130,188 | -0.075 | -0.020 |
| top 64 by activation difference x write along our direction | 130,526 | -0.074 | -0.019 |
| published 64 neurons, our direction | 129,374 | -0.072 | -0.023 |
| top 64 by activation difference only | 126,958 | -0.038 | -0.011 |
| top 64 by weights only (our direction) | 130,746 | -0.043 | -0.016 |
| top 64 by the between-prompt ranking | 126,483 | -0.022 | -0.008 |
| all 3,584 columns, our direction, alpha 6 / 4 / 2 / 1 | about 6.4-7.0 M | -0.080 / -0.043 / -0.017 / -0.007 | -0.036 / -0.019 / -0.008 / -0.003 |
| top 16 / 32 / 128 / 256 by the write ranking | 33 k / 65 k / 261 k / 520 k | -0.041 / -0.057 / -0.095 / -0.112 | -0.014 / -0.017 / -0.021 / -0.022 |
| published 64 columns zeroed | 131,072 | -0.007 | -0.005 |
| 64 random neurons (3 draws) | about 124 k | -0.002 to 0.000 | -0.002 to 0.000 |
| published 64 neurons, random direction | 127,405 | 0.000 | 0.000 |

Reading: a ranking built without the published edit (activation difference x write along our own difference-of-means
direction) matches the published edit at the same weight count. Activation difference alone gets about half.
The full-direction edit needs about 50x the weights for the same loop effect and costs more on productive text.
Next: confirm with generations (scripts/gen_eval.py).

## Step 3 (2026-11-10): do the edited neurons have one clear function? Mostly, but less often than other neurons.
512 units, blinded packets (10 strong + 5 moderate excerpts, positive pole) from 3.94 M tokens (2.0 M FineWeb,
1.94 M unedited-model math reasoning). 16 labelling agents, 4 more relabelled a random quarter.
Files: results/step3/analysis.log, labels_unblinded.csv, unit_stats.jsonl; scripts/step3_*.py.

| Unit type (128 each) | One job | + related offshoots | One job, either | 95% CI | Repetition-related |
| --- | --- | --- | --- | --- | --- |
| Edited neurons | 25 | 57 | 82 (64%) | 56-72% | 18 |
| Matched neurons (variance, index) | 34 | 59 | 93 (73%) | 64-80% | 6 |
| Random neurons | 35 | 70 | 105 (82%) | 75-88% | 1 |
| Random directions | 12 | 22 | 34 (27%) | 20-35% | 2 |

- Fisher exact, one job either: edited vs matched p = 0.18; edited vs random p = 0.002; every neuron group vs
  directions p < 0.001. Repetition-related, edited vs matched: p = 0.017.
- Agreement between the two labelling passes (128 units): 90% on one-job-either, kappa 0.79. Same model both passes.
- Hand check: I judged 30 random packets blind; 24/30 agree with the agents on one-job-either, 3 disagreements each way.
- Density matters within type (denser units are labelled one job less often), as in the already-clean replication.
- Limits: positive pole only; half the edited neurons have their largest activation at the negative pole.

## Step 4 generation check (2026-11-10, in progress)
36 MATH train practice problems, 1 sample each, 8,192-token cap (deviation from 32,768), post's sampling incl.
presence penalty 1.5; results/gen_eval.jsonl; scripts/gen_eval.py.

| Condition | Looped by the cap | Finished | Correct |
| --- | --- | --- | --- |
| unedited | 26/36 | 1 | 1 |
| published edit | 3/36 | 8 | 8 |
| our top 64 (write ranking, our direction, alpha 6) | 5/36 | 1 | 1 |
| all columns, our direction, alpha 6 | 6/36 | 1 | 1 |

Our edits remove loops about as well as the published one (5/36 vs 3/36, p = 0.71) but do not make the model
finish within the cap (1/36 vs 8/36, p = 0.028). Reading 4 random problems: text under our edits is coherent
math reasoning that keeps second-guessing ("Wait, ...") without exact repetition. The proxy did not predict this.
Queued: published neurons with our direction, and our neurons with the published direction, to see whether the
direction or the neuron choice carries the finishing effect.

## Difference-of-means directions (rollout-level contrasts; results/directions_cosines.txt)
Cosine with the loop direction: incorrect-vs-correct 0.76 / 0.77 (L15 / L19), cut-vs-finished 0.79 / 0.80,
unedited-vs-edited text 0.99 / 0.99. Cosine with the published u: loop 0.73 / 0.75, incorrect 0.70 / 0.69,
cut 0.72 / 0.70. Only 15 correct and 14 finished rollouts among the 200.
