# Does neuron-level interpretability recover a known weight edit?

Started 2026-11-10 on zeus (GTX 1650 Ti, 4 GB). Stage: understand (a hypothesis test with a known answer).

## Question

Astra's edit to Qwen3.5-0.8B (Sparks of Weight-Space Programming, experiment 8, run p50c) fixes thinking loops on
math problems by changing 64 MLP neurons' output weights in each of layers 15 and 19. Given only the unedited model
and looping / non-looping behavior, does ranking MLP neurons by activation difference find those neurons? Are they
individually interpretable? Does a neuron-level edit need fewer changed weights than a difference-of-means direction?

## What step 1 established (verified, see results/step1_diff.log and results/verify_step1.log)

- Edited checkpoint vs Qwen/Qwen3.5-0.8B @ 2fc0636: 2 of 488 tensors differ, L15 and L19 `mlp.down_proj.weight`.
- 64 input columns changed per layer = 64 MLP neurons per layer (down_proj is Linear(3584 -> 1024), weight
  [1024, 3584]; a column is one neuron's output weights). The columns equal Astra's `selected_neurons` lists.
- 131,072 values (64 x 1024 x 2); 130,188 differ in bfloat16 bits, the rest round back.
- The change is rank 1 per layer: W = W0 - 6 u (u^T W0) on the 64 columns (alpha fit 6.000; top singular value holds
  99.999% of the energy). After the edit each selected neuron writes -5x its original amount along u: the edit
  reverses the neuron's contribution along u, it does not remove it.
- u is a residual-stream direction (one per layer, cosine 0.85 between layers) that Astra got from a
  difference between repetitive and productive contexts. So the published edit is itself a difference-of-means
  direction restricted to 64 neurons. Step 4's baseline is the unrestricted version of the same idea.
- The 64 were chosen with activations, not weights alone: the top 64 by |u . W0[:, j]| overlap the edited set in
  31/64 (L15) and 18/64 (L19). The selected neurons hold 25% / 16% of the squared projection of W0 on u.
- Selected neurons are concentrated at low indices: 40/64 and 35/64 have index < 512 (9 expected by chance). The
  weights-only top 64 show the same skew (25 and 28 below 512), so neuron index carries structure in this model.

## Consequences for the design

1. **The "right answer" is one sufficient edit, not the unique set of responsible neurons.** Astra picked the 64
   strongest "repetition writers" from one forward pass. Precision and recall against the 64 measure agreement with
   Astra's criterion. So every ranking is also scored causally: apply the same edit formula to the method's top k
   neurons and measure looping.
2. **Looping is a property of where in a generation the model is, more than of the prompt.** The unedited model
   hits the token limit on about 93% of MATH problems. A contrast between prompts that loop and prompts that do not
   has few negatives and is confounded with problem difficulty. I will run the between-prompt contrast as asked and
   add a within-generation contrast (tokens inside the repetitive tail vs earlier productive tokens of the same
   generation), which is closer to what Astra used.
3. **A ranking by (activation difference) x (write along u) would recover Astra's set almost by construction.** The
   primary method uses activations only. The weighted version is reported separately and labeled as such.

## Data

- 200 full generations from the unedited model and 200 from the edited model on the same MATH-500 problems
  (first sample per problem), published in the logs repo (`exp8_math/eval/{unedited,p50c}/parts/math500_s*of10/`).
  Teacher-forcing these through the unedited model needs no generation.
- 7,275 MATH train practice problems in the logs repo (`exp8_math/practice/math_train.jsonl`) for our own generations.
- Prompt format (thinking on): user message = problem + "\nPlease reason step by step, and put your final answer
  within \boxed{}.", Qwen chat template, generation starts after "<think>\n". Sampling: temperature 0.6, top_p 0.95,
  top_k 20, presence penalty 1.5, max 32,768 new tokens.

## Steps

**Step 2. Neuron ranking on the unedited model.**
- Activations: input of `mlp.down_proj` in layers 15 and 19 (3,584 neurons each), all token positions.
- Labels: (a) per generation, looped (hit the limit with a repetitive tail) vs finished; (b) per token, inside the
  repetitive tail vs before loop onset. Loop onset from repeated-line detection, checked by reading 10 random cases.
- Rankings: absolute difference of mean activation; Cohen's d; AUROC per neuron. Split problems into discovery and
  held-out halves.
- Scores: precision and recall at k = 16, 32, 64, 128, 256 against the edited set, and average precision, per layer.
  Baselines: random ranking (expected precision 64/3584 = 1.8%), ranking by activation variance, ranking by index
  (because of the low-index skew), weights-only ranking by |u . W0[:, j]|.
- Also: cosine between our difference-of-means residual direction (outputs of layers 15 and 19) and the published u.

**Step 3. Do the edited neurons have one clear function?**
- For the 128 edited neurons and 128 random neurons from the same layers: top-activating contexts from math
  generations and from general text, labeled blind (unit type hidden) as one job / related offshoots / unrelated
  mixture, the protocol from the already-clean replication. Also 128 random directions in neuron space.
- Quantitative: share of each neuron's top activations that fall on repetitive-tail tokens; AUROC of each neuron
  for tail vs productive tokens.
- "Already clean" predicts the edited neurons are mostly one-job and that the job relates to repetition. The
  alternative is that they are generic writers along u with mixed or unrelated top contexts.

**Step 4. Edits compared by weights changed.**
- Neuron edits: W0 - alpha u_hat (u_hat^T W0) on the top-k columns from step 2 (u_hat = our own direction), and
  plain zeroing of the top-k columns; k from 1 to 256. Weights changed = 1024 k per layer.
- Difference-of-means baseline: the same rank-1 subtraction on all 3,584 columns (3.67 M weights per layer; one
  1024-vector to describe), with alpha swept.
- Controls: Astra's published edit (reference), random k neurons, random unit direction of equal norm.
- Metrics: (i) teacher-forced log-prob change on repetitive vs productive continuations (cheap, for the sweep);
  (ii) loop rate and accuracy from sampled generations for the few best settings.

## Fit on the 4 GB GPU (results/fitcheck.log, fitcheck2_*.log)

- Text model 752 M parameters; weights take 1,459 MiB of 3,715 MiB in 16-bit. The vision tower is dropped.
- fp16 is numerically closer to fp32 than bf16 (L15 activation relative error 0.17% vs 1.3%; KL to fp32 logits
  5e-6 vs 3e-4). bf16 does run on this card and is 2.6x faster for teacher forcing (about 900 vs 350 tokens/s).
  The published evaluation used bf16. bf16 -> fp16 conversion changes 267,427 of 752 M weights by at most 3e-8.
- Teacher forcing: a single pass fits up to 8,192 tokens in fp16 (2,587 MiB) and 4,096 in bf16 (3,000 MiB). Longer
  texts need chunked passes with the cache carried forward (not yet tested).
- Generation: about 31 tokens/s for one sequence, 46 tokens/s total at batch 4 (bf16); fp16 at batch 4 was slower
  than batch 1 (15 tokens/s). transformers is using slow fallback code for the linear-attention layers because
  `flash-linear-attention` and `causal_conv1d` are not installed.
- One 32,768-token generation takes about 18 minutes. Full-length loop evaluation is the part that does not fit in
  a reasonable time: 64 problems x 2 models would take about a day.

## Decisions needed before long runs

See the message to John of 2026-11-10: dtype, how to measure looping within the time available, and whether to
install the faster linear-attention kernels.

## Storage

Code and notes: ~/code/wsp-neurons. Results: ~/code/wsp-neurons/results. Model weights: ~/.cache/huggingface on the
NVMe disk, because $HF_HOME (/tmp/john-scratch) is a RAM-backed tmpfs. Logs repo clone: ext/ (untracked).
