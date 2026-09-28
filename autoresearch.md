# Autoresearch: 1secimage Qwen Image 2.1 warm-path latency

## Objective

Reduce warm 512px and 768px end-to-end generation latency on the local RTX 4060 Ti 16GB toward 1 second while preserving usable image quality, prompt fidelity, VRAM headroom, and reproducibility. The active work is inference optimization for the installed Viggle v0.2.1 Turbo student.

## Metrics

- **Primary**: median warm 512px prompt-submission-to-success seconds, lower is better.
- **Secondary**: 512px text-encode, denoise, and VAE seconds; 768px latency; peak VRAM; decoded output stability.
- **Quality gate**: compare fixed-seed outputs against the official v0.2.1 six-step reference. A large unexplained image or prompt-alignment change is rejected for production adoption.

## Measurement protocol

Run `D:\Project\1secimage\scripts\comfy_bench.py` against ComfyUI at `http://127.0.0.1:8188`. Use two warmups and at least five measured requests with unique prompt suffixes (`--vary-prompt`). Treat `execution_cached` sampler nodes as invalid for latency samples. The lean server uses classic cache, no previews, no API nodes, in-memory database, 1.0 GiB reserve, PyTorch `2.14.0+cu130`, native Comfy Kitchen CUDA, and three async offload streams.

## Current accepted profiles

- **Quality default**: v0.2.1 r128, prescribed six-step schedule. Final clean stock-loader warm 512px median `2.085519 s`; text `0.602064 s`, denoise `1.156417 s`, VAE `0.303648 s`; peak `15,582 MiB`.
- **Fast default**: v0.2.1 r128, publisher-supported five-step compromise. Final clean stock-loader warm 512px median `1.890984 s`; text `0.570453 s`, denoise `0.982212 s`, VAE `0.295339 s`; peak `15,662 MiB`. Final clean 768px median `3.343637 s`; text `0.626469 s`, denoise `2.078024 s`, VAE `0.561162 s`; peak `15,588 MiB`.
- **Experimental**: v0.2.1 r128, four-step schedule. Warm 512px median `2.294962 s`, but its fixed-seed DINO cosine versus the official six-step reference is `0.949876` and DINO L2 is `0.316621`; it is not the production default.
- **Prescribed r256 reference**: v0.2.1 r256, six-step. Warm 512px median `2.725699 s`; used as the quality reference, not as the fast production path.

The historical old rank-64 result was approximately `1.655577 s` at 512px and `2.915651 s` at 768px. It must not be reported as a v0.2.1 result.

## What has been tried

- RAM-pressure cache under severe Windows commit pressure: rejected as a confounded baseline. The lean classic-cache server restored the intended policy.
- Positive-only CFG-1 conditioning with a GPU-resident lossless prefix cache: accepted. This removes the redundant negative text encode without changing the CFG-1 sampler contract.
- PyTorch `2.14.0+cu130` with native Comfy Kitchen CUDA: accepted for the old path and retained for v0.2.1. Logs report zero graph breaks and zero rogues.
- Three async offload streams: retained from the old-path sweep. No rank-256-sized penalty was observed on already-resident denoise steps; fault-time dense merge, ConvRot re-rotation, and re-quantization dominate LoRA cost.
- v0.2.1 r256 six-step: true warm baseline, approximately `2.726 s` at 512px.
- v0.2.1 r128 six-step: final clean stock-loader warm median `2.085519 s` at 512px; effectively unchanged from r256 in the local quality gate.
- v0.2.1 r128 five-step: final clean stock-loader warm median `1.890984 s` at 512px; selected fast compromise after quality review.
- v0.2.1 r128 four-step: approximately `2.295 s` at 512px; faster than six-step in some runs but retained as experimental because the local quality distance is larger.
- `LoraLoaderBypassModelOnly`: approximately `3.635 s`; rejected.
- Official-style unmerged runtime LoRA: approximately `4.029 s` for r128 and `4.270 s` for r256; rejected.
- No-LoRA and pre-bake controls: no accepted one-second path; pre-baking caused material output drift and no speedup.
- Direct per-block CUDA graph capture, QKV fusion, CPU text encoding, W4A8 text, lossy cache precision, alternate stream counts, cuDNN autotune, Comfy Kitchen attention, `--disable-fast-disk`, and direct Qwen graph replay: rejected based on latency, quality, VRAM, or correctness evidence.
- Scale-one LoRA allocation reduction: clean matched output was pixel-identical, but the timing comparison was cache-state-confounded and did not prove a win. The source change was reverted.
- Specialized exact INT8 ConvRot bake diagnostic: representative float patch math matched the runtime path, but quantized bytes differed in a few stochastic-rounding positions and the combined diagnostic hit a CUDA illegal-memory-access. No checkpoint was written; rejected without a speed claim.

## Quality evidence

The fixed-seed quality set is in `results/quality_v021/quality_v021_p1_metrics.json`. It compares r256 six-step, r128 six-step, r128 five-step, r128 four-step, and r256 four-step. The r128 six-step result is the closest to the official reference. Five-step is the selected fast compromise. Four-step remains experimental.

## Current hypothesis

1. The current v0.2.1 fast path is approximately `1.89 s` at 512px and `3.34 s` at 768px, not one second.
2. The fixed costs are material: five-step text encoding is about `0.57 s`, denoise about `0.98 s`, and VAE about `0.30 s`; the first prefix-cache/model-fault pass is especially expensive.
3. The stock runtime LoRA path is the correct local path. Generic pre-baking, exact-bake experiments, and unmerged runtime branches are not correctness-preserving or faster here.
4. A one-second result on this card likely requires a smaller purpose-built student, a different architecture, more aggressive model distillation/pruning, or different hardware rather than another cache/schedule flag.

## Future research: 2-step and 1-step students

This is a separate model-research phase. It must not replace or silently alter the accepted v0.2.1 r128 5-step/6-step runtime paths.

### Baseline lock

- Keep the v0.2.1 r128 six-step output as the quality reference and the five-step output as the fast baseline.
- Store every 2-step or 1-step candidate as a separate artifact, schedule definition, and result record.
- Do not describe sigma-list truncation as student distillation. A candidate is a new student only if its weights or adapter are trained for that step count.
- The current five-step timing has approximately `0.57 s` text encoding, `0.98 s` denoise, and `0.30 s` VAE. A rough step-count-only extrapolation gives about `1.06 s` for one step and `1.26 s` for two steps; these are planning estimates, not measurements. The fixed text and VAE costs must still be optimized or overlapped.

### Phase 0: reproduce the training context

Before local training, identify the exact teacher, student initialization, dataset construction, noise/timestep schedule, loss, optimizer, LoRA/full-parameter choice, precision, and checkpoint format. The reported Viggle scale of approximately `19.4k` real user requests, roughly `1,000` student optimizer steps, and `16x B200` GPUs is useful provenance, but it is not a reproducible recipe without those details. Confirm data and weight licensing before collecting or training on local user data.

### Phase 1: candidate evaluation harness

Build a fixed evaluation set before training candidates. Include multiple prompt categories, seeds, square and rectangular resolutions, and reference-image cases. Compare every candidate against the locked six-step reference and the current five-step fast baseline using:

- CLIP image cosine and prompt alignment.
- DINO image cosine and L2 distance.
- Fixed-seed visual stability and human review for artifacts or prompt failures.
- Warm 512px end-to-end latency, stage timings, peak VRAM, and execution success.

A candidate is not production-ready merely because it generates an image. A large unexplained quality regression, malformed output, excessive VRAM, or unrepeatable result rejects the candidate.

### Phase 2: two-step student

Initialize from the accepted r128 student and test a genuine two-step training objective. The default teacher target should be the accepted six-step student, with the 40-step Qwen teacher retained as an upper-bound quality reference when the training recipe requires it. Compare several small, isolated objectives rather than combining data, schedule, and architecture changes at once. Start with a short run and validate the output contract before scaling training.

### Phase 3: one-step student

Attempt a one-step student only after the two-step candidate has a stable artifact, a measured quality result, and a reproducible training command. Prefer a progressive or curriculum path from two steps to one where evidence supports it. A one-step candidate remains experimental even if it reaches the latency target until the broad quality gate and human review pass.

### Phase 4: runtime adoption

Only after a candidate passes training, quality, and reproducibility gates should it be loaded into the INT8 ComfyUI runtime. Measure the complete text-encode, denoise, and VAE path, not just the sampler. Keep the current dashboard, cache policy, and v0.2.1 LoRA path available as a rollback path. Runtime optimization and student research must be measured as separate changes.

## Two-step student: result

A genuine two-step student was built and measured end to end. It is fast and correct as a schedule, but it does not recover six-step quality, and training a correction adapter on top did not change that.

- Teacher export: `scripts/export_qwen21_teacher.py` wrote 8 exact six-step teacher trajectories (pre-final state and final denoised latent) plus conditioning to `results/qwen21_teacher_trajectories.json`.
- Training: `scripts/train_qwen21_two_step.py` trains a rank-1 correction on the 32 attention output projections, regressing the first student Euler state and the final denoised latent.
- Evaluation: `scripts/eval_qwen21_student.py` measures warm latency with the same cache-defeating prompt variation as the accepted harness, and `scripts/quality_eval.py` scores against the locked six-step reference.

Measured results:

- Two-step schedule, no adapter: warm median `1.445099 s`, versus the accepted five-step `1.890984 s` and six-step `2.085519 s`. The schedule is a real `0.44 s` win over the current fast default.
- Two-step schedule with the trained rank-1 adapter: warm median `1.406741 s`.
- Quality gate against the locked six-step reference: trained DINO cosine `0.5216`, CLIP cosine `0.8433`; untrained two-step DINO cosine `0.5183`, CLIP cosine `0.8491`. The trained student is slightly worse than no training at all.
- Holdout experiment: 24 updates at LR `4e-4` with linear decay, 6 train and 2 holdout samples. Holdout loss moved `0.46706` to `0.46405` and `0.76459` to `0.76571`. Flat within noise, so this is not an under-trained run.

Conclusion: the two-step schedule is the fastest local path, but a rank-1 attention output-projection correction on 8 samples cannot close the two-step quality gap. The binding constraint is objective capacity and data, not steps, rank within this configuration, or learning rate. The accepted five-step and six-step profiles remain production. This is a documented limitation, not a reason to adopt a degraded student.

## Training-path fixes in ComfyUI

Two real defects blocked training Qwen-Image-2.1. Both are guarded by `comfy.model_management.in_training`, so accepted inference is unchanged.

- `_gated_residual` in `comfy/ldm/qwen_image21/model.py` mutated the running hidden state in place across all 32 blocks. Autograd rejected this, and with checkpointing it produced silently wrong gradients at roughly `1e17`. The training branch is now out-of-place.
- `cast_bias_weight` in `comfy/ops.py` handed out views into a reusable asynchronous cast buffer. A later block's cast overwrote the same storage before backward consumed a saved RMSNorm weight. During training it now allocates a dedicated cast destination and still uses the offload stream.

Verified inference-neutral: a control run on clean code and a run with both fixes produced pixel-identical output, MAE `0.0`. The `0.015108` MAE against the stored accepted image reproduces exactly with and without the fixes, so it is pre-existing run-to-run nondeterminism in this stack.

Checkpointed backward additionally requires synchronous offload. With async streams, 32-block checkpointing produced `253,952` non-finite gradient elements. Enabling anomaly detection changed timing enough to hide the race, so anomaly mode must not be used to validate this path.

## Phase 2 findings: what the 2-step problem actually is

This phase was preregistered before measurement in `research/PREREGISTRATION_2STEP.md`, with
rival hypotheses and discriminating predictions in `research/prediction_rival_matrix.csv`
(schema v2.0, validated). The previous section above is preserved unchanged as the prior
state, including its negative result.

### The previous conclusion was not supported by its own evidence

The prior conclusion was that "the binding constraint is objective capacity and data, not
steps, rank, or learning rate." Two things are now established about that claim.

- **"Not rank" was never tested.** Only rank 1 was ever run. A rank-1 result cannot exclude
  rank as a constraint.
- **The evaluation was one prompt and one seed.** The eight teacher trajectories came from a
  single prompt, and the quality gate used a single image. A twelve-cell re-measurement of
  that same cell shows the single-sample numbers are systematically optimistic and, worse,
  have a heavy failure tail the single sample could not see.

A second rival was also never excluded: that the trained adapter was inert in the forward
pass. The saved adapters are not zero (up-matrix L2 `0.134` for the 1-step file and `0.913`
for the 2-step file), so the adapter is not trivially empty, but whether the injection
actually reached the forward pass during the end-to-end evaluation is still unverified.
That check remains open.

### Validity control

`NC-HARNESS` passed. A regenerated six-step output at the locked prompt and seed scored
DINO cosine `0.9990` and pixel MAE `0.00155` against the stored locked reference, matching
the recorded `0.999034` and `0.0015497`. The harness reproduces the reference.

One deviation: the first sweep was run at the bench default seed 42 while the locked gate
uses seed 101, which produced a spurious DINO of `0.8665` for the six-step control. That was
a seed error, not a harness failure. The lock is seed 101.

### Finding 1: the quality cliff sits between three and two steps

Fixed prompt, seed 101, 512px, untrained r128 student, one sigma node removed at a time,
scored against the locked six-step reference.

| steps | schedule (raw sigma) | warm median s | DINO | CLIP |
|---|---|---|---|---|
| 1 | `1` | 1.118 | 0.4411 | 0.7335 |
| 2 | `1,0.78` | 1.275 | 0.8472 | 0.9104 |
| 3 | `1,0.75,0.25` | 1.446 | 0.9339 | 0.9727 |
| 4 | `1,0.75,0.5,0.25` | 1.596 | 0.9499 | 0.9838 |
| 5 | `1,0.875,0.75,0.5,0.25` | 1.772 | 0.9653 | 0.9867 |
| 6 | `1,0.9375,0.875,0.75,0.5,0.25` | 1.840 | 0.9990 | 0.9996 |

Sources: `results/frontier.md`, `results/p1b_cliff_quality.json`, `results/sweep_p3b_floor.json`.
Six to three costs only `0.065` DINO. Three to two costs `0.384` on this cell.

### Finding 2: across twelve cells, three steps equals four steps and two steps is bimodal

Four prompts and three seeds, every candidate scored against a six-step reference generated
for the same prompt and seed. `results/schedule_stepcount_generalization.json`.

| grid | mean DINO | median | min | max | mean CLIP |
|---|---|---|---|---|---|
| 1 step | 0.1997 | 0.0879 | -0.0143 | 0.5480 | 0.6625 |
| 2 steps, prescribed `1,0.25` | 0.4446 | 0.4324 | 0.1720 | 0.8199 | 0.7904 |
| 2 steps, tuned `1,0.85` | 0.6456 | 0.7813 | 0.1445 | 0.8973 | 0.8203 |
| 3 steps `1,0.75,0.25` | 0.8345 | 0.8411 | 0.5913 | 0.9567 | 0.9116 |
| 4 steps `1,0.75,0.5,0.25` | 0.8448 | 0.8413 | 0.6387 | 0.9519 | 0.9323 |

Three consequences.

1. **Three steps is the efficient frontier.** Its mean and median are indistinguishable from
   four steps (`0.8345` vs `0.8448` mean, `0.8411` vs `0.8413` median) while costing one
   fewer denoiser call, about `0.143 s`. Four steps buys almost nothing over three.
2. **Two steps is not merely weaker, it is unreliable.** The tuned two-step grid has median
   `0.7813` but mean `0.6456` and minimum `0.1445`. The typical case is close to three steps
   and the tail collapses. A reliability distribution, not a mean, is the right summary.
3. **One step is not a student at all.** Median `0.0879` and minimum `-0.0143` mean the
   typical one-step image is uncorrelated with, or anti-correlated to, the six-step reference.

The single prompt and seed used throughout the earlier work overstates quality by about
`0.10` for both three steps (`0.9339` observed versus `0.8345` measured mean) and the
prescribed two steps (`0.5503` observed versus `0.4446` measured mean). That cell is a
favourable draw, so single-cell conclusions about this model are unsafe.

### Finding 3: schedule choice explains a large share of the recorded 2-step gap

Searching the second sigma of an untrained two-step Euler grid, one prompt, seed 101,
DINO against the locked reference: `0.4228` at 0.10, `0.5503` at 0.25, `0.6629` at 0.40,
`0.7138` at 0.60, `0.7858` at 0.70, `0.8127` at 0.75, `0.8472` at 0.78, `0.8219` at 0.80,
`0.7930` at 0.85, `0.7751` at 0.90. Sources: `results/p2_grid_quality.json`,
`results/p2b_grid_quality.json`.

On this cell the best grid beats the prescribed grid by `+0.297` DINO with no training at
all. The prescribed `[1.0, 0.25]` grid is a poor two-step schedule, and the earlier
correction experiment was trained to compensate for that choice rather than for a real
student deficiency.

Two corrections to this finding. The single-cell optimum at 0.78 was found at the edge of
the searched range, so per the preregistration it is reported as a bound and the range was
widened. Over twelve cells the ranking is monotone in the second sigma across the tested
values but the best value again sits at the edge of that range, so the optimum is not
bracketed. Grids starting below sigma 1.0 are rejected by the accepted harness and were not
tested. And per this project's own rule, **none of this is distillation**: a tuned schedule
is an ablation, not a student.

### Finding 4: step reduction cannot reach one second on this card

Fitted from six measured step counts at a fixed prompt, 2 warmups and 5 measured requests
each with prompt variation, every sample verified latency-valid with no cached sampler node.
`results/latency_floor.json`.

- `denoise(N) = 0.3654 + 0.1434 * (N-1)` seconds, max absolute fit residual `0.0223 s`.
- The first denoiser call costs `0.3654 s`, a marginal call costs `0.1434 s`. The one-off
  **first-call surcharge is `0.2220 s`**, about 1.55 marginal calls.
- Non-denoise fixed cost is `0.7799 s`: text encode `0.4830`, VAE decode `0.2735`, save
  `0.0228`.
- One measured step already costs `1.1177 s`. The model predicts `1.1453 s`.
- A zero-denoise pipeline floor is `0.7799 s`.

Therefore a **perfect** one-step student lands at about `1.12 s`, and the floor with no
denoising at all is `0.78 s`. The one-second target is not reachable by reducing step count
on this card. Three specific terms must be attacked: the `0.780 s` non-denoise fixed cost,
the `0.365 s` first denoiser call, and the per-step `0.143 s`.

The first-call surcharge is the most actionable of these, and it is not a quality problem.
It is LoRA and ConvRot machinery, consistent with the recorded hypothesis that fault-time
dense merge plus re-rotation and re-quantization dominate. Note the ordering this implies:
**the first-call surcharge `0.2220 s` is larger than the entire `0.1434 s` saved by dropping
from two steps to one**, and it is larger than the `0.1434 s` saved by dropping from four to
three. Removing it is worth more than any single step reduction except going from six to one.

Caveat on the fixed-cost term: text-encode cost scales with prompt length. This fit used a
16-word prompt and measured text encode at `0.483 s`; the accepted long-prompt run measured
`0.570 s`. The `0.780 s` floor is therefore a lower bound for long prompts, and the
one-step figure for the accepted long-prompt profile is closer to `1.34 s`.

### What this changes

- The distillation target is two steps only if a real student can make it *reliable*, which
  means closing a gap of `0.19` against the tuned untrained two-step mean and eliminating a
  failure tail. A two-step student that merely matches the tuned untrained grid is worth
  nothing.
- The immediately available, no-training result is the **three-step profile**: it matches
  four steps in quality at about `1.45 s` with this prompt, against `1.89 s` for the accepted
  five-step default. It needs a wider quality gate before adoption because the current gate
  is one prompt and one seed, and it needs latency re-measured on the accepted long-prompt
  protocol.
- A one-step student is not currently a credible target at this quality bar and should be
  deferred, not retried at a higher rank.
- The largest single latency term is not the denoiser step count. The first-call surcharge
  and the text encoder plus VAE floor dominate, and a student whose LoRA is baked into its
  own base weights would remove the first-call surcharge entirely.

### Finding 5: on the accepted long-prompt protocol, three steps to two steps buys nothing

Re-measured with the accepted benchmark prompt, 2 warmups and 5 measured requests, prompt
varied, every sample latency-valid with no cached sampler node. `results/sweep_p4_longprompt.json`.

| grid | warm median s | text s | denoise s | VAE s |
|---|---|---|---|---|
| 5 steps (accepted fast default grid) | 2.3393 | 0.7520 | 1.1970 | 0.3435 |
| 4 steps | 2.1412 | 0.7025 | 1.0906 | 0.3325 |
| 3 steps | 2.0208 | 0.7050 | 0.9505 | 0.3280 |
| 2 steps, tuned grid | 2.0122 | 0.7900 | 0.8260 | 0.3380 |

Three steps is `0.319 s` faster than five. Three to two is `0.009 s`, which is inside the
run-to-run spread of the text-encode stage. So on the accepted protocol, going from three
steps to two steps costs a large amount of reliability, at `0.845` mean DINO against
`0.6456` with a failure tail down to `0.1445`, and buys no measurable latency.

That is the practical conclusion of this phase: the two-step target that motivated the
earlier program is not worth pursuing on this card, because the latency it was supposed to
buy is not there.

**Machine-state caveat, and it matters.** The five-step accepted profile measures
`1.890984 s` in the recorded accepted session but `2.3393 s` in this session, with every
stage inflated by roughly 20 to 45%: text `0.570` versus `0.752`, sampler `0.982` versus
`1.197`, VAE `0.307` versus `0.344`. `nvidia-smi` during the run showed no clock or power
throttling (`clocks_event_reasons.active` `0x0`, 60 degrees C, 2790 of 3105 MHz) and the
GPU was otherwise idle, so the most likely cause is host-side contention from background
desktop software. Consequences: absolute latencies in this session must not be compared to
the accepted numbers, and the absolute `1.12 s` one-step projection in Finding 4 is a
lower bound. The step-to-step ratios, the first-call surcharge fraction, and the ordering
across step counts are internally consistent within a single session and are what the
conclusions rest on.

## Next experiments

Ordered by what the phase-2 evidence now supports. Each names the decision it informs.

1. **Bracket the two-step sigma optimum and check the failure tail per prompt.** The
   twelve-cell sweep is monotone in the second sigma with the best value at the edge of the
   tested range, so the optimum is not bracketed, and the tuned two-step grid has a
   minimum of `0.1445` across cells. Widen the range and record the full per-cell
   distribution, not the mean. This is cheap and it decides whether a two-step student is
   worth training at all.
2. **Adopt or reject the three-step profile.** It matched four steps on twelve cells at
   about `0.143 s` less. It needs latency re-measured on the accepted long-prompt protocol
   and a wider quality gate before it can replace the five-step default.
3. **Fix the evaluation before any further training.** Every quality number in this project
   so far rests on one prompt and one seed, which the twelve-cell sweep shows overstates
   quality by about `0.10`. Any distillation run judged on the old single-cell gate will
   produce a result that does not survive. Export teacher trajectories across several
   prompts, not one.
4. **Close `NC-INJECTION`.** Generate with and without the trained adapter on identical
   seed, prompt and schedule and confirm the images differ by more than the `0.0151`
   nondeterminism floor. If they are identical, the earlier training result is untested
   rather than negative, and that must be recorded before any new training claim.
5. **Only then train a two-step student**, against the *tuned* grid, with the objective and
   capacity ablations from the preregistration: capacity first at fixed objective, then
   objective at fixed capacity, decided on holdout latent error before any quality gate.
6. **Attack the first-call surcharge, not the step count.** It is `0.2220 s`, larger than any
   single step saving, and it is LoRA plus ConvRot machinery rather than a quality term. A
   distilled student whose adapter is baked into its own base weights should not pay it.
   This is the highest-leverage latency work available and it does not require a new model.
7. **Treat the text encoder and VAE as first-class research targets.** Together they are
   `0.780 s`, the majority of any one-step pipeline. No amount of denoiser distillation
   touches them.

Retained from the previous list, still valid:

- Do not retry a rank-1 `to_out.0`-only correction with more steps, higher learning rate, or
  more samples at this objective. That combination is a recorded dead end, and the phase-2
  evidence adds that it was also trained against a needlessly poor target grid.
- Do not retry the exact bake in its current form. Its representative float patch math
  matched, but quantized requantization differed in a few stochastic-rounding bytes and the
  combined diagnostic hit a CUDA illegal-memory-access. Item 6 is a different bake with a
  different goal and needs its own isolated per-layer seed and byte-equivalence harness.
- Do not retry CPU text encoding, lossy cache precision, QKV fusion, direct graph capture,
  pre-baked checkpoints, bypass LoRA, or unmerged LoRA without a new measured bottleneck.
- Expand the fixed-seed quality gate to more prompts, seeds, reference-image cases, and
  768px before widening any production default. This is now a blocking prerequisite rather
  than a nice-to-have, for the reason in item 3.
