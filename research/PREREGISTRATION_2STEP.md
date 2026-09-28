# Preregistration: few-step Qwen-Image-2.1 student research (phase: 2-step)

Frozen **before** any new measurement for this phase. Timestamped 2026-09-27 14:58 IST.
Prediction/rival matrix: `research/prediction_rival_matrix.csv` (schema v2.0, validated `VALID_FOR_HUMAN_REVIEW`).

Accountable human owner: the project owner. AI assistance is material here and is recorded
in `autoresearch.md`. No claim in this file is a finding; every item is a candidate or a
planned measurement.

---

## 1. Scope and safety gate

- All artifacts are local. No prompt, image, or weight is sent to any external service.
- No personal data, no clinical or dual-use content. Model weights and a public LoRA only.
- **License precondition (P7 in the original matrix, held out of it because it is a
  compliance condition rather than a scientific prediction).** Before training on any
  artifact, its license and the authority to use it for local research must be recorded.
  These are Qwen-Image-2.1 and Qwen3-VL weights plus the Viggle v0.2.1 Turbo LoRA, already
  installed locally and already used for local inference in accepted production paths. If any
  license forbids distillation-based modification for research, that artifact is quarantined
  and findings are reported without it. This is a human accountability item and is not
  settled by any measurement in this file.
- Feasibility limits that shape every design choice: one RTX 4060 Ti 16GB, local inference
  stack, no multi-GPU training.

## 2. Frozen observations

Provenance for every row. These are what is already on disk; they are not re-derived here.

| ID | Observation | Source |
|---|---|---|
| O1 | Step-count quality curve, DINO cosine vs the locked r256 six-step reference, one prompt, one seed, 512px, untrained r128 student: **6-step 0.9990, 5-step 0.9653, 4-step 0.9499** | `results/quality_v021/quality_v021_p1_metrics.json` |
| O2 | **2-step** on the raw-sigma grid `[1.0, 0.25]` gives DINO cosine **0.5183**, CLIP cosine 0.8491 | `results/qwen21_student_eval_two_step_untrained.json` |
| O3 | Rank-1 LoRA on the 32 `to_out.0` projections, detached midpoint-state plus final-latent MSE, 24 updates at LR 4e-4, 8 trajectories, holdout flat; trained 2-step scored DINO 0.5216 / CLIP 0.8433, i.e. slightly worse than untrained | `autoresearch.md` §"Two-step student: result"; `results/qwen21_two_step_train_r1_output_6train_2holdout.json` |
| O4 | Accepted warm 512px medians: 6-step 2.085519 s, 5-step 1.890984 s, 2-step 1.445099 s, 2-step trained 1.406741 s | `autoresearch.md` |
| O5 | Per-stage cost on the accepted 5-step run: text encode 0.570453 s and 0.616525 s, sampler 0.982212 s and 1.021382 s, VAE decode 0.307065 s and 0.280132 s, SaveImage 0.022 s | `results/cu130_stream3_v021_final_r128_512_5step.json` |
| O6 | Per-step intervals inside the accepted 5-step sampler node: first transition 0.412067 s, then 0.199399, 0.136883, 0.133717 s | same file, `step_intervals` |
| O7 | Run-to-run pixel MAE of 0.015108 against the stored accepted image reproduces with and without two training-only ComfyUI fixes, so it is pre-existing nondeterminism of this stack, not a regression | `autoresearch.md`; `results/qwen21_regression_*.json` |
| O8 | Six-step teacher grid, raw sigma nodes `[1.0, 0.9375, 0.875, 0.75, 0.5, 0.25]`, shifted at 512px to `[1.0, 0.96256, 0.92306, 0.83717, 0.63151, 0.36357, 0.0]`. The 2-step grid `[1.0, 0.25]` shifts to `[1.0, 0.36357, 0.0]`, which is exactly the 3x coarsening of the six-step grid. **The 2-step grid is therefore grid-correct, not an arbitrary truncation.** | `scripts/export_qwen21_teacher.py`, `results/qwen21_teacher_trajectories.json` |

### 2.1 Derived quantity: the per-NFE cost model

From O5, O6 and O4, denoise cost is well fitted by `denoise(N) = a + (N-1)c`:

- `0.982 = a + 4c` (accepted five-step sampler node)
- `0.575 = a + 1c` (two-step end-to-end 1.445 minus text 0.570 minus VAE 0.305)
- therefore `c = 0.1357 s` per additional step and `a = 0.4393 s` for the first step.

Consequences, stated as **predictions to be re-measured, not results**:

- `a` is roughly 3.2x `c`, so a large one-off cost lands on the first denoiser evaluation.
  This is consistent with the already-recorded hypothesis that fault-time dense LoRA merge
  plus ConvRot re-rotation and re-quantization dominate LoRA cost (O5, O6).
- A perfect one-step student implies end-to-end ≈ `0.570 + 0.439 + 0.305 + 0.022 = 1.336 s`.
- A zero-step pipeline floor is ≈ `0.897 s`, of which the VAE decode alone is 0.305 s.
- Going from two steps to one step buys only `c = 0.136 s`, which is 10% of the 1.34 s
  total. The other 90% is text encode, first-step fault, and VAE decode.

This is the single most decision-relevant quantity in the file, and it is not yet
re-measured for this phase. P3 re-measures it.

## 3. Research question

For the accepted Viggle v0.2.1 r128 six-step Qwen-Image-2.1 student on one RTX 4060 Ti 16GB,
what is the minimum number of solver steps that preserves the locked six-step reference
quality, which correction objective and capacity close the gap, and does any point on the
step-count axis make a one-second end-to-end target reachable at all?

## 4. Candidate hypotheses

All labeled `candidate`. None is a finding.

- **H1-CLIFF.** The 6-to-2 gap is dominated by the step-halving ratio, not by adapter
  capacity. The recorded 2-step collapse is a 3x reduction attempted in a single jump, and
  the decline between 6, 5 and 4 steps is gentle. *This contradicts the previous run's
  stated conclusion.*
- **H2-CAPACITY.** Correction capacity is the binding constraint. Rank-1 on `to_out.0` only
  cannot express the correction, and the useful capacity lives in the modulation and
  timestep-embedder projections. This is the previous run's conclusion. **Its own evidence
  does not support it:** rank 1 is the only rank ever tried, so "not rank" was never tested.
- **H3-OBJECTIVE.** Detached-state regression is the wrong signal. Regressing on the
  teacher's midpoint state and final latent does not correct the compounding error that
  makes multi-step rollouts drift; backpropagating through the student's own rollout does.
- **H4-SCHEDULE.** A material part of the recorded 2-step gap is the choice of sigma nodes
  rather than the student. Note the codebase's `candidate_2step` uses `[1.0, 0.5]`, whereas
  the failed run used `[1.0, 0.25]`; **neither has been compared against the other, and
  neither has been optimized.**
- **H5-FIXED-COST.** One second is unreachable by step distillation on this card, because
  non-denoise fixed cost plus the first-step fault dominate every candidate. The necessary
  research target is the text encoder and the VAE decoder, not the denoiser step count.
- **H-ARTIFACT.** Rival that must stay alive: the 2-step collapse is a property of the
  measurement or sampler path rather than of the solver, e.g. cache interaction, the
  `QwenImage21Cache` node, or the way 2-step sigmas are threaded through the graph.
- **H-NOOP.** Rival that must stay alive: the trained adapter was not actually in the
  forward pass, so O3 is an untested run rather than a negative result. A reported "trained
  slightly worse than untrained" is equally consistent with an inert injection and with a
  real but unhelpful gradient. **This was never excluded.**
- **H-NFE-ONLY.** Rival to H5: the fixed-cost fit is confounded and a one-step pipeline
  really does land below one second.

## 5. Validity controls, and the order they run in

Validity is established before any hypothesis is tested. A failed control voids the
downstream reading rather than counting as a null result.

1. **NC-HARNESS (P0-VALID-HARNESS).** Regenerate the accepted six-step output and compare
   it with the stored locked reference. Pass if the difference is of the order of the
   recorded nondeterminism (O7, pixel MAE ~0.0151). A large mismatch implicates the harness
   and every candidate reading is suspended.
2. **NC-INJECTION (P0-VALID-INJECTION).** Generate twice from identical seed, prompt and
   conditioning, once with the trained LoRA injection active and once removed. Pass if the
   images differ by more than the nondeterminism floor. **If they are identical, O3 is void
   and must be recorded as untested, not as evidence about capacity or objective.** This
   control has never been run and is the cheapest high-value check in the program.
3. **NC-HOLDOUT.** Holdout seeds and prompts never enter an optimizer.
4. **NC-LICENSING.** Section 1.

## 6. Planned measurement sequence

- **P1-CLIFF-LOCATION.** Untrained r128 student, one sigma node at a time removed, fixed
  seed and locked prompt, 6/5/4/3/2/1 steps. Locates the cliff. Cheapest experiment that
  decides where training should start.
- **P2-GRID-OPTIMALITY.** Two-step Euler, grid search over the first and second shifted
  sigma, same fixed seed and prompt, untrained. Separates schedule error from capacity error.
- **P3-LATENCY-FLOOR.** Re-measure stage and per-step cost under the accepted protocol, fit
  `denoise(N) = a + (N-1)c`, and report the one-step and zero-step implications.
- **P4-CAPACITY.** Objective fixed to O3's, optimizer and update count fixed, rank 1 to 16
  and target modules extended from `to_out.0` to attention plus modulation plus timestep
  embedder. Decided on holdout latent error.
- **P5-OBJECTIVE.** Capacity fixed, objective changed to backpropagation through the
  student's own two-step rollout. Decided on holdout final-latent error.
- **P6-LADDER.** Incremental 6-to-3 then 3-to-2 halving instead of a single 6-to-2 jump.

## 7. Operationalization

| ID | Construct | Operational definition |
|---|---|---|
| M-QUALITY | Candidate quality | DINO image cosine and CLIP image cosine against the locked six-step reference, same seed, same prompt, 512px. DINO is the gate; CLIP is reported. |
| M-LATENCY | Warm end-to-end latency | Median prompt-submission-to-success seconds, 2 warmups and at least 5 measured requests, prompt varied every request so the classic graph cache cannot return identical nodes. |
| M-STAGE | Stage attribution | Per-node seconds for text encode, sampler and VAE decode, plus peak VRAM, from the same run. |
| M-HOLDOUT | Objective progress | Holdout latent MSE on seeds and prompts excluded from the optimizer, reported for the first Euler state and the final denoised latent. |
| M-ONLINE | Adapter validity | Pixel MAE and DINO cosine between adapter-on and adapter-off generation. |

- Unit of observation: one generation. Unit of analysis for quality: one (prompt, seed) pair.
- Known limitation of M-QUALITY: the existing fixed-seed set is effectively a single prompt
  at a single seed. A one-prompt sweep locates a cliff; it is not a distribution claim. Any
  decision that changes a production profile must be confirmed on more prompts and seeds
  first.
- Known limitation of M-HOLDOUT: a latent-space proxy. It is not assumed to track image
  quality. P4 reports both, and a holdout improvement with no quality movement is treated
  as evidence that the proxy is suspect.

## 8. Analysis plan and decision rules

- P1, P2: report the full curve or grid, not a selected maximum. A best grid found at the
  edge of the searched range is reported as a bound, not an optimum.
- P3: least-squares fit of the two anchors, with the anchors named. Report the residual of
  the fit against the two accepted end-to-end medians as a check.
- P4, P5, P6: holdout error decides. The rule adopted from the previous run's own lesson is
  kept: **measure holdout before running any end-to-end quality gate.** A flat holdout ends
  that configuration without spending a generate-and-score cycle.
- Only a configuration that passes holdout gets the end-to-end quality gate, and only a
  configuration that passes the quality gate gets a latency measurement. This ordering is
  what makes the program affordable on one card.
- Multiplicity: each of P4, P5 and P6 changes one factor relative to its declared parent.
  They are not pooled. No "best of N" is reported without the full set.
- Indeterminate results are reported as indeterminate, with what would be learned from them,
  as required per row of the matrix.

## 9. HARKing controls and deviations

- This file and the matrix were written before any measurement in section 6 was run.
  Timestamps above are the freeze time.
- Anything decided after seeing an outcome is labeled exploratory in `autoresearch.md`,
  with date, rationale, and who decided.
- Deviations are appended to section 10 rather than edited into the body.
- Negative results are recorded, not hidden. The previous run's negative result is preserved
  as O3 and is not overwritten.

## 10. Deviations log

- 2026-09-27: P7-LICENSING was removed from `prediction_rival_matrix.csv` and moved into
  section 1 as a compliance precondition. Rationale: the matrix schema requires a rival for
  every row, and a licensing condition has no scientific rival. It is still binding.
- 2026-09-27: the first cliff sweep was run at the `comfy_bench` default seed 42 while the
  locked quality gate uses seed 101. That produced a spurious six-step control DINO of
  `0.8665` against the locked reference and briefly looked like a harness failure. The
  harness was correct; the seed was wrong. Every later measurement uses seed 101 for the
  locked cell. Recorded because it is exactly the kind of error that would have been
  reported as a negative result about the model.
- 2026-09-27: the first cliff sweep used a per-grid warmup, which executed each grid and
  then let the measured run return the same graph from the classic cache. The **images**
  were valid, because each grid's graph had genuinely executed during its own warmup, but
  every **timing** in that run was invalid at about `0.03 s`. `sweep_sigmas.py` was changed
  to warm once on a throwaway prompt in quality mode and to enforce prompt variation in
  latency mode, and to report `cached_runs` and `latency_valid` per grid.
- 2026-09-27: the first latency sweep omitted `--vary-prompt`, so every measured sample was
  served from the cache and reported about `0.03 s` across all step counts. This was caught
  because the numbers were implausibly identical, not because the harness complained. The
  driver now enforces it rather than offering it as a flag.
- 2026-09-27 (exploratory, post-hoc): the two-step sigma search range was extended from
  `0.10` to `0.70` to `0.72` to `0.90` after the best value fell at the edge of the original
  range. This is data-dependent and is labeled exploratory. The twelve-cell sweep in run 51
  is a separate, independently motivated measurement and was not chosen in response to it.
- 2026-09-27 (exploratory, post-hoc): the two-step sigma search was extended again over
  `0.25, 0.50, 0.65, 0.75, 0.78, 0.85` across twelve cells because the single-cell optimum
  was not bracketed. The resulting best value is again at the edge of the tested set, so the
  optimum remains unbracketed and is reported as a bound, per section 8.
- 2026-09-27: grids whose first sigma is below `1.0` are rejected by the accepted harness
  (`comfy_bench.qwen_sigmas` requires the first raw sigma to be exactly `1.0`). Whether a
  two-step schedule starting at lower noise helps is therefore untested and is a boundary
  condition of every sigma finding in this file, not a tested null.
