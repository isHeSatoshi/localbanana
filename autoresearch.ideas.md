# Ideas Backlog

- **Phase 2 changed the operating point.** Across 4 prompts x 3 seeds, three steps
  (`1,0.75,0.25`) matches four steps (mean DINO 0.8345 vs 0.8448, median 0.8411 vs 0.8413)
  for about 0.143 s less. Four steps is nearly wasted over three. Re-measure three-step
  latency on the accepted long-prompt protocol and widen its quality gate before it can
  replace the five-step default.
- **The one-second target is not reachable by step reduction on this card.** Fitted
  `denoise(N) = 0.3654 + 0.1434*(N-1)`, non-denoise fixed 0.7799 s, so a perfect one-step
  student is ~1.12 s (short prompt) or ~1.34 s (accepted long prompt) and a zero-denoise
  floor is 0.78 s. Stop treating NFE as the lever.
- **The first-call surcharge is the best latency target available.** 0.2220 s on the first
  denoiser call, which is 1.55 marginal calls and larger than dropping two steps to one or
  four to three. It is LoRA plus ConvRot fault machinery, not a quality term, so a
  distilled student with its adapter baked into its own base weights should not pay it.
- **Text encoder plus VAE decode is 0.780 s** and is the majority of any one-step pipeline.
  No denoiser work touches it. Treat both as first-class research targets.
- **The prescribed two-step grid `[1.0, 0.25]` is a bad schedule.** On the tuning cell,
  searching the second sigma moved DINO from 0.5503 at 0.25 to 0.8472 at 0.78 with no
  training. Across 12 cells the ranking is monotone in that sigma but the best value is
  still at the edge of the tested range, so the optimum is not bracketed. Per project rule
  this is an ablation, never a student.
- **Two steps is unreliable, not just weaker.** The tuned two-step grid has median DINO
  0.7813 but mean 0.6456 and minimum 0.1445 across 12 cells. Report the per-cell
  distribution for any few-step candidate; a mean hides the failure tail.
- **One step is not currently a credible target.** Median DINO 0.0879, minimum -0.0143
  across 12 cells. Defer rather than retrying at higher rank.
- **Every quality number in this project rested on one prompt and one seed, which
  overstates quality by about 0.10.** The twelve-cell sweep put that cell at 0.9339 for
  three steps where the mean is 0.8345, and 0.5503 for the prescribed two steps where the
  mean is 0.4446. Fix the gate before any further training; a run judged on the old
  single-cell gate will not survive contact with a wider set.
- **The previous "capacity and data, not rank" conclusion was not supported by its own
  evidence.** Only rank 1 was ever tried, so rank was never excluded, and the evaluation
  was one cell. Treat that result as untested rather than as a negative finding about
  capacity.
- **`NC-INJECTION` is still open.** The saved adapters are non-zero (up-matrix L2 0.134 and
  0.913) so the adapter is not trivially empty, but whether the injection reached the
  forward pass during end-to-end evaluation was never verified. If adapter-on and
  adapter-off images are identical, the earlier training result is void, not negative.
- **The Gradio dashboard now surfaces the phase-2 evidence.** `dashboard.py` gained a
  three-step "Balanced" profile (raw sigma `[1.0, 0.75, 0.25]`, correctly shifting to
  `[1.0, 0.837, 0.364, 0.0]` at 512px) and a "Few-step research" tab that renders the
  12-cell step-count table, the latency floor fit, and the long-prompt step comparison
  directly from `results/schedule_stepcount_generalization.json`,
  `results/latency_floor.json`, and `results/sweep_p4_longprompt.json`. It reads those
  files at render time, so it shows nothing that is not on disk and reports a missing file
  rather than a stale number. The Balanced profile is labelled experimental: it is not an
  adopted production default. The old single-cell quality table now carries an explicit
  caveat that it overstates quality by roughly 0.10.
- **The two-step training infrastructure works and is reusable**: `scripts/export_qwen21_teacher.py`,
  `scripts/train_qwen21_two_step.py`, `scripts/eval_qwen21_student.py`. New this phase:
  `scripts/sweep_sigmas.py` (sigma-grid driver over the accepted `comfy_bench.py`),
  `scripts/schedule_generalization.py` (multi-prompt, multi-seed scoring against per-cell
  six-step references), `scripts/fit_latency_floor.py`, `scripts/build_frontier.py`.
- **Always force prompt variation in any latency measurement, and verify it.** The classic
  graph cache will silently return the whole graph and report ~0.03 s. `sweep_sigmas.py` now
  enforces prompt variation in latency mode and reports `cached_runs` and `latency_valid`
  per grid, and excludes any grid that is not latency-valid. A per-grid warmup in quality
  mode causes the same trap; warm once on a throwaway prompt instead.
- **The locked quality gate uses seed 101, not the bench default 42.** Comparing across
  seeds produced a spurious DINO 0.8665 for a correct six-step control.
- Two ComfyUI training-path defects were found and fixed in ComfyUI source. `_gated_residual`
  mutated a tensor autograd still needs, across a 32-block in-place chain. `cast_bias_weight`
  handed out views into a reusable async cast buffer that a later block overwrote before
  backward consumed the saved norm weight. Both fixes are guarded by
  `comfy.model_management.in_training` and do not change inference.
- Checkpointed backward is only valid with synchronous offload. With async offload streams,
  32-block checkpointing produced 253,952 non-finite gradient elements. Anomaly detection
  changed timing enough to hide that race, so never use anomaly mode to validate this path.
  Disable async offload for training.
- Always measure holdout loss before running an end-to-end quality gate. A flat holdout means
  the objective or capacity is wrong, and it saves a full generate-and-score cycle.
- The Viggle v0.2.1 r128 five-step fast profile and six-step quality default remain the
  accepted production paths. Nothing in phase 2 changed a production profile.
- Do not retry CPU text encoding, W4A8 text, lossy prefix-cache precision, QKV fusion, direct
  per-block Qwen graph capture, cuDNN autotune, Comfy Kitchen attention, zero/four streams,
  0.5 GiB reserve, `--disable-fast-disk`, generic pre-baked Turbo checkpoints,
  `LoraLoaderBypassModelOnly`, or official-style unmerged LoRA without a new measured
  bottleneck and correctness gate.
