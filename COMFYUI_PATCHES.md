# Two ComfyUI defects that block training Qwen-Image-2.1

Found while trying to distill a few-step student out of the Viggle Turbo v0.2.1 adapter.
Both are guarded by `comfy.model_management.in_training`, so neither changes inference
behaviour. Verified: a run on clean code and a run with both fixes produce pixel-identical
output, MAE `0.0`.

## 1. `_gated_residual` mutates a tensor autograd still needs

**File:** `comfy/ldm/qwen_image21/model.py`

The function added its residual contribution to the running hidden state in place, inside a
loop over all 32 transformer blocks. That builds a 32-deep in-place chain on a tensor that
backward still has to read.

Two failure modes showed up:

- Without checkpointing, autograd rejects the operation outright.
- With block checkpointing, it does not reject. It silently produces gradients at roughly
  `1e17`. That is the dangerous case, because training appears to run and the weights are
  simply destroyed.

The fix is to make the training branch out-of-place, leaving inference on the original
in-place path.

## 2. `cast_bias_weight` hands out views into a reused async buffer

**File:** `comfy/ops.py`

The function cast a bias weight on the offload stream and returned **views into a shared,
reusable cast buffer**. A later block's cast overwrote the same storage before backward
consumed a saved RMSNorm weight.

The effect is a use-after-overwrite in the autograd graph, and it shows up as wrong or
non-finite gradients rather than as an error. The fix allocates a dedicated cast destination
for the training path while still using the offload stream.

## 3. Not a bug, but a landmine: async offload plus checkpointing

Checkpointed backward is only valid with **synchronous** offload. With the async offload
streams enabled, 32-block checkpointing produced `253,952` non-finite gradient elements.

Disable async offload for training.

One trap worth naming: `torch.autograd.set_detect_anomaly(True)` changes timing enough to
hide this race. Anomaly mode reported clean gradients on a configuration that was producing
non-finite ones. Never use anomaly mode to validate this path.

## Reproducing

Both defects only appear in the training path, so you need a backward pass through
`comfy/ldm/qwen_image21/model.py` to hit them. Any Qwen-Image-2.1 training script will do.

What a correct run looks like: gradient L2 in a sane range and zero non-finite gradient
elements. The broken configurations report non-finite counts, or gradients near `1e17` when
checkpointing is enabled.

To see defect 2 specifically, run training with the async offload streams enabled. The
disposable cast buffer is only reused when offload is asynchronous, so a synchronous run can
look correct while the aliasing is still there.

## Inference neutrality

| configuration | pixel MAE vs stored accepted image |
|---|---:|
| clean code, no fixes | 0.015108 |
| with both fixes | 0.015108 |
| clean vs fixed, same run | 0.0 |

The `0.015108` offset reproduces with and without the patches, so it is pre-existing
run-to-run nondeterminism in this stack, not a regression from these changes. A consequence
worth knowing when benchmarking: any latency or quality comparison needs a tolerance of at
least that size before you call a difference real.
