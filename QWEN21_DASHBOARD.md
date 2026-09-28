# Qwen-Image-2.1 Local Dashboard and Optimization Notes

## Run the dashboard

The dashboard is a local Gradio client. ComfyUI must already be running with the accepted model/runtime flags.

```powershell
Set-Location D:\Project\1secimage
& .\.venv-cu130\Scripts\python.exe .\dashboard.py --server-name 127.0.0.1 --server-port 7860
```

Or:

```powershell
Set-Location D:\Project\1secimage
.\start_dashboard.ps1
```

Open <http://127.0.0.1:7860>. The ComfyUI API remains at <http://127.0.0.1:8188>.

Install dashboard dependencies in the isolated environment if needed:

```powershell
python -m pip install --python .\.venv-cu130\Scripts\python.exe -r .\requirements-dashboard.txt
```

The dashboard does not load PyTorch or the model. It sends API graphs to ComfyUI, so it does not duplicate model VRAM.

## Current v0.2.1 stack

| Artifact | Local file size | Role |
|---|---:|---|
| `qwen_image_2.1_int8_convrot.safetensors` | 6.758 GiB | INT8 ConvRot visual DiT |
| `qwen3vl_8b_int8_convrot.safetensors` | 8.709 GiB | INT8 ConvRot Qwen3-VL prompt and reference encoder |
| `qwen_image_2.1_vae_bf16.safetensors` | 0.629 GiB | BF16 VAE |
| `Qwen-Image-2.1-viggle-turbo-v0.2.1-6step-lora-r128.safetensors` | 0.633 GiB | Official rank-128 v0.2.1 student adapter |
| Total listed artifacts | about 16.7 GiB | Disk storage, not a VRAM guarantee |

The local stack uses PyTorch `2.14.0+cu130`, native Comfy Kitchen CUDA kernels, DynamicVRAM, three asynchronous offload streams, a 1.0 GiB VRAM reserve, and the lossless GPU Qwen prefix cache. Turbo defaults to positive-only CFG-1 with `BasicGuider` for the lowest latency. The dashboard's optional negative-prompt control switches to a second text-conditioning pass and `CFGGuider`; this is slower, and negative prompting with a Turbo profile is experimental because v0.2.1 is designed for CFG-1. ComfyUI dynamically stages and offloads model chunks, so disk size is not simultaneous VRAM residency.

## Dashboard profiles

- **Quality (6-step Turbo v0.2.1 r128):** the official prescribed raw sigmas `[1.0, 0.9375, 0.875, 0.75, 0.5, 0.25]`, positive-only CFG-1, GPU lossless prefix cache. This is the default selection.
- **Fast (5-step Turbo v0.2.1 r128):** raw sigmas `[1.0, 0.875, 0.75, 0.5, 0.25]`. This is the selected fast compromise. The publisher documents five-step use as valid, with a detail/diversity tradeoff.
- **Balanced (3-step Turbo v0.2.1 r128, experimental):** raw sigmas `[1.0, 0.75, 0.25]`. Measured over 4 prompts and 3 seeds, this matches the four-step schedule in quality (mean DINO 0.8345 vs 0.8448, median 0.8411 vs 0.8413) for one fewer denoiser call, and saves about 0.319 s against the five-step profile. It is experimental until its quality gate is widened. See `README.md` and the "Few-step research" tab.
- **Experimental (4-step Turbo v0.2.1 r128):** raw sigmas `[1.0, 0.75, 0.5, 0.25]`. Three steps now makes this schedule nearly redundant, since the two measure the same on the local quality gate.
- **Base reference (40-step no Turbo):** the same INT8 weights, GPU cache, and CFG-1 conditioning with Turbo removed. This is a local reference, not a BF16 full-fidelity benchmark.

Two steps is deliberately not offered as a profile. Over 4 prompts and 3 seeds it is unreliable rather than merely weaker (tuned grid: median DINO 0.7813, mean 0.6456, minimum 0.1445), and on the accepted long-prompt protocol going from three steps to two steps saves 0.009 s, which is inside run-to-run spread.

With **Use negative prompt** disabled, the dashboard sends `encode_negative=false` and uses `BasicGuider`, preserving the measured CFG-1 fast path. When enabled, it encodes the negative text, uses `CFGGuider` at the selected CFG scale, and keeps separate prompt-keyed prefix-cache slots. Turbo negative prompting is experimental and slower; use the Base reference for the supported no-Turbo negative-prompt path. The same-seed validation outputs are recorded in `results/negative_prompt_gradio_validation.json`.

## Negative-prompt control

The Generate tab includes a **Negative prompt** accordion with:

- **Use negative prompt:** off by default, preserving the fast CFG-1 path.
- **Negative prompt:** text passed to `TextEncodeQwenImage21` when enabled.
- **CFG scale:** used by `CFGGuider` only when enabled, from 1.0 to 10.0.

Turbo profiles remain faster without negative prompting. Enabling negative prompting adds a second conditioning pass and is off the official v0.2.1 CFG-1 contract. The Base reference is the supported no-Turbo path for negative prompting.

## Output size and aspect ratio

The Generate tab includes independent **Area** and **Aspect ratio** selectors. It retains the existing 512² and 768² square areas and adds the requested 1024² and 2048² areas. Presets cover Instagram, YouTube, Reels/Shorts, and common photo ratios. The exact requested 16:9, 9:16, 4:3, 3:4, 3:2, and 2:3 dimensions are preserved at 1024² and 2048²; additional 4:5, 5:4, 15:9, and 21:9 presets are rounded to the model's required 16-pixel grid.

Choose **Custom** to enter width and height directly. Both values must be positive multiples of 16 and must not exceed ComfyUI's 16,384-pixel limit. The dashboard shows the resolved dimensions before submission. Rectangular and 2048² output use substantially more image tokens and can be slower and use more VRAM than the measured square fast path. When reference images are supplied, the existing Qwen reference path derives its edit latent from the reference image dimensions; the selected output shape applies to text-to-image generation.

## Warm measurements

Measurements are prompt-submission-to-ComfyUI-success on one RTX 4060 Ti 16GB, with two warmup requests and five post-warmup requests using unique prompt markers.

| Profile | Resolution | Total median | Text median | Denoise median | VAE median | Peak VRAM |
|---|---:|---:|---:|---:|---:|---:|
| v0.2.1 r128, 6-step quality | 512 | **2.086 s** | 0.602 s | 1.156 s | 0.304 s | 15,582 MiB |
| v0.2.1 r128, 5-step fast | 512 | **1.891 s** | 0.570 s | 0.982 s | 0.295 s | 15,662 MiB |
| v0.2.1 r128, 4-step experimental | 512 | **2.295 s** | 0.817 s | 1.126 s | 0.315 s | 15,421 MiB |
| v0.2.1 r128, 5-step fast | 768 | **3.344 s** | 0.626 s | 2.078 s | 0.561 s | 15,588 MiB |

The v0.2.1 path does not reach one second on this hardware. The historical `1.656 s` 512px result belongs to the earlier rank-64 adapter and is not a v0.2.1 measurement.

## Quality gate

The timing column is from the final clean stock-loader validation runs. The image-quality proxies use the first real execution of each fixed-seed candidate run; graph-cached repeats were excluded. Full data is in `results/quality_v021/quality_v021_p1_metrics.json`.

| Candidate | CLIP image cosine | DINO image cosine | DINO L2 | CLIP prompt alignment |
|---|---:|---:|---:|---:|
| r256 six-step reference | 1.000000 | 1.000000 | 0.000 | 0.336403 |
| r128 six-step | 0.999572 | 0.999034 | 0.044 | 0.335129 |
| r128 five-step | 0.986657 | 0.965302 | 0.263 | 0.337517 |
| r128 four-step | 0.983807 | 0.949876 | 0.317 | 0.344306 |

The r128 six-step result is effectively unchanged from the r256 reference in this sample. Five-step is the selected fast compromise because it is faster and remains closer to the six-step reference than four-step. These proxies do not establish equivalence to the 40-step base model or cover all prompts, seeds, resolutions, and edit cases. Full data is in `results/quality_v021/quality_v021_p1_metrics.json`.

## Rejected paths

- RAM-pressure cache under severe host commit pressure: confounded and not an admissible baseline; use classic cache.
- `LoraLoaderBypassModelOnly`: about 3.635 s at 512px, slower than the stock runtime loader.
- Official-style unmerged r128: about 4.029 s; unmerged r256: about 4.270 s.
- Pre-baked Turbo checkpoint: no speedup and material output drift; generic serialization is not the exact BF16/ConvRot/stochastic-rounding path.
- Specialized exact-bake probe: representative float patch math matched, but quantized requantization differed in a few stochastic-rounding bytes and the combined diagnostic hit a CUDA illegal-memory-access. No bake artifact was written.
- Direct per-block CUDA graph capture: graph breaks, rogue allocations, and wrong outputs.
- CPU text encoder, lossy cache precision, QKV fusion, alternate stream counts, cuDNN autotune, and Comfy Kitchen attention: no accepted latency/quality win.
- A scale-one LoRA allocation reduction was pixel-identical in a clean A/B but did not establish a latency win and was reverted.

The experimental `ComfyUI/custom_nodes/viggle_turbo.py` unmerged node is retained as a diagnostic only. The dashboard does not use it.

## What is upstream and what is local

Upstream components include the Qwen architecture, Qwen3-VL encoder, VAE, Viggle Turbo adapter/distillation, prefix-cache design, INT8 artifacts, Comfy Kitchen kernels, and ComfyUI offload machinery.

Local work includes the CFG-1 `encode_negative` graph option, the optional Gradio negative-prompt and CFG-scale controls, the exact artifact/runtime combination, RTX 4060 Ti stream tuning, benchmark harness, quality evaluation, profile migration, and the durable research journal.

## Future student research

The current dashboard remains pinned to the accepted v0.2.1 r128 5-step and 6-step profiles. A separate future research phase will evaluate genuine 2-step and 1-step students, initialized from the accepted student and trained with an explicit distillation objective. Sigma-list truncation is only a sampler ablation and is not counted as a new student.

The candidate harness must compare fixed-seed outputs against the locked six-step reference and five-step fast baseline using CLIP/DINO quality, prompt alignment, human review, warm 512px latency, stage timings, and peak VRAM. A one-step or two-step candidate is not adopted automatically, even if it is faster. The current runtime, cache, and Turbo paths remain available for rollback.

The reported Viggle training scale of roughly 19.4k requests, 1,000 student steps, and 16x B200 GPUs is recorded as provenance only. The exact teacher, data construction, loss, schedule, precision, and licensing must be verified before any local training is attempted.
