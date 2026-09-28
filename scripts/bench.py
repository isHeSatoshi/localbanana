import argparse
import json
import os
import time
from pathlib import Path

import torch

MODEL_DIR = os.environ.get("QWEN21_DIR", r"C:\qwen_weights\Qwen-Image-2.1")
OUT_DIR = Path(__file__).resolve().parents[1] / "results"
IMG_DIR = Path(__file__).resolve().parents[1] / "out"

PROMPT = (
    "A neon shop sign that reads \"QWEN IMAGE 2.1\", rainy night, "
    "reflections on wet pavement, cinematic photo"
)


def peak_mb():
    if not torch.cuda.is_available():
        return 0.0
    return torch.cuda.max_memory_allocated() / (1024**2)


def reset_peak():
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()


def load_pipeline(dtype: str = "bfloat16", cpu_offload: bool = False):
    from diffusers import QwenImage21Pipeline

    torch_dtype = getattr(torch, dtype)
    t0 = time.perf_counter()
    pipe = QwenImage21Pipeline.from_pretrained(
        MODEL_DIR,
        torch_dtype=torch_dtype,
        low_cpu_mem_usage=True,
    )
    load_s = time.perf_counter() - t0
    if cpu_offload:
        pipe.enable_model_cpu_offload()
    else:
        pipe.to("cuda")
    return pipe, load_s


def bench_once(pipe, width: int, height: int, steps: int, seed: int = 42):
    reset_peak()
    gen = torch.Generator(device="cuda").manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    image = pipe(
        prompt=PROMPT,
        width=width,
        height=height,
        num_inference_steps=steps,
        true_cfg_scale=1.0,
        generator=gen,
    ).images[0]
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    dt = time.perf_counter() - t0
    return dt, peak_mb(), image


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, nargs="+", default=[40])
    ap.add_argument("--sizes", type=int, nargs="+", default=[512, 768, 1024])
    ap.add_argument("--warmup", type=int, default=1)
    ap.add_argument("--repeats", type=int, default=1)
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--cpu-offload", action="store_true")
    ap.add_argument("--tag", default="baseline")
    args = ap.parse_args()

    assert torch.cuda.is_available(), "CUDA required"
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    IMG_DIR.mkdir(parents=True, exist_ok=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    pipe, load_s = load_pipeline(args.dtype, args.cpu_offload)
    rows = []
    print(f"model_load_s={load_s:.2f}", flush=True)

    for size in args.sizes:
        for steps in args.steps:
            for i in range(args.warmup):
                dt, mem, _ = bench_once(pipe, size, size, steps)
                print(f"warmup {size} steps={steps} {dt:.3f}s {mem:.0f}MB", flush=True)
            times = []
            mems = []
            for i in range(args.repeats):
                dt, mem, img = bench_once(pipe, size, size, steps)
                times.append(dt)
                mems.append(mem)
                print(f"run {size} steps={steps} {dt:.3f}s {mem:.0f}MB", flush=True)
                if i == args.repeats - 1:
                    out = IMG_DIR / f"{args.tag}_{size}_{steps}step.png"
                    img.save(out)
            row = {
                "tag": args.tag,
                "width": size,
                "height": size,
                "steps": steps,
                "model_load_s": load_s,
                "warmup_s": None,
                "time_s": times,
                "time_mean_s": sum(times) / len(times),
                "peak_vram_mb": mems,
                "dtype": args.dtype,
                "cpu_offload": args.cpu_offload,
                "gpu": torch.cuda.get_device_name(0),
                "torch": torch.__version__,
                "prompt": PROMPT,
            }
            rows.append(row)
            print(json.dumps(row), flush=True)

    path = OUT_DIR / f"bench_{args.tag}.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2)
    print(f"wrote {path}", flush=True)


if __name__ == "__main__":
    main()
