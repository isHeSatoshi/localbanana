"""Export fixed Qwen-Image-2.1 six-step teacher trajectories for distillation."""

from __future__ import annotations

import argparse
import json
import math
import shutil
import time
import uuid
from pathlib import Path

import requests


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "out" / "research" / "teacher_trajectories"
DEFAULT_RESULTS = ROOT / "results" / "qwen21_teacher_trajectories.json"
DEFAULT_PROMPTS = [
    "A small blue glass sphere on a dark studio background, soft rim light, product photograph, centered composition",
    "A red vintage bicycle leaning against a sunlit limewashed wall, realistic editorial photography",
    "A misty mountain lake at sunrise, layered ridges, calm water, wide landscape photograph, natural colors",
    "A chef plating handmade noodles in a warm modern kitchen, candid documentary photograph",
    "A futuristic transparent mechanical wristwatch on black acrylic, detailed product macro, cyan reflections",
    "A rain-covered neon street in a dense Asian city at night, cinematic photography, reflections",
    "A studio portrait of an elderly fisherman mending a net, warm rim light, 85mm, natural skin texture",
    "A minimalist poster reading CREATE MORE, bold geometric typography, cream and black, gallery quality",
]


def shifted_sigmas(size: int, raw: list[float]) -> list[float]:
    sequence_length = (size // 16) ** 2
    slope = (0.9 - 0.5) / (8192 - 256)
    mu = sequence_length * slope + (0.5 - slope * 256)
    alpha = math.exp(mu)
    return [alpha * sigma / (1 + (alpha - 1) * sigma) for sigma in raw] + [0.0]


def build_graph(args, prompt: str, sigmas: list[float]) -> dict[str, dict]:
    state_sigmas = sigmas[:-1]
    state_text = ", ".join(f"{sigma:.9f}" for sigma in state_sigmas)
    final_text = ", ".join(f"{sigma:.9f}" for sigma in sigmas)
    return {
        "1": {
            "class_type": "UNETLoader",
            "inputs": {"unet_name": args.model, "weight_dtype": "default"},
        },
        "2": {
            "class_type": "LoraLoaderModelOnly",
            "inputs": {
                "model": ["1", 0],
                "lora_name": args.lora,
                "strength_model": 1.0,
            },
        },
        "3": {
            "class_type": "CLIPLoader",
            "inputs": {
                "clip_name": args.text_encoder,
                "type": "qwen_image",
                "device": "default",
            },
        },
        "4": {
            "class_type": "TextEncodeQwenImage21",
            "inputs": {
                "clip": ["3", 0],
                "prompt": prompt,
                "negative_prompt": "",
                "encode_negative": False,
                "resolution": args.size,
                "images": {},
            },
        },
        "5": {
            "class_type": "EmptyLatentImage",
            "inputs": {"width": args.size, "height": args.size, "batch_size": 1},
        },
        "6": {"class_type": "RandomNoise", "inputs": {"noise_seed": args.seed}},
        "7": {
            "class_type": "KSamplerSelect",
            "inputs": {"sampler_name": "euler"},
        },
        "8": {"class_type": "ManualSigmas", "inputs": {"sigmas": state_text}},
        "9": {
            "class_type": "BasicGuider",
            "inputs": {"model": ["15", 0], "conditioning": ["4", 0]},
        },
        "10": {
            "class_type": "SamplerCustomAdvanced",
            "inputs": {
                "noise": ["6", 0],
                "guider": ["9", 0],
                "sampler": ["7", 0],
                "sigmas": ["8", 0],
                "latent_image": ["5", 0],
            },
        },
        "11": {
            "class_type": "SaveLatent",
            "inputs": {
                "samples": ["10", 0],
                "filename_prefix": f"{args.filename_prefix}_state",
            },
        },
        "12": {
            "class_type": "SaveConditioning",
            "inputs": {
                "conditioning": ["4", 0],
                "filename_prefix": args.filename_prefix,
            },
        },
        "13": {
            "class_type": "SamplerCustomAdvanced",
            "inputs": {
                "noise": ["6", 0],
                "guider": ["9", 0],
                "sampler": ["7", 0],
                "sigmas": ["16", 0],
                "latent_image": ["5", 0],
            },
        },
        "14": {
            "class_type": "SaveLatent",
            "inputs": {
                "samples": ["13", 0],
                "filename_prefix": f"{args.filename_prefix}_final",
            },
        },
        "15": {
            "class_type": "QwenImage21Cache",
            "inputs": {"model": ["2", 0], "device": "gpu", "dtype": "default"},
        },
        "16": {"class_type": "ManualSigmas", "inputs": {"sigmas": final_text}},
    }


def newest_file(directory: Path, pattern: str) -> Path:
    matches = [path for path in directory.glob(pattern) if path.is_file()]
    if not matches:
        raise FileNotFoundError(f"No teacher artifact matched {directory / pattern}")
    return max(matches, key=lambda path: path.stat().st_mtime_ns)


def run_sample(session: requests.Session, args, index: int, prompt: str, seed: int, tag: str) -> dict:
    sample_id = f"{tag}_{index:03d}_seed{seed}"
    filename_prefix = f"research/teacher_trajectories/{tag}/{sample_id}"
    export_args = argparse.Namespace(**vars(args))
    export_args.seed = seed
    export_args.filename_prefix = filename_prefix

    teacher_sigmas = shifted_sigmas(args.size, [1.0, 0.9375, 0.875, 0.75, 0.5, 0.25])
    capture_sigmas = teacher_sigmas
    graph = build_graph(export_args, prompt, capture_sigmas)
    prompt_id = str(uuid.uuid4())
    started = time.perf_counter()
    response = session.post(
        f"{args.base_url}/prompt",
        json={"prompt": graph, "client_id": args.client_id, "prompt_id": prompt_id},
        timeout=30,
    )
    response.raise_for_status()
    submission = response.json()
    if submission.get("prompt_id") != prompt_id:
        raise RuntimeError(f"ComfyUI returned a different prompt ID: {submission}")

    deadline = time.monotonic() + args.timeout
    entry = None
    while time.monotonic() < deadline:
        history = session.get(f"{args.base_url}/history/{prompt_id}", timeout=30)
        history.raise_for_status()
        entry = history.json().get(prompt_id)
        if entry is not None:
            status = entry.get("status", {})
            if status.get("completed") or status.get("status_str") in {"success", "error"}:
                break
        time.sleep(0.1)
    if entry is None:
        raise TimeoutError(f"Timed out waiting for teacher export {prompt_id}")

    status = entry.get("status", {})
    if status.get("status_str") == "error":
        raise RuntimeError(json.dumps(status, indent=2))
    if not status.get("completed"):
        raise RuntimeError(f"Teacher export did not complete: {status}")

    destination = args.output_root / tag
    destination.mkdir(parents=True, exist_ok=True)
    latent_source = newest_file(
        args.comfy_output / "research" / "teacher_trajectories" / tag,
        f"{sample_id}_state_*.latent",
    )
    final_source = newest_file(
        args.comfy_output / "research" / "teacher_trajectories" / tag,
        f"{sample_id}_final_*.latent",
    )
    conditioning_source = newest_file(
        args.comfy_output / "research" / "teacher_trajectories" / tag,
        f"{sample_id}_*.safetensors",
    )
    latent_path = destination / f"{sample_id}.state.latent"
    final_path = destination / f"{sample_id}.final.latent"
    conditioning_path = destination / f"{sample_id}.conditioning.safetensors"
    shutil.copy2(latent_source, latent_path)
    shutil.copy2(final_source, final_path)
    shutil.copy2(conditioning_source, conditioning_path)

    return {
        "index": index,
        "sample_id": sample_id,
        "prompt": prompt,
        "seed": seed,
        "width": args.size,
        "height": args.size,
        "prompt_id": prompt_id,
        "elapsed_seconds": time.perf_counter() - started,
        "teacher_raw_sigmas": [1.0, 0.9375, 0.875, 0.75, 0.5, 0.25],
        "teacher_sigmas": teacher_sigmas,
        "capture_sigmas": capture_sigmas,
        "two_step_raw_sigmas": [1.0, 0.25],
        "two_step_sigmas": shifted_sigmas(args.size, [1.0, 0.25]),
        "state_latent": str(latent_path),
        "final_latent": str(final_path),
        "conditioning": str(conditioning_path),
        "source_state_latent": str(latent_source),
        "source_final_latent": str(final_source),
        "source_conditioning": str(conditioning_source),
    }


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8188")
    parser.add_argument("--comfy-output", type=Path, default=ROOT / "out" / "comfy")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--results", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--tag", default="fixed_p1")
    parser.add_argument("--size", type=int, default=512)
    parser.add_argument("--model", default="qwen_image_2.1_int8_convrot.safetensors")
    parser.add_argument(
        "--lora",
        default="Qwen-Image-2.1-viggle-turbo-v0.2.1-6step-lora-r128.safetensors",
    )
    parser.add_argument("--text-encoder", default="qwen3vl_8b_int8_convrot.safetensors")
    parser.add_argument("--limit", type=int, default=8)
    parser.add_argument("--timeout", type=float, default=600.0)
    parser.add_argument("--seed-start", type=int, default=42)
    parser.add_argument("--prompt", action="append")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.size % 16:
        raise ValueError("Size must be divisible by 16")
    prompts = args.prompt or DEFAULT_PROMPTS
    count = min(args.limit, len(prompts))
    args.client_id = str(uuid.uuid4())
    records = []
    started = time.perf_counter()
    with requests.Session() as session:
        session.get(f"{args.base_url}/system_stats", timeout=30).raise_for_status()
        for index, prompt in enumerate(prompts[:count]):
            record = run_sample(session, args, index, prompt, args.seed_start + index, args.tag)
            records.append(record)
            print(json.dumps(record, indent=2), flush=True)

    payload = {
        "tag": args.tag,
        "base_url": args.base_url,
        "size": args.size,
        "teacher": "Qwen-Image-2.1 + Viggle v0.2.1 r128 six-step",
        "objective_contract": {
            "state_before_final": "teacher Euler state at the shifted raw sigma 0.25 node",
            "final": "teacher denoised latent at sigma 0",
            "student_schedule": "shifted raw sigmas [1.0, 0.25, 0.0]",
            "loss": "MSE on the first student Euler state plus MSE on the final denoised latent",
        },
        "total_elapsed_seconds": time.perf_counter() - started,
        "records": records,
    }
    args.results.parent.mkdir(parents=True, exist_ok=True)
    args.results.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"wrote {args.results}")


if __name__ == "__main__":
    main()
