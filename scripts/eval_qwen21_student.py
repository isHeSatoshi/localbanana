"""Benchmark and quality-gate a trained Qwen-Image-2.1 student correction adapter."""

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
DEFAULT_BASE_URL = "http://127.0.0.1:8188"
DEFAULT_OUTPUT = ROOT / "out" / "research" / "student_eval"
DEFAULT_TEACHER_LORA = "Qwen-Image-2.1-viggle-turbo-v0.2.1-6step-lora-r128.safetensors"
TEACHER_RAW = [1.0, 0.9375, 0.875, 0.75, 0.5, 0.25]
STUDENT_RAW = [1.0, 0.25]


def shifted_sigmas(size: int, raw: list[float]) -> list[float]:
    sequence_length = (size // 16) ** 2
    slope = (0.9 - 0.5) / (8192 - 256)
    mu = sequence_length * slope + (0.5 - slope * 256)
    alpha = math.exp(mu)
    return [alpha * sigma / (1 + (alpha - 1) * sigma) for sigma in raw] + [0.0]


def build_graph(args, prompt: str, sigmas: list[float], filename_prefix: str) -> dict[str, dict]:
    sigmas_text = ", ".join(f"{sigma:.9f}" for sigma in sigmas)
    graph: dict[str, dict] = {
        "1": {
            "class_type": "UNETLoader",
            "inputs": {"unet_name": args.model, "weight_dtype": "default"},
        },
        "2": {
            "class_type": "LoraLoaderModelOnly",
            "inputs": {"model": ["1", 0], "lora_name": args.teacher_lora, "strength_model": 1.0},
        },
    }
    if args.student_lora:
        graph["3"] = {
            "class_type": "LoraLoaderModelOnly",
            "inputs": {"model": ["2", 0], "lora_name": args.student_lora, "strength_model": 1.0},
        }
        model_source = ["3", 0]
    else:
        model_source = ["2", 0]
    graph.update({
        "10": {
            "class_type": "CLIPLoader",
            "inputs": {"clip_name": args.text_encoder, "type": "qwen_image", "device": "default"},
        },
        "11": {
            "class_type": "TextEncodeQwenImage21",
            "inputs": {
                "clip": ["10", 0],
                "prompt": prompt,
                "negative_prompt": "",
                "encode_negative": False,
                "resolution": args.size,
                "images": {},
            },
        },
        "12": {"class_type": "EmptyLatentImage", "inputs": {"width": args.size, "height": args.size, "batch_size": 1}},
        "13": {"class_type": "RandomNoise", "inputs": {"noise_seed": args.seed}},
        "14": {"class_type": "KSamplerSelect", "inputs": {"sampler_name": "euler"}},
        "15": {"class_type": "ManualSigmas", "inputs": {"sigmas": sigmas_text}},
        "16": {"class_type": "BasicGuider", "inputs": {"model": ["21", 0], "conditioning": ["11", 0]}},
        "17": {
            "class_type": "SamplerCustomAdvanced",
            "inputs": {
                "noise": ["13", 0],
                "guider": ["16", 0],
                "sampler": ["14", 0],
                "sigmas": ["15", 0],
                "latent_image": ["12", 0],
            },
        },
        "18": {"class_type": "VAEDecode", "inputs": {"samples": ["17", 0], "vae": ["19", 0]}},
        "19": {"class_type": "VAELoader", "inputs": {"vae_name": args.vae}},
        "20": {"class_type": "SaveImage", "inputs": {"images": ["18", 0], "filename_prefix": filename_prefix}},
        "21": {
            "class_type": "QwenImage21Cache",
            "inputs": {"model": model_source, "device": "gpu", "dtype": "default"},
        },
    })
    return graph


def newest_file(directory: Path, pattern: str) -> Path:
    matches = [path for path in directory.glob(pattern) if path.is_file()]
    if not matches:
        raise FileNotFoundError(f"No output matched {directory / pattern}")
    return max(matches, key=lambda path: path.stat().st_mtime_ns)


def run_prompt(session, args, prompt: str, seed: int, index: int) -> dict:
    sample_id = f"{args.tag}_{index:03d}_seed{seed}"
    filename_prefix = f"research/student_eval/{args.tag}/{sample_id}"
    prompt_id = str(uuid.uuid4())
    graph = build_graph(args, prompt, shifted_sigmas(args.size, args.raw_sigmas), filename_prefix)
    started = time.perf_counter()
    response = session.post(
        f"{args.base_url}/prompt",
        json={"prompt": graph, "client_id": args.client_id, "prompt_id": prompt_id},
        timeout=30,
    )
    response.raise_for_status()
    if response.json().get("prompt_id") != prompt_id:
        raise RuntimeError("ComfyUI returned a different prompt id")

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
        time.sleep(0.05)
    if entry is None:
        raise TimeoutError(f"Timed out waiting for {prompt_id}")
    status = entry.get("status", {})
    if status.get("status_str") == "error":
        raise RuntimeError(json.dumps(status, indent=2))
    if not status.get("completed"):
        raise RuntimeError(f"Run did not complete: {status}")

    elapsed = time.perf_counter() - started
    source = newest_file(args.comfy_output / "research" / "student_eval" / args.tag, f"{sample_id}_*.png")
    destination = args.output_root / args.tag / source.name
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)
    return {
        "index": index,
        "sample_id": sample_id,
        "prompt": prompt,
        "seed": seed,
        "elapsed_seconds": elapsed,
        "raw_sigmas": args.raw_sigmas,
        "image": str(destination),
        "source": str(source),
        "filename_prefix": filename_prefix,
    }


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--comfy-output", type=Path, default=ROOT / "out" / "comfy")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--results", type=Path, default=ROOT / "results" / "qwen21_student_eval.json")
    parser.add_argument("--tag", default="two_step_r1")
    parser.add_argument("--size", type=int, default=512)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model", default="qwen_image_2.1_int8_convrot.safetensors")
    parser.add_argument("--teacher-lora", default=DEFAULT_TEACHER_LORA)
    parser.add_argument("--student-lora", default="")
    parser.add_argument("--text-encoder", default="qwen3vl_8b_int8_convrot.safetensors")
    parser.add_argument("--vae", default="qwen_image_2.1_vae_bf16.safetensors")
    parser.add_argument("--steps", type=int, choices=(1, 2, 6), default=2)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--measured", type=int, default=5)
    parser.add_argument("--timeout", type=float, default=600.0)
    parser.add_argument("--vary-prompt", action="store_true")
    parser.add_argument("--prompt", default="")
    return parser.parse_args()


def main():
    args = parse_args()
    args.raw_sigmas = TEACHER_RAW if args.steps == 6 else STUDENT_RAW[: args.steps]
    if args.size % 16:
        raise ValueError("Size must be divisible by 16")
    args.client_id = str(uuid.uuid4())
    prompt = args.prompt or (
        "A small blue glass sphere on a dark studio background, soft rim light, "
        "product photograph, centered composition"
    )
    records = []
    started = time.perf_counter()
    with requests.Session() as session:
        session.get(f"{args.base_url}/system_stats", timeout=30).raise_for_status()
        for index in range(args.warmup + args.measured):
            run_text = prompt
            if args.vary_prompt:
                run_text = f"{prompt} Variation marker {index}."
            record = run_prompt(session, args, run_text, args.seed, index)
            record["warmup"] = index < args.warmup
            records.append(record)
            print(json.dumps({key: value for key, value in record.items() if key != "source"}, indent=2), flush=True)

    measured = sorted(record["elapsed_seconds"] for record in records[args.warmup:])
    payload = {
        "tag": args.tag,
        "base_url": args.base_url,
        "size": args.size,
        "seed": args.seed,
        "steps": args.steps,
        "teacher_lora": args.teacher_lora,
        "student_lora": args.student_lora,
        "raw_sigmas": args.raw_sigmas,
        "latency_median_s": measured[len(measured) // 2] if measured else None,
        "latency_min_s": measured[0] if measured else None,
        "latency_max_s": measured[-1] if measured else None,
        "warmup_runs": args.warmup,
        "measured_runs": args.measured,
        "total_elapsed_seconds": time.perf_counter() - started,
        "records": records,
    }
    args.results.parent.mkdir(parents=True, exist_ok=True)
    args.results.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps({key: value for key, value in payload.items() if key != "records"}, indent=2))


if __name__ == "__main__":
    main()
