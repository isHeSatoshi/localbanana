"""Test whether the best two-step sigma generalizes across prompts and seeds.

P2 found, on a single prompt and seed, that the prescribed two-step grid [1.0, 0.25]
scores far worse than an interior optimum near [1.0, 0.78]. A single-prompt result is not
a finding, so this script re-measures the same comparison across several prompts and
seeds, scoring every candidate against a six-step reference generated for the same prompt
and seed.

Every generation goes through ``comfy_bench.py``, the accepted harness, so the graph and
measurement path are the ones the production numbers come from.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BENCH = ROOT / "scripts" / "comfy_bench.py"
EVAL = ROOT / "scripts" / "quality_eval.py"

PROMPTS = [
    "A small blue glass sphere on a dark studio background, soft rim light, product photograph, centered composition",
    "A red vintage bicycle leaning against a sunlit limewashed wall, realistic editorial photography",
    "A misty mountain lake at sunrise, layered ridges, calm water, wide landscape photograph, natural colors",
    "A futuristic transparent mechanical wristwatch on black acrylic, detailed product macro, cyan reflections",
]

CLIP_ENV = "QWEN_CLIP_MODEL"
DINO_ENV = "QWEN_DINO_MODEL"
CLIP_DEFAULT_ID = "openai/clip-vit-base-patch32"
DINO_DEFAULT_ID = "facebook/dinov2-base"


def resolve_model(name_or_id: str) -> str:
    """Return a local snapshot path if one is given, else the hub id.

    quality_eval.py loads with local_files_only, so a bare hub id works as long as the
    checkpoint is already in the local Hugging Face cache. Set the environment variables
    below to pin an explicit snapshot directory instead.
    """
    candidate = Path(name_or_id)
    if candidate.is_dir():
        return str(candidate)
    return name_or_id

SIX_STEP = [1.0, 0.9375, 0.875, 0.75, 0.5, 0.25]


def generate(base_url: str, tag: str, prompt: str, seed: int, raw: list[float], size: int, results: Path) -> str:
    command = [
        sys.executable, str(BENCH),
        "--base-url", base_url,
        "--tag", tag,
        "--size", str(size),
        "--seed", str(seed),
        "--prompt", prompt,
        "--raw-sigmas", ",".join(f"{value:g}" for value in raw),
        "--positive-only",
        "--cache-device", "gpu",
        "--cache-dtype", "default",
        "--warmup", "0",
        "--repeats", "1",
        "--results-path", str(results),
    ]
    completed = subprocess.run(command, capture_output=True, text=True, timeout=900)
    if completed.returncode != 0:
        raise RuntimeError(f"bench failed for {tag}:\n{completed.stdout[-2000:]}\n{completed.stderr[-2000:]}")
    records = json.loads(results.read_text(encoding="utf-8"))
    measured = [record for record in records if not record.get("warmup")]
    if not measured or not measured[0].get("outputs"):
        raise RuntimeError(f"no image produced for {tag}")
    return measured[0]["outputs"][0]["copy"]


def score(reference: str, candidates: list[str], prompt: str, output: Path, clip: str, dino: str) -> list[dict]:
    command = [
        sys.executable, str(EVAL),
        "--reference", reference,
        "--prompt", prompt,
        "--clip-model", clip,
        "--dino-model", dino,
        "--output", str(output),
    ]
    for candidate in candidates:
        command += ["--candidate", candidate]
    completed = subprocess.run(command, capture_output=True, text=True, timeout=900)
    if completed.returncode != 0:
        raise RuntimeError(f"eval failed:\n{completed.stdout[-2000:]}\n{completed.stderr[-2000:]}")
    return json.loads(output.read_text(encoding="utf-8"))["candidates"]


def parse_grids(spec: str) -> list[tuple[str, list[float]]]:
    grids: list[tuple[str, list[float]]] = []
    for item in spec.split(";"):
        item = item.strip()
        if not item:
            continue
        label, _, values = item.partition("=")
        if not _:
            label, values = item, item
        grids.append((label.strip(), [float(value) for value in values.split(",")]))
    if not grids:
        raise ValueError("no grids given")
    return grids


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8188")
    parser.add_argument("--tag", default="schedgen")
    parser.add_argument("--seeds", default="101,202,303")
    parser.add_argument(
        "--grids",
        default="b025=1.0,0.25;b050=1.0,0.50;b065=1.0,0.65;b075=1.0,0.75;b078=1.0,0.78;b085=1.0,0.85",
        help="semicolon separated label=comma,sigmas; the first grid is the candidate set",
    )
    parser.add_argument("--size", type=int, default=512)
    parser.add_argument("--prompts", type=int, default=4)
    parser.add_argument(
        "--clip-model",
        default=os.environ.get(CLIP_ENV, CLIP_DEFAULT_ID),
        help="local snapshot directory, or a hub id already present in the local cache",
    )
    parser.add_argument(
        "--dino-model",
        default=os.environ.get(DINO_ENV, DINO_DEFAULT_ID),
        help="local snapshot directory, or a hub id already present in the local cache",
    )
    parser.add_argument("--work-dir", type=Path, default=ROOT / "results" / "sweep_raw")
    parser.add_argument("--results-path", type=Path, default=ROOT / "results" / "schedule_generalization.json")
    return parser.parse_args()


def main():
    args = parse_args()
    seeds = [int(value) for value in args.seeds.split(",") if value.strip()]
    grids = parse_grids(args.grids)
    clip = resolve_model(args.clip_model)
    dino = resolve_model(args.dino_model)
    prompts = PROMPTS[: args.prompts]
    args.work_dir.mkdir(parents=True, exist_ok=True)

    started = time.perf_counter()
    combos = []
    for prompt_index, prompt in enumerate(prompts):
        for seed in seeds:
            key = f"p{prompt_index}_s{seed}"
            reference = generate(
                args.base_url, f"{args.tag}_{key}_ref6", prompt, seed, SIX_STEP, args.size,
                args.work_dir / f"{args.tag}_{key}_ref6.json",
            )
            candidates = []
            for label, raw in grids:
                candidates.append(
                    generate(
                        args.base_url, f"{args.tag}_{key}_{label}", prompt, seed, raw, args.size,
                        args.work_dir / f"{args.tag}_{key}_{label}.json",
                    )
                )
            scored = score(
                reference,
                candidates,
                prompt,
                args.work_dir / f"{args.tag}_{key}_score.json",
                clip,
                dino,
            )
            row = {
                "key": key,
                "prompt_index": prompt_index,
                "seed": seed,
                "prompt": prompt,
                "reference": reference,
                "results": {},
            }
            for (label, _raw), record in zip(grids, scored):
                row["results"][label] = {
                    "dino_cosine": record["dino_image_cosine_vs_reference"],
                    "clip_cosine": record["clip_image_cosine_vs_reference"],
                    "pixel_mae": record["pixel_mae_rgb"],
                }
            combos.append(row)
            means = {
                label: round(
                    statistics.fmean([combo["results"][label]["dino_cosine"] for combo in combos]), 4
                )
                for label, _ in grids
            }
            print(
                json.dumps(
                    {
                        "key": key,
                        "dino": {label: round(row["results"][label]["dino_cosine"], 4) for label, _ in grids},
                        "running_mean": means,
                    }
                ),
                flush=True,
            )

    labels = [label for label, _ in grids]
    summary = {
        "grids": dict(grids),
        "n_prompts": len(prompts),
        "n_seeds": len(seeds),
        "n_combos": len(combos),
        "mean_dino_cosine": {
            label: statistics.fmean([combo["results"][label]["dino_cosine"] for combo in combos])
            for label in labels
        },
        "median_dino_cosine": {
            label: statistics.median([combo["results"][label]["dino_cosine"] for combo in combos])
            for label in labels
        },
        "min_dino_cosine": {
            label: min(combo["results"][label]["dino_cosine"] for combo in combos)
            for label in labels
        },
        "max_dino_cosine": {
            label: max(combo["results"][label]["dino_cosine"] for combo in combos)
            for label in labels
        },
        "mean_clip_cosine": {
            label: statistics.fmean([combo["results"][label]["clip_cosine"] for combo in combos])
            for label in labels
        },
        "per_prompt_mean_dino": {
            str(prompt_index): {
                label: round(
                    statistics.fmean(
                        [c["results"][label]["dino_cosine"] for c in combos if c["prompt_index"] == prompt_index]
                    ),
                    4,
                )
                for label in labels
            }
            for prompt_index in range(len(prompts))
        },
    }
    payload = {
        "tag": args.tag,
        "size": args.size,
        "seeds": seeds,
        "prompts": prompts,
        "summary": summary,
        "combos": combos,
        "total_elapsed_seconds": time.perf_counter() - started,
    }
    args.results_path.parent.mkdir(parents=True, exist_ok=True)
    args.results_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"wrote {args.results_path}")


if __name__ == "__main__":
    main()
