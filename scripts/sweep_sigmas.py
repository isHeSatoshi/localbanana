"""Drive the accepted comfy_bench harness over a list of sigma grids.

This deliberately shells out to ``comfy_bench.py`` rather than reimplementing graph
construction, so every grid is measured by exactly the same code path as the accepted
production numbers.

Two modes, because the accepted protocol and the quality protocol need different prompt
handling:

- ``--mode quality`` keeps prompt and seed fixed across grids so the images are directly
  comparable. The text-encode node is then served from cache, so its reported duration is
  NOT a valid latency sample. Sampler nodes differ between grids, so the sampler always
  re-executes. This mode is for image quality only.
- ``--mode latency`` varies the prompt every request so the classic graph cache cannot
  return identical nodes. Without this the harness reports a false ~0.06 s.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BENCH = ROOT / "scripts" / "comfy_bench.py"
DEFAULT_PROMPT = "A small blue glass sphere on a dark studio background, soft rim light, product photograph, centered composition"


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


def run_bench(args, label: str, raw: list[float], repeats: int, warmup: int, results: Path) -> dict:
    command = [
        sys.executable,
        str(BENCH),
        "--tag", f"{args.tag}_{label}",
        "--size", str(args.size),
        "--seed", str(args.seed),
        "--prompt", args.prompt,
        "--raw-sigmas", ",".join(f"{value:g}" for value in raw),
        "--positive-only",
        "--cache-device", args.cache_device,
        "--cache-dtype", args.cache_dtype,
        "--warmup", str(warmup),
        "--repeats", str(repeats),
        "--results-path", str(results),
    ]
    if args.vary_prompt:
        command.append("--vary-prompt")
    if args.vary_seed:
        command.append("--vary-seed")
    completed = subprocess.run(command, capture_output=True, text=True, timeout=args.timeout)
    if completed.returncode != 0:
        raise RuntimeError(f"bench failed for {label}:\n{completed.stdout[-3000:]}\n{completed.stderr[-3000:]}")
    return json.loads(results.read_text(encoding="utf-8"))


def cached_sampler_runs(records: list[dict]) -> list[int]:
    """Runs where ComfyUI reported execution_cached.

    A cached run means the graph was not re-executed, so its elapsed time is not a
    latency sample. In latency mode any hit here invalidates the configuration.
    """
    hits = []
    for record in records:
        messages = (record.get("history_status") or {}).get("messages") or []
        for message in messages:
            if isinstance(message, list) and message and message[0] == "execution_cached":
                payload = message[1] if len(message) > 1 and isinstance(message[1], dict) else {}
                nodes = [str(node) for node in (payload.get("nodes") or [])]
                if not nodes or "11" in nodes:
                    hits.append(record.get("run_index"))
                break
    return hits


def summarize(label: str, raw: list[float], records: list[dict]) -> dict:
    measured = [record for record in records if not record.get("warmup")]
    elapsed = sorted(record["elapsed_seconds"] for record in measured)
    median = elapsed[len(elapsed) // 2] if elapsed else None

    stages: dict[str, list[float]] = {}
    for record in measured:
        for node_id, info in (record.get("node_durations") or {}).items():
            stages.setdefault(info["class_type"], []).append(info["seconds"])
    stage_medians = {
        name: sorted(values)[len(values) // 2] for name, values in stages.items() if values
    }

    images = [record["outputs"][0]["copy"] for record in measured if record.get("outputs")]
    intervals = [record.get("step_intervals") or [] for record in measured]
    cached = cached_sampler_runs(records)

    return {
        "label": label,
        "raw_sigmas": raw,
        "steps": len(raw),
        "median_elapsed_seconds": median,
        "measured_runs": len(measured),
        "stage_medians_seconds": stage_medians,
        "images": images,
        "cached_runs": cached,
        "latency_valid": not cached and bool(measured),
        "sampler_step_intervals": intervals[0] if intervals else [],
    }


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tag", default="step_curve")
    parser.add_argument("--mode", choices=["quality", "latency"], default="quality")
    parser.add_argument("--grids", required=True, help="semicolon separated label=comma,sigmas")
    parser.add_argument("--size", type=int, default=512)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--vary-prompt", action="store_true")
    parser.add_argument("--vary-seed", action="store_true")
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--cache-device", default="gpu")
    parser.add_argument("--cache-dtype", default="default")
    parser.add_argument("--results-path", type=Path)
    parser.add_argument("--raw-results-dir", type=Path, default=ROOT / "results" / "sweep_raw")
    parser.add_argument("--timeout", type=float, default=3600.0)
    args = parser.parse_args()
    if args.size % 32:
        parser.error("--size must be divisible by 32")
    if args.results_path is None:
        args.results_path = ROOT / "results" / f"sweep_{args.tag}.json"
    if args.mode == "quality":
        args.vary_prompt = False
        args.vary_seed = False
        # A per-grid warmup would execute the grid and then let the measured run hit the
        # classic cache, returning the same image with a meaningless ~0.03 s timing. In
        # quality mode we warm the model once on a throwaway prompt and then run every grid
        # exactly once, so each grid's graph is unique and genuinely executes.
        args.warmup = 0
    else:
        # Latency mode must defeat the classic cache or every sample is a false ~0.03 s.
        # Enforced rather than left as a flag, because getting this wrong silently
        # invalidates the whole measurement.
        args.vary_prompt = True
    return args


def main():
    args = parse_args()
    grids = parse_grids(args.grids)
    args.raw_results_dir.mkdir(parents=True, exist_ok=True)
    summaries = []
    started = time.perf_counter()

    if args.mode == "quality":
        warm_args = argparse.Namespace(**vars(args))
        warm_args.prompt = f"{args.prompt} Model warmup, discard this image."
        warm_path = args.raw_results_dir / f"{args.tag}_warmup.json"
        run_bench(warm_args, "warmup", grids[0][1], 1, 0, warm_path)

    for label, raw in grids:
        repeats = max(1, args.repeats)
        result_path = args.raw_results_dir / f"{args.tag}_{label}.json"
        records = run_bench(args, label, raw, repeats, args.warmup, result_path)
        summary = summarize(label, raw, records)
        summaries.append(summary)
        print(
            json.dumps(
                {
                    "label": summary["label"],
                    "steps": summary["steps"],
                    "median_s": summary["median_elapsed_seconds"],
                    "stages": summary["stage_medians_seconds"],
                    "images": len(summary["images"]),
                }
            ),
            flush=True,
        )

    payload = {
        "tag": args.tag,
        "mode": args.mode,
        "size": args.size,
        "seed": args.seed,
        "prompt": args.prompt,
        "vary_prompt": args.vary_prompt,
        "grids": summaries,
        "total_elapsed_seconds": time.perf_counter() - started,
    }
    args.results_path.parent.mkdir(parents=True, exist_ok=True)
    args.results_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"wrote {args.results_path}")


if __name__ == "__main__":
    main()
