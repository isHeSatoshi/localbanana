"""Fit the per-NFE latency floor model from a measured step-count sweep.

Model: denoise(N) = a + (N-1) * c, fitted by ordinary least squares on the sampler node
median at each step count. The intercept ``a`` is the cost of the first denoiser call and
``c`` is the marginal cost of each additional call, so ``a - c`` is the one-off surcharge
paid only on the first call. The non-denoise fixed cost is the sum of the text-encode,
VAE-decode and image-save stage medians, and it is what remains even with zero denoising.
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--latency", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=ROOT / "results" / "latency_floor.json")
    args = parser.parse_args()

    payload = json.loads(args.latency.read_text(encoding="utf-8-sig"))
    grids = [grid for grid in payload["grids"] if grid.get("latency_valid")]
    if len(grids) < 2:
        raise SystemExit("need at least two latency-valid grids")

    xs, ys = [], []
    stages: dict[str, list[float]] = {}
    for grid in grids:
        sampler = (grid.get("stage_medians_seconds") or {}).get("SamplerCustomAdvanced")
        if sampler is None:
            continue
        xs.append(grid["steps"] - 1)
        ys.append(sampler)
        for name, value in (grid.get("stage_medians_seconds") or {}).items():
            stages.setdefault(name, []).append(value)

    n = len(xs)
    mean_x = statistics.fmean(xs)
    mean_y = statistics.fmean(ys)
    sxy = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
    sxx = sum((x - mean_x) ** 2 for x in xs)
    c = sxy / sxx
    a = mean_y - c * mean_x

    residuals = [y - (a + c * x) for x, y in zip(xs, ys)]
    stage_means = {name: statistics.fmean(values) for name, values in stages.items() if name != "SamplerCustomAdvanced"}
    fixed = sum(stage_means.values())

    result = {
        "source": str(args.latency),
        "prompt": payload.get("prompt"),
        "n_points": n,
        "denoise_model": "denoise(N) = a + (N-1)*c",
        "a_first_call_seconds": a,
        "c_marginal_step_seconds": c,
        "first_call_surcharge_seconds": a - c,
        "non_denoise_fixed_seconds": fixed,
        "stage_mean_seconds": stage_means,
        "fit_residuals_seconds": residuals,
        "max_abs_residual_seconds": max(abs(r) for r in residuals),
        "predicted_total_seconds": {str(n_): fixed + a + c * (n_ - 1) for n_ in range(1, 9)},
        "measured_total_seconds": {
            str(grid["steps"]): grid["median_elapsed_seconds"] for grid in grids if grid["median_elapsed_seconds"]
        },
        "one_step_seconds_predicted": fixed + a,
        "zero_denoise_floor_seconds": fixed,
    }
    args.out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
