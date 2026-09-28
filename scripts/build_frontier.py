"""Combine measured step-count quality and latency into one quality-versus-time frontier.

Both inputs are required to exist and to be latency-valid. Any grid whose latency samples
were served from the classic graph cache is excluded rather than reported, because such a
sample is not a measurement.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def load_quality(path: Path) -> dict[str, dict]:
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    out: dict[str, dict] = {}
    for record in payload["candidates"]:
        label = record["label"]
        for prefix in ("p1b_cliff_", "p2_grid_b", "p2b_grid_b"):
            if label.startswith(prefix):
                key = label.replace(prefix, "").split("_512_")[0]
                out[key] = {
                    "dino_cosine": record["dino_image_cosine_vs_reference"],
                    "clip_cosine": record["clip_image_cosine_vs_reference"],
                    "pixel_mae": record["pixel_mae_rgb"],
                }
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quality", type=Path, action="append", required=True)
    parser.add_argument("--latency", type=Path, required=True)
    parser.add_argument("--label-map", type=Path, required=True, help="JSON mapping latency label -> quality key")
    parser.add_argument("--out", type=Path, default=ROOT / "results" / "frontier.json")
    parser.add_argument("--markdown-out", type=Path, default=ROOT / "results" / "frontier.md")
    args = parser.parse_args()

    quality: dict[str, dict] = {}
    for path in args.quality:
        quality.update(load_quality(path))
    latency = json.loads(args.latency.read_text(encoding="utf-8-sig"))
    mapping = json.loads(args.label_map.read_text(encoding="utf-8-sig"))

    rows = []
    for grid in latency["grids"]:
        key = mapping.get(grid["label"])
        if key is None or key not in quality:
            continue
        if not grid.get("latency_valid"):
            continue
        stages = grid.get("stage_medians_seconds") or {}
        rows.append(
            {
                "label": grid["label"],
                "raw_sigmas": grid["raw_sigmas"],
                "steps": grid["steps"],
                "median_seconds": grid["median_elapsed_seconds"],
                "text_seconds": stages.get("TextEncodeQwenImage21"),
                "denoise_seconds": stages.get("SamplerCustomAdvanced"),
                "vae_seconds": stages.get("VAEDecode"),
                "measured_runs": grid["measured_runs"],
                **quality[key],
            }
        )
    rows.sort(key=lambda row: row["median_seconds"])
    args.out.write_text(json.dumps({"rows": rows}, indent=2), encoding="utf-8")

    lines = [
        "| steps | schedule (raw sigma) | warm median s | DINO | CLIP | text s | denoise s | VAE s |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for row in rows:
        schedule = ",".join(f"{value:g}" for value in row["raw_sigmas"])
        lines.append(
            f"| {row['steps']} | `{schedule}` | {row['median_seconds']:.3f} | "
            f"{row['dino_cosine']:.4f} | {row['clip_cosine']:.4f} | "
            f"{row['text_seconds']:.3f} | {row['denoise_seconds']:.3f} | {row['vae_seconds']:.3f} |"
        )
    args.markdown_out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    print(f"\nwrote {args.out} and {args.markdown_out}")


if __name__ == "__main__":
    main()
