"""Score generated images against a reference with local CLIP and DINO checkpoints."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModel, CLIPModel, CLIPProcessor


def image_array(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0


def normalize(features: torch.Tensor) -> torch.Tensor:
    return torch.nn.functional.normalize(features.float(), dim=-1)


def pixel_metrics(reference: np.ndarray, candidate: np.ndarray) -> dict[str, float]:
    if reference.shape != candidate.shape:
        raise ValueError(f"Image shapes differ: {reference.shape} vs {candidate.shape}")
    difference = candidate - reference
    return {
        "pixel_mae_rgb": float(np.abs(difference).mean()),
        "pixel_rmse_rgb": float(np.sqrt(np.mean(difference * difference))),
        "pixel_max_abs_rgb": float(np.abs(difference).max()),
        "neighbor_gradient_mean": float(np.mean(np.abs(np.gradient(candidate, axis=(0, 1))))),
        "laplacian_mean_abs": float(np.mean(np.abs(4.0 * candidate - np.roll(candidate, 1, axis=0) - np.roll(candidate, -1, axis=0) - np.roll(candidate, 1, axis=1) - np.roll(candidate, -1, axis=1)))),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, action="append", required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--clip-model", type=Path, required=True)
    parser.add_argument("--dino-model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    paths = [args.reference, *args.candidate]
    images = [Image.open(path).convert("RGB") for path in paths]
    reference_pixels = image_array(args.reference)

    clip_processor = CLIPProcessor.from_pretrained(args.clip_model, local_files_only=True)
    clip_model = CLIPModel.from_pretrained(args.clip_model, local_files_only=True).eval()
    dino_processor = AutoImageProcessor.from_pretrained(args.dino_model, local_files_only=True)
    dino_model = AutoModel.from_pretrained(args.dino_model, local_files_only=True).eval()

    with torch.inference_mode():
        clip_inputs = clip_processor(text=[args.prompt] * len(images), images=images, return_tensors="pt", padding=True)
        clip_outputs = clip_model(**clip_inputs)
        clip_images = normalize(clip_outputs.image_embeds)
        clip_text = normalize(clip_outputs.text_embeds)
        dino_inputs = dino_processor(images=images, return_tensors="pt")
        dino_outputs = dino_model(**dino_inputs)
        dino_images = normalize(dino_outputs.last_hidden_state[:, 0])

    records = []
    for index, path in enumerate(paths):
        if index == 0:
            continue
        candidate_pixels = image_array(path)
        record = {
            "label": path.parent.name or path.stem,
            "image": str(path),
            "pixel_mae_rgb": pixel_metrics(reference_pixels, candidate_pixels)["pixel_mae_rgb"],
            "pixel_rmse_rgb": pixel_metrics(reference_pixels, candidate_pixels)["pixel_rmse_rgb"],
            "pixel_max_abs_rgb": pixel_metrics(reference_pixels, candidate_pixels)["pixel_max_abs_rgb"],
            "neighbor_gradient_mean": pixel_metrics(reference_pixels, candidate_pixels)["neighbor_gradient_mean"],
            "laplacian_mean_abs": pixel_metrics(reference_pixels, candidate_pixels)["laplacian_mean_abs"],
            "clip_prompt_alignment": float((clip_images[index] * clip_text[index]).sum().item()),
            "clip_image_cosine_vs_reference": float((clip_images[index] * clip_images[0]).sum().item()),
            "dino_image_cosine_vs_reference": float((dino_images[index] * dino_images[0]).sum().item()),
            "dino_l2_vs_reference": float(torch.linalg.vector_norm(dino_images[index] - dino_images[0]).item()),
        }
        records.append(record)

    payload = {
        "reference": str(args.reference),
        "prompt": args.prompt,
        "clip_model": str(args.clip_model),
        "dino_model": str(args.dino_model),
        "candidates": records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
