"""Generate the sample gallery published in the README.

Runs through the same ComfyClient and build_graph the dashboard uses, so every sample is
produced by the code in this repository rather than by a separate path. Outputs land in
samples/ with a manifest recording the prompt, seed, profile, size, and measured latency for
each one.
"""

from __future__ import annotations

import importlib.util
import json
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
SAMPLES = ROOT / "samples"

# Five steps is the accepted fast profile and holds the sub-two-second claim, so that is
# what the gallery uses. Seeds were chosen by generating three per prompt and keeping the
# strongest result.
PROFILE = "Fast (5-step Turbo v0.2.1 r128)"

TEXTS = [
    {
        "name": "portrait-film-grain",
        "prompt": (
            "Cinematic photorealistic film portrait of a young woman in a worn leather jacket, "
            "soft warm window light, heavy 35mm film grain, muted teal and amber tones, shallow "
            "depth of field, shot on Kodak Portra 400"
        ),
        "size": (768, 768),
        "seed": 20260928,
    },
    {
        "name": "portrait-chinese",
        "prompt": "生成一张电影感写实人像，柔和自然光与暖色调，浅景深，韩系都市偶像剧质感，细腻胶片颗粒",
        "size": (768, 768),
        "seed": 20260928,
    },
    {
        "name": "infographic-bioluminescent",
        "prompt": (
            "Editorial magazine infographic spread about deep-sea bioluminescence. Dark navy "
            "background, a large detailed cutaway illustration of a giant squid on the left, "
            "connected by thin leader lines to columns of small caps fact text on the right. "
            "Accent colours in cyan and warm amber. Clean grid, generous margins, crisp "
            "technical linework, flat vector style with subtle grain"
        ),
        "size": (1024, 640),
        "seed": 303,
    },
    {
        "name": "poster-night-train",
        "prompt": (
            "Design a refined travel poster for a fictional night train called the Meridian. "
            "Deep indigo sky, a single lit carriage on a viaduct, art-deco lettering, "
            "grain-textured print, limited palette of indigo, brass and cream"
        ),
        "size": (896, 1152),
        "seed": 3141,
    },
    {
        "name": "poster-swiss-rail",
        "prompt": (
            "Vintage 1960s Swiss railway poster infographic, landscape. Bold geometric mountain "
            "range stacked in receding planes, a sleek train crossing a viaduct, large "
            "condensed sans-serif type, four-colour screenprint palette of red, cream, teal and "
            "black, visible halftone texture, clean 1960s modernist layout"
        ),
        "size": (1024, 640),
        "seed": 303,
    },
    {
        "name": "product-watch",
        "prompt": (
            "Macro studio photograph of a polished stainless steel chronograph wristwatch lying "
            "at an angle on dark slate, three small dial sub-registers, applied steel hour "
            "markers, a domed sapphire crystal showing a soft window reflection, controlled rim "
            "light along the case edge, extremely sharp detail"
        ),
        "size": (768, 768),
        "seed": 303,
    },
    {
        "name": "product-decanter",
        "prompt": (
            "Studio product photograph of a heavy faceted crystal decanter with a ground glass "
            "stopper, filled with amber liquid, standing on a pale grey seamless backdrop. Two "
            "large softbox reflections visible in the glass, a crisp caustic light pattern on "
            "the surface, deep controlled shadow, sharp focus on the facets"
        ),
        "size": (768, 768),
        "seed": 303,
    },
    {
        "name": "graphic-launch",
        "prompt": (
            "Premium minimalist launch graphic for an AI image tool, a single luminous sphere "
            "over a matte black field, restrained sans-serif wordmark, generous negative space, "
            "printed-brochure finish"
        ),
        "size": (1024, 640),
        "seed": 5150,
    },
]

# A reference-image sample, run after the portrait so it has a real input to condition on.
REFERENCE_SAMPLE = {
    "name": "reference-wardrobe-change",
    "prompt": (
        "The person from the reference photo is now wearing a tailored cream wool coat, standing "
        "in the same room, same lighting, same film grain"
    ),
    "size": (768, 768),
    "seed": 20260929,
    "reference": "portrait-film-grain",
}


def load_dashboard():
    spec = importlib.util.spec_from_file_location("dashboard_module", ROOT / "dashboard.py")
    module = importlib.util.module_from_spec(spec)
    sys.argv = ["dashboard.py"]
    spec.loader.exec_module(module)
    return module


def main() -> int:
    SAMPLES.mkdir(parents=True, exist_ok=True)
    dash = load_dashboard()
    client = dash.ComfyClient()
    manifest: list[dict] = []

    def run(entry: dict, reference_path: Path | None = None) -> None:
        uploaded = []
        if reference_path is not None:
            uploaded.append(client.upload_image(str(reference_path)))
        width, height = entry["size"]
        graph = dash.build_graph(
            entry["prompt"], width, height, entry["seed"], PROFILE, uploaded
        )
        started = time.perf_counter()
        prompt_id = client.submit(graph)
        history = client.wait_for_history(prompt_id)
        image_path, _ = client.image_output(history)
        elapsed = time.perf_counter() - started

        destination = SAMPLES / f"{entry['name']}.png"
        source = Path(image_path)
        destination.write_bytes(source.read_bytes())
        manifest.append(
            {
                "name": entry["name"],
                "file": f"samples/{destination.name}",
                "prompt": entry["prompt"],
                "seed": entry["seed"],
                "width": width,
                "height": height,
                "profile": PROFILE,
                "reference": reference_path.name if reference_path else None,
                "elapsed_seconds": round(elapsed, 3),
            }
        )
        print(f"{entry['name']:26s} {width}x{height}  {elapsed:.2f}s", flush=True)

    for entry in TEXTS:
        run(entry)

    portrait = SAMPLES / f"{REFERENCE_SAMPLE['reference']}.png"
    if portrait.is_file():
        run(REFERENCE_SAMPLE, reference_path=portrait)
    else:
        print(f"skipped {REFERENCE_SAMPLE['name']}: {portrait} missing")

    (SAMPLES / "manifest.json").write_text(
        json.dumps({"profile": PROFILE, "samples": manifest}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"wrote {SAMPLES / 'manifest.json'} with {len(manifest)} samples")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
