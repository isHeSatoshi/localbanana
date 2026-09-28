"""Scratch: generate seed variants so the weakest gallery samples can be replaced.

Not part of the published tooling. Generates a few seeds per candidate prompt at the
five-step profile so the best result can be chosen by eye before it goes in samples/.
"""

from __future__ import annotations

import importlib.util
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "out" / "sample_candidates"

CANDIDATES = [
    {
        "tag": "infographic",
        "prompts": {
            "a": (
                "Editorial magazine infographic spread about deep-sea bioluminescence. Dark navy "
                "background, a large detailed cutaway illustration of a giant squid on the left, "
                "connected by thin leader lines to columns of small caps fact text on the right. "
                "Accent colours in cyan and warm amber. Clean grid, generous margins, crisp "
                "technical linework, flat vector style with subtle grain"
            ),
            "b": (
                "Vintage 1960s Swiss railway poster infographic, landscape. Bold geometric "
                "mountain range stacked in receding planes, a sleek train crossing a viaduct, "
                "large condensed sans-serif type, four-colour screenprint palette of red, cream, "
                "teal and black, visible halftone texture, clean 1960s modernist layout"
            ),
        },
        "size": (1024, 640),
    },
    {
        "tag": "stilllife",
        "prompts": {
            "a": (
                "Studio product photograph of a heavy faceted crystal decanter with a ground "
                "glass stopper, filled with amber liquid, standing on a pale grey seamless "
                "backdrop. Two large softbox reflections visible in the glass, a crisp caustic "
                "light pattern on the surface, deep controlled shadow, sharp focus on the facets"
            ),
            "b": (
                "Macro studio photograph of a polished stainless steel chronograph wristwatch "
                "lying at an angle on dark slate, three small dial sub-registers, applied steel "
                "hour markers, a domed sapphire crystal showing a soft window reflection, "
                "controlled rim light along the case edge, extremely sharp detail"
            ),
        },
        "size": (768, 768),
    },
]

SEEDS = [101, 202, 303]


def load_dashboard():
    spec = importlib.util.spec_from_file_location("dashboard_module", ROOT / "dashboard.py")
    module = importlib.util.module_from_spec(spec)
    sys.argv = ["dashboard.py"]
    spec.loader.exec_module(module)
    return module


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    dash = load_dashboard()
    client = dash.ComfyClient()
    profile = dash.FAST_PROFILE

    for candidate in CANDIDATES:
        width, height = candidate["size"]
        for variant, prompt in candidate["prompts"].items():
            for seed in SEEDS:
                graph = dash.build_graph(prompt, width, height, seed, profile, [])
                started = time.perf_counter()
                history = client.wait_for_history(client.submit(graph))
                image_path, _ = client.image_output(history)
                elapsed = time.perf_counter() - started
                name = f"{candidate['tag']}_{variant}_{seed}.png"
                (OUT / name).write_bytes(Path(image_path).read_bytes())
                print(f"{name:26s} {width}x{height} {elapsed:.2f}s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
