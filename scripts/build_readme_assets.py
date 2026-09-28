"""Build the README hero assets: a cropped dashboard shot and a step-count sample strip.

Both are assembled from files already produced by the pipeline, so nothing in the README is
a mock-up.
"""

from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw


ROOT = Path(__file__).resolve().parent.parent
BENCH = ROOT / "out" / "benchmarks"
DOCS = ROOT / "docs"

SAMPLES = [
    ("quality_v021_p1_r256_6_512_000/quality_v021_p1_r256_6_512_000_00001_.png", "6 steps, 1.84 s"),
    ("p1b_cliff_s5_512_000/p1b_cliff_s5_512_000_00001_.png", "5 steps, 1.77 s"),
    ("p1b_cliff_s3_512_000/p1b_cliff_s3_512_000_00001_.png", "3 steps, 1.45 s"),
]

LABEL_BAR = 34
GAP = 12
BG = (13, 17, 23)
FG = (222, 228, 236)
MUTED = (139, 148, 158)


def build_samples() -> Path:
    images = [Image.open(BENCH / rel).convert("RGB") for rel, _ in SAMPLES]
    width, height = images[0].size
    total = len(images) * width + GAP * (len(images) - 1)
    canvas = Image.new("RGB", (total, height + LABEL_BAR), BG)
    draw = ImageDraw.Draw(canvas)
    for index, (image, (_, caption)) in enumerate(zip(images, SAMPLES)):
        x = index * (width + GAP)
        canvas.paste(image, (x, 0))
        draw.text((x + 10, height + 10), caption, fill=FG)
    out = DOCS / "samples.png"
    canvas.save(out, optimize=True)
    return out


def crop_dashboard() -> Path:
    """Trim the empty right-hand gutter from the raw capture.

    Reads docs/dashboard_raw.png and writes docs/dashboard.png, so running this twice
    cannot crop an already-cropped image.
    """
    raw = DOCS / "dashboard_raw.png"
    if not raw.is_file():
        return DOCS / "dashboard.png"
    source = Image.open(raw).convert("RGB")
    width, height = source.size
    keep_width = int(width * 0.88)
    out = source.crop((0, 0, keep_width, height))
    out.save(DOCS / "dashboard.png", optimize=True)
    return DOCS / "dashboard.png"


if __name__ == "__main__":
    DOCS.mkdir(parents=True, exist_ok=True)
    for path in (crop_dashboard(), build_samples()):
        print(f"{path.relative_to(ROOT)}  {path.stat().st_size} bytes")
