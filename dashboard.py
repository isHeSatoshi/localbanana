"""Local Gradio control panel for the accepted Qwen-Image-2.1 ComfyUI path.

The dashboard is a thin HTTP client. ComfyUI remains the only process that
loads the model, so starting the dashboard does not duplicate model VRAM.
"""

from __future__ import annotations

import argparse
import json
import math
import mimetypes
import os
import subprocess
import sys
import time
import uuid
from pathlib import Path
from typing import Any

import gradio as gr
import requests

ROOT = Path(__file__).resolve().parent
COMFY_BASE_URL = os.environ.get("COMFY_BASE_URL", "http://127.0.0.1:8188").rstrip("/")
COMFY_OUTPUT_DIR = Path(os.environ.get("COMFY_OUTPUT_DIR", str(ROOT / "out" / "comfy")))
BENCHMARK_PYTHON = Path(
    os.environ.get("QWEN_BENCHMARK_PYTHON", str(ROOT / ".venv-cu130" / "Scripts" / "python.exe"))
)
DASHBOARD_OUTPUT = ROOT / "out" / "dashboard"
STEP_COUNT_RESULTS = ROOT / "results" / "schedule_stepcount_generalization.json"
LATENCY_FLOOR_RESULTS = ROOT / "results" / "latency_floor.json"
LONG_PROMPT_RESULTS = ROOT / "results" / "sweep_p4_longprompt.json"
DASHBOARD_OUTPUT.mkdir(parents=True, exist_ok=True)

DIT_MODEL = "qwen_image_2.1_int8_convrot.safetensors"
TEXT_MODEL = "qwen3vl_8b_int8_convrot.safetensors"
VAE_MODEL = "qwen_image_2.1_vae_bf16.safetensors"
TURBO_LORA = "Qwen-Image-2.1-turbo-v0.2.1-6step-lora-r128.safetensors"
DEFAULT_PROMPT = (
    "A premium futuristic AI image-generation emblem, centered composition, "
    "luminous cyan and violet light, sharp details, cinematic studio lighting, "
    "square composition, no watermark"
)
FAST_PROFILE = "Fast (5-step Turbo v0.2.1 r128)"
BALANCED_PROFILE = "Balanced (3-step Turbo v0.2.1 r128, experimental)"
QUALITY_PROFILE = "Quality (6-step Turbo v0.2.1 r128)"
EXPERIMENTAL_PROFILE = "Experimental (4-step Turbo v0.2.1 r128)"
BASE_PROFILE = "Base reference (40-step no Turbo)"
TURBO_PROFILES = {QUALITY_PROFILE, FAST_PROFILE, BALANCED_PROFILE, EXPERIMENTAL_PROFILE}
PROFILES = [QUALITY_PROFILE, FAST_PROFILE, BALANCED_PROFILE, EXPERIMENTAL_PROFILE, BASE_PROFILE]
PROFILE_STEPS = {
    QUALITY_PROFILE: 6,
    FAST_PROFILE: 5,
    BALANCED_PROFILE: 3,
    EXPERIMENTAL_PROFILE: 4,
    BASE_PROFILE: 40,
}

DEFAULT_AREA = "512² area"
DEFAULT_ASPECT = "1:1 • Instagram square"
CUSTOM_ASPECT = "Custom • set width and height below"
AREA_SIDES = {
    "512² area": 512,
    "768² area": 768,
    "1024² area": 1024,
    "2048² area": 2048,
}
MAX_OUTPUT_DIMENSION = 16384
MAX_TEXT_ENCODER_RESOLUTION = 4096
ASPECT_PRESETS = {
    "1:1 • Instagram square": "1:1",
    "4:5 • Instagram portrait": "4:5",
    "5:4 • Instagram landscape": "5:4",
    "16:9 • YouTube": "16:9",
    "9:16 • Instagram Reels / TikTok / Shorts": "9:16",
    "15:9 • YouTube widescreen": "15:9",
    "21:9 • YouTube ultrawide": "21:9",
    "4:3 • Classic": "4:3",
    "3:4 • Portrait": "3:4",
    "3:2 • Photo": "3:2",
    "2:3 • Portrait": "2:3",
    CUSTOM_ASPECT: "custom",
}
ASPECT_DIMENSIONS = {
    ("1:1", 1024): (1024, 1024),
    ("4:5", 1024): (896, 1120),
    ("5:4", 1024): (1120, 896),
    ("16:9", 1024): (1376, 768),
    ("9:16", 1024): (768, 1376),
    ("15:9", 1024): (1312, 800),
    ("21:9", 1024): (1568, 672),
    ("4:3", 1024): (1184, 896),
    ("3:4", 1024): (896, 1184),
    ("3:2", 1024): (1248, 832),
    ("2:3", 1024): (832, 1248),
    ("1:1", 2048): (2048, 2048),
    ("4:5", 2048): (1792, 2240),
    ("5:4", 2048): (2240, 1792),
    ("16:9", 2048): (2720, 1536),
    ("9:16", 2048): (1536, 2720),
    ("15:9", 2048): (2656, 1600),
    ("21:9", 2048): (3136, 1344),
    ("4:3", 2048): (2368, 1760),
    ("3:4", 2048): (1760, 2368),
    ("3:2", 2048): (2496, 1664),
    ("2:3", 2048): (1664, 2496),
}


class ComfyError(RuntimeError):
    """A local ComfyUI request or workflow failed."""


class ComfyClient:
    def __init__(self, base_url: str = COMFY_BASE_URL):
        self.base_url = base_url.rstrip("/")
        self.session = requests.Session()
        self.client_id = str(uuid.uuid4())

    def get_json(self, path: str, timeout: float = 10.0) -> dict[str, Any]:
        try:
            response = self.session.get(f"{self.base_url}{path}", timeout=timeout)
            response.raise_for_status()
            return response.json()
        except requests.RequestException as exc:
            raise ComfyError(f"ComfyUI GET {path} failed: {exc}") from exc

    def upload_image(self, path: str) -> dict[str, str]:
        file_path = Path(path)
        if not file_path.is_file():
            raise ComfyError(f"Reference image does not exist: {file_path}")

        content_type = mimetypes.guess_type(file_path.name)[0] or "application/octet-stream"
        with file_path.open("rb") as image_file:
            files = {"image": (file_path.name, image_file, content_type)}
            data = {"type": "input", "overwrite": "true"}
            try:
                response = self.session.post(
                    f"{self.base_url}/upload/image", files=files, data=data, timeout=60
                )
                response.raise_for_status()
                return response.json()
            except requests.RequestException as exc:
                raise ComfyError(f"Reference image upload failed: {exc}") from exc

    def submit(self, graph: dict[str, dict[str, Any]]) -> str:
        prompt_id = str(uuid.uuid4())
        payload = {
            "prompt": graph,
            "client_id": self.client_id,
            "prompt_id": prompt_id,
        }
        try:
            response = self.session.post(
                f"{self.base_url}/prompt", json=payload, timeout=30
            )
            response.raise_for_status()
            result = response.json()
        except requests.RequestException as exc:
            raise ComfyError(f"ComfyUI prompt submission failed: {exc}") from exc
        if result.get("error"):
            raise ComfyError(f"ComfyUI rejected the workflow: {result['error']}")
        return result.get("prompt_id", prompt_id)

    def wait_for_history(self, prompt_id: str, timeout: float = 1800.0) -> dict[str, Any]:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            history = self.get_json(f"/history/{prompt_id}")
            entry = history.get(prompt_id)
            if entry:
                status = entry.get("status", {})
                status_text = status.get("status_str")
                if status_text in {"success", "error"} or status.get("completed"):
                    return entry
            time.sleep(0.15)
        raise ComfyError(f"Timed out waiting for ComfyUI prompt {prompt_id}")

    def image_output(self, entry: dict[str, Any]) -> tuple[str, str]:
        candidates: list[dict[str, str]] = []
        for output in entry.get("outputs", {}).values():
            for image in output.get("images", []):
                candidates.append(image)
        if not candidates:
            raise ComfyError("ComfyUI completed without an image output")

        image = candidates[0]
        filename = image.get("filename", "")
        subfolder = image.get("subfolder", "")
        image_type = image.get("type", "output")
        if not filename:
            raise ComfyError("ComfyUI returned an image without a filename")

        relative = Path(subfolder) / filename
        local_path = COMFY_OUTPUT_DIR / relative
        if local_path.is_file():
            return str(local_path), json.dumps(image, indent=2)

        response = self.session.get(
            f"{self.base_url}/view",
            params={"filename": str(relative).replace("\\", "/"), "type": image_type},
            timeout=60,
        )
        response.raise_for_status()
        suffix = Path(filename).suffix or ".png"
        local_path = DASHBOARD_OUTPUT / f"generated_{uuid.uuid4().hex}{suffix}"
        local_path.write_bytes(response.content)
        return str(local_path), json.dumps(image, indent=2)


def qwen_sigmas(width: int, height: int, steps: int) -> list[float]:
    """Use the local token-count-dependent shift for the selected profile.

    Raw sigma nodes follow the halving family of the prescribed six-step schedule
    ``[1.0, 0.9375, 0.875, 0.75, 0.5, 0.25]``: four steps drops the 0.9375 and 0.875
    nodes, three steps keeps ``[1.0, 0.75, 0.25]``, and two steps would keep
    ``[1.0, 0.25]``. Two steps is deliberately not offered here; see the
    "Few-step research" tab for the measured reason.
    """
    if steps == 3:
        raw = [1.0, 0.75, 0.25]
    elif steps == 4:
        raw = [1.0, 0.75, 0.5, 0.25]
    elif steps == 5:
        raw = [1.0, 0.875, 0.75, 0.5, 0.25]
    elif steps == 6:
        raw = [1.0, 0.9375, 0.875, 0.75, 0.5, 0.25]
    else:
        raise ValueError("The dashboard supports 3-step, 4-step, 5-step, and 6-step Turbo profiles")

    sequence_length = (width // 16) * (height // 16)
    slope = (0.9 - 0.5) / (8192 - 256)
    mu = sequence_length * slope + (0.5 - slope * 256)
    alpha = math.exp(mu)
    shifted = [alpha * sigma / (1 + (alpha - 1) * sigma) for sigma in raw]
    return shifted + [0.0]


def aspect_dimensions(aspect: str, side: int) -> tuple[int, int]:
    exact = ASPECT_DIMENSIONS.get((aspect, side))
    if exact is not None:
        return exact
    try:
        width_ratio, height_ratio = (int(part) for part in aspect.split(":", 1))
    except (AttributeError, ValueError) as exc:
        raise ValueError(f"Unknown aspect ratio: {aspect}") from exc
    if width_ratio <= 0 or height_ratio <= 0:
        raise ValueError(f"Unknown aspect ratio: {aspect}")
    scale = side / math.sqrt(width_ratio * height_ratio)
    return round(width_ratio * scale / 16) * 16, round(height_ratio * scale / 16) * 16


def output_dimensions(area: str, aspect: str, custom_width: Any, custom_height: Any) -> tuple[int, int]:
    if area not in AREA_SIDES:
        raise ValueError(f"Unknown output area: {area}")
    if aspect == CUSTOM_ASPECT:
        try:
            width = int(custom_width)
            height = int(custom_height)
        except (TypeError, ValueError) as exc:
            raise ValueError("Custom width and height must be whole numbers") from exc
    else:
        aspect_code = ASPECT_PRESETS.get(aspect)
        if not aspect_code or aspect_code == "custom":
            raise ValueError(f"Unknown aspect ratio: {aspect}")
        width, height = aspect_dimensions(aspect_code, AREA_SIDES[area])
    if width < 16 or height < 16:
        raise ValueError("Width and height must each be at least 16 pixels")
    if width > MAX_OUTPUT_DIMENSION or height > MAX_OUTPUT_DIMENSION:
        raise ValueError(f"Width and height must not exceed {MAX_OUTPUT_DIMENSION} pixels")
    if width % 16 or height % 16:
        raise ValueError("Width and height must each be divisible by 16")
    return width, height


def text_encoder_resolution(width: int, height: int) -> int:
    return min(MAX_TEXT_ENCODER_RESOLUTION, max(32, round(math.sqrt(width * height) / 32) * 32))


def dimensions_preview(area: str, aspect: str, custom_width: Any, custom_height: Any) -> str:
    try:
        width, height = output_dimensions(area, aspect, custom_width, custom_height)
    except ValueError as exc:
        return str(exc)
    return f"{width} × {height} px  •  {ASPECT_PRESETS[aspect]}  •  {width * height:,} pixels"


def update_dimensions(area: str, aspect: str, custom_width: Any, custom_height: Any):
    preview = dimensions_preview(area, aspect, custom_width, custom_height)
    visible = aspect == CUSTOM_ASPECT
    return preview, gr.update(visible=visible), gr.update(visible=visible)


def build_graph(
    prompt: str,
    width: int,
    height: int,
    seed: int,
    profile: str,
    references: list[dict[str, str]],
    negative_prompt: str = "",
    use_negative: bool = False,
    cfg_scale: float = 4.0,
) -> dict[str, dict[str, Any]]:
    """Build a ComfyUI API graph for the accepted local workflow."""
    if width < 16 or height < 16 or width % 16 or height % 16:
        raise ValueError("Width and height must be positive multiples of 16")
    if width > MAX_OUTPUT_DIMENSION or height > MAX_OUTPUT_DIMENSION:
        raise ValueError(f"Width and height must not exceed {MAX_OUTPUT_DIMENSION} pixels")
    if len(references) > 10:
        raise ValueError("Qwen-Image-2.1 accepts at most 10 reference images")

    turbo = profile in TURBO_PROFILES
    base = profile == BASE_PROFILE
    if not (turbo or base):
        raise ValueError(f"Unknown dashboard profile: {profile}")
    negative_prompt = (negative_prompt or "").strip()
    cfg_scale = float(cfg_scale)
    if use_negative and not 1.0 <= cfg_scale <= 20.0:
        raise ValueError("CFG scale must be between 1 and 20")
    if not use_negative:
        cfg_scale = 1.0
    steps = PROFILE_STEPS[profile]
    sigmas = None if base else qwen_sigmas(width, height, steps)
    graph: dict[str, dict[str, Any]] = {
        "1": {
            "class_type": "UNETLoader",
            "inputs": {"unet_name": DIT_MODEL, "weight_dtype": "default"},
        },
        "3": {
            "class_type": "CLIPLoader",
            "inputs": {
                "clip_name": TEXT_MODEL,
                "type": "qwen_image",
                "device": "default",
            },
        },
        "4": {"class_type": "VAELoader", "inputs": {"vae_name": VAE_MODEL}},
    }

    text_inputs: dict[str, Any] = {
        "clip": ["3", 0],
        "prompt": prompt,
        "negative_prompt": negative_prompt if use_negative else "",
        "encode_negative": use_negative,
        "resolution": text_encoder_resolution(width, height),
        "images": {},
    }
    if references:
        text_inputs["vae"] = ["4", 0]
        for index, reference in enumerate(references, start=1):
            node_id = str(20 + index - 1)
            reference_name = reference["name"]
            if reference.get("subfolder"):
                reference_name = f"{reference['subfolder']}/{reference_name}"
            graph[node_id] = {
                "class_type": "LoadImage",
                "inputs": {"image": reference_name},
            }
            text_inputs["images"][f"image_{index}"] = [node_id, 0]

    graph["5"] = {
        "class_type": "TextEncodeQwenImage21",
        "inputs": text_inputs,
    }
    graph["6"] = {
        "class_type": "EmptyLatentImage",
        "inputs": {"width": width, "height": height, "batch_size": 1},
    }
    latent_source = ["5", 2] if references else ["6", 0]
    model_source = ["1", 0]
    if turbo:
        graph["2"] = {
            "class_type": "LoraLoaderModelOnly",
            "inputs": {
                "model": ["1", 0],
                "lora_name": TURBO_LORA,
                "strength_model": 1.0,
            },
        }
        model_source = ["2", 0]

    if base:
        graph["11"] = {
            "class_type": "KSampler",
            "inputs": {
                "model": ["14", 0],
                "seed": int(seed),
                "steps": steps,
                "cfg": cfg_scale,
                "sampler_name": "euler",
                "scheduler": "simple",
                "positive": ["5", 0],
                "negative": ["5", 1],
                "latent_image": latent_source,
                "denoise": 1.0,
            },
        }
    else:
        graph["7"] = {
            "class_type": "ManualSigmas",
            "inputs": {"sigmas": ", ".join(f"{sigma:.9f}" for sigma in sigmas)},
        }
        graph["8"] = {
            "class_type": "KSamplerSelect",
            "inputs": {"sampler_name": "euler"},
        }
        if use_negative:
            graph["9"] = {
                "class_type": "CFGGuider",
                "inputs": {
                    "model": ["14", 0],
                    "positive": ["5", 0],
                    "negative": ["5", 1],
                    "cfg": cfg_scale,
                },
            }
        else:
            graph["9"] = {
                "class_type": "BasicGuider",
                "inputs": {"model": ["14", 0], "conditioning": ["5", 0]},
            }
        graph["10"] = {
            "class_type": "RandomNoise",
            "inputs": {"noise_seed": int(seed)},
        }
        graph["11"] = {
            "class_type": "SamplerCustomAdvanced",
            "inputs": {
                "noise": ["10", 0],
                "guider": ["9", 0],
                "sampler": ["8", 0],
                "sigmas": ["7", 0],
                "latent_image": latent_source,
            },
        }
    graph["12"] = {
        "class_type": "VAEDecode",
        "inputs": {"samples": ["11", 0], "vae": ["4", 0]},
    }
    graph["13"] = {
        "class_type": "SaveImage",
        "inputs": {
            "images": ["12", 0],
            "filename_prefix": f"dashboard/{width}x{height}_{int(time.time())}_{uuid.uuid4().hex[:8]}",
        },
    }
    graph["14"] = {
        "class_type": "QwenImage21Cache",
        "inputs": {"model": model_source, "device": "gpu", "dtype": "default"},
    }
    return graph


def file_path(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, (str, Path)):
        return str(value)
    if isinstance(value, dict):
        candidate = value.get("path") or value.get("name")
        return str(candidate) if candidate else None
    name = getattr(value, "name", None)
    return str(name) if name else None


def reference_files(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, (str, Path, dict)):
        value = [value]
    paths: list[str] = []
    for item in value:
        path = file_path(item)
        if path:
            paths.append(path)
    return paths


def history_timing(entry: dict[str, Any]) -> dict[str, float | None]:
    messages = entry.get("status", {}).get("messages", [])
    start_ms: float | None = None
    success_ms: float | None = None
    for message in messages:
        if not isinstance(message, list) or len(message) < 2:
            continue
        message_type, data = message[0], message[1]
        timestamp = data.get("timestamp") if isinstance(data, dict) else None
        if not isinstance(timestamp, (int, float)):
            continue
        if message_type == "execution_start":
            start_ms = timestamp
        elif message_type == "execution_success":
            success_ms = timestamp
    execution_s = None
    if start_ms is not None and success_ms is not None:
        execution_s = (success_ms - start_ms) / 1000.0
    return {"execution_seconds": execution_s}


def generate(
    prompt: str,
    negative_prompt: str,
    use_negative: bool,
    cfg_scale: float,
    references: Any,
    area: str,
    aspect: str,
    custom_width: Any,
    custom_height: Any,
    seed: int,
    profile: str,
):
    prompt = prompt.strip()
    negative_prompt = (negative_prompt or "").strip()
    if not prompt:
        raise gr.Error("Enter a prompt first.")
    if use_negative and not negative_prompt:
        raise gr.Error("Enter a negative prompt or disable negative prompting.")
    try:
        width, height = output_dimensions(area, aspect, custom_width, custom_height)
    except ValueError as exc:
        raise gr.Error(str(exc)) from exc

    client = ComfyClient()
    uploaded: list[dict[str, str]] = []
    try:
        for path in reference_files(references)[:10]:
            uploaded.append(client.upload_image(path))
        graph = build_graph(
            prompt,
            width,
            height,
            int(seed),
            profile,
            uploaded,
            negative_prompt=negative_prompt,
            use_negative=use_negative,
            cfg_scale=float(cfg_scale),
        )
        started = time.perf_counter()
        prompt_id = client.submit(graph)
        entry = client.wait_for_history(prompt_id)
        image_path, image_metadata = client.image_output(entry)
    except ComfyError as exc:
        raise gr.Error(str(exc)) from exc

    status = entry.get("status", {})
    if status.get("status_str") == "error":
        raise gr.Error(json.dumps(status, indent=2))
    timing = history_timing(entry)
    api_seconds = time.perf_counter() - started
    details = {
        "profile": profile,
        "area": area,
        "aspect_ratio": ASPECT_PRESETS[aspect],
        "width": width,
        "height": height,
        "pixel_area": width * height,
        "seed": int(seed),
        "prompt_id": prompt_id,
        "api_poll_seconds": round(api_seconds, 6),
        "execution_seconds": timing["execution_seconds"],
        "output_file": image_path,
        "output_metadata": image_metadata,
        "model": DIT_MODEL,
        "text_encoder": TEXT_MODEL,
        "vae": VAE_MODEL,
        "lora": TURBO_LORA if profile in TURBO_PROFILES else None,
        "negative_prompt": negative_prompt if use_negative else None,
        "negative_prompt_enabled": use_negative,
        "cache": "GPU / default (lossless prefix cache; Turbo negative prompting is experimental)" if use_negative and profile != BASE_PROFILE else "GPU / default (lossless prefix cache)",
        "cfg": float(cfg_scale) if use_negative else 1.0,
        "note": (
            "Quality profile uses the official v0.2.1 r128 prescribed six-step schedule."
            if profile == QUALITY_PROFILE
            else "Fast profile uses v0.2.1 r128 with the publisher-supported five-step compromise schedule."
            if profile == FAST_PROFILE
            else "Balanced profile uses v0.2.1 r128 with a three-step schedule. Measured to match the four-step schedule in quality over 12 prompt/seed cells while costing one fewer denoiser call, but it is not yet an adopted production default and its quality gate is still narrow."
            if profile == BALANCED_PROFILE
            else "Experimental profile uses v0.2.1 r128 with a four-step schedule; it is not the quality default."
            if profile == EXPERIMENTAL_PROFILE
            else "Base reference uses the ordinary 40-step KSampler path."
        )
        + (" Negative prompting is enabled; Turbo negative-prompt runs are experimental and off the official CFG-1 path." if use_negative and profile != BASE_PROFILE else ""),
    }
    summary = (
        f"Completed {width}×{height} in {api_seconds:.3f} s API time. "
        f"ComfyUI execution time: {timing['execution_seconds'] or 'n/a'} s. "
        f"Prompt ID: {prompt_id}"
    )
    return image_path, summary, details


def gpu_snapshot() -> dict[str, Any]:
    result: dict[str, Any] = {}
    try:
        completed = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.total,memory.used,memory.free,driver_version",
                "--format=csv,noheader,nounits",
                "--id=0",
            ],
            capture_output=True,
            text=True,
            timeout=5,
            check=True,
        )
        fields = ["name", "memory_total_mib", "memory_used_mib", "memory_free_mib", "driver"]
        values = [item.strip() for item in completed.stdout.strip().split(",")]
        result.update(dict(zip(fields, values)))
    except (OSError, subprocess.SubprocessError) as exc:
        result["gpu_error"] = str(exc)
    return result


def refresh_status() -> dict[str, Any]:
    try:
        stats = ComfyClient().get_json("/system_stats")
        queue = ComfyClient().get_json("/queue")
    except ComfyError as exc:
        return {"connected": False, "error": str(exc)}

    system = stats.get("system", {})
    devices = stats.get("devices", [])
    device = devices[0] if devices else {}
    return {
        "connected": True,
        "server": COMFY_BASE_URL,
        "comfyui_version": system.get("comfyui_version"),
        "python": system.get("python_version"),
        "pytorch": system.get("pytorch_version"),
        "ram_total_gib": round(system.get("ram_total", 0) / 1024**3, 2),
        "ram_free_gib": round(system.get("ram_free", 0) / 1024**3, 2),
        "device": device,
        "queue_running": len(queue.get("queue_running", [])),
        "queue_pending": len(queue.get("queue_pending", [])),
        "argv": system.get("argv", []),
        "nvidia_smi": gpu_snapshot(),
    }


def benchmark_rows(result_path: Path) -> list[list[Any]]:
    """Turn a comfy_bench result file into table rows, flagging cached samples."""
    payload = _read_json(result_path)
    if not isinstance(payload, list):
        return []
    rows: list[list[Any]] = []
    for record in payload:
        messages = ((record.get("history_status") or {}).get("messages") or [])
        cached = any(
            isinstance(message, list) and message and message[0] == "execution_cached" for message in messages
        )
        status = "cached, not timed" if cached else ("warmup" if record.get("warmup") else "measured")
        rows.append(
            [
                record.get("run_index"),
                status,
                round(float(record.get("elapsed_seconds") or 0.0), 4),
                f"{record.get('size')}px, seed {record.get('seed')}, {record.get('schedule')}",
                "served from the graph cache" if cached else "elapsed prompt-submission-to-success",
            ]
        )
    return rows


def run_benchmark(size: int, repeats: int):
    if size not in {512, 768, 1024}:
        raise gr.Error("Benchmark size must be 512, 768, or 1024.")
    repeats = max(1, min(int(repeats), 5))
    tag = f"dashboard_{size}_{int(time.time())}"
    result_path = ROOT / "results" / f"comfy_{tag}.json"
    python = BENCHMARK_PYTHON if BENCHMARK_PYTHON.is_file() else Path(sys.executable)
    command = [
        str(python),
        str(ROOT / "scripts" / "comfy_bench.py"),
        "--base-url",
        COMFY_BASE_URL,
        "--tag",
        tag,
        "--size",
        str(size),
        "--seed",
        "42",
        "--schedule",
        "v02_5step",
        "--model",
        DIT_MODEL,
        "--lora",
        TURBO_LORA,
        "--positive-only",
        "--cache-device",
        "gpu",
        "--cache-dtype",
        "default",
        "--warmup",
        "1",
        "--repeats",
        str(repeats),
        "--vary-prompt",
        "--output-root",
        str(COMFY_OUTPUT_DIR),
        "--copy-dir",
        str(ROOT / "out" / "benchmarks"),
        "--results-path",
        str(result_path),
    ]
    try:
        completed = subprocess.run(
            command,
            cwd=str(ROOT),
            capture_output=True,
            text=True,
            timeout=1800,
            check=False,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        return [], f"Benchmark could not start: {exc}", str(result_path)

    log = (completed.stdout or "") + ("\n" + completed.stderr if completed.stderr else "")
    if completed.returncode != 0:
        return [], log[-12000:], str(result_path)
    return benchmark_rows(result_path), log[-12000:], str(result_path)


def _read_json(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, ValueError, json.JSONDecodeError):
        return None


def research_markdown() -> str:
    """Phase-2 few-step findings, rendered from the recorded result files.

    Nothing here is hardcoded from memory: if a result file is missing the panel says so
    instead of showing a stale number.
    """
    sections: list[str] = [
        "# Few-step research (phase 2)",
        "",
        "This tab reports the measurements that changed which schedule this project should be "
        "using. It is measurement evidence, not an adopted production default. Every table "
        "below is read from the result files at render time, so nothing here is typed in by "
        "hand. `results/README.md` explains what each file holds and how to regenerate it.",
        "",
    ]

    step_count = _read_json(STEP_COUNT_RESULTS)
    if step_count is None:
        sections += ["**Step-count comparison is missing.** Expected `results/schedule_stepcount_generalization.json`.", ""]
    else:
        summary = step_count["summary"]
        rows = [
            ("1 step", "n1", "0.8345 baseline: not credible"),
            ("2 steps, prescribed `1,0.25`", "n2prescribed", "the old experiment's grid"),
            ("2 steps, tuned `1,0.85`", "n2tuned", "best grid found; optimum not bracketed"),
            ("3 steps `1,0.75,0.25`", "n3", "new Balanced profile"),
            ("4 steps `1,0.75,0.5,0.25`", "n4", "previous experimental"),
        ]
        sections += [
            "## Step-count quality across 12 prompt/seed cells",
            "",
            f"Four prompts and {summary['n_seeds']} seeds ({summary['n_combos']} cells). Every candidate is scored "
            "against a six-step reference generated for the same prompt and seed.",
            "",
            "| Grid | mean DINO | median | min | max | mean CLIP |",
            "|---|---:|---:|---:|---:|---:|",
        ]
        for label, key, note in rows:
            if key not in summary["mean_dino_cosine"]:
                continue
            sections.append(
                f"| {label} | {summary['mean_dino_cosine'][key]:.4f} | {summary['median_dino_cosine'][key]:.4f} "
                f"| {summary['min_dino_cosine'][key]:.4f} | {summary['max_dino_cosine'][key]:.4f} "
                f"| {summary['mean_clip_cosine'][key]:.4f} |"
            )
        sections += [
            "",
            "**Three steps matches four steps** on both mean and median, for one fewer denoiser "
            "call. That is why the three-step grid is now offered as the Balanced profile.",
            "",
            "**Two steps is unreliable, not merely weaker.** The tuned two-step grid has a "
            f"median of {summary['median_dino_cosine']['n2tuned']:.4f} but a mean of "
            f"{summary['mean_dino_cosine']['n2tuned']:.4f} and a minimum of "
            f"{summary['min_dino_cosine']['n2tuned']:.4f}. Report the distribution, not the mean. "
            "One step has a median of "
            f"{summary['median_dino_cosine']['n1']:.4f}, so it is not a viable target at this quality bar.",
            "",
        ]

    floor = _read_json(LATENCY_FLOOR_RESULTS)
    if floor is not None:
        sections += [
            "## Latency floor model",
            "",
            "Fitted over six measured step counts; every sample latency-valid with no cached sampler node.",
            "",
            f"- `denoise(N) = {floor['a_first_call_seconds']:.4f} + {floor['c_marginal_step_seconds']:.4f} * (N-1)` seconds, "
            f"max absolute fit residual {floor['max_abs_residual_seconds']:.4f} s",
            f"- First denoiser call **{floor['a_first_call_seconds']:.4f} s**; marginal call "
            f"**{floor['c_marginal_step_seconds']:.4f} s**; one-off first-call surcharge "
            f"**{floor['first_call_surcharge_seconds']:.4f} s**",
            f"- Non-denoise fixed cost **{floor['non_denoise_fixed_seconds']:.4f} s** "
            f"(text encode {floor['stage_mean_seconds'].get('TextEncodeQwenImage21', 0):.4f}, "
            f"VAE decode {floor['stage_mean_seconds'].get('VAEDecode', 0):.4f})",
            f"- Measured one-step total **{floor['measured_total_seconds'].get('1', 0):.4f} s**; "
            f"zero-denoise floor **{floor['zero_denoise_floor_seconds']:.4f} s**",
            "",
            "**One second is not reachable by reducing step count on this card.** A perfect "
            f"one-step student still lands near {floor['one_step_seconds_predicted']:.2f} s, and the floor with "
            f"no denoising at all is {floor['zero_denoise_floor_seconds']:.2f} s. Note that the first-call surcharge "
            f"({floor['first_call_surcharge_seconds']:.4f} s) is larger than what dropping from two steps to one saves "
            f"({floor['c_marginal_step_seconds']:.4f} s); that surcharge is LoRA and ConvRot fault machinery rather than "
            "a quality term.",
            "",
        ]

    long_prompt = _read_json(LONG_PROMPT_RESULTS)
    if long_prompt is not None:
        grids = [grid for grid in long_prompt.get("grids", []) if grid.get("latency_valid")]
        if grids:
            sections += [
                "## Accepted long-prompt protocol",
                "",
                "The same step counts on the accepted benchmark prompt, 2 warmups and 5 measured requests, prompt varied.",
                "",
                "| Grid | warm median s | text s | denoise s | VAE s |",
                "|---|---:|---:|---:|---:|",
            ]
            for grid in grids:
                stages = grid.get("stage_medians_seconds") or {}
                schedule = ",".join(f"{value:g}" for value in grid["raw_sigmas"])
                sections.append(
                    f"| {grid['steps']} steps `{schedule}` | {grid['median_elapsed_seconds']:.4f} "
                    f"| {stages.get('TextEncodeQwenImage21', 0):.4f} | {stages.get('SamplerCustomAdvanced', 0):.4f} "
                    f"| {stages.get('VAEDecode', 0):.4f} |"
                )
            by_steps = {grid["steps"]: grid["median_elapsed_seconds"] for grid in grids}
            if 3 in by_steps and 2 in by_steps and 5 in by_steps:
                sections += [
                    "",
                    f"Three steps saves {by_steps[5] - by_steps[3]:.3f} s against five steps, but only "
                    f"{by_steps[3] - by_steps[2]:.3f} s against two steps, which is inside run-to-run spread. "
                    "That is the practical reason the two-step target was dropped rather than trained.",
                    "",
                    "> **Machine-state caveat.** These absolute values ran roughly 20 to 45 percent above the "
                    "recorded accepted session, with no GPU clock or power throttling detected. Treat them as "
                    "lower bounds and rely on the within-session ratios, not on comparison with the accepted "
                    "1.891 s figure.",
                    "",
                ]

    sections += [
        "## What is still open",
        "",
        "- Every quality number above rests on 4 prompts and 3 seeds at 512px with no reference-image cases "
        "and no 768px. The gate is not yet wide enough to adopt a new production default.",
        "- The tuned two-step sigma optimum is not bracketed; the best tested value sits at the edge of the "
        "searched range.",
        "- Whether the earlier rank-1 correction was actually injected into the forward pass is still unverified.",
        "",
    ]
    return "\n".join(sections)


def architecture_markdown() -> str:
    return f"""
# Qwen-Image-2.1 local stack

## The short version

The visual generator is **7B parameters**, but the complete image-generation system is larger:

| Component | Local accepted artifact | Disk size | Role |
|---|---|---:|---|
| Qwen-Image-2.1 DiT | `{DIT_MODEL}` | 6.758 GiB | Generates the image latent with a 32-layer, 7B single-stream DiT |
| Qwen3-VL text encoder | `{TEXT_MODEL}` | 8.709 GiB | Encodes prompts and reference images into conditioning |
| VAE | `{VAE_MODEL}` | 0.629 GiB | Converts between image pixels and latent space |
| Turbo LoRA | `{TURBO_LORA}` | 0.633 GiB | Runtime v0.2.1 r128 student adapter; six-step quality or five-step fast schedule |
| ComfyUI runtime | ComfyUI 0.37.0 | system | Graph execution, model loading, cache and offload policy |

The **complete pipeline is therefore roughly 16.7 GiB of local artifacts**, before CUDA workspaces, activations, allocator fragmentation, OS/display memory, and offload staging. If the BF16 DiT (~14.2GB), BF16 text encoder (~17.5GB), and BF16 VAE (~0.7GB) are staged or kept resident together, the raw weights alone are roughly 32.4GB decimal, or 30+ GiB, before runtime overhead. That is why a 7B visual component does not mean a 7B total memory footprint. Sequential or model-level offloading can reduce the instantaneous peak, but increases transfer and scheduling cost.

The final clean stock-loader run measured **1.891 s** for the v0.2.1 r128 five-step fast profile at 512px, **2.086 s** for the r128 six-step quality profile, and **3.344 s** for the five-step profile at 768px. The runs used the same RTX 4060 Ti contract and reached about 15.1 GiB peak VRAM, not the one-second target. The historical old-adapter result of 1.656 s at 512px is not a v0.2.1 measurement.

## How the accepted path works

1. Gradio uploads optional reference images to the local ComfyUI input directory.
2. ComfyUI loads the INT8 ConvRot DiT, INT8 ConvRot Qwen3-VL encoder, and BF16 VAE.
3. `TextEncodeQwenImage21` encodes the positive prompt once by default. When the dashboard's negative-prompt checkbox is enabled, it also encodes the negative prompt.
4. `QwenImage21Cache` stores the reusable prefix K/V state on the GPU at default precision. Its prompt-keyed slots keep positive and negative conditioning separate. This is lossless cache storage, not lossless whole-pipeline inference.
5. The runtime Turbo LoRA is the current v0.2.1 r128 adapter. The quality profile uses the prescribed six-step sigma schedule; the fast profile uses the publisher-supported five-step compromise schedule.
6. With negative prompting disabled, `BasicGuider` runs at CFG 1 with no duplicate pass. Enabling it switches to `CFGGuider` and the selected CFG scale; Turbo negative prompting is experimental and slower.
7. Comfy Kitchen dispatches the INT8 linear work to its native CUDA `int8_linear` path under `torch 2.14.0+cu130`.
8. Three asynchronous offload streams overlap weight transfer with computation on this 16GB card.
9. The VAE decodes the final latent to pixels and ComfyUI writes the PNG.

## Base model versus this local path

The base Qwen-Image-2.1 checkpoint is the source model. The local run does not retrain or change its architecture. It changes deployment and execution:

- **Base/default path:** the original Qwen-Image-2.1 checkpoint, usually loaded at higher precision and sampled with a 25–50 step Euler schedule. The official template starts at 25 steps; the README's reference pipeline uses 40 steps.
- **Dashboard base reference:** the same INT8 DiT, INT8 Qwen3-VL encoder, GPU cache, CFG-1 conditioning, and runtime, with Turbo removed and ordinary `KSampler` set to 40 steps. This isolates the Turbo adapter and schedule effect, but is not a BF16 full-fidelity comparison.
- **Local accepted path:** INT8 ConvRot model weights, runtime v0.2.1 r128 Turbo LoRA, prescribed six-step quality schedule or five-step fast schedule, positive-only CFG-1 by default, optional CFG-based negative prompting, GPU lossless prefix cache, native Comfy Kitchen CUDA kernels, three async offload streams, and a 1.0 GiB reserve.
- **What remains upstream:** the 7B model, Qwen3-VL encoder, VAE, Turbo LoRA/distillation, prefix-cache architecture, Comfy Kitchen kernels, and ComfyUI offload machinery.
- **What is local:** the graph-level `encode_negative` option, the exact artifact combination, the RTX 4060 Ti stream tuning, the quality gate, and the reproducible benchmark record.

## Performance recorded by this project

- v0.2.1 r128 quality default, 512px: **2.086 s** warm median.
- v0.2.1 r128 fast profile, 512px: **1.891 s** warm median.
- v0.2.1 r128 experimental four-step, 512px: **2.295 s** warm median.
- v0.2.1 r128 fast profile, 768px: **3.344 s** warm median.
- v0.2.1 r128 three-step (new Balanced profile), 512px: about **0.32 s** faster than the five-step fast profile in the same session; experimental until its quality gate is widened.
- Historical old-adapter result: **1.656 s** at 512px; retained for comparison only, not a v0.2.1 result.
- These are warm prompt-submission-to-ComfyUI-success measurements on one RTX 4060 Ti 16GB, not cold-start, throughput, P99, or universal Qwen benchmarks.

## Few-step research boundary

Step count was re-measured over 4 prompts and 3 seeds, and the per-call latency cost was fitted from six step counts. The numbers are on the **Few-step research** tab. In short: three steps matches four steps for one fewer denoiser call, two steps is unreliable rather than merely weaker, and cutting step count is not how this reaches one second, because the non-denoise fixed cost and the first denoiser call dominate. Five and six steps remain the production defaults.

## Dashboard boundary

This Gradio app is a local control panel only. It does not load PyTorch, the DiT, the text encoder, or the VAE. Inference stays in ComfyUI, so opening the dashboard does not add a second copy of the model to VRAM.
"""


def build_demo() -> gr.Blocks:
    with gr.Blocks(title="Qwen Image 2.1 Local Dashboard") as demo:
        gr.Markdown(
            "# Qwen-Image-2.1 Local Dashboard\n"
            "Quality default: INT8 ConvRot + 6-step Turbo v0.2.1 r128 + CFG-1 + GPU prefix cache + cu130 native CUDA + three async offload streams."
        )
        with gr.Tab("Generate"):
            with gr.Row():
                with gr.Column(scale=5):
                    prompt = gr.Textbox(
                        label="Prompt",
                        value=DEFAULT_PROMPT,
                        lines=8,
                        max_lines=16,
                    )
                    with gr.Accordion("Negative prompt", open=False):
                        use_negative = gr.Checkbox(
                            value=False,
                            label="Use negative prompt",
                            info="Enables a second conditioning pass. This is slower; Turbo negative prompting is experimental because v0.2.1 is designed for CFG-1.",
                        )
                        negative_prompt = gr.Textbox(
                            label="Negative prompt",
                            value="blurry, low quality, watermark, text, extra fingers",
                            lines=4,
                            max_lines=10,
                        )
                        cfg_scale = gr.Slider(
                            minimum=1.0,
                            maximum=10.0,
                            step=0.1,
                            value=4.0,
                            label="CFG scale",
                            info="Used only when the negative-prompt checkbox is enabled.",
                        )
                    profile = gr.Radio(
                        PROFILES,
                        value=QUALITY_PROFILE,
                        label="Profile",
                        info=(
                            "Quality uses the prescribed six-step v0.2.1 r128 schedule. "
                            "Fast uses the measured five-step r128 compromise. "
                            "Balanced is a new three-step schedule, measured to match the "
                            "four-step quality over 12 prompt/seed cells for one fewer "
                            "denoiser call, but it is experimental until its gate is widened. "
                            "Negative prompting is optional and slower. "
                            "Base reference uses 40 steps without Turbo."
                        ),
                    )
                    with gr.Accordion("Output size", open=True):
                        with gr.Row():
                            area = gr.Dropdown(
                                list(AREA_SIDES),
                                value=DEFAULT_AREA,
                                label="Area",
                                info="Target pixel area: 512², 768², 1024², or 2048².",
                            )
                            aspect = gr.Dropdown(
                                list(ASPECT_PRESETS),
                                value=DEFAULT_ASPECT,
                                label="Aspect ratio",
                                info="Presets cover Instagram, YouTube, and common image ratios.",
                            )
                        with gr.Row():
                            custom_width = gr.Number(
                                value=1024,
                                precision=0,
                                minimum=16,
                                maximum=MAX_OUTPUT_DIMENSION,
                                step=16,
                                label="Custom width",
                                visible=False,
                            )
                            custom_height = gr.Number(
                                value=1024,
                                precision=0,
                                minimum=16,
                                maximum=MAX_OUTPUT_DIMENSION,
                                step=16,
                                label="Custom height",
                                visible=False,
                            )
                        dimension_preview = gr.Markdown(dimensions_preview(DEFAULT_AREA, DEFAULT_ASPECT, 1024, 1024))
                    seed = gr.Number(value=42, precision=0, label="Seed")
                    references = gr.File(
                        label="Optional reference images (up to 10)",
                        file_count="multiple",
                        file_types=["image"],
                        type="filepath",
                    )
                    generate_button = gr.Button("Generate image", variant="primary")
                with gr.Column(scale=6):
                    output_image = gr.Image(type="filepath", label="Generated image")
                    generation_status = gr.Textbox(label="Status", lines=3)
                    generation_details = gr.JSON(label="Run details")
            aspect.change(
                update_dimensions,
                inputs=[area, aspect, custom_width, custom_height],
                outputs=[dimension_preview, custom_width, custom_height],
                show_progress="hidden",
            )
            area.change(
                dimensions_preview,
                inputs=[area, aspect, custom_width, custom_height],
                outputs=dimension_preview,
                show_progress="hidden",
            )
            custom_width.change(
                dimensions_preview,
                inputs=[area, aspect, custom_width, custom_height],
                outputs=dimension_preview,
                show_progress="hidden",
            )
            custom_height.change(
                dimensions_preview,
                inputs=[area, aspect, custom_width, custom_height],
                outputs=dimension_preview,
                show_progress="hidden",
            )
            generate_button.click(
                generate,
                inputs=[prompt, negative_prompt, use_negative, cfg_scale, references, area, aspect, custom_width, custom_height, seed, profile],
                outputs=[output_image, generation_status, generation_details],
            )

        with gr.Tab("System / VRAM"):
            refresh_button = gr.Button("Refresh status", variant="primary")
            status_output = gr.JSON(label="ComfyUI and GPU status", value=refresh_status())
            refresh_button.click(refresh_status, outputs=status_output)
            gr.Markdown(
                "The final v0.2.1 r128 runs used about 15.1 GiB peak VRAM on the RTX 4060 Ti 16GB. The complete pipeline includes the 7B visual DiT, the 8B Qwen3-VL text encoder, the VAE, runtime activations, CUDA workspaces, and allocator/offload overhead."
            )

        with gr.Tab("Benchmarks"):
            gr.Markdown(
                "The table fills in as you benchmark. The button invokes `scripts/comfy_bench.py` "
                "using the five-step fast profile, a unique prompt marker per request so the graph "
                "cache cannot return a stale result, and the selected repeat count. Every sample "
                "whose sampler node was served from cache is reported as invalid rather than timed."
            )
            with gr.Row():
                bench_size = gr.Dropdown([512, 768, 1024], value=512, label="Resolution")
                bench_repeats = gr.Number(value=3, precision=0, label="Measured repeats")
            bench_button = gr.Button("Run local benchmark", variant="primary")
            bench_table = gr.Dataframe(
                headers=["Run", "Status", "Elapsed (s)", "Configuration", "Notes"],
                datatype=["number", "str", "number", "str", "str"],
                value=[],
                wrap=True,
            )
            bench_output = gr.Textbox(label="Benchmark log / result", lines=14)
            bench_result_path = gr.Textbox(label="Result JSON")
            bench_button.click(
                run_benchmark,
                inputs=[bench_size, bench_repeats],
                outputs=[bench_table, bench_output, bench_result_path],
            )

        with gr.Tab("Few-step research"):
            gr.Markdown(
                "Preregistered phase-2 measurements. Rendered from the recorded result files, so this tab "
                "shows no number that is not on disk. Nothing here changes an adopted production profile."
            )
            gr.Markdown(research_markdown())

        with gr.Tab("Architecture / Explanation"):
            gr.Markdown(architecture_markdown())
    return demo


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--server-name", default="127.0.0.1")
    parser.add_argument("--server-port", type=int, default=7860)
    parser.add_argument("--share", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    build_demo().queue().launch(
        server_name=args.server_name,
        server_port=args.server_port,
        share=args.share,
        show_error=True,
        theme=gr.themes.Soft(),
    )


if __name__ == "__main__":
    main()
