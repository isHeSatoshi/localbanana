import argparse
import asyncio
import json
import math
import shutil
import subprocess
import threading
import time
import uuid
from pathlib import Path

import aiohttp

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RESULTS = ROOT / "results"
DEFAULT_OUTPUT = ROOT / "out" / "comfy"


def qwen_sigmas(size: int, schedule: str, raw_sigmas: list[float] | None = None) -> list[float]:
    if raw_sigmas:
        raw = [float(value) for value in raw_sigmas]
    elif schedule == "v02_6step":
        raw = [1.0, 0.9375, 0.875, 0.75, 0.5, 0.25]
    elif schedule == "v02_5step":
        raw = [1.0, 0.875, 0.75, 0.5, 0.25]
    elif schedule == "v02_4step":
        raw = [1.0, 0.75, 0.5, 0.25]
    elif schedule == "candidate_2step":
        raw = [1.0, 0.5]
    elif schedule == "candidate_1step":
        raw = [1.0]
    else:
        raise ValueError(f"Unknown schedule: {schedule}")
    if len(raw) < 1 or raw[0] != 1.0 or any(value <= 0 or value >= 1 for value in raw[1:]):
        raise ValueError("Raw sigmas must start at 1.0 and remaining values must be in (0, 1)")

    sequence_length = (size // 16) ** 2
    slope = (0.9 - 0.5) / (8192 - 256)
    mu = sequence_length * slope + (0.5 - slope * 256)
    alpha = math.exp(mu)
    shifted = [alpha * sigma / (1 + (alpha - 1) * sigma) for sigma in raw]
    return shifted + [0.0]


def parse_csv_row(line: str) -> list[float]:
    return [float(value.strip()) for value in line.split(",")]


class NvidiaSampler:
    fields = [
        "memory.used",
        "memory.free",
        "utilization.gpu",
        "power.draw",
        "temperature.gpu",
        "clocks.current.sm",
    ]

    def __init__(self, interval: float):
        self.interval = interval
        self.samples = []
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self._run, daemon=True)

    @staticmethod
    def sample() -> dict:
        command = [
            "nvidia-smi",
            f"--query-gpu={','.join(NvidiaSampler.fields)}",
            "--format=csv,noheader,nounits",
            "--id=0",
        ]
        completed = subprocess.run(command, capture_output=True, text=True, timeout=5, check=True)
        values = parse_csv_row(completed.stdout.strip())
        return {"time": time.time(), **dict(zip(NvidiaSampler.fields, values))}

    def _run(self):
        while not self.stop_event.is_set():
            try:
                self.samples.append(self.sample())
            except Exception as exc:
                self.samples.append({"time": time.time(), "error": str(exc)})
            self.stop_event.wait(self.interval)

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, exc_type, exc, tb):
        self.stop_event.set()
        self.thread.join(timeout=5)


async def wait_for_server(session: aiohttp.ClientSession, base_url: str, timeout: float = 120):
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            async with session.get(f"{base_url}/system_stats") as response:
                if response.status == 200:
                    return await response.json()
        except aiohttp.ClientError:
            pass
        await asyncio.sleep(1)
    raise TimeoutError(f"ComfyUI did not become ready at {base_url}")


def build_prompt(
    args,
    prompt: str,
    sigmas: list[float],
) -> dict:
    sigmas_text = ", ".join(f"{sigma:.9f}" for sigma in sigmas)
    model_names = [args.model]
    if args.lora:
        model_names.append(args.lora)
    common = {
        "1": {
            "class_type": "UnetLoaderGGUF" if args.dit_loader == "gguf" else "UNETLoader",
            "inputs": {
                "unet_name": args.model,
                **({} if args.dit_loader == "gguf" else {"weight_dtype": "default"}),
            },
        },
        "3": {
            "class_type": "CLIPLoader",
            "inputs": {
                "clip_name": args.text_encoder,
                "type": "qwen_image",
                "device": args.text_encoder_device,
            },
        },
        "4": {
            "class_type": "VAELoader",
            "inputs": {"vae_name": args.vae},
        },
        "5": {
            "class_type": "TextEncodeQwenImage21",
            "inputs": {
                "clip": ["3", 0],
                "prompt": prompt,
                "negative_prompt": "",
                "encode_negative": not args.positive_only,
                "resolution": args.size,
                "images": {},
            },
        },
        "6": {
            "class_type": "EmptyLatentImage",
            "inputs": {
                "width": args.size,
                "height": args.size,
                "batch_size": 1,
            },
        },
        "7": {
            "class_type": "ManualSigmas",
            "inputs": {"sigmas": sigmas_text},
        },
        "8": {
            "class_type": "KSamplerSelect",
            "inputs": {"sampler_name": "euler"},
        },
        "9": ({
            "class_type": "BasicGuider",
            "inputs": {
                "model": ["2", 0] if args.lora else ["1", 0],
                "conditioning": ["5", 0],
            },
        } if args.positive_only else {
            "class_type": "CFGGuider",
            "inputs": {
                "model": ["2", 0] if args.lora else ["1", 0],
                "positive": ["5", 0],
                "negative": ["5", 1],
                "cfg": 1.0,
            },
        }),
        "10": {
            "class_type": "RandomNoise",
            "inputs": {"noise_seed": args.seed},
        },
        "11": {
            "class_type": "SamplerCustomAdvanced",
            "inputs": {
                "noise": ["10", 0],
                "guider": ["9", 0],
                "sampler": ["8", 0],
                "sigmas": ["7", 0],
                "latent_image": ["6", 0],
            },
        },
        "12": {
            "class_type": "VAEDecode",
            "inputs": {
                "samples": ["11", 0],
                "vae": ["4", 0],
            },
        },
        "13": {
            "class_type": "SaveImage",
            "inputs": {
                "images": ["12", 0],
                "filename_prefix": args.filename_prefix,
            },
        },
    }

    if args.lora:
        lora_class = "LoraLoaderModelOnly"
        lora_inputs = {
            "model": ["1", 0],
            "lora_name": args.lora,
            "strength_model": args.strength,
        }
        if args.lora_bypass:
            lora_class = "LoraLoaderBypassModelOnly"
        elif args.unmerged_lora:
            lora_class = "TurboLora"
            lora_inputs = {
                "model": ["1", 0],
                "lora_name": args.lora,
                "strength": args.strength,
            }
        common["2"] = {
            "class_type": lora_class,
            "inputs": lora_inputs,
        }

    model_source = ["2", 0] if args.lora else ["1", 0]
    if args.attention_backend != "default":
        common["15"] = {
            "class_type": "ModelAttentionBackend",
            "inputs": {
                "model": model_source,
                "attention": args.attention_backend,
            },
        }
        model_source = ["15", 0]

    if args.cache_device != "bypass":
        common["14"] = {
            "class_type": "QwenImage21Cache",
            "inputs": {
                "model": model_source,
                "device": args.cache_device,
                "dtype": args.cache_dtype,
            },
        }
        common["9"]["inputs"]["model"] = ["14", 0]

    return common


async def run_once(session, args, prompt: str, output_root: Path) -> dict:
    sigmas = qwen_sigmas(args.size, args.schedule, args.raw_sigmas)
    graph = build_prompt(args, prompt, sigmas)
    prompt_id = str(uuid.uuid4())
    events = []
    node_states = {}
    node_transitions = []
    sampler_progress = []
    done = asyncio.get_running_loop().create_future()
    websocket = None
    client_id = None

    telemetry = NvidiaSampler(args.telemetry_interval)
    with telemetry:
        async with session.ws_connect(f"{args.base_url}/ws", heartbeat=20) as ws:
            initial = await ws.receive_json(timeout=10)
            if initial.get("type") != "status":
                raise RuntimeError(f"Unexpected initial WebSocket message: {initial}")
            client_id = initial["data"]["sid"]
            await ws.send_json({"type": "feature_flags", "data": {"supports_preview_metadata": True}})

            async def receive_events():
                nonlocal websocket
                websocket = ws
                async for message in ws:
                    if message.type != aiohttp.WSMsgType.TEXT:
                        if message.type in (aiohttp.WSMsgType.CLOSED, aiohttp.WSMsgType.ERROR):
                            if not done.done():
                                done.set_exception(RuntimeError("ComfyUI WebSocket closed early"))
                            return
                        continue
                    payload = json.loads(message.data)
                    message_type = payload.get("type")
                    data = payload.get("data", {})
                    event_time = time.time()
                    events.append({"type": message_type, "time": event_time, "data": data})
                    if message_type == "execution_success" and data.get("prompt_id") == prompt_id:
                        if not done.done():
                            done.set_result("success")
                    elif message_type == "execution_error" and data.get("prompt_id") == prompt_id:
                        if not done.done():
                            done.set_exception(RuntimeError(json.dumps(data, indent=2)))
                    elif message_type == "progress_state" and data.get("prompt_id") == prompt_id:
                        for node_id, state in data.get("nodes", {}).items():
                            previous = node_states.get(node_id)
                            if previous != state:
                                transition = {
                                    "node": node_id,
                                    "state": state.get("state"),
                                    "value": state.get("value"),
                                    "max": state.get("max"),
                                    "time": event_time,
                                }
                                node_transitions.append(transition)
                                node_states[node_id] = state
                                if node_id == "11" and state.get("state") in ("running", "finished"):
                                    sampler_progress.append(transition)

            receiver = asyncio.create_task(receive_events())
            start = time.perf_counter()
            payload = {
                "prompt": graph,
                "client_id": client_id,
                "prompt_id": prompt_id,
                "extra_data": {"extra_pnginfo": {}},
            }
            async with session.post(f"{args.base_url}/prompt", json=payload) as response:
                response_payload = await response.json()
                if response.status != 200:
                    receiver.cancel()
                    raise RuntimeError(f"Queue rejected prompt: {json.dumps(response_payload, indent=2)}")
                if response_payload.get("prompt_id") != prompt_id:
                    receiver.cancel()
                    raise RuntimeError(f"Prompt ID mismatch: {response_payload}")

            outcome = await asyncio.wait_for(done, timeout=args.timeout)
            elapsed = time.perf_counter() - start
            if outcome != "success":
                raise RuntimeError(f"Unexpected execution outcome: {outcome}")
            receiver.cancel()
            try:
                await receiver
            except asyncio.CancelledError:
                pass

    history = None
    for _ in range(30):
        async with session.get(f"{args.base_url}/history/{prompt_id}") as response:
            history_payload = await response.json()
        history = history_payload.get(prompt_id)
        if history is not None:
            break
        await asyncio.sleep(0.1)
    if history is None:
        raise RuntimeError(f"No history found for {prompt_id}")

    output_records = []
    for node_id, output in history.get("outputs", {}).items():
        for image in output.get("images", []):
            source = output_root / image.get("subfolder", "") / image["filename"]
            if not source.exists():
                raise FileNotFoundError(source)
            destination = args.copy_dir / args.filename_prefix / image["filename"]
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)
            output_records.append(
                {
                    "node": node_id,
                    "source": str(source),
                    "copy": str(destination),
                    "filename": image["filename"],
                    "subfolder": image.get("subfolder", ""),
                }
            )
    if not output_records:
        raise RuntimeError("Execution succeeded but produced no image output")

    node_durations = {}
    for node_id, transitions in ((node_id, [t for t in node_transitions if t["node"] == node_id]) for node_id in node_states):
        running = next((t["time"] for t in transitions if t["state"] == "running"), None)
        finished = next((t["time"] for t in transitions if t["state"] == "finished"), None)
        if running is not None and finished is not None:
            node_durations[node_id] = {
                "class_type": graph[node_id]["class_type"],
                "seconds": finished - running,
            }

    step_intervals = []
    for index, update in enumerate(sampler_progress):
        if update["state"] != "running":
            continue
        value = int(update["value"])
        previous_time = sampler_progress[index - 1]["time"] if index else update["time"]
        step_intervals.append(
            {
                "step": value,
                "seconds_since_previous_event": update["time"] - previous_time,
            }
        )

    valid_gpu = [sample for sample in telemetry.samples if "error" not in sample]
    gpu_summary = {}
    if valid_gpu:
        gpu_summary = {
            "peak_memory_used_mib": max(sample["memory.used"] for sample in valid_gpu),
            "minimum_memory_free_mib": min(sample["memory.free"] for sample in valid_gpu),
            "maximum_utilization_percent": max(sample["utilization.gpu"] for sample in valid_gpu),
            "mean_power_watts": sum(sample["power.draw"] for sample in valid_gpu) / len(valid_gpu),
            "maximum_power_watts": max(sample["power.draw"] for sample in valid_gpu),
            "maximum_temperature_c": max(sample["temperature.gpu"] for sample in valid_gpu),
            "minimum_sm_clock_mhz": min(sample["clocks.current.sm"] for sample in valid_gpu),
            "sample_count": len(valid_gpu),
        }
    return {
        "prompt_id": prompt_id,
        "elapsed_seconds": elapsed,
        "size": args.size,
        "seed": args.seed,
        "prompt": prompt,
        "schedule": args.schedule,
        "raw_sigmas": args.raw_sigmas or {
            "v02_6step": [1.0, 0.9375, 0.875, 0.75, 0.5, 0.25],
            "v02_5step": [1.0, 0.875, 0.75, 0.5, 0.25],
            "v02_4step": [1.0, 0.75, 0.5, 0.25],
            "candidate_2step": [1.0, 0.5],
            "candidate_1step": [1.0],
        }.get(args.schedule),
        "effective_sigmas": sigmas,
        "models": {
            "dit": args.model,
            "dit_loader": args.dit_loader,
            "text_encoder": args.text_encoder,
            "vae": args.vae,
            "lora": args.lora,
            "lora_strength": args.strength if args.lora else None,
        },
        "settings": {
            "cache_device": args.cache_device,
            "cache_dtype": args.cache_dtype,
            "positive_only": args.positive_only,
            "text_encoder_device": args.text_encoder_device,
            "attention_backend": args.attention_backend,
            "sampler": "euler",
            "cfg": 1.0,
        },
        "node_durations": node_durations,
        "sampler_progress": sampler_progress,
        "step_intervals": step_intervals,
        "outputs": output_records,
        "history_status": history.get("status"),
        "gpu_summary": gpu_summary,
        "gpu_samples": telemetry.samples,
    }


async def async_main(args):
    args.copy_dir.mkdir(parents=True, exist_ok=True)
    timeout = aiohttp.ClientTimeout(total=args.timeout + 60)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        system = await wait_for_server(session, args.base_url)
        print(json.dumps({"server_ready": system}, indent=2), flush=True)
        records = []
        for run_index in range(args.warmup + args.repeats):
            args.filename_prefix = f"{args.tag}_{args.size}_{run_index:03d}"
            run_args = argparse.Namespace(**vars(args))
            if args.vary_seed:
                run_args.seed = args.seed + run_index
            started = time.time()
            print(f"starting run {run_index + 1}/{args.warmup + args.repeats}", flush=True)
            run_prompt = args.prompt
            if args.vary_prompt:
                run_prompt = f"{args.prompt} Variation marker {run_index}."
            record = await run_once(session, run_args, run_prompt, args.output_root)
            record["run_index"] = run_index
            record["warmup"] = run_index < args.warmup
            record["started_at"] = started
            record["tag"] = args.tag
            record["gpu"] = system["devices"][0]
            records.append(record)
            print(json.dumps(record, indent=2), flush=True)

        args.results_path.parent.mkdir(parents=True, exist_ok=True)
        with args.results_path.open("w", encoding="utf-8") as file:
            json.dump(records, file, indent=2)
        print(f"wrote {args.results_path}", flush=True)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8188")
    parser.add_argument("--tag", default="v02_int8")
    parser.add_argument("--size", type=int, default=512)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--schedule", choices=["v02_6step", "v02_5step", "v02_4step", "candidate_2step", "candidate_1step"], default="v02_6step")
    parser.add_argument("--raw-sigmas", help="Comma-separated raw sigma nodes; overrides --schedule.", type=str)
    parser.add_argument("--model", default="qwen_image_2.1_int8_convrot.safetensors")
    parser.add_argument("--dit-loader", choices=["safetensors", "gguf"], default="safetensors")
    parser.add_argument("--text-encoder", default="qwen3vl_8b_int8_convrot.safetensors")
    parser.add_argument("--text-encoder-device", choices=["default", "cpu"], default="default")
    parser.add_argument("--vae", default="qwen_image_2.1_vae_bf16.safetensors")
    parser.add_argument("--lora", default="Qwen-Image-2.1-turbo-v0.2.1-6step-lora-r128.safetensors")
    parser.add_argument("--no-lora", action="store_true", help="Disable the model adapter for a counterfactual run.")
    parser.add_argument("--lora-bypass", action="store_true", help="Apply the model LoRA through the quantized-safe bypass path.")
    parser.add_argument("--unmerged-lora", action="store_true", help="Apply the Turbo adapter as an unmerged runtime branch.")
    parser.add_argument("--strength", type=float, default=1.0)
    parser.add_argument("--cache-device", choices=["auto", "gpu", "cpu", "off", "bypass"], default="auto")
    parser.add_argument("--cache-dtype", choices=["default", "int8", "int4"], default="default")
    parser.add_argument("--positive-only", action="store_true", help="Skip negative text encoding and use BasicGuider for CFG 1 benchmarking.")
    parser.add_argument("--attention-backend", choices=["default", "pytorch attention", "comfy kitchen attention"], default="default")
    parser.add_argument("--prompt", default='A premium futuristic AI image-generation emblem designed as a Telegram bot profile picture, centered composition, a sleek abstract letter "R" formed from flowing luminous light and tiny digital particles, subtle neural-network geometry and generative-art patterns surrounding it, cinematic sci-fi aesthetic, deep black background, electric cyan and violet highlights, glossy glass-metal details, professional technology logo, sharp details, dramatic studio lighting, perfectly centered, square composition, no watermark')
    parser.add_argument("--vary-prompt", action="store_true", help="Append a unique neutral variation marker so prompt encoding is not cached while model loaders can be reused.")
    parser.add_argument("--vary-seed", action="store_true", help="Change only the random noise seed for each run while preserving the prompt and conditioning.")
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--timeout", type=float, default=1800)
    parser.add_argument("--telemetry-interval", type=float, default=0.1)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--copy-dir", type=Path, default=ROOT / "out" / "benchmarks")
    parser.add_argument("--results-path", type=Path)
    args = parser.parse_args()
    if args.raw_sigmas:
        args.raw_sigmas = [float(value.strip()) for value in args.raw_sigmas.split(",") if value.strip()]
        args.schedule = "custom"
    if args.no_lora:
        args.lora = ""
    if args.size % 32:
        parser.error("--size must be divisible by 32")
    if args.results_path is None:
        args.results_path = DEFAULT_RESULTS / f"comfy_{args.tag}.json"
    return args


def main():
    args = parse_args()
    asyncio.run(async_main(args))


if __name__ == "__main__":
    main()
