"""Train a genuine two-step Qwen-Image-2.1 student from exported six-step teacher states."""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from safetensors import safe_open


ROOT = Path(__file__).resolve().parents[1]
COMFY_ROOT = ROOT / "ComfyUI"
DEFAULT_MANIFEST = ROOT / "results" / "qwen21_teacher_trajectories.json"
DEFAULT_REPORT = ROOT / "results" / "qwen21_two_step_gradient_probe.json"
DEFAULT_OUTPUT = ROOT / "out" / "research" / "qwen21_two_step"
DEFAULT_DIT = Path(r"H:\1secimage\models\comfy-qwen21\diffusion_models\qwen_image_2.1_int8_convrot.safetensors")
DEFAULT_LORA = Path(r"H:\1secimage\models\comfy-qwen21\loras\Qwen-Image-2.1-viggle-turbo-v0.2.1-6step-lora-r128.safetensors")

sys.path.insert(0, str(COMFY_ROOT))

import comfy.model_management
import comfy.sd
import comfy.samplers
import comfy.sampler_helpers
import comfy.utils
from comfy.weight_adapter import adapter_maps
from comfy.weight_adapter.bypass import BypassInjectionManager
from comfy_extras.nodes_train import TrainGuider, patch, unpatch


CORRECTION_TARGETS = {
    "attention": {"to_q", "to_k", "to_v", "to_out.0"},
    "output": {"to_out.0"},
}


def shifted_sigmas(size: int, raw: list[float]) -> list[float]:
    sequence_length = (size // 16) ** 2
    slope = (0.9 - 0.5) / (8192 - 256)
    mu = sequence_length * slope + (0.5 - slope * 256)
    alpha = math.exp(mu)
    return [alpha * sigma / (1 + (alpha - 1) * sigma) for sigma in raw] + [0.0]


def load_safetensor(path: Path, key: str) -> torch.Tensor:
    with safe_open(str(path), framework="pt", device="cpu") as file:
        return file.get_tensor(key)


def load_conditioning(path: Path) -> list[list]:
    with safe_open(str(path), framework="pt", device="cpu") as file:
        metadata = file.metadata() or {}
        conditioning = file.get_tensor("conditioning")
        options = json.loads(metadata.get("conditioning_options", "{}"))
        stored = {key: file.get_tensor(key) for key in file.keys() if key != "conditioning"}
    lists: dict[str, dict[int, torch.Tensor]] = {}
    for key, value in stored.items():
        name, _, index = key.rpartition(".")
        if index.isdigit():
            lists.setdefault(name, {})[int(index)] = value
        else:
            options[key] = value
    for name, values in lists.items():
        options[name] = [values[index] for index in sorted(values)]
    return [[conditioning, options]]


def load_noise(size: int, seed: int, dtype: torch.dtype) -> torch.Tensor:
    latent = torch.zeros(1, 64, size // 16, size // 16, dtype=dtype)
    generator = torch.manual_seed(seed)
    return torch.randn(latent.shape, generator=generator, dtype=torch.float32, layout=latent.layout).to(dtype)


def correction_keys(diffusion_model: torch.nn.Module, targets: set[str]) -> list[str]:
    keys = []
    for name, _module in diffusion_model.named_modules():
        if name.endswith(".to_out.0"):
            leaf = "to_out.0"
        else:
            _, _, leaf = name.rpartition(".")
        if name.startswith("transformer_blocks.") and leaf in targets:
            keys.append(name)
    return keys


def setup_correction(model, rank: int, device: torch.device, targets: set[str]):
    diffusion_model = model.model.diffusion_model
    manager = BypassInjectionManager()
    parameters = {}
    adapter_type = adapter_maps["LoRA"]
    module_names = correction_keys(diffusion_model, targets)
    for module_name in module_names:
        module = diffusion_model.get_submodule(module_name)
        adapter = adapter_type.create_train(module.weight, rank=rank, alpha=float(rank)).to(torch.bfloat16)
        with torch.no_grad():
            torch.nn.init.kaiming_uniform_(adapter.lora_down.weight, a=math.sqrt(5.0))
            adapter.lora_up.weight.zero_()
        adapter.to(device=device)
        adapter.train()
        adapter.lora_down.weight.requires_grad_(True)
        adapter.lora_up.weight.requires_grad_(True)
        parameters[f"diffusion_model.{module_name}.lora_down.weight"] = adapter.lora_down.weight
        parameters[f"diffusion_model.{module_name}.lora_up.weight"] = adapter.lora_up.weight
        manager.add_adapter(f"diffusion_model.{module_name}.weight", adapter, strength=1.0)
    injections = manager.create_injections(model.model)
    model.set_injections("qwen21_two_step_correction", injections)
    return manager, parameters, len(module_names)


def save_correction(parameters: dict[str, torch.Tensor], path: Path, rank: int, steps: int) -> None:
    tensors = {}
    for name, parameter in parameters.items():
        prefix = name.removeprefix("diffusion_model.").removesuffix(".weight")
        down = prefix.removesuffix(".lora_down")
        up = prefix.removesuffix(".lora_up")
        key = "transformer." + (down if name.endswith(".lora_down.weight") else up)
        tensors[f"{key}.lora_{'A' if name.endswith('.lora_down.weight') else 'B'}.weight"] = (
            parameter.detach().to(torch.bfloat16).cpu().contiguous()
        )
    metadata = {
        "format": "pt",
        "lora_adapter_metadata": json.dumps(
            {
                "transformer.alpha_pattern": {},
                "transformer.bias": "none",
                "transformer.init_lora_weights": True,
                "transformer.lora_alpha": rank,
                "transformer.lora_dropout": 0.0,
                "transformer.peft_type": "LORA",
                "transformer.r": rank,
                "transformer.rank_pattern": {},
                "transformer.target_modules": ["to_q", "to_k", "to_v", "to_out"],
                "transformer.use_rslora": False,
            }
        ),
        "training_steps": str(steps),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    comfy.utils.save_torch_file(tensors, path, metadata=metadata)


class TwoStepObjective:
    def __init__(self, state_target: torch.Tensor, final_target: torch.Tensor, optimizer, parameter_names, step: bool, grad_scale: float = 1.0):
        self.state_target = state_target
        self.final_target = final_target
        self.optimizer = optimizer
        self.parameter_names = parameter_names
        self.step = step
        self.grad_scale = grad_scale
        self.result = {}

    def sample(
        self,
        model_wrap,
        sigmas,
        extra_args,
        callback,
        noise,
        latent_image=None,
        denoise_mask=None,
        disable_pbar=False,
    ):
        started = time.perf_counter()
        self.optimizer.zero_grad(set_to_none=True)
        x = noise.detach().requires_grad_(True)
        s_in = x.new_ones(x.shape[0])
        sigma_start = sigmas[0]
        sigma_state = sigmas[1]

        denoised_start = model_wrap(x, sigma_start * s_in, **extra_args)
        velocity_start = (x - denoised_start) / sigma_start
        state = x + (sigma_state - sigma_start) * velocity_start
        state_loss = F.mse_loss(state.float(), self.state_target.float())
        (state_loss * self.grad_scale).backward()
        state_gradient = gradient_summary(
            dict(zip(self.parameter_names, self.optimizer.param_groups[0]["params"]))
        )

        state_input = state.detach()
        denoised_final = model_wrap(state_input, sigma_state * s_in, **extra_args)
        final_loss = F.mse_loss(denoised_final.float(), self.final_target.float())
        (final_loss * self.grad_scale).backward()

        gradient_tensors = []
        gradient_norm = 0.0
        gradient_max_abs = 0.0
        gradient_nonfinite = 0
        gradient_top = {}
        for parameter_name, parameter in zip(self.parameter_names, self.optimizer.param_groups[0]["params"]):
            if parameter.grad is None:
                continue
            gradient = parameter.grad.detach()
            gradient_tensors.append(gradient)
            gradient_finite = torch.isfinite(gradient)
            gradient_nonfinite += int((~gradient_finite).sum().item())
            finite_gradient = gradient[gradient_finite].double()
            if finite_gradient.numel():
                tensor_norm = float(torch.linalg.vector_norm(finite_gradient).item())
                tensor_max = float(finite_gradient.abs().max().item())
                gradient_norm = math.hypot(gradient_norm, tensor_norm)
                gradient_max_abs = max(gradient_max_abs, tensor_max)
                if tensor_norm > gradient_top.get("l2", 0.0):
                    gradient_top = {"parameter": parameter_name, "l2": tensor_norm, "max_abs": tensor_max}

        if self.step and self.grad_scale > 0.0:
            self.optimizer.step()
        self.result = {
            "state_loss": float(state_loss.detach().item()),
            "final_loss": float(final_loss.detach().item()),
            "total_loss": float((state_loss + final_loss).detach().item()),
            "gradient_tensors": len(gradient_tensors),
            "gradient_l2": gradient_norm,
            "gradient_max_abs": gradient_max_abs,
            "gradient_nonfinite_values": gradient_nonfinite,
            "gradient_finite": gradient_nonfinite == 0 and bool(gradient_tensors),
            "gradient_top": gradient_top,
            "state_gradient": state_gradient,
            "predictions": {
                "state_target_mean": float(self.state_target.mean().item()),
                "state_target_std": float(self.state_target.std().item()),
                "state_pred_mean": float(state.mean().item()),
                "state_pred_std": float(state.std().item()),
                "final_target_mean": float(self.final_target.mean().item()),
                "final_target_std": float(self.final_target.std().item()),
                "final_pred_mean": float(denoised_final.mean().item()),
                "final_pred_std": float(denoised_final.std().item()),
            },
            "elapsed_seconds": time.perf_counter() - started,
        }
        return state_input.detach()


def step_scale(args, offset: int) -> float:
    if not args.gradient_only and args.warmup_steps and offset < args.warmup_steps:
        return (offset + 1) / args.warmup_steps
    return 1.0


def gradient_summary(parameters: dict[str, torch.Tensor]) -> dict:
    norm = 0.0
    maximum = 0.0
    tensors = 0
    nonfinite = 0
    for parameter in parameters.values():
        if parameter.grad is None:
            continue
        gradient = parameter.grad.detach()
        tensors += 1
        finite = torch.isfinite(gradient)
        nonfinite += int((~finite).sum().item())
        finite_gradient = gradient[finite].double()
        if finite_gradient.numel():
            norm = math.hypot(norm, float(torch.linalg.vector_norm(finite_gradient).item()))
            maximum = max(maximum, float(finite_gradient.abs().max().item()))
    return {
        "tensors": tensors,
        "l2": norm,
        "max_abs": maximum,
        "nonfinite_values": nonfinite,
    }


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--dit", type=Path, default=DEFAULT_DIT)
    parser.add_argument("--teacher-lora", type=Path, default=DEFAULT_LORA)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--limit", type=int, default=1)
    parser.add_argument("--rank", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--targets", choices=CORRECTION_TARGETS, default="attention")
    parser.add_argument("--steps", type=int, default=1)
    parser.add_argument("--gradient-only", action="store_true")
    parser.add_argument("--save-lora", action="store_true")
    parser.add_argument("--checkpoint-depth", type=int, choices=(0, 2), default=2)
    parser.add_argument("--detect-anomaly", action="store_true")
    parser.add_argument("--sync-offload", action="store_true")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--holdout", type=int, default=0)
    parser.add_argument("--warmup-steps", type=int, default=0)
    parser.add_argument("--decay-to", type=float, default=0.0)
    return parser.parse_args()


def main():
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the Qwen-Image-2.1 training probe")
    if args.rank < 1:
        raise ValueError("Rank must be positive")

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    all_records = manifest["records"][: args.limit or None]
    holdout = all_records[args.holdout:] if args.holdout else []
    records = all_records[: args.holdout] if args.holdout else all_records
    if not records:
        raise ValueError("The teacher manifest contains no training records")

    device = torch.device("cuda:0")
    if args.sync_offload:
        comfy.model_management.NUM_STREAMS = 0
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()

    model = comfy.sd.load_diffusion_model(
        str(args.dit),
        model_options={"load_device": device, "offload_device": torch.device("cpu")},
        disable_dynamic=True,
    )
    teacher_lora, metadata = comfy.utils.load_torch_file(str(args.teacher_lora), return_metadata=True)
    model, _ = comfy.sd.load_lora_for_models(model, None, teacher_lora, 1.0, 0.0, lora_metadata=metadata)
    model.model.requires_grad_(False).train()
    model.model_options["transformer_options"]["qwen_image21_cache"] = {"device": "off", "dtype": "default"}

    manager, parameters, correction_modules = setup_correction(
        model, args.rank, device, CORRECTION_TARGETS[args.targets]
    )
    optimizer = torch.optim.AdamW(parameters.values(), lr=args.learning_rate, betas=(0.9, 0.95))
    scheduler = None
    if args.decay_to > 0.0:
        scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=1.0, end_factor=args.decay_to / args.learning_rate, total_iters=len(records) * args.epochs
        )
    checkpointed = list(model.model.diffusion_model.transformer_blocks) if args.checkpoint_depth else []
    for module in checkpointed:
        patch(module, offloading=True)

    run_records = []
    holdout_records = []
    try:
        comfy.model_management.in_training = True
        if args.detect_anomaly:
            torch.autograd.set_detect_anomaly(True)
        for offset, record in enumerate(records * args.epochs):
            state_target = load_safetensor(Path(record["state_latent"]), "latent_tensor")
            final_target = load_safetensor(Path(record["final_latent"]), "latent_tensor")
            conditioning = load_conditioning(Path(record["conditioning"]))
            noise = load_noise(record["width"], record["seed"], torch.float32)
            sigmas = torch.tensor(shifted_sigmas(record["width"], [1.0, 0.25]), dtype=torch.float32)
            dummy_latent = torch.zeros_like(noise)
            target_device = model.load_device
            state_target = model.model.latent_format.process_in(state_target.to(target_device))
            final_target = model.model.latent_format.process_in(final_target.to(target_device))

            objective = TwoStepObjective(
                state_target,
                final_target,
                optimizer,
                list(parameters),
                not args.gradient_only,
                step_scale(args, offset),
            )
            guider = TrainGuider(model, offloading=True)
            guider.set_conds(conditioning)
            run_started = time.perf_counter()
            guider.sample(noise, dummy_latent, objective, sigmas, seed=record["seed"])
            run_records.append(
                {
                    "index": record["index"],
                    "sample_id": record["sample_id"],
                    "prompt": record["prompt"],
                    "seed": record["seed"],
                    "objective": objective.result,
                    "elapsed_seconds": time.perf_counter() - run_started,
                }
            )
            print(json.dumps(run_records[-1], indent=2), flush=True)
            if scheduler is not None:
                scheduler.step()
            if args.gradient_only:
                break

        for record in holdout:
            state_target = load_safetensor(Path(record["state_latent"]), "latent_tensor")
            final_target = load_safetensor(Path(record["final_latent"]), "latent_tensor")
            conditioning = load_conditioning(Path(record["conditioning"]))
            noise = load_noise(record["width"], record["seed"], torch.float32)
            sigmas = torch.tensor(shifted_sigmas(record["width"], [1.0, 0.25]), dtype=torch.float32)
            dummy_latent = torch.zeros_like(noise)
            target_device = model.load_device
            state_target = model.model.latent_format.process_in(state_target.to(target_device))
            final_target = model.model.latent_format.process_in(final_target.to(target_device))
            objective = TwoStepObjective(
                state_target, final_target, optimizer, list(parameters), False, 0.0
            )
            guider = TrainGuider(model, offloading=True)
            guider.set_conds(conditioning)
            run_started = time.perf_counter()
            guider.sample(noise, dummy_latent, objective, sigmas, seed=record["seed"])
            holdout_records.append(
                {
                    "index": record["index"],
                    "sample_id": record["sample_id"],
                    "seed": record["seed"],
                    "state_loss": objective.result.get("state_loss"),
                    "final_loss": objective.result.get("final_loss"),
                    "total_loss": objective.result.get("total_loss"),
                    "elapsed_seconds": time.perf_counter() - run_started,
                }
            )
            print(json.dumps(holdout_records[-1], indent=2), flush=True)
    finally:
        comfy.model_management.in_training = False
        for module in checkpointed:
            unpatch(module)
        model.cleanup()
        manager.clear_adapters()

    if args.save_lora:
        output = args.output_root / f"qwen21_two_step_r{args.rank}_{args.steps}step.safetensors"
        save_correction(parameters, output, args.rank, args.steps)
    else:
        output = None

    report = {
        "mode": "gradient_only" if args.gradient_only else "train",
        "manifest": str(args.manifest),
        "dit": str(args.dit),
        "teacher_lora": str(args.teacher_lora),
        "rank": args.rank,
        "targets": args.targets,
        "learning_rate": args.learning_rate,
        "requested_steps": args.steps,
        "epochs": args.epochs,
        "train_records": len(records),
        "holdout_records": len(holdout),
        "sync_offload": args.sync_offload,
        "correction_modules": correction_modules,
        "checkpoint_depth": args.checkpoint_depth,
        "checkpoint_modules": len(checkpointed),
        "gradient_after_clear": gradient_summary(parameters),
        "cuda_peak_allocated_mib": torch.cuda.max_memory_allocated(device) / 1024**2,
        "cuda_peak_reserved_mib": torch.cuda.max_memory_reserved(device) / 1024**2,
        "total_elapsed_seconds": time.perf_counter() - started,
        "output_lora": str(output) if output else None,
        "records": run_records,
        "holdout": holdout_records,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
