# Fundamentals And Hardware

Use this reference for LEARN mode and for the first pass of DESIGN.

## Workload Contract

Capture or explicitly mark unknown:

- model family, revision, architecture, modality, tokenizer, chat template, precision, and license;
- input and output shape distributions, including context, resolution, duration, and sampling settings;
- online, offline, streaming, or mixed interaction;
- normal, peak, and burst concurrency and arrival pattern;
- first useful user outcome and end-to-end latency tolerance;
- task-specific quality failures that are unacceptable;
- cost ceiling per request, user, workload unit, and representative time window;
- privacy, data-residency, provider, device, region, and availability constraints.

When most of this is unknown, recommend a shared service and measurement plan. Do not invent precise infrastructure.

Classify each material path as online, offline, streaming, or mixed. When online and offline paths have material volume or incompatible latency and throughput objectives, compare separate pools or deployments with one shared pool. Do not force separation when measured demand does not justify it.

## Metrics By Interaction

- **Streaming text:** time to first token (TTFT), inter-token latency (ITL), perceived tokens per second, end-to-end percentiles.
- **Machine-consumed or tool output:** total response time; partial tokens do not create value.
- **Offline work:** completed workload per wall-clock time and cost per successful unit.
- **Production:** P50/P90/P95/P99, queue time, errors, rejections, retries, and inference-only versus end-to-end latency.

State whether throughput is per user, replica, GPU, node, or total service. State the workload shape and traffic level beside every latency or throughput number.

## Phase Model

- **Prefill** processes the input and creates the KV cache. It often controls TTFT and often begins compute-bound.
- **Decode** produces one token at a time while reading weights and cache state. It often controls ITL and often begins memory-bandwidth-bound.
- Batching can move decode toward compute saturation while increasing queueing, KV memory, and tail latency.
- Tokenization, templates, special tokens, stop rules, sampling, and structured-output constraints affect both correctness and performance. Include them in every manifest.

For Mixture-of-Experts models, active parameters approximate per-token compute; total parameters still control weight memory and distribution. Measure expert imbalance and communication.

## Bottleneck Test

Use the roofline model as a hypothesis generator:

```text
arithmetic_intensity = operations / bytes_moved
hardware_balance = peak_operations_per_second / peak_bytes_per_second
performance_ceiling = min(peak_compute, arithmetic_intensity * memory_bandwidth)
```

Below the hardware balance point suggests bandwidth pressure; above it suggests compute pressure. Confirm with achieved FLOPS, achieved bandwidth, traces, utilization, queueing, and end-to-end measurements. Peak specifications are ceilings, not capacity plans.

## Memory Fit

First-pass weight estimate:

```text
weight_bytes ~= parameter_count * bits_per_weight / 8
minimum_devices_by_memory = ceil(total_required_VRAM / usable_VRAM_per_device)
```

Approximate LLM KV cache:

```text
kv_bytes ~= 2 * layers * cached_tokens * kv_heads * head_dim * bytes_per_value * concurrent_sequences
```

Add quantization metadata, activations, temporary workspaces, allocator fragmentation, speculative branches, replicas, and operating margin. A successful model load proves only that one configuration fit once.

## Hardware Decision

1. Size memory before comparing speed.
2. Predict the dominant limit by phase and modality.
3. Compare the same precision, density or real sparsity mode, exact device variant, and form factor.
4. Inspect CPU, host RAM, storage, network, GPU-to-GPU links, node-to-node links, virtualization, and topology.
5. Prefer the smallest topology that meets the service objective. Price communication before adding devices or nodes.
6. Gate new hardware on current kernel, compiler, runtime, profiling, observability, supply, and operator maturity.
7. Select by measured cost per useful output at the required quality and latency, not FLOPS per dollar.

Useful lower bound for communication:

```text
communication_time_floor ~= bytes_transferred / effective_link_bandwidth
```

If this is comparable to or larger than the divided compute time, more devices can make the system slower.

## Learning Completion Test

LEARN is complete when the user can state the workload, first useful outcome, quality floor, metric meanings, likely phase, likely bottleneck, unknowns, and evidence that would change the diagnosis.
