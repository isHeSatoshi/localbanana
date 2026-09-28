# Benchmark And Capacity

Use this reference for performance claims, engine comparisons, optimization tests, and sizing.

## Benchmark Contract

Freeze and preserve:

- model revision, tokenizer, template, precision, quality settings, and sampling parameters;
- runtime or engine version, container digest, dependencies, flags, and quantization artifact;
- exact hardware, form factor, host resources, topology, region, and power mode;
- input and output distributions, media shape, concurrency, burst and jitter pattern, and cache state;
- workload generator version, seed where relevant, warmup, duration, repetitions, and raw result paths.

Prefer privacy-reviewed production shadowing. Otherwise replay sanitized traces or generate traffic that preserves the distributions that control work. Fixed average lengths and fixed-rate traffic can hide queueing and tail failure.

## Required Measurements

- first useful result: TTFT, first phrase, partial transcript, image, or other modality-native outcome;
- continuation speed: ITL, perceived TPS, audio real-time behavior, or stage cadence;
- end-to-end P50/P90/P95/P99 and inference-only latency;
- request and workload throughput;
- queue time, errors, timeouts, rejections, retries, and saturation behavior;
- device memory, compute, bandwidth, utilization, interconnect, CPU, and network when diagnosing;
- task-specific quality before and after the change;
- cost per successful useful unit and representative time window.

One latency or throughput number is not a decision record.

## Experiment Discipline

1. Capture a baseline.
2. Warm the system under a declared policy.
3. Run enough work to expose variance and tails.
4. Repeat runs and report variation.
5. Change one major variable.
6. Test interactions after isolated changes.
7. Profile only when the benchmark shows a gap but not the cause.
8. Return profiling changes to the full end-to-end benchmark.
9. Compare with a service objective and quality floor, not only the previous run.

Static engine rankings are invalid outside the stated model, version, hardware, workload, and quality envelope.

## Capacity And Cost

Derive per-replica capacity from the highest tested concurrency that still meets the tail-latency and quality objectives for expected and outlier request shapes. Apply a target utilization margin; do not plan at measured peak.

```text
effective_capacity = measured_peak_capacity * target_utilization
required_replicas = ceil(peak_required_capacity / effective_capacity)
cost_per_million_tokens = hourly_instance_price / sustained_tokens_per_hour * 1_000_000
cost_per_successful_request = hourly_instance_price / successful_requests_per_hour
```

Add warm and idle capacity, cold starts, failed or retried work, CPU, storage, network, cross-region transfer, orchestration, and engineering ownership.

## Result Record

```text
Question:
Service and quality gates:
Manifest:
Traffic and workload envelope:
Baseline:
Single change:
Repeated results and variance:
Quality delta:
Cost delta:
Confounders:
Decision: adopt / tune / reject
Rollback threshold:
Raw evidence:
```

Label each claim as measured here, sourced from current official documentation, estimated, or hypothesized.

