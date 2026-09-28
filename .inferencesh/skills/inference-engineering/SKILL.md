---
name: inference-engineering
description: Design, size, benchmark, tune, debug, or review machine-learning model inference on local or production hardware. Use for accelerator fit, memory and bandwidth limits, serving runtimes, quantization, batching, KV cache, speculation, parallelism, latency, throughput, cost, capacity, reliability, and production operations. Do not use for hosted-model API documentation or pricing alone, AI media generation through inference.sh, GPU graphics, or generic application deployment.
---

# Inference Engineering

Treat inference as a product system, not a model-server setting. The model, runtime, hardware, router, autoscaler, network, client, workload, and quality threshold jointly determine the result.

## Select A Mode

- **LEARN:** Explain the system or derive a mental model. Read [fundamentals-and-hardware.md](references/fundamentals-and-hardware.md).
- **DESIGN:** Turn a workload into a model, hardware, runtime, and topology decision. Read the fundamentals reference, then [serving-and-optimization.md](references/serving-and-optimization.md). For speech, image, video, embeddings, or VLMs, also read [modalities.md](references/modalities.md).
- **MEASURE:** Design, run, or review a benchmark, experiment, or capacity model. Read [benchmark-and-capacity.md](references/benchmark-and-capacity.md) and the reference for the surface being measured.
- **OPERATE:** Design or review scaling, routing, deployment, observability, failure recovery, or cost controls. Read [production-operations.md](references/production-operations.md).

If the task depends on current models, hardware, prices, runtime features, commands, compatibility, or benchmarks, also read [source-map.md](references/source-map.md) and verify the relevant official source live.

## Governing Sequence

Follow this causal order:

1. Define the workload and first useful user outcome.
2. Set a task-specific quality floor.
3. Separate prefill, decode, preprocessing, queueing, transport, and postprocessing.
4. Classify the limiting resource: quality, compute, bandwidth, memory capacity, communication, queueing, or operations.
5. Choose the smallest intervention that can remove the measured limit.
6. Compare against a preserved baseline under production-shaped traffic.
7. Adopt only when quality, service, cost, and rollback gates pass.

Do not name a GPU, runtime, or optimization before the workload is clear. Do not accept a faster kernel, higher token rate, or successful model load as product proof.

## Invariants

- Use product-specific evaluations before model optimization. Prefer the smallest well-supported model that clears the quality gate.
- Keep shared inference while requirements are unclear. Recommend dedicated serving only for a measurable scale, specialization, control, privacy, or orchestration need.
- Treat prefill as commonly compute-bound and decode as commonly bandwidth-bound only as starting hypotheses. Profile the exact workload.
- Include weights, KV cache, activations, workspaces, fragmentation, replicas, and margin in memory sizing.
- Require a full-precision or current accepted-quality baseline before lossy quantization, changed attention, caching approximations, or distilled models.
- Change one major variable per experiment. Test the intended combination after isolated tests because optimizations can interfere.
- Separate measured facts, documentation facts, estimates, and hypotheses.
- Treat vendor claims and named product rankings as time-sensitive. Current official documentation and target-stack measurements control.

## Default Output

Return only the parts the task needs, but preserve this decision record when making a recommendation:

```text
Mode:
Workload and first useful outcome:
Quality gate:
Service objectives by workload bucket and cache state:
Cost objectives:
Known facts:
Assumptions and unknowns:
Phase and bottleneck diagnosis:
Candidate intervention:
Baseline and experiment:
Measured or expected tradeoffs:
Acceptance and rollback:
Current facts that need verification:
```

Lead with the decision in product language. Keep calculations and evidence available as supporting detail.
