# Serving And Optimization

Use this reference to select a runtime level and conditional optimization.

## Abstraction Ladder

Start at the highest level that meets the measured requirement:

1. Managed or shared API while requirements and demand are unclear.
2. Production inference engine for a supported model and hardware target.
3. Exported graph or framework implementation when engine support or control is insufficient.
4. Custom kernels only after profiling identifies an operator-level limit.
5. Distributed serving only when model or traffic scale makes its benefit material.

Lower-level code is justified by evidence, not prestige. Return every component improvement to an end-to-end workload test.

## Technique Selection

| Technique | Use when | Main gain | Main price |
| --- | --- | --- | --- |
| Quantization | Weight, KV, bandwidth, or memory capacity is limiting | Less movement, more capacity, sometimes more compute | Quality risk, conversion, scale metadata, kernel support |
| Batching | Aggregate demand exists and throughput or unit cost matters | Utilization and total throughput | Queueing, tail latency, cache pressure, less spare compute |
| Prefix/KV caching | Requests repeat a long exact prefix | TTFT, throughput, and input-cost reduction | VRAM, eviction, affinity routing, tenant isolation |
| Speculative decoding | Decode is bandwidth-bound, concurrency is not saturated, and acceptance is high | Lower ITL and higher per-user TPS | Draft/validation work, memory, coordination, possible throughput loss |
| Model parallelism | Model plus runtime headroom cannot fit, or latency/throughput justifies multiple devices | Capacity or speed | Communication and topology complexity |
| Prefill/decode disaggregation | Sustained large, prefill-heavy traffic causes phase contention | Independent scaling and specialization | KV transfer, two queues, routing, capacity fragmentation |

## Rules

### Quantization

- Start with weights or selected linear layers; widen to activations and KV cache only for a measured need.
- Keep sensitive operations and layers at higher precision unless product evaluations prove safety.
- Match the format to current hardware kernels and runtime support.
- Compare the same model and workload with task-specific quality checks, long-context cases when relevant, and performance variance.
- If quality changes, reduce scope or use a less aggressive format before rejecting quantization entirely.

### Batching And Speculation

- Sweep concurrency and batch policy together with speculation.
- Speculation helps decode, not TTFT. It loses value when larger batches consume spare compute or token acceptance falls.
- Measure draft cost, accepted tokens, rejected work, memory, per-user latency, aggregate throughput, and total cost.
- Disable speculation when the validation work no longer pays for itself.

### Prefix And KV Caching

- Standard reuse requires exact tokens from the beginning and stops at the first difference.
- Reuse also requires an equivalent authorization and execution key. Bind private cache entries to the tenant or principal, permission scope, model revision, tokenizer and template, adapter, and privacy or retention class. Define ownership, lifetime, and invalidation before enabling reuse.
- Forbid cross-tenant reuse of private context unless an approved design proves equivalent authority and isolation.
- Put stable system instructions, tool schemas, document context, and conversation history before request-specific values.
- Measure repeated prefix length and hit value on real prompts.
- Balance load and cache locality; neither pure least-loaded routing nor pure affinity is sufficient.
- Budget active and reusable KV separately. Track occupancy, hits, eviction, tier transfer, authorization failures, invalidation, tenant isolation, and node churn.

### Parallelism

- Prefer horizontal replicas when one node fits the model and service headroom.
- Tensor parallelism is the first within-node latency option when fast collective links can pay for frequent synchronization.
- Expert parallelism serves sparse MoE layers and often favors aggregate throughput.
- Pipeline parallelism is mainly a cross-node necessity and adds bubbles.
- Select mixed topology from measured memory, latency, throughput, and collective costs.

### Disaggregation

Test after simpler controls. Require sustained volume, a large enough model, long uncached inputs, visible prefill/decode contention, and a measured KV transfer path. Route short prompts and cache hits locally when that is cheaper. Operate explicit prefill and decode queues, admission limits, and a changing worker ratio.

## Experiment Order

1. Freeze model, runtime, hardware, request distributions, sampling, and quality baseline.
2. Establish low-, medium-, and saturation-concurrency behavior.
3. Remove capacity or bandwidth pressure with selective quantization.
4. Improve prompt stability and cache reuse.
5. Tune batching and speculation together.
6. Add model parallelism only for proven fit or speed constraints.
7. Test disaggregation last.

For every technique, record the observed bottleneck, mechanism, competing effect, quality gate, performance gate, failure signal, rollback, and result.
