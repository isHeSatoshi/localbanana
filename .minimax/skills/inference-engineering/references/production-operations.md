# Production Operations

Use this reference for OPERATE mode.

## Serving Unit

Pin the model, runtime, accelerator stack, dependencies, build inputs, image digest, and hardware target. Keep large weights separate from the runtime image when this improves distribution and startup. Readiness must prove that the target workload can be served, not only that the process started.

## Capacity And Autoscaling

Treat autoscaling as feedback control. Combine workload-aware demand with observed saturation:

- arrivals, admitted work, input/output or modality-specific work units, cache state, and queue age;
- active concurrency, batch state, GPU compute and memory, host resources, and engine saturation.

Request count alone is not capacity. Define minimum and maximum replicas, measurement window, scale-down delay, per-replica concurrency, budget or quota ceiling, and overload response. The autoscaler and engine must share the same tested capacity envelope.

## Cold Start

Measure separately:

```text
capacity acquisition -> node startup -> image transfer -> weight transfer -> process startup -> compile or engine load -> readiness
```

Warm capacity and scale-down policy follow from the measured chain. Scale-to-zero is valid only when the first-request delay is acceptable or the work is asynchronous.

## Routing, Queues, And Admission

Keep the concerns distinct:

- routing chooses an eligible destination using request cost, cache or adapter affinity, modality, locality, residency, health, and remaining concurrency;
- load balancing distributes work across eligible capacity;
- bounded queues hold accepted work and define expiry, cancellation, retries, priority, and maximum wait;
- admission control rejects or defers work before overload becomes unbounded latency.

Compound pipelines need stage-specific capacity and queues plus end-to-end backpressure. Co-locate tightly coupled stages when network cost is material.

## Failure And Deployment

- Design for accelerator, node, cluster, region, provider, and control-plane failure.
- Multi-region or multi-provider operation is justified only by tested capacity, traffic shifting, compatible artifacts, data rules, and observability.
- Use synthetic checks, load tests, privacy-reviewed shadow traffic, and staged canaries.
- Pre-warm each canary step, compare candidate with baseline, stop on a violated gate, and retain a tested rollback through the observation window.

## Observability

Join demand shape, cache state, queueing, user latency, engine state, fleet lifecycle, hardware, client and network timing, and change history. Alert on user risk or exhaustion: sustained tail breach, rejection or error rise, queue-age breach, lost redundancy, capacity ceiling, repeated cold-start failure, or rollout regression.

High utilization without user impact is context, not automatically an incident.

## Client And Cost

Budget DNS, connection setup, TLS, upload, queue, server work, first result, streaming, parsing, and rendering. Reuse sessions. Choose synchronous, asynchronous, streaming, or bidirectional protocols from the interaction contract and include connection concurrency.

Compare API and dedicated infrastructure over representative normal, peak, and failure periods. Include active, warm, and stranded accelerator time; CPU; storage; network; commitments; observability; incidents; upgrades; and engineering ownership.

## Full Production-Readiness Record

Require all nine parts only for a full readiness review, complete operating design, or production acceptance claim. For a narrow OPERATE question, return only the relevant fields and state which broader readiness claims were not evaluated.

The full record contains:

1. service and data contract;
2. immutable runtime manifest;
3. tested capacity and cold-start envelope;
4. routing, queue, and admission policy;
5. failure and recovery plan;
6. rollout gates and rollback;
7. observability and ownership map;
8. workload-level cost model;
9. client behavior and end-to-end latency budget.
