# Published measurements

Every number in the top-level README comes from one of these files. Nothing here is
hand-edited.

| file | what it holds |
|---|---|
| `latency_floor.json` | Fitted per-call cost: `denoise(N) = a + (N-1)*c`, the first-call surcharge, and the non-denoise fixed cost. Source of the speed table and the "cutting steps is not how this reaches one second" result. |
| `sweep_p4_longprompt.json` | Warm latency at 1 through 5 steps on the long-prompt protocol, with per-stage node durations. Shows the 0.009 s three-to-two-step gap. |
| `schedule_stepcount_generalization.json` | Step-count quality over 4 prompts x 3 seeds (12 cells), each candidate scored against a six-step reference from the same prompt and seed. Source of the 1/2/3/4-step comparison. |

## Reproducing the speed numbers

```bash
# latency at 1 through 6 steps, prompt varied to defeat the graph cache
python scripts/sweep_sigmas.py --tag floor --mode latency --warmup 1 --repeats 5 \
  --grids "n1=1.0;n2=1.0,0.78;n3=1.0,0.75,0.25;n4=1.0,0.75,0.5,0.25;n5=1.0,0.875,0.75,0.5,0.25;n6=1.0,0.9375,0.875,0.75,0.5,0.25"

python scripts/fit_latency_floor.py --latency results/sweep_floor.json
```

The step-count file was produced by a CLIP and DINOv2 comparison against a six-step
reference. That evaluation script is not part of this repository; the file is kept as the
record behind the claims in the README.

## Caveat

Absolute latency moved 20 to 45 percent between sessions on the same machine with no GPU
throttling detected, so compare step counts against each other rather than against totals
from another run.
