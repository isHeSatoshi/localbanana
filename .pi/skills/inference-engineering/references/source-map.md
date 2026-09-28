# Source Map

Checked 2026-08-27. Use this file to separate stable concepts from facts that must be refreshed.

## Source Book

- Philip Kiely, *Inference Engineering*, Baseten Books, 2026.
- Canonical page: https://www.baseten.co/inference-engineering/
- If the user supplies a local copy of the book, use it as the source artifact for page-specific analysis.
- The skill is an original operating method derived from the source. It does not reproduce the book, figures, or glossary.

## Durable Primary Sources

- Transformer attention: https://arxiv.org/abs/1706.03762
- FlashAttention: https://arxiv.org/abs/2205.14135
- FlashAttention-2: https://arxiv.org/abs/2307.08691
- PagedAttention and vLLM: https://arxiv.org/abs/2309.06180
- Speculative decoding: https://arxiv.org/abs/2211.17192
- Ring Attention: https://arxiv.org/abs/2310.01889
- Site Reliability Engineering: https://sre.google/books/

## Current Official Implementation Sources

Read the relevant source live before giving commands, compatibility claims, or rankings:

- CUDA guide: https://docs.nvidia.com/cuda/cuda-c-programming-guide/
- CUTLASS: https://github.com/NVIDIA/cutlass
- PyTorch Profiler: https://pytorch.org/tutorials/recipes/recipes/profiler_recipe.html
- Transformers: https://huggingface.co/docs/transformers/index
- Diffusers: https://huggingface.co/docs/diffusers/index
- vLLM: https://github.com/vllm-project/vllm
- SGLang: https://github.com/sgl-project/sglang
- TensorRT-LLM: https://github.com/NVIDIA/TensorRT-LLM
- NVIDIA Dynamo: https://docs.nvidia.com/dynamo/latest/index.html
- Kubernetes: https://kubernetes.io/docs/home/

## Always Refresh

- accelerator names, variants, memory, bandwidth, precision support, prices, regions, availability, and roadmaps;
- model capabilities, licenses, architecture variants, context limits, rankings, and provider status;
- runtime versions, flags, defaults, supported models, kernels, quantization formats, and interoperability;
- performance, quality, cost, reliability, uptime, speedup, and market-share claims;
- benchmark versions, datasets, harness commits, prompt templates, scoring, contamination status, and leaderboards.

Current documentation forms a shortlist. Only target-stack measurements under the user's workload and quality threshold make the final decision.

