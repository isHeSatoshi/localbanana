# Modality Profiles

Use this reference when the workload is not only autoregressive text. Measure the first useful user outcome and profile every pipeline stage.

| Modality | First useful outcome | Main planning shape | Quality gate |
| --- | --- | --- | --- |
| VLM | First correct text result | Visual tokens, text context, prefill, KV cache, sampled frames | Task accuracy across resolution and frame rate |
| Embeddings | Retrieved or ranked result | Separate interactive lookups from bulk backfills; tokenization, batching, queue, vector store | Recall@k, nDCG, MRR, and product relevance |
| ASR | Useful partial or final transcript | Live chunks and sequential context versus VAD-parallelized files | WER/CER, timestamp quality, hallucination, diarization error |
| TTS | First intelligible phrase | Token model, waveform decoder, streaming, concurrent stable sessions | Intelligibility, preference, voice identity, prosody |
| Image | Acceptable image | Resolution, denoising steps, passes, kernels, precision | Human preference, prompt and reference fidelity, defect rate |
| Video | Acceptable motion output | Resolution, duration, frame rate, steps, attention, communication, batch-one node use | Temporal consistency, motion, identity persistence, artifacts |

## Rules

- Use a general multimodal model only when the product needs joint reasoning. Use specialized OCR, PDF extraction, ASR, diarization, or codecs for narrow correctness targets.
- VLM performance is often an input-shape problem. Segment media decode, vision encoding, prefill, and text decode. Treat resolution and frame sampling as quality controls.
- Run embedding backfills and interactive lookups as distinct capacity profiles. A new embedding model requires re-embedding and re-indexing. Validate quantization with downstream retrieval, not cosine similarity alone.
- For live ASR, measure microphone-to-partial and microphone-to-final latency. For files, report audio-seconds processed per wall-second and include VAD, stitching, retry, and diarization.
- For TTS, optimize time to first intelligible phrase, then stable real-time concurrency. Benchmark the token model and waveform decoder separately.
- Image and video speed controls can change visible quality. Freeze prompts, seeds or seed policy, size, scheduler, steps, guidance, precision, and hardware before comparison.
- Video often uses one request across several devices. Measure wall seconds and node cost per generated second, attention share, communication, cache reuse, and temporal quality.

Do not compare cross-modality token rates as if they represented the same work.

