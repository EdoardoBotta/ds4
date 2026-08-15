# Development notes

This branch is intentionally limited to Qwen3.6 27B Q8_0 inference on Apple
Metal. Keep changes within that scope.

## Performance contract

- `ds4_metal.m` and `metal/*.metal` are the production performance substrate.
- Do not replace GPU operations with CPU reference code.
- Keep the model mmap-backed and pass its tensor region to Metal as no-copy
  shared buffers.
- Preserve the default 1024-token batched prefill, chunkwise Gated DeltaNet,
  exact FlashAttention, fused residual/RMSNorm, Q8_0 matmuls, and the pre-M5
  F16 FFN intermediate.
- Treat a measurable prefill-throughput or decode-latency regression as a bug.

## Verification

Run:

```sh
make clean && make -j
make test
DS4_TEST_MODEL=gguf/Qwen3.6-27B-Q8_0.gguf make test
```

For performance changes, alternate the candidate and a baseline binary on the
same prompt and model. Use multiple warm runs and compare medians; model
residency and thermal state can otherwise dominate small differences.

Only one model process should run at once. The CLI enforces this with
`/tmp/ds4.lock` because a second mapping can create severe memory pressure.
