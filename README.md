# Minimal Qwen3.6 Metal inference

This branch is an educational extraction of the production Qwen3.6 27B path.
It keeps the mmap GGUF loader and the existing optimized Metal backend, but
removes the server, agent, distributed runtimes, other GPU backends, other
model families, and the general-purpose engine API.

The intended reading order is:

1. `qwen36.c`: GGUF parsing, fixed-shape validation, tokenizer, graph storage,
   prefill/decode scheduling, and the small CLI.
2. `ds4_gpu.h`: the C interface exposed by the Metal backend.
3. `ds4_metal.m`: Metal buffers, command encoding, pipeline selection, and
   model mmap residency.
4. `metal/qwen35.metal`: Qwen full-attention and Gated DeltaNet kernels.
5. Shared kernels in `metal/dense.metal`, `metal/flash_attn.metal`,
   `metal/norm.metal`, and `metal/glu.metal`.

## The inference path

```text
Q8_0 GGUF
  -> mmap + metadata/tensor directory
  -> Qwen chat-template tokenization
  -> persistent Metal graph
  -> 64 transformer blocks
       [Gated DeltaNet, Gated DeltaNet, Gated DeltaNet, full attention] x 16
  -> RMSNorm + Q8_0 output projection
  -> greedy token
```

The runtime deliberately validates one shape instead of implementing a model
configuration framework: 64 layers, width 5120, 24 query heads, 4 KV heads,
head dimension 256, dense FFN width 17408, and a 262144-token model context.
Unexpected GGUF metadata or tensor layouts fail early.

## Performance-critical behavior retained

- The model file is memory-mapped and exposed to Metal without copying every
  weight into a second host allocation.
- Prefill is batched in chunks of up to 1024 tokens.
- Gated DeltaNet prefill uses the 64-row chunkwise WY path when eligible.
- Full-attention layers use exact FlashAttention by default.
- Prefill keeps fused residual-add/RMSNorm enabled.
- Pre-M5 Apple Silicon uses an F16 FFN intermediate to reduce bandwidth.
- Decode keeps persistent KV, convolution, and recurrent SSM state on GPU.
- The original Metal implementation and shader set are byte-for-byte retained,
  including Q8_0 matmuls and the optional decode-fusion experiments.

Keeping all Metal sources is intentional. `ds4_metal.m` builds a shared kernel
library and eagerly creates a broad pipeline set during initialization; pruning
that substrate would introduce a separate performance and correctness risk.
This branch minimizes the host semantics around it instead.

## Build and run

Requirements: macOS, Apple Silicon, Clang, and the Metal framework.

```sh
make -j

DS4_LOG=timing ./ds4 \
  -m gguf/Qwen3.6-27B-Q8_0.gguf \
  -p 'Reply with exactly: hello' \
  -n 8 -c 64 --temp 0 --nothink
```

Metal and greedy decoding are the only backend and sampler. `--metal` and
`--temp 0` remain accepted so existing baseline commands still work.

Useful diagnostic/ablation variables are:

- `DS4_METAL_MEMORY_REPORT=1`: print Metal memory accounting.
- `DS4_TOKEN_TIMING=1`: print per-token decode latency.
- `DS4_QWEN_PREFILL_CHUNK=N`: choose a prefill chunk from 1 through 1024.
- `DS4_QWEN_BATCHED_PREFILL=0`: disable batched prefill.
- `DS4_QWEN_GDN_CHUNKWISE=0`: disable chunkwise Gated DeltaNet prefill.
- `DS4_QWEN_FLASH_ATTN=0`: disable FlashAttention.
- `DS4_QWEN_PREFILL_FUSED_RESIDUAL_NORM=0`: disable the prefill fusion.
- `DS4_QWEN_DISABLE_FFN_MID_F16=1`: force the F32 FFN intermediate.

The last five switches are for studying optimizations; their fast defaults are
the normal configuration.

## Tests

```sh
make test
DS4_TEST_MODEL=gguf/Qwen3.6-27B-Q8_0.gguf make test
```

The first command checks the CLI without loading a model. The second also runs
a deterministic live-model smoke test.

## What was cut

The original default macOS CLI compiled 159023 implementation lines: 132228
lines of C/Objective-C plus 26795 lines of runtime-compiled Metal. This branch
compiles 74083: 2685 in the Qwen-only host, the unchanged 44603-line Metal
backend, and the same 26795 shader lines.

| Surface | Before | After | Removed |
|---|---:|---:|---:|
| Main host engine | 70897 | 2685 | 68212 (96.2%) |
| Compiled implementation | 159023 | 74083 | 84940 (53.4%) |

The repository also loses millions of lines of generated experiment fixtures,
but those are not counted as inference-engine simplification.

Removed features include DeepSeek and GLM execution, CUDA/ROCm/CPU backends,
server and agent frontends, SSD streaming, tensor parallelism, distributed
execution, eval/benchmark products, and optional support-model speculative
decoding. The retained single-model Qwen path has the same hot graph and Metal
kernels as the original.

On an Apple M4 Pro with `Qwen3.6-27B-Q8_0.gguf`, three alternating 16-token
runs gave the following means. Decode latency contains 45 timed graph
evaluations per runtime.

| Runtime | Prefill | Generation | Decode evaluation |
|---|---:|---:|---:|
| Original CLI | 73.30 tok/s | 9.41 tok/s | 113.388 ms |
| Minimal CLI | 74.13 tok/s | 9.45 tok/s | 112.827 ms |

The difference is measurement noise; the generated-token path is unchanged.

## Scope

This is a focused inference example, not a drop-in replacement for the full
product. It supports one prompt, Qwen's think/no-think chat prefix, greedy
streaming output, and one resident Metal model. That narrow contract is what
makes the control flow readable without replacing optimized kernels with toy
implementations.
