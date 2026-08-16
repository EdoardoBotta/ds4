# Minimal Qwen3.6 Metal inference

This branch is an educational extraction of the production Qwen3.6 27B path.
It keeps the mmap GGUF loader, Qwen3.6 NextN speculative decoding, and the
existing optimized Metal backend, but removes the server, agent, distributed
runtimes, other GPU backends, other model families, and the general-purpose
engine API.

The intended reading order is:

1. `qwen36.c`: GGUF parsing, fixed-shape validation, tokenizer, target and MTP
   graph storage, prefill/speculative-decode scheduling, and the small CLI.
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
  -> greedy token ────────────────┐
                                  v
       NextN block -> draft suffix -> batched target verification
                                  -> accepted tokens
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
- The optional Qwen3.6 NextN model drafts one token ahead; the target verifies
  a short suffix in one multi-row graph dispatch.
- Speculative verification can perform top-k on GPU, elide the recurrent-state
  snapshot, fuse draft-cache catch-up, and retain accepted Gated DeltaNet
  frontiers without replaying target layers.
- The Qwen, dense, and FlashAttention kernel bodies are unchanged, including
  every Q8_0 matmul specialization and optional decode-fusion experiment.
- The Metal library contains only kernels reachable from the Qwen target and
  NextN graphs. Static and shape-specialized pipeline selection is preserved.

The backend reduction follows the graph rather than replacing it: trace the 43
GPU calls made by `qwen36.c`, retain their transitive Objective-C helpers, then
retain the Metal entry points those helpers load. That removes other model
families without simplifying Qwen's hot graph or its kernels.

## Build and run

Requirements: macOS, Apple Silicon, Clang, and the Metal framework.

```sh
make -j

DS4_LOG=timing ./ds4 \
  -m gguf/Qwen3.6-27B-Q8_0.gguf \
  -p 'Reply with exactly: hello' \
  -n 8 -c 64 --temp 0 --nothink
```

Enable greedy speculative decoding with the matching NextN GGUF. As in the
larger CLI, `--mtp-draft 1` means ordinary decoding and values 2 through 16
select the maximum verifier width.

```sh
DS4_QWEN_MTP_GPU_TOPK=1 \
DS4_QWEN_MTP_NO_SNAPSHOT=1 \
DS4_QWEN_MTP_FUSED_CATCHUP=1 \
./ds4 \
  -m gguf/Qwen3.6-27B-Q8_0.gguf \
  --mtp gguf/mtp-Qwen3.6-27B-Q8_0.gguf \
  --mtp-draft 3 --mtp-margin 3 \
  -p 'Reply with exactly: hello' \
  -n 32 -c 512 --temp 0 --nothink
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
- `DS4_QWEN_MTP_GPU_TOPK=1`: keep draft/verification argmax selection on GPU.
- `DS4_QWEN_MTP_NO_SNAPSHOT=1`: commit retained recurrent frontiers directly.
- `DS4_QWEN_MTP_FUSED_CATCHUP=1`: encode MTP KV catch-up in one command buffer.
- `DS4_QWEN_MTP_WIDE_FRONTIERS=1`: retain the extra frontier needed by a
  four-row verifier.
- `DS4_QWEN_MTP_STATS=1`: print proposal and acceptance counters.

The prefill switches are ablations whose fast settings are the defaults. The
MTP acceleration switches are explicit experiments in the larger runtime and
remain opt-in here so both binaries expose the same peak configuration.

## Tests

```sh
make test
DS4_TEST_MODEL=gguf/Qwen3.6-27B-Q8_0.gguf make test
DS4_TEST_MODEL=gguf/Qwen3.6-27B-Q8_0.gguf \
DS4_TEST_MTP=gguf/mtp-Qwen3.6-27B-Q8_0.gguf make test
```

The first command checks the CLI without loading a model. The second also runs
a deterministic live-model smoke test.

## What was cut

The original default macOS CLI compiled 159023 implementation lines: 132228
lines of C/Objective-C plus 26795 lines of runtime-compiled Metal. This branch
compiles 16211: 3798 in the Qwen host, 6094 in the Qwen-only Metal backend, and
6319 in its nine runtime-compiled shader files. The public GPU header is another
442 lines, down from 3196.

| Surface | Before | After | Removed |
|---|---:|---:|---:|
| Main host engine | 70897 | 3798 | 67099 (94.6%) |
| Metal host backend | 44603 | 6094 | 38509 (86.3%) |
| Runtime Metal shaders | 26795 | 6319 | 20476 (76.4%) |
| Compiled implementation | 159023 | 16211 | 142812 (89.8%) |

The repository also loses millions of lines of generated experiment fixtures,
but those are not counted as inference-engine simplification.

Removed features include DeepSeek and GLM execution, CUDA/ROCm/CPU backends,
server and agent frontends, SSD streaming, tensor parallelism, distributed
execution, eval/benchmark products, and optional support-model speculative
decoding for other model families. The retained Qwen target and NextN paths
use the same hot graphs and Metal kernels as the original.

Final validation used the saved pre-prune binary and this branch on an Apple M4
Pro. Every comparison used the same Q8_0 files and peak MTP settings; measured
ordinary and speculative outputs were identical.

| Gate | Pre-prune | Minimal | Change |
|---|---:|---:|---:|
| Long-prompt prefill, 5-run median | 66.08 tok/s | 71.08 tok/s | +7.6% |
| Ordinary generation, 3 alternating-run median | 5.48 tok/s | 5.63 tok/s | +2.7% |
| MTP total cycle, 37-cycle median | 395.087 ms | 373.319 ms | -5.5% |

The absolute throughput varied with mmap residency pressure, so the decode
gate also compares the per-cycle medians within a longer run. The reduced
runtime did not lose peak prefill, ordinary decode, or speculative performance.

## Scope

This is a focused inference example, not a drop-in replacement for the full
product. It supports one prompt, Qwen's think/no-think chat prefix, greedy
streaming output, one resident Metal target, and an optional matching NextN
support model. That narrow contract is what makes the control flow readable
without replacing optimized kernels with toy implementations.
