# Minimal Qwen3.6 Metal inference

This branch is an educational extraction of the production Qwen3.6 27B path.
It keeps the mmap GGUF loader, Qwen3.6 NextN speculative decoding, and the
existing optimized Metal backend. A small persistent chat loop and serial
OpenAI-compatible server are retained; the general server/agent framework,
distributed runtimes, other GPU backends, other model families, and the
general-purpose engine API are removed.

The intended reading order is:

1. `qwen36.c`: GGUF parsing, tokenizer, persistent target/MTP sessions,
   prefill/speculative-decode scheduling, and the small CLI.
2. `qwen_frontend.m`: the minimal HTTP and OpenAI JSON boundary.
3. `ds4_gpu.h`: the C interface exposed by the Metal backend.
4. `ds4_metal.m`: Metal buffers, command encoding, pipeline selection, and
   model mmap residency.
5. `metal/qwen35.metal`: Qwen full-attention and Gated DeltaNet kernels.
6. Shared kernels in `metal/dense.metal`, `metal/flash_attn.metal`,
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
  -> sampled token ───────────────┐
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
- Interactive turns append only their new chat-template suffix to that state;
  the HTTP server reuses the same allocated graph between serial requests.
- The optional Qwen3.6 NextN model drafts one token ahead; the target verifies
  a short suffix in one multi-row graph dispatch.
- Temperature zero retains GPU top-k equality verification. Positive
  temperatures use exact speculative acceptance/rejection, including sampling
  rejected proposals from the `(target - draft)+` residual distribution.
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

Enable greedy speculative decoding with the matching NextN GGUF. Supplying
`--mtp` selects the measured three-row verifier and margin 3 by default.
`--mtp-draft 1` means ordinary decoding, while values 2 through 16 override
the maximum verifier width.

```sh
./ds4 \
  -m gguf/Qwen3.6-27B-Q8_0.gguf \
  --mtp gguf/mtp-Qwen3.6-27B-Q8_0.gguf \
  -p 'Reply with exactly: hello' \
  -n 32 -c 512 --temp 0 --nothink
```

Metal is the only backend. `--temp 0` selects the unchanged greedy fast path;
positive values enable temperature sampling, and `--seed` makes a run
reproducible. The same sampler works with or without `--mtp`.

```sh
./ds4 \
  -m gguf/Qwen3.6-27B-Q8_0.gguf \
  --mtp gguf/mtp-Qwen3.6-27B-Q8_0.gguf \
  -p 'Write one sentence about the moon' \
  -n 32 -c 512 --temp 0.8 --seed 123 --nothink
```

### Interactive session

Omit `-p` (or add `--interactive`) to keep the model, Metal graph, KV cache,
and Gated DeltaNet state resident across turns. `/reset` starts a new context
and `/quit` exits. Speculative decoding uses the same options as one-shot mode.

```sh
./ds4 \
  -m gguf/Qwen3.6-27B-Q8_0.gguf \
  --mtp gguf/mtp-Qwen3.6-27B-Q8_0.gguf \
  -n 256 -c 32768 --nothink
```

### Inference server

`make` also creates `ds4-server`, a symlink to the same binary. The compact
server keeps one graph resident and processes requests serially, which keeps
the state machine visible and peak single-request Qwen performance unchanged.
It implements health/model discovery plus OpenAI non-streaming and SSE APIs at
`/v1/chat/completions` and `/v1/completions`.

```sh
./ds4-server \
  -m gguf/Qwen3.6-27B-Q8_0.gguf \
  --mtp gguf/mtp-Qwen3.6-27B-Q8_0.gguf \
  -n 256 -c 32768 --host 127.0.0.1 --port 8000

curl http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"messages":[{"role":"user","content":"Hello"}],"temperature":0.8,"seed":123,"stream":true}'
```

Useful diagnostic/ablation variables are:

- `DS4_METAL_MEMORY_REPORT=1`: print Metal memory accounting.
- `DS4_TOKEN_TIMING=1`: print per-token decode latency.
- `DS4_QWEN_PREFILL_CHUNK=N`: choose a prefill chunk from 1 through 1024.
- `DS4_QWEN_BATCHED_PREFILL=0`: disable batched prefill.
- `DS4_QWEN_GDN_CHUNKWISE=0`: disable chunkwise Gated DeltaNet prefill.
- `DS4_QWEN_FLASH_ATTN=0`: disable FlashAttention.
- `DS4_QWEN_PREFILL_FUSED_RESIDUAL_NORM=0`: disable the prefill fusion.
- `DS4_QWEN_DISABLE_FFN_MID_F16=1`: force the F32 FFN intermediate.
- `DS4_QWEN_MTP_GPU_TOPK=0`: move draft/verification argmax selection to CPU.
- `DS4_QWEN_MTP_NO_SNAPSHOT=0`: restore recurrent-state snapshot copies.
- `DS4_QWEN_MTP_FUSED_CATCHUP=0`: split MTP KV catch-up across command buffers.
- `DS4_QWEN_MTP_WIDE_FRONTIERS=1`: retain the extra frontier needed by a
  four-row verifier.
- `DS4_QWEN_MTP_STATS=1`: print proposal and acceptance counters.

The prefill switches use their fast settings by default. GPU top-k, recurrent
snapshot elision, and fused KV catch-up are also enabled by default for MTP;
set their variables to `0` for baseline ablations. Wide frontiers remain
opt-in because the default three-row verifier does not need them.

## Tests

```sh
make test
DS4_TEST_MODEL=gguf/Qwen3.6-27B-Q8_0.gguf make test
DS4_TEST_MODEL=gguf/Qwen3.6-27B-Q8_0.gguf \
DS4_TEST_MTP=gguf/mtp-Qwen3.6-27B-Q8_0.gguf make test
```

The first command checks the CLI without loading a model. The live-model test
checks greedy output and fixed-seed temperature sampling; setting
`DS4_TEST_MTP` exercises the exact speculative sampler too.

## What was cut

The original default macOS CLI compiled 159023 implementation lines: 132228
lines of C/Objective-C plus 26795 lines of runtime-compiled Metal. This branch
compiles 16549: 4628 in the Qwen host and frontends, 5808 in the Qwen-only Metal
backend, and 6113 in its nine runtime-compiled shader files. The public GPU
header is another 442 lines, down from 3196.

| Surface | Before | After | Removed |
|---|---:|---:|---:|
| Main host engine + frontends | 70897 | 4628 | 66269 (93.5%) |
| Metal host backend | 44603 | 5808 | 38795 (87.0%) |
| Runtime Metal shaders | 26795 | 6113 | 20682 (77.2%) |
| Compiled implementation | 159023 | 16549 | 142474 (89.6%) |

The repository also loses millions of lines of generated experiment fixtures,
but those are not counted as inference-engine simplification.

Removed features include DeepSeek and GLM execution, CUDA/ROCm/CPU backends,
the general concurrent server and agent frontends, SSD streaming, tensor
parallelism, distributed execution, eval/benchmark products, and optional
support-model speculative decoding for other model families. The retained Qwen
target and NextN paths use the same hot graphs and Metal kernels as the original.

Final validation used the saved pre-prune binary and this branch on an Apple M4
Pro. Every comparison used the same Q8_0 files and peak MTP settings; measured
ordinary and speculative outputs were identical.

| Gate | Pre-prune | Minimal | Change |
|---|---:|---:|---:|
| Long-prompt prefill, 5-run median | 66.08 tok/s | 71.08 tok/s | +7.6% |
| Ordinary generation, 3 alternating-run median | 5.48 tok/s | 5.63 tok/s | +2.7% |
| MTP total cycle, 37-cycle median | 395.087 ms | 373.319 ms | -5.5% |

An additional alternating A/B after restoring the frontends compared this
branch with its immediately preceding commit: ordinary generation differed by
-0.5% and MTP generation by +0.9%, both within run-to-run noise.

The absolute throughput varied with mmap residency pressure, so the decode
gate also compares the per-cycle medians within a longer run. The reduced
runtime did not lose peak prefill, ordinary decode, or speculative performance.

## Scope

This is a focused inference example, not a drop-in replacement for the full
product. It supports one-shot and persistent chat, a serial OpenAI-compatible
HTTP boundary, Qwen's think/no-think prefix, greedy or temperature-sampled
streaming output, one resident Metal target, and an optional matching NextN
support model. That
narrow contract keeps the control flow readable without replacing optimized
kernels with toy implementations.
