# Qwen3.6 prefill experiments

This directory records the local Qwen3.6-27B Q8_0 Metal optimization campaign.
Run only one model process at a time. The engine lock and Qwen memory guard must
remain enabled for every benchmark.

## Retained baseline

- Workload: `BENCHMARK_PROMPT.txt`, 671 raw tokens, context 1024, one greedy
  output token.
- Default path: 1024-token layer-major prefill, token-wise Gated DeltaNet,
  grouped-query FlashAttention, and fused residual-add plus RMSNorm.
- Final retained trials: 129.47, 129.41, and 129.37 tokens/s.
- Resident model: 26.62 GiB. Default context/runtime estimate at context 1024:
  0.85 GiB.
- The Q8 projection profile measured 5.20 seconds in 496 dense projections;
  these projections dominate the roughly 5.2-second prefill.

The agent system-prompt snapshot is also retained. The first 1821-token system
prefill took about 14.1 seconds and wrote the Qwen KV/recurrent snapshot. A warm
snapshot hit reduced total agent startup to 1.41 seconds.

## Fused residual + RMSNorm experiment

The default prefill path uses a row-batched form of the existing residual-add
plus RMSNorm kernel. It prepares the FFN norm and the next layer's attention
norm in fused dispatches, removing 127 Metal dispatches over 64 layers without
allocating another buffer. Set `DS4_QWEN_PREFILL_FUSED_RESIDUAL_NORM=0` to use
the unfused fallback.

The guarded 671-token Metal validation completed on 2026-08-10. The raw prompt
must be passed with `--raw` and without the file's trailing newline to reproduce
the retained token stream.

- Correctness: all 248320 logits were bit-identical to
  `results/raw671-flash-attn-logits.json`; both paths selected token 198.
- Fused prefill trials: 121.94, 122.36, and 121.94 tokens/s; median 121.94.
- Contemporaneous default trials: 121.92, 121.89, and 122.05 tokens/s; median
  121.92.
- Fused wall times: 5.80, 5.81, and 5.84 seconds; median 5.81. Default wall
  times: 5.83, 5.85, and 5.83 seconds; median 5.83.
- Both paths retained the same 27.47 GiB planned memory estimate and completed
  without swaps.

The 0.02 tokens/s median prefill difference is below the run-to-run noise. The
fusion is nevertheless the retained default because it is bit-identical,
memory-neutral, and eliminates redundant dispatches; the unfused path remains
available for diagnostics.

## Dense Q8 projection kernel experiment

Dense Q8 prefill now defaults to eight aligned 16-bit Q8 loads per dequantized
half tile instead of sixteen byte loads. The arithmetic and half staging are
unchanged. Set `DS4_METAL_Q8_PREFILL_VARIANT=legacy` to select the old byte-load
kernel, or `pairs64` to reproduce the rejected 64-token/256-thread tile.

The guarded experiment completed on 2026-08-10:

- Synthetic boundary exactness used 95 tokens; the default, explicit
  `pairs32`, and `pairs64` variants were bit-identical to the legacy kernel.
- The realistic 5120x17408x695 projection test compared 12,098,560 outputs;
  both candidates were bit-identical. Peak footprint was 378.7 MB with zero
  swaps. The final retained-harness medians were 18.858 ms for legacy and
  18.478 ms for paired loads (+2.06% throughput); the 64-token tile took
  19.152 ms.
- Two full-model projection profiles measured all 496 calls per variant.
  Legacy totals were 5237.182 and 5145.383 ms; paired-load totals were 5087.362
  and 5103.399 ms. Average Q8 time fell from 5191.283 to 5095.381 ms, a 1.85%
  time reduction.
- The first three-way end-to-end set was noisy: legacy median 120.34 tokens/s,
  paired-load median 120.30, and 64-token median 117.19. A balanced legacy vs
  paired-load confirmation measured 123.44 versus 125.93 tokens/s (+2.02%).
- The 64-token tile was rejected: its 256-thread group lost enough occupancy to
  make end-to-end prefill 2.62% slower in the three-way set despite reducing
  repeated weight reads.
- Full-model default and `pairs64` logits were each byte-identical to the
  retained 248,320-logit golden file. The memory guard reported 27.72 GiB
  required against a 33.70 GiB budget; both runs completed with zero swaps.

The paired-load kernel is retained because the projection-level measurement is
repeatable, it is bit-identical and memory-neutral, and the end-to-end
confirmation is positive. The larger token tile remains opt-in only for future
occupancy work.

## Memory safety

- Do not set `DS4_QWEN_MEMORY_GUARD=0` during experiments.
- Do not run CPU and Metal baselines concurrently.
- Keep decode A/B runs in the benchmark's default serial-session mode. The
  two-live-session path now requires an explicit `--parallel-sessions` opt-in.
- Do not opt into chunkwise Gated DeltaNet above 16 tokens; the code caps it to
  avoid quadratic scratch growth.
- Rejected CPU configurations are checked before the CPU `WILLNEED` prefetch,
  so an oversized context cannot pull the entire GGUF into the page cache before
  the guard refuses it.
- `tests/test_engine_mgpu_placement` pins both FlashAttention and non-flash
  Qwen scratch accounting.
