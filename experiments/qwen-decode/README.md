# Qwen3.6 decode experiments

This directory records the Qwen3.6-27B Q8_0 Metal decode campaign on the
Apple M4 Pro. Run only one model process and one live session at a time. Keep
`DS4_QWEN_MEMORY_GUARD` enabled.

## Retained baseline and fusion bundle

The bounded workload uses 128 forced prefix tokens, 8 warmup tokens, 64 timed
tokens, context 224, greedy selection, and the serial-session benchmark. Its
planned footprint is 26.92 GiB: 26.62 GiB resident model, 0.15 GiB recurrent
state, 0.01 GiB attention KV, and 0.14 GiB buffers.

The existing `DS4_METAL_QWEN_DECODE_FUSIONS=1` bundle combines decode FFN
SwiGLU, cross-layer residual/RMS normalization, and GDN parameter transforms.
The retained pre-continuation order-balanced pair was bit-identical over 72
rows (17,879,040 logits per run):

- Control-first: 7.422975 s control, 7.394530 s candidate.
- Candidate-first: 7.640135 s control, 7.375310 s candidate.
- Mean elapsed time fell from 7.531555 s to 7.384920 s, about 1.95%.

The bundle remains opt-in. Individual effects are small enough that longer or
independent-process confirmation is required before making it the default.

## Rejected continuation experiments

Two additional memory-neutral dispatch fusions were implemented, tested, and
then removed on 2026-08-10:

1. GDN convolution + parameter + Q/K-normalization preparation fusion was
   bit-identical through a 272-row control-first run (67,543,040 compared
   logits), but its 256-token timing was neutral: 29.875545 s control versus
   29.930143 s candidate (-0.18%). Four shorter 64-token trials averaged a
   noisy +0.46%. The added kernel was not retained.
2. A pair-of-pairs Q8 projection grid combined QKV/gate and alpha/beta regions
   into one dispatch. Its focused synthetic kernel test was exact, but a
   256-token full-model run diverged at step 93. The kernel, host hook, flag,
   and test were removed.

Neither experiment increased persistent allocation. Context 416 planned
27.05 GiB during the long trials, and all runs exited normally without OOM.

## Correctness-oracle limit

Repeated long runs inside one process are not a reliable Qwen correctness
oracle yet. A no-op environment comparison also produced a full-logit mismatch
after replay/reset (step 15 in a 128-token reverse-order run); an earlier
session-recreating no-op run diverged at step 23 of a 96-token window. This is
independent of the rejected kernels.

Use the 64-token exact A/B window for local dispatch experiments. For longer
correctness validation, compare fresh-process outputs against a retained
golden artifact rather than interpreting an in-process replay mismatch as a
candidate regression.

## Memory-safe command

```sh
DS4_QWEN_MEMORY_GUARD=1 \
./speed-bench/metal_decode_schedule_bench \
  -m gguf/Qwen3.6-27B-Q8_0.gguf \
  --prompt-file ds4.c \
  --prefix-tokens 128 \
  --ctx 224 \
  --warmup 8 \
  --tokens 64 \
  --candidate-env DS4_METAL_QWEN_DECODE_FUSIONS \
  --include-selection
```

Run the same command with `--serial-reverse` to balance order. Do not add
`--parallel-sessions` for this model.
