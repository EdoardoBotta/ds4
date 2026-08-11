#!/bin/zsh
set -eu
setopt pipefail

ROOT=${0:A:h:h:h}
OUT=${OUT:-$ROOT/experiments/ssd-cache-policy/results}
MODEL=${MODEL:-$ROOT/gguf/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix.gguf}
PROMPT=${PROMPT:-$ROOT/speed-bench/promessi_sposi.txt}
TRACE=$OUT/promessi_2048_gen128_selected_ids.bin
BENCH_CSV=$OUT/promessi_2048_gen128_bench.csv
LOG=$OUT/promessi_2048_gen128.log

mkdir -p "$OUT"

env \
  DS4_MOE_RECORD_SELECTED_IDS="$TRACE" \
  DS4_METAL_STREAMING_EXPERT_TIMING_SUMMARY=1 \
  "$ROOT/ds4-bench" \
    -m "$MODEL" \
    --prompt-file "$PROMPT" \
    --ctx-start 2048 \
    --ctx-max 2048 \
    --ctx-alloc 2177 \
    --gen-tokens 128 \
    --sample-mode non-eos-greedy \
    --ssd-streaming \
    --ssd-streaming-cold \
    --ssd-streaming-cache-experts 4096 \
    --csv "$BENCH_CSV" 2>&1 | tee "$LOG"

python3 "$ROOT/experiments/ssd-cache-policy/analyze_cache_policy.py" \
  "$TRACE" \
  --layers 43 \
  --experts 256 \
  --selected 6 \
  --expert-bytes 7077888 \
  --cache-gib 1,2,4,8,12,16,24,27,32,40,48,56,64,72,73 \
  --csv "$OUT/cache_policy_results.csv" \
  --svg "$OUT/cache_policy_gap.svg" \
  --title "SSD streaming policy gap — Promessi sposi, ctx 2048, 128 decode tokens"
