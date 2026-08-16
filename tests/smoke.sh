#!/bin/sh
set -eu

./ds4 --help >/dev/null 2>&1

if ./ds4 --temp 0.7 -m unused.gguf -p unused >/dev/null 2>&1; then
    echo "smoke: non-greedy temperature was unexpectedly accepted" >&2
    exit 1
fi

if ./ds4 --mtp-draft 2 -m unused.gguf -p unused >/dev/null 2>&1; then
    echo "smoke: speculative decoding was accepted without --mtp" >&2
    exit 1
fi

if [ -n "${DS4_TEST_MODEL:-}" ]; then
    set -- -m "$DS4_TEST_MODEL" \
        -p "Reply with exactly: hello" \
        -n 8 -c 64 --temp 0 --nothink
    if [ -n "${DS4_TEST_MTP:-}" ]; then
        set -- "$@" --mtp "$DS4_TEST_MTP" --mtp-draft 3
    fi
    output=$(DS4_LOG=error ./ds4 "$@")
    case "$output" in
        *hello*) ;;
        *)
            echo "smoke: expected model output to contain 'hello', got: $output" >&2
            exit 1
            ;;
    esac
fi

echo "smoke: ok"
