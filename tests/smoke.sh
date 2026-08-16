#!/bin/sh
set -eu

./ds4 --help >/dev/null 2>&1
./ds4-server --help >/dev/null 2>&1

if [ ! -L ./ds4-server ]; then
    echo "smoke: ds4-server compatibility entry point is missing" >&2
    exit 1
fi

./ds4 --temp 0.7 --seed 1 --help >/dev/null 2>&1

if ./ds4 --temp -1 --help >/dev/null 2>&1; then
    echo "smoke: negative temperature was unexpectedly accepted" >&2
    exit 1
fi

if ./ds4 --seed -1 --help >/dev/null 2>&1; then
    echo "smoke: negative seed was unexpectedly accepted" >&2
    exit 1
fi

if ./ds4 --mtp-draft 2 -m unused.gguf -p unused >/dev/null 2>&1; then
    echo "smoke: speculative decoding was accepted without --mtp" >&2
    exit 1
fi

if ./ds4 --server -m unused.gguf -p unused >/dev/null 2>&1; then
    echo "smoke: server mode unexpectedly accepted a one-shot prompt" >&2
    exit 1
fi

if [ -n "${DS4_TEST_MODEL:-}" ]; then
    set -- -m "$DS4_TEST_MODEL" \
        -p "Reply with exactly: hello" \
        -n 8 -c 64 --temp 0 --nothink
    if [ -n "${DS4_TEST_MTP:-}" ]; then
        set -- "$@" --mtp "$DS4_TEST_MTP"
    fi
    output=$(DS4_LOG=error ./ds4 "$@")
    case "$output" in
        *hello*) ;;
        *)
            echo "smoke: expected model output to contain 'hello', got: $output" >&2
            exit 1
            ;;
    esac

    set -- -m "$DS4_TEST_MODEL" \
        -p "Write a very short greeting" \
        -n 4 -c 64 --temp 0.8 --seed 123 --nothink
    if [ -n "${DS4_TEST_MTP:-}" ]; then
        set -- "$@" --mtp "$DS4_TEST_MTP"
    fi
    sampled_a=$(DS4_LOG=error ./ds4 "$@")
    sampled_b=$(DS4_LOG=error ./ds4 "$@")
    if [ -z "$sampled_a" ] || [ "$sampled_a" != "$sampled_b" ]; then
        echo "smoke: fixed-seed temperature sampling is not reproducible" >&2
        exit 1
    fi
fi

echo "smoke: ok"
