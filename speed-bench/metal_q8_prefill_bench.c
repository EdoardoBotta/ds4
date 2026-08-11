#define _DARWIN_C_SOURCE

#include "ds4_gpu.h"

#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

enum {
    IN_DIM = 5120,
    OUT_DIM = 17408,
    N_TOK = 695,
    TIMED_RUNS = 9,
};

static const char *const variants[] = {"baseline", "pairs32", "pairs64"};

bool ds4_log_is_tty(FILE *fp) {
    (void)fp;
    return false;
}

static uint64_t round_up(uint64_t value, uint64_t alignment) {
    return (value + alignment - 1u) / alignment * alignment;
}

static double now_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec * 1000.0 + (double)ts.tv_nsec / 1.0e6;
}

static int compare_double(const void *a, const void *b) {
    const double da = *(const double *)a;
    const double db = *(const double *)b;
    return (da > db) - (da < db);
}

static int select_variant(size_t variant) {
    return variant == 0
        ? setenv("DS4_METAL_Q8_PREFILL_VARIANT", "legacy", 1)
        : setenv("DS4_METAL_Q8_PREFILL_VARIANT", variants[variant], 1);
}

static void fill_q8_weights(uint8_t *weights, uint64_t row_bytes) {
    const uint32_t blocks = IN_DIM / 32u;
    for (uint32_t row = 0; row < OUT_DIM; row++) {
        uint8_t *dst = weights + (uint64_t)row * row_bytes;
        for (uint32_t block = 0; block < blocks; block++) {
            /* IEEE half 1/256, followed by a deterministic signed Q8 block. */
            dst[0] = 0x00;
            dst[1] = 0x1c;
            for (uint32_t i = 0; i < 32u; i++) {
                dst[2u + i] = (uint8_t)(int8_t)(
                    (int)((row * 17u + block * 13u + i * 7u) % 127u) - 63);
            }
            dst += 34u;
        }
    }
}

int main(void) {
    const uint64_t page = (uint64_t)getpagesize();
    const uint64_t row_bytes = (uint64_t)(IN_DIM / 32u) * 34u;
    const uint64_t weight_bytes = (uint64_t)OUT_DIM * row_bytes;
    const uint64_t model_bytes = round_up(weight_bytes, page);
    const uint64_t x_count = (uint64_t)N_TOK * IN_DIM;
    const uint64_t out_count = (uint64_t)N_TOK * OUT_DIM;
    const uint64_t x_bytes = x_count * sizeof(float);
    const uint64_t out_bytes = out_count * sizeof(float);

    fprintf(stderr,
            "Q8 prefill bench shape K=%u M=%u N=%u model=%.2f MiB scratch=%.2f MiB\n",
            IN_DIM, OUT_DIM, N_TOK, (double)model_bytes / (1024.0 * 1024.0),
            (double)(x_bytes + out_bytes) / (1024.0 * 1024.0));

    void *model = NULL;
    float *x_host = NULL;
    float *baseline = NULL;
    float *candidate = NULL;
    ds4_gpu_tensor *x = NULL;
    ds4_gpu_tensor *out = NULL;
    int ok = posix_memalign(&model, (size_t)page, (size_t)model_bytes) == 0;
    x_host = malloc((size_t)x_bytes);
    baseline = malloc((size_t)out_bytes);
    candidate = malloc((size_t)out_bytes);
    ok = ok && model && x_host && baseline && candidate;
    if (!ok) {
        fprintf(stderr, "Q8 prefill bench host allocation failed\n");
        goto cleanup;
    }

    memset(model, 0, (size_t)model_bytes);
    fill_q8_weights(model, row_bytes);
    for (uint64_t i = 0; i < x_count; i++) {
        x_host[i] = (float)((int)((i * 19u + (i >> 5u) * 3u) % 97u) - 48) /
                      64.0f;
    }

    ok = ds4_gpu_init() && ds4_gpu_set_model_map(model, model_bytes);
    x = ds4_gpu_tensor_alloc(x_bytes);
    out = ds4_gpu_tensor_alloc(out_bytes);
    ok = ok && x && out && ds4_gpu_tensor_write(x, 0, x_host, x_bytes);
    if (!ok) {
        fprintf(stderr, "Q8 prefill bench Metal setup failed\n");
        goto cleanup;
    }

    ds4_gpu_set_quality(false);
    for (size_t variant = 0; variant < 3u; variant++) {
        ok = select_variant(variant) == 0 &&
             ds4_gpu_matmul_q8_0_tensor(out, model, model_bytes, 0,
                                        IN_DIM, OUT_DIM, x, N_TOK);
        if (!ok) {
            fprintf(stderr, "Q8 prefill bench warmup failed for %s\n",
                    variants[variant]);
            goto cleanup;
        }
    }

    ok = select_variant(0) == 0 &&
         ds4_gpu_matmul_q8_0_tensor(out, model, model_bytes, 0,
                                    IN_DIM, OUT_DIM, x, N_TOK) &&
         ds4_gpu_tensor_read(out, 0, baseline, out_bytes);
    for (size_t variant = 1; ok && variant < 3u; variant++) {
        ok = select_variant(variant) == 0 &&
             ds4_gpu_matmul_q8_0_tensor(out, model, model_bytes, 0,
                                        IN_DIM, OUT_DIM, x, N_TOK) &&
             ds4_gpu_tensor_read(out, 0, candidate, out_bytes);
        if (ok && memcmp(baseline, candidate, (size_t)out_bytes) != 0) {
            uint64_t mismatches = 0;
            for (uint64_t i = 0; i < out_count; i++) {
                mismatches += memcmp(&baseline[i], &candidate[i], sizeof(float)) != 0;
            }
            fprintf(stderr, "Q8 prefill bench %s mismatch=%llu/%llu\n",
                    variants[variant], (unsigned long long)mismatches,
                    (unsigned long long)out_count);
            ok = 0;
        } else if (ok) {
            fprintf(stderr, "Q8 prefill bench %s bit-exact over %llu outputs\n",
                    variants[variant], (unsigned long long)out_count);
        }
    }
    if (!ok) goto cleanup;

    double elapsed[3][TIMED_RUNS] = {{0}};
    size_t counts[3] = {0};
    for (size_t round = 0; round < TIMED_RUNS; round++) {
        const size_t order[3] = {
            round % 2u == 0 ? 0u : 2u,
            1u,
            round % 2u == 0 ? 2u : 0u,
        };
        for (size_t oi = 0; oi < 3u; oi++) {
            const size_t variant = order[oi];
            ok = select_variant(variant) == 0;
            const double t0 = now_ms();
            ok = ok && ds4_gpu_matmul_q8_0_tensor(
                           out, model, model_bytes, 0,
                           IN_DIM, OUT_DIM, x, N_TOK);
            const double dt = now_ms() - t0;
            if (!ok) {
                fprintf(stderr, "Q8 prefill bench timed run failed for %s\n",
                        variants[variant]);
                goto cleanup;
            }
            elapsed[variant][counts[variant]++] = dt;
        }
    }

    double medians[3];
    for (size_t variant = 0; variant < 3u; variant++) {
        qsort(elapsed[variant], counts[variant], sizeof(double), compare_double);
        medians[variant] = elapsed[variant][counts[variant] / 2u];
        fprintf(stderr, "Q8 prefill bench %-8s median=%7.3f ms runs=%zu\n",
                variants[variant], medians[variant], counts[variant]);
    }
    fprintf(stderr,
            "Q8 prefill bench speedup pairs32=%+.2f%% pairs64=%+.2f%%\n",
            100.0 * (medians[0] / medians[1] - 1.0),
            100.0 * (medians[0] / medians[2] - 1.0));

cleanup:
    unsetenv("DS4_METAL_Q8_PREFILL_VARIANT");
    ds4_gpu_tensor_free(out);
    ds4_gpu_tensor_free(x);
    ds4_gpu_cleanup();
    free(candidate);
    free(baseline);
    free(x_host);
    free(model);
    return ok ? 0 : 1;
}
