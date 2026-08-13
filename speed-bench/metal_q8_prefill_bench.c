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
    MID_DIM = 17408,
    OUT_DIM = 5120,
    N_TOK = 1024,
    TIMED_RUNS = 31,
};

static const char *const variants[] = {
    "baseline", "mid_f16", "baseline_repeat"
};

bool ds4_log_is_tty(FILE *fp) {
    (void)fp;
    return false;
}

static uint64_t round_up(uint64_t value, uint64_t alignment) {
    return (value + alignment - 1u)/alignment*alignment;
}

static double now_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec*1000.0 + (double)ts.tv_nsec/1.0e6;
}

static int compare_double(const void *a, const void *b) {
    const double da = *(const double *)a;
    const double db = *(const double *)b;
    return (da > db) - (da < db);
}

static void fill_q8_weights(uint8_t *weights, uint32_t in_dim,
                            uint32_t out_dim, uint64_t row_bytes,
                            uint32_t salt) {
    const uint32_t blocks = in_dim/32u;
    for (uint32_t row = 0; row < out_dim; row++) {
        uint8_t *dst = weights + (uint64_t)row*row_bytes;
        for (uint32_t block = 0; block < blocks; block++) {
            dst[0] = 0x00;
            dst[1] = 0x1c; /* IEEE half 1/256. */
            for (uint32_t i = 0; i < 32u; i++) {
                dst[2u + i] = (uint8_t)(int8_t)(
                    (int)((row*17u + block*13u + i*7u + salt)%127u) - 63);
            }
            dst += 34u;
        }
    }
}

static int run_variant(size_t variant,
                       ds4_gpu_tensor *gate,
                       ds4_gpu_tensor *up,
                       ds4_gpu_tensor *mid_f32,
                       ds4_gpu_tensor *mid_f16,
                       ds4_gpu_tensor *out,
                       const void *model,
                       uint64_t model_bytes,
                       uint64_t up_offset,
                       uint64_t down_offset,
                       const ds4_gpu_tensor *x) {
    int ok = ds4_gpu_begin_commands();
    if (ok) ok = ds4_gpu_matmul_q8_0_tensor(
        gate, model, model_bytes, 0u, IN_DIM, MID_DIM, x, N_TOK);
    if (ok) ok = ds4_gpu_matmul_q8_0_tensor(
        up, model, model_bytes, up_offset, IN_DIM, MID_DIM, x, N_TOK);
    if (ok && variant == 1u) {
        ok = ds4_gpu_swiglu_f16_tensor(
            mid_f16, gate, up, N_TOK*MID_DIM, 0.0f, 1.0f);
        if (ok) ok = ds4_gpu_matmul_q8_0_f16_rhs_tensor(
            out, model, model_bytes, down_offset,
            MID_DIM, OUT_DIM, mid_f16, N_TOK);
    } else if (ok) {
        ok = ds4_gpu_swiglu_tensor(
            mid_f32, gate, up, N_TOK*MID_DIM, 0.0f, 1.0f);
        if (ok) ok = ds4_gpu_matmul_q8_0_tensor(
            out, model, model_bytes, down_offset,
            MID_DIM, OUT_DIM, mid_f32, N_TOK);
    }
    if (ok) ok = ds4_gpu_end_commands();
    return ok;
}

int main(void) {
    const uint64_t page = (uint64_t)getpagesize();
    const uint64_t gate_row_bytes = (uint64_t)(IN_DIM/32u)*34u;
    const uint64_t gate_weight_bytes = (uint64_t)MID_DIM*gate_row_bytes;
    const uint64_t down_row_bytes = (uint64_t)(MID_DIM/32u)*34u;
    const uint64_t down_weight_bytes = (uint64_t)OUT_DIM*down_row_bytes;
    const uint64_t up_offset = round_up(gate_weight_bytes, page);
    const uint64_t down_offset = round_up(up_offset + gate_weight_bytes, page);
    const uint64_t model_bytes = round_up(down_offset + down_weight_bytes, page);
    const uint64_t x_count = (uint64_t)N_TOK*IN_DIM;
    const uint64_t mid_count = (uint64_t)N_TOK*MID_DIM;
    const uint64_t out_count = (uint64_t)N_TOK*OUT_DIM;
    const uint64_t x_bytes = x_count*sizeof(float);
    const uint64_t mid_f32_bytes = mid_count*sizeof(float);
    const uint64_t mid_f16_bytes = mid_count*sizeof(uint16_t);
    const uint64_t out_bytes = out_count*sizeof(float);

    fprintf(stderr,
            "Q8 FFN bench N=%u K=%u mid=%u M=%u model=%.2f MiB scratch=%.2f MiB\n",
            N_TOK, IN_DIM, MID_DIM, OUT_DIM,
            (double)model_bytes/(1024.0*1024.0),
            (double)(x_bytes + 3u*mid_f32_bytes + mid_f16_bytes + out_bytes)/
                (1024.0*1024.0));

    void *model = NULL;
    float *x_host = NULL;
    float *baseline = NULL;
    float *candidate = NULL;
    ds4_gpu_tensor *x = NULL;
    ds4_gpu_tensor *gate = NULL;
    ds4_gpu_tensor *up = NULL;
    ds4_gpu_tensor *mid_f32 = NULL;
    ds4_gpu_tensor *mid_f16 = NULL;
    ds4_gpu_tensor *out = NULL;
    int ok = posix_memalign(&model, (size_t)page, (size_t)model_bytes) == 0;
    x_host = malloc((size_t)x_bytes);
    baseline = malloc((size_t)out_bytes);
    candidate = malloc((size_t)out_bytes);
    ok = ok && model && x_host && baseline && candidate;
    if (!ok) {
        fprintf(stderr, "Q8 FFN bench host allocation failed\n");
        goto cleanup;
    }

    memset(model, 0, (size_t)model_bytes);
    fill_q8_weights(model, IN_DIM, MID_DIM, gate_row_bytes, 0u);
    fill_q8_weights((uint8_t *)model + up_offset,
                    IN_DIM, MID_DIM, gate_row_bytes, 29u);
    fill_q8_weights((uint8_t *)model + down_offset,
                    MID_DIM, OUT_DIM, down_row_bytes, 53u);
    for (uint64_t i = 0; i < x_count; i++) {
        x_host[i] = (float)((int)((i*19u + (i >> 5u)*3u)%97u) - 48)/64.0f;
    }

    ok = ds4_gpu_init() && ds4_gpu_set_model_map(model, model_bytes);
    x = ds4_gpu_tensor_alloc(x_bytes);
    gate = ds4_gpu_tensor_alloc(mid_f32_bytes);
    up = ds4_gpu_tensor_alloc(mid_f32_bytes);
    mid_f32 = ds4_gpu_tensor_alloc(mid_f32_bytes);
    mid_f16 = ds4_gpu_tensor_alloc(mid_f16_bytes);
    out = ds4_gpu_tensor_alloc(out_bytes);
    ok = ok && x && gate && up && mid_f32 && mid_f16 && out &&
         ds4_gpu_tensor_write(x, 0, x_host, x_bytes);
    if (!ok) {
        fprintf(stderr, "Q8 FFN bench Metal setup failed\n");
        goto cleanup;
    }

    ds4_gpu_set_quality(false);
    for (size_t variant = 0; variant < 3u; variant++) {
        ok = run_variant(variant, gate, up, mid_f32, mid_f16, out,
                         model, model_bytes, up_offset, down_offset, x);
        if (!ok) {
            fprintf(stderr, "Q8 FFN bench warmup failed for %s\n",
                    variants[variant]);
            goto cleanup;
        }
    }

    ok = run_variant(0u, gate, up, mid_f32, mid_f16, out,
                     model, model_bytes, up_offset, down_offset, x) &&
         ds4_gpu_tensor_read(out, 0, baseline, out_bytes);
    for (size_t variant = 1; ok && variant < 3u; variant++) {
        ok = run_variant(variant, gate, up, mid_f32, mid_f16, out,
                         model, model_bytes, up_offset, down_offset, x) &&
             ds4_gpu_tensor_read(out, 0, candidate, out_bytes);
        if (ok && memcmp(baseline, candidate, (size_t)out_bytes) != 0) {
            uint64_t mismatches = 0;
            for (uint64_t i = 0; i < out_count; i++) {
                mismatches += memcmp(&baseline[i], &candidate[i], sizeof(float)) != 0;
            }
            fprintf(stderr, "Q8 FFN bench %s mismatch=%llu/%llu\n",
                    variants[variant], (unsigned long long)mismatches,
                    (unsigned long long)out_count);
            ok = 0;
        } else if (ok) {
            fprintf(stderr, "Q8 FFN bench %s bit-exact over %llu outputs\n",
                    variants[variant], (unsigned long long)out_count);
        }
    }
    if (!ok) goto cleanup;

    double elapsed[3][TIMED_RUNS] = {{0}};
    size_t counts[3] = {0};
    for (size_t round = 0; round < TIMED_RUNS; round++) {
        const size_t order[3] = {
            round%2u == 0 ? 0u : 2u,
            1u,
            round%2u == 0 ? 2u : 0u,
        };
        for (size_t oi = 0; oi < 3u; oi++) {
            const size_t variant = order[oi];
            const double t0 = now_ms();
            ok = run_variant(variant, gate, up, mid_f32, mid_f16, out,
                             model, model_bytes, up_offset, down_offset, x);
            const double dt = now_ms() - t0;
            if (!ok) {
                fprintf(stderr, "Q8 FFN bench timed run failed for %s\n",
                        variants[variant]);
                goto cleanup;
            }
            elapsed[variant][counts[variant]++] = dt;
        }
    }

    double medians[3];
    for (size_t variant = 0; variant < 3u; variant++) {
        qsort(elapsed[variant], counts[variant], sizeof(double), compare_double);
        medians[variant] = elapsed[variant][counts[variant]/2u];
        fprintf(stderr, "Q8 FFN bench %-15s median=%7.3f ms runs=%zu\n",
                variants[variant], medians[variant], counts[variant]);
    }
    fprintf(stderr,
            "Q8 FFN bench speedup mid_f16=%+.2f%% baseline_repeat=%+.2f%%\n",
            100.0*(medians[0]/medians[1] - 1.0),
            100.0*(medians[0]/medians[2] - 1.0));

cleanup:
    ds4_gpu_tensor_free(out);
    ds4_gpu_tensor_free(mid_f16);
    ds4_gpu_tensor_free(mid_f32);
    ds4_gpu_tensor_free(up);
    ds4_gpu_tensor_free(gate);
    ds4_gpu_tensor_free(x);
    ds4_gpu_cleanup();
    free(candidate);
    free(baseline);
    free(x_host);
    free(model);
    return ok ? 0 : 1;
}
