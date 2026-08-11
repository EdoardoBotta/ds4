#define _DARWIN_C_SOURCE

#include "ds4_gpu.h"

#include <errno.h>
#include <fcntl.h>
#include <inttypes.h>
#include <math.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <time.h>
#include <unistd.h>

/* ds4_metal.o uses this helper only to decide whether to draw progress text. */
int ds4_log_is_tty(FILE *fp) {
    (void)fp;
    return 0;
}

typedef enum {
    BENCH_PROJECT = 0,
    BENCH_GREEDY = 1,
    BENCH_FULL = 2,
    BENCH_COUNT = 3,
} bench_kind;

typedef struct {
    double *v;
    uint32_t n;
} samples;

typedef struct {
    const void *model_map;
    uint64_t model_size;
    uint64_t weight_offset;
    uint32_t in_dim;
    uint32_t out_dim;
    ds4_gpu_tensor *x;
    ds4_gpu_tensor *logits;
    ds4_gpu_tensor **result;
    uint32_t result_count;
} bench_ctx;

static double now_sec(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC_RAW, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1.0e-9;
}

static int cmp_double(const void *a, const void *b) {
    const double da = *(const double *)a;
    const double db = *(const double *)b;
    return (da > db) - (da < db);
}

static double sample_mean(const samples *s) {
    double sum = 0.0;
    for (uint32_t i = 0; i < s->n; i++) sum += s->v[i];
    return s->n ? sum / (double)s->n : 0.0;
}

static double sample_median(const samples *s) {
    if (!s->n) return 0.0;
    double *copy = malloc((size_t)s->n * sizeof(copy[0]));
    if (!copy) return 0.0;
    memcpy(copy, s->v, (size_t)s->n * sizeof(copy[0]));
    qsort(copy, s->n, sizeof(copy[0]), cmp_double);
    const double median = (s->n & 1u)
        ? copy[s->n / 2u]
        : 0.5 * (copy[s->n / 2u - 1u] + copy[s->n / 2u]);
    free(copy);
    return median;
}

static double sample_stddev(const samples *s) {
    if (s->n < 2) return 0.0;
    const double mean = sample_mean(s);
    double ss = 0.0;
    for (uint32_t i = 0; i < s->n; i++) {
        const double d = s->v[i] - mean;
        ss += d * d;
    }
    return sqrt(ss / (double)(s->n - 1u));
}

static const char *bench_name(bench_kind kind) {
    switch (kind) {
    case BENCH_PROJECT: return "q8_project";
    case BENCH_GREEDY:  return "fused_greedy";
    case BENCH_FULL:    return "fused_full_gumbel";
    default:            return "unknown";
    }
}

static bool encode_one(bench_ctx *ctx, bench_kind kind, uint32_t result_index,
                       uint64_t seed) {
    if (kind == BENCH_PROJECT) {
        return ds4_gpu_matmul_q8_0_tensor(ctx->logits,
                                          ctx->model_map,
                                          ctx->model_size,
                                          ctx->weight_offset,
                                          ctx->in_dim,
                                          ctx->out_dim,
                                          ctx->x,
                                          1) != 0;
    }
    if (result_index >= ctx->result_count) return false;
    return ds4_gpu_output_sample_q8_0_tensor(ctx->result[result_index],
                                             ctx->model_map,
                                             ctx->model_size,
                                             ctx->weight_offset,
                                             ctx->in_dim,
                                             ctx->out_dim,
                                             ctx->x,
                                             1.0f,
                                             0,
                                             seed,
                                             kind == BENCH_GREEDY ? 0u : 1u) != 0;
}

static bool run_completed_batch(bench_ctx *ctx, bench_kind kind, uint32_t count,
                                uint64_t seed_base, double *elapsed_sec) {
    if (count == 0 || count > ctx->result_count) return false;
    const double t0 = now_sec();
    if (!ds4_gpu_begin_commands()) return false;
    bool ok = true;
    for (uint32_t i = 0; ok && i < count; i++) {
        ok = encode_one(ctx, kind, i, seed_base + i);
    }
    if (ds4_gpu_end_commands() == 0) ok = false;
    const double t1 = now_sec();
    if (elapsed_sec) *elapsed_sec = t1 - t0;
    return ok;
}

static void print_stats(const char *measurement, bench_kind kind,
                        const samples *s, uint64_t weight_bytes,
                        double peak_gbs) {
    const double mean_ms = sample_mean(s) * 1000.0;
    const double median_ms = sample_median(s) * 1000.0;
    const double stddev_ms = sample_stddev(s) * 1000.0;
    const double effective_gbs = (double)weight_bytes / (median_ms * 1.0e6);
    const double roof_pct = peak_gbs > 0.0 ? 100.0 * effective_gbs / peak_gbs : 0.0;
    printf("%-12s %-19s n=%u mean_ms=%.4f median_ms=%.4f stddev_ms=%.4f "
           "weight_GBps=%.2f roof_pct=%.2f\n",
           measurement,
           bench_name(kind),
           s->n,
           mean_ms,
           median_ms,
           stddev_ms,
           effective_gbs,
           roof_pct);
}

static bool parse_u64(const char *s, uint64_t *out) {
    if (!s || !s[0] || !out) return false;
    errno = 0;
    char *end = NULL;
    const unsigned long long v = strtoull(s, &end, 0);
    if (errno != 0 || end == s || *end != '\0') return false;
    *out = (uint64_t)v;
    return true;
}

int main(int argc, char **argv) {
    if (argc < 5 || argc > 8) {
        fprintf(stderr,
                "usage: %s MODEL OUTPUT_OFFSET IN_DIM OUT_DIM [PEAK_GBPS [TRIALS [BATCH]]]\n",
                argv[0]);
        return 2;
    }

    uint64_t weight_offset = 0;
    uint64_t in_dim64 = 0;
    uint64_t out_dim64 = 0;
    if (!parse_u64(argv[2], &weight_offset) ||
        !parse_u64(argv[3], &in_dim64) ||
        !parse_u64(argv[4], &out_dim64) ||
        in_dim64 == 0 || in_dim64 > UINT32_MAX ||
        out_dim64 == 0 || out_dim64 > UINT32_MAX) {
        fprintf(stderr, "invalid offset or dimensions\n");
        return 2;
    }
    const uint32_t in_dim = (uint32_t)in_dim64;
    const uint32_t out_dim = (uint32_t)out_dim64;
    const double peak_gbs = argc > 5 ? strtod(argv[5], NULL) : 273.0;
    const uint32_t trials = argc > 6 ? (uint32_t)strtoul(argv[6], NULL, 10) : 9u;
    const uint32_t batch = argc > 7 ? (uint32_t)strtoul(argv[7], NULL, 10) : 12u;
    if (trials < 3 || batch == 0 || batch > 64) {
        fprintf(stderr, "trials must be >= 3 and batch must be in 1..64\n");
        return 2;
    }

    const uint64_t blocks = in_dim / 32u;
    if ((in_dim & 31u) != 0 || blocks > UINT64_MAX / 34u) {
        fprintf(stderr, "input dimension must be a positive multiple of 32\n");
        return 2;
    }
    const uint64_t row_bytes = blocks * 34u;
    if (out_dim > UINT64_MAX / row_bytes) {
        fprintf(stderr, "weight byte count overflow\n");
        return 2;
    }
    const uint64_t weight_bytes = (uint64_t)out_dim * row_bytes;

    int fd = open(argv[1], O_RDONLY);
    if (fd < 0) {
        fprintf(stderr, "open %s: %s\n", argv[1], strerror(errno));
        return 1;
    }
    struct stat st;
    if (fstat(fd, &st) != 0 || st.st_size <= 0) {
        fprintf(stderr, "stat %s: %s\n", argv[1], strerror(errno));
        close(fd);
        return 1;
    }
    const uint64_t model_size = (uint64_t)st.st_size;
    if (weight_offset > model_size || weight_bytes > model_size - weight_offset) {
        fprintf(stderr, "output weight range exceeds model file\n");
        close(fd);
        return 1;
    }
    void *model_map = mmap(NULL, (size_t)model_size, PROT_READ, MAP_SHARED, fd, 0);
    if (model_map == MAP_FAILED) {
        fprintf(stderr, "mmap %s: %s\n", argv[1], strerror(errno));
        close(fd);
        return 1;
    }

    int rc = 1;
    bench_ctx ctx = {
        .model_map = model_map,
        .model_size = model_size,
        .weight_offset = weight_offset,
        .in_dim = in_dim,
        .out_dim = out_dim,
        .result_count = batch,
    };
    ctx.result = calloc(batch, sizeof(ctx.result[0]));
    samples batched[BENCH_COUNT] = {0};
    samples single[BENCH_COUNT] = {0};
    for (uint32_t k = 0; k < BENCH_COUNT; k++) {
        batched[k].n = trials;
        batched[k].v = calloc(trials, sizeof(double));
        single[k].n = trials;
        single[k].v = calloc(trials, sizeof(double));
    }

    if (!ctx.result || !ds4_gpu_init()) goto cleanup;
    ds4_gpu_set_quality(false);
    ds4_gpu_set_ssd_streaming(false);
    if (!ds4_gpu_set_model_fd_for_map(fd, model_map) ||
        !ds4_gpu_set_model_map_range(model_map,
                                     model_size,
                                     weight_offset,
                                     weight_bytes,
                                     weight_bytes)) {
        goto cleanup;
    }

    ctx.x = ds4_gpu_tensor_alloc((uint64_t)in_dim * sizeof(float));
    ctx.logits = ds4_gpu_tensor_alloc((uint64_t)out_dim * sizeof(float));
    for (uint32_t i = 0; i < batch; i++) {
        ctx.result[i] = ds4_gpu_tensor_alloc(DS4_GPU_OUTPUT_SAMPLE_RESULT_BYTES);
    }
    if (!ctx.x || !ctx.logits) goto cleanup;
    for (uint32_t i = 0; i < batch; i++) if (!ctx.result[i]) goto cleanup;

    float *x_host = malloc((size_t)in_dim * sizeof(x_host[0]));
    if (!x_host) goto cleanup;
    for (uint32_t i = 0; i < in_dim; i++) {
        x_host[i] = (float)((int)(i % 127u) - 63) / 64.0f;
    }
    const bool wrote_x = ds4_gpu_tensor_write(ctx.x,
                                               0,
                                               x_host,
                                               (uint64_t)in_dim * sizeof(x_host[0])) != 0;
    free(x_host);
    if (!wrote_x) goto cleanup;

    /* Compile all pipelines and make every output-weight page resident before timing. */
    for (uint32_t k = 0; k < BENCH_COUNT; k++) {
        for (uint32_t i = 0; i < 3; i++) {
            double ignored = 0.0;
            if (!run_completed_batch(&ctx, (bench_kind)k, 1,
                                     UINT64_C(0x1234567800000000) + i,
                                     &ignored)) {
                goto cleanup;
            }
        }
    }

    /* Rotate order each trial to distribute frequency/thermal drift. */
    for (uint32_t trial = 0; trial < trials; trial++) {
        for (uint32_t step = 0; step < BENCH_COUNT; step++) {
            const bench_kind kind = (bench_kind)((trial + step) % BENCH_COUNT);
            double elapsed = 0.0;
            if (!run_completed_batch(&ctx,
                                     kind,
                                     batch,
                                     UINT64_C(0x9e3779b97f4a7c15) * (trial + 1u),
                                     &elapsed)) {
                goto cleanup;
            }
            batched[kind].v[trial] = elapsed / (double)batch;
        }
    }

    for (uint32_t trial = 0; trial < trials; trial++) {
        for (uint32_t step = 0; step < BENCH_COUNT; step++) {
            const bench_kind kind = (bench_kind)((trial + step) % BENCH_COUNT);
            double elapsed = 0.0;
            if (!run_completed_batch(&ctx,
                                     kind,
                                     1,
                                     UINT64_C(0xd1b54a32d192ed03) * (trial + 1u),
                                     &elapsed)) {
                goto cleanup;
            }
            single[kind].v[trial] = elapsed;
        }
    }

    printf("model=%s\n", argv[1]);
    printf("in_dim=%u out_dim=%u row_bytes=%" PRIu64
           " weight_bytes=%" PRIu64 " weight_GiB=%.6f\n",
           in_dim,
           out_dim,
           row_bytes,
           weight_bytes,
           (double)weight_bytes / 1073741824.0);
    printf("peak_GBps=%.2f bandwidth_floor_ms=%.4f trials=%u batch=%u\n",
           peak_gbs,
           (double)weight_bytes / (peak_gbs * 1.0e6),
           trials,
           batch);
    for (uint32_t k = 0; k < BENCH_COUNT; k++) {
        print_stats("batched", (bench_kind)k, &batched[k], weight_bytes, peak_gbs);
    }
    for (uint32_t k = 0; k < BENCH_COUNT; k++) {
        print_stats("single_wall", (bench_kind)k, &single[k], weight_bytes, peak_gbs);
    }
    rc = 0;

cleanup:
    if (ctx.result) {
        for (uint32_t i = 0; i < batch; i++) ds4_gpu_tensor_free(ctx.result[i]);
    }
    ds4_gpu_tensor_free(ctx.logits);
    ds4_gpu_tensor_free(ctx.x);
    free(ctx.result);
    for (uint32_t k = 0; k < BENCH_COUNT; k++) {
        free(batched[k].v);
        free(single[k].v);
    }
    ds4_gpu_cleanup();
    munmap(model_map, (size_t)model_size);
    close(fd);
    return rc;
}
