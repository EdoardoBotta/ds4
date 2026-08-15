/* Qwen3.6 27B inference on Apple Metal.
 *
 * This is a deliberately narrow, vertical runtime: mmap a Q8_0 GGUF, parse
 * its tokenizer and weights, construct the persistent Qwen graph, prefill in
 * chunks, and decode greedily. The optimized Metal backend is kept intact;
 * only unrelated model families, frontends, and host execution paths were
 * removed.
 */

#include <errno.h>
#include <fcntl.h>
#include <inttypes.h>
#include <ctype.h>
#include <limits.h>
#include <math.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <strings.h>
#include <sys/file.h>
#include <sys/mman.h>
#include <sys/stat.h>

#include <stdarg.h>
#include <time.h>
#include <unistd.h>

#include "ds4.h"
#include "ds4_gpu.h"

#define DS4_NEG_INF (-1.0e30f)

enum {
    DS4_MAX_LAYER = 64,
};

/* Fixed dimensions make the graph easy to follow and turn incompatible GGUFs
 * into clear startup errors instead of late kernel failures. */
typedef struct {
    const char *name;
    uint32_t n_layer;
    uint32_t n_embd;
    uint32_t n_vocab;
    uint32_t n_head;
    uint32_t n_head_kv;
    uint32_t n_head_dim;
    uint32_t n_value_dim;
    uint32_t n_rot;
    uint32_t n_ff_dense;
    uint32_t n_gdn_inner;
    uint32_t n_gdn_qk_head;
    uint32_t n_gdn_v_head;
    uint32_t n_gdn_state;
    uint32_t n_gdn_dt_rank;
    uint32_t n_gdn_conv;
    uint32_t n_full_attn_interval;
    float rms_eps;
    float rope_freq_base;
    uint64_t rope_orig_ctx;
} ds4_shape;

static const ds4_shape g_ds4_shape = {
    .name = "Qwen3.6 27B",
    .n_layer = 64,
    .n_embd = 5120,
    .n_vocab = 248320,
    .n_head = 24,
    .n_head_kv = 4,
    .n_head_dim = 256,
    .n_value_dim = 256,
    .n_rot = 64,
    .n_ff_dense = 17408,
    .n_gdn_inner = 6144,
    .n_gdn_qk_head = 16,
    .n_gdn_v_head = 48,
    .n_gdn_state = 128,
    .n_gdn_dt_rank = 48,
    .n_gdn_conv = 4,
    .n_full_attn_interval = 4,
    .rms_eps = 1.0e-6f,
    .rope_freq_base = 10000000.0f,
    .rope_orig_ctx = 262144,
};

#define DS4_MODEL_SHAPE_NAME          (g_ds4_shape.name)
#define DS4_N_LAYER                   (g_ds4_shape.n_layer)
#define DS4_N_EMBD                    (g_ds4_shape.n_embd)
#define DS4_N_VOCAB                   (g_ds4_shape.n_vocab)
#define DS4_N_HEAD                    (g_ds4_shape.n_head)
#define DS4_N_HEAD_KV                 (g_ds4_shape.n_head_kv)
#define DS4_N_HEAD_DIM                (g_ds4_shape.n_head_dim)
#define DS4_N_VALUE_DIM               (g_ds4_shape.n_value_dim)
#define DS4_N_ROT                     (g_ds4_shape.n_rot)
#define DS4_N_FF_DENSE                (g_ds4_shape.n_ff_dense)
#define DS4_N_GDN_INNER               (g_ds4_shape.n_gdn_inner)
#define DS4_N_GDN_QK_HEAD             (g_ds4_shape.n_gdn_qk_head)
#define DS4_N_GDN_V_HEAD              (g_ds4_shape.n_gdn_v_head)
#define DS4_N_GDN_STATE               (g_ds4_shape.n_gdn_state)
#define DS4_N_GDN_DT_RANK             (g_ds4_shape.n_gdn_dt_rank)
#define DS4_N_GDN_CONV                (g_ds4_shape.n_gdn_conv)
#define DS4_N_FULL_ATTN_INTERVAL       (g_ds4_shape.n_full_attn_interval)
#define DS4_RMS_EPS                   (g_ds4_shape.rms_eps)
#define DS4_ROPE_FREQ_BASE            (g_ds4_shape.rope_freq_base)
#define DS4_ROPE_ORIG_CTX             (g_ds4_shape.rope_orig_ctx)

/* The loader records offsets into one read-only mmap. Metal later wraps the
 * tensor region in shared, no-copy buffers. */
#define DS4_GGUF_MAGIC 0x46554747u
#define DS4_MAX_DIMS   8

typedef struct {
    const char *ptr;
    uint64_t len;
} ds4_str;

typedef ds4_tokens token_vec;

typedef struct {
    const uint8_t *base;
    uint64_t size;
    uint64_t pos;
    char error[256];
} ds4_cursor;

static void ds4_die(const char *msg) {
    fprintf(stderr, "ds4: %s\n", msg);
    exit(1);
}

static void ds4_die_errno(const char *what, const char *path) {
    fprintf(stderr, "ds4: %s '%s': %s\n", what, path, strerror(errno));
    exit(1);
}

static bool ds4_streq(ds4_str s, const char *z) {
    size_t n = strlen(z);
    return s.len == n && memcmp(s.ptr, z, n) == 0;
}

static bool ds4_str_eq(ds4_str a, ds4_str b) {
    return a.len == b.len && memcmp(a.ptr, b.ptr, a.len) == 0;
}

static uint64_t hash_bytes(const void *ptr, uint64_t len) {
    const uint8_t *p = ptr;
    uint64_t h = 1469598103934665603ull;
    for (uint64_t i = 0; i < len; i++) {
        h ^= p[i];
        h *= 1099511628211ull;
    }
    return h;
}

static void *xcalloc(size_t n, size_t size) {
    void *p = calloc(n, size);
    if (!p) ds4_die("out of memory");
    return p;
}

static void *xmalloc(size_t size) {
    void *p = malloc(size);
    if (!p) ds4_die("out of memory");
    return p;
}

static void *xrealloc(void *ptr, size_t size) {
    void *p = realloc(ptr, size);
    if (!p) ds4_die("out of memory");
    return p;
}

static double now_sec(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1.0e-9;
}

static const char *ds4_log_color_code(ds4_log_type type) {
    switch (type) {
    case DS4_LOG_PREFILL:
    case DS4_LOG_TIMING:
        return "\x1b[36m";
    case DS4_LOG_GENERATION:
    case DS4_LOG_OK:
        return "\x1b[32m";
    case DS4_LOG_KVCACHE:
        return "\x1b[33m";
    case DS4_LOG_TOOL:
        return "\x1b[90m";
    case DS4_LOG_WARNING:
        return "\x1b[38;5;208m";
    case DS4_LOG_ERROR:
        return "\x1b[31m";
    default:
        return "";
    }
}

bool ds4_log_is_tty(FILE *fp) {
    int fd = fileno(fp);
    return fd >= 0 && isatty(fd) != 0;
}

static void ds4_vlog(FILE *fp, ds4_log_type type, const char *fmt, va_list ap) {
    const bool colorize = type != DS4_LOG_DEFAULT && ds4_log_is_tty(fp);
    if (colorize) fputs(ds4_log_color_code(type), fp);
    vfprintf(fp, fmt, ap);
    if (colorize) fputs("\x1b[0m", fp);
}

void ds4_log(FILE *fp, ds4_log_type type, const char *fmt, ...) {
    va_list ap;
    va_start(ap, fmt);
    ds4_vlog(fp, type, fmt, ap);
    va_end(ap);
}

static void cursor_error(ds4_cursor *c, const char *msg) {
    if (c->error[0] == '\0') {
        snprintf(c->error, sizeof(c->error), "%s at byte %" PRIu64, msg, c->pos);
    }
}

static bool cursor_has(ds4_cursor *c, uint64_t n) {
    if (n > c->size || c->pos > c->size - n) {
        cursor_error(c, "truncated GGUF file");
        return false;
    }
    return true;
}

static bool cursor_read(ds4_cursor *c, void *dst, uint64_t n) {
    if (!cursor_has(c, n)) return false;
    memcpy(dst, c->base + c->pos, (size_t)n);
    c->pos += n;
    return true;
}

static bool cursor_skip(ds4_cursor *c, uint64_t n) {
    if (!cursor_has(c, n)) return false;
    c->pos += n;
    return true;
}

static bool cursor_u32(ds4_cursor *c, uint32_t *v) {
    return cursor_read(c, v, sizeof(*v));
}

static bool cursor_u64(ds4_cursor *c, uint64_t *v) {
    return cursor_read(c, v, sizeof(*v));
}

static bool cursor_string(ds4_cursor *c, ds4_str *s) {
    uint64_t len;
    if (!cursor_u64(c, &len)) return false;
    if (!cursor_has(c, len)) return false;
    s->ptr = (const char *)(c->base + c->pos);
    s->len = len;
    c->pos += len;
    return true;
}

static uint64_t align_up(uint64_t value, uint64_t alignment) {
    uint64_t rem = value % alignment;
    return rem == 0 ? value : value + alignment - rem;
}

enum {
    GGUF_VALUE_UINT8   = 0,
    GGUF_VALUE_INT8    = 1,
    GGUF_VALUE_UINT16  = 2,
    GGUF_VALUE_INT16   = 3,
    GGUF_VALUE_UINT32  = 4,
    GGUF_VALUE_INT32   = 5,
    GGUF_VALUE_FLOAT32 = 6,
    GGUF_VALUE_BOOL    = 7,
    GGUF_VALUE_STRING  = 8,
    GGUF_VALUE_ARRAY   = 9,
    GGUF_VALUE_UINT64  = 10,
    GGUF_VALUE_INT64   = 11,
    GGUF_VALUE_FLOAT64 = 12,
};

typedef struct {
    const char *name;
    uint32_t block_elems;
    uint32_t block_bytes;
} gguf_type_info;

static const gguf_type_info gguf_types[] = {
    [0]  = {"f32",      1,   4},
    [1]  = {"f16",      1,   2},
    [2]  = {"q4_0",    32,  18},
    [3]  = {"q4_1",    32,  20},
    [6]  = {"q5_0",    32,  22},
    [7]  = {"q5_1",    32,  24},
    [8]  = {"q8_0",    32,  34},
    [9]  = {"q8_1",    32,  40},
    [10] = {"q2_k",   256,  84},
    [11] = {"q3_k",   256, 110},
    [12] = {"q4_k",   256, 144},
    [13] = {"q5_k",   256, 176},
    [14] = {"q6_k",   256, 210},
    [15] = {"q8_k",   256, 292},
    [16] = {"iq2_xxs",256,  66},
    [17] = {"iq2_xs", 256,  74},
    [18] = {"iq3_xxs",256,  98},
    [19] = {"iq1_s",  256, 110},
    [20] = {"iq4_nl", 256,  50},
    [21] = {"iq3_s",  256, 110},
    [22] = {"iq2_s",  256,  82},
    [23] = {"iq4_xs", 256, 136},
    [24] = {"i8",       1,   1},
    [25] = {"i16",      1,   2},
    [26] = {"i32",      1,   4},
    [27] = {"i64",      1,   8},
    [28] = {"f64",      1,   8},
    [29] = {"iq1_m",  256,  56},
    [30] = {"bf16",     1,   2},
    [39] = {"mxfp4",   32,  17},
};

enum {
    DS4_TENSOR_F32      = 0,
    DS4_TENSOR_Q8_0     = 8,
};

typedef struct {
    ds4_str key;
    uint32_t type;
    uint64_t value_pos;
} ds4_kv;

typedef struct {
    ds4_str name;
    uint32_t ndim;
    uint64_t dim[DS4_MAX_DIMS];
    uint32_t type;
    uint64_t rel_offset;
    uint64_t abs_offset;
    uint64_t elements;
    uint64_t bytes;
} ds4_tensor;

typedef struct {
    int fd;
    const uint8_t *map;
    uint64_t size;

    uint32_t version;
    uint64_t n_kv;
    uint64_t n_tensors;
    uint64_t alignment;
    uint64_t tensor_data_pos;
    uint64_t max_tensor_bytes;

    ds4_kv *kv;
    ds4_tensor *tensors;
} ds4_model;

static uint64_t scalar_value_size(uint32_t type) {
    switch (type) {
    case GGUF_VALUE_UINT8:
    case GGUF_VALUE_INT8:
    case GGUF_VALUE_BOOL:
        return 1;
    case GGUF_VALUE_UINT16:
    case GGUF_VALUE_INT16:
        return 2;
    case GGUF_VALUE_UINT32:
    case GGUF_VALUE_INT32:
    case GGUF_VALUE_FLOAT32:
        return 4;
    case GGUF_VALUE_UINT64:
    case GGUF_VALUE_INT64:
    case GGUF_VALUE_FLOAT64:
        return 8;
    default:
        return 0;
    }
}

static bool skip_value(ds4_cursor *c, uint32_t type, int depth) {
    if (depth > 8) {
        cursor_error(c, "metadata array nesting is too deep");
        return false;
    }

    uint64_t scalar = scalar_value_size(type);
    if (scalar != 0) return cursor_skip(c, scalar);

    if (type == GGUF_VALUE_STRING) {
        ds4_str ignored;
        return cursor_string(c, &ignored);
    }

    if (type == GGUF_VALUE_ARRAY) {
        uint32_t item_type;
        uint64_t len;

        if (!cursor_u32(c, &item_type)) return false;
        if (!cursor_u64(c, &len)) return false;

        uint64_t item_size = scalar_value_size(item_type);
        if (item_size != 0) {
            if (len > UINT64_MAX / item_size) {
                cursor_error(c, "metadata array is too large");
                return false;
            }
            return cursor_skip(c, len * item_size);
        }

        for (uint64_t i = 0; i < len; i++) {
            if (!skip_value(c, item_type, depth + 1)) return false;
        }
        return true;
    }

    cursor_error(c, "unknown GGUF metadata type");
    return false;
}

static const gguf_type_info *tensor_type(uint32_t type) {
    uint32_t n = sizeof(gguf_types) / sizeof(gguf_types[0]);
    if (type >= n || gguf_types[type].name == NULL) return NULL;
    return &gguf_types[type];
}

static const char *tensor_type_name(uint32_t type) {
    const gguf_type_info *info = tensor_type(type);
    return info ? info->name : "unknown";
}

static bool tensor_nbytes(uint32_t type, uint64_t elements, uint64_t *bytes) {
    const gguf_type_info *info = tensor_type(type);
    if (!info || info->block_elems == 0) return false;
    uint64_t blocks = (elements + info->block_elems - 1) / info->block_elems;
    if (blocks > UINT64_MAX / info->block_bytes) return false;
    *bytes = blocks * info->block_bytes;
    return true;
}

static ds4_cursor cursor_at(const ds4_model *m, uint64_t pos) {
    ds4_cursor c = {
        .base = m->map,
        .size = m->size,
        .pos = pos,
        .error = {0},
    };
    return c;
}

static ds4_kv *model_find_kv(const ds4_model *m, const char *key) {
    for (uint64_t i = 0; i < m->n_kv; i++) {
        if (ds4_streq(m->kv[i].key, key)) return &m->kv[i];
    }
    return NULL;
}

static bool model_get_string(const ds4_model *m, const char *key, ds4_str *out) {
    ds4_kv *kv = model_find_kv(m, key);
    if (!kv || kv->type != GGUF_VALUE_STRING) return false;
    ds4_cursor c = cursor_at(m, kv->value_pos);
    return cursor_string(&c, out);
}

static bool model_get_u32(const ds4_model *m, const char *key, uint32_t *out) {
    ds4_kv *kv = model_find_kv(m, key);
    if (!kv || kv->type != GGUF_VALUE_UINT32) return false;
    ds4_cursor c = cursor_at(m, kv->value_pos);
    return cursor_u32(&c, out);
}

static bool model_get_token_id(const ds4_model *m, const char *key, int *out) {
    ds4_kv *kv = model_find_kv(m, key);
    if (!kv) return false;

    ds4_cursor c = cursor_at(m, kv->value_pos);
    switch (kv->type) {
    case GGUF_VALUE_UINT32: {
        uint32_t v = 0;
        if (!cursor_u32(&c, &v) || v > (uint32_t)INT_MAX) return false;
        *out = (int)v;
        return true;
    }
    case GGUF_VALUE_INT32: {
        int32_t v = 0;
        if (!cursor_read(&c, &v, sizeof(v)) || v < 0) return false;
        *out = (int)v;
        return true;
    }
    case GGUF_VALUE_UINT64: {
        uint64_t v = 0;
        if (!cursor_u64(&c, &v) || v > (uint64_t)INT_MAX) return false;
        *out = (int)v;
        return true;
    }
    case GGUF_VALUE_INT64: {
        int64_t v = 0;
        if (!cursor_read(&c, &v, sizeof(v)) || v < 0 || v > (int64_t)INT_MAX) return false;
        *out = (int)v;
        return true;
    }
    default:
        return false;
    }
}

static bool model_get_f32_compat(const ds4_model *m, const char *key, float *out) {
    ds4_kv *kv = model_find_kv(m, key);
    if (!kv) return false;
    ds4_cursor c = cursor_at(m, kv->value_pos);
    if (kv->type == GGUF_VALUE_FLOAT32) {
        return cursor_read(&c, out, sizeof(*out));
    }
    if (kv->type == GGUF_VALUE_FLOAT64) {
        double v = 0.0;
        if (!cursor_read(&c, &v, sizeof(v))) return false;
        *out = (float)v;
        return true;
    }
    if (kv->type == GGUF_VALUE_UINT32) {
        uint32_t v = 0;
        if (!cursor_u32(&c, &v)) return false;
        *out = (float)v;
        return true;
    }
    if (kv->type == GGUF_VALUE_INT32) {
        int32_t v = 0;
        if (!cursor_read(&c, &v, sizeof(v))) return false;
        *out = (float)v;
        return true;
    }
    return false;
}

typedef struct {
    uint32_t type;
    uint64_t len;
    uint64_t data_pos;
} ds4_array_ref;

static bool model_get_array(const ds4_model *m, const char *key, ds4_array_ref *out) {
    ds4_kv *kv = model_find_kv(m, key);
    if (!kv || kv->type != GGUF_VALUE_ARRAY) return false;

    ds4_cursor c = cursor_at(m, kv->value_pos);
    if (!cursor_u32(&c, &out->type)) return false;
    if (!cursor_u64(&c, &out->len)) return false;
    out->data_pos = c.pos;
    return true;
}

static void model_close(ds4_model *m) {
    if (!m) return;
    free(m->kv);
    free(m->tensors);
    if (m->map) munmap((void *)m->map, (size_t)m->size);
    if (m->fd >= 0) close(m->fd);
    memset(m, 0, sizeof(*m));
    m->fd = -1;
}

static void parse_metadata(ds4_model *m, ds4_cursor *c) {

    if (m->n_kv > c->size - c->pos) ds4_die("GGUF metadata count exceeds file size");
    m->kv = calloc((size_t)m->n_kv, sizeof(m->kv[0]));
    if (!m->kv) ds4_die("out of memory while allocating metadata table");

    m->alignment = 32;

    for (uint64_t i = 0; i < m->n_kv; i++) {
        ds4_kv *kv = &m->kv[i];

        if (!cursor_string(c, &kv->key)) ds4_die(c->error);
        if (!cursor_u32(c, &kv->type)) ds4_die(c->error);

        kv->value_pos = c->pos;

        if (ds4_streq(kv->key, "general.alignment") &&
            kv->type == GGUF_VALUE_UINT32)
        {
            ds4_cursor tmp = cursor_at(m, kv->value_pos);
            uint32_t alignment;
            if (cursor_u32(&tmp, &alignment) && alignment != 0) {
                m->alignment = alignment;
            }
        }

        if (!skip_value(c, kv->type, 0)) ds4_die(c->error);
    }
}

static void parse_tensors(ds4_model *m, ds4_cursor *c) {

    if (m->n_tensors > c->size - c->pos) ds4_die("GGUF tensor count exceeds file size");
    m->tensors = calloc((size_t)m->n_tensors, sizeof(m->tensors[0]));
    if (!m->tensors) ds4_die("out of memory while allocating tensor table");

    for (uint64_t i = 0; i < m->n_tensors; i++) {
        ds4_tensor *t = &m->tensors[i];

        if (!cursor_string(c, &t->name)) ds4_die(c->error);
        if (!cursor_u32(c, &t->ndim)) ds4_die(c->error);
        if (t->ndim == 0 || t->ndim > DS4_MAX_DIMS) {
            ds4_die("tensor has an unsupported number of dimensions");
        }

        t->elements = 1;
        for (uint32_t d = 0; d < t->ndim; d++) {
            if (!cursor_u64(c, &t->dim[d])) ds4_die(c->error);
            if (t->dim[d] != 0 && t->elements > UINT64_MAX / t->dim[d]) {
                ds4_die("tensor element count overflow");
            }
            t->elements *= t->dim[d];
        }

        if (!cursor_u32(c, &t->type)) ds4_die(c->error);
        if (!cursor_u64(c, &t->rel_offset)) ds4_die(c->error);

        if (!tensor_nbytes(t->type, t->elements, &t->bytes)) {
            ds4_log(stderr,
                DS4_LOG_WARNING,
                "ds4: warning: tensor %.*s has unsupported GGUF type %u\n",
                (int)t->name.len, t->name.ptr, t->type);
        }
    }

    m->tensor_data_pos = align_up(c->pos, m->alignment);

    for (uint64_t i = 0; i < m->n_tensors; i++) {
        ds4_tensor *t = &m->tensors[i];
        if (t->rel_offset > UINT64_MAX - m->tensor_data_pos) {
            ds4_die("tensor offset overflow");
        }
        t->abs_offset = m->tensor_data_pos + t->rel_offset;
        if (t->bytes != 0 &&
            (t->abs_offset > m->size || t->bytes > m->size - t->abs_offset))
        {
            ds4_die("tensor points outside GGUF file");
        }
        if (t->bytes > m->max_tensor_bytes) {
            m->max_tensor_bytes = t->bytes;
        }
    }
}

static void model_open(ds4_model *m, const char *path) {
    memset(m, 0, sizeof(*m));
    m->fd = -1;

    int fd = open(path, O_RDONLY);
    if (fd == -1) ds4_die_errno("cannot open model", path);

    struct stat st;
    if (fstat(fd, &st) == -1) ds4_die_errno("cannot stat model", path);
    if (st.st_size < 32) ds4_die("model file is too small to be GGUF");

    void *map = mmap(NULL, (size_t)st.st_size, PROT_READ, MAP_SHARED, fd, 0);
    if (map == MAP_FAILED) ds4_die_errno("cannot mmap model", path);

    m->fd = fd;
    m->map = map;
    m->size = (uint64_t)st.st_size;

    ds4_cursor c = cursor_at(m, 0);
    uint32_t magic;
    if (!cursor_u32(&c, &magic)) ds4_die(c.error);
    if (magic != DS4_GGUF_MAGIC) ds4_die("model is not a GGUF file");
    if (!cursor_u32(&c, &m->version)) ds4_die(c.error);
    if (!cursor_u64(&c, &m->n_tensors)) ds4_die(c.error);
    if (!cursor_u64(&c, &m->n_kv)) ds4_die(c.error);

    if (m->version != 3) ds4_die("only GGUF v3 is supported");

    parse_metadata(m, &c);
    parse_tensors(m, &c);
}

static ds4_tensor *model_find_tensor(const ds4_model *m, const char *name) {
    const size_t len = strlen(name);
    for (uint64_t i = 0; i < m->n_tensors; i++) {
        if (m->tensors[i].name.len == len &&
            memcmp(m->tensors[i].name.ptr, name, len) == 0) {
            return &m->tensors[i];
        }
    }
    return NULL;
}

typedef struct {
    ds4_tensor *attn_norm;
    ds4_tensor *attn_output;
    ds4_tensor *qwen_attn_q;
    ds4_tensor *qwen_attn_k;
    ds4_tensor *qwen_attn_v;
    ds4_tensor *qwen_attn_q_norm;
    ds4_tensor *qwen_attn_k_norm;
    ds4_tensor *qwen_attn_qkv;
    ds4_tensor *qwen_attn_gate;
    ds4_tensor *qwen_ssm_a;
    ds4_tensor *qwen_ssm_alpha;
    ds4_tensor *qwen_ssm_beta;
    ds4_tensor *qwen_ssm_conv1d;
    ds4_tensor *qwen_ssm_dt;
    ds4_tensor *qwen_ssm_norm;
    ds4_tensor *qwen_ssm_out;
    ds4_tensor *ffn_norm;
    ds4_tensor *ffn_gate;
    ds4_tensor *ffn_up;
    ds4_tensor *ffn_down;
} ds4_layer_weights;

typedef struct {
    ds4_tensor *token_embd;
    ds4_tensor *output_norm;
    ds4_tensor *output;
    ds4_layer_weights layer[DS4_MAX_LAYER];
} ds4_weights;

static uint32_t required_u32(const ds4_model *m, const char *key) {
    uint32_t v = 0;
    if (!model_get_u32(m, key, &v)) {
        fprintf(stderr, "ds4: required metadata key is missing: %s\n", key);
        exit(1);
    }
    return v;
}

static float required_f32(const ds4_model *m, const char *key) {
    float v = 0.0f;
    if (!model_get_f32_compat(m, key, &v)) {
        fprintf(stderr, "ds4: required metadata key is missing: %s\n", key);
        exit(1);
    }
    return v;
}

static ds4_tensor *required_tensor(const ds4_model *m, const char *name) {
    ds4_tensor *t = model_find_tensor(m, name);
    if (!t) {
        fprintf(stderr, "ds4: required tensor is missing: %s\n", name);
        exit(1);
    }
    return t;
}

static ds4_tensor *required_tensorf(const ds4_model *m, const char *fmt, uint32_t layer) {
    char name[128];
    int n = snprintf(name, sizeof(name), fmt, layer);
    if (n < 0 || (size_t)n >= sizeof(name)) ds4_die("tensor name is too long");
    return required_tensor(m, name);
}

static void tensor_expect_layout(
        const ds4_tensor *t,
        uint32_t          type,
        uint32_t          ndim,
        uint64_t          d0,
        uint64_t          d1,
        uint64_t          d2) {
    if (!t) ds4_die("internal error: missing tensor while validating layout");
    if (t->type != type) {
        fprintf(stderr,
                "ds4: tensor %.*s has type %s, expected %s\n",
                (int)t->name.len,
                t->name.ptr,
                tensor_type_name(t->type),
                tensor_type_name(type));
        exit(1);
    }
    if (t->ndim != ndim) {
        fprintf(stderr,
                "ds4: tensor %.*s has %u dimensions, expected %u\n",
                (int)t->name.len,
                t->name.ptr,
                t->ndim,
                ndim);
        exit(1);
    }

    const uint64_t want[3] = { d0, d1, d2 };
    for (uint32_t i = 0; i < ndim; i++) {
        if (t->dim[i] == want[i]) continue;
        fprintf(stderr,
                "ds4: tensor %.*s has dim[%u]=%" PRIu64 ", expected %" PRIu64 "\n",
                (int)t->name.len,
                t->name.ptr,
                i,
                t->dim[i],
                want[i]);
        exit(1);
    }
}

static void config_expect_f32(const char *name, float got, float expected);

static void config_expect_u32(const char *name, uint32_t got, uint32_t expected) {
    if (got == expected) return;
    fprintf(stderr, "ds4: expected %s=%u for %s, got %u\n",
            name, expected, DS4_MODEL_SHAPE_NAME, got);
    exit(1);
}

static void config_expect_f32(const char *name, float got, float expected) {
    const float scale = fabsf(expected) > 1.0f ? fabsf(expected) : 1.0f;
    if (fabsf(got - expected) <= scale * 1.0e-6f) return;
    fprintf(stderr, "ds4: expected %s=%.9g for %s, got %.9g\n",
            name, (double)expected, DS4_MODEL_SHAPE_NAME, (double)got);
    exit(1);
}

/* Validate every dimension that the fixed graph and kernels assume. */
static void config_validate_qwen35_model(const ds4_model *m) {
    ds4_array_ref tokens;
    if (!model_get_array(m, "tokenizer.ggml.tokens", &tokens) ||
        tokens.type != GGUF_VALUE_STRING) {
        ds4_die("Qwen3.6 tokenizer.ggml.tokens is missing or invalid");
    }
    if (tokens.len > UINT32_MAX) ds4_die("Qwen3.6 vocabulary is too large");

    config_expect_u32("block_count", required_u32(m, "qwen35.block_count"), DS4_N_LAYER);
    config_expect_u32("context_length", required_u32(m, "qwen35.context_length"), (uint32_t)DS4_ROPE_ORIG_CTX);
    config_expect_u32("embedding_length", required_u32(m, "qwen35.embedding_length"), DS4_N_EMBD);
    config_expect_u32("vocab_size", (uint32_t)tokens.len, DS4_N_VOCAB);
    config_expect_u32("feed_forward_length", required_u32(m, "qwen35.feed_forward_length"), DS4_N_FF_DENSE);
    config_expect_u32("attention.head_count", required_u32(m, "qwen35.attention.head_count"), DS4_N_HEAD);
    config_expect_u32("attention.head_count_kv", required_u32(m, "qwen35.attention.head_count_kv"), DS4_N_HEAD_KV);
    config_expect_u32("attention.key_length", required_u32(m, "qwen35.attention.key_length"), DS4_N_HEAD_DIM);
    config_expect_u32("attention.value_length", required_u32(m, "qwen35.attention.value_length"), DS4_N_VALUE_DIM);
    config_expect_u32("rope.dimension_count", required_u32(m, "qwen35.rope.dimension_count"), DS4_N_ROT);
    config_expect_u32("ssm.conv_kernel", required_u32(m, "qwen35.ssm.conv_kernel"), DS4_N_GDN_CONV);
    config_expect_u32("ssm.state_size", required_u32(m, "qwen35.ssm.state_size"), DS4_N_GDN_STATE);
    config_expect_u32("ssm.group_count", required_u32(m, "qwen35.ssm.group_count"), DS4_N_GDN_QK_HEAD);
    config_expect_u32("ssm.time_step_rank", required_u32(m, "qwen35.ssm.time_step_rank"), DS4_N_GDN_DT_RANK);
    config_expect_u32("ssm.inner_size", required_u32(m, "qwen35.ssm.inner_size"), DS4_N_GDN_INNER);
    config_expect_u32("full_attention_interval", required_u32(m, "qwen35.full_attention_interval"), DS4_N_FULL_ATTN_INTERVAL);
    config_expect_f32("rope.freq_base", required_f32(m, "qwen35.rope.freq_base"), DS4_ROPE_FREQ_BASE);
    config_expect_f32("attention.layer_norm_rms_epsilon",
                      required_f32(m, "qwen35.attention.layer_norm_rms_epsilon"),
                      DS4_RMS_EPS);
}

static void weights_bind_qwen35_layer(ds4_layer_weights *l, const ds4_model *m, uint32_t il) {
    l->attn_norm = required_tensorf(m, "blk.%u.attn_norm.weight", il);
    l->ffn_norm  = required_tensorf(m, "blk.%u.post_attention_norm.weight", il);
    l->ffn_gate  = required_tensorf(m, "blk.%u.ffn_gate.weight", il);
    l->ffn_up    = required_tensorf(m, "blk.%u.ffn_up.weight", il);
    l->ffn_down  = required_tensorf(m, "blk.%u.ffn_down.weight", il);

    if ((il + 1u) % DS4_N_FULL_ATTN_INTERVAL == 0) {
        l->qwen_attn_q      = required_tensorf(m, "blk.%u.attn_q.weight", il);
        l->qwen_attn_k      = required_tensorf(m, "blk.%u.attn_k.weight", il);
        l->qwen_attn_v      = required_tensorf(m, "blk.%u.attn_v.weight", il);
        l->qwen_attn_q_norm = required_tensorf(m, "blk.%u.attn_q_norm.weight", il);
        l->qwen_attn_k_norm = required_tensorf(m, "blk.%u.attn_k_norm.weight", il);
        l->attn_output      = required_tensorf(m, "blk.%u.attn_output.weight", il);
    } else {
        l->qwen_attn_gate   = required_tensorf(m, "blk.%u.attn_gate.weight", il);
        l->qwen_attn_qkv    = required_tensorf(m, "blk.%u.attn_qkv.weight", il);
        l->qwen_ssm_a       = required_tensorf(m, "blk.%u.ssm_a", il);
        l->qwen_ssm_alpha   = required_tensorf(m, "blk.%u.ssm_alpha.weight", il);
        l->qwen_ssm_beta    = required_tensorf(m, "blk.%u.ssm_beta.weight", il);
        l->qwen_ssm_conv1d  = required_tensorf(m, "blk.%u.ssm_conv1d.weight", il);
        l->qwen_ssm_dt      = required_tensorf(m, "blk.%u.ssm_dt.bias", il);
        l->qwen_ssm_norm    = required_tensorf(m, "blk.%u.ssm_norm.weight", il);
        l->qwen_ssm_out     = required_tensorf(m, "blk.%u.ssm_out.weight", il);
    }
}

static void weights_free(ds4_weights *w) {
    memset(w, 0, sizeof(*w));
}

static int sample_argmax(const float *logits, uint32_t n_vocab);

#define DS4_QWEN_PREFILL_CHUNK_DEFAULT 1024u
#define DS4_QWEN_PREFILL_CHUNK_MAX 1024u
#define DS4_QWEN_GDN_WY_CHUNK 64u

static uint32_t qwen_prefill_chunk_tokens(void) {
    const char *env = getenv("DS4_QWEN_PREFILL_CHUNK");
    if (!env || !env[0]) return DS4_QWEN_PREFILL_CHUNK_DEFAULT;
    char *end = NULL;
    errno = 0;
    unsigned long value = strtoul(env, &end, 10);
    if (errno != 0 || !end || *end != '\0' || value == 0 ||
        value > DS4_QWEN_PREFILL_CHUNK_MAX) {
        fprintf(stderr,
                "ds4: ignoring invalid DS4_QWEN_PREFILL_CHUNK=%s "
                "(expected 1..%u)\n",
                env, DS4_QWEN_PREFILL_CHUNK_MAX);
        return DS4_QWEN_PREFILL_CHUNK_DEFAULT;
    }
    return (uint32_t)value;
}

static bool qwen_batched_prefill_enabled(void) {
    const char *env = getenv("DS4_QWEN_BATCHED_PREFILL");
    return !env || (env[0] != '\0' && strcmp(env, "0") != 0 &&
                    strcasecmp(env, "false") != 0 &&
                    strcasecmp(env, "off") != 0);
}

static bool qwen_gdn_chunkwise_enabled(void) {
    const char *env = getenv("DS4_QWEN_GDN_CHUNKWISE");
    return !env || (env[0] != '\0' && strcmp(env, "0") != 0 &&
                    strcasecmp(env, "false") != 0 &&
                    strcasecmp(env, "off") != 0);
}

static bool qwen_gdn_chunkwise_eligible(uint32_t n_tokens) {
    return qwen_gdn_chunkwise_enabled() && n_tokens >= 16u &&
           n_tokens <= DS4_QWEN_GDN_WY_CHUNK && (n_tokens & 7u) == 0u;
}

static bool qwen_flash_attention_enabled(void) {
    const char *env = getenv("DS4_QWEN_FLASH_ATTN");
    return !env || (env[0] != '\0' && strcmp(env, "0") != 0 &&
                    strcasecmp(env, "false") != 0 &&
                    strcasecmp(env, "off") != 0);
}

static bool qwen_prefill_fused_residual_norm_enabled(void) {
    const char *env = getenv("DS4_QWEN_PREFILL_FUSED_RESIDUAL_NORM");
    return !env || (env[0] != '\0' && strcmp(env, "0") != 0 &&
                    strcasecmp(env, "false") != 0 &&
                    strcasecmp(env, "off") != 0);
}

static bool qwen_prefill_ffn_mid_f16_enabled(void) {
    return ds4_gpu_device_is_pre_m5_apple_silicon() &&
           getenv("DS4_QWEN_DISABLE_FFN_MID_F16") == NULL;
}

static bool qwen_decode_experiment_enabled(const char *name) {
    const char *env = getenv(name);
    return env && env[0] != '\0' && strcmp(env, "0") != 0 &&
           strcasecmp(env, "false") != 0 &&
           strcasecmp(env, "off") != 0;
}

static bool qwen_decode_fused_ffn_enabled(void) {
    return qwen_decode_experiment_enabled("DS4_METAL_QWEN_DECODE_FUSIONS") ||
           qwen_decode_experiment_enabled("DS4_METAL_QWEN_DECODE_FUSED_FFN");
}

static bool qwen_decode_fused_residual_norm_enabled(void) {
    return qwen_decode_experiment_enabled("DS4_METAL_QWEN_DECODE_FUSIONS") ||
           qwen_decode_experiment_enabled(
               "DS4_METAL_QWEN_DECODE_FUSED_RESIDUAL_NORM");
}

static uint32_t qwen_runtime_prefill_cap(void) {
    return qwen_batched_prefill_enabled() ?
        qwen_prefill_chunk_tokens() : 1u;
}

/* Qwen uses a GPT-2 byte mapping plus ranked BPE merges stored in GGUF. */
typedef struct ds4_vocab ds4_vocab;

typedef struct {
    ds4_str key;
    int value;
    bool used;
} str_i32_entry;

typedef struct {
    str_i32_entry *entry;
    uint64_t cap;
    uint64_t used;
} str_i32_table;

static uint64_t next_pow2(uint64_t n) {
    uint64_t p = 1;
    while (p < n) p <<= 1;
    return p;
}

static void table_init(str_i32_table *t, uint64_t expected) {
    t->cap = next_pow2(expected * 2 + 16);
    t->used = 0;
    t->entry = xcalloc((size_t)t->cap, sizeof(t->entry[0]));
}

static void table_free(str_i32_table *t) {
    free(t->entry);
    memset(t, 0, sizeof(*t));
}

static void table_put(str_i32_table *t, ds4_str key, int value) {
    uint64_t mask = t->cap - 1;
    uint64_t i = hash_bytes(key.ptr, key.len) & mask;

    while (t->entry[i].used) {
        if (ds4_str_eq(t->entry[i].key, key)) {
            t->entry[i].value = value;
            return;
        }
        i = (i + 1) & mask;
    }

    t->entry[i].used = true;
    t->entry[i].key = key;
    t->entry[i].value = value;
    t->used++;
}

static bool table_get(const str_i32_table *t, const char *ptr, uint64_t len, int *value) {
    if (t->cap == 0) return false;

    uint64_t mask = t->cap - 1;
    uint64_t i = hash_bytes(ptr, len) & mask;

    while (t->entry[i].used) {
        ds4_str key = t->entry[i].key;
        if (key.len == len && memcmp(key.ptr, ptr, len) == 0) {
            *value = t->entry[i].value;
            return true;
        }
        i = (i + 1) & mask;
    }
    return false;
}

static void token_vec_push(token_vec *tv, int token) {
    if (tv->len == tv->cap) {
        tv->cap = tv->cap ? tv->cap * 2 : 64;
        tv->v = xrealloc(tv->v, (size_t)tv->cap * sizeof(tv->v[0]));
    }
    tv->v[tv->len++] = token;
}

static void token_vec_free(token_vec *tv) {
    free(tv->v);
    memset(tv, 0, sizeof(*tv));
}

struct ds4_vocab {
    ds4_str *token;
    int n_vocab;
    int eos_id;
    int system_id;
    int user_id;
    int assistant_id;
    int think_start_id;
    int think_end_id;
    str_i32_table token_to_id;
    str_i32_table merge_rank;
};

static void utf8_put(char **p, uint32_t cp) {
    if (cp <= 0x7f) {
        *(*p)++ = (char)cp;
    } else if (cp <= 0x7ff) {
        *(*p)++ = (char)(0xc0 | (cp >> 6));
        *(*p)++ = (char)(0x80 | (cp & 0x3f));
    } else if (cp <= 0xffff) {
        *(*p)++ = (char)(0xe0 | (cp >> 12));
        *(*p)++ = (char)(0x80 | ((cp >> 6) & 0x3f));
        *(*p)++ = (char)(0x80 | (cp & 0x3f));
    } else {
        *(*p)++ = (char)(0xf0 | (cp >> 18));
        *(*p)++ = (char)(0x80 | ((cp >> 12) & 0x3f));
        *(*p)++ = (char)(0x80 | ((cp >> 6) & 0x3f));
        *(*p)++ = (char)(0x80 | (cp & 0x3f));
    }
}

static uint32_t gpt2_byte_to_codepoint(uint8_t b) {
    if ((b >= 33 && b <= 126) || (b >= 161 && b <= 172) || (b >= 174)) {
        return b;
    }

    uint32_t n = 0;
    for (uint32_t x = 0; x < 256; x++) {
        if ((x >= 33 && x <= 126) || (x >= 161 && x <= 172) || (x >= 174)) {
            continue;
        }
        if (x == b) return 256 + n;
        n++;
    }
    return b;
}

static char *byte_encode(ds4_str in, uint64_t *out_len) {
    char *out = xmalloc((size_t)in.len * 4 + 1);
    char *p = out;

    for (uint64_t i = 0; i < in.len; i++) {
        utf8_put(&p, gpt2_byte_to_codepoint((uint8_t)in.ptr[i]));
    }
    *p = '\0';
    *out_len = (uint64_t)(p - out);
    return out;
}

static int utf8_len_from_first_byte(uint8_t c) {
    if (c < 0x80) return 1;
    if ((c & 0xe0) == 0xc0) return 2;
    if ((c & 0xf0) == 0xe0) return 3;
    if ((c & 0xf8) == 0xf0) return 4;
    return 1;
}

typedef struct {
    char *ptr;
    uint64_t len;
} owned_str;

static owned_str owned_copy(const char *ptr, uint64_t len) {
    owned_str s;
    s.ptr = xmalloc((size_t)len);
    memcpy(s.ptr, ptr, (size_t)len);
    s.len = len;
    return s;
}

static int bpe_rank(const ds4_vocab *vocab, const owned_str *a, const owned_str *b) {
    uint64_t len = a->len + 1 + b->len;
    char stack[512];
    char *buf = len <= sizeof(stack) ? stack : xmalloc((size_t)len);

    memcpy(buf, a->ptr, (size_t)a->len);
    buf[a->len] = ' ';
    memcpy(buf + a->len + 1, b->ptr, (size_t)b->len);

    int rank = -1;
    table_get(&vocab->merge_rank, buf, len, &rank);

    if (buf != stack) free(buf);
    return rank;
}

static void bpe_emit_piece(const ds4_vocab *vocab, ds4_str raw_piece, token_vec *out) {
    uint64_t encoded_len = 0;
    char *encoded = byte_encode(raw_piece, &encoded_len);

    int n_sym = 0;
    int cap_sym = 32;
    owned_str *sym = xcalloc((size_t)cap_sym, sizeof(sym[0]));

    for (uint64_t off = 0; off < encoded_len;) {
        int n = utf8_len_from_first_byte((uint8_t)encoded[off]);
        if (off + (uint64_t)n > encoded_len) n = 1;
        if (n_sym == cap_sym) {
            cap_sym *= 2;
            sym = xrealloc(sym, (size_t)cap_sym * sizeof(sym[0]));
        }
        sym[n_sym++] = owned_copy(encoded + off, (uint64_t)n);
        off += (uint64_t)n;
    }

    for (;;) {
        int best_i = -1;
        int best_rank = INT32_MAX;

        for (int i = 0; i + 1 < n_sym; i++) {
            int rank = bpe_rank(vocab, &sym[i], &sym[i + 1]);
            if (rank >= 0 && rank < best_rank) {
                best_rank = rank;
                best_i = i;
            }
        }

        if (best_i < 0) break;

        owned_str merged;
        merged.len = sym[best_i].len + sym[best_i + 1].len;
        merged.ptr = xmalloc((size_t)merged.len);
        memcpy(merged.ptr, sym[best_i].ptr, (size_t)sym[best_i].len);
        memcpy(merged.ptr + sym[best_i].len, sym[best_i + 1].ptr, (size_t)sym[best_i + 1].len);

        free(sym[best_i].ptr);
        free(sym[best_i + 1].ptr);
        sym[best_i] = merged;

        for (int j = best_i + 1; j + 1 < n_sym; j++) {
            sym[j] = sym[j + 1];
        }
        n_sym--;
    }

    for (int i = 0; i < n_sym; i++) {
        int token = -1;
        if (table_get(&vocab->token_to_id, sym[i].ptr, sym[i].len, &token)) {
            token_vec_push(out, token);
        } else {
            for (uint64_t j = 0; j < sym[i].len; j++) {
                if (table_get(&vocab->token_to_id, sym[i].ptr + j, 1, &token)) {
                    token_vec_push(out, token);
                }
            }
        }
        free(sym[i].ptr);
    }

    free(sym);
    free(encoded);
}

static uint64_t next_utf8_char(const char *s, uint64_t len, uint64_t pos) {
    int n = utf8_len_from_first_byte((uint8_t)s[pos]);
    if (pos + (uint64_t)n > len) n = 1;
    return pos + (uint64_t)n;
}

static bool ascii_alpha(uint8_t c) {
    return (c >= 'A' && c <= 'Z') || (c >= 'a' && c <= 'z');
}

static bool ascii_digit(uint8_t c) {
    return c >= '0' && c <= '9';
}

static bool ascii_space(uint8_t c) {
    return c == ' ' || c == '\t' || c == '\n' || c == '\r' ||
           c == '\v' || c == '\f';
}

static bool qwen_ascii_punct_symbol(uint8_t c) {
    return (c >= '!' && c <= '/') ||
           (c >= ':' && c <= '@') ||
           (c >= '[' && c <= '`') ||
           (c >= '{' && c <= '~');
}

static uint32_t utf8_peek_one(const char *s, uint64_t len, uint64_t pos, uint64_t *next) {
    const uint8_t c0 = (uint8_t)s[pos];
    int n = utf8_len_from_first_byte(c0);
    if (pos + (uint64_t)n > len) n = 1;
    *next = pos + (uint64_t)n;

    if (n == 1) return c0;
    if (n == 2) {
        return ((uint32_t)(c0 & 0x1f) << 6) |
               ((uint32_t)((uint8_t)s[pos + 1] & 0x3f));
    }
    if (n == 3) {
        return ((uint32_t)(c0 & 0x0f) << 12) |
               ((uint32_t)((uint8_t)s[pos + 1] & 0x3f) << 6) |
               ((uint32_t)((uint8_t)s[pos + 2] & 0x3f));
    }
    return ((uint32_t)(c0 & 0x07) << 18) |
           ((uint32_t)((uint8_t)s[pos + 1] & 0x3f) << 12) |
           ((uint32_t)((uint8_t)s[pos + 2] & 0x3f) << 6) |
           ((uint32_t)((uint8_t)s[pos + 3] & 0x3f));
}

typedef struct {
    uint32_t cp;
    uint64_t next;
    bool valid;
    bool is_letter;
    bool is_number;
    bool is_whitespace;
} qwen_char_info;

static bool qwen_unicode_whitespace(uint32_t cp) {
    if (cp < 128) return ascii_space((uint8_t)cp);
    return cp == 0x0085 ||
           cp == 0x00a0 ||
           cp == 0x1680 ||
           (cp >= 0x2000 && cp <= 0x200a) ||
           cp == 0x2028 ||
           cp == 0x2029 ||
           cp == 0x202f ||
           cp == 0x205f ||
           cp == 0x3000;
}

static bool qwen_unicode_number(uint32_t cp) {
    if (cp < 128) return ascii_digit((uint8_t)cp);
    return (cp >= 0x0660 && cp <= 0x0669) ||
           (cp >= 0x06f0 && cp <= 0x06f9) ||
           (cp >= 0x07c0 && cp <= 0x07c9) ||
           (cp >= 0x0966 && cp <= 0x096f) ||
           (cp >= 0x09e6 && cp <= 0x09ef) ||
           (cp >= 0x0a66 && cp <= 0x0a6f) ||
           (cp >= 0x0ae6 && cp <= 0x0aef) ||
           (cp >= 0x0b66 && cp <= 0x0b6f) ||
           (cp >= 0x0be6 && cp <= 0x0bef) ||
           (cp >= 0x0c66 && cp <= 0x0c6f) ||
           (cp >= 0x0ce6 && cp <= 0x0cef) ||
           (cp >= 0x0d66 && cp <= 0x0d6f) ||
           (cp >= 0x0de6 && cp <= 0x0def) ||
           (cp >= 0x0e50 && cp <= 0x0e59) ||
           (cp >= 0x0ed0 && cp <= 0x0ed9) ||
           (cp >= 0x0f20 && cp <= 0x0f29) ||
           (cp >= 0x1040 && cp <= 0x1049) ||
           (cp >= 0x1090 && cp <= 0x1099) ||
           (cp >= 0x17e0 && cp <= 0x17e9) ||
           (cp >= 0x1810 && cp <= 0x1819) ||
           (cp >= 0xff10 && cp <= 0xff19);
}

static bool qwen_unicode_punct_symbol(uint32_t cp) {
    if (cp < 128) return qwen_ascii_punct_symbol((uint8_t)cp);
    return (cp >= 0x00a1 && cp <= 0x00a9) ||
           (cp >= 0x00ab && cp <= 0x00ac) ||
           (cp >= 0x00ae && cp <= 0x00b1) ||
           cp == 0x00b4 ||
           (cp >= 0x00b6 && cp <= 0x00b8) ||
           cp == 0x00bb ||
           cp == 0x00bf ||
           cp == 0x00d7 ||
           cp == 0x00f7 ||
           (cp >= 0x02c2 && cp <= 0x02df) ||
           (cp >= 0x02e5 && cp <= 0x02eb) ||
           (cp >= 0x02ed && cp <= 0x02ff) ||
           (cp >= 0x0375 && cp <= 0x037e) ||
           (cp >= 0x0384 && cp <= 0x0385) ||
           cp == 0x0387 ||
           (cp >= 0x055a && cp <= 0x055f) ||
           (cp >= 0x0589 && cp <= 0x058a) ||
           (cp >= 0x05be && cp <= 0x05c0) ||
           cp == 0x05c3 ||
           (cp >= 0x05c6 && cp <= 0x05c7) ||
           (cp >= 0x0609 && cp <= 0x060a) ||
           (cp >= 0x060c && cp <= 0x060d) ||
           cp == 0x061b ||
           (cp >= 0x061e && cp <= 0x061f) ||
           cp == 0x066a ||
           cp == 0x066d ||
           cp == 0x06d4 ||
           (cp >= 0x2000 && cp <= 0x206f) ||
           (cp >= 0x20a0 && cp <= 0x20cf) ||
           (cp >= 0x2100 && cp <= 0x214f) ||
           (cp >= 0x2190 && cp <= 0x23ff) ||
           (cp >= 0x2460 && cp <= 0x24ff) ||
           (cp >= 0x2500 && cp <= 0x2775) ||
           (cp >= 0x2794 && cp <= 0x2bff) ||
           (cp >= 0x2e00 && cp <= 0x2e7f) ||
           (cp >= 0x3000 && cp <= 0x303f) ||
           (cp >= 0xfd3e && cp <= 0xfd3f) ||
           (cp >= 0xfe10 && cp <= 0xfe6f) ||
           (cp >= 0xff01 && cp <= 0xff0f) ||
           (cp >= 0xff1a && cp <= 0xff20) ||
           (cp >= 0xff3b && cp <= 0xff40) ||
           (cp >= 0xff5b && cp <= 0xff65) ||
           (cp >= 0x1f000 && cp <= 0x1faff);
}

static qwen_char_info qwen_char_at(const char *s, uint64_t len, uint64_t pos) {
    qwen_char_info info;
    memset(&info, 0, sizeof(info));
    if (pos >= len) return info;

    info.valid = true;
    info.cp = utf8_peek_one(s, len, pos, &info.next);
    info.is_whitespace = qwen_unicode_whitespace(info.cp);
    info.is_number = qwen_unicode_number(info.cp);
    if (info.cp < 128) {
        info.is_letter = ascii_alpha((uint8_t)info.cp);
    } else {
        info.is_letter =
            !info.is_whitespace &&
            !info.is_number &&
            !qwen_unicode_punct_symbol(info.cp);
    }
    return info;
}

static uint32_t ascii_tolower_cp(uint32_t cp) {
    if (cp >= 'A' && cp <= 'Z') return cp + ('a' - 'A');
    return cp;
}

static void qwen_bpe_tokenize_text(const ds4_vocab *vocab, const char *text,
                                   int max_digits, token_vec *out) {
    const uint64_t len = strlen(text);
    uint64_t pos = 0;

    while (pos < len) {
        uint64_t start = pos;
        qwen_char_info cur = qwen_char_at(text, len, pos);

        if (!cur.valid) break;

        if (cur.cp == '\'' && cur.next < len) {
            qwen_char_info next = qwen_char_at(text, len, cur.next);
            uint32_t n1 = ascii_tolower_cp(next.cp);
            if (n1 == 's' || n1 == 't' || n1 == 'm' || n1 == 'd') {
                pos = next.next;
                bpe_emit_piece(vocab, (ds4_str){ text + start, pos - start }, out);
                continue;
            }
            if (next.valid && next.next < len) {
                qwen_char_info next2 = qwen_char_at(text, len, next.next);
                uint32_t n2 = ascii_tolower_cp(next2.cp);
                if ((n1 == 'r' && n2 == 'e') ||
                    (n1 == 'v' && n2 == 'e') ||
                    (n1 == 'l' && n2 == 'l')) {
                    pos = next2.next;
                    bpe_emit_piece(vocab, (ds4_str){ text + start, pos - start }, out);
                    continue;
                }
            }
        }

        if (!(cur.cp == '\r' || cur.cp == '\n' || cur.is_number)) {
            qwen_char_info next = qwen_char_at(text, len, cur.next);
            if (cur.is_letter || next.is_letter) {
                pos = cur.next;
                while (pos < len) {
                    qwen_char_info scan = qwen_char_at(text, len, pos);
                    if (!scan.valid || !scan.is_letter) break;
                    pos = scan.next;
                }
                bpe_emit_piece(vocab, (ds4_str){ text + start, pos - start }, out);
                continue;
            }
        }

        if (cur.is_number) {
            int ndigits = 0;
            while (pos < len && ndigits < max_digits) {
                qwen_char_info scan = qwen_char_at(text, len, pos);
                if (!scan.valid || !scan.is_number) break;
                pos = scan.next;
                ndigits++;
            }
            bpe_emit_piece(vocab, (ds4_str){ text + start, pos - start }, out);
            continue;
        }

        qwen_char_info punct = cur;
        uint64_t punct_pos = pos;
        if (cur.cp == ' ') {
            punct_pos = cur.next;
            punct = qwen_char_at(text, len, punct_pos);
        }
        if (punct.valid &&
            !punct.is_whitespace &&
            !punct.is_letter &&
            !punct.is_number) {
            pos = punct_pos;
            while (pos < len) {
                qwen_char_info scan = qwen_char_at(text, len, pos);
                if (!scan.valid ||
                    scan.is_whitespace ||
                    scan.is_letter ||
                    scan.is_number) {
                    break;
                }
                pos = scan.next;
            }
            while (pos < len) {
                qwen_char_info scan = qwen_char_at(text, len, pos);
                if (!scan.valid || !(scan.cp == '\r' || scan.cp == '\n')) break;
                pos = scan.next;
            }
            bpe_emit_piece(vocab, (ds4_str){ text + start, pos - start }, out);
            continue;
        }

        if (cur.is_whitespace) {
            uint64_t p = pos;
            uint64_t last_newline_end = 0;
            uint64_t last_ws_start = pos;
            int nspace = 0;
            while (p < len) {
                qwen_char_info scan = qwen_char_at(text, len, p);
                if (!scan.valid || !scan.is_whitespace) break;
                last_ws_start = p;
                if (scan.cp == '\r' || scan.cp == '\n') last_newline_end = scan.next;
                p = scan.next;
                nspace++;
            }
            if (last_newline_end) {
                pos = last_newline_end;
            } else if (nspace > 1 && p < len) {
                pos = last_ws_start;
            } else {
                pos = p;
            }
            bpe_emit_piece(vocab, (ds4_str){ text + start, pos - start }, out);
            continue;
        }

        pos = cur.next;
        if (pos == start) pos = next_utf8_char(text, len, pos);
        bpe_emit_piece(vocab, (ds4_str){ text + start, pos - start }, out);
    }
}

static int vocab_lookup_optional(const ds4_vocab *vocab, const char *text) {
    int token = -1;
    if (!table_get(&vocab->token_to_id, text, strlen(text), &token)) return -1;
    return token;
}

static void vocab_free(ds4_vocab *vocab) {
    free(vocab->token);
    table_free(&vocab->token_to_id);
    table_free(&vocab->merge_rank);
    memset(vocab, 0, sizeof(*vocab));
}

static uint32_t utf8_decode_one(const char *s, uint64_t len, uint64_t *pos) {
    const uint8_t c = (uint8_t)s[*pos];
    if (c < 0x80 || *pos + 1 >= len) {
        (*pos)++;
        return c;
    }
    if ((c & 0xe0) == 0xc0 && *pos + 1 < len) {
        uint32_t cp = ((uint32_t)(c & 0x1f) << 6) | ((uint8_t)s[*pos + 1] & 0x3f);
        *pos += 2;
        return cp;
    }
    if ((c & 0xf0) == 0xe0 && *pos + 2 < len) {
        uint32_t cp = ((uint32_t)(c & 0x0f) << 12) |
                      ((uint32_t)((uint8_t)s[*pos + 1] & 0x3f) << 6) |
                      ((uint8_t)s[*pos + 2] & 0x3f);
        *pos += 3;
        return cp;
    }
    if ((c & 0xf8) == 0xf0 && *pos + 3 < len) {
        uint32_t cp = ((uint32_t)(c & 0x07) << 18) |
                      ((uint32_t)((uint8_t)s[*pos + 1] & 0x3f) << 12) |
                      ((uint32_t)((uint8_t)s[*pos + 2] & 0x3f) << 6) |
                      ((uint8_t)s[*pos + 3] & 0x3f);
        *pos += 4;
        return cp;
    }
    (*pos)++;
    return c;
}

static int gpt2_codepoint_to_byte(uint32_t cp) {
    if ((cp >= 33 && cp <= 126) || (cp >= 161 && cp <= 172) || (cp >= 174 && cp <= 255)) {
        return (int)cp;
    }

    uint32_t n = 0;
    for (uint32_t b = 0; b < 256; b++) {
        if ((b >= 33 && b <= 126) || (b >= 161 && b <= 172) || (b >= 174)) {
            continue;
        }
        if (cp == 256 + n) return (int)b;
        n++;
    }
    return -1;
}

static bool vocab_token_is_literal_special(ds4_str s) {
    const unsigned char bar[] = {0xef, 0xbd, 0x9c};
    if (s.len < sizeof(bar)) return false;
    for (uint64_t i = 0; i + sizeof(bar) <= s.len; i++) {
        if (!memcmp(s.ptr + i, bar, sizeof(bar))) return true;
    }
    return false;
}

static inline void argmax_f32_unrolled8_range(
        const float *logits,
        uint32_t     begin,
        uint32_t     end,
        int         *best,
        float       *best_v) {
    uint32_t i = begin;
    int b0 = *best, b1 = *best, b2 = *best, b3 = *best;
    int b4 = *best, b5 = *best, b6 = *best, b7 = *best;
    float v0 = *best_v, v1 = *best_v, v2 = *best_v, v3 = *best_v;
    float v4 = *best_v, v5 = *best_v, v6 = *best_v, v7 = *best_v;

    while (end - i >= 8u) {
        const float x0 = logits[i + 0u];
        const float x1 = logits[i + 1u];
        const float x2 = logits[i + 2u];
        const float x3 = logits[i + 3u];
        const float x4 = logits[i + 4u];
        const float x5 = logits[i + 5u];
        const float x6 = logits[i + 6u];
        const float x7 = logits[i + 7u];
        if (x0 > v0) { v0 = x0; b0 = (int)(i + 0u); }
        if (x1 > v1) { v1 = x1; b1 = (int)(i + 1u); }
        if (x2 > v2) { v2 = x2; b2 = (int)(i + 2u); }
        if (x3 > v3) { v3 = x3; b3 = (int)(i + 3u); }
        if (x4 > v4) { v4 = x4; b4 = (int)(i + 4u); }
        if (x5 > v5) { v5 = x5; b5 = (int)(i + 5u); }
        if (x6 > v6) { v6 = x6; b6 = (int)(i + 6u); }
        if (x7 > v7) { v7 = x7; b7 = (int)(i + 7u); }
        i += 8u;
    }

#define DS4_ARGMAX_MERGE_LANE(b, v) \
    do { \
        if ((v) > *best_v || ((v) == *best_v && (b) < *best)) { \
            *best_v = (v); \
            *best = (b); \
        } \
    } while (0)
    DS4_ARGMAX_MERGE_LANE(b0, v0);
    DS4_ARGMAX_MERGE_LANE(b1, v1);
    DS4_ARGMAX_MERGE_LANE(b2, v2);
    DS4_ARGMAX_MERGE_LANE(b3, v3);
    DS4_ARGMAX_MERGE_LANE(b4, v4);
    DS4_ARGMAX_MERGE_LANE(b5, v5);
    DS4_ARGMAX_MERGE_LANE(b6, v6);
    DS4_ARGMAX_MERGE_LANE(b7, v7);
#undef DS4_ARGMAX_MERGE_LANE

    for (; i < end; i++) {
        const float v = logits[i];
        if (v > *best_v) {
            *best_v = v;
            *best = (int)i;
        }
    }
}

static int sample_argmax_unrolled8(const float *logits, uint32_t n_vocab) {
    int best = 0;
    float best_v = DS4_NEG_INF;
    argmax_f32_unrolled8_range(logits, 0, n_vocab, &best, &best_v);
    return best;
}

static int sample_argmax(const float *logits, uint32_t n_vocab) {
    if (getenv("DS4_CPU_DISABLE_UNROLLED_ARGMAX") == NULL) {
        return sample_argmax_unrolled8(logits, n_vocab);
    }
    int best = 0;
    float best_v = DS4_NEG_INF;
    for (uint32_t i = 0; i < n_vocab; i++) {
        const float v = logits[i];
        if (v > best_v) {
            best_v = v;
            best = (int)i;
        }
    }
    return best;
}

static void print_top_logits(
        FILE          * fp,
        const char    * label,
        const ds4_vocab * vocab,
        const float   * logits,
        uint32_t        n_vocab,
        int             k) {
    int best[16];
    if (k > 16) k = 16;
    for (int i = 0; i < k; i++) best[i] = -1;

    for (uint32_t i = 0; i < n_vocab; i++) {
        for (int j = 0; j < k; j++) {
            if (best[j] < 0 || logits[i] > logits[best[j]]) {
                for (int l = k - 1; l > j; l--) best[l] = best[l - 1];
                best[j] = (int)i;
                break;
            }
        }
    }

    fprintf(fp, "ds4: top logits %s:\n", label);
    for (int i = 0; i < k && best[i] >= 0; i++) {
        const int id = best[i];
        fprintf(fp, "  %2d %7d % .9g  ", i, id, logits[id]);
        if (id >= 0 && id < vocab->n_vocab) {
            fprintf(fp, "%.*s", (int)vocab->token[id].len, vocab->token[id].ptr);
        }
        fputc('\n', fp);
    }
}

static int generate_qwen_metal_argmax(
        const ds4_model   * model,
        const ds4_vocab   * vocab,
        const ds4_weights * weights,
        const token_vec   * prompt,
        int                 n_predict,
        int                 ctx_size,
        ds4_token_emit_fn   emit,
        ds4_generation_done_fn done,
        void              * emit_ud,
        ds4_session_progress_fn progress,
        void              * progress_ud);

bool ds4_think_mode_enabled(ds4_think_mode mode) {
    return mode == DS4_THINK_HIGH || mode == DS4_THINK_MAX;
}

/* One allocation owns all persistent activations and attention/GDN state.
 * Prefill and decode reuse it; neither hot loop performs host allocation. */
typedef struct {
    uint32_t ctx_cap;
    uint32_t prefill_cap;
    ds4_gpu_tensor *tokens;
    ds4_gpu_tensor *cur;
    ds4_gpu_tensor *attn_norm;
    ds4_gpu_tensor *attn_out;
    ds4_gpu_tensor *after_attn;
    ds4_gpu_tensor *ffn_norm;
    ds4_gpu_tensor *ffn_gate;
    ds4_gpu_tensor *ffn_up;
    ds4_gpu_tensor *ffn_mid;
    ds4_gpu_tensor *ffn_out;
    ds4_gpu_tensor *qkv;
    ds4_gpu_tensor *z;
    ds4_gpu_tensor *alpha;
    ds4_gpu_tensor *beta;
    ds4_gpu_tensor *gdn_prepared;
    ds4_gpu_tensor *gdn_g;
    ds4_gpu_tensor *gdn_b;
    ds4_gpu_tensor *gdn_recurrent;
    ds4_gpu_tensor *gdn_chunk_w;
    ds4_gpu_tensor *gdn_chunk_u;
    ds4_gpu_tensor *gdn_chunk_qk;
    ds4_gpu_tensor *gdn_chunk_cumulative_g;
    ds4_gpu_tensor *gdn_out;
    ds4_gpu_tensor *qg;
    ds4_gpu_tensor *k;
    ds4_gpu_tensor *v;
    ds4_gpu_tensor *q;
    ds4_gpu_tensor *q_gate;
    ds4_gpu_tensor *heads;
    ds4_gpu_tensor *scores;
    ds4_gpu_tensor *logits;
    ds4_gpu_tensor *key_cache[DS4_MAX_LAYER];
    ds4_gpu_tensor *value_cache[DS4_MAX_LAYER];
    ds4_gpu_tensor *conv_state[DS4_MAX_LAYER];
    ds4_gpu_tensor *ssm_state[DS4_MAX_LAYER];
} ds4_qwen_gpu_graph;

static void qwen_graph_free(ds4_qwen_gpu_graph *g) {
    if (!g) return;
#define DS4_QWEN_FREE(field_) do { ds4_gpu_tensor_free(g->field_); g->field_ = NULL; } while (0)
    DS4_QWEN_FREE(tokens);
    DS4_QWEN_FREE(cur);
    DS4_QWEN_FREE(attn_norm);
    DS4_QWEN_FREE(attn_out);
    DS4_QWEN_FREE(after_attn);
    DS4_QWEN_FREE(ffn_norm);
    DS4_QWEN_FREE(ffn_gate);
    DS4_QWEN_FREE(ffn_up);
    DS4_QWEN_FREE(ffn_mid);
    DS4_QWEN_FREE(ffn_out);
    DS4_QWEN_FREE(qkv);
    DS4_QWEN_FREE(z);
    DS4_QWEN_FREE(alpha);
    DS4_QWEN_FREE(beta);
    DS4_QWEN_FREE(gdn_prepared);
    DS4_QWEN_FREE(gdn_g);
    DS4_QWEN_FREE(gdn_b);
    DS4_QWEN_FREE(gdn_recurrent);
    DS4_QWEN_FREE(gdn_chunk_w);
    DS4_QWEN_FREE(gdn_chunk_u);
    DS4_QWEN_FREE(gdn_chunk_qk);
    DS4_QWEN_FREE(gdn_chunk_cumulative_g);
    DS4_QWEN_FREE(gdn_out);
    DS4_QWEN_FREE(qg);
    DS4_QWEN_FREE(k);
    DS4_QWEN_FREE(v);
    DS4_QWEN_FREE(q);
    DS4_QWEN_FREE(q_gate);
    DS4_QWEN_FREE(heads);
    DS4_QWEN_FREE(scores);
    DS4_QWEN_FREE(logits);
#undef DS4_QWEN_FREE
    for (uint32_t i = 0; i < DS4_MAX_LAYER; i++) {
        ds4_gpu_tensor_free(g->key_cache[i]);
        ds4_gpu_tensor_free(g->value_cache[i]);
        ds4_gpu_tensor_free(g->conv_state[i]);
        ds4_gpu_tensor_free(g->ssm_state[i]);
    }
    memset(g, 0, sizeof(*g));
}

static bool qwen_graph_alloc(ds4_qwen_gpu_graph *g, uint32_t ctx_cap,
                             uint32_t prefill_cap) {
    if (!g || ctx_cap == 0 || prefill_cap == 0) return false;
    if (prefill_cap > ctx_cap) prefill_cap = ctx_cap;
    memset(g, 0, sizeof(*g));
    g->ctx_cap = ctx_cap;
    g->prefill_cap = prefill_cap;
#define DS4_QWEN_ALLOC(field_, count_) do {                                      \
        g->field_ = ds4_gpu_tensor_alloc((uint64_t)(count_) * sizeof(float));     \
        if (!g->field_) goto fail;                                                \
    } while (0)
    g->tokens = ds4_gpu_tensor_alloc((uint64_t)prefill_cap * sizeof(uint32_t));
    if (!g->tokens) goto fail;
    DS4_QWEN_ALLOC(cur, (uint64_t)prefill_cap * DS4_N_EMBD);
    DS4_QWEN_ALLOC(attn_norm, (uint64_t)prefill_cap * DS4_N_EMBD);
    DS4_QWEN_ALLOC(attn_out, (uint64_t)prefill_cap * DS4_N_EMBD);
    DS4_QWEN_ALLOC(after_attn, (uint64_t)prefill_cap * DS4_N_EMBD);
    DS4_QWEN_ALLOC(ffn_norm, (uint64_t)prefill_cap * DS4_N_EMBD);
    DS4_QWEN_ALLOC(ffn_gate, (uint64_t)prefill_cap * DS4_N_FF_DENSE);
    DS4_QWEN_ALLOC(ffn_up, (uint64_t)prefill_cap * DS4_N_FF_DENSE);
    {
        uint64_t ffn_mid_bytes =
            (uint64_t)prefill_cap * DS4_N_FF_DENSE * sizeof(uint16_t);
        const uint64_t decode_mid_bytes =
            (uint64_t)DS4_N_FF_DENSE * sizeof(float);
        if (ffn_mid_bytes < decode_mid_bytes) ffn_mid_bytes = decode_mid_bytes;
        g->ffn_mid = ds4_gpu_tensor_alloc(ffn_mid_bytes);
        if (!g->ffn_mid) goto fail;
    }
    DS4_QWEN_ALLOC(ffn_out, (uint64_t)prefill_cap * DS4_N_EMBD);
    DS4_QWEN_ALLOC(qkv, (uint64_t)prefill_cap *
        (2u * DS4_N_GDN_QK_HEAD + DS4_N_GDN_V_HEAD) * DS4_N_GDN_STATE);
    DS4_QWEN_ALLOC(z, (uint64_t)prefill_cap * DS4_N_GDN_INNER);
    DS4_QWEN_ALLOC(alpha, (uint64_t)prefill_cap * DS4_N_GDN_DT_RANK);
    DS4_QWEN_ALLOC(beta, (uint64_t)prefill_cap * DS4_N_GDN_DT_RANK);
    DS4_QWEN_ALLOC(gdn_prepared, (uint64_t)prefill_cap *
        (2u * DS4_N_GDN_QK_HEAD + DS4_N_GDN_V_HEAD) * DS4_N_GDN_STATE);
    DS4_QWEN_ALLOC(gdn_g, (uint64_t)prefill_cap * DS4_N_GDN_V_HEAD);
    DS4_QWEN_ALLOC(gdn_b, (uint64_t)prefill_cap * DS4_N_GDN_V_HEAD);
    DS4_QWEN_ALLOC(gdn_recurrent, (uint64_t)prefill_cap * DS4_N_GDN_INNER);
    if (qwen_gdn_chunkwise_enabled()) {
        DS4_QWEN_ALLOC(gdn_chunk_w,
            (uint64_t)DS4_QWEN_GDN_WY_CHUNK * DS4_N_GDN_INNER);
        DS4_QWEN_ALLOC(gdn_chunk_u,
            (uint64_t)DS4_QWEN_GDN_WY_CHUNK * DS4_N_GDN_INNER);
        DS4_QWEN_ALLOC(gdn_chunk_qk,
            (uint64_t)DS4_QWEN_GDN_WY_CHUNK * DS4_QWEN_GDN_WY_CHUNK *
            DS4_N_GDN_V_HEAD);
        DS4_QWEN_ALLOC(gdn_chunk_cumulative_g,
            (uint64_t)DS4_QWEN_GDN_WY_CHUNK * DS4_N_GDN_V_HEAD);
    }
    DS4_QWEN_ALLOC(gdn_out, (uint64_t)prefill_cap * DS4_N_GDN_INNER);
    DS4_QWEN_ALLOC(qg, (uint64_t)prefill_cap * 2u * DS4_N_HEAD * DS4_N_HEAD_DIM);
    DS4_QWEN_ALLOC(k, (uint64_t)prefill_cap * DS4_N_HEAD_KV * DS4_N_HEAD_DIM);
    DS4_QWEN_ALLOC(v, (uint64_t)prefill_cap * DS4_N_HEAD_KV * DS4_N_VALUE_DIM);
    DS4_QWEN_ALLOC(q, (uint64_t)prefill_cap * DS4_N_HEAD * DS4_N_HEAD_DIM);
    DS4_QWEN_ALLOC(q_gate, (uint64_t)prefill_cap * DS4_N_HEAD * DS4_N_HEAD_DIM);
    DS4_QWEN_ALLOC(heads, (uint64_t)prefill_cap * DS4_N_HEAD * DS4_N_VALUE_DIM);
    DS4_QWEN_ALLOC(scores,
        (uint64_t)(qwen_flash_attention_enabled() ? 1u : prefill_cap) *
        DS4_N_HEAD * ctx_cap);
    DS4_QWEN_ALLOC(logits, DS4_N_VOCAB);
#undef DS4_QWEN_ALLOC

    uint32_t full_slot = 0;
    uint32_t linear_slot = 0;
    const uint64_t kv_elems = (uint64_t)ctx_cap * DS4_N_HEAD_KV * DS4_N_HEAD_DIM;
    const uint64_t conv_elems =
        (uint64_t)(2u * DS4_N_GDN_QK_HEAD + DS4_N_GDN_V_HEAD) *
        DS4_N_GDN_STATE * (DS4_N_GDN_CONV - 1u);
    const uint64_t state_elems =
        (uint64_t)DS4_N_GDN_V_HEAD * DS4_N_GDN_STATE * DS4_N_GDN_STATE;
    for (uint32_t il = 0; il < DS4_N_LAYER; il++) {
        if ((il + 1u) % DS4_N_FULL_ATTN_INTERVAL == 0) {
            g->key_cache[full_slot] = ds4_gpu_tensor_alloc(kv_elems * sizeof(uint16_t));
            g->value_cache[full_slot] = ds4_gpu_tensor_alloc(kv_elems * sizeof(uint16_t));
            if (!g->key_cache[full_slot] || !g->value_cache[full_slot]) goto fail;
            full_slot++;
        } else {
            g->conv_state[linear_slot] = ds4_gpu_tensor_alloc(conv_elems * sizeof(float));
            g->ssm_state[linear_slot] = ds4_gpu_tensor_alloc(state_elems * sizeof(float));
            if (!g->conv_state[linear_slot] || !g->ssm_state[linear_slot]) goto fail;
            if (!ds4_gpu_tensor_fill_f32(g->conv_state[linear_slot], 0.0f, conv_elems) ||
                !ds4_gpu_tensor_fill_f32(g->ssm_state[linear_slot], 0.0f, state_elems)) goto fail;
            linear_slot++;
        }
    }
    return true;
fail:
    qwen_graph_free(g);
    return false;
}

/* Decode one token through three GDN layers followed by each fourth full-
 * attention layer. Paired Q8 projections and optional fusions stay enabled. */
static bool qwen_graph_forward_token(
        ds4_qwen_gpu_graph *g,
        const ds4_model    *model,
        const ds4_weights  *weights,
        uint32_t            token,
        uint32_t            pos,
        float              *logits_out) {
    if (!g || !model || !weights || token >= DS4_N_VOCAB || pos >= g->ctx_cap) return false;
    bool ok = ds4_gpu_begin_commands() != 0;
    const bool fused_ffn = qwen_decode_fused_ffn_enabled();
    const bool fused_residual_norm =
        qwen_decode_fused_residual_norm_enabled();
    if (ok) ok = ds4_gpu_embed_token_q8_0_tensor(g->cur, model->map, model->size,
                                                   weights->token_embd->abs_offset,
                                                   DS4_N_VOCAB, token, DS4_N_EMBD) != 0;
    uint32_t full_slot = 0;
    uint32_t linear_slot = 0;
    const uint32_t qkv_dim =
        (2u * DS4_N_GDN_QK_HEAD + DS4_N_GDN_V_HEAD) * DS4_N_GDN_STATE;
    for (uint32_t il = 0; ok && il < DS4_N_LAYER; il++) {
        const ds4_layer_weights *l = &weights->layer[il];
        if (!fused_residual_norm || il == 0u) {
            ok = ds4_gpu_rms_norm_weight_tensor(g->attn_norm, g->cur,
                                                 model->map, model->size,
                                                 l->attn_norm->abs_offset,
                                                 DS4_N_EMBD, DS4_RMS_EPS) != 0;
        }
        if (!ok) break;
        if ((il + 1u) % DS4_N_FULL_ATTN_INTERVAL == 0) {
            ok = ds4_gpu_matmul_q8_0_pair_tensor(g->qg, g->k,
                    model->map, model->size,
                    l->qwen_attn_q->abs_offset, l->qwen_attn_k->abs_offset,
                    DS4_N_EMBD, 2u * DS4_N_HEAD * DS4_N_HEAD_DIM,
                    DS4_N_HEAD_KV * DS4_N_HEAD_DIM, g->attn_norm, 1) != 0;
            if (ok) ok = ds4_gpu_matmul_q8_0_tensor(g->v, model->map, model->size,
                    l->qwen_attn_v->abs_offset, DS4_N_EMBD,
                    DS4_N_HEAD_KV * DS4_N_VALUE_DIM, g->attn_norm, 1) != 0;
            if (ok) ok = ds4_gpu_qwen35_full_prepare_tensor(
                    g->q, g->q_gate, g->key_cache[full_slot],
                    g->value_cache[full_slot], g->qg, g->k, g->v,
                    model->map, model->size,
                    l->qwen_attn_q_norm->abs_offset,
                    l->qwen_attn_k_norm->abs_offset,
                    pos, g->ctx_cap, DS4_N_HEAD, DS4_N_HEAD_KV,
                    DS4_N_HEAD_DIM, DS4_N_ROT, DS4_RMS_EPS,
                    DS4_ROPE_FREQ_BASE) != 0;
            if (ok) ok = ds4_gpu_qwen35_attention_tensor(
                    g->heads, g->scores, g->q, g->q_gate,
                    g->key_cache[full_slot], g->value_cache[full_slot],
                    pos + 1u, g->ctx_cap, DS4_N_HEAD, DS4_N_HEAD_KV,
                    DS4_N_HEAD_DIM) != 0;
            if (ok) ok = ds4_gpu_matmul_q8_0_tensor(g->attn_out,
                    model->map, model->size, l->attn_output->abs_offset,
                    DS4_N_HEAD * DS4_N_VALUE_DIM, DS4_N_EMBD, g->heads, 1) != 0;
            full_slot++;
        } else {
            ok = ds4_gpu_matmul_q8_0_pair_tensor(g->qkv, g->z,
                    model->map, model->size,
                    l->qwen_attn_qkv->abs_offset, l->qwen_attn_gate->abs_offset,
                    DS4_N_EMBD, qkv_dim, DS4_N_GDN_INNER,
                    g->attn_norm, 1) != 0;
            if (ok) ok = ds4_gpu_matmul_q8_0_pair_tensor(g->alpha, g->beta,
                    model->map, model->size,
                    l->qwen_ssm_alpha->abs_offset, l->qwen_ssm_beta->abs_offset,
                    DS4_N_EMBD, DS4_N_GDN_DT_RANK, DS4_N_GDN_DT_RANK,
                    g->attn_norm, 1) != 0;
            if (ok) ok = ds4_gpu_qwen35_gdn_tensor(
                    g->gdn_out, g->gdn_prepared, g->gdn_g, g->gdn_b,
                    g->gdn_recurrent, g->conv_state[linear_slot],
                    g->ssm_state[linear_slot], g->qkv, g->z, g->alpha, g->beta,
                    model->map, model->size,
                    l->qwen_ssm_conv1d->abs_offset, l->qwen_ssm_dt->abs_offset,
                    l->qwen_ssm_a->abs_offset, l->qwen_ssm_norm->abs_offset,
                    qkv_dim, DS4_N_GDN_QK_HEAD, DS4_N_GDN_V_HEAD,
                    DS4_N_GDN_STATE, DS4_N_GDN_CONV, DS4_RMS_EPS) != 0;
            if (ok) ok = ds4_gpu_matmul_q8_0_tensor(g->attn_out,
                    model->map, model->size, l->qwen_ssm_out->abs_offset,
                    DS4_N_GDN_INNER, DS4_N_EMBD, g->gdn_out, 1) != 0;
            linear_slot++;
        }
        if (ok) ok = ds4_gpu_add_rms_norm_weight_tensor(
                g->ffn_norm, g->after_attn, g->cur, g->attn_out,
                model->map, model->size, l->ffn_norm->abs_offset,
                DS4_N_EMBD, DS4_RMS_EPS) != 0;
        if (ok && fused_ffn) {
            ok = ds4_gpu_shared_gate_up_swiglu_q8_0_tensor(
                    g->ffn_gate, g->ffn_up, g->ffn_mid,
                    model->map, model->size,
                    l->ffn_gate->abs_offset, l->ffn_up->abs_offset,
                    DS4_N_EMBD, DS4_N_FF_DENSE, g->ffn_norm, 0.0f) != 0;
        } else if (ok) {
            ok = ds4_gpu_matmul_q8_0_pair_tensor(
                    g->ffn_gate, g->ffn_up, model->map, model->size,
                    l->ffn_gate->abs_offset, l->ffn_up->abs_offset,
                    DS4_N_EMBD, DS4_N_FF_DENSE, DS4_N_FF_DENSE,
                    g->ffn_norm, 1) != 0;
            if (ok) ok = ds4_gpu_swiglu_tensor(
                    g->ffn_mid, g->ffn_gate, g->ffn_up,
                    DS4_N_FF_DENSE, 0.0f, 1.0f) != 0;
        }
        if (ok) ok = ds4_gpu_matmul_q8_0_tensor(g->ffn_out,
                model->map, model->size, l->ffn_down->abs_offset,
                DS4_N_FF_DENSE, DS4_N_EMBD, g->ffn_mid, 1) != 0;
        if (ok && fused_residual_norm && il + 1u < DS4_N_LAYER) {
            ok = ds4_gpu_add_rms_norm_weight_tensor(
                    g->attn_norm, g->cur, g->after_attn, g->ffn_out,
                    model->map, model->size,
                    weights->layer[il + 1u].attn_norm->abs_offset,
                    DS4_N_EMBD, DS4_RMS_EPS) != 0;
        } else if (ok) {
            ok = ds4_gpu_add_tensor(g->cur, g->after_attn, g->ffn_out,
                                     DS4_N_EMBD) != 0;
        }
    }
    if (ok && logits_out) {
        ok = ds4_gpu_rms_norm_weight_tensor(
                g->attn_norm, g->cur, model->map, model->size,
                weights->output_norm->abs_offset,
                DS4_N_EMBD, DS4_RMS_EPS) != 0;
        if (ok) ok = ds4_gpu_matmul_q8_0_tensor(g->logits,
                model->map, model->size, weights->output->abs_offset,
                DS4_N_EMBD, DS4_N_VOCAB, g->attn_norm, 1) != 0;
    }
    if (ok) ok = ds4_gpu_end_commands() != 0;
    if (ok && logits_out) {
        ok = ds4_gpu_tensor_read(g->logits, 0, logits_out,
                                  (uint64_t)DS4_N_VOCAB * sizeof(float)) != 0;
    }
    if (!ok) (void)ds4_gpu_synchronize();
    return ok;
}

static bool qwen_graph_gdn_wy_sliced(
        ds4_qwen_gpu_graph   *g,
        const ds4_model      *model,
        const ds4_layer_weights *l,
        uint32_t              linear_slot,
        uint32_t              n_tokens,
        uint32_t              qkv_dim) {
    if (!g || !model || !l || n_tokens == 0u) return false;
    bool ok = true;
    for (uint32_t row0 = 0; ok && row0 < n_tokens; ) {
        uint32_t rows = n_tokens - row0;
        if (rows > DS4_QWEN_GDN_WY_CHUNK) rows = DS4_QWEN_GDN_WY_CHUNK;
        const bool use_wy = qwen_gdn_chunkwise_eligible(rows);
        if (use_wy) {
            ok = ds4_gpu_qwen35_gdn_chunk_offset_tensor(
                    g->gdn_out, g->gdn_prepared, g->gdn_g, g->gdn_b,
                    g->gdn_recurrent, g->gdn_chunk_w, g->gdn_chunk_u,
                    g->gdn_chunk_qk, g->gdn_chunk_cumulative_g,
                    g->conv_state[linear_slot], g->ssm_state[linear_slot],
                    g->qkv, g->z, g->alpha, g->beta,
                    model->map, model->size,
                    l->qwen_ssm_conv1d->abs_offset,
                    l->qwen_ssm_dt->abs_offset,
                    l->qwen_ssm_a->abs_offset,
                    l->qwen_ssm_norm->abs_offset,
                    row0, rows, qkv_dim, DS4_N_GDN_QK_HEAD,
                    DS4_N_GDN_V_HEAD, DS4_N_GDN_STATE,
                    DS4_N_GDN_CONV, DS4_RMS_EPS) != 0;
        } else {
            ok = ds4_gpu_qwen35_gdn_batch_offset_tensor(
                    g->gdn_out, g->gdn_prepared, g->gdn_g, g->gdn_b,
                    g->gdn_recurrent,
                    g->conv_state[linear_slot], g->ssm_state[linear_slot],
                    g->qkv, g->z, g->alpha, g->beta,
                    model->map, model->size,
                    l->qwen_ssm_conv1d->abs_offset,
                    l->qwen_ssm_dt->abs_offset,
                    l->qwen_ssm_a->abs_offset,
                    l->qwen_ssm_norm->abs_offset,
                    row0, rows, qkv_dim, DS4_N_GDN_QK_HEAD,
                    DS4_N_GDN_V_HEAD, DS4_N_GDN_STATE,
                    DS4_N_GDN_CONV, DS4_RMS_EPS) != 0;
        }
        row0 += rows;
    }
    return ok;
}

/* Prefill follows the same graph in token batches. Eligible GDN slices use
 * the 64-row chunkwise WY kernels; full-attention layers use FlashAttention. */
static bool qwen_graph_forward_chunk(
        ds4_qwen_gpu_graph *g,
        const ds4_model    *model,
        const ds4_weights  *weights,
        const int          *tokens,
        uint32_t            pos0,
        uint32_t            n_tokens,
        float              *logits_out) {
    if (!g || !model || !weights || !tokens || n_tokens < 2u ||
        n_tokens > g->prefill_cap || pos0 >= g->ctx_cap ||
        n_tokens > g->ctx_cap - pos0) {
        return false;
    }
    for (uint32_t i = 0; i < n_tokens; i++) {
        if (tokens[i] < 0 || (uint32_t)tokens[i] >= DS4_N_VOCAB) return false;
    }
    if (!ds4_gpu_tensor_write(g->tokens, 0, tokens,
                              (uint64_t)n_tokens * sizeof(uint32_t))) {
        return false;
    }

    const uint32_t qkv_dim =
        (2u * DS4_N_GDN_QK_HEAD + DS4_N_GDN_V_HEAD) * DS4_N_GDN_STATE;
    const uint32_t flat_embd = n_tokens * DS4_N_EMBD;
    const uint32_t flat_ff = n_tokens * DS4_N_FF_DENSE;
    const bool chunkwise_gdn =
        qwen_gdn_chunkwise_enabled() && n_tokens >= 16u;
    const bool flash_attention = qwen_flash_attention_enabled();
    const bool fused_residual_norm =
        qwen_prefill_fused_residual_norm_enabled();
    bool ok = ds4_gpu_begin_commands() != 0;
    if (ok) ok = ds4_gpu_embed_tokens_q8_0_tensor(
        g->cur, g->tokens, model->map, model->size,
        weights->token_embd->abs_offset, DS4_N_VOCAB,
        n_tokens, DS4_N_EMBD) != 0;
    uint32_t full_slot = 0;
    uint32_t linear_slot = 0;
    for (uint32_t il = 0; ok && il < DS4_N_LAYER; il++) {
        const ds4_layer_weights *l = &weights->layer[il];
        if (!fused_residual_norm || il == 0u) {
            ok = ds4_gpu_rms_norm_weight_rows_tensor(
                     g->attn_norm, g->cur, model->map, model->size,
                     l->attn_norm->abs_offset, DS4_N_EMBD, n_tokens,
                     DS4_RMS_EPS) != 0;
        }
        if (!ok) break;

        if ((il + 1u) % DS4_N_FULL_ATTN_INTERVAL == 0) {
            ok = ds4_gpu_matmul_q8_0_tensor(
                     g->qg, model->map, model->size,
                     l->qwen_attn_q->abs_offset, DS4_N_EMBD,
                     2u * DS4_N_HEAD * DS4_N_HEAD_DIM,
                     g->attn_norm, n_tokens) != 0;
            if (ok) ok = ds4_gpu_matmul_q8_0_tensor(
                     g->k, model->map, model->size,
                     l->qwen_attn_k->abs_offset, DS4_N_EMBD,
                     DS4_N_HEAD_KV * DS4_N_HEAD_DIM,
                     g->attn_norm, n_tokens) != 0;
            if (ok) ok = ds4_gpu_matmul_q8_0_tensor(
                     g->v, model->map, model->size,
                     l->qwen_attn_v->abs_offset, DS4_N_EMBD,
                     DS4_N_HEAD_KV * DS4_N_VALUE_DIM,
                     g->attn_norm, n_tokens) != 0;
            if (ok) ok = ds4_gpu_qwen35_full_prepare_batch_tensor(
                     g->q, g->q_gate,
                     g->key_cache[full_slot], g->value_cache[full_slot],
                     g->qg, g->k, g->v, model->map, model->size,
                     l->qwen_attn_q_norm->abs_offset,
                     l->qwen_attn_k_norm->abs_offset,
                     pos0, n_tokens, g->ctx_cap,
                     DS4_N_HEAD, DS4_N_HEAD_KV, DS4_N_HEAD_DIM,
                     DS4_N_ROT, DS4_RMS_EPS, DS4_ROPE_FREQ_BASE) != 0;
            if (ok && flash_attention) {
                ok = ds4_gpu_qwen35_attention_flash_batch_tensor(
                         g->heads, g->q, g->q_gate,
                         g->key_cache[full_slot], g->value_cache[full_slot],
                         pos0, n_tokens, g->ctx_cap,
                         DS4_N_HEAD, DS4_N_HEAD_KV,
                         DS4_N_HEAD_DIM) != 0;
            } else if (ok) {
                ok = ds4_gpu_qwen35_attention_batch_tensor(
                         g->heads, g->scores, g->q, g->q_gate,
                         g->key_cache[full_slot], g->value_cache[full_slot],
                         pos0, n_tokens, g->ctx_cap,
                         DS4_N_HEAD, DS4_N_HEAD_KV, DS4_N_HEAD_DIM) != 0;
            }
            if (ok) ok = ds4_gpu_matmul_q8_0_tensor(
                     g->attn_out, model->map, model->size,
                     l->attn_output->abs_offset,
                     DS4_N_HEAD * DS4_N_VALUE_DIM, DS4_N_EMBD,
                     g->heads, n_tokens) != 0;
            full_slot++;
        } else {
            ok = ds4_gpu_matmul_q8_0_tensor(
                     g->qkv, model->map, model->size,
                     l->qwen_attn_qkv->abs_offset, DS4_N_EMBD,
                     qkv_dim, g->attn_norm, n_tokens) != 0;
            if (ok) ok = ds4_gpu_matmul_q8_0_tensor(
                     g->z, model->map, model->size,
                     l->qwen_attn_gate->abs_offset, DS4_N_EMBD,
                     DS4_N_GDN_INNER, g->attn_norm, n_tokens) != 0;
            if (ok) ok = ds4_gpu_matmul_q8_0_tensor(
                     g->alpha, model->map, model->size,
                     l->qwen_ssm_alpha->abs_offset, DS4_N_EMBD,
                     DS4_N_GDN_DT_RANK, g->attn_norm, n_tokens) != 0;
            if (ok) ok = ds4_gpu_matmul_q8_0_tensor(
                     g->beta, model->map, model->size,
                     l->qwen_ssm_beta->abs_offset, DS4_N_EMBD,
                     DS4_N_GDN_DT_RANK, g->attn_norm, n_tokens) != 0;
            if (ok && chunkwise_gdn) {
                ok = qwen_graph_gdn_wy_sliced(
                         g, model, l, linear_slot, n_tokens, qkv_dim);
            } else if (ok) {
                ok = ds4_gpu_qwen35_gdn_batch_tensor(
                         g->gdn_out, g->gdn_prepared, g->gdn_g, g->gdn_b,
                         g->gdn_recurrent, g->conv_state[linear_slot],
                         g->ssm_state[linear_slot], g->qkv, g->z,
                         g->alpha, g->beta, model->map, model->size,
                         l->qwen_ssm_conv1d->abs_offset,
                         l->qwen_ssm_dt->abs_offset,
                         l->qwen_ssm_a->abs_offset,
                         l->qwen_ssm_norm->abs_offset,
                         n_tokens, qkv_dim, DS4_N_GDN_QK_HEAD,
                         DS4_N_GDN_V_HEAD, DS4_N_GDN_STATE,
                         DS4_N_GDN_CONV, DS4_RMS_EPS) != 0;
            }
            if (ok) ok = ds4_gpu_matmul_q8_0_tensor(
                     g->attn_out, model->map, model->size,
                     l->qwen_ssm_out->abs_offset,
                     DS4_N_GDN_INNER, DS4_N_EMBD,
                     g->gdn_out, n_tokens) != 0;
            linear_slot++;
        }

        if (ok && fused_residual_norm) {
            ok = ds4_gpu_add_rms_norm_weight_rows_tensor(
                    g->ffn_norm, g->after_attn, g->cur, g->attn_out,
                    model->map, model->size, l->ffn_norm->abs_offset,
                    DS4_N_EMBD, n_tokens, DS4_RMS_EPS) != 0;
        } else if (ok) {
            ok = ds4_gpu_add_tensor(
                    g->after_attn, g->cur, g->attn_out, flat_embd) != 0;
            if (ok) ok = ds4_gpu_rms_norm_weight_rows_tensor(
                    g->ffn_norm, g->after_attn, model->map, model->size,
                    l->ffn_norm->abs_offset, DS4_N_EMBD, n_tokens,
                    DS4_RMS_EPS) != 0;
        }
        const bool ffn_mid_f16 =
            qwen_prefill_ffn_mid_f16_enabled() &&
            n_tokens >= 32u && (n_tokens % 32u) == 0u;
        ds4_gpu_tensor *ffn_mid = ffn_mid_f16 ? g->ffn_mid : g->ffn_gate;
        if (ok) {
            ok = ds4_gpu_matmul_q8_0_tensor(
                    g->ffn_gate, model->map, model->size,
                    l->ffn_gate->abs_offset, DS4_N_EMBD, DS4_N_FF_DENSE,
                    g->ffn_norm, n_tokens) != 0;
            if (ok) ok = ds4_gpu_matmul_q8_0_tensor(
                    g->ffn_up, model->map, model->size,
                    l->ffn_up->abs_offset, DS4_N_EMBD, DS4_N_FF_DENSE,
                    g->ffn_norm, n_tokens) != 0;
            if (ok) ok = ffn_mid_f16
                ? ds4_gpu_swiglu_f16_tensor(
                    ffn_mid, g->ffn_gate, g->ffn_up,
                    flat_ff, 0.0f, 1.0f) != 0
                : ds4_gpu_swiglu_tensor(
                    ffn_mid, g->ffn_gate, g->ffn_up,
                    flat_ff, 0.0f, 1.0f) != 0;
        }
        if (ok) ok = ffn_mid_f16
            ? ds4_gpu_matmul_q8_0_f16_rhs_tensor(
                g->ffn_out, model->map, model->size,
                l->ffn_down->abs_offset, DS4_N_FF_DENSE, DS4_N_EMBD,
                ffn_mid, n_tokens) != 0
            : ds4_gpu_matmul_q8_0_tensor(
                g->ffn_out, model->map, model->size,
                l->ffn_down->abs_offset, DS4_N_FF_DENSE, DS4_N_EMBD,
                ffn_mid, n_tokens) != 0;
        if (ok && fused_residual_norm && il + 1u < DS4_N_LAYER) {
            ok = ds4_gpu_add_rms_norm_weight_rows_tensor(
                    g->attn_norm, g->cur, g->after_attn, g->ffn_out,
                    model->map, model->size,
                    weights->layer[il + 1u].attn_norm->abs_offset,
                    DS4_N_EMBD, n_tokens, DS4_RMS_EPS) != 0;
        } else if (ok) {
            ok = ds4_gpu_add_tensor(
                    g->cur, g->after_attn, g->ffn_out, flat_embd) != 0;
        }
    }

    ds4_gpu_tensor *last = NULL;
    if (ok && logits_out) {
        const uint64_t row_bytes = (uint64_t)DS4_N_EMBD * sizeof(float);
        last = ds4_gpu_tensor_view(g->cur,
                                   (uint64_t)(n_tokens - 1u) * row_bytes,
                                   row_bytes);
        ok = last != NULL;
        if (ok) ok = ds4_gpu_rms_norm_weight_tensor(
                g->attn_norm, last, model->map, model->size,
                weights->output_norm->abs_offset,
                DS4_N_EMBD, DS4_RMS_EPS) != 0;
        if (ok) ok = ds4_gpu_matmul_q8_0_tensor(
                g->logits, model->map, model->size,
                weights->output->abs_offset, DS4_N_EMBD, DS4_N_VOCAB,
                g->attn_norm, 1) != 0;
    }
    if (ok) ok = ds4_gpu_end_commands() != 0;
    if (ok && logits_out) {
        ok = ds4_gpu_tensor_read(g->logits, 0, logits_out,
                                 (uint64_t)DS4_N_VOCAB * sizeof(float)) != 0;
    }
    ds4_gpu_tensor_free(last);
    if (!ok) (void)ds4_gpu_synchronize();
    return ok;
}

/* Prefill once, then feed each greedy token back through the persistent graph. */
static int generate_qwen_metal_argmax(
        const ds4_model   * model,
        const ds4_vocab   * vocab,
        const ds4_weights * weights,
        const token_vec   * prompt,
        int                 n_predict,
        int                 ctx_size,
        ds4_token_emit_fn   emit,
        ds4_generation_done_fn done,
        void              * emit_ud,
        ds4_session_progress_fn progress,
        void              * progress_ud) {
    fprintf(stderr, "ds4: using Qwen3.6 resident Metal generation path\n");

    if (!prompt || prompt->len <= 0 || prompt->len > ctx_size) {
        fprintf(stderr, "ds4: prompt is empty or exceeds context size\n");
        return 1;
    }
    if (n_predict <= 0) {
        if (done) done(emit_ud);
        return 0;
    }
    if (prompt->len >= ctx_size) {
        fprintf(stderr, "ds4: prompt leaves no context room for generation\n");
        return 1;
    }

    ds4_qwen_gpu_graph g = {0};
    if (!qwen_graph_alloc(&g, (uint32_t)ctx_size,
                          qwen_runtime_prefill_cap())) {
        fprintf(stderr, "ds4: failed to allocate Qwen3.6 graph runtime\n");
        return 1;
    }
    const bool memory_report = getenv("DS4_METAL_MEMORY_REPORT") != NULL;
    if (memory_report) ds4_gpu_print_memory_report("after Qwen3.6 graph alloc");

    float *logits = xmalloc((size_t)DS4_N_VOCAB * sizeof(logits[0]));
    const double t_prefill0 = now_sec();
    const bool batched_prefill = qwen_batched_prefill_enabled();
    for (int i = 0; i < prompt->len; ) {
        uint32_t chunk = 1u;
        if (batched_prefill && prompt->len - i > 1) {
            chunk = (uint32_t)(prompt->len - i);
            if (chunk > g.prefill_cap) chunk = g.prefill_cap;
        }
        const bool last = i + (int)chunk == prompt->len;
        const bool step_ok = chunk > 1u
            ? qwen_graph_forward_chunk(&g, model, weights,
                                       prompt->v + i, (uint32_t)i, chunk,
                                       last ? logits : NULL)
            : qwen_graph_forward_token(&g, model, weights,
                                       (uint32_t)prompt->v[i], (uint32_t)i,
                                       last ? logits : NULL);
        if (!step_ok) {
            fprintf(stderr, "ds4: Qwen3.6 prefill failed at position %d\n", i);
            free(logits);
            qwen_graph_free(&g);
            return 1;
        }
        i += (int)chunk;
        if (progress) progress(progress_ud, "prefill_chunk", i, prompt->len);
    }
    const double t_prefill1 = now_sec();
    if (memory_report) ds4_gpu_print_memory_report("after Qwen3.6 prefill");

    int n_generated = 0;
    uint32_t pos = (uint32_t)prompt->len;
    const bool token_timing = getenv("DS4_TOKEN_TIMING") != NULL;
    const double t_decode0 = now_sec();
    for (int i = 0; i < n_predict && pos < g.ctx_cap; i++) {
        if (getenv("DS4_TRACE_TOP") != NULL) {
            char label[64];
            snprintf(label, sizeof(label), "Qwen3.6 step %d", i);
            print_top_logits(stderr, label, vocab, logits, DS4_N_VOCAB, 10);
        }
        const int token = sample_argmax(logits, DS4_N_VOCAB);
        if (token == vocab->eos_id) break;
        if (emit) emit(emit_ud, token);
        n_generated++;

        if (i == n_predict - 1 || pos + 1u >= g.ctx_cap) break;
        const double t_eval0 = token_timing ? now_sec() : 0.0;
        if (!qwen_graph_forward_token(&g, model, weights,
                                      (uint32_t)token, pos, logits)) {
            fprintf(stderr, "ds4: Qwen3.6 decode failed at position %u\n", pos);
            free(logits);
            qwen_graph_free(&g);
            return 1;
        }
        if (token_timing) {
            const double t_eval1 = now_sec();
            fprintf(stderr, "ds4: Qwen3.6 decode eval %d took %.3f ms\n",
                    i + 1, (t_eval1 - t_eval0) * 1000.0);
        }
        pos++;
    }
    const double t_decode1 = now_sec();
    if (done) done(emit_ud);

    const double prefill_s = t_prefill1 - t_prefill0;
    const double decode_s = t_decode1 - t_decode0;
    ds4_log(stderr, DS4_LOG_TIMING,
            "ds4: Qwen3.6 prefill: %.2f t/s, generation: %.2f t/s\n",
            prefill_s > 0.0 ? (double)prompt->len / prefill_s : 0.0,
            decode_s > 0.0 ? (double)n_generated / decode_s : 0.0);

    if (memory_report) ds4_gpu_print_memory_report("before Qwen3.6 graph free");
    free(logits);
    qwen_graph_free(&g);
    return 0;
}
typedef struct {
    ds4_vocab *vocab;
} qwen_emit_ctx;

static void qwen_validate_model(const ds4_model *model) {
    ds4_str arch = {0};
    if (!model_get_string(model, "general.architecture", &arch) ||
        !ds4_streq(arch, "qwen35")) {
        ds4_die("this educational runtime only accepts qwen35 GGUFs");
    }
    config_validate_qwen35_model(model);
}

static void qwen_validate_weights(const ds4_weights *w) {
    const uint64_t full_qg = (uint64_t)DS4_N_HEAD * 2u * DS4_N_HEAD_DIM;
    const uint64_t full_kv = (uint64_t)DS4_N_HEAD_KV * DS4_N_HEAD_DIM;
    const uint64_t full_out = (uint64_t)DS4_N_HEAD * DS4_N_VALUE_DIM;
    const uint64_t gdn_qkv =
        (uint64_t)(2u * DS4_N_GDN_QK_HEAD + DS4_N_GDN_V_HEAD) *
        DS4_N_GDN_STATE;

    tensor_expect_layout(w->token_embd, DS4_TENSOR_Q8_0, 2,
                         DS4_N_EMBD, DS4_N_VOCAB, 0);
    tensor_expect_layout(w->output_norm, DS4_TENSOR_F32, 1,
                         DS4_N_EMBD, 0, 0);
    tensor_expect_layout(w->output, DS4_TENSOR_Q8_0, 2,
                         DS4_N_EMBD, DS4_N_VOCAB, 0);
    for (uint32_t il = 0; il < DS4_N_LAYER; il++) {
        const ds4_layer_weights *l = &w->layer[il];
        if (!l->attn_norm || !l->ffn_norm || !l->ffn_gate ||
            !l->ffn_up || !l->ffn_down) {
            ds4_die("required Qwen3.6 layer tensor is missing");
        }
        tensor_expect_layout(l->attn_norm, DS4_TENSOR_F32, 1,
                             DS4_N_EMBD, 0, 0);
        tensor_expect_layout(l->ffn_norm, DS4_TENSOR_F32, 1,
                             DS4_N_EMBD, 0, 0);
        tensor_expect_layout(l->ffn_gate, DS4_TENSOR_Q8_0, 2,
                             DS4_N_EMBD, DS4_N_FF_DENSE, 0);
        tensor_expect_layout(l->ffn_up, DS4_TENSOR_Q8_0, 2,
                             DS4_N_EMBD, DS4_N_FF_DENSE, 0);
        tensor_expect_layout(l->ffn_down, DS4_TENSOR_Q8_0, 2,
                             DS4_N_FF_DENSE, DS4_N_EMBD, 0);
        if ((il + 1u) % DS4_N_FULL_ATTN_INTERVAL == 0) {
            tensor_expect_layout(l->qwen_attn_q, DS4_TENSOR_Q8_0, 2,
                                 DS4_N_EMBD, full_qg, 0);
            tensor_expect_layout(l->qwen_attn_k, DS4_TENSOR_Q8_0, 2,
                                 DS4_N_EMBD, full_kv, 0);
            tensor_expect_layout(l->qwen_attn_v, DS4_TENSOR_Q8_0, 2,
                                 DS4_N_EMBD, full_kv, 0);
            tensor_expect_layout(l->qwen_attn_q_norm, DS4_TENSOR_F32, 1,
                                 DS4_N_HEAD_DIM, 0, 0);
            tensor_expect_layout(l->qwen_attn_k_norm, DS4_TENSOR_F32, 1,
                                 DS4_N_HEAD_DIM, 0, 0);
            tensor_expect_layout(l->attn_output, DS4_TENSOR_Q8_0, 2,
                                 full_out, DS4_N_EMBD, 0);
        } else {
            tensor_expect_layout(l->qwen_attn_qkv, DS4_TENSOR_Q8_0, 2,
                                 DS4_N_EMBD, gdn_qkv, 0);
            tensor_expect_layout(l->qwen_attn_gate, DS4_TENSOR_Q8_0, 2,
                                 DS4_N_EMBD, DS4_N_GDN_INNER, 0);
            tensor_expect_layout(l->qwen_ssm_a, DS4_TENSOR_F32, 1,
                                 DS4_N_GDN_DT_RANK, 0, 0);
            tensor_expect_layout(l->qwen_ssm_alpha, DS4_TENSOR_Q8_0, 2,
                                 DS4_N_EMBD, DS4_N_GDN_DT_RANK, 0);
            tensor_expect_layout(l->qwen_ssm_beta, DS4_TENSOR_Q8_0, 2,
                                 DS4_N_EMBD, DS4_N_GDN_DT_RANK, 0);
            tensor_expect_layout(l->qwen_ssm_conv1d, DS4_TENSOR_F32, 2,
                                 DS4_N_GDN_CONV, gdn_qkv, 0);
            tensor_expect_layout(l->qwen_ssm_dt, DS4_TENSOR_F32, 1,
                                 DS4_N_GDN_DT_RANK, 0, 0);
            tensor_expect_layout(l->qwen_ssm_norm, DS4_TENSOR_F32, 1,
                                 DS4_N_GDN_STATE, 0, 0);
            tensor_expect_layout(l->qwen_ssm_out, DS4_TENSOR_Q8_0, 2,
                                 DS4_N_GDN_INNER, DS4_N_EMBD, 0);
        }
    }
}

static void qwen_bind_weights(ds4_weights *w, const ds4_model *model) {
    memset(w, 0, sizeof(*w));
    w->token_embd = required_tensor(model, "token_embd.weight");
    w->output_norm = required_tensor(model, "output_norm.weight");
    w->output = required_tensor(model, "output.weight");
    for (uint32_t il = 0; il < DS4_N_LAYER; il++) {
        weights_bind_qwen35_layer(&w->layer[il], model, il);
    }
    qwen_validate_weights(w);
}

static void qwen_vocab_load(ds4_vocab *vocab, const ds4_model *model) {
    memset(vocab, 0, sizeof(*vocab));
    ds4_array_ref tokens;
    ds4_array_ref merges;
    if (!model_get_array(model, "tokenizer.ggml.tokens", &tokens) ||
        tokens.type != GGUF_VALUE_STRING || tokens.len > INT32_MAX) {
        ds4_die("GGUF tokenizer token table is missing or invalid");
    }
    if (!model_get_array(model, "tokenizer.ggml.merges", &merges) ||
        merges.type != GGUF_VALUE_STRING) {
        ds4_die("GGUF tokenizer merge table is missing or invalid");
    }
    vocab->n_vocab = (int)tokens.len;
    vocab->token = xcalloc((size_t)vocab->n_vocab, sizeof(vocab->token[0]));
    table_init(&vocab->token_to_id, tokens.len);
    ds4_cursor c = cursor_at(model, tokens.data_pos);
    for (int i = 0; i < vocab->n_vocab; i++) {
        if (!cursor_string(&c, &vocab->token[i])) ds4_die(c.error);
        table_put(&vocab->token_to_id, vocab->token[i], i);
    }
    table_init(&vocab->merge_rank, merges.len);
    c = cursor_at(model, merges.data_pos);
    for (uint64_t i = 0; i < merges.len; i++) {
        ds4_str merge;
        if (!cursor_string(&c, &merge)) ds4_die(c.error);
        table_put(&vocab->merge_rank, merge, (int)i);
    }
    if (!model_get_token_id(model, "tokenizer.ggml.eos_token_id", &vocab->eos_id))
        vocab->eos_id = vocab_lookup_optional(vocab, "<|im_end|>");
    const int im_start = vocab_lookup_optional(vocab, "<|im_start|>");
    vocab->system_id = im_start;
    vocab->user_id = im_start;
    vocab->assistant_id = im_start;
    vocab->think_start_id = vocab_lookup_optional(vocab, "<think>");
    vocab->think_end_id = vocab_lookup_optional(vocab, "</think>");
    if (vocab->eos_id < 0 || im_start < 0 ||
        vocab->think_start_id < 0 || vocab->think_end_id < 0) {
        ds4_die("Qwen3.6 tokenizer is missing required chat special tokens");
    }
}

static void qwen_tokenize(const ds4_vocab *vocab, const char *text,
                          token_vec *out) {
    qwen_bpe_tokenize_text(vocab, text, 1, out);
}

static void qwen_encode_prompt(const ds4_vocab *vocab, const char *system,
                               const char *prompt, ds4_think_mode think_mode,
                               token_vec *out) {
    if (system && system[0]) {
        token_vec_push(out, vocab->system_id);
        qwen_tokenize(vocab, "system\n", out);
        qwen_tokenize(vocab, system, out);
        token_vec_push(out, vocab->eos_id);
        qwen_tokenize(vocab, "\n", out);
    }
    token_vec_push(out, vocab->user_id);
    qwen_tokenize(vocab, "user\n", out);
    qwen_tokenize(vocab, prompt, out);
    token_vec_push(out, vocab->eos_id);
    qwen_tokenize(vocab, "\n", out);
    token_vec_push(out, vocab->assistant_id);
    qwen_tokenize(vocab, "assistant\n", out);
    token_vec_push(out, vocab->think_start_id);
    qwen_tokenize(vocab, "\n", out);
    if (!ds4_think_mode_enabled(think_mode)) {
        qwen_tokenize(vocab, "\n", out);
        token_vec_push(out, vocab->think_end_id);
        qwen_tokenize(vocab, "\n\n", out);
    }
}

static char *qwen_token_text(ds4_vocab *vocab, int token, size_t *len) {
    if (token < 0 || token >= vocab->n_vocab) {
        if (len) *len = 0;
        return xcalloc(1, 1);
    }
    ds4_str s = vocab->token[token];
    char *out = xmalloc((size_t)s.len + 1);
    if (vocab_token_is_literal_special(s)) {
        memcpy(out, s.ptr, (size_t)s.len);
        out[s.len] = '\0';
        if (len) *len = (size_t)s.len;
        return out;
    }
    size_t n = 0;
    uint64_t pos = 0;
    while (pos < s.len) {
        uint32_t cp = utf8_decode_one(s.ptr, s.len, &pos);
        int b = gpt2_codepoint_to_byte(cp);
        if (b >= 0) out[n++] = (char)b;
    }
    out[n] = '\0';
    if (len) *len = n;
    return out;
}

static void qwen_emit(void *ud, int token) {
    qwen_emit_ctx *ctx = ud;
    size_t len = 0;
    char *text = qwen_token_text(ctx->vocab, token, &len);
    fwrite(text, 1, len, stdout);
    fflush(stdout);
    free(text);
}

static void qwen_done(void *ud) {
    (void)ud;
    fputc('\n', stdout);
}

static void qwen_usage(const char *argv0) {
    fprintf(stderr,
            "usage: %s -m MODEL -p PROMPT [-n TOKENS] [-c CONTEXT] [--think|--nothink]\n",
            argv0);
}

static int qwen_acquire_instance_lock(void) {
    int fd = open("/tmp/ds4.lock", O_RDWR | O_CREAT, 0600);
    if (fd < 0) {
        fprintf(stderr, "ds4: cannot open /tmp/ds4.lock: %s\n", strerror(errno));
        return -1;
    }
    if (flock(fd, LOCK_EX | LOCK_NB) != 0) {
        fprintf(stderr, "ds4: another inference process is already running\n");
        close(fd);
        return -1;
    }
    if (ftruncate(fd, 0) == 0) dprintf(fd, "%ld\n", (long)getpid());
    return fd;
}

/* The intentionally small CLI exposes only choices that do not replace the
 * optimized execution path: context, token budget, and think/no-think. */
int main(int argc, char **argv) {
    const char *model_path = NULL;
    const char *prompt_text = NULL;
    int n_predict = 32;
    int ctx_size = 4096;
    ds4_think_mode think_mode = DS4_THINK_NONE;

    for (int i = 1; i < argc; i++) {
        if ((!strcmp(argv[i], "-m") || !strcmp(argv[i], "--model")) && i + 1 < argc) {
            model_path = argv[++i];
        } else if ((!strcmp(argv[i], "-p") || !strcmp(argv[i], "--prompt")) && i + 1 < argc) {
            prompt_text = argv[++i];
        } else if ((!strcmp(argv[i], "-n") || !strcmp(argv[i], "--tokens")) && i + 1 < argc) {
            n_predict = atoi(argv[++i]);
        } else if ((!strcmp(argv[i], "-c") || !strcmp(argv[i], "--ctx")) && i + 1 < argc) {
            ctx_size = atoi(argv[++i]);
        } else if (!strcmp(argv[i], "--think")) {
            think_mode = DS4_THINK_HIGH;
        } else if (!strcmp(argv[i], "--nothink")) {
            think_mode = DS4_THINK_NONE;
        } else if (!strcmp(argv[i], "--metal")) {
            /* Compatibility no-op: Metal is the only backend. */
        } else if (!strcmp(argv[i], "--temp") && i + 1 < argc) {
            const char *temperature = argv[++i];
            if (strcmp(temperature, "0") && strcmp(temperature, "0.0")) {
                fprintf(stderr, "ds4: this minimal runtime supports greedy decoding only (--temp 0)\n");
                return 2;
            }
        } else if (!strcmp(argv[i], "-h") || !strcmp(argv[i], "--help")) {
            qwen_usage(argv[0]);
            return 0;
        } else {
            fprintf(stderr, "unknown option: %s\n", argv[i]);
            qwen_usage(argv[0]);
            return 2;
        }
    }
    if (!model_path || !prompt_text || n_predict < 0 || ctx_size <= 0 ||
        ctx_size > 262144) {
        qwen_usage(argv[0]);
        return 2;
    }

    int lock_fd = qwen_acquire_instance_lock();
    if (lock_fd < 0) return 1;

    ds4_model model;
    ds4_weights weights;
    ds4_vocab vocab;
    token_vec prompt = {0};
    model_open(&model, model_path);
    qwen_validate_model(&model);
    qwen_bind_weights(&weights, &model);
    qwen_vocab_load(&vocab, &model);
    qwen_encode_prompt(&vocab, "You are a helpful assistant", prompt_text,
                       think_mode, &prompt);

    if (!ds4_gpu_init() ||
        !ds4_gpu_set_model_fd(model.fd) ||
        !ds4_gpu_set_model_map_range(model.map, model.size,
                                     model.tensor_data_pos,
                                     model.size - model.tensor_data_pos,
                                     model.max_tensor_bytes) ||
        !ds4_gpu_set_model_fd_for_map(model.fd, model.map)) {
        fprintf(stderr, "ds4: Metal initialization failed\n");
        close(lock_fd);
        return 1;
    }
    ds4_gpu_set_quality(0);
    ds4_gpu_set_glm_model(0);
    ds4_gpu_set_ssd_streaming(0);

    qwen_emit_ctx emit = {.vocab = &vocab};
    int rc = generate_qwen_metal_argmax(&model, &vocab, &weights, &prompt,
                                         n_predict, ctx_size, qwen_emit,
                                         qwen_done, &emit, NULL, NULL);
    ds4_gpu_cleanup();
    token_vec_free(&prompt);
    vocab_free(&vocab);
    weights_free(&weights);
    model_close(&model);
    close(lock_fd);
    return rc;
}
