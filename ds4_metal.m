#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <stdint.h>
#include <inttypes.h>
#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include <math.h>
#include <float.h>
#include <fcntl.h>
#include <limits.h>
#include <time.h>
#include <pthread.h>
#include <unistd.h>
#include <sys/mman.h>
#include <sys/sysctl.h>
#include <mach/mach.h>

#include "ds4.h"
#include "ds4_gpu.h"

/*
 * Objective-C Metal glue for the C engine.
 *
 * The C code owns model semantics and graph scheduling.  This file owns only
 * Metal objects: device/queue/library setup, mmap-backed weight views, command
 * batching, persistent tensors, scratch buffers, and thin wrappers around the
 * kernel files in the metal directory.  Keeping this boundary narrow makes the
 * inference path readable from C while still using Objective-C where Metal
 * requires it.
 */

enum {
    DS4_METAL_TENSOR_Q4_0    = 2,
    DS4_METAL_TENSOR_Q8_0    = 8,
    DS4_METAL_TENSOR_Q2_K    = 10,
    DS4_METAL_TENSOR_Q4_K    = 12,
    DS4_METAL_TENSOR_Q5_K    = 13,
    DS4_METAL_TENSOR_Q6_K    = 14,
    DS4_METAL_TENSOR_Q8_K    = 15,
    DS4_METAL_TENSOR_IQ2_XXS = 16,
    DS4_METAL_TENSOR_MXFP4   = 39,
};

@class DS4MetalQ4ExpertTable;

static id<MTLDevice> g_device;
static id<MTLCommandQueue> g_queue;
static id<MTLLibrary> g_library;
static id<MTLCommandBuffer> g_batch_cb;
static id<MTLComputeCommandEncoder> g_batch_enc;
static BOOL g_batch_has_work;
static NSMutableArray<id<MTLCommandBuffer>> *g_pending_cbs;
static id<MTLSharedEvent> g_selected_readback_event;
static uint64_t g_selected_readback_event_value;
static id<MTLComputePipelineState> g_get_rows_q8_0_pipeline;
static id<MTLComputePipelineState> g_cpy_f32_f16_pipeline;
static id<MTLComputePipelineState> g_cpy_f16_f16_pipeline;
static id<MTLComputePipelineState> g_swiglu_flat_pipeline;
static id<MTLComputePipelineState> g_add2_pipeline;
static id<MTLComputePipelineState> g_rms_norm_pipeline;
static id<MTLComputePipelineState> g_add_rms_norm_pipeline;
static id<MTLComputePipelineState> g_argsort_f32_i32_desc_pipeline;
static id<MTLComputePipelineState> g_argsort_merge_f32_i32_desc_pipeline;
static NSMutableDictionary<NSString *, id<MTLComputePipelineState>> *g_pipeline_cache;

enum {
    DS4_METAL_DECODE_PIPELINE_FAST_CACHE_SLOTS = 64,
    DS4_METAL_DECODE_PIPELINE_FAST_NAME_BYTES = 96,
};
/*
 * Sentinel nxpsg stored by the single-constant ds4_gpu_get_mul_mv_pipeline
 * path, which has no second SIMD-group constant. The extended
 * ds4_gpu_get_mul_mv_ext_pipeline path derives nxpsg as a positive divisor of
 * the 32-row output tile, so INT16_MIN can never collide with a real value.
 */
#define DS4_METAL_DECODE_PIPELINE_FAST_NXPSG_NONE INT16_MIN
_Static_assert(
    (DS4_METAL_DECODE_PIPELINE_FAST_CACHE_SLOTS &
     (DS4_METAL_DECODE_PIPELINE_FAST_CACHE_SLOTS - 1u)) == 0,
    "decode pipeline fast cache slot count must be a power of two");

typedef struct {
    id<MTLComputePipelineState> __strong pipeline;
    uint64_t hash;
    int16_t nsg;
    int16_t nxpsg;
    uint16_t name_len;
    bool used;
    char name[DS4_METAL_DECODE_PIPELINE_FAST_NAME_BYTES];
} ds4_gpu_decode_pipeline_fast_cache_entry;

static ds4_gpu_decode_pipeline_fast_cache_entry
    g_decode_pipeline_fast_cache[DS4_METAL_DECODE_PIPELINE_FAST_CACHE_SLOTS];
/* Command encoding and the existing NSMutableDictionary pipeline cache are
 * backend-serialized; this hot-path mirror intentionally shares that model. */
static bool g_decode_pipeline_fast_lookup_active;
static uint32_t g_decode_pipeline_fast_cache_entries;
static NSMutableDictionary<NSString *, id<MTLBuffer>> *g_model_buffer_cache;
static NSMutableArray<id<MTLBuffer>> *g_transient_buffers;
static id g_model_residency_set;

enum {
    DS4_GPU_PREFILL_MASK_CACHE_RAW = 1,
    DS4_GPU_PREFILL_MASK_CACHE_RATIO4 = 2,
    DS4_GPU_PREFILL_MASK_CACHE_RATIO128 = 3,
    DS4_GPU_PREFILL_MASK_CACHE_SLOTS = 3,
};

static id<MTLBuffer> g_flash_attn_pad_buffer;
static id<MTLBuffer> g_flash_attn_tmp_buffer;
static id<MTLBuffer> g_flash_attn_blk_buffer;
static id<MTLBuffer> g_flash_attn_kv_buffer;
static id<MTLBuffer> g_glm_flash_attn_mask_buffer;
static id<MTLBuffer> g_indexer_topk_buffer;
static int g_model_fd = -1;
static const void *g_model_map_ptr;
static uint64_t g_model_map_size;
static uint64_t g_model_mapped_offset;
static uint64_t g_model_mapped_size;
static uint64_t g_model_mapped_max_tensor_bytes;
static uint64_t g_tensor_alloc_live_bytes;
static uint64_t g_tensor_alloc_peak_bytes;
static pthread_mutex_t g_tensor_mu = PTHREAD_MUTEX_INITIALIZER;
static uintptr_t *g_tensor_live_slots;
static size_t g_tensor_live_cap;
static size_t g_tensor_live_count;
static size_t g_tensor_live_tombs;
static uint64_t g_model_wrap_count;
static uint64_t g_model_wrap_bytes;
static uint64_t g_model_wrap_max_bytes;
static uint64_t g_model_buffer_cache_bytes;
static uint64_t g_model_buffer_cache_evictions;
static int g_model_buffer_cache_over_limit;
static uint64_t g_model_residency_count;
static int g_model_residency_added_to_queue;
static int g_ssd_streaming_mode;
static int g_metal4_runtime_available;
static int g_metal4_family_supported;
static int g_metal4_queue_supported;
static int g_metal4_m5_neural_accelerators_hint;
static int g_metal4_tensor_api_enabled;
static int g_metal4_tensor_api_compile_supported;
static char g_metal_device_name[128];
static int ds4_gpu_model_map_log_enabled(void);

/* The async selected-load worker registers itself so cache paths that would
 * flush/wait on command buffers (a race against the encoding thread) fail
 * the load instead; the caller then retries on the main thread. */

static NSUInteger g_flash_attn_pad_bytes;
static NSUInteger g_flash_attn_tmp_bytes;
static NSUInteger g_flash_attn_blk_bytes;
static NSUInteger g_flash_attn_kv_bytes;
static NSUInteger g_glm_flash_attn_mask_bytes;
static uint32_t g_glm_flash_attn_mask_pos0;
static uint32_t g_glm_flash_attn_mask_tokens;
static uint32_t g_glm_flash_attn_mask_cache_len;
static int g_glm_flash_attn_mask_valid;
static NSUInteger g_indexer_topk_bytes;
static int g_initialized;
static int g_quality_mode;
static int g_mpp_invalid_env_reported;
#define DS4_METAL_MAX_ROUTED_EXPERT_USED 8

static double ds4_gpu_gib(uint64_t bytes);

static uint64_t ds4_gpu_system_memory_bytes(void) {
    uint64_t bytes = 0;
    size_t len = sizeof(bytes);
    if (sysctlbyname("hw.memsize", &bytes, &len, NULL, 0) != 0) return 0;
    return len == sizeof(bytes) ? bytes : 0;
}

static void ds4_gpu_print_device_summary(void) {
    const char *name = g_device.name ? [g_device.name UTF8String] : "unknown Metal device";
    uint64_t mem = ds4_gpu_system_memory_bytes();
    if (mem) {
        double gib = (double)mem / 1024.0 / 1024.0 / 1024.0;
        fprintf(stderr, "ds4: Metal device %s, %.2f GiB RAM\n", name, gib);
    } else {
        fprintf(stderr, "ds4: Metal device %s\n", name);
    }
}

#define DS4_METAL_MAX_MODEL_VIEWS 4096
/* Compatibility fallback for callers that cannot provide a parsed GGUF tensor
 * span. The normal DS4 engine passes the exact maximum tensor byte size. */
#define DS4_METAL_FALLBACK_MAX_TENSOR_BYTES (4ull * 1024ull * 1024ull * 1024ull)

typedef struct {
    __strong id<MTLBuffer> buffer;
    const void *model_map;
    uint64_t model_size;
    uint64_t model_offset;
    uint64_t bytes;
} ds4_gpu_model_view;

static ds4_gpu_model_view g_model_views[DS4_METAL_MAX_MODEL_VIEWS];
static uint32_t g_model_view_count;

enum {
    DS4_METAL_STREAM_EXPERT_CACHE_MAX_LAYER = 80,
    DS4_METAL_STREAM_EXPERT_CACHE_MAX_EXPERT = 384,
    DS4_METAL_STREAM_EXPERT_CACHE_MAX_SELECTED = DS4_METAL_MAX_ROUTED_EXPERT_USED,
    DS4_METAL_STREAM_EXPERT_CACHE_MAX_ENTRIES =
        DS4_METAL_STREAM_EXPERT_CACHE_MAX_LAYER *
        DS4_METAL_STREAM_EXPERT_CACHE_MAX_EXPERT,
    DS4_METAL_STREAM_EXPERT_CACHE_MAX_SLABS = 256,
    DS4_METAL_STREAM_EXPERT_HOTNESS_DECAY_TOKENS = 16,
    DS4_METAL_STREAM_EXPERT_VALIDATE_WORDS = 16,
};

@interface DS4MetalTensor : NSObject
@property(nonatomic, strong) id<MTLBuffer> buffer;
@property(nonatomic, assign) uint64_t offset;
@property(nonatomic, assign) uint64_t bytes;
@property(nonatomic, assign) uint8_t owner;
@end

@implementation DS4MetalTensor
@end

@interface DS4MetalQ4ExpertTable : NSObject
@property(nonatomic, strong) id<MTLBuffer> argumentBuffer;
@property(nonatomic, strong) id<MTLBuffer> addressBuffer;
@property(nonatomic, strong) NSMutableArray<id<MTLBuffer>> *expertBuffers;
@property(nonatomic, strong) id residencySet;
@property(nonatomic, assign) BOOL residencySetAddedToQueue;
@property(nonatomic, assign) uint32_t nExpert;
@property(nonatomic, assign) uint64_t expertBytes;
@end

@implementation DS4MetalQ4ExpertTable
- (void)dealloc {
#if TARGET_OS_OSX
    if (@available(macOS 15.0, *)) {
        if (_residencySet) {
            if (_residencySetAddedToQueue &&
                g_queue &&
                [g_queue respondsToSelector:@selector(removeResidencySet:)]) {
                [g_queue removeResidencySet:_residencySet];
            }
            [_residencySet endResidency];
        }
    }
#endif
}
@end

@interface DS4MetalQ4LayerResidency : NSObject
@property(nonatomic, strong) id residencySet;
@property(nonatomic, assign) BOOL addedToQueue;
@end

@implementation DS4MetalQ4LayerResidency
- (void)dealloc {
#if TARGET_OS_OSX
    if (@available(macOS 15.0, *)) {
        if (_residencySet) {
            if (_addedToQueue &&
                g_queue &&
                [g_queue respondsToSelector:@selector(removeResidencySet:)]) {
                [g_queue removeResidencySet:_residencySet];
            }
            [_residencySet endResidency];
        }
    }
#endif
}
@end

static DS4MetalTensor *ds4_gpu_tensor_obj(ds4_gpu_tensor *tensor) {
    return (__bridge DS4MetalTensor *)tensor;
}

static const DS4MetalTensor *ds4_gpu_tensor_const_obj(const ds4_gpu_tensor *tensor) {
    return (__bridge const DS4MetalTensor *)tensor;
}

/* C code owns ds4_gpu_tensor handles as retained Objective-C objects.  Freeing
 * the same opaque handle twice would make the second __bridge_transfer release
 * an already-deallocated object, which macOS reports as malloc corruption.  The
 * live table lets free validate a handle before touching Objective-C state; the
 * same mutex also serializes the diagnostic allocation counters. */
static uint64_t ds4_gpu_tensor_ptr_hash(uintptr_t ptr) {
    uint64_t x = (uint64_t)(ptr >> 4);
    x ^= x >> 33;
    x *= UINT64_C(0xff51afd7ed558ccd);
    x ^= x >> 33;
    x *= UINT64_C(0xc4ceb9fe1a85ec53);
    x ^= x >> 33;
    return x;
}

static int ds4_gpu_tensor_live_resize_locked(size_t min_cap) {
    size_t new_cap = 1024;
    while (new_cap < min_cap) new_cap <<= 1;

    uintptr_t *new_slots = calloc(new_cap, sizeof(new_slots[0]));
    if (!new_slots) return 0;

    for (size_t i = 0; i < g_tensor_live_cap; i++) {
        const uintptr_t key = g_tensor_live_slots[i];
        if (key == 0 || key == UINTPTR_MAX) continue;

        size_t idx = (size_t)ds4_gpu_tensor_ptr_hash(key) & (new_cap - 1);
        while (new_slots[idx] != 0) idx = (idx + 1) & (new_cap - 1);
        new_slots[idx] = key;
    }

    free(g_tensor_live_slots);
    g_tensor_live_slots = new_slots;
    g_tensor_live_cap = new_cap;
    g_tensor_live_tombs = 0;
    return 1;
}

static int ds4_gpu_tensor_live_insert_locked(const void *ptr) {
    if (!ptr || (uintptr_t)ptr == UINTPTR_MAX) return 0;
    if ((g_tensor_live_count + g_tensor_live_tombs + 1) * 10 >=
        g_tensor_live_cap * 7)
    {
        const size_t min_cap = g_tensor_live_cap ? g_tensor_live_cap * 2 : 1024;
        if (!ds4_gpu_tensor_live_resize_locked(min_cap)) return 0;
    }

    const uintptr_t key = (uintptr_t)ptr;
    size_t idx = (size_t)ds4_gpu_tensor_ptr_hash(key) & (g_tensor_live_cap - 1);
    size_t tomb = (size_t)-1;
    for (;;) {
        const uintptr_t cur = g_tensor_live_slots[idx];
        if (cur == key) return 0;
        if (cur == UINTPTR_MAX) {
            if (tomb == (size_t)-1) tomb = idx;
        } else if (cur == 0) {
            if (tomb != (size_t)-1) {
                idx = tomb;
                g_tensor_live_tombs--;
            }
            g_tensor_live_slots[idx] = key;
            g_tensor_live_count++;
            return 1;
        }
        idx = (idx + 1) & (g_tensor_live_cap - 1);
    }
}

static int ds4_gpu_tensor_live_remove_locked(const void *ptr) {
    if (!ptr || g_tensor_live_cap == 0) return 0;

    const uintptr_t key = (uintptr_t)ptr;
    size_t idx = (size_t)ds4_gpu_tensor_ptr_hash(key) & (g_tensor_live_cap - 1);
    for (;;) {
        const uintptr_t cur = g_tensor_live_slots[idx];
        if (cur == 0) return 0;
        if (cur == key) {
            g_tensor_live_slots[idx] = UINTPTR_MAX;
            g_tensor_live_count--;
            g_tensor_live_tombs++;
            return 1;
        }
        idx = (idx + 1) & (g_tensor_live_cap - 1);
    }
}

static int ds4_gpu_tensor_track_alloc_locked(
        const void *ptr,
        uint64_t bytes,
        uint64_t *live_snap,
        uint64_t *peak_snap)
{
    if (!ds4_gpu_tensor_live_insert_locked(ptr)) return 0;

    g_tensor_alloc_live_bytes += bytes;
    if (g_tensor_alloc_live_bytes > g_tensor_alloc_peak_bytes) {
        g_tensor_alloc_peak_bytes = g_tensor_alloc_live_bytes;
    }
    if (live_snap) *live_snap = g_tensor_alloc_live_bytes;
    if (peak_snap) *peak_snap = g_tensor_alloc_peak_bytes;
    return 1;
}

static int ds4_gpu_tensor_track_view_locked(const void *ptr) {
    return ds4_gpu_tensor_live_insert_locked(ptr);
}

static int ds4_gpu_tensor_prepare_free(
        ds4_gpu_tensor *tensor,
        uint8_t *owner,
        uint64_t *bytes,
        uint64_t *live_snap,
        uint64_t *peak_snap)
{
    pthread_mutex_lock(&g_tensor_mu);
    if (!ds4_gpu_tensor_live_remove_locked(tensor)) {
        pthread_mutex_unlock(&g_tensor_mu);
        fprintf(stderr,
                "ds4: Metal tensor free ignored for unknown handle %p\n",
                (void *)tensor);
        return 0;
    }

    DS4MetalTensor *obj = ds4_gpu_tensor_obj(tensor);
    const uint8_t obj_owner = obj.owner;
    const uint64_t obj_bytes = obj.bytes;
    if (obj_owner) {
        if (obj_bytes <= g_tensor_alloc_live_bytes) {
            g_tensor_alloc_live_bytes -= obj_bytes;
        } else {
            g_tensor_alloc_live_bytes = 0;
        }
    }
    if (owner) *owner = obj_owner;
    if (bytes) *bytes = obj_bytes;
    if (live_snap) *live_snap = g_tensor_alloc_live_bytes;
    if (peak_snap) *peak_snap = g_tensor_alloc_peak_bytes;
    pthread_mutex_unlock(&g_tensor_mu);
    return 1;
}

static void ds4_gpu_tensor_tracking_reset(void) {
    pthread_mutex_lock(&g_tensor_mu);
    if (g_tensor_live_count != 0) {
        fprintf(stderr,
                "ds4: Metal cleanup discarded %zu live tensor handles\n",
                g_tensor_live_count);
    }
    free(g_tensor_live_slots);
    g_tensor_live_slots = NULL;
    g_tensor_live_cap = 0;
    g_tensor_live_count = 0;
    g_tensor_live_tombs = 0;
    g_tensor_alloc_live_bytes = 0;
    g_tensor_alloc_peak_bytes = 0;
    pthread_mutex_unlock(&g_tensor_mu);
}

static id<MTLBuffer> ds4_gpu_tensor_buffer(const ds4_gpu_tensor *tensor) {
    if (!tensor) return nil;
    const DS4MetalTensor *obj = ds4_gpu_tensor_const_obj(tensor);
    return obj.buffer;
}

static NSUInteger ds4_gpu_tensor_offset(const ds4_gpu_tensor *tensor) {
    if (!tensor) return 0;
    const DS4MetalTensor *obj = ds4_gpu_tensor_const_obj(tensor);
    return (NSUInteger)obj.offset;
}

static id<MTLCommandBuffer> ds4_gpu_new_command_buffer(void);

static id<MTLCommandBuffer> ds4_gpu_command_buffer(int *owned) {
    if (g_batch_cb) {
        *owned = 0;
        return g_batch_cb;
    }
    *owned = 1;
    return ds4_gpu_new_command_buffer();
}

static id<MTLComputeCommandEncoder> ds4_gpu_compute_encoder(id<MTLCommandBuffer> cb) {
    if (g_batch_cb && cb == g_batch_cb) {
        g_batch_has_work = YES;
        if (!g_batch_enc) {
            g_batch_enc = [cb computeCommandEncoder];
        }
        return g_batch_enc;
    }
    return [cb computeCommandEncoder];
}

static void ds4_gpu_end_compute_encoder(id<MTLCommandBuffer> cb, id<MTLComputeCommandEncoder> enc) {
    if (!enc) return;
    if (g_batch_cb && cb == g_batch_cb && enc == g_batch_enc) return;
    [enc endEncoding];
}

static void ds4_gpu_close_batch_encoder(void) {
    if (!g_batch_enc) return;
    [g_batch_enc endEncoding];
    g_batch_enc = nil;
}

static int ds4_gpu_wait_command_buffer(id<MTLCommandBuffer> cb, const char *label) {
    [cb waitUntilCompleted];
    if (cb.status == MTLCommandBufferStatusError) {
        fprintf(stderr, "ds4: Metal %s failed: %s\n",
                label, [[cb.error localizedDescription] UTF8String]);
        return 0;
    }
    return 1;
}

static id<MTLCommandBuffer> ds4_gpu_new_command_buffer(void) {
    static int initialized;
    static int use_unretained;
    if (!initialized) {
        use_unretained = getenv("DS4_METAL_UNRETAINED_COMMAND_BUFFERS") != NULL;
        initialized = 1;
    }
    if (use_unretained) {
        return [g_queue commandBufferWithUnretainedReferences];
    }
    return [g_queue commandBuffer];
}

static uint64_t ds4_gpu_exact_view_cache_limit_bytes(void) {
    static int initialized;
    static uint64_t limit_bytes;
    if (initialized) return limit_bytes;

    const uint64_t mib = 1024ull * 1024ull;
    const uint64_t gib = 1024ull * mib;
    limit_bytes = 64ull * gib;

    const char *gib_env = getenv("DS4_METAL_EXACT_VIEW_CACHE_GIB");
    if (gib_env && gib_env[0]) {
        char *end = NULL;
        unsigned long long v = strtoull(gib_env, &end, 10);
        if (end != gib_env && *end == '\0') {
            limit_bytes = v > UINT64_MAX / gib ? UINT64_MAX : (uint64_t)v * gib;
        }
    }

    const char *mib_env = getenv("DS4_METAL_EXACT_VIEW_CACHE_MIB");
    if (mib_env && mib_env[0]) {
        char *end = NULL;
        unsigned long long v = strtoull(mib_env, &end, 10);
        if (end != mib_env && *end == '\0') {
            limit_bytes = v > UINT64_MAX / mib ? UINT64_MAX : (uint64_t)v * mib;
        }
    }

    initialized = 1;
    return limit_bytes;
}

static void ds4_gpu_model_buffer_cache_note_insert(uint64_t bytes) {
    if (g_model_buffer_cache_bytes > UINT64_MAX - bytes) {
        g_model_buffer_cache_bytes = UINT64_MAX;
    } else {
        g_model_buffer_cache_bytes += bytes;
    }

    const uint64_t limit = ds4_gpu_exact_view_cache_limit_bytes();
    if (limit != 0 && g_model_buffer_cache_bytes > limit) {
        g_model_buffer_cache_over_limit = 1;
    }
}

static void ds4_gpu_model_buffer_cache_clear(const char *reason) {
    if (!g_model_buffer_cache) {
        g_model_buffer_cache_bytes = 0;
        g_model_buffer_cache_over_limit = 0;
        return;
    }

    const NSUInteger entries = [g_model_buffer_cache count];
    if (entries != 0) {
        if (getenv("DS4_METAL_EXACT_VIEW_CACHE_PROFILE") != NULL) {
            fprintf(stderr,
                    "ds4: Metal exact model view cache evict reason=%s entries=%lu bytes=%.2f GiB limit=%.2f GiB\n",
                    reason ? reason : "unknown",
                    (unsigned long)entries,
                    ds4_gpu_gib(g_model_buffer_cache_bytes),
                    ds4_gpu_gib(ds4_gpu_exact_view_cache_limit_bytes()));
        }
        [g_model_buffer_cache removeAllObjects];
        g_model_buffer_cache_evictions++;
    }
    g_model_buffer_cache_bytes = 0;
    g_model_buffer_cache_over_limit = 0;
}

static void ds4_gpu_model_buffer_cache_maybe_evict(const char *reason) {
    if (g_model_buffer_cache_over_limit) {
        ds4_gpu_model_buffer_cache_clear(reason);
    }
}

static int ds4_gpu_finish_command_buffer(id<MTLCommandBuffer> cb, int owned, const char *label) {
    if (!owned) return 1;

    [cb commit];
    const int ok = ds4_gpu_wait_command_buffer(cb, label);
    [g_transient_buffers removeAllObjects];
    ds4_gpu_model_buffer_cache_maybe_evict(label);
    return ok;
}

static int ds4_gpu_device_name_contains(const char *needle);

static int ds4_gpu_use_m5_private_scratch(void) {
    static int initialized;
    static int enabled;
    if (!initialized) {
        enabled = ds4_gpu_device_name_contains("M5");
        initialized = 1;
    }
    return enabled;
}

static int ds4_gpu_scratch_needs_cpu_access(const char *label) {
    if (!label) return 0;
    return strstr(label, "mask") != NULL ||
           strcmp(label, "ds4_attention_output_group_ids") == 0;
}

static MTLResourceOptions ds4_gpu_model_resource_options(void) {
    MTLResourceOptions options = MTLResourceStorageModeShared;
    if (getenv("DS4_METAL_MODEL_UNTRACKED") != NULL) {
        options |= MTLResourceHazardTrackingModeUntracked;
    }
    return options;
}

static int ds4_gpu_ensure_scratch_buffer(
        id<MTLBuffer> __strong *buffer,
        NSUInteger    *capacity,
        NSUInteger     bytes,
        const char    *label) {
    if (*buffer && *capacity >= bytes) return 1;
    if (bytes == 0) bytes = 1;
    if (bytes > NSUIntegerMax) return 0;

    MTLResourceOptions options = MTLResourceStorageModeShared;
    if (ds4_gpu_use_m5_private_scratch() &&
        !ds4_gpu_scratch_needs_cpu_access(label)) {
        /*
         * M5 scratch buffers that only flow between Metal kernels do not need
         * CPU-visible shared storage. This reduces shared-memory traffic and
         * residency pressure for the long prefill scratch pools without
         * changing the public buffer lifetime model. Keep default hazard
         * tracking because the graph reuses these buffers across dependent
         * compute encoders.
         */
        options = MTLResourceStorageModePrivate;
    }

    *buffer = [g_device newBufferWithLength:bytes options:options];
    if (!*buffer && options != MTLResourceStorageModeShared) {
        *buffer = [g_device newBufferWithLength:bytes options:MTLResourceStorageModeShared];
    }
    if (!*buffer) {
        fprintf(stderr, "ds4: failed to allocate Metal scratch buffer %s (%llu bytes)\n",
                label, (unsigned long long)bytes);
        *capacity = 0;
        return 0;
    }
    (*buffer).label = [NSString stringWithUTF8String:label];
    *capacity = bytes;
    return 1;
}

static uint64_t round_up_u64(uint64_t v, uint64_t align) {
    return (v + align - 1) & ~(align - 1);
}

static uint64_t ds4_gpu_effective_model_max_tensor_bytes(uint64_t map_size, uint64_t max_tensor_bytes) {
    if (max_tensor_bytes != 0) return max_tensor_bytes;
    return map_size < DS4_METAL_FALLBACK_MAX_TENSOR_BYTES ?
           map_size : DS4_METAL_FALLBACK_MAX_TENSOR_BYTES;
}

static id<MTLComputePipelineState> ds4_gpu_get_pipeline(const char *function_name);
static int ds4_gpu_warm_model_views(void);
static double ds4_gpu_gib(uint64_t bytes);

static double ds4_gpu_now_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1000.0 + ts.tv_nsec / 1000000.0;
}

static int ds4_gpu_progress_enabled(void) {
    return ds4_log_is_tty(stderr);
}

static void ds4_gpu_progress_begin(const char *what) {
    if (!ds4_gpu_progress_enabled()) return;
    fprintf(stderr, "ds4: %s...", what);
    fflush(stderr);
}

static void ds4_gpu_progress_done(void) {
    if (!ds4_gpu_progress_enabled()) return;
    fputs(" done\n", stderr);
    fflush(stderr);
}

static void ds4_gpu_progress_failed(void) {
    if (!ds4_gpu_progress_enabled()) return;
    fputs(" failed\n", stderr);
    fflush(stderr);
}

static void ds4_gpu_model_views_clear(void) {
    for (uint32_t i = 0; i < g_model_view_count; i++) {
        g_model_views[i].buffer = nil;
        g_model_views[i].model_map = NULL;
        g_model_views[i].model_size = 0;
        g_model_views[i].model_offset = 0;
        g_model_views[i].bytes = 0;
    }
    g_model_view_count = 0;
}

static void ds4_gpu_model_residency_clear(void) {
#if TARGET_OS_OSX
    if (@available(macOS 15.0, *)) {
        if (g_model_residency_set) {
            if (g_model_residency_added_to_queue &&
                g_queue &&
                [g_queue respondsToSelector:@selector(removeResidencySet:)]) {
                [g_queue removeResidencySet:g_model_residency_set];
            }
            [g_model_residency_set endResidency];
            [g_model_residency_set removeAllAllocations];
            g_model_residency_set = nil;
        }
    }
#endif
    g_model_residency_count = 0;
    g_model_residency_added_to_queue = 0;
}

/* TP sharding keeps only this rank's expert ranges warm,
 * so whole-view residency requests (which would page in the full file)
 * must be skipped; pages fault in lazily through the same view buffers,
 * exactly like ssd-streaming mode. */
static int g_model_residency_skipped;

static int ds4_gpu_model_residency_request_views(void) {
    if (g_model_view_count == 0 ||
        g_ssd_streaming_mode ||
        g_model_residency_skipped ||
        getenv("DS4_METAL_NO_RESIDENCY") != NULL) {
        return 1;
    }

#if TARGET_OS_OSX
    if (@available(macOS 15.0, *)) {
        /*
         * Register all model views as one residency set before inference. This
         * is a GPU residency/budgeting hint, not a request to fault the whole
         * 80+ GB file into memory. Its purpose is to make the driver see the
         * complete set of large shared allocations during setup instead of
         * discovering them lazily from the first measured graph command, where
         * VM validation and residency accounting would look like model compute.
         */
        MTLResidencySetDescriptor *desc = [[MTLResidencySetDescriptor alloc] init];
        desc.label = @"ds4_model";
        desc.initialCapacity = g_model_view_count;

        NSError *error = nil;
        g_model_residency_set = [g_device newResidencySetWithDescriptor:desc error:&error];
        if (!g_model_residency_set) {
            fprintf(stderr, "ds4: Metal model residency set creation failed: %s\n",
                    [[error localizedDescription] UTF8String]);
            return 0;
        }

        for (uint32_t i = 0; i < g_model_view_count; i++) {
            [g_model_residency_set addAllocation:g_model_views[i].buffer];
        }
        [g_model_residency_set commit];
        [g_model_residency_set requestResidency];
        if (getenv("DS4_METAL_DISABLE_QUEUE_RESIDENCY_SET") == NULL &&
            g_queue &&
            [g_queue respondsToSelector:@selector(addResidencySet:)]) {
            [g_queue addResidencySet:g_model_residency_set];
            g_model_residency_added_to_queue = 1;
        }
        g_model_residency_count = g_model_view_count;
    }
#endif

    return 1;
}

static int ds4_gpu_add_model_view_range(
        const void *model_map,
        uint64_t    model_size,
        uint64_t    map_offset,
        uint64_t    map_size,
        uint64_t    max_tensor_bytes,
        bool        use_default_view_cap,
        uint64_t   *mapped_model_size_out) {
    const uint64_t page = (uint64_t)getpagesize();
    const uintptr_t model_addr = (uintptr_t)model_map;

    if ((model_addr & (uintptr_t)(page - 1)) != 0) {
        fprintf(stderr, "ds4: Metal model mmap base is not page aligned\n");
        return 0;
    }
    if (map_offset > model_size || map_size > model_size - map_offset) {
        fprintf(stderr, "ds4: Metal model mapped range is outside the GGUF mapping\n");
        return 0;
    }
    const uint64_t page_model_offset = map_offset & ~(page - 1);
    const uint64_t leading = map_offset - page_model_offset;
    if (map_size > UINT64_MAX - leading ||
        leading + map_size > UINT64_MAX - (page - 1))
    {
        fprintf(stderr, "ds4: Metal model mapped range overflows page alignment\n");
        return 0;
    }
    const uint64_t mapped_model_size = round_up_u64(leading + map_size, page);
    uint64_t max_buffer = (uint64_t)[g_device maxBufferLength];
    max_buffer &= ~(page - 1);

    /*
     * Wrap only the tensor-data part of the GGUF file. Metadata is parsed by the
     * CPU and is never dereferenced by kernels, so exposing it to Metal only
     * grows the residency set and the VM range the driver must validate.
     *
     * Metal buffers have a device-specific maximum length, and this model is
     * larger than that maximum on the target machines. Creating one no-copy
     * buffer per tensor would avoid the length limit, but it would also move a
     * lot of VM-object creation and residency bookkeeping into graph setup. The
     * stable shape here is a tiny number of page-aligned views created once.
     *
     * Adjacent views intentionally overlap by more than the largest tensor, plus
     * one page for alignment. That invariant guarantees every tensor lies wholly
     * inside at least one view, so hot paths pass one buffer and one inner byte
     * offset. We never split a weight tensor across command encoders.
     */
    if (max_tensor_bytes > map_size) {
        fprintf(stderr, "ds4: Metal model max tensor span is larger than a mapped tensor span\n");
        return 0;
    }
    if (max_tensor_bytes > UINT64_MAX - (page - 1)) {
        fprintf(stderr, "ds4: Metal model max tensor span overflows page alignment\n");
        return 0;
    }
    const uint64_t max_tensor_rounded = round_up_u64(max_tensor_bytes, page);
    if (max_tensor_rounded > UINT64_MAX - page) {
        fprintf(stderr, "ds4: Metal model view overlap overflows page slack\n");
        return 0;
    }
    const uint64_t overlap = max_tensor_rounded + page;
    if (max_buffer == 0 || max_buffer <= overlap) {
        fprintf(stderr,
                "ds4: Metal maxBufferLength is too small for DS4 model views "
                "(max tensor %.2f GiB, max buffer %.2f GiB)\n",
                ds4_gpu_gib(max_tensor_bytes),
                ds4_gpu_gib(max_buffer));
        return 0;
    }

    uint64_t view_limit = max_buffer;
    const char *view_limit_env = getenv("DS4_METAL_MODEL_VIEW_MAX_GIB");
    if (view_limit_env && view_limit_env[0]) {
        char *end = NULL;
        unsigned long long gib = strtoull(view_limit_env, &end, 10);
        if (end != view_limit_env && gib > 0) {
            uint64_t env_limit = gib * 1024ull * 1024ull * 1024ull;
            env_limit &= ~(page - 1);
            if (env_limit > 0) view_limit = env_limit;
        }
    } else if (use_default_view_cap && mapped_model_size > max_buffer) {
        /*
         * Very large no-copy buffers can make Metal's VM validation dominate
         * startup or the first graph command on multi-hundred-GiB slices. Keep
         * ordinary contiguous model mappings unchanged, but let distributed
         * span maps use smaller overlapping views when a range already has to
         * be split.
         */
        const uint64_t default_limit = 128ull * 1024ull * 1024ull * 1024ull;
        if (view_limit > default_limit) view_limit = default_limit;
    }
    if (view_limit > max_buffer) view_limit = max_buffer;
    view_limit &= ~(page - 1);
    if (view_limit == 0 || view_limit <= overlap) {
        fprintf(stderr,
                "ds4: Metal model view cap is too small for DS4 model views "
                "(cap %.2f GiB, max tensor %.2f GiB)\n",
                ds4_gpu_gib(view_limit),
                ds4_gpu_gib(max_tensor_bytes));
        return 0;
    }

    const uint64_t step = view_limit - overlap;
    uint64_t off = 0;
    while (off < mapped_model_size) {
        if (g_model_view_count == DS4_METAL_MAX_MODEL_VIEWS) {
            fprintf(stderr, "ds4: Metal model needs more mapped views than expected\n");
            return 0;
        }

        uint64_t view_bytes = mapped_model_size - off;
        if (view_bytes > view_limit) view_bytes = view_limit;

        id<MTLBuffer> buffer = [g_device newBufferWithBytesNoCopy:(void *)(model_addr + page_model_offset + off)
                                                           length:(NSUInteger)view_bytes
                                                          options:ds4_gpu_model_resource_options()
                                                      deallocator:nil];
        if (!buffer) {
            fprintf(stderr,
                    "ds4: Metal could not wrap mmaped model view at %.2f GiB, size %.2f GiB\n",
                    (double)(page_model_offset + off) / (1024.0 * 1024.0 * 1024.0),
                    (double)view_bytes / (1024.0 * 1024.0 * 1024.0));
            return 0;
        }
        buffer.label = [NSString stringWithFormat:@"ds4_model_view_%u", g_model_view_count];

        g_model_views[g_model_view_count].buffer = buffer;
        g_model_views[g_model_view_count].model_map = model_map;
        g_model_views[g_model_view_count].model_size = model_size;
        g_model_views[g_model_view_count].model_offset = page_model_offset + off;
        g_model_views[g_model_view_count].bytes = view_bytes;
        g_model_view_count++;

        g_model_wrap_count++;
        g_model_wrap_bytes += view_bytes;
        if (view_bytes > g_model_wrap_max_bytes) g_model_wrap_max_bytes = view_bytes;

        if (off + view_bytes >= mapped_model_size) break;
        off += step;
    }

    if (mapped_model_size_out) *mapped_model_size_out += mapped_model_size;
    return 1;
}

static int ds4_gpu_finish_model_views(
        double t0,
        uint64_t mapped_model_size,
        uint64_t display_offset) {
    const double t_mapped = ds4_gpu_now_ms();
    const int request_residency =
        !g_ssd_streaming_mode &&
        getenv("DS4_METAL_NO_RESIDENCY") == NULL;
    if (request_residency) ds4_gpu_progress_begin("requesting Metal residency (may take tens of seconds)");
    if (!ds4_gpu_model_residency_request_views()) {
        if (request_residency) ds4_gpu_progress_failed();
        return 0;
    }
    if (request_residency) ds4_gpu_progress_done();
    const double t_resident = ds4_gpu_now_ms();
    int warmed = 1;
    const double t_warm0 = ds4_gpu_now_ms();
    const int warm_model_views = !g_ssd_streaming_mode &&
                                 getenv("DS4_METAL_NO_RESIDENCY") == NULL &&
                                 getenv("DS4_METAL_NO_MODEL_WARMUP") == NULL;
    if (warm_model_views) {
        /*
         * The first GPU command touching no-copy mmap storage can pay command
         * queue setup, page-table validation, and shared-allocation residency
         * costs. Sample each model view here so timed graph execution starts
         * after that one-time work. The stride is intentionally coarse: this is
         * a validation touch over the VM ranges, not a full model prefetch. A
         * dense prefetch would create exactly the kind of memory pressure and
         * startup stalls this path is designed to avoid.
         */
        if (g_model_residency_skipped) {
            /* TP sharding: a single command buffer binding every
             * view demands residency of them all and OOMs; the engine's
             * CPU-side sharded warm pre-faults the owned bytes instead. */
            warmed = 1;
        } else {
            ds4_gpu_progress_begin("warming Metal model views");
            warmed = ds4_gpu_warm_model_views();
            if (warmed) ds4_gpu_progress_done();
            else ds4_gpu_progress_failed();
        }
    }
    const double t_warm = ds4_gpu_now_ms();
    if (ds4_gpu_model_map_log_enabled()) {
        fprintf(stderr,
                "ds4: Metal model views created in %.3f ms, residency requested in %.3f ms, warmup %.3f ms (mapped %.2f MiB from offset %.2f MiB)\n",
                t_mapped - t0,
                t_resident - t_mapped,
                t_warm - t_warm0,
                mapped_model_size / 1024.0 / 1024.0,
                display_offset / 1024.0 / 1024.0);
    }
    if (!warmed) return 0;
    return 1;
}

static int ds4_gpu_map_model_views(
        const void *model_map,
        uint64_t    model_size,
        uint64_t    map_offset,
        uint64_t    map_size,
        uint64_t    max_tensor_bytes) {
    const double t0 = ds4_gpu_now_ms();
    uint64_t mapped_model_size = 0;
    if (!ds4_gpu_add_model_view_range(model_map,
                                      model_size,
                                      map_offset,
                                      map_size,
                                      max_tensor_bytes,
                                      false,
                                      &mapped_model_size)) {
        return 0;
    }
    return ds4_gpu_finish_model_views(t0, mapped_model_size, map_offset);
}

static id<MTLComputePipelineState> ds4_gpu_get_mul_mm_pipeline(
        const char *function_name,
        bool        bc_inp,
        bool        bc_out) {
    NSString *key = [NSString stringWithFormat:@"%s_bci=%d_bco=%d",
                     function_name, bc_inp ? 1 : 0, bc_out ? 1 : 0];
    id<MTLComputePipelineState> cached = [g_pipeline_cache objectForKey:key];
    if (cached) return cached;

    MTLFunctionConstantValues *constants = [[MTLFunctionConstantValues alloc] init];
    [constants setConstantValue:&bc_inp type:MTLDataTypeBool atIndex:700];
    [constants setConstantValue:&bc_out type:MTLDataTypeBool atIndex:701];

    NSError *error = nil;
    NSString *name = [NSString stringWithUTF8String:function_name];
    id<MTLFunction> fn = [g_library newFunctionWithName:name
                                         constantValues:constants
                                                  error:&error];
    if (!fn) {
        fprintf(stderr, "ds4: Metal %s function not found: %s\n",
                function_name, [[error localizedDescription] UTF8String]);
        return nil;
    }

    error = nil;
    id<MTLComputePipelineState> pipeline = [g_device newComputePipelineStateWithFunction:fn error:&error];
    if (!pipeline) {
        fprintf(stderr, "ds4: Metal %s pipeline failed: %s\n",
                function_name, [[error localizedDescription] UTF8String]);
        return nil;
    }

    [g_pipeline_cache setObject:pipeline forKey:key];
    return pipeline;
}

static id<MTLComputePipelineState> ds4_gpu_get_pipeline(
        const char *function_name) {
    NSString *key = [NSString stringWithFormat:@"%s", function_name];
    id<MTLComputePipelineState> cached = [g_pipeline_cache objectForKey:key];
    if (cached) return cached;

    NSError *error = nil;
    NSString *name = [NSString stringWithUTF8String:function_name];
    id<MTLFunction> fn = [g_library newFunctionWithName:name];
    if (!fn) {
        fprintf(stderr, "ds4: Metal %s function not found\n", function_name);
        return nil;
    }

    id<MTLComputePipelineState> pipeline = [g_device newComputePipelineStateWithFunction:fn error:&error];
    if (!pipeline) {
        fprintf(stderr, "ds4: Metal %s pipeline failed: %s\n",
                function_name, [[error localizedDescription] UTF8String]);
        return nil;
    }

    [g_pipeline_cache setObject:pipeline forKey:key];
    return pipeline;
}

static int ds4_gpu_device_name_contains(const char *needle);

static int ds4_gpu_env_value_eq(const char *v, size_t n, const char *literal) {
    size_t m = strlen(literal);
    if (n != m) return 0;
    for (size_t i = 0; i < n; i++) {
        if (tolower((unsigned char)v[i]) != tolower((unsigned char)literal[i])) return 0;
    }
    return 1;
}

static int ds4_gpu_env_bool(const char *name) {
    const char *v = getenv(name);
    if (!v) return -1;

    while (isspace((unsigned char)*v)) v++;
    size_t n = strlen(v);
    while (n > 0 && isspace((unsigned char)v[n - 1])) n--;
    if (n == 0) return 1;

    if (ds4_gpu_env_value_eq(v, n, "1") ||
        ds4_gpu_env_value_eq(v, n, "true") ||
        ds4_gpu_env_value_eq(v, n, "yes") ||
        ds4_gpu_env_value_eq(v, n, "on")) {
        return 1;
    }
    if (ds4_gpu_env_value_eq(v, n, "0") ||
        ds4_gpu_env_value_eq(v, n, "false") ||
        ds4_gpu_env_value_eq(v, n, "no") ||
        ds4_gpu_env_value_eq(v, n, "off")) {
        return 0;
    }

    if (!g_mpp_invalid_env_reported) {
        fprintf(stderr,
                "ds4: invalid Metal boolean environment value %s=%.*s; treating presence as enabled\n",
                name, (int)n, v);
        g_mpp_invalid_env_reported = 1;
    }
    return 1;
}

static uint64_t ds4_gpu_env_u64(const char *name,
                                uint64_t    fallback,
                                uint64_t    min_value,
                                uint64_t    max_value) {
    const char *v = getenv(name);
    if (!v) return fallback;
    while (isspace((unsigned char)*v)) v++;
    if (!*v) return fallback;

    errno = 0;
    char *end = NULL;
    unsigned long long parsed = strtoull(v, &end, 10);
    if (end == v || errno == ERANGE) return fallback;
    while (isspace((unsigned char)*end)) end++;
    if (*end) return fallback;

    if (parsed < min_value) return fallback;
    uint64_t value = (uint64_t)parsed;
    if (value > max_value) value = max_value;
    return value;
}

static int ds4_gpu_mpp_available(void) {
    return g_metal4_tensor_api_enabled && !g_quality_mode;
}

/*
 * Retained Metal4 defaults live here instead of behind user-visible options.
 * The public runtime has one automatic accelerated path plus the global
 * DS4_METAL_DISABLE_METAL4 comparison switch.  Benchmark-only alternatives that
 * lost during M5 work are removed or kept out of the dispatch path so future
 * changes do not accidentally turn old experiments into new modes.
 */

enum {
    DS4_METAL_ATTN_OUT_MPP_TILE_N = 64,
};

static void ds4_gpu_warn_mpp_fallback(void) {
    static int warned;
    if (!warned) {
        fprintf(stderr, "ds4: accelerated Metal prefill matmul unavailable; falling back to legacy kernel\n");
        warned = 1;
    }
}

static int ds4_gpu_device_name_contains(const char *needle) {
    return g_metal_device_name[0] != '\0' && strstr(g_metal_device_name, needle) != NULL;
}

int ds4_gpu_device_is_pre_m5_apple_silicon(void) {
    return strncmp(g_metal_device_name, "Apple M", 7) == 0 &&
           g_metal_device_name[7] >= '1' &&
           g_metal_device_name[7] <= '4' &&
           (g_metal_device_name[8] == '\0' ||
            g_metal_device_name[8] == ' ');
}

static void ds4_gpu_detect_metal4_features(void) {
    g_metal4_runtime_available = 0;
    g_metal4_family_supported = 0;
    g_metal4_queue_supported = 0;
    g_metal4_m5_neural_accelerators_hint = 0;
    g_metal4_tensor_api_enabled = 0;
    g_metal4_tensor_api_compile_supported = 0;
    g_metal_device_name[0] = '\0';

    if (!g_device) return;

    const char *name = [[g_device name] UTF8String];
    if (name) {
        snprintf(g_metal_device_name, sizeof(g_metal_device_name), "%s", name);
    }

#if defined(__MAC_OS_X_VERSION_MAX_ALLOWED) && __MAC_OS_X_VERSION_MAX_ALLOWED >= 260000
    const int metal4_disabled =
        ds4_gpu_env_bool("DS4_METAL_DISABLE_METAL4") > 0;
    if (@available(macOS 26.0, *)) {
        g_metal4_runtime_available = 1;
        g_metal4_family_supported =
            !metal4_disabled && [g_device supportsFamily:MTLGPUFamilyMetal4] ? 1 : 0;
        g_metal4_queue_supported = [g_device respondsToSelector:@selector(newMTL4CommandQueue)] ? 1 : 0;

        /*
         * Apple does not currently expose a separate "Neural Accelerator" bit
         * through Metal. On public M5 systems the hardware signal is the device
         * generation plus Metal 4 support, so keep this as a conservative hint.
         */
        if (g_metal4_family_supported && ds4_gpu_device_name_contains("M5")) {
            g_metal4_m5_neural_accelerators_hint = 1;
        }

        if (g_metal4_family_supported) {
            const int default_enable =
                ds4_gpu_device_name_contains("M5") ||
                ds4_gpu_device_name_contains("M6") ||
                ds4_gpu_device_name_contains("A19") ||
                ds4_gpu_device_name_contains("A20");

            /*
             * Metal 4 TensorOps are portable in source, but on pre-M5 hardware
             * they can map to ordinary shader fallbacks.  Keep the automatic
             * fast path restricted to hardware generations where the Neural
             * Accelerator/TensorOps path is expected to pay off; older Metal
             * machines continue to use the established kernels unless a future
             * device is explicitly added here.
             */
            if (default_enable) {
                g_metal4_tensor_api_compile_supported = ds4_gpu_compile_tensor_probe();
                g_metal4_tensor_api_enabled = g_metal4_tensor_api_compile_supported;
                if (!g_metal4_tensor_api_enabled) {
                    fprintf(stderr, "ds4: Metal 4 tensor API probe failed; using legacy Metal kernels\n");
                }
            } else {
                fprintf(stderr, "ds4: Metal 4 tensor API disabled for pre-M5/pre-A19 devices\n");
            }
        }
    }
#endif
}

static int ds4_gpu_warm_model_views(void) {
    if (g_model_view_count == 0) return 1;

    id<MTLComputePipelineState> pipeline = ds4_gpu_get_pipeline("kernel_touch_u8_stride");
    if (!pipeline) return 0;

    uint64_t stride = 1024ull * 1024ull;
    const char *stride_env = getenv("DS4_METAL_MODEL_WARMUP_STRIDE_MB");
    if (stride_env && stride_env[0]) {
        char *end = NULL;
        unsigned long long mb = strtoull(stride_env, &end, 10);
        if (end != stride_env && mb > 0 && mb <= 1024) {
            stride = mb * 1024ull * 1024ull;
        }
    }
    const char *stride_kb_env = getenv("DS4_METAL_MODEL_WARMUP_STRIDE_KB");
    if (stride_kb_env && stride_kb_env[0]) {
        char *end = NULL;
        unsigned long long kb = strtoull(stride_kb_env, &end, 10);
        if (end != stride_kb_env && kb > 0 && kb <= 1024ull * 1024ull) {
            stride = kb * 1024ull;
            const uint64_t page = (uint64_t)getpagesize();
            if (stride < page) stride = page;
        }
    }

    uint64_t total_touches = 0;
    for (uint32_t i = 0; i < g_model_view_count; i++) {
        total_touches += (g_model_views[i].bytes + stride - 1) / stride;
    }
    if (total_touches == 0 || total_touches > (uint64_t)NSUIntegerMax) return 0;

    const NSUInteger out_bytes = (NSUInteger)total_touches;
    id<MTLBuffer> out = [g_device newBufferWithLength:out_bytes
                                             options:MTLResourceStorageModeShared];
    if (!out) {
        fprintf(stderr, "ds4: Metal model warmup scratch allocation failed\n");
        return 0;
    }
    out.label = @"ds4_model_warmup";

    id<MTLCommandBuffer> cb = ds4_gpu_new_command_buffer();
    if (!cb) {
        fprintf(stderr, "ds4: Metal model warmup command buffer allocation failed\n");
        return 0;
    }

    id<MTLComputeCommandEncoder> enc = ds4_gpu_compute_encoder(cb);
    [enc setComputePipelineState:pipeline];
    uint64_t dst_offset = 0;
    for (uint32_t i = 0; i < g_model_view_count; i++) {
        const uint64_t bytes = g_model_views[i].bytes;
        const uint64_t n = (bytes + stride - 1) / stride;
        [enc setBuffer:g_model_views[i].buffer offset:0 atIndex:0];
        [enc setBuffer:out offset:0 atIndex:1];
        [enc setBytes:&stride length:sizeof(stride) atIndex:2];
        [enc setBytes:&bytes length:sizeof(bytes) atIndex:3];
        [enc setBytes:&dst_offset length:sizeof(dst_offset) atIndex:4];
        [enc dispatchThreadgroups:MTLSizeMake((NSUInteger)((n + 255) / 256), 1, 1)
             threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        dst_offset += n;
    }
    ds4_gpu_end_compute_encoder(cb, enc);

    [cb commit];
    [cb waitUntilCompleted];

    if (cb.status == MTLCommandBufferStatusError) {
        fprintf(stderr, "ds4: Metal model warmup failed: %s\n",
                [[cb.error localizedDescription] UTF8String]);
        return 0;
    }

    return 1;
}

static void ds4_gpu_decode_pipeline_fast_cache_reset(void) {
    for (uint32_t i = 0; i < DS4_METAL_DECODE_PIPELINE_FAST_CACHE_SLOTS; i++) {
        ds4_gpu_decode_pipeline_fast_cache_entry *entry =
            &g_decode_pipeline_fast_cache[i];
        /* ARC must see every strong release; do not memset this table. */
        entry->pipeline = nil;
        entry->hash = 0;
        entry->nsg = 0;
        entry->name_len = 0;
        entry->used = false;
        entry->name[0] = '\0';
    }
    g_decode_pipeline_fast_lookup_active = false;
    g_decode_pipeline_fast_cache_entries = 0;
}

static bool ds4_gpu_decode_pipeline_fast_key(
        const char *function_name,
        int16_t     nsg,
        int16_t     nxpsg,
        uint16_t   *name_len_out,
        uint64_t   *hash_out) {
    if (!function_name || !name_len_out || !hash_out) return false;

    uint64_t hash = UINT64_C(14695981039346656037);
    size_t name_len = 0;
    while (function_name[name_len] != '\0') {
        if (name_len + 1u >= DS4_METAL_DECODE_PIPELINE_FAST_NAME_BYTES) {
            return false;
        }
        hash ^= (uint8_t)function_name[name_len++];
        hash *= UINT64_C(1099511628211);
    }
    const uint16_t nsg_bits = (uint16_t)nsg;
    hash ^= (uint8_t)nsg_bits;
    hash *= UINT64_C(1099511628211);
    hash ^= (uint8_t)(nsg_bits >> 8u);
    hash *= UINT64_C(1099511628211);
    const uint16_t nxpsg_bits = (uint16_t)nxpsg;
    hash ^= (uint8_t)nxpsg_bits;
    hash *= UINT64_C(1099511628211);
    hash ^= (uint8_t)(nxpsg_bits >> 8u);
    hash *= UINT64_C(1099511628211);

    *name_len_out = (uint16_t)name_len;
    *hash_out = hash;
    return true;
}

static id<MTLComputePipelineState> ds4_gpu_decode_pipeline_fast_cache_lookup(
        const char *function_name,
        int16_t     nsg,
        int16_t     nxpsg,
        uint16_t    name_len,
        uint64_t    hash) {
    const uint32_t mask = DS4_METAL_DECODE_PIPELINE_FAST_CACHE_SLOTS - 1u;
    uint32_t slot = (uint32_t)hash & mask;
    for (uint32_t probe = 0;
         probe < DS4_METAL_DECODE_PIPELINE_FAST_CACHE_SLOTS;
         probe++, slot = (slot + 1u) & mask) {
        ds4_gpu_decode_pipeline_fast_cache_entry *entry =
            &g_decode_pipeline_fast_cache[slot];
        if (!entry->used) break;
        if (entry->hash == hash &&
            entry->nsg == nsg &&
            entry->nxpsg == nxpsg &&
            entry->name_len == name_len &&
            memcmp(entry->name, function_name, name_len) == 0) {
            return entry->pipeline;
        }
    }
    return nil;
}

static void ds4_gpu_decode_pipeline_fast_cache_insert(
        const char                  *function_name,
        int16_t                      nsg,
        int16_t                      nxpsg,
        uint16_t                     name_len,
        uint64_t                     hash,
        id<MTLComputePipelineState>  pipeline) {
    if (!pipeline ||
        g_decode_pipeline_fast_cache_entries >=
            DS4_METAL_DECODE_PIPELINE_FAST_CACHE_SLOTS) {
        return;
    }

    const uint32_t mask = DS4_METAL_DECODE_PIPELINE_FAST_CACHE_SLOTS - 1u;
    uint32_t slot = (uint32_t)hash & mask;
    for (uint32_t probe = 0;
         probe < DS4_METAL_DECODE_PIPELINE_FAST_CACHE_SLOTS;
         probe++, slot = (slot + 1u) & mask) {
        ds4_gpu_decode_pipeline_fast_cache_entry *entry =
            &g_decode_pipeline_fast_cache[slot];
        if (entry->used) {
            if (entry->hash == hash &&
                entry->nsg == nsg &&
                entry->nxpsg == nxpsg &&
                entry->name_len == name_len &&
                memcmp(entry->name, function_name, name_len) == 0) {
                entry->pipeline = pipeline;
                return;
            }
            continue;
        }
        memcpy(entry->name, function_name, name_len);
        entry->name[name_len] = '\0';
        entry->hash = hash;
        entry->nsg = nsg;
        entry->nxpsg = nxpsg;
        entry->name_len = name_len;
        entry->pipeline = pipeline;
        entry->used = true;
        g_decode_pipeline_fast_cache_entries++;
        return;
    }
}

static id<MTLComputePipelineState> ds4_gpu_get_mul_mv_pipeline(
        const char *function_name,
        int16_t     nsg) {
    uint16_t fast_name_len = 0;
    uint64_t fast_hash = 0;
    const bool fast_key_valid =
        g_decode_pipeline_fast_lookup_active &&
        ds4_gpu_decode_pipeline_fast_key(
            function_name, nsg,
            DS4_METAL_DECODE_PIPELINE_FAST_NXPSG_NONE,
            &fast_name_len, &fast_hash);
    if (fast_key_valid) {
        id<MTLComputePipelineState> fast_cached =
            ds4_gpu_decode_pipeline_fast_cache_lookup(
                function_name, nsg,
                DS4_METAL_DECODE_PIPELINE_FAST_NXPSG_NONE,
                fast_name_len, fast_hash);
        if (fast_cached) return fast_cached;
    }

    NSString *key = [NSString stringWithFormat:@"%s_nsg=%d", function_name, (int)nsg];
    id<MTLComputePipelineState> cached = [g_pipeline_cache objectForKey:key];
    if (cached) {
        if (fast_key_valid) {
            ds4_gpu_decode_pipeline_fast_cache_insert(
                function_name, nsg,
                DS4_METAL_DECODE_PIPELINE_FAST_NXPSG_NONE,
                fast_name_len, fast_hash, cached);
        }
        return cached;
    }

    MTLFunctionConstantValues *constants = [[MTLFunctionConstantValues alloc] init];
    [constants setConstantValue:&nsg type:MTLDataTypeShort atIndex:600];

    NSError *error = nil;
    NSString *name = [NSString stringWithUTF8String:function_name];
    id<MTLFunction> fn = [g_library newFunctionWithName:name
                                         constantValues:constants
                                                  error:&error];
    if (!fn) {
        fprintf(stderr, "ds4: Metal %s function not found: %s\n",
                function_name, [[error localizedDescription] UTF8String]);
        return nil;
    }

    error = nil;
    id<MTLComputePipelineState> pipeline = [g_device newComputePipelineStateWithFunction:fn error:&error];
    if (!pipeline) {
        fprintf(stderr, "ds4: Metal %s pipeline failed: %s\n",
                function_name, [[error localizedDescription] UTF8String]);
        return nil;
    }

    [g_pipeline_cache setObject:pipeline forKey:key];
    if (fast_key_valid) {
        ds4_gpu_decode_pipeline_fast_cache_insert(
            function_name, nsg,
            DS4_METAL_DECODE_PIPELINE_FAST_NXPSG_NONE,
            fast_name_len, fast_hash, pipeline);
    }
    return pipeline;
}

/* The ordinary mul-mv cache key covers function name and nsg only. Keep the
 * descriptor-hinted PSO separate so a cache hit cannot erase this compiler
 * contract or substitute it for the fallback pipeline. */

/*
 * Exercises the second SIMD-group constant key shape used by
 * ds4_gpu_get_mul_mv_ext_pipeline (name, nsg, nxpsg). Confirms the extended key
 * populates and hits, distinguishes a real nxpsg from the single-constant
 * sentinel, and coexists with the single-constant mv entries in the shared
 * 64-slot table. Uses baseline as a stand-in pipeline object so the guard does
 * not depend on compiling a specific extended-kernel function.
 */

static id<MTLComputePipelineState> ds4_gpu_get_mul_mv_ext_pipeline(
        const char *function_name,
        int16_t     nsg,
        int16_t     nxpsg) {
    uint16_t fast_name_len = 0;
    uint64_t fast_hash = 0;
    const bool fast_key_valid =
        g_decode_pipeline_fast_lookup_active &&
        ds4_gpu_decode_pipeline_fast_key(
            function_name, nsg, nxpsg, &fast_name_len, &fast_hash);
    if (fast_key_valid) {
        id<MTLComputePipelineState> fast_cached =
            ds4_gpu_decode_pipeline_fast_cache_lookup(
                function_name, nsg, nxpsg, fast_name_len, fast_hash);
        if (fast_cached) return fast_cached;
    }

    NSString *key = [NSString stringWithFormat:@"%s_nsg=%d_nxpsg=%d",
                     function_name, (int)nsg, (int)nxpsg];
    id<MTLComputePipelineState> cached = [g_pipeline_cache objectForKey:key];
    if (cached) {
        if (fast_key_valid) {
            ds4_gpu_decode_pipeline_fast_cache_insert(
                function_name, nsg, nxpsg, fast_name_len, fast_hash, cached);
        }
        return cached;
    }

    MTLFunctionConstantValues *constants = [[MTLFunctionConstantValues alloc] init];
    [constants setConstantValue:&nsg   type:MTLDataTypeShort atIndex:600];
    [constants setConstantValue:&nxpsg type:MTLDataTypeShort atIndex:601];

    NSError *error = nil;
    NSString *name = [NSString stringWithUTF8String:function_name];
    id<MTLFunction> fn = [g_library newFunctionWithName:name
                                         constantValues:constants
                                                  error:&error];
    if (!fn) {
        fprintf(stderr, "ds4: Metal %s function not found: %s\n",
                function_name, [[error localizedDescription] UTF8String]);
        return nil;
    }

    error = nil;
    id<MTLComputePipelineState> pipeline = [g_device newComputePipelineStateWithFunction:fn error:&error];
    if (!pipeline) {
        fprintf(stderr, "ds4: Metal %s pipeline failed: %s\n",
                function_name, [[error localizedDescription] UTF8String]);
        return nil;
    }

    [g_pipeline_cache setObject:pipeline forKey:key];
    if (fast_key_valid) {
        ds4_gpu_decode_pipeline_fast_cache_insert(
            function_name, nsg, nxpsg, fast_name_len, fast_hash, pipeline);
    }
    return pipeline;
}

static id<MTLComputePipelineState> ds4_gpu_get_flash_attn_pad_pipeline(
        bool    has_mask,
        int32_t ncpsg) {
    /*
     * Decode calls this once per layer with identical arguments, so memoize
     * the last hit and skip the NSString key + dictionary lookup on the hot
     * path.  The generic cache below remains the fallback for new variants.
     * The rollback switch restores the dictionary path for same-binary A/B.
     */
    static struct {
        bool m;
        int32_t nc;
        id<MTLComputePipelineState> pipeline;
    } memo;
    const bool memo_disabled =
        getenv("DS4_METAL_DISABLE_PRE_M5_FLASH_ATTN_PAD_BLK_MEMO") != NULL;
    if (!memo_disabled && memo.pipeline && memo.m == has_mask && memo.nc == ncpsg) {
        return memo.pipeline;
    }

    NSString *key = [NSString stringWithFormat:@"kernel_flash_attn_ext_pad_mask=%d_ncpsg=%d",
                     has_mask ? 1 : 0, (int)ncpsg];
    id<MTLComputePipelineState> cached = [g_pipeline_cache objectForKey:key];
    if (cached) {
        if (!memo_disabled) {
            memo = (typeof(memo)){ has_mask, ncpsg, cached };
        }
        return cached;
    }

    MTLFunctionConstantValues *constants = [[MTLFunctionConstantValues alloc] init];
    [constants setConstantValue:&has_mask type:MTLDataTypeBool atIndex:100];
    [constants setConstantValue:&ncpsg type:MTLDataTypeInt atIndex:125];

    NSError *error = nil;
    id<MTLFunction> fn = [g_library newFunctionWithName:@"kernel_flash_attn_ext_pad"
                                         constantValues:constants
                                                  error:&error];
    if (!fn) {
        fprintf(stderr, "ds4: Metal kernel_flash_attn_ext_pad function not found: %s\n",
                [[error localizedDescription] UTF8String]);
        return nil;
    }

    error = nil;
    id<MTLComputePipelineState> pipeline = [g_device newComputePipelineStateWithFunction:fn error:&error];
    if (!pipeline) {
        fprintf(stderr, "ds4: Metal kernel_flash_attn_ext_pad pipeline failed: %s\n",
                [[error localizedDescription] UTF8String]);
        return nil;
    }

    [g_pipeline_cache setObject:pipeline forKey:key];
    if (!memo_disabled) {
        memo = (typeof(memo)){ has_mask, ncpsg, pipeline };
    }
    return pipeline;
}

static id<MTLComputePipelineState> ds4_gpu_get_flash_attn_blk_pipeline(
        int32_t nqptg,
        int32_t ncpsg) {
    /*
     * Decode calls this once per layer with identical arguments, so memoize
     * the last hit and skip the NSString key + dictionary lookup on the hot
     * path.  The generic cache below remains the fallback for new variants.
     * The rollback switch restores the dictionary path for same-binary A/B.
     */
    static struct {
        int32_t nq;
        int32_t nc;
        id<MTLComputePipelineState> pipeline;
    } memo;
    const bool memo_disabled =
        getenv("DS4_METAL_DISABLE_PRE_M5_FLASH_ATTN_PAD_BLK_MEMO") != NULL;
    if (!memo_disabled && memo.pipeline && memo.nq == nqptg && memo.nc == ncpsg) {
        return memo.pipeline;
    }

    NSString *key = [NSString stringWithFormat:@"kernel_flash_attn_ext_blk_nqptg=%d_ncpsg=%d",
                     (int)nqptg, (int)ncpsg];
    id<MTLComputePipelineState> cached = [g_pipeline_cache objectForKey:key];
    if (cached) {
        if (!memo_disabled) {
            memo = (typeof(memo)){ nqptg, ncpsg, cached };
        }
        return cached;
    }

    MTLFunctionConstantValues *constants = [[MTLFunctionConstantValues alloc] init];
    [constants setConstantValue:&nqptg type:MTLDataTypeInt atIndex:224];
    [constants setConstantValue:&ncpsg type:MTLDataTypeInt atIndex:225];

    NSError *error = nil;
    id<MTLFunction> fn = [g_library newFunctionWithName:@"kernel_flash_attn_ext_blk"
                                         constantValues:constants
                                                  error:&error];
    if (!fn) {
        fprintf(stderr, "ds4: Metal kernel_flash_attn_ext_blk function not found: %s\n",
                [[error localizedDescription] UTF8String]);
        return nil;
    }

    error = nil;
    id<MTLComputePipelineState> pipeline = [g_device newComputePipelineStateWithFunction:fn error:&error];
    if (!pipeline) {
        fprintf(stderr, "ds4: Metal kernel_flash_attn_ext_blk pipeline failed: %s\n",
                [[error localizedDescription] UTF8String]);
        return nil;
    }

    [g_pipeline_cache setObject:pipeline forKey:key];
    if (!memo_disabled) {
        memo = (typeof(memo)){ nqptg, ncpsg, pipeline };
    }
    return pipeline;
}

static id<MTLComputePipelineState> ds4_gpu_get_flash_attn_pipeline(
        const char *function_name,
        bool        has_mask,
        bool        has_sinks,
        bool        has_bias,
        bool        has_scap,
        bool        has_kvpad,
        bool        bc_mask,
        int32_t     ns10,
        int32_t     ns20,
        int32_t     nsg) {
    /*
     * Prefill and batched decode call this once per layer with identical
     * arguments, so memoize the last hit and skip the NSString key + dictionary
     * lookup on the hot path.  The generic cache below remains the fallback for
     * new variants.  The rollback switch restores the dictionary path for
     * same-binary A/B.
     */
    static struct {
        const char *fn;
        bool m, s, b, c, k, bc;
        int32_t n10, n20, sg;
        id<MTLComputePipelineState> pipeline;
    } memo;
    const bool memo_disabled =
        getenv("DS4_METAL_DISABLE_PRE_M5_FLASH_ATTN_BATCHED_MEMO") != NULL;
    if (!memo_disabled && memo.pipeline && memo.fn != NULL &&
        strcmp(memo.fn, function_name) == 0 &&
        memo.m == has_mask && memo.s == has_sinks && memo.b == has_bias &&
        memo.c == has_scap && memo.k == has_kvpad && memo.bc == bc_mask &&
        memo.n10 == ns10 && memo.n20 == ns20 && memo.sg == nsg) {
        return memo.pipeline;
    }

    NSString *key = [NSString stringWithFormat:@"%s_mask=%d_sinks=%d_bias=%d_scap=%d_kvpad=%d_bcm=%d_ns10=%d_ns20=%d_nsg=%d",
                     function_name,
                     has_mask ? 1 : 0,
                     has_sinks ? 1 : 0,
                     has_bias ? 1 : 0,
                     has_scap ? 1 : 0,
                     has_kvpad ? 1 : 0,
                     bc_mask ? 1 : 0,
                     (int)ns10,
                     (int)ns20,
                     (int)nsg];
    id<MTLComputePipelineState> cached = [g_pipeline_cache objectForKey:key];
    if (cached) {
        if (!memo_disabled) {
            memo = (typeof(memo)){ function_name, has_mask, has_sinks, has_bias,
                                   has_scap, has_kvpad, bc_mask, ns10, ns20,
                                   nsg, cached };
        }
        return cached;
    }

    MTLFunctionConstantValues *constants = [[MTLFunctionConstantValues alloc] init];
    [constants setConstantValue:&has_mask  type:MTLDataTypeBool atIndex:300];
    [constants setConstantValue:&has_sinks type:MTLDataTypeBool atIndex:301];
    [constants setConstantValue:&has_bias  type:MTLDataTypeBool atIndex:302];
    [constants setConstantValue:&has_scap  type:MTLDataTypeBool atIndex:303];
    [constants setConstantValue:&has_kvpad type:MTLDataTypeBool atIndex:304];
    [constants setConstantValue:&bc_mask   type:MTLDataTypeBool atIndex:310];
    [constants setConstantValue:&ns10 type:MTLDataTypeInt atIndex:320];
    [constants setConstantValue:&ns20 type:MTLDataTypeInt atIndex:321];
    [constants setConstantValue:&nsg  type:MTLDataTypeInt atIndex:322];

    NSError *error = nil;
    NSString *name = [NSString stringWithUTF8String:function_name];
    id<MTLFunction> fn = [g_library newFunctionWithName:name
                                         constantValues:constants
                                                  error:&error];
    if (!fn) {
        fprintf(stderr, "ds4: Metal %s function not found: %s\n",
                function_name, [[error localizedDescription] UTF8String]);
        return nil;
    }

    error = nil;
    id<MTLComputePipelineState> pipeline = [g_device newComputePipelineStateWithFunction:fn error:&error];
    if (!pipeline) {
        fprintf(stderr, "ds4: Metal %s pipeline failed: %s\n",
                function_name, [[error localizedDescription] UTF8String]);
        return nil;
    }

    [g_pipeline_cache setObject:pipeline forKey:key];
    if (!memo_disabled) {
        memo = (typeof(memo)){ function_name, has_mask, has_sinks, has_bias,
                               has_scap, has_kvpad, bc_mask, ns10, ns20,
                               nsg, pipeline };
    }
    return pipeline;
}

static id<MTLComputePipelineState> ds4_gpu_get_flash_attn_vec_pipeline(
        const char *function_name,
        bool        has_mask,
        bool        has_sinks,
        bool        has_bias,
        bool        has_scap,
        bool        has_kvpad,
        bool        shared_kvpad,
        bool        strided_kv,
        int32_t     ns10,
        int32_t     ns20,
        int32_t     nsg,
        int32_t     nwg) {
    /*
     * Decode calls this once per layer with identical arguments, so memoize
     * the last hit and skip the NSString key + dictionary lookup on the hot
     * path.  The generic cache below remains the fallback for new variants.
     */
    static struct {
        const char *fn;
        bool m, s, b, c, k, sp, st;
        int32_t n10, n20, sg, wg;
        id<MTLComputePipelineState> pipeline;
    } memo;
    if (memo.pipeline && memo.fn != NULL && strcmp(memo.fn, function_name) == 0 &&
        memo.m == has_mask && memo.s == has_sinks && memo.b == has_bias &&
        memo.c == has_scap && memo.k == has_kvpad && memo.sp == shared_kvpad &&
        memo.st == strided_kv &&
        memo.n10 == ns10 && memo.n20 == ns20 && memo.sg == nsg && memo.wg == nwg) {
        return memo.pipeline;
    }

    NSString *key = [NSString stringWithFormat:@"%s_mask=%d_sinks=%d_bias=%d_scap=%d_kvpad=%d_sharedpad=%d_strided=%d_ns10=%d_ns20=%d_nsg=%d_nwg=%d",
                     function_name,
                     has_mask ? 1 : 0,
                     has_sinks ? 1 : 0,
                     has_bias ? 1 : 0,
                     has_scap ? 1 : 0,
                     has_kvpad ? 1 : 0,
                     shared_kvpad ? 1 : 0,
                     strided_kv ? 1 : 0,
                     (int)ns10,
                     (int)ns20,
                     (int)nsg,
                     (int)nwg];
    id<MTLComputePipelineState> cached = [g_pipeline_cache objectForKey:key];
    if (cached) {
        memo = (typeof(memo)){ function_name, has_mask, has_sinks, has_bias,
                               has_scap, has_kvpad, shared_kvpad, strided_kv,
                               ns10, ns20,
                               nsg, nwg, cached };
        return cached;
    }

    MTLFunctionConstantValues *constants = [[MTLFunctionConstantValues alloc] init];
    [constants setConstantValue:&has_mask  type:MTLDataTypeBool atIndex:400];
    [constants setConstantValue:&has_sinks type:MTLDataTypeBool atIndex:401];
    [constants setConstantValue:&has_bias  type:MTLDataTypeBool atIndex:402];
    [constants setConstantValue:&has_scap  type:MTLDataTypeBool atIndex:403];
    [constants setConstantValue:&has_kvpad type:MTLDataTypeBool atIndex:404];
    [constants setConstantValue:&shared_kvpad type:MTLDataTypeBool atIndex:405];
    [constants setConstantValue:&strided_kv type:MTLDataTypeBool atIndex:406];
    [constants setConstantValue:&ns10 type:MTLDataTypeInt atIndex:420];
    [constants setConstantValue:&ns20 type:MTLDataTypeInt atIndex:421];
    [constants setConstantValue:&nsg  type:MTLDataTypeInt atIndex:422];
    [constants setConstantValue:&nwg  type:MTLDataTypeInt atIndex:423];

    NSError *error = nil;
    NSString *name = [NSString stringWithUTF8String:function_name];
    id<MTLFunction> fn = [g_library newFunctionWithName:name
                                         constantValues:constants
                                                  error:&error];
    if (!fn) {
        fprintf(stderr, "ds4: Metal %s function not found: %s\n",
                function_name, [[error localizedDescription] UTF8String]);
        return nil;
    }

    error = nil;
    id<MTLComputePipelineState> pipeline = [g_device newComputePipelineStateWithFunction:fn error:&error];
    if (!pipeline) {
        fprintf(stderr, "ds4: Metal %s pipeline failed: %s\n",
                function_name, [[error localizedDescription] UTF8String]);
        return nil;
    }

    [g_pipeline_cache setObject:pipeline forKey:key];
    memo = (typeof(memo)){ function_name, has_mask, has_sinks, has_bias,
                           has_scap, has_kvpad, shared_kvpad, strided_kv,
                           ns10, ns20,
                           nsg, nwg, pipeline };
    return pipeline;
}

/* Pipeline for the RoPE-fused decode reduce. Same function constants as the
 * plain reduce so the split-K geometry is identical. */

static id<MTLComputePipelineState> ds4_gpu_get_flash_attn_reduce_pipeline(
        int32_t dv,
        int32_t nwg) {
    /* Same per-layer memo pattern as the vec getter above. */
    static int32_t memo_dv, memo_nwg;
    static id<MTLComputePipelineState> memo_pipeline;
    if (memo_pipeline && memo_dv == dv && memo_nwg == nwg) {
        return memo_pipeline;
    }

    NSString *key = [NSString stringWithFormat:@"kernel_flash_attn_ext_vec_reduce_dv=%d_nwg=%d",
                     (int)dv, (int)nwg];
    id<MTLComputePipelineState> cached = [g_pipeline_cache objectForKey:key];
    if (cached) {
        memo_dv = dv; memo_nwg = nwg; memo_pipeline = cached;
        return cached;
    }

    MTLFunctionConstantValues *constants = [[MTLFunctionConstantValues alloc] init];
    [constants setConstantValue:&dv  type:MTLDataTypeInt atIndex:500];
    [constants setConstantValue:&nwg type:MTLDataTypeInt atIndex:501];

    NSError *error = nil;
    id<MTLFunction> fn = [g_library newFunctionWithName:@"kernel_flash_attn_ext_vec_reduce"
                                         constantValues:constants
                                                  error:&error];
    if (!fn) {
        fprintf(stderr, "ds4: Metal kernel_flash_attn_ext_vec_reduce function not found: %s\n",
                [[error localizedDescription] UTF8String]);
        return nil;
    }

    error = nil;
    id<MTLComputePipelineState> pipeline = [g_device newComputePipelineStateWithFunction:fn error:&error];
    if (!pipeline) {
        fprintf(stderr, "ds4: Metal kernel_flash_attn_ext_vec_reduce pipeline failed: %s\n",
                [[error localizedDescription] UTF8String]);
        return nil;
    }

    [g_pipeline_cache setObject:pipeline forKey:key];
    memo_dv = dv; memo_nwg = nwg; memo_pipeline = pipeline;
    return pipeline;
}

static uint32_t ds4_gpu_flash_attn_vec_nsg(uint32_t n_keys, uint32_t nwg, uint32_t ncpsg) {
    uint32_t nsg = 1;
    while (2u * nwg * nsg * ncpsg < n_keys && nsg < 4u) {
        nsg *= 2u;
    }
    return nsg;
}

static int ds4_gpu_trace_allocs(void) {
    static int initialized;
    static int enabled;
    if (!initialized) {
        enabled = getenv("DS4_METAL_TRACE_ALLOCS") != NULL;
        initialized = 1;
    }
    return enabled;
}

static double ds4_gpu_mib(uint64_t bytes) {
    return (double)bytes / (1024.0 * 1024.0);
}

static double ds4_gpu_gib(uint64_t bytes) {
    return (double)bytes / (1024.0 * 1024.0 * 1024.0);
}

static void ds4_gpu_print_task_memory_report(void) {
    task_vm_info_data_t info;
    mach_msg_type_number_t count = TASK_VM_INFO_COUNT;
    const kern_return_t kr = task_info(mach_task_self(),
                                       TASK_VM_INFO,
                                       (task_info_t)&info,
                                       &count);
    if (kr != KERN_SUCCESS) return;

    fprintf(stderr,
            "ds4:   macOS task memory footprint %.2f GiB, resident %.2f GiB, virtual %.2f GiB\n",
            ds4_gpu_gib((uint64_t)info.phys_footprint),
            ds4_gpu_gib((uint64_t)info.resident_size),
            ds4_gpu_gib((uint64_t)info.virtual_size));
}

void ds4_gpu_print_memory_report(const char *label) {
    pthread_mutex_lock(&g_tensor_mu);
    const uint64_t live = g_tensor_alloc_live_bytes;
    const uint64_t peak = g_tensor_alloc_peak_bytes;
    pthread_mutex_unlock(&g_tensor_mu);

    const uint64_t scratch = (uint64_t)g_flash_attn_pad_bytes +
        (uint64_t)g_flash_attn_tmp_bytes +
        (uint64_t)g_flash_attn_blk_bytes +
        (uint64_t)g_flash_attn_kv_bytes +
        (uint64_t)g_glm_flash_attn_mask_bytes +
        (uint64_t)g_indexer_topk_bytes;
    fprintf(stderr,
            "ds4: Metal memory%s%s: live %.2f MiB, peak %.2f MiB, "
            "scratch %.2f MiB\n",
            label && label[0] ? " " : "",
            label && label[0] ? label : "",
            ds4_gpu_mib(live), ds4_gpu_mib(peak), ds4_gpu_mib(scratch));
    ds4_gpu_print_task_memory_report();
    fprintf(stderr,
            "ds4:   mmap model %llu views, %.2f GiB total, %.2f GiB max; "
            "residency requests %llu\n",
            (unsigned long long)g_model_wrap_count,
            ds4_gpu_gib(g_model_wrap_bytes),
            ds4_gpu_gib(g_model_wrap_max_bytes),
            (unsigned long long)g_model_residency_count);
    fprintf(stderr, "ds4:   device %s, Metal tensor API %s\n",
            g_metal_device_name[0] ? g_metal_device_name : "(unknown)",
            g_metal4_tensor_api_enabled ? "enabled" : "disabled");
}

static int ds4_gpu_model_map_log_enabled(void) {
    if (!g_ssd_streaming_mode) return 1;
    const char *trace = getenv("DS4_METAL_STREAMING_MAP_TRACE");
    return trace && trace[0] && strcmp(trace, "0") != 0;
}

static id<MTLBuffer> ds4_gpu_wrap_model_range(
        const void *model_map,
        uint64_t    model_size,
        uint64_t    offset,
        uint64_t    len,
        uint64_t   *inner_offset);

static id<MTLBuffer> ds4_gpu_wrap_model_exact_range(
        const void *model_map,
        uint64_t    model_size,
        uint64_t    offset,
        uint64_t    len,
        uint64_t   *inner_offset);

static const char *ds4_gpu_source =
"#include <metal_stdlib>\n"
"#ifdef DS4_METAL_HAS_TENSOR\n"
"#include <metal_tensor>\n"
"#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>\n"
"#endif\n"
"using namespace metal;\n"
"#ifdef DS4_METAL_HAS_TENSOR\n"
"using namespace mpp::tensor_ops;\n"
"#endif\n"
"\n"
"#define MAX(x, y) ((x) > (y) ? (x) : (y))\n"
"#define MIN(x, y) ((x) < (y) ? (x) : (y))\n"
"#define SWAP(x, y) { auto tmp = (x); (x) = (y); (y) = tmp; }\n"
"#define QK8_0 32\n"
"#ifndef QK_K\n"
"#define QK_K 256\n"
"#endif\n"
"#define N_SIMDWIDTH 32\n"
"#define N_R0_Q8_0 2\n"
"#define N_SG_Q8_0 4\n"
"#define FC_MUL_MV 600\n"
"#define FC_MUL_MM 700\n"
"#define FC_BIN 1300\n"
"#define FOR_UNROLL(x) _Pragma(\"clang loop unroll(full)\") for (x)\n"
"#define M_PI_F 3.14159265358979323846f\n"
"\n"
"// Reads one byte per stride to warm model-backed pages without copying the\n"
"// model. This is outside inference and exists only to reduce first-use stalls.\n"
"kernel void kernel_touch_u8_stride(\n"
"        device const uchar    *src        [[buffer(0)]],\n"
"        device uchar          *dst        [[buffer(1)]],\n"
"        constant ulong        &stride     [[buffer(2)]],\n"
"        constant ulong        &bytes      [[buffer(3)]],\n"
"        constant ulong        &dst_offset [[buffer(4)]],\n"
"        uint gid [[thread_position_in_grid]]) {\n"
"    ulong off = (ulong)gid * stride;\n"
"    if (off >= bytes) return;\n"
"    dst[dst_offset + (ulong)gid] = src[off];\n"
"}\n"
"\n"
"enum ds4_sort_order {\n"
"    DS4_SORT_ORDER_ASC,\n"
"    DS4_SORT_ORDER_DESC,\n"
"};\n"
"\n"
"struct block_q8_0 {\n"
"    half d;\n"
"    int8_t qs[QK8_0];\n"
"};\n"
"\n"
"struct block_q8_K {\n"
"    float d;\n"
"    int8_t qs[QK_K];\n"
"    int16_t bsums[QK_K / 16];\n"
"};\n"
"\n"
"\n";

static NSString *ds4_gpu_full_source(void) {
    NSString *base = [NSString stringWithUTF8String:ds4_gpu_source];
    NSFileManager *fm = [NSFileManager defaultManager];
    /*
     * Kernels are kept as separate files for review, then concatenated into one
     * Metal library.  Environment overrides are still honored so a diagnostic
     * run can swap one source file without changing the executable.
     */
    NSArray<NSArray<NSString *> *> *required_sources = @[
        @[@"DS4_METAL_FLASH_ATTN_SOURCE", @"metal/flash_attn.metal"],
        @[@"DS4_METAL_DENSE_SOURCE",      @"metal/dense.metal"],
        @[@"DS4_METAL_QWEN35_SOURCE",     @"metal/qwen35.metal"],
        @[@"DS4_METAL_ARGSORT_SOURCE",    @"metal/argsort.metal"],
        @[@"DS4_METAL_CPY_SOURCE",        @"metal/cpy.metal"],
        @[@"DS4_METAL_GET_ROWS_SOURCE",   @"metal/get_rows.metal"],
        @[@"DS4_METAL_GLU_SOURCE",        @"metal/glu.metal"],
        @[@"DS4_METAL_NORM_SOURCE",       @"metal/norm.metal"],
        @[@"DS4_METAL_BIN_SOURCE",        @"metal/bin.metal"],
    ];

    NSMutableString *source = [NSMutableString stringWithString:base];
    for (NSArray<NSString *> *spec in required_sources) {
        const char *override_path = getenv([spec[0] UTF8String]);
        NSMutableArray<NSString *> *paths = [NSMutableArray array];
        if (override_path && override_path[0]) {
            [paths addObject:[NSString stringWithUTF8String:override_path]];
        }
        [paths addObject:spec[1]];
        [paths addObject:[@"./" stringByAppendingString:spec[1]]];

        NSString *loaded = nil;
        NSString *loaded_path = nil;
        for (NSString *path in paths) {
            if (![fm fileExistsAtPath:path]) continue;

            NSError *error = nil;
            loaded = [NSString stringWithContentsOfFile:path
                                               encoding:NSUTF8StringEncoding
                                                  error:&error];
            if (!loaded) {
                fprintf(stderr, "ds4: failed to read Metal source %s: %s\n",
                        [path UTF8String], [[error localizedDescription] UTF8String]);
                return nil;
            }
            loaded_path = path;
            break;
        }

        if (!loaded) {
            fprintf(stderr,
                    "ds4: Metal source %s not found (set %s to override)\n",
                    [spec[1] UTF8String], [spec[0] UTF8String]);
            return nil;
        }
        [source appendFormat:@"\n// appended %@\n%@\n", loaded_path, loaded];
    }
    return source;
}

typedef struct {
    uint32_t pos0;
    uint32_t n_tokens;
    uint32_t cache_cap;
    uint32_t n_head;
    uint32_t n_head_kv;
    uint32_t head_dim;
    uint32_t rot_dim;
    float eps;
    float freq_base;
} ds4_gpu_qwen35_full_prepare_args;

typedef struct {
    uint32_t pos0;
    uint32_t n_tokens;
    uint32_t cache_cap;
    uint32_t n_head;
    uint32_t n_head_kv;
    uint32_t head_dim;
    float scale;
} ds4_gpu_qwen35_attention_args;

typedef struct {
    uint32_t n_tokens;
    uint32_t channels;
    uint32_t qk_heads;
    uint32_t v_heads;
    uint32_t state_dim;
    uint32_t conv_width;
    float eps;
    float scale;
} ds4_gpu_qwen35_gdn_args;

typedef struct {
    int32_t  n_embd;
    int32_t  n_vocab;
    int32_t  n_tokens;
    uint64_t src_row_bytes;
    uint64_t dst_row_bytes;
    uint64_t token_stride;
} ds4_gpu_get_rows_q8_0_args;

typedef struct {
    int64_t  nk0;
    int64_t  ne00;
    int64_t  ne01;
    int64_t  ne02;
    int64_t  ne03;
    uint64_t nb00;
    uint64_t nb01;
    uint64_t nb02;
    uint64_t nb03;
    int64_t  ne0;
    int64_t  ne1;
    int64_t  ne2;
    int64_t  ne3;
    uint64_t nb0;
    uint64_t nb1;
    uint64_t nb2;
    uint64_t nb3;
} ds4_gpu_cpy_args;

static NSUInteger ds4_gpu_cpy_threads(uint32_t n, id<MTLComputePipelineState> pipeline) {
    NSUInteger nth = 32u;
    const NSUInteger max_threads = pipeline.maxTotalThreadsPerThreadgroup;
    while (nth < (NSUInteger)n && nth < max_threads) nth *= 2u;
    if (nth > max_threads) nth = max_threads;
    if (nth > (NSUInteger)n) nth = (NSUInteger)n;
    return nth ? nth : 1u;
}

typedef struct {
    int32_t  ne00;
    uint64_t nb01;
    int32_t  ne10;
    uint64_t nb11;
    int32_t  ne0;
    uint64_t nb1;
    int32_t  i00;
    int32_t  i10;
    float    alpha;
    float    limit;
} ds4_gpu_glu_args;

typedef struct {
    uint32_t n;
} ds4_gpu_add_flat_args;

typedef struct {
    int32_t  ne00;
    int32_t  ne01;
    int32_t  ne02;
    uint64_t nb00;
    uint64_t nb01;
    uint64_t nb02;
    uint64_t nb03;
    int32_t  ne10;
    int32_t  ne11;
    int32_t  ne12;
    uint64_t nb10;
    uint64_t nb11;
    uint64_t nb12;
    uint64_t nb13;
    int32_t  ne0;
    int32_t  ne1;
    int32_t  nr0;
    int16_t  r2;
    int16_t  r3;
} ds4_gpu_q8_0_matvec_args;

typedef struct {
    int32_t  ne00;
    int32_t  ne02;
    uint64_t nb01;
    uint64_t nb02;
    uint64_t nb03;
    int32_t  ne12;
    uint64_t nb10;
    uint64_t nb11;
    uint64_t nb12;
    uint64_t nb13;
    int32_t  ne0;
    int32_t  ne1;
    int16_t  r2;
    int16_t  r3;
} ds4_gpu_mul_mm_args;

typedef struct {
    int32_t  ne00;
    int32_t  ne01;
    int32_t  ne02;
    uint64_t nb00;
    uint64_t nb01;
    uint64_t nb02;
    uint64_t nb03;
    int32_t  ne10;
    int32_t  ne11;
    int32_t  ne12;
    uint64_t nb10;
    uint64_t nb11;
    uint64_t nb12;
    uint64_t nb13;
    int32_t  ne0;
    int32_t  ne1;
    int16_t  r2;
    int16_t  r3;
} ds4_gpu_mul_mv_ext_args;

static ds4_gpu_q8_0_matvec_args ds4_gpu_make_q8_0_mv_args(uint64_t in_dim, uint64_t out_dim) {
    const uint64_t row_bytes = (in_dim / 32u) * 34u;
    return (ds4_gpu_q8_0_matvec_args) {
        .ne00 = (int32_t)in_dim,
        .ne01 = (int32_t)out_dim,
        .ne02 = 1,
        .nb00 = 34,
        .nb01 = row_bytes,
        .nb02 = row_bytes * out_dim,
        .nb03 = row_bytes * out_dim,
        .ne10 = (int32_t)in_dim,
        .ne11 = 1,
        .ne12 = 1,
        .nb10 = sizeof(float),
        .nb11 = in_dim * sizeof(float),
        .nb12 = in_dim * sizeof(float),
        .nb13 = in_dim * sizeof(float),
        .ne0 = (int32_t)out_dim,
        .ne1 = 1,
        .nr0 = 2,
        .r2 = 1,
        .r3 = 1,
    };
}

typedef struct {
    const char *function_name;
    int16_t     nsg;
    int32_t     nr0;
    NSUInteger  smem;
} ds4_gpu_mv_dispatch;

static int ds4_gpu_tp_world_is_two(void);

static ds4_gpu_mv_dispatch ds4_gpu_make_q8_0_mv_dispatch(void) {
    const uint64_t default_nsg = ds4_gpu_tp_world_is_two() ? 2u : 4u;
    const int16_t nsg =
        (int16_t)ds4_gpu_env_u64("DS4_METAL_Q8_MV_NSG", default_nsg, 1u, 8u);
    return (ds4_gpu_mv_dispatch) {
        .function_name = "kernel_mul_mv_q8_0_f32",
        .nsg = nsg,
        .nr0 = 2,
        .smem = 32u * 2u * sizeof(float),
    };
}

/* Standalone decode matvec row-ownership experiment.  NR4 does not change the
 * per-row K traversal or reduction tree and is retained as a bit-identical
 * diagnostic.  Paired/fused projection kernels keep their tuned NR2 shape. */
static ds4_gpu_mv_dispatch ds4_gpu_make_q8_0_single_mv_dispatch(void) {
    ds4_gpu_mv_dispatch dispatch = ds4_gpu_make_q8_0_mv_dispatch();
    const uint64_t nr0 = ds4_gpu_env_u64(
        "DS4_METAL_Q8_MV_SINGLE_NR0", 2u, 2u, 4u);
    if (nr0 == 4u) {
        dispatch.function_name = "kernel_mul_mv_q8_0_f32_nr4";
        dispatch.nr0 = 4;
    }
    dispatch.smem = 32u * (NSUInteger)dispatch.nr0 * sizeof(float);
    return dispatch;
}

static ds4_gpu_mul_mm_args ds4_gpu_make_mm_args(
        uint64_t in_dim,
        uint64_t out_dim,
        uint64_t n_tok,
        uint64_t row_bytes) {
    return (ds4_gpu_mul_mm_args) {
        .ne00 = (int32_t)in_dim,
        .ne02 = 1,
        .nb01 = row_bytes,
        .nb02 = row_bytes * out_dim,
        .nb03 = row_bytes * out_dim,
        .ne12 = 1,
        .nb10 = sizeof(float),
        .nb11 = in_dim * sizeof(float),
        .nb12 = in_dim * n_tok * sizeof(float),
        .nb13 = in_dim * n_tok * sizeof(float),
        .ne0 = (int32_t)out_dim,
        .ne1 = (int32_t)n_tok,
        .r2 = 1,
        .r3 = 1,
    };
}

static ds4_gpu_mul_mv_ext_args ds4_gpu_make_mv_ext_args(
        uint64_t in_dim,
        uint64_t out_dim,
        uint64_t n_tok,
        uint64_t elem_bytes,
        uint64_t row_bytes) {
    return (ds4_gpu_mul_mv_ext_args) {
        .ne00 = (int32_t)in_dim,
        .ne01 = (int32_t)out_dim,
        .ne02 = 1,
        .nb00 = elem_bytes,
        .nb01 = row_bytes,
        .nb02 = row_bytes * out_dim,
        .nb03 = row_bytes * out_dim,
        .ne10 = (int32_t)in_dim,
        .ne11 = (int32_t)n_tok,
        .ne12 = 1,
        .nb10 = sizeof(float),
        .nb11 = in_dim * sizeof(float),
        .nb12 = in_dim * n_tok * sizeof(float),
        .nb13 = in_dim * n_tok * sizeof(float),
        .ne0 = (int32_t)out_dim,
        .ne1 = (int32_t)n_tok,
        .r2 = 1,
        .r3 = 1,
    };
}

static int16_t ds4_gpu_mv_ext_nxpsg(uint64_t in_dim, uint64_t n_tok) {
    if ((in_dim % 256u) == 0 && n_tok < 3) return 16;
    if ((in_dim % 128u) == 0) return 8;
    return 4;
}

static int16_t ds4_gpu_mv_ext_r1ptg(uint64_t n_tok) {
    switch (n_tok) {
    case 2: return 2;
    case 3:
    case 6: return 3;
    case 4:
    case 7:
    case 8: return 4;
    case 5: return 5;
    default: return n_tok > 8 ? 4 : 0;
    }
}

static const char *ds4_gpu_mv_ext_name(int q8, int16_t r1ptg) {
    if (q8) {
        switch (r1ptg) {
        case 2: return "kernel_mul_mv_ext_q8_0_f32_r1_2";
        case 3: return "kernel_mul_mv_ext_q8_0_f32_r1_3";
        case 4: return "kernel_mul_mv_ext_q8_0_f32_r1_4";
        case 5: return "kernel_mul_mv_ext_q8_0_f32_r1_5";
        default: return NULL;
        }
    }

    switch (r1ptg) {
    case 2: return "kernel_mul_mv_ext_f16_f32_r1_2";
    case 3: return "kernel_mul_mv_ext_f16_f32_r1_3";
    case 4: return "kernel_mul_mv_ext_f16_f32_r1_4";
    case 5: return "kernel_mul_mv_ext_f16_f32_r1_5";
    default: return NULL;
    }
}

typedef struct {
    int32_t  ne00;
    int32_t  ne00_t;
    uint64_t nb1;
    uint64_t nb2;
    uint64_t nb3;
    float    eps;
    int32_t  nef1[3];
    int32_t  nef2[3];
    int32_t  nef3[3];
    uint64_t nbf1[3];
    uint64_t nbf2[3];
    uint64_t nbf3[3];
} ds4_gpu_rms_norm_args;

static ds4_gpu_rms_norm_args ds4_gpu_make_rms_norm_args(uint32_t n, uint32_t rows, float eps) {
    const uint64_t row_bytes = (uint64_t)n * sizeof(float);
    return (ds4_gpu_rms_norm_args) {
        .ne00 = (int32_t)n,
        .ne00_t = (int32_t)(n / 4u),
        .nb1 = row_bytes,
        .nb2 = row_bytes * rows,
        .nb3 = row_bytes * rows,
        .eps = eps,
        .nef1 = { (int32_t)rows, 1, 1 },
        .nef2 = { 1, 1, 1 },
        .nef3 = { 1, 1, 1 },
        .nbf1 = { row_bytes, row_bytes, row_bytes },
        .nbf2 = { row_bytes * rows, row_bytes, row_bytes },
        .nbf3 = { row_bytes * rows, row_bytes, row_bytes },
    };
}

static NSUInteger ds4_gpu_rms_norm_threads(uint32_t n) {
    NSUInteger ne00_t = n / 4u;
    NSUInteger nth = 32u;
    while (nth < ne00_t && nth < 1024u) nth *= 2u;
    if (nth > ne00_t) nth = ne00_t;
    return nth ? nth : 1u;
}

typedef struct {
    int32_t  ne11;
    int32_t  ne_12_2;
    int32_t  ne_12_3;
    uint64_t nb11;
    uint64_t nb12;
    uint64_t nb13;
    uint64_t nb21;
    uint64_t nb22;
    uint64_t nb23;
    int32_t  ne31;
    int32_t  ne32;
    int32_t  ne33;
    uint64_t nb31;
    uint64_t nb32;
    uint64_t nb33;
} ds4_gpu_flash_attn_pad_args;

typedef struct {
    int32_t  ne01;
    int32_t  ne30;
    int32_t  ne31;
    int32_t  ne32;
    int32_t  ne33;
    uint64_t nb31;
    uint64_t nb32;
    uint64_t nb33;
} ds4_gpu_flash_attn_blk_args;

typedef struct {
    int32_t  ne01;
    int32_t  ne02;
    int32_t  ne03;
    uint64_t nb01;
    uint64_t nb02;
    uint64_t nb03;
    int32_t  ne11;
    int32_t  ne_12_2;
    int32_t  ne_12_3;
    int32_t  ns10;
    uint64_t nb11;
    uint64_t nb12;
    uint64_t nb13;
    int32_t  ns20;
    uint64_t nb21;
    uint64_t nb22;
    uint64_t nb23;
    int32_t  ne31;
    int32_t  ne32;
    int32_t  ne33;
    uint64_t nb31;
    uint64_t nb32;
    uint64_t nb33;
    int32_t  ne1;
    int32_t  ne2;
    int32_t  ne3;
    float    scale;
    float    max_bias;
    float    m0;
    float    m1;
    int32_t  n_head_log2;
    float    logit_softcap;
} ds4_gpu_flash_attn_vec_args;

typedef struct {
    int32_t nrows;
} ds4_gpu_flash_attn_reduce_args;

typedef struct {
    uint64_t row_bytes;
    uint64_t token_bytes;
    int32_t head_dim;
    int32_t n_dims;
    int32_t n_ctx_orig;
    int32_t inverse;
    uint32_t pos0;
    uint32_t pos_step;
    float freq_base;
    float freq_scale;
    float ext_factor;
    float attn_factor;
    float beta_fast;
    float beta_slow;
} ds4_gpu_rope_affine_pair_args;

/* Matches ds4_metal_args_dsv4_comp_finalize in dsv4_rope.metal. */

/* Set by ds4.c immediately before the decode attention call when the inverse
 * RoPE tail is deferred into the reduce kernel. Cleared by the encoder. */
/* Set only by the encoder that actually applied the deferred rotation, so ds4.c
 * can fall back to the standalone RoPE on any attention path that does not
 * consume it (for example ratio-0 layers that take the raw-heads encoder). */

_Static_assert(sizeof(ds4_gpu_rope_affine_pair_args) == 64,
               "Metal affine RoPE argument ABI changed");

typedef struct {
    int32_t  ne00;
    int32_t  ne01;
    int32_t  ne02;
    int32_t  ne03;
    uint64_t nb00;
    uint64_t nb01;
    uint64_t nb02;
    uint64_t nb03;
    int32_t  ne0;
    int32_t  ne1;
    int32_t  ne2;
    int32_t  ne3;
    int32_t  top_k;
} ds4_gpu_kargs_argsort;

typedef struct {
    int64_t  ne00;
    int64_t  ne01;
    int64_t  ne02;
    int64_t  ne03;
    uint64_t nb00;
    uint64_t nb01;
    uint64_t nb02;
    uint64_t nb03;
    int32_t  ne0;
    int32_t  ne1;
    int32_t  ne2;
    int32_t  ne3;
    int32_t  top_k;
    int32_t  len;
} ds4_gpu_kargs_argsort_merge;

/* Compile the single in-repo Metal source and create the pipelines that every
 * session uses. Shape-dependent kernels with function constants are built
 * lazily by the small ds4_gpu_get_* caches, so startup stays predictable
 * while long-context prefill and decode can still pick specialized variants. */
int ds4_gpu_init(void) {
    if (g_initialized) return 1;

    @autoreleasepool {
        ds4_gpu_decode_pipeline_fast_cache_reset();
        g_device = MTLCreateSystemDefaultDevice();
        if (!g_device) {
            fprintf(stderr, "ds4: Metal device not available\n");
            return 0;
        }
        ds4_gpu_print_device_summary();
        ds4_gpu_detect_metal4_features();

        g_queue = [g_device newCommandQueue];
        g_model_buffer_cache = [NSMutableDictionary dictionary];
        g_pipeline_cache = [NSMutableDictionary dictionary];
        g_transient_buffers = [NSMutableArray array];
        g_pending_cbs = [NSMutableArray array];
        if (!g_queue || !g_model_buffer_cache || !g_pipeline_cache ||
            !g_transient_buffers || !g_pending_cbs) {
            fprintf(stderr, "ds4: Metal bookkeeping allocation failed\n");
            g_queue = nil;
            g_device = nil;
            return 0;
        }

        NSError *error = nil;
        NSString *source = ds4_gpu_full_source();
        if (!source) return 0;
        MTLCompileOptions *options = [MTLCompileOptions new];
        NSMutableDictionary *macros = [NSMutableDictionary new];
        if (g_metal4_tensor_api_enabled) {
            macros[@"DS4_METAL_HAS_TENSOR"] = @"1";
            fprintf(stderr, "ds4: Metal 4 tensor API enabled for Tensor kernels\n");
        }
        options.preprocessorMacros = macros;
        g_library = [g_device newLibraryWithSource:source
                                           options:options
                                             error:&error];
        if (!g_library) {
            fprintf(stderr, "ds4: Metal shader compilation failed: %s\n",
                    [[error localizedDescription] UTF8String]);
            return 0;
        }

#define DS4_QWEN_PIPELINE(slot_, name_) do { \
        id<MTLFunction> fn_ = [g_library newFunctionWithName:@name_]; \
        error = nil; \
        slot_ = fn_ ? [g_device newComputePipelineStateWithFunction:fn_ \
                                                             error:&error] : nil; \
        if (!slot_) { \
            fprintf(stderr, "ds4: Metal %s pipeline failed: %s\n", name_, \
                    error ? [[error localizedDescription] UTF8String] : \
                            "function not found"); \
            return 0; \
        } \
    } while (0)
        DS4_QWEN_PIPELINE(g_get_rows_q8_0_pipeline,
                          "kernel_get_rows_q8_0_f32");
        DS4_QWEN_PIPELINE(g_cpy_f32_f16_pipeline,
                          "kernel_cpy_f32_f16");
        DS4_QWEN_PIPELINE(g_cpy_f16_f16_pipeline,
                          "kernel_cpy_f16_f16");
        DS4_QWEN_PIPELINE(g_swiglu_flat_pipeline,
                          "kernel_swiglu_flat_f32");
        DS4_QWEN_PIPELINE(g_add2_pipeline,
                          "kernel_add2_f32");
        DS4_QWEN_PIPELINE(g_rms_norm_pipeline,
                          "kernel_rms_norm_mul_f32_4");
        DS4_QWEN_PIPELINE(g_add_rms_norm_pipeline,
                          "kernel_add_rms_norm_mul_f32_4");
        DS4_QWEN_PIPELINE(g_argsort_f32_i32_desc_pipeline,
                          "kernel_argsort_f32_i32_desc");
        DS4_QWEN_PIPELINE(g_argsort_merge_f32_i32_desc_pipeline,
                          "kernel_argsort_merge_f32_i32_desc");
#undef DS4_QWEN_PIPELINE
        g_initialized = 1;
    }
    return 1;
}

ds4_gpu_tensor *ds4_gpu_tensor_alloc(uint64_t bytes) {
    if (!g_initialized && !ds4_gpu_init()) return NULL;
    if (bytes == 0 || bytes > (uint64_t)NSUIntegerMax) return NULL;

    @autoreleasepool {
        DS4MetalTensor *tensor = [DS4MetalTensor new];
        tensor.buffer = [g_device newBufferWithLength:(NSUInteger)bytes
                                              options:MTLResourceStorageModeShared];
        if (!tensor.buffer) {
            return NULL;
        }
        tensor.offset = 0;
        tensor.bytes = bytes;
        tensor.owner = 1;
        uint64_t live_snap = 0;
        uint64_t peak_snap = 0;
        pthread_mutex_lock(&g_tensor_mu);
        const int tracked = ds4_gpu_tensor_track_alloc_locked(
                (__bridge const void *)tensor,
                bytes,
                &live_snap,
                &peak_snap);
        pthread_mutex_unlock(&g_tensor_mu);
        if (!tracked) {
            fprintf(stderr, "ds4: failed to track Metal tensor allocation\n");
            tensor.buffer = nil;
            return NULL;
        }
        if (ds4_gpu_trace_allocs()) {
            fprintf(stderr,
                    "ds4: Metal tensor alloc %.3f MiB live %.3f MiB peak %.3f MiB\n",
                    (double)bytes / (1024.0 * 1024.0),
                    (double)live_snap / (1024.0 * 1024.0),
                    (double)peak_snap / (1024.0 * 1024.0));
        }
        return (__bridge_retained ds4_gpu_tensor *)tensor;
    }
}

ds4_gpu_tensor *ds4_gpu_tensor_view(const ds4_gpu_tensor *base, uint64_t offset, uint64_t bytes) {
    if (!base) return NULL;
    const DS4MetalTensor *base_obj = ds4_gpu_tensor_const_obj(base);
    if (offset > base_obj.bytes || bytes > base_obj.bytes - offset) return NULL;
    if (base_obj.offset > UINT64_MAX - offset) return NULL;
    const uint64_t absolute_offset = base_obj.offset + offset;
    if (absolute_offset > (uint64_t)NSUIntegerMax) return NULL;

    @autoreleasepool {
        DS4MetalTensor *view = [DS4MetalTensor new];
        view.buffer = base_obj.buffer;
        view.offset = absolute_offset;
        view.bytes = bytes;
        view.owner = 0;
        pthread_mutex_lock(&g_tensor_mu);
        const int tracked = ds4_gpu_tensor_track_view_locked((__bridge const void *)view);
        pthread_mutex_unlock(&g_tensor_mu);
        if (!tracked) {
            fprintf(stderr, "ds4: failed to track Metal tensor view\n");
            view.buffer = nil;
            return NULL;
        }
        return (__bridge_retained ds4_gpu_tensor *)view;
    }
}

void ds4_gpu_tensor_free(ds4_gpu_tensor *tensor) {
    if (!tensor) return;
    @autoreleasepool {
        uint8_t owner = 0;
        uint64_t bytes = 0;
        uint64_t live_snap = 0;
        uint64_t peak_snap = 0;
        if (!ds4_gpu_tensor_prepare_free(tensor,
                                         &owner,
                                         &bytes,
                                         &live_snap,
                                         &peak_snap)) {
            return;
        }
        DS4MetalTensor *obj = (__bridge_transfer DS4MetalTensor *)tensor;
        if (owner) {
            if (ds4_gpu_trace_allocs()) {
                fprintf(stderr,
                        "ds4: Metal tensor free %.3f MiB live %.3f MiB peak %.3f MiB\n",
                        (double)bytes / (1024.0 * 1024.0),
                        (double)live_snap / (1024.0 * 1024.0),
                        (double)peak_snap / (1024.0 * 1024.0));
            }
        }
        obj.buffer = nil;
        obj.offset = 0;
        obj.bytes = 0;
        obj.owner = 0;
    }
}

uint64_t ds4_gpu_tensor_bytes(const ds4_gpu_tensor *tensor) {
    if (!tensor) return 0;
    const DS4MetalTensor *obj = ds4_gpu_tensor_const_obj(tensor);
    return obj.bytes;
}

void *ds4_gpu_tensor_contents(ds4_gpu_tensor *tensor) {
    if (!tensor) return NULL;
    DS4MetalTensor *obj = ds4_gpu_tensor_obj(tensor);
    return (uint8_t *)[obj.buffer contents] + obj.offset;
}

int ds4_gpu_tensor_fill_f32(ds4_gpu_tensor *tensor, float value, uint64_t count) {
    if (!tensor || count > ds4_gpu_tensor_bytes(tensor) / sizeof(float)) return 0;
    float *p = ds4_gpu_tensor_contents(tensor);
    if (!p && count != 0) return 0;
    for (uint64_t i = 0; i < count; i++) p[i] = value;
    return 1;
}

int ds4_gpu_tensor_write(ds4_gpu_tensor *tensor, uint64_t offset, const void *data, uint64_t bytes) {
    if (!tensor || (!data && bytes != 0)) return 0;
    DS4MetalTensor *obj = ds4_gpu_tensor_obj(tensor);
    if (offset > obj.bytes || bytes > obj.bytes - offset) return 0;
    if (bytes != 0) {
        memcpy((uint8_t *)[obj.buffer contents] + obj.offset + offset, data, (size_t)bytes);
    }
    return 1;
}

int ds4_gpu_tensor_read(const ds4_gpu_tensor *tensor, uint64_t offset, void *data, uint64_t bytes) {
    if (!tensor || (!data && bytes != 0)) return 0;
    const DS4MetalTensor *obj = ds4_gpu_tensor_const_obj(tensor);
    if (offset > obj.bytes || bytes > obj.bytes - offset) return 0;
    if (bytes != 0) {
        memcpy(data, (const uint8_t *)[obj.buffer contents] + obj.offset + offset, (size_t)bytes);
    }
    return 1;
}

int ds4_gpu_tensor_copy(ds4_gpu_tensor *dst, uint64_t dst_offset,
                          const ds4_gpu_tensor *src, uint64_t src_offset,
                          uint64_t bytes) {
    if (!dst || !src) return 0;
    if (!g_initialized && !ds4_gpu_init()) return 0;
    DS4MetalTensor *d = ds4_gpu_tensor_obj(dst);
    const DS4MetalTensor *s = ds4_gpu_tensor_const_obj(src);
    if (dst_offset > d.bytes || bytes > d.bytes - dst_offset) return 0;
    if (src_offset > s.bytes || bytes > s.bytes - src_offset) return 0;
    if (bytes == 0) return 1;
    if (!g_batch_cb) return 0;

    ds4_gpu_close_batch_encoder();
    g_batch_has_work = YES;
    id<MTLBlitCommandEncoder> blit = [g_batch_cb blitCommandEncoder];
    if (!blit) return 0;
    [blit copyFromBuffer:s.buffer
            sourceOffset:(NSUInteger)(s.offset + src_offset)
                toBuffer:d.buffer
       destinationOffset:(NSUInteger)(d.offset + dst_offset)
                    size:(NSUInteger)bytes];
    [blit endEncoding];
    return 1;
}

int ds4_gpu_begin_commands(void) {
    if (!g_initialized && !ds4_gpu_init()) return 0;
    if (g_batch_cb) return 0;
    g_batch_cb = ds4_gpu_new_command_buffer();
    g_batch_has_work = NO;
    return g_batch_cb != nil;
}

int ds4_gpu_commands_active(void) {
    return g_batch_cb != nil;
}

/* Exact M5 full-FFN overlap inside one concurrent compute encoder.  Shared
 * gate/up and routed IQ2 pair-SwiGLU launch together; explicit level barriers
 * precede the routed Q2 and shared Q8 down consumers. */

/* Reset is deliberately idempotent. Closing the concurrent encoder preserves
 * work already encoded, while clearing every admission/reference field keeps
 * a failed FFN path from turning later ordinary dispatches concurrent. */

/* Clang cleanup makes every early return from the large generic routed-MoE
 * wrapper abort an armed concurrent FFN without touching its many
 * established fallback branches. */

/*
 * Tensor-parallel gates.
 *
 * A TP gate is a mid-command-stream rendezvous with the peer machine: the
 * kernels ahead of the gate leave a partial block output in a slab slot,
 * the GPU signals g_tp_gpu_event, and the pre-encoded combine kernel waits
 * on g_tp_cpu_event.  A dedicated service thread bridges the two: it spins
 * until the GPU reaches the gate, runs the transport exchange (RDMA WRITE
 * plus flag poll, or a TCP write/read pair — behind the callback), and
 * CPU-signals the release.  On exchange failure the release is signaled
 * anyway so the GPU never deadlocks; the failure latches in g_tp_failed
 * and the eval aborts at the next command-buffer boundary.
 *
 * Gate sequence values increase monotonically per encoded gate.  Both ranks
 * encode the identical graph, so the values agree by construction and slots
 * never need resetting between tokens.
 */

enum { DS4_GPU_TP_QUEUE = 1024 };

/* Batch (verify-block) gates run on their own sequence space and release
 * event: the row-gate seq feeds the RDMA pre-posted recv accounting, which
 * requires consecutive values, and a shared release event would make a
 * small batch value satisfy waits armed against the larger row seq. */
/* Batch flag values are tagged so a stale row-gate seq in the reused FFN
 * flag word can never satisfy a batch arrival spin (and vice versa). */
#define DS4_TP_BATCH_FLAG_TAG 0x80000000u
/* Expert-ownership split parameters for routed kernels. World 1 means TP is
 * not bound; world 2 assigns each rank one contiguous expert range. */
static int32_t g_tp_split_world = 1;

static int ds4_gpu_tp_world_is_two(void) {
    return g_tp_split_world == 2;
}

/* Return the contiguous routed-expert range backed by this process. Rank 1
 * owns the high range and receives any odd-count remainder. */

/* Attention head split for GLM batch prefill: each rank computes a
 * contiguous half of the heads in the qk-low / attention-lora /
 * value-project batch kernels; the caller zeroes the unowned head range
 * of the heads buffer and combines the attn-output partials over the
 * TP big-gate exchange. */

/* Flag gates (DS4_TP_FLAG_GATES): the GPU publishes gate arrival by storing
 * the sequence number into a slab word instead of signaling the shared
 * event; the service thread spin-reads it from shared memory, which wakes
 * hundreds of microseconds earlier than signaledValue polling.  The
 * CPU->GPU release direction stays on the shared event. */

/* GPU keep-alive (see kernel_dsv4_tp_keepalive): its own queue and thread,
 * alive exactly as long as the TP gate machinery. */

/* Nonzero while a verify block runs: the GPU is genuinely busy
 * there, so the keep-alive is a pure parasite (~2.3ms per 5-row block
 * measured against the single-machine verify). */

/* Verify-block batch gate: same arrival/release machinery as the row gate
 * (the FFN flag word and event pair are reused — a decode gate and a batch
 * gate are never in flight together, and seq values stay globally unique),
 * but the service thread runs the multi-row exchange callback. */

/* Prefill batch gate kick: same seq space and release event as the verify
 * batch gate, but the service thread exchanges big_bytes directly between
 * the two shared bounce buffers instead of slab slots.  The kick only
 * publishes the GPU arrival marker and queues the exchange; the caller
 * encodes the release wait later through ds4_gpu_tp_big_gate_wait, which
 * lets it interleave more GPU work with the wire exchange.  Arrival always
 * uses the batch shared event, NOT the flag word: a flag write carries no
 * memory-visibility guarantee for the payload buffer, and once the GPU
 * keeps running past the kick (no event wait right behind it) the service
 * thread can observe the flag before the producing kernels' stores reach
 * CPU-visible memory (measured: stale rows in the first sub-kick).  The
 * shared-event signal only fires after every preceding command completes,
 * which is exactly the payload ordering the exchange needs; the ~10 us
 * slower arrival detection is noise against a multi-ms exchange. */

/* Encode the GPU-side release wait for a previously kicked big gate.  The
 * batch release event is monotonic and the service thread completes queued
 * exchanges in kick order, so waiting on the LAST kicked seq of a stage
 * also covers every earlier kick. */

int ds4_gpu_end_commands(void) {
    if (!g_batch_cb) return 0;
    ds4_gpu_close_batch_encoder();
    id<MTLCommandBuffer> cb = g_batch_cb;
    g_batch_cb = nil;
    g_batch_has_work = NO;
    return ds4_gpu_finish_command_buffer(cb, 1, "command batch");
}

int ds4_gpu_synchronize(void) {
    if (!g_initialized && !ds4_gpu_init()) return 0;
    if (g_batch_cb) return ds4_gpu_end_commands();
    id<MTLCommandBuffer> cb = ds4_gpu_new_command_buffer();
    if (!cb) return 0;
    return ds4_gpu_finish_command_buffer(cb, 1, "synchronize");
}

void ds4_gpu_cleanup(void) {
    if (!g_initialized) return;
    @autoreleasepool {
        if (g_batch_cb) {
            ds4_gpu_close_batch_encoder();
            [g_batch_cb commit];
            [g_batch_cb waitUntilCompleted];
            g_batch_cb = nil;
        }
        g_get_rows_q8_0_pipeline = nil;
        g_cpy_f32_f16_pipeline = nil;
        g_cpy_f16_f16_pipeline = nil;
        g_swiglu_flat_pipeline = nil;
        g_add2_pipeline = nil;
        g_rms_norm_pipeline = nil;
        g_add_rms_norm_pipeline = nil;
        g_argsort_f32_i32_desc_pipeline = nil;
        g_argsort_merge_f32_i32_desc_pipeline = nil;
        g_selected_readback_event = nil;
        g_selected_readback_event_value = 0;
        ds4_gpu_tensor_tracking_reset();
        ds4_gpu_model_residency_clear();
        ds4_gpu_model_views_clear();
        [g_pipeline_cache removeAllObjects];
        [g_model_buffer_cache removeAllObjects];
        [g_transient_buffers removeAllObjects];
        g_pipeline_cache = nil;
        g_model_buffer_cache = nil;
        g_transient_buffers = nil;
        g_library = nil;
        g_queue = nil;
        g_device = nil;
        g_model_fd = -1;
        g_model_map_ptr = NULL;
        g_model_map_size = 0;
        g_model_mapped_offset = 0;
        g_model_mapped_size = 0;
        g_model_mapped_max_tensor_bytes = 0;
        g_initialized = 0;
    }
}

static uint64_t ds4_gpu_q8_0_row_bytes(uint32_t n_embd) {
    return (((uint64_t)n_embd + 31u) / 32u) * 34u;
}

static int ds4_gpu_q8_0_table_bytes(
        uint32_t  n_vocab,
        uint32_t  n_embd,
        uint64_t *bytes_out) {
    if (!bytes_out || n_vocab == 0 || n_embd == 0) return 0;
    const uint64_t row_bytes = ds4_gpu_q8_0_row_bytes(n_embd);
    if (row_bytes != 0 && (uint64_t)n_vocab > UINT64_MAX / row_bytes) return 0;
    *bytes_out = (uint64_t)n_vocab * row_bytes;
    return 1;
}

static int ds4_gpu_encode_get_rows_q8_0(
        id<MTLCommandBuffer> cb,
        id<MTLBuffer>        weight,
        NSUInteger           weight_offset,
        id<MTLBuffer>        tokens,
        NSUInteger           tokens_offset,
        const int32_t       *single_token,
        id<MTLBuffer>        out,
        NSUInteger           out_offset,
        uint32_t             n_vocab,
        uint32_t             n_tokens,
        uint32_t             n_embd) {
    if (!cb || !weight || !out || n_vocab == 0 || n_tokens == 0 || n_embd == 0) {
        return 0;
    }
    if (!tokens && (!single_token || n_tokens != 1)) {
        return 0;
    }

    ds4_gpu_get_rows_q8_0_args args = {
        .n_embd = (int32_t)n_embd,
        .n_vocab = (int32_t)n_vocab,
        .n_tokens = (int32_t)n_tokens,
        .src_row_bytes = ds4_gpu_q8_0_row_bytes(n_embd),
        .dst_row_bytes = (uint64_t)n_embd * sizeof(float),
        .token_stride = sizeof(int32_t),
    };

    NSUInteger nth = 32u;
    const NSUInteger max_threads = g_get_rows_q8_0_pipeline.maxTotalThreadsPerThreadgroup;
    if (nth > max_threads) nth = max_threads;
    if (nth == 0) nth = 1;
    const NSUInteger nblocks = ((NSUInteger)n_embd + 31u) / 32u;

    id<MTLComputeCommandEncoder> enc = ds4_gpu_compute_encoder(cb);
    [enc setComputePipelineState:g_get_rows_q8_0_pipeline];
    [enc setBytes:&args length:sizeof(args) atIndex:0];
    [enc setBuffer:weight offset:weight_offset atIndex:1];
    if (tokens) {
        [enc setBuffer:tokens offset:tokens_offset atIndex:2];
    } else {
        [enc setBytes:single_token length:sizeof(*single_token) atIndex:2];
    }
    [enc setBuffer:out offset:out_offset atIndex:3];
    [enc dispatchThreadgroups:MTLSizeMake(nblocks, n_tokens, 1)
         threadsPerThreadgroup:MTLSizeMake(nth, 1, 1)];
    ds4_gpu_end_compute_encoder(cb, enc);
    return 1;
}

int ds4_gpu_embed_token_q8_0_tensor(
        ds4_gpu_tensor *out,
        const void       *model_map,
        uint64_t          model_size,
        uint64_t          weight_offset,
        uint32_t          n_vocab,
        uint32_t          token,
        uint32_t          n_embd) {
    if (!g_initialized && !ds4_gpu_init()) return 0;
    if (!out || !model_map || n_vocab == 0 || token >= n_vocab || n_embd == 0) {
        return 0;
    }

    @autoreleasepool {
        id<MTLBuffer> outbuf = ds4_gpu_tensor_buffer(out);
        const uint64_t out_bytes = (uint64_t)n_embd * sizeof(float);
        if (!outbuf || ds4_gpu_tensor_bytes(out) < out_bytes) {
            fprintf(stderr, "ds4: Metal Q8_0 embedding received undersized output buffer\n");
            return 0;
        }

        uint64_t weight_bytes = 0;
        if (!ds4_gpu_q8_0_table_bytes(n_vocab, n_embd, &weight_bytes) ||
            weight_offset > model_size ||
            weight_bytes > model_size - weight_offset) {
            fprintf(stderr, "ds4: Metal Q8_0 embedding range is outside the mapped model\n");
            return 0;
        }

        uint64_t inner_offset = 0;
        uint32_t token_for_kernel = token;
        id<MTLBuffer> wbuf = nil;
        const bool exact_token_row =
            getenv("DS4_METAL_DISABLE_TOKEN_EMBED_EXACT_VIEW") == NULL;
        if (exact_token_row) {
            const uint64_t row_bytes = ds4_gpu_q8_0_row_bytes(n_embd);
            const uint64_t token_rel = (uint64_t)token * row_bytes;
            if (token_rel > weight_bytes || row_bytes > weight_bytes - token_rel) {
                fprintf(stderr, "ds4: Metal Q8_0 embedding token row is outside the mapped table\n");
                return 0;
            }
            wbuf = ds4_gpu_wrap_model_exact_range(model_map,
                                                  model_size,
                                                  weight_offset + token_rel,
                                                  row_bytes,
                                                  &inner_offset);
            token_for_kernel = 0;
        } else {
            wbuf = ds4_gpu_wrap_model_range(model_map,
                                            model_size,
                                            weight_offset,
                                            weight_bytes,
                                            &inner_offset);
        }
        if (!wbuf) return 0;

        int owned = 0;
        id<MTLCommandBuffer> cb = ds4_gpu_command_buffer(&owned);
        if (!cb) return 0;

        const int32_t token_i32 = (int32_t)token_for_kernel;
        if (!ds4_gpu_encode_get_rows_q8_0(cb,
                                           wbuf,
                                           (NSUInteger)inner_offset,
                                           nil,
                                           0,
                                           &token_i32,
                                           outbuf,
                                           ds4_gpu_tensor_offset(out),
                                           n_vocab,
                                           1,
                                           n_embd)) {
            return 0;
        }

        if (!ds4_gpu_finish_command_buffer(cb, owned, "q8_0 embed token")) return 0;
    }

    return 1;
}

int ds4_gpu_embed_tokens_q8_0_tensor(
        ds4_gpu_tensor       *out,
        const ds4_gpu_tensor *tokens,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                weight_offset,
        uint32_t                n_vocab,
        uint32_t                n_tokens,
        uint32_t                n_embd) {
    if (!g_initialized && !ds4_gpu_init()) return 0;
    if (!out || !tokens || !model_map || n_vocab == 0 || n_tokens == 0 || n_embd == 0) {
        return 0;
    }

    @autoreleasepool {
        id<MTLBuffer> outbuf = ds4_gpu_tensor_buffer(out);
        id<MTLBuffer> tokbuf = ds4_gpu_tensor_buffer(tokens);
        const uint64_t out_bytes = (uint64_t)n_tokens * n_embd * sizeof(float);
        const uint64_t token_bytes = (uint64_t)n_tokens * sizeof(int32_t);
        if (!outbuf || !tokbuf ||
            ds4_gpu_tensor_bytes(out) < out_bytes ||
            ds4_gpu_tensor_bytes(tokens) < token_bytes) {
            fprintf(stderr, "ds4: Metal Q8_0 batched embedding received undersized buffers\n");
            return 0;
        }

        uint64_t weight_bytes = 0;
        if (!ds4_gpu_q8_0_table_bytes(n_vocab, n_embd, &weight_bytes) ||
            weight_offset > model_size ||
            weight_bytes > model_size - weight_offset) {
            fprintf(stderr, "ds4: Metal Q8_0 batched embedding range is outside the mapped model\n");
            return 0;
        }

        uint64_t inner_offset = 0;
        id<MTLBuffer> wbuf =
            ds4_gpu_wrap_model_range(model_map,
                                     model_size,
                                     weight_offset,
                                     weight_bytes,
                                     &inner_offset);
        if (!wbuf) return 0;

        int owned = 0;
        id<MTLCommandBuffer> cb = ds4_gpu_command_buffer(&owned);
        if (!cb) return 0;

        if (!ds4_gpu_encode_get_rows_q8_0(cb,
                                           wbuf,
                                           (NSUInteger)inner_offset,
                                           tokbuf,
                                           ds4_gpu_tensor_offset(tokens),
                                           NULL,
                                           outbuf,
                                           ds4_gpu_tensor_offset(out),
                                           n_vocab,
                                           n_tokens,
                                           n_embd)) {
            return 0;
        }

        if (!ds4_gpu_finish_command_buffer(cb, owned, "q8_0 embed tokens")) return 0;
    }

    return 1;
}

int ds4_gpu_set_model_map_range(const void *model_map, uint64_t model_size, uint64_t map_offset, uint64_t map_size, uint64_t max_tensor_bytes) {
    if (!g_initialized && !ds4_gpu_init()) return 0;
    if (!model_map || model_size == 0) return 0;
    if (map_offset > model_size || map_size == 0 || map_size > model_size - map_offset) return 0;
    max_tensor_bytes = ds4_gpu_effective_model_max_tensor_bytes(map_size, max_tensor_bytes);

    @autoreleasepool {
        if (g_model_map_ptr == model_map &&
            g_model_map_size == model_size &&
            g_model_mapped_offset == map_offset &&
            g_model_mapped_size == map_size &&
            g_model_mapped_max_tensor_bytes == max_tensor_bytes) {
            return 1;
        }

        for (uint32_t i = 0; i < g_model_view_count; i++) {
            if (g_model_views[i].model_map == model_map &&
                g_model_views[i].model_size == model_size &&
                map_offset >= g_model_views[i].model_offset &&
                map_offset + map_size <= g_model_views[i].model_offset + g_model_views[i].bytes) {
                return 1;
            }
        }

        ds4_gpu_model_residency_clear();
        if (!ds4_gpu_map_model_views(model_map, model_size, map_offset, map_size, max_tensor_bytes)) {
            ds4_gpu_model_residency_clear();
            return 0;
        }
        g_model_map_ptr = model_map;
        g_model_map_size = model_size;
        g_model_mapped_offset = map_offset;
        g_model_mapped_size = map_size;
        g_model_mapped_max_tensor_bytes = max_tensor_bytes;
        if (ds4_gpu_model_map_log_enabled()) {
            fprintf(stderr,
                    "ds4: Metal mapped mmaped model as %u overlapping shared buffers\n",
                    g_model_view_count);
        }
        return 1;
    }
}

int ds4_gpu_set_model_fd(int fd) {
    g_model_fd = fd;
    return 1;
}

int ds4_gpu_set_model_fd_for_map(int fd, const void *model_map) {
    (void)fd;
    (void)model_map;
    return 1;
}

static id<MTLBuffer> ds4_gpu_wrap_model_range(
        const void *model_map,
        uint64_t    model_size,
        uint64_t    offset,
        uint64_t    len,
        uint64_t   *inner_offset) {
    (void)model_map;
    if (model_size == 0 || offset > model_size || len > model_size - offset) {
        fprintf(stderr, "ds4: Metal model range is outside the mapped model\n");
        return nil;
    }

    const uint64_t end = offset + len;
    for (uint32_t i = 0; i < g_model_view_count; i++) {
        if (g_model_views[i].model_map != model_map ||
            g_model_views[i].model_size != model_size) {
            continue;
        }
        const uint64_t view_start = g_model_views[i].model_offset;
        const uint64_t view_end = view_start + g_model_views[i].bytes;
        if (offset >= view_start && end <= view_end) {
            *inner_offset = offset - view_start;
            return g_model_views[i].buffer;
        }
    }

    fprintf(stderr,
            "ds4: Metal model range %.2f..%.2f GiB is not covered by mapped model views\n",
            ds4_gpu_gib(offset),
            ds4_gpu_gib(end));
    return nil;
}

typedef enum {
    DS4_GPU_EXACT_VIEW_CACHED,
    DS4_GPU_EXACT_VIEW_TRANSIENT,
    DS4_GPU_EXACT_VIEW_OWNED,
} ds4_gpu_exact_view_lifetime;

static id<MTLBuffer> ds4_gpu_wrap_model_exact_range_impl(
        const void *model_map,
        uint64_t    model_size,
        uint64_t    offset,
        uint64_t    len,
        uint64_t   *inner_offset,
        ds4_gpu_exact_view_lifetime lifetime) {
    const bool cache_view = lifetime == DS4_GPU_EXACT_VIEW_CACHED;
    const bool transient_view = lifetime == DS4_GPU_EXACT_VIEW_TRANSIENT;
    if (!model_map || !g_device ||
        (cache_view && !g_model_buffer_cache) ||
        (transient_view && !g_transient_buffers) ||
        model_size == 0 || offset > model_size || len > model_size - offset) {
        fprintf(stderr, "ds4: Metal exact model range is outside the mapped model\n");
        return nil;
    }

    const uint64_t page = (uint64_t)getpagesize();
    const uint64_t page_offset = offset & ~(page - 1);
    const uint64_t leading = offset - page_offset;
    if (len > UINT64_MAX - leading ||
        leading + len > UINT64_MAX - (page - 1)) {
        fprintf(stderr, "ds4: Metal exact model range overflows page alignment\n");
        return nil;
    }
    uint64_t view_bytes = round_up_u64(leading + len, page);
    if (view_bytes > model_size - page_offset) view_bytes = model_size - page_offset;
    if (leading + len > view_bytes) {
        fprintf(stderr, "ds4: Metal exact model range alignment exceeds mapped model\n");
        return nil;
    }
    if (view_bytes > (uint64_t)[g_device maxBufferLength]) {
        fprintf(stderr,
                "ds4: Metal exact model range %.2f GiB exceeds maxBufferLength %.2f GiB\n",
                ds4_gpu_gib(view_bytes),
                ds4_gpu_gib((uint64_t)[g_device maxBufferLength]));
        return nil;
    }

    NSString *key = nil;
    id<MTLBuffer> buffer = nil;
    if (cache_view) {
        key = [NSString stringWithFormat:@"%p:%llu:%llu:%llu",
               model_map,
               (unsigned long long)model_size,
               (unsigned long long)page_offset,
               (unsigned long long)view_bytes];
        buffer = [g_model_buffer_cache objectForKey:key];
    }
    if (!buffer) {
        const uintptr_t base = (uintptr_t)model_map;
        buffer = [g_device newBufferWithBytesNoCopy:(void *)(base + page_offset)
                                             length:(NSUInteger)view_bytes
                                            options:ds4_gpu_model_resource_options()
                                        deallocator:nil];
        if (!buffer) {
            fprintf(stderr,
                    "ds4: Metal could not wrap exact mmaped model range at %.2f GiB, size %.2f MiB\n",
                    ds4_gpu_gib(page_offset),
                    ds4_gpu_mib(view_bytes));
            return nil;
        }
        if (cache_view) {
            buffer.label = @"ds4_model_exact_view";
        } else if (transient_view) {
            buffer.label = @"ds4_model_exact_transient_view";
        } else {
            buffer.label = @"ds4_model_exact_owned_view";
        }
        if (cache_view) {
            [g_model_buffer_cache setObject:buffer forKey:key];
            ds4_gpu_model_buffer_cache_note_insert(view_bytes);
        } else if (transient_view) {
            [g_transient_buffers addObject:buffer];
        }
    }

    if (inner_offset) *inner_offset = leading;
    return buffer;
}

static id<MTLBuffer> ds4_gpu_wrap_model_exact_range(
        const void *model_map,
        uint64_t    model_size,
        uint64_t    offset,
        uint64_t    len,
        uint64_t   *inner_offset) {
    return ds4_gpu_wrap_model_exact_range_impl(model_map,
                                               model_size,
                                               offset,
                                               len,
                                               inner_offset,
                                               DS4_GPU_EXACT_VIEW_CACHED);
}

/*
 * Large PRO caches otherwise create thousands of small shared Metal buffers.
 * Slabs keep the buffer object set small while locking pages only for slots
 * that actually hold a streamed expert.
 */

int ds4_gpu_indexer_topk_tensor(
        ds4_gpu_tensor       *selected,
        const ds4_gpu_tensor *scores,
        uint32_t                n_comp,
        uint32_t                n_tokens,
        uint32_t                top_k) {
    if (!g_initialized && !ds4_gpu_init()) return 0;
    if (!selected || !scores || n_comp == 0 || n_tokens == 0 || top_k == 0 || top_k > n_comp) return 0;

    @autoreleasepool {
        const uint64_t score_bytes = (uint64_t)n_comp * n_tokens * sizeof(float);
        const uint64_t selected_bytes = (uint64_t)top_k * n_tokens * sizeof(uint32_t);
        id<MTLBuffer> scorebuf = ds4_gpu_tensor_buffer(scores);
        id<MTLBuffer> selbuf = ds4_gpu_tensor_buffer(selected);
        if (!scorebuf || !selbuf ||
            ds4_gpu_tensor_bytes(scores) < score_bytes ||
            ds4_gpu_tensor_bytes(selected) < selected_bytes) {
            fprintf(stderr, "ds4: Metal graph indexer top-k received undersized buffers\n");
            return 0;
        }
        NSUInteger max_threads = g_argsort_f32_i32_desc_pipeline.maxTotalThreadsPerThreadgroup;
        if (max_threads == 0) max_threads = 256;
        int32_t nth = 1;
        while ((uint32_t)nth < n_comp && (uint64_t)2u * (uint64_t)nth <= (uint64_t)max_threads) {
            nth *= 2;
        }
        const int32_t npr = (int32_t)((n_comp + (uint32_t)nth - 1u) / (uint32_t)nth);
        const int32_t block_top_k = (int32_t)(top_k < (uint32_t)nth ? top_k : (uint32_t)nth);
        int32_t work_width = (int32_t)top_k;
        if (npr > 1) {
            const int32_t last_block = (int32_t)n_comp - (npr - 1) * nth;
            work_width = (npr - 1) * block_top_k + (last_block < block_top_k ? last_block : block_top_k);
        }
        const uint64_t scratch_row_bytes = (uint64_t)work_width * sizeof(uint32_t);
        const bool one_pass = npr <= 1;
        const uint64_t scratch_bytes = one_pass ? scratch_row_bytes * n_tokens :
            2u * scratch_row_bytes * n_tokens;
        if (!ds4_gpu_ensure_scratch_buffer(&g_indexer_topk_buffer,
                                             &g_indexer_topk_bytes,
                                             (NSUInteger)scratch_bytes,
                                             "ds4_indexer_topk")) {
            return 0;
        }

        ds4_gpu_kargs_argsort args = {
            .ne00 = (int32_t)n_comp,
            .ne01 = (int32_t)n_tokens,
            .ne02 = 1,
            .ne03 = 1,
            .nb00 = sizeof(float),
            .nb01 = (uint64_t)n_comp * sizeof(float),
            .nb02 = (uint64_t)n_comp * n_tokens * sizeof(float),
            .nb03 = (uint64_t)n_comp * n_tokens * sizeof(float),
            .ne0 = work_width,
            .ne1 = (int32_t)n_tokens,
            .ne2 = 1,
            .ne3 = 1,
            .top_k = block_top_k,
        };
        // kernel_argsort_f32_i32_desc stages the block's scores behind the
        // index array: nth int32 indices + nth float scores.
        const NSUInteger smem = (((NSUInteger)nth * (sizeof(int32_t) + sizeof(float))) + 15u) & ~(NSUInteger)15u;

        NSUInteger cur_off = 0;
        NSUInteger next_off = (NSUInteger)scratch_row_bytes * n_tokens;
        int owned = 0;
        id<MTLCommandBuffer> cb = ds4_gpu_command_buffer(&owned);
        if (!cb) return 0;

        id<MTLComputeCommandEncoder> enc = ds4_gpu_compute_encoder(cb);
        [enc setComputePipelineState:g_argsort_f32_i32_desc_pipeline];
        [enc setBytes:&args length:sizeof(args) atIndex:0];
        [enc setBuffer:scorebuf offset:ds4_gpu_tensor_offset(scores) atIndex:1];
        [enc setBuffer:one_pass ? selbuf : g_indexer_topk_buffer
              offset:one_pass ? ds4_gpu_tensor_offset(selected) : cur_off
             atIndex:2];
        [enc setThreadgroupMemoryLength:smem atIndex:0];
        [enc dispatchThreadgroups:MTLSizeMake((NSUInteger)npr * n_tokens, 1, 1)
             threadsPerThreadgroup:MTLSizeMake((NSUInteger)nth, 1, 1)];
        ds4_gpu_end_compute_encoder(cb, enc);

        int32_t len = block_top_k;
        while (len < work_width) {
            const int32_t nm = (work_width + 2 * len - 1) / (2 * len);
            const bool final_merge = nm == 1;
            NSUInteger merge_threads = g_argsort_merge_f32_i32_desc_pipeline.maxTotalThreadsPerThreadgroup;
            if (merge_threads == 0 || merge_threads > 512u) merge_threads = 512u;
            if (merge_threads > (NSUInteger)len) merge_threads = (NSUInteger)len;
            if (merge_threads == 0) merge_threads = 1;

            ds4_gpu_kargs_argsort_merge merge_args = {
                .ne00 = (int64_t)n_comp,
                .ne01 = (int64_t)n_tokens,
                .ne02 = 1,
                .ne03 = 1,
                .nb00 = sizeof(float),
                .nb01 = (uint64_t)n_comp * sizeof(float),
                .nb02 = (uint64_t)n_comp * n_tokens * sizeof(float),
                .nb03 = (uint64_t)n_comp * n_tokens * sizeof(float),
                .ne0 = work_width,
                .ne1 = (int32_t)n_tokens,
                .ne2 = 1,
                .ne3 = 1,
                .top_k = nm == 1 ? (int32_t)top_k : work_width,
                .len = len,
            };

            enc = ds4_gpu_compute_encoder(cb);
            [enc setComputePipelineState:g_argsort_merge_f32_i32_desc_pipeline];
            [enc setBytes:&merge_args length:sizeof(merge_args) atIndex:0];
            [enc setBuffer:scorebuf offset:ds4_gpu_tensor_offset(scores) atIndex:1];
            [enc setBuffer:g_indexer_topk_buffer offset:cur_off atIndex:2];
            [enc setBuffer:final_merge ? selbuf : g_indexer_topk_buffer
                  offset:final_merge ? ds4_gpu_tensor_offset(selected) : next_off
                 atIndex:3];
            [enc dispatchThreadgroups:MTLSizeMake((NSUInteger)nm * n_tokens, 1, 1)
                 threadsPerThreadgroup:MTLSizeMake(merge_threads, 1, 1)];
            ds4_gpu_end_compute_encoder(cb, enc);

            const NSUInteger tmp = cur_off;
            cur_off = next_off;
            next_off = tmp;
            len <<= 1;
        }

        if (!ds4_gpu_finish_command_buffer(cb, owned, "indexer top-k")) return 0;
    }

    return 1;
}

static int ds4_gpu_matmul_q8_0_legacy_tensor(
        ds4_gpu_tensor       *out,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                weight_offset,
        uint64_t                in_dim,
        uint64_t                out_dim,
        const ds4_gpu_tensor *x,
        uint64_t                n_tok,
        bool                    prefer_decode_mpp) {
    if (!g_initialized && !ds4_gpu_init()) return 0;
    if ((in_dim & 31u) != 0 ||
        in_dim > UINT32_MAX || out_dim > UINT32_MAX || n_tok > UINT32_MAX) {
        return 0;
    }

    @autoreleasepool {
        id<MTLBuffer> xbuf = ds4_gpu_tensor_buffer(x);
        id<MTLBuffer> outbuf = ds4_gpu_tensor_buffer(out);
        const uint64_t x_bytes = n_tok * in_dim * sizeof(float);
        const uint64_t out_bytes = n_tok * out_dim * sizeof(float);
        if (!xbuf || !outbuf ||
            ds4_gpu_tensor_bytes(x) < x_bytes ||
            ds4_gpu_tensor_bytes(out) < out_bytes) {
            fprintf(stderr, "ds4: Metal Q8_0 tensor matmul received undersized activation buffers\n");
            return 0;
        }

        const uint64_t blocks = in_dim / 32;
        const uint64_t row_bytes = blocks * 34;
        const uint64_t weight_bytes = out_dim * row_bytes;
        if (weight_offset > model_size || weight_bytes > model_size - weight_offset) {
            fprintf(stderr, "ds4: Metal Q8_0 tensor matmul range is outside the mapped model\n");
            return 0;
        }

        uint64_t inner_offset = 0;
        id<MTLBuffer> wbuf = ds4_gpu_wrap_model_range(model_map,
                                                      model_size,
                                                      weight_offset,
                                                      weight_bytes,
                                                      &inner_offset);
        if (!wbuf) {
            return 0;
        }

        int owned = 0;
        id<MTLCommandBuffer> cb = ds4_gpu_command_buffer(&owned);
        if (!cb) return 0;

        if (n_tok == 1) {
            if (ds4_gpu_mpp_available() &&
                prefer_decode_mpp &&
                (in_dim % 64u) == 0) {
                const char *nax_fn = "kernel_mul_mm_q8_0_f32_nax_direct_rhs";
                id<MTLComputePipelineState> mpp_pipeline =
                    ds4_gpu_get_mul_mm_pipeline(nax_fn, false, false);
                if (mpp_pipeline) {
                    ds4_gpu_mul_mm_args args =
                        ds4_gpu_make_mm_args(in_dim, out_dim, n_tok, row_bytes);

                    id<MTLComputeCommandEncoder> enc = ds4_gpu_compute_encoder(cb);
                    [enc setComputePipelineState:mpp_pipeline];
                    [enc setBytes:&args length:sizeof(args) atIndex:0];
                    [enc setBuffer:wbuf offset:(NSUInteger)inner_offset atIndex:1];
                    [enc setBuffer:xbuf offset:ds4_gpu_tensor_offset(x) atIndex:2];
                    [enc setBuffer:outbuf offset:ds4_gpu_tensor_offset(out) atIndex:3];
                    [enc setThreadgroupMemoryLength:64u * 32u * sizeof(uint16_t) atIndex:0];
                    [enc dispatchThreadgroups:MTLSizeMake(1u,
                                                          ((NSUInteger)out_dim + 63u) / 64u,
                                                          1u)
                         threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
                    ds4_gpu_end_compute_encoder(cb, enc);

                    if (!ds4_gpu_finish_command_buffer(cb, owned, "Q8_0 decode MPP matmul")) {
                        return 0;
                    }
                    return 1;
                }
                ds4_gpu_warn_mpp_fallback();
            }

            ds4_gpu_q8_0_matvec_args mv_args = ds4_gpu_make_q8_0_mv_args(in_dim, out_dim);
            ds4_gpu_mv_dispatch mv_dispatch =
                ds4_gpu_make_q8_0_single_mv_dispatch();
            if (out_dim > 65536u) mv_dispatch.nsg = 8;
            mv_args.nr0 = mv_dispatch.nr0;
            id<MTLComputePipelineState> pipeline =
                ds4_gpu_get_mul_mv_pipeline(mv_dispatch.function_name, mv_dispatch.nsg);
            if (!pipeline) return 0;

            id<MTLComputeCommandEncoder> enc = ds4_gpu_compute_encoder(cb);
            [enc setComputePipelineState:pipeline];
            [enc setBytes:&mv_args length:sizeof(mv_args) atIndex:0];
            [enc setBuffer:wbuf offset:(NSUInteger)inner_offset atIndex:1];
            [enc setBuffer:xbuf offset:ds4_gpu_tensor_offset(x) atIndex:2];
            [enc setBuffer:outbuf offset:ds4_gpu_tensor_offset(out) atIndex:3];
            [enc setThreadgroupMemoryLength:mv_dispatch.smem atIndex:0];
            [enc dispatchThreadgroups:MTLSizeMake(((NSUInteger)out_dim + (NSUInteger)mv_dispatch.nr0 - 1u) / (NSUInteger)mv_dispatch.nr0,
                                                  1,
                                                  1)
                 threadsPerThreadgroup:MTLSizeMake(32, (NSUInteger)mv_dispatch.nsg, 1)];
            ds4_gpu_end_compute_encoder(cb, enc);

            if (!ds4_gpu_finish_command_buffer(cb, owned, "Q8_0 tensor matvec")) {
                return 0;
            }
            return 1;
        }

        const uint64_t mv_ext_max_tokens =
            ds4_gpu_env_u64("DS4_METAL_Q8_MV_EXT_MAX_TOKENS", 16u, 2u, 128u);
        if (n_tok <= mv_ext_max_tokens && (in_dim % 128u) == 0) {
            const int16_t nsg = 2;
            const int16_t nxpsg = ds4_gpu_mv_ext_nxpsg(in_dim, n_tok);
            const int16_t r1ptg = ds4_gpu_mv_ext_r1ptg(n_tok);
            const char *fn_name = ds4_gpu_mv_ext_name(1, r1ptg);
            id<MTLComputePipelineState> pipeline =
                fn_name ? ds4_gpu_get_mul_mv_ext_pipeline(fn_name, nsg, nxpsg) : nil;
            if (!pipeline) return 0;

            const int16_t nypsg = 32 / nxpsg;
            const uint64_t r0ptg = (uint64_t)nypsg * (uint64_t)nsg;
            ds4_gpu_mul_mv_ext_args args =
                ds4_gpu_make_mv_ext_args(in_dim, out_dim, n_tok, 34, row_bytes);

            id<MTLComputeCommandEncoder> enc = ds4_gpu_compute_encoder(cb);
            [enc setComputePipelineState:pipeline];
            [enc setBytes:&args length:sizeof(args) atIndex:0];
            [enc setBuffer:wbuf offset:(NSUInteger)inner_offset atIndex:1];
            [enc setBuffer:xbuf offset:ds4_gpu_tensor_offset(x) atIndex:2];
            [enc setBuffer:outbuf offset:ds4_gpu_tensor_offset(out) atIndex:3];
            [enc dispatchThreadgroups:MTLSizeMake(((NSUInteger)out_dim + (NSUInteger)r0ptg - 1u) / (NSUInteger)r0ptg,
                                                  ((NSUInteger)n_tok + (NSUInteger)r1ptg - 1u) / (NSUInteger)r1ptg,
                                                  1)
                 threadsPerThreadgroup:MTLSizeMake(32, (NSUInteger)nsg, 1)];
            ds4_gpu_end_compute_encoder(cb, enc);

            if (!ds4_gpu_finish_command_buffer(cb, owned, "Q8_0 tensor mul_mv_ext")) {
                return 0;
            }
            return 1;
        }

        /*
         * Dense Q8_0 prefill is the cleanest DS4 TensorOps shape: M/N/K are
         * aligned and the RHS activation matrix is already dense.  The retained
         * kernel dequantizes each 64x32 weight tile to half in threadgroup
         * memory, then uses direct-RHS MPP for the activation tile.  This avoids
         * staging RHS into threadgroup memory and was the direct replacement for
         * the slower generic MPP prototype.
         */
        /*
         * An unaligned token count used to send the entire projection through
         * the generic kernel.  Run its aligned prefix through TensorOps and
         * leave only the final partial tile to the boundary-safe kernel.
         * Tiny prompts do not amortize the second dispatch, while --quality
         * deliberately retains the single-kernel arithmetic schedule.
         */
        const bool split_nax_prefix =
            !g_quality_mode && n_tok >= 192u && (n_tok % 32u) != 0u;
        const uint64_t nax_rows =
            (n_tok % 32u) == 0u ? n_tok :
            (split_nax_prefix ? n_tok - (n_tok % 32u) : 0u);
        uint64_t generic_row0 = 0u;
        uint64_t generic_rows = n_tok;
        if (ds4_gpu_mpp_available() &&
            nax_rows >= 32u &&
            (in_dim % 64u) == 0 &&
            (out_dim % 64u) == 0) {
            uint64_t nax_tile_n = 32u;
            if ((nax_rows % 128u) == 0) {
                nax_tile_n = 128u;
            } else if ((nax_rows % 64u) == 0) {
                nax_tile_n = 64u;
            }
            const char *nax_fn = nax_tile_n == 128u
                ? "kernel_mul_mm_q8_0_f32_nax_direct_rhs_n128"
                : (nax_tile_n == 64u
                    ? "kernel_mul_mm_q8_0_f32_nax_direct_rhs_n64"
                    : "kernel_mul_mm_q8_0_f32_nax_direct_rhs");
            id<MTLComputePipelineState> pipeline =
                ds4_gpu_get_mul_mm_pipeline(nax_fn, false, false);
            if (pipeline) {
                ds4_gpu_mul_mm_args args =
                    ds4_gpu_make_mm_args(in_dim, out_dim, nax_rows, row_bytes);

                id<MTLComputeCommandEncoder> enc = ds4_gpu_compute_encoder(cb);
                [enc setComputePipelineState:pipeline];
                [enc setBytes:&args length:sizeof(args) atIndex:0];
                [enc setBuffer:wbuf offset:(NSUInteger)inner_offset atIndex:1];
                [enc setBuffer:xbuf offset:ds4_gpu_tensor_offset(x) atIndex:2];
                [enc setBuffer:outbuf offset:ds4_gpu_tensor_offset(out) atIndex:3];
                [enc setThreadgroupMemoryLength:2u * 64u * 32u * sizeof(uint16_t) atIndex:0];
                [enc dispatchThreadgroups:MTLSizeMake((NSUInteger)(nax_rows / nax_tile_n),
                                                      (NSUInteger)out_dim / 64u,
                                                      1)
                     threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
                ds4_gpu_end_compute_encoder(cb, enc);

                if (nax_rows == n_tok) {
                    if (!ds4_gpu_finish_command_buffer(cb, owned, "Q8_0 NAX tensor matmul")) {
                        return 0;
                    }
                    return 1;
                }
                generic_row0 = nax_rows;
                generic_rows = n_tok - nax_rows;
            }
            if (!pipeline) ds4_gpu_warn_mpp_fallback();
        }

        /*
         * Paired 16-bit Q8 loads are bit-identical to the legacy byte loads
         * and reduce Qwen3.6 projection time on pre-M5 Apple GPUs.  Keep the
         * old kernel and the rejected 64-token occupancy experiment available
         * for reproducible diagnostics.
         */
        const char *q8_prefill_variant = getenv("DS4_METAL_Q8_PREFILL_VARIANT");
        const bool q8_pairs64 =
            q8_prefill_variant && !strcmp(q8_prefill_variant, "pairs64") &&
            generic_rows >= 64u;
        const bool q8_legacy =
            q8_prefill_variant && !strcmp(q8_prefill_variant, "legacy");
        const uint64_t generic_tile_n = q8_pairs64 ? 64u : 32u;
        const char *generic_fn = q8_pairs64
            ? "kernel_mul_mm_q8_0_f32_pairs_n64"
            : (q8_legacy
                ? "kernel_mul_mm_q8_0_f32"
                : "kernel_mul_mm_q8_0_f32_pairs");

        const bool bc_inp = (in_dim % 32u) != 0;
        const bool bc_out =
            (out_dim % 64u) != 0 || (generic_rows % generic_tile_n) != 0;
        id<MTLComputePipelineState> pipeline =
            ds4_gpu_get_mul_mm_pipeline(generic_fn, bc_inp, bc_out);
        if (!pipeline) return 0;

        ds4_gpu_mul_mm_args args =
            ds4_gpu_make_mm_args(in_dim, out_dim, generic_rows, row_bytes);

        id<MTLComputeCommandEncoder> enc = ds4_gpu_compute_encoder(cb);
        [enc setComputePipelineState:pipeline];
        [enc setBytes:&args length:sizeof(args) atIndex:0];
        [enc setBuffer:wbuf offset:(NSUInteger)inner_offset atIndex:1];
        [enc setBuffer:xbuf
                offset:ds4_gpu_tensor_offset(x) +
                       (NSUInteger)(generic_row0 * in_dim * sizeof(float))
               atIndex:2];
        [enc setBuffer:outbuf
                offset:ds4_gpu_tensor_offset(out) +
                       (NSUInteger)(generic_row0 * out_dim * sizeof(float))
               atIndex:3];
        const NSUInteger stage_bytes =
            4096u + (NSUInteger)generic_tile_n * 32u * sizeof(uint16_t);
        const NSUInteger boundary_bytes =
            (NSUInteger)generic_tile_n * 64u * sizeof(float);
        [enc setThreadgroupMemoryLength:(bc_out && boundary_bytes > stage_bytes)
                                               ? boundary_bytes
                                               : stage_bytes
                                   atIndex:0];
        [enc dispatchThreadgroups:MTLSizeMake(((NSUInteger)generic_rows +
                                               (NSUInteger)generic_tile_n - 1u) /
                                                  (NSUInteger)generic_tile_n,
                                              ((NSUInteger)out_dim + 63u) / 64u,
                                              1)
             threadsPerThreadgroup:MTLSizeMake((NSUInteger)generic_tile_n * 4u,
                                               1,
                                               1)];
        ds4_gpu_end_compute_encoder(cb, enc);

        if (!ds4_gpu_finish_command_buffer(cb, owned, "Q8_0 tensor matmul")) {
            return 0;
        }
    }

    return 1;
}

int ds4_gpu_matmul_q8_0_tensor(
        ds4_gpu_tensor       *out,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                weight_offset,
        uint64_t                in_dim,
        uint64_t                out_dim,
        const ds4_gpu_tensor *x,
        uint64_t                n_tok) {
    if (!g_initialized && !ds4_gpu_init()) return 0;
    if ((in_dim & 31u) != 0 ||
        in_dim > UINT32_MAX || out_dim > UINT32_MAX || n_tok > UINT32_MAX) {
        return 0;
    }

    const int profile_requested =
        n_tok > 8u && ds4_gpu_env_bool("DS4_METAL_Q8_PREFILL_PROFILE") > 0;
    int profile_prefill = 0;
    int split_batch_for_profile = 0;
    const char *profile_label = NULL;
    char profile_label_buf[128];
    char profile_fallback[128];
    if (profile_requested) {
        snprintf(profile_fallback, sizeof(profile_fallback),
                 "q8 weight_off=%llu in=%llu out=%llu tok=%llu",
                 (unsigned long long)weight_offset,
                 (unsigned long long)in_dim,
                 (unsigned long long)out_dim,
                 (unsigned long long)n_tok);
        snprintf(profile_label_buf, sizeof(profile_label_buf), "%s", profile_fallback);
        profile_label = profile_label_buf;
        const char *profile_filter = getenv("DS4_METAL_Q8_PREFILL_PROFILE_FILTER");
        profile_prefill =
            profile_requested &&
            (!profile_filter || !profile_filter[0] ||
             strstr(profile_label, profile_filter) != NULL);
    }
    if (profile_prefill) {
        if (g_batch_cb) {
            if (ds4_gpu_end_commands() == 0 || ds4_gpu_begin_commands() == 0) {
                return 0;
            }
            split_batch_for_profile = 1;
        }
    }

    const double profile_t0 = profile_prefill ? ds4_gpu_now_ms() : 0.0;
    int ok = ds4_gpu_matmul_q8_0_legacy_tensor(out, model_map, model_size,
                                               weight_offset, in_dim, out_dim,
                                               x, n_tok, false);
    if (profile_prefill) {
        if (split_batch_for_profile && ds4_gpu_end_commands() == 0) {
            ok = 0;
        }
        const double elapsed_ms = ds4_gpu_now_ms() - profile_t0;
        fprintf(stderr,
                "ds4: Metal Q8_0 prefill profile %s in=%llu out=%llu tok=%llu %.3f ms\n",
                profile_label ? profile_label : profile_fallback,
                (unsigned long long)in_dim,
                (unsigned long long)out_dim,
                (unsigned long long)n_tok,
                elapsed_ms);
        if (split_batch_for_profile && ds4_gpu_begin_commands() == 0) {
            ok = 0;
        }
    }
    return ok;
}

int ds4_gpu_matmul_q8_0_f16_rhs_tensor(
        ds4_gpu_tensor       *out,
        const void           *model_map,
        uint64_t              model_size,
        uint64_t              weight_offset,
        uint64_t              in_dim,
        uint64_t              out_dim,
        const ds4_gpu_tensor *x_f16,
        uint64_t              n_tok) {
    if (!g_initialized && !ds4_gpu_init()) return 0;
    if (!out || !model_map || !x_f16 || n_tok == 0u ||
        (n_tok % 32u) != 0u || (in_dim % 32u) != 0u ||
        (out_dim % 64u) != 0u || in_dim > UINT32_MAX ||
        out_dim > UINT32_MAX || n_tok > UINT32_MAX) {
        return 0;
    }
    @autoreleasepool {
        id<MTLBuffer> xbuf = ds4_gpu_tensor_buffer(x_f16);
        id<MTLBuffer> outbuf = ds4_gpu_tensor_buffer(out);
        const uint64_t x_bytes = n_tok*in_dim*sizeof(uint16_t);
        const uint64_t out_bytes = n_tok*out_dim*sizeof(float);
        if (!xbuf || !outbuf || ds4_gpu_tensor_bytes(x_f16) < x_bytes ||
            ds4_gpu_tensor_bytes(out) < out_bytes) return 0;

        const uint64_t row_bytes = (in_dim/32u)*34u;
        const uint64_t weight_bytes = out_dim*row_bytes;
        if (weight_offset > model_size ||
            weight_bytes > model_size - weight_offset) return 0;
        uint64_t inner_offset = 0u;
        id<MTLBuffer> wbuf = ds4_gpu_wrap_model_range(
            model_map, model_size, weight_offset, weight_bytes, &inner_offset);
        id<MTLComputePipelineState> pipeline = ds4_gpu_get_mul_mm_pipeline(
            "kernel_mul_mm_q8_0_f16_pairs", false, false);
        if (!wbuf || !pipeline) return 0;

        int owned = 0;
        id<MTLCommandBuffer> cb = ds4_gpu_command_buffer(&owned);
        if (!cb) return 0;
        ds4_gpu_mul_mm_args args =
            ds4_gpu_make_mm_args(in_dim, out_dim, n_tok, row_bytes);
        args.nb10 = sizeof(uint16_t);
        args.nb11 = in_dim*sizeof(uint16_t);
        args.nb12 = in_dim*n_tok*sizeof(uint16_t);
        args.nb13 = args.nb12;
        id<MTLComputeCommandEncoder> enc = ds4_gpu_compute_encoder(cb);
        [enc setComputePipelineState:pipeline];
        [enc setBytes:&args length:sizeof(args) atIndex:0];
        [enc setBuffer:wbuf offset:(NSUInteger)inner_offset atIndex:1];
        [enc setBuffer:xbuf offset:ds4_gpu_tensor_offset(x_f16) atIndex:2];
        [enc setBuffer:outbuf offset:ds4_gpu_tensor_offset(out) atIndex:3];
        [enc setThreadgroupMemoryLength:4096u + 32u*32u*sizeof(uint16_t)
                                atIndex:0];
        [enc dispatchThreadgroups:MTLSizeMake((NSUInteger)(n_tok/32u),
                                              (NSUInteger)(out_dim/64u), 1u)
             threadsPerThreadgroup:MTLSizeMake(128u, 1u, 1u)];
        ds4_gpu_end_compute_encoder(cb, enc);
        return ds4_gpu_finish_command_buffer(
            cb, owned, "Q8_0 F16-RHS tensor matmul");
    }
}

int ds4_gpu_matmul_q8_0_pair_tensor(
        ds4_gpu_tensor       *out0,
        ds4_gpu_tensor       *out1,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                weight0_offset,
        uint64_t                weight1_offset,
        uint64_t                in_dim,
        uint64_t                out0_dim,
        uint64_t                out1_dim,
        const ds4_gpu_tensor *x,
        uint64_t                n_tok) {
    if (!g_initialized && !ds4_gpu_init()) return 0;
    if (!out0 || !out1 || !model_map || !x || n_tok != 1 ||
        out0_dim == 0 || out1_dim == 0 || (in_dim & 31u) != 0 ||
        in_dim > UINT32_MAX || out0_dim > UINT32_MAX || out1_dim > UINT32_MAX) {
        return 0;
    }

    @autoreleasepool {
        id<MTLBuffer> xbuf = ds4_gpu_tensor_buffer(x);
        id<MTLBuffer> out0buf = ds4_gpu_tensor_buffer(out0);
        id<MTLBuffer> out1buf = ds4_gpu_tensor_buffer(out1);
        const uint64_t x_bytes = in_dim * sizeof(float);
        const uint64_t out0_bytes = out0_dim * sizeof(float);
        const uint64_t out1_bytes = out1_dim * sizeof(float);
        if (!xbuf || !out0buf || !out1buf ||
            ds4_gpu_tensor_bytes(x) < x_bytes ||
            ds4_gpu_tensor_bytes(out0) < out0_bytes ||
            ds4_gpu_tensor_bytes(out1) < out1_bytes) {
            fprintf(stderr, "ds4: Metal paired Q8_0 matvec received undersized activation buffers\n");
            return 0;
        }

        const uint64_t row_bytes = (in_dim / 32u) * 34u;
        const uint64_t weight0_bytes = out0_dim * row_bytes;
        const uint64_t weight1_bytes = out1_dim * row_bytes;
        if (weight0_offset > model_size || weight0_bytes > model_size - weight0_offset ||
            weight1_offset > model_size || weight1_bytes > model_size - weight1_offset) {
            fprintf(stderr, "ds4: Metal paired Q8_0 matvec range is outside the mapped model\n");
            return 0;
        }

        uint64_t inner0 = 0;
        uint64_t inner1 = 0;
        id<MTLBuffer> weight0buf =
            ds4_gpu_wrap_model_range(model_map, model_size,
                                     weight0_offset, weight0_bytes, &inner0);
        id<MTLBuffer> weight1buf =
            ds4_gpu_wrap_model_range(model_map, model_size,
                                     weight1_offset, weight1_bytes, &inner1);
        if (!weight0buf || !weight1buf) return 0;

        ds4_gpu_mv_dispatch dispatch0 = ds4_gpu_make_q8_0_mv_dispatch();
        ds4_gpu_mv_dispatch dispatch1 = ds4_gpu_make_q8_0_mv_dispatch();
        if (out0_dim > 65536u) dispatch0.nsg = 8;
        if (out1_dim > 65536u) dispatch1.nsg = 8;
        /* A common threadgroup shape is required to retain each standalone
         * reduction tree. Mixed 4/8-simdgroup extents use the existing fallback. */
        if (dispatch0.nsg != dispatch1.nsg) return 0;

        ds4_gpu_q8_0_matvec_args args0 = ds4_gpu_make_q8_0_mv_args(in_dim, out0_dim);
        ds4_gpu_q8_0_matvec_args args1 = ds4_gpu_make_q8_0_mv_args(in_dim, out1_dim);
        args0.nr0 = dispatch0.nr0;
        args1.nr0 = dispatch1.nr0;
        id<MTLComputePipelineState> pipeline =
            ds4_gpu_get_mul_mv_pipeline("kernel_mul_mv_q8_0_f32_pair", dispatch0.nsg);
        if (!pipeline) return 0;

        int owned = 0;
        id<MTLCommandBuffer> cb = ds4_gpu_command_buffer(&owned);
        if (!cb) return 0;

        id<MTLComputeCommandEncoder> enc = ds4_gpu_compute_encoder(cb);
        [enc setComputePipelineState:pipeline];
        [enc setBytes:&args0 length:sizeof(args0) atIndex:0];
        [enc setBytes:&args1 length:sizeof(args1) atIndex:1];
        [enc setBuffer:weight0buf offset:(NSUInteger)inner0 atIndex:2];
        [enc setBuffer:weight1buf offset:(NSUInteger)inner1 atIndex:3];
        [enc setBuffer:xbuf offset:ds4_gpu_tensor_offset(x) atIndex:4];
        [enc setBuffer:out0buf offset:ds4_gpu_tensor_offset(out0) atIndex:5];
        [enc setBuffer:out1buf offset:ds4_gpu_tensor_offset(out1) atIndex:6];
        [enc setThreadgroupMemoryLength:2u * dispatch0.smem atIndex:0];
        const uint64_t max_out_dim = out0_dim > out1_dim ? out0_dim : out1_dim;
        [enc dispatchThreadgroups:MTLSizeMake(((NSUInteger)max_out_dim +
                                               (NSUInteger)dispatch0.nr0 - 1u) /
                                              (NSUInteger)dispatch0.nr0,
                                              1,
                                              1)
             threadsPerThreadgroup:MTLSizeMake(32, (NSUInteger)dispatch0.nsg, 1)];
        ds4_gpu_end_compute_encoder(cb, enc);

        if (!ds4_gpu_finish_command_buffer(cb, owned, "paired Q8_0 matvec")) return 0;
    }

    return 1;
}

static int ds4_gpu_shared_gate_up_swiglu_q8_0_impl(
        ds4_gpu_tensor       *gate,
        ds4_gpu_tensor       *up,
        ds4_gpu_tensor       *mid,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                gate_offset,
        uint64_t                up_offset,
        uint64_t                in_dim,
        uint64_t                out_dim,
        const ds4_gpu_tensor *x,
        float                   clamp,
        int                     store_gate_up) {
    if (!g_initialized && !ds4_gpu_init()) return 0;
    if (!mid || !x || !model_map ||
        (store_gate_up && (!gate || !up)) ||
        (in_dim & 31u) != 0 ||
        in_dim > UINT32_MAX || out_dim > UINT32_MAX ||
        !isfinite(clamp) || clamp < 0.0f) {
        return 0;
    }

    @autoreleasepool {
        id<MTLBuffer> xbuf = ds4_gpu_tensor_buffer(x);
        id<MTLBuffer> midbuf = ds4_gpu_tensor_buffer(mid);
        id<MTLBuffer> gatebuf = store_gate_up ?
            ds4_gpu_tensor_buffer(gate) : midbuf;
        id<MTLBuffer> upbuf = store_gate_up ?
            ds4_gpu_tensor_buffer(up) : midbuf;
        const uint64_t x_bytes = in_dim * sizeof(float);
        const uint64_t out_bytes = out_dim * sizeof(float);
        if (!xbuf || !gatebuf || !upbuf || !midbuf ||
            ds4_gpu_tensor_bytes(x) < x_bytes ||
            (store_gate_up && ds4_gpu_tensor_bytes(gate) < out_bytes) ||
            (store_gate_up && ds4_gpu_tensor_bytes(up) < out_bytes) ||
            ds4_gpu_tensor_bytes(mid) < out_bytes) {
            fprintf(stderr, "ds4: Metal shared expert fused gate/up received undersized activation buffers\n");
            return 0;
        }

        const uint64_t blocks = in_dim / 32;
        const uint64_t row_bytes = blocks * 34;
        const uint64_t weight_bytes = out_dim * row_bytes;
        if (gate_offset > model_size || weight_bytes > model_size - gate_offset ||
            up_offset > model_size || weight_bytes > model_size - up_offset) {
            fprintf(stderr, "ds4: Metal shared expert fused gate/up range is outside the mapped model\n");
            return 0;
        }

        uint64_t gate_inner = 0;
        uint64_t up_inner = 0;
        id<MTLBuffer> gate_wbuf = ds4_gpu_wrap_model_range(model_map,
                                                           model_size,
                                                           gate_offset,
                                                           weight_bytes,
                                                           &gate_inner);
        id<MTLBuffer> up_wbuf = ds4_gpu_wrap_model_range(model_map,
                                                         model_size,
                                                         up_offset,
                                                         weight_bytes,
                                                         &up_inner);
        if (!gate_wbuf || !up_wbuf) return 0;

        ds4_gpu_q8_0_matvec_args args = ds4_gpu_make_q8_0_mv_args(in_dim, out_dim);
        ds4_gpu_mv_dispatch mv_dispatch = ds4_gpu_make_q8_0_mv_dispatch();
        args.nr0 = mv_dispatch.nr0;
        const char *fn_name = store_gate_up ?
            "kernel_dsv4_shared_gate_up_swiglu_q8_0" :
            "kernel_dsv4_shared_mid_swiglu_q8_0";
        id<MTLComputePipelineState> pipeline =
            ds4_gpu_get_mul_mv_pipeline(fn_name, mv_dispatch.nsg);
        if (!pipeline) return 0;

        int owned = 0;
        id<MTLCommandBuffer> cb = ds4_gpu_command_buffer(&owned);
        if (!cb) return 0;

        id<MTLComputeCommandEncoder> enc = ds4_gpu_compute_encoder(cb);
        [enc setComputePipelineState:pipeline];
        [enc setBytes:&args length:sizeof(args) atIndex:0];
        [enc setBuffer:gate_wbuf offset:(NSUInteger)gate_inner atIndex:1];
        [enc setBuffer:up_wbuf offset:(NSUInteger)up_inner atIndex:2];
        [enc setBuffer:xbuf offset:ds4_gpu_tensor_offset(x) atIndex:3];
        [enc setBuffer:gatebuf offset:(store_gate_up ?
                                       ds4_gpu_tensor_offset(gate) :
                                       ds4_gpu_tensor_offset(mid)) atIndex:4];
        [enc setBuffer:upbuf offset:(store_gate_up ?
                                     ds4_gpu_tensor_offset(up) :
                                     ds4_gpu_tensor_offset(mid)) atIndex:5];
        [enc setBuffer:midbuf offset:ds4_gpu_tensor_offset(mid) atIndex:6];
        [enc setBytes:&clamp length:sizeof(clamp) atIndex:7];
        [enc setThreadgroupMemoryLength:2u * mv_dispatch.smem atIndex:0];
        [enc dispatchThreadgroups:MTLSizeMake(((NSUInteger)out_dim + (NSUInteger)mv_dispatch.nr0 - 1u) /
                                                  (NSUInteger)mv_dispatch.nr0,
                                              1,
                                              1)
             threadsPerThreadgroup:MTLSizeMake(32, (NSUInteger)mv_dispatch.nsg, 1)];
        ds4_gpu_end_compute_encoder(cb, enc);

        if (!ds4_gpu_finish_command_buffer(cb,
                                           owned,
                                           store_gate_up ?
                                           "shared expert fused gate/up" :
                                           "shared expert fused mid")) {
            return 0;
        }
    }

    return 1;
}

/* Decode-only fusion of the router logits matvec with the shared-expert
 * gate/up SwiGLU: one dispatch instead of two on the same normalized FFN
 * input.  Bit-exact by construction (see metal/dense.metal).  Returns 1 on
 * success, 0 when the shape is unsupported (caller falls back), -1 on
 * error. */

int ds4_gpu_shared_gate_up_swiglu_q8_0_tensor(
        ds4_gpu_tensor       *gate,
        ds4_gpu_tensor       *up,
        ds4_gpu_tensor       *mid,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                gate_offset,
        uint64_t                up_offset,
        uint64_t                in_dim,
        uint64_t                out_dim,
        const ds4_gpu_tensor *x,
        float                   clamp) {
    return ds4_gpu_shared_gate_up_swiglu_q8_0_impl(gate,
                                                   up,
                                                   mid,
                                                   model_map,
                                                   model_size,
                                                   gate_offset,
                                                   up_offset,
                                                   in_dim,
                                                   out_dim,
                                                   x,
                                                   clamp,
                                                   1);
}

/* Quad variant of the paired compressor projection: the attention compressor
 * and indexer compressor pairs share the input activation and F16 matvec
 * shape, so one dispatch covers all four matrices.  Bit-exact by
 * construction (see metal/dense.metal).  Returns 1 when the fused dispatch
 * ran, 0 when the caller should use the separate paths, -1 on error. */

/* Decode-only emit-path fusion: finalize the freshly pooled attention and
 * indexer compressor rows (norm + rope + fp8/commit + qat) in one dispatch
 * instead of seven.  Bit-exact vs the separate dispatches; see the kernel
 * comment.  Returns 1 when the fused dispatch ran, 0 to fall back. */

/* Decode-only fusion: q_a/kv Q8 pair projection + F16 quad compressor
 * projection/store in one dispatch (both read the same normalized input).
 * Bit-exact vs the separate dispatches; see the kernel comment.  Returns
 * 1 when the fused dispatch ran, 0 when the caller must use the separate
 * paths, -1 on error. */

int ds4_gpu_rms_norm_weight_tensor(
        ds4_gpu_tensor       *out,
        const ds4_gpu_tensor *x,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                weight_offset,
        uint32_t                n,
        float                   eps) {
    return ds4_gpu_rms_norm_weight_rows_tensor(out, x, model_map, model_size, weight_offset, n, 1, eps);
}

int ds4_gpu_rms_norm_weight_rows_tensor(
        ds4_gpu_tensor       *out,
        const ds4_gpu_tensor *x,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                weight_offset,
        uint32_t                n,
        uint32_t                rows,
        float                   eps) {
    if (!g_initialized && !ds4_gpu_init()) return 0;
    if (n == 0 || rows == 0 || (n & 3u) != 0) return 0;

    @autoreleasepool {
        id<MTLBuffer> xbuf = ds4_gpu_tensor_buffer(x);
        id<MTLBuffer> outbuf = ds4_gpu_tensor_buffer(out);
        const uint64_t row_bytes = (uint64_t)n * sizeof(float);
        const uint64_t bytes = row_bytes * rows;
        if (!xbuf || !outbuf ||
            ds4_gpu_tensor_bytes(x) < bytes ||
            ds4_gpu_tensor_bytes(out) < bytes) {
            fprintf(stderr, "ds4: Metal weighted RMS norm received undersized activation buffers\n");
            return 0;
        }
        if (weight_offset > model_size || row_bytes > model_size - weight_offset) {
            fprintf(stderr, "ds4: Metal weighted RMS norm range is outside the mapped model\n");
            return 0;
        }

        uint64_t inner_offset = 0;
        id<MTLBuffer> wbuf = ds4_gpu_wrap_model_range(model_map,
                                                       model_size,
                                                       weight_offset,
                                                       row_bytes,
                                                       &inner_offset);
        if (!wbuf) return 0;

        ds4_gpu_rms_norm_args args = ds4_gpu_make_rms_norm_args(n, rows, eps);
        int owned = 0;
        id<MTLCommandBuffer> cb = ds4_gpu_command_buffer(&owned);
        if (!cb) return 0;

        id<MTLComputeCommandEncoder> enc = ds4_gpu_compute_encoder(cb);
        [enc setComputePipelineState:g_rms_norm_pipeline];
        [enc setBytes:&args length:sizeof(args) atIndex:0];
        [enc setBuffer:xbuf offset:ds4_gpu_tensor_offset(x) atIndex:1];
        [enc setBuffer:wbuf offset:(NSUInteger)inner_offset atIndex:2];
        [enc setBuffer:xbuf offset:ds4_gpu_tensor_offset(x) atIndex:3];
        [enc setBuffer:outbuf offset:ds4_gpu_tensor_offset(out) atIndex:4];
        [enc setThreadgroupMemoryLength:32u * sizeof(float) atIndex:0];
        [enc dispatchThreadgroups:MTLSizeMake(rows, 1, 1)
             threadsPerThreadgroup:MTLSizeMake(ds4_gpu_rms_norm_threads(n), 1, 1)];
        ds4_gpu_end_compute_encoder(cb, enc);

        if (!ds4_gpu_finish_command_buffer(cb, owned, "weighted RMS norm")) return 0;
    }

    return 1;
}

int ds4_gpu_add_rms_norm_weight_tensor(
        ds4_gpu_tensor       *norm_out,
        ds4_gpu_tensor       *sum_out,
        const ds4_gpu_tensor *a,
        const ds4_gpu_tensor *b,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                weight_offset,
        uint32_t                n,
        float                   eps) {
    return ds4_gpu_add_rms_norm_weight_rows_tensor(
        norm_out, sum_out, a, b, model_map, model_size,
        weight_offset, n, 1u, eps);
}

int ds4_gpu_add_rms_norm_weight_rows_tensor(
        ds4_gpu_tensor       *norm_out,
        ds4_gpu_tensor       *sum_out,
        const ds4_gpu_tensor *a,
        const ds4_gpu_tensor *b,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                weight_offset,
        uint32_t                n,
        uint32_t                rows,
        float                   eps) {
    if (!g_initialized && !ds4_gpu_init()) return 0;
    if (!norm_out || !sum_out || !a || !b || n == 0 || rows == 0 ||
        (n & 3u) != 0) return 0;

    @autoreleasepool {
        id<MTLBuffer> abuf = ds4_gpu_tensor_buffer(a);
        id<MTLBuffer> bbuf = ds4_gpu_tensor_buffer(b);
        id<MTLBuffer> sumbuf = ds4_gpu_tensor_buffer(sum_out);
        id<MTLBuffer> normbuf = ds4_gpu_tensor_buffer(norm_out);
        const uint64_t row_bytes = (uint64_t)n * sizeof(float);
        if ((uint64_t)rows > UINT64_MAX / row_bytes) return 0;
        const uint64_t bytes = row_bytes * rows;
        if (!abuf || !bbuf || !sumbuf || !normbuf ||
            ds4_gpu_tensor_bytes(a) < bytes ||
            ds4_gpu_tensor_bytes(b) < bytes ||
            ds4_gpu_tensor_bytes(sum_out) < bytes ||
            ds4_gpu_tensor_bytes(norm_out) < bytes) {
            fprintf(stderr, "ds4: Metal add+RMS norm received undersized activation buffers\n");
            return 0;
        }
        if (weight_offset > model_size || row_bytes > model_size - weight_offset) {
            fprintf(stderr, "ds4: Metal add+RMS norm range is outside the mapped model\n");
            return 0;
        }

        uint64_t inner_offset = 0;
        id<MTLBuffer> wbuf = ds4_gpu_wrap_model_range(model_map,
                                                       model_size,
                                                       weight_offset,
                                                       row_bytes,
                                                       &inner_offset);
        if (!wbuf) return 0;

        ds4_gpu_rms_norm_args args = ds4_gpu_make_rms_norm_args(n, rows, eps);
        int owned = 0;
        id<MTLCommandBuffer> cb = ds4_gpu_command_buffer(&owned);
        if (!cb) return 0;

        id<MTLComputeCommandEncoder> enc = ds4_gpu_compute_encoder(cb);
        [enc setComputePipelineState:g_add_rms_norm_pipeline];
        [enc setBytes:&args length:sizeof(args) atIndex:0];
        [enc setBuffer:abuf offset:ds4_gpu_tensor_offset(a) atIndex:1];
        [enc setBuffer:bbuf offset:ds4_gpu_tensor_offset(b) atIndex:2];
        [enc setBuffer:wbuf offset:(NSUInteger)inner_offset atIndex:3];
        [enc setBuffer:sumbuf offset:ds4_gpu_tensor_offset(sum_out) atIndex:4];
        [enc setBuffer:normbuf offset:ds4_gpu_tensor_offset(norm_out) atIndex:5];
        [enc setThreadgroupMemoryLength:32u * sizeof(float) atIndex:0];
        [enc dispatchThreadgroups:MTLSizeMake(rows, 1, 1)
             threadsPerThreadgroup:MTLSizeMake(ds4_gpu_rms_norm_threads(n), 1, 1)];
        ds4_gpu_end_compute_encoder(cb, enc);

        if (!ds4_gpu_finish_command_buffer(cb, owned, "add+RMS norm")) return 0;
    }

    return 1;
}

/* Decode-only triple fusion: q/kv RMS norm + KV RoPE tail + FP8/raw store.
 * Same arithmetic, order and rounding as the three separate dispatches (see
 * metal/norm.metal); gated and verified against full-vocabulary logits.
 * Returns 1 on success, 0 on unsupported shape (caller falls back). */

/* Release decode fused KV finalizer.  Reference paths are selected by the C
 * graph driver; this Objective-C entry point always means "use the fused
 * Metal kernel." */
/* The fused KV RoPE/FP8 kernel replicates the affine decode RoPE specialisation,
 * so it is only valid where that specialisation is the path in use. */

/* Decode-only fusion: one dispatch does the KV RoPE tail and the FP8/raw
 * finalizer that previously cost two. Same arithmetic, same order; gated and
 * verified against full-vocabulary logits. */

static NSUInteger ds4_gpu_align_up_ns(NSUInteger value, NSUInteger align) {
    return (value + align - 1u) & ~(align - 1u);
}

static int ds4_gpu_encode_cpy_f32_f16_3d(
        id<MTLCommandBuffer> cb,
        id<MTLBuffer>        src,
        NSUInteger           src_off,
        id<MTLBuffer>        dst,
        NSUInteger           dst_off,
        uint32_t             cols,
        uint32_t             rows,
        uint32_t             planes,
        uint64_t             src_row_stride,
        uint64_t             src_plane_stride,
        uint64_t             dst_row_stride,
        uint64_t             dst_plane_stride) {
    if (!cb || !src || !dst || cols == 0 || rows == 0 || planes == 0) return 0;

    ds4_gpu_cpy_args args = {
        .nk0 = (int64_t)cols,
        .ne00 = (int64_t)cols,
        .ne01 = (int64_t)rows,
        .ne02 = (int64_t)planes,
        .ne03 = 1,
        .nb00 = sizeof(float),
        .nb01 = src_row_stride,
        .nb02 = src_plane_stride,
        .nb03 = (uint64_t)planes * src_plane_stride,
        .ne0 = (int64_t)cols,
        .ne1 = (int64_t)rows,
        .ne2 = (int64_t)planes,
        .ne3 = 1,
        .nb0 = sizeof(uint16_t),
        .nb1 = dst_row_stride,
        .nb2 = dst_plane_stride,
        .nb3 = (uint64_t)planes * dst_plane_stride,
    };
    const NSUInteger nth = ds4_gpu_cpy_threads(cols, g_cpy_f32_f16_pipeline);
    const NSUInteger col_groups = ((NSUInteger)cols + nth - 1u) / nth;

    id<MTLComputeCommandEncoder> enc = ds4_gpu_compute_encoder(cb);
    [enc setComputePipelineState:g_cpy_f32_f16_pipeline];
    [enc setBytes:&args length:sizeof(args) atIndex:0];
    [enc setBuffer:src offset:src_off atIndex:1];
    [enc setBuffer:dst offset:dst_off atIndex:2];
    [enc dispatchThreadgroups:MTLSizeMake(col_groups * rows, planes, 1)
         threadsPerThreadgroup:MTLSizeMake(nth, 1, 1)];
    ds4_gpu_end_compute_encoder(cb, enc);

    return 1;
}

static int ds4_gpu_encode_cpy_f16_f16_3d(
        id<MTLCommandBuffer> cb,
        id<MTLBuffer>        src,
        NSUInteger           src_off,
        id<MTLBuffer>        dst,
        NSUInteger           dst_off,
        uint32_t             cols,
        uint32_t             rows,
        uint32_t             planes,
        uint64_t             src_row_stride,
        uint64_t             src_plane_stride,
        uint64_t             dst_row_stride,
        uint64_t             dst_plane_stride) {
    if (!cb || !src || !dst || cols == 0 || rows == 0 || planes == 0) return 0;

    ds4_gpu_cpy_args args = {
        .nk0 = (int64_t)cols,
        .ne00 = (int64_t)cols,
        .ne01 = (int64_t)rows,
        .ne02 = (int64_t)planes,
        .ne03 = 1,
        .nb00 = sizeof(uint16_t),
        .nb01 = src_row_stride,
        .nb02 = src_plane_stride,
        .nb03 = (uint64_t)planes * src_plane_stride,
        .ne0 = (int64_t)cols,
        .ne1 = (int64_t)rows,
        .ne2 = (int64_t)planes,
        .ne3 = 1,
        .nb0 = sizeof(uint16_t),
        .nb1 = dst_row_stride,
        .nb2 = dst_plane_stride,
        .nb3 = (uint64_t)planes * dst_plane_stride,
    };
    const NSUInteger nth = ds4_gpu_cpy_threads(cols, g_cpy_f16_f16_pipeline);
    const NSUInteger col_groups = ((NSUInteger)cols + nth - 1u) / nth;

    id<MTLComputeCommandEncoder> enc = ds4_gpu_compute_encoder(cb);
    [enc setComputePipelineState:g_cpy_f16_f16_pipeline];
    [enc setBytes:&args length:sizeof(args) atIndex:0];
    [enc setBuffer:src offset:src_off atIndex:1];
    [enc setBuffer:dst offset:dst_off atIndex:2];
    [enc dispatchThreadgroups:MTLSizeMake(col_groups * rows, planes, 1)
         threadsPerThreadgroup:MTLSizeMake(nth, 1, 1)];
    ds4_gpu_end_compute_encoder(cb, enc);

    return 1;
}

/* Rectangular causal/window mask for raw prefill: q covers rows
 * [q_row0, q_row0 + n_q) of an n_kv-token block whose keys all live in
 * rows [0, n_kv).  The square prefill case is q_row0 == 0, n_q == n_kv. */

static void ds4_gpu_fill_glm_prefill_mask(
        uint16_t *mask,
        uint32_t  pos0,
        uint32_t  n_tokens,
        uint32_t  cache_len) {
    const uint16_t neg_inf_half = 0xfc00u;
    for (uint32_t q = 0; q < n_tokens; q++) {
        const uint32_t qpos = pos0 + q;
        uint16_t *row = mask + (uint64_t)q * cache_len;
        for (uint32_t k = 0; k < cache_len; k++) {
            row[k] = k <= qpos ? 0u : neg_inf_half;
        }
    }
}

static id<MTLBuffer> ds4_gpu_glm_prefill_mask_buffer(
        uint32_t pos0,
        uint32_t n_tokens,
        uint32_t cache_len,
        NSUInteger mask_bytes) {
    const int same_shape =
        g_glm_flash_attn_mask_valid &&
        g_glm_flash_attn_mask_buffer &&
        g_glm_flash_attn_mask_bytes >= mask_bytes &&
        g_glm_flash_attn_mask_pos0 == pos0 &&
        g_glm_flash_attn_mask_tokens == n_tokens &&
        g_glm_flash_attn_mask_cache_len == cache_len;
    if (same_shape) return g_glm_flash_attn_mask_buffer;

    if (g_glm_flash_attn_mask_buffer) {
        [g_transient_buffers addObject:g_glm_flash_attn_mask_buffer];
        g_glm_flash_attn_mask_buffer = nil;
    }
    g_glm_flash_attn_mask_bytes = 0;
    g_glm_flash_attn_mask_valid = 0;
    if (!ds4_gpu_ensure_scratch_buffer(&g_glm_flash_attn_mask_buffer,
                                        &g_glm_flash_attn_mask_bytes,
                                        mask_bytes,
                                        "ds4_glm_flash_attn_mask")) {
        return nil;
    }

    ds4_gpu_fill_glm_prefill_mask((uint16_t *)[g_glm_flash_attn_mask_buffer contents],
                                  pos0,
                                  n_tokens,
                                  cache_len);
    g_glm_flash_attn_mask_pos0 = pos0;
    g_glm_flash_attn_mask_tokens = n_tokens;
    g_glm_flash_attn_mask_cache_len = cache_len;
    g_glm_flash_attn_mask_valid = 1;
    return g_glm_flash_attn_mask_buffer;
}

/* Rectangular causal/window + compressed-key visibility mask: q covers rows
 * [q_row0, q_row0 + n_q) of an n_tokens-token chunk whose raw keys all stay
 * resident, followed by n_comp compressed keys.  The square prefill case is
 * q_row0 == 0, n_q == n_tokens. */

/* Static-mixed prefill FlashAttention over a rectangular problem: q holds
 * n_q query rows for token positions [q_row0, q_row0 + n_q) of the chunk,
 * while the keys stay full (all n_tokens raw rows plus n_comp compressed
 * rows).  The classic square prefill is q_row0 == 0, n_q == n_tokens. */

/* Raw prefill FlashAttention over a rectangular problem: q holds n_q query
 * rows that correspond to token positions [q_row0, q_row0 + n_q) of the
 * chunk, raw_kv holds all n_kv key rows, and heads receives one output row
 * per query row.  The classic square prefill is q_row0 == 0, n_q == n_kv. */

int ds4_gpu_swiglu_tensor(
        ds4_gpu_tensor       *out,
        const ds4_gpu_tensor *gate,
        const ds4_gpu_tensor *up,
        uint32_t                n,
        float                   clamp,
        float                   weight) {
    if (!g_initialized && !ds4_gpu_init()) return 0;
    if (!out || !gate || !up || n == 0) return 0;
    if (!isfinite(clamp) || clamp < 0.0f || !isfinite(weight)) return 0;

    @autoreleasepool {
        id<MTLBuffer> gatebuf = ds4_gpu_tensor_buffer(gate);
        id<MTLBuffer> upbuf = ds4_gpu_tensor_buffer(up);
        id<MTLBuffer> outbuf = ds4_gpu_tensor_buffer(out);
        const uint64_t bytes = (uint64_t)n * sizeof(float);
        if (!gatebuf || !upbuf || !outbuf ||
            ds4_gpu_tensor_bytes(gate) < bytes ||
            ds4_gpu_tensor_bytes(up) < bytes ||
            ds4_gpu_tensor_bytes(out) < bytes) {
            fprintf(stderr, "ds4: Metal SwiGLU received undersized buffers\n");
            return 0;
        }

        int owned = 0;
        id<MTLCommandBuffer> cb = ds4_gpu_command_buffer(&owned);
        if (!cb) return 0;

        ds4_gpu_glu_args args = {
            .ne00 = (int32_t)n,
            .nb01 = (uint64_t)n * sizeof(float),
            .ne10 = (int32_t)n,
            .nb11 = (uint64_t)n * sizeof(float),
            .ne0 = (int32_t)n,
            .nb1 = (uint64_t)n * sizeof(float),
            .i00 = 0,
            .i10 = 0,
            .alpha = weight,
            .limit = clamp,
        };
        NSUInteger nth = g_swiglu_flat_pipeline.maxTotalThreadsPerThreadgroup;
        if (nth > 256u) nth = 256u;
        if (nth > (NSUInteger)n) nth = (NSUInteger)n;
        if (nth == 0u) nth = 1u;
        const NSUInteger groups = ((NSUInteger)n + nth - 1u) / nth;

        id<MTLComputeCommandEncoder> enc = ds4_gpu_compute_encoder(cb);
        [enc setComputePipelineState:g_swiglu_flat_pipeline];
        [enc setBytes:&args length:sizeof(args) atIndex:0];
        [enc setBuffer:gatebuf offset:ds4_gpu_tensor_offset(gate) atIndex:1];
        [enc setBuffer:upbuf offset:ds4_gpu_tensor_offset(up) atIndex:2];
        [enc setBuffer:outbuf offset:ds4_gpu_tensor_offset(out) atIndex:3];
        [enc dispatchThreadgroups:MTLSizeMake(groups, 1, 1)
             threadsPerThreadgroup:MTLSizeMake(nth, 1, 1)];
        ds4_gpu_end_compute_encoder(cb, enc);

        if (!ds4_gpu_finish_command_buffer(cb, owned, "SwiGLU")) return 0;
    }

    return 1;
}

int ds4_gpu_swiglu_f16_tensor(
        ds4_gpu_tensor       *out,
        const ds4_gpu_tensor *gate,
        const ds4_gpu_tensor *up,
        uint32_t              n,
        float                 clamp,
        float                 weight) {
    if (!g_initialized && !ds4_gpu_init()) return 0;
    if (!out || !gate || !up || n == 0u || !isfinite(clamp) ||
        clamp < 0.0f || !isfinite(weight)) return 0;
    @autoreleasepool {
        id<MTLBuffer> gatebuf = ds4_gpu_tensor_buffer(gate);
        id<MTLBuffer> upbuf = ds4_gpu_tensor_buffer(up);
        id<MTLBuffer> outbuf = ds4_gpu_tensor_buffer(out);
        if (!gatebuf || !upbuf || !outbuf ||
            ds4_gpu_tensor_bytes(gate) < (uint64_t)n*sizeof(float) ||
            ds4_gpu_tensor_bytes(up) < (uint64_t)n*sizeof(float) ||
            ds4_gpu_tensor_bytes(out) < (uint64_t)n*sizeof(uint16_t)) return 0;
        id<MTLComputePipelineState> pipeline =
            ds4_gpu_get_pipeline("kernel_swiglu_flat_f16");
        if (!pipeline) return 0;
        int owned = 0;
        id<MTLCommandBuffer> cb = ds4_gpu_command_buffer(&owned);
        if (!cb) return 0;
        ds4_gpu_glu_args args = {
            .ne00 = (int32_t)n, .nb01 = (uint64_t)n*sizeof(float),
            .ne10 = (int32_t)n, .nb11 = (uint64_t)n*sizeof(float),
            .ne0 = (int32_t)n, .nb1 = (uint64_t)n*sizeof(uint16_t),
            .i00 = 0, .i10 = 0, .alpha = weight, .limit = clamp,
        };
        NSUInteger nth = pipeline.maxTotalThreadsPerThreadgroup;
        if (nth > 256u) nth = 256u;
        if (nth > n) nth = n;
        if (nth == 0u) nth = 1u;
        id<MTLComputeCommandEncoder> enc = ds4_gpu_compute_encoder(cb);
        [enc setComputePipelineState:pipeline];
        [enc setBytes:&args length:sizeof(args) atIndex:0];
        [enc setBuffer:gatebuf offset:ds4_gpu_tensor_offset(gate) atIndex:1];
        [enc setBuffer:upbuf offset:ds4_gpu_tensor_offset(up) atIndex:2];
        [enc setBuffer:outbuf offset:ds4_gpu_tensor_offset(out) atIndex:3];
        [enc dispatchThreadgroups:MTLSizeMake(((NSUInteger)n+nth-1u)/nth, 1u, 1u)
             threadsPerThreadgroup:MTLSizeMake(nth, 1u, 1u)];
        ds4_gpu_end_compute_encoder(cb, enc);
        return ds4_gpu_finish_command_buffer(cb, owned, "SwiGLU F16");
    }
}

int ds4_gpu_add_tensor(
        ds4_gpu_tensor       *out,
        const ds4_gpu_tensor *a,
        const ds4_gpu_tensor *b,
        uint32_t                n) {
    if (!g_initialized && !ds4_gpu_init()) return 0;
    if (!out || !a || !b || n == 0) return 0;

    @autoreleasepool {
        id<MTLBuffer> abuf = ds4_gpu_tensor_buffer(a);
        id<MTLBuffer> bbuf = ds4_gpu_tensor_buffer(b);
        id<MTLBuffer> outbuf = ds4_gpu_tensor_buffer(out);
        const uint64_t bytes = (uint64_t)n * sizeof(float);
        if (!abuf || !bbuf || !outbuf ||
            ds4_gpu_tensor_bytes(a) < bytes ||
            ds4_gpu_tensor_bytes(b) < bytes ||
            ds4_gpu_tensor_bytes(out) < bytes) {
            fprintf(stderr, "ds4: Metal tensor add received undersized buffers\n");
            return 0;
        }

        int owned = 0;
        id<MTLCommandBuffer> cb = ds4_gpu_command_buffer(&owned);
        if (!cb) return 0;

        ds4_gpu_add_flat_args args = { .n = n };
        NSUInteger nth = g_add2_pipeline.maxTotalThreadsPerThreadgroup;
        if (nth > 256u) nth = 256u;
        if (nth > (NSUInteger)n) nth = (NSUInteger)n;
        if (nth == 0u) nth = 1u;
        const NSUInteger groups = ((NSUInteger)n + nth - 1u) / nth;

        id<MTLComputeCommandEncoder> enc = ds4_gpu_compute_encoder(cb);
        [enc setComputePipelineState:g_add2_pipeline];
        [enc setBytes:&args length:sizeof(args) atIndex:0];
        [enc setBuffer:abuf offset:ds4_gpu_tensor_offset(a) atIndex:1];
        [enc setBuffer:bbuf offset:ds4_gpu_tensor_offset(b) atIndex:2];
        [enc setBuffer:outbuf offset:ds4_gpu_tensor_offset(out) atIndex:3];
        [enc dispatchThreadgroups:MTLSizeMake(groups, 1, 1)
             threadsPerThreadgroup:MTLSizeMake(nth, 1, 1)];
        ds4_gpu_end_compute_encoder(cb, enc);

        if (!ds4_gpu_finish_command_buffer(cb, owned, "tensor add")) return 0;
    }

    return 1;
}

int ds4_gpu_qwen35_full_prepare_tensor(
        ds4_gpu_tensor       *q_out,
        ds4_gpu_tensor       *gate_out,
        ds4_gpu_tensor       *key_cache,
        ds4_gpu_tensor       *value_cache,
        const ds4_gpu_tensor *qg,
        const ds4_gpu_tensor *k,
        const ds4_gpu_tensor *v,
        const void           *model_map,
        uint64_t              model_size,
        uint64_t              q_norm_offset,
        uint64_t              k_norm_offset,
        uint32_t              pos,
        uint32_t              cache_cap,
        uint32_t              n_head,
        uint32_t              n_head_kv,
        uint32_t              head_dim,
        uint32_t              rot_dim,
        float                 eps,
        float                 freq_base) {
    return ds4_gpu_qwen35_full_prepare_batch_tensor(
        q_out, gate_out, key_cache, value_cache, qg, k, v,
        model_map, model_size, q_norm_offset, k_norm_offset,
        pos, 1u, cache_cap, n_head, n_head_kv, head_dim, rot_dim,
        eps, freq_base);
}

int ds4_gpu_qwen35_full_prepare_batch_tensor(
        ds4_gpu_tensor       *q_out,
        ds4_gpu_tensor       *gate_out,
        ds4_gpu_tensor       *key_cache,
        ds4_gpu_tensor       *value_cache,
        const ds4_gpu_tensor *qg,
        const ds4_gpu_tensor *k,
        const ds4_gpu_tensor *v,
        const void           *model_map,
        uint64_t              model_size,
        uint64_t              q_norm_offset,
        uint64_t              k_norm_offset,
        uint32_t              pos0,
        uint32_t              n_tokens,
        uint32_t              cache_cap,
        uint32_t              n_head,
        uint32_t              n_head_kv,
        uint32_t              head_dim,
        uint32_t              rot_dim,
        float                 eps,
        float                 freq_base) {
    if (!g_initialized && !ds4_gpu_init()) return 0;
    if (!q_out || !gate_out || !key_cache || !value_cache || !qg || !k || !v ||
        !model_map || n_tokens == 0 || cache_cap == 0 || pos0 >= cache_cap ||
        n_tokens > cache_cap - pos0 || n_head == 0 ||
        n_head_kv == 0 || n_head % n_head_kv != 0 || head_dim == 0 ||
        head_dim > 256u || (head_dim & (head_dim - 1u)) != 0 ||
        rot_dim > head_dim || (rot_dim & 1u) != 0 || eps <= 0.0f ||
        !isfinite(freq_base) || freq_base <= 0.0f) return 0;

    @autoreleasepool {
        const uint64_t q_bytes =
            (uint64_t)n_tokens * n_head * head_dim * sizeof(float);
        const uint64_t kv_bytes =
            (uint64_t)n_tokens * n_head_kv * head_dim * sizeof(float);
        const uint64_t cache_bytes = (uint64_t)cache_cap * n_head_kv * head_dim * sizeof(uint16_t);
        const uint64_t norm_bytes = (uint64_t)head_dim * sizeof(float);
        if (ds4_gpu_tensor_bytes(q_out) < q_bytes ||
            ds4_gpu_tensor_bytes(gate_out) < q_bytes ||
            ds4_gpu_tensor_bytes(qg) < 2u * q_bytes ||
            ds4_gpu_tensor_bytes(k) < kv_bytes ||
            ds4_gpu_tensor_bytes(v) < kv_bytes ||
            ds4_gpu_tensor_bytes(key_cache) < cache_bytes ||
            ds4_gpu_tensor_bytes(value_cache) < cache_bytes ||
            q_norm_offset > model_size || norm_bytes > model_size - q_norm_offset ||
            k_norm_offset > model_size || norm_bytes > model_size - k_norm_offset) return 0;

        id<MTLBuffer> qoutbuf = ds4_gpu_tensor_buffer(q_out);
        id<MTLBuffer> gatebuf = ds4_gpu_tensor_buffer(gate_out);
        id<MTLBuffer> kcachebuf = ds4_gpu_tensor_buffer(key_cache);
        id<MTLBuffer> vcachebuf = ds4_gpu_tensor_buffer(value_cache);
        id<MTLBuffer> qgbuf = ds4_gpu_tensor_buffer(qg);
        id<MTLBuffer> kbuf = ds4_gpu_tensor_buffer(k);
        id<MTLBuffer> vbuf = ds4_gpu_tensor_buffer(v);
        uint64_t qnorm_inner = 0, knorm_inner = 0;
        id<MTLBuffer> qnormbuf = ds4_gpu_wrap_model_range(model_map, model_size,
                                                           q_norm_offset, norm_bytes,
                                                           &qnorm_inner);
        id<MTLBuffer> knormbuf = ds4_gpu_wrap_model_range(model_map, model_size,
                                                           k_norm_offset, norm_bytes,
                                                           &knorm_inner);
        id<MTLComputePipelineState> pipeline =
            ds4_gpu_get_pipeline("kernel_qwen35_full_prepare");
        if (!qoutbuf || !gatebuf || !kcachebuf || !vcachebuf || !qgbuf ||
            !kbuf || !vbuf || !qnormbuf || !knormbuf || !pipeline) return 0;

        ds4_gpu_qwen35_full_prepare_args args = {
            .pos0 = pos0, .n_tokens = n_tokens,
            .cache_cap = cache_cap, .n_head = n_head,
            .n_head_kv = n_head_kv, .head_dim = head_dim, .rot_dim = rot_dim,
            .eps = eps, .freq_base = freq_base,
        };
        int owned = 0;
        id<MTLCommandBuffer> cb = ds4_gpu_command_buffer(&owned);
        if (!cb) return 0;
        id<MTLComputeCommandEncoder> enc = ds4_gpu_compute_encoder(cb);
        [enc setComputePipelineState:pipeline];
        [enc setBytes:&args length:sizeof(args) atIndex:0];
        [enc setBuffer:qgbuf offset:ds4_gpu_tensor_offset(qg) atIndex:1];
        [enc setBuffer:kbuf offset:ds4_gpu_tensor_offset(k) atIndex:2];
        [enc setBuffer:vbuf offset:ds4_gpu_tensor_offset(v) atIndex:3];
        [enc setBuffer:qnormbuf offset:(NSUInteger)qnorm_inner atIndex:4];
        [enc setBuffer:knormbuf offset:(NSUInteger)knorm_inner atIndex:5];
        [enc setBuffer:qoutbuf offset:ds4_gpu_tensor_offset(q_out) atIndex:6];
        [enc setBuffer:gatebuf offset:ds4_gpu_tensor_offset(gate_out) atIndex:7];
        [enc setBuffer:kcachebuf offset:ds4_gpu_tensor_offset(key_cache) atIndex:8];
        [enc setBuffer:vcachebuf offset:ds4_gpu_tensor_offset(value_cache) atIndex:9];
        [enc setThreadgroupMemoryLength:(NSUInteger)head_dim * sizeof(float) atIndex:0];
        [enc dispatchThreadgroups:MTLSizeMake(n_head, n_tokens, 1)
             threadsPerThreadgroup:MTLSizeMake(head_dim, 1, 1)];
        ds4_gpu_end_compute_encoder(cb, enc);
        return ds4_gpu_finish_command_buffer(cb, owned, "Qwen3.6 full attention prepare");
    }
}

static uint32_t ds4_gpu_qwen35_flash_decode_splits(uint32_t cache_len) {
    /*
     * An M4 Pro sweep at 2K and 8K selected four split-K workgroups.  Below
     * 512 keys their reduction and gate dispatch overhead did not pay back, so
     * retain the legacy fused two-pass kernel for short contexts.  Other GPU
     * families stay on that path until they have their own measured launch
     * configuration.
     */
    if (cache_len < 512u ||
        !ds4_gpu_device_name_contains("M4 Pro") ||
        ds4_gpu_env_bool("DS4_METAL_DISABLE_QWEN_FLASH_DECODE") > 0) {
        return 0u;
    }
    return 4u;
}

static int ds4_gpu_qwen35_attention_flash_decode_tensor(
        ds4_gpu_tensor       *out,
        const ds4_gpu_tensor *q,
        const ds4_gpu_tensor *gate,
        const ds4_gpu_tensor *key_cache,
        const ds4_gpu_tensor *value_cache,
        uint32_t              cache_len,
        uint32_t              cache_cap,
        uint32_t              n_head,
        uint32_t              n_head_kv,
        uint32_t              head_dim,
        uint32_t              nwg) {
    if (head_dim != 256u || nwg == 0u || nwg > 32u) return 0;
    const NSUInteger row_bytes = (NSUInteger)head_dim * sizeof(float);
    const NSUInteger row_bytes_f16 =
        (NSUInteger)head_dim * sizeof(uint16_t);
    const NSUInteger cache_row_stride =
        (NSUInteger)n_head_kv * row_bytes_f16;
    const uint64_t q_bytes = (uint64_t)n_head * row_bytes;
    const uint64_t cache_bytes =
        (uint64_t)cache_cap * n_head_kv * row_bytes_f16;
    id<MTLBuffer> outbuf = ds4_gpu_tensor_buffer(out);
    id<MTLBuffer> qbuf = ds4_gpu_tensor_buffer(q);
    id<MTLBuffer> gatebuf = ds4_gpu_tensor_buffer(gate);
    id<MTLBuffer> keybuf = ds4_gpu_tensor_buffer(key_cache);
    id<MTLBuffer> valbuf = ds4_gpu_tensor_buffer(value_cache);
    if (!outbuf || !qbuf || !gatebuf || !keybuf || !valbuf ||
        ds4_gpu_tensor_bytes(out) < q_bytes ||
        ds4_gpu_tensor_bytes(q) < q_bytes ||
        ds4_gpu_tensor_bytes(gate) < q_bytes ||
        ds4_gpu_tensor_bytes(key_cache) < cache_bytes ||
        ds4_gpu_tensor_bytes(value_cache) < cache_bytes) {
        return 0;
    }

    const uint32_t ncpsg = 32u;
    const bool has_kvpad = (cache_len % ncpsg) != 0u;
    const uint32_t nsg = ds4_gpu_flash_attn_vec_nsg(cache_len, nwg, ncpsg);
    const NSUInteger pad_bytes = has_kvpad
        ? 2u * (NSUInteger)ncpsg * cache_row_stride * n_head_kv
        : 1u;
    const NSUInteger nrows = n_head;
    const NSUInteger tmp_bytes =
        nrows * head_dim * (NSUInteger)nwg * sizeof(float) +
        nrows * 2u * (NSUInteger)nwg * sizeof(float);
    if (!ds4_gpu_ensure_scratch_buffer(&g_flash_attn_pad_buffer,
                                        &g_flash_attn_pad_bytes,
                                        pad_bytes,
                                        "ds4_qwen_flash_attn_pad") ||
        (nwg > 1u &&
         !ds4_gpu_ensure_scratch_buffer(&g_flash_attn_tmp_buffer,
                                         &g_flash_attn_tmp_bytes,
                                         tmp_bytes,
                                         "ds4_qwen_flash_attn_tmp"))) {
        return 0;
    }

    id<MTLComputePipelineState> pad_pipeline = has_kvpad
        ? ds4_gpu_get_flash_attn_pad_pipeline(false, (int32_t)ncpsg) : nil;
    id<MTLComputePipelineState> vec_pipeline =
        ds4_gpu_get_flash_attn_vec_pipeline(
            "kernel_qwen35_flash_attn_ext_vec_f16_dk256_dv256",
            false, false, false, false, has_kvpad, false, true,
            (int32_t)head_dim, (int32_t)head_dim,
            (int32_t)nsg, (int32_t)nwg);
    id<MTLComputePipelineState> reduce_pipeline = nwg > 1u
        ? ds4_gpu_get_flash_attn_reduce_pipeline(
              (int32_t)head_dim, (int32_t)nwg) : nil;
    id<MTLComputePipelineState> gate_pipeline =
        ds4_gpu_get_pipeline("kernel_qwen35_attention_gate");
    if ((has_kvpad && !pad_pipeline) || !vec_pipeline ||
        (nwg > 1u && !reduce_pipeline) || !gate_pipeline) {
        return 0;
    }

    int owned = 0;
    id<MTLCommandBuffer> cb = ds4_gpu_command_buffer(&owned);
    if (!cb) return 0;
    if (has_kvpad) {
        ds4_gpu_flash_attn_pad_args pad_args = {
            .ne11 = (int32_t)cache_len,
            .ne_12_2 = (int32_t)n_head_kv,
            .ne_12_3 = 1,
            .nb11 = cache_row_stride,
            .nb12 = row_bytes_f16,
            .nb13 = (uint64_t)cache_cap * cache_row_stride,
            .nb21 = cache_row_stride,
            .nb22 = row_bytes_f16,
            .nb23 = (uint64_t)cache_cap * cache_row_stride,
            .ne31 = 1,
            .ne32 = 1,
            .ne33 = 1,
            .nb31 = 0,
            .nb32 = 0,
            .nb33 = 0,
        };
        id<MTLComputeCommandEncoder> enc = ds4_gpu_compute_encoder(cb);
        [enc setComputePipelineState:pad_pipeline];
        [enc setBytes:&pad_args length:sizeof(pad_args) atIndex:0];
        [enc setBuffer:keybuf offset:ds4_gpu_tensor_offset(key_cache) atIndex:1];
        [enc setBuffer:valbuf offset:ds4_gpu_tensor_offset(value_cache) atIndex:2];
        [enc setBuffer:qbuf offset:ds4_gpu_tensor_offset(q) atIndex:3];
        [enc setBuffer:g_flash_attn_pad_buffer offset:0 atIndex:4];
        [enc dispatchThreadgroups:MTLSizeMake(ncpsg, n_head_kv, 1)
             threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
        ds4_gpu_end_compute_encoder(cb, enc);
    }

    ds4_gpu_flash_attn_vec_args vec_args = {
        .ne01 = 1,
        .ne02 = (int32_t)n_head,
        .ne03 = 1,
        .nb01 = (uint64_t)n_head * row_bytes,
        .nb02 = row_bytes,
        .nb03 = (uint64_t)n_head * row_bytes,
        .ne11 = (int32_t)cache_len,
        .ne_12_2 = (int32_t)n_head_kv,
        .ne_12_3 = 1,
        .ns10 = (int32_t)head_dim,
        .nb11 = cache_row_stride,
        .nb12 = row_bytes_f16,
        .nb13 = (uint64_t)cache_cap * cache_row_stride,
        .ns20 = (int32_t)head_dim,
        .nb21 = cache_row_stride,
        .nb22 = row_bytes_f16,
        .nb23 = (uint64_t)cache_cap * cache_row_stride,
        .ne31 = 1,
        .ne32 = 1,
        .ne33 = 1,
        .nb31 = 0,
        .nb32 = 0,
        .nb33 = 0,
        .ne1 = (int32_t)n_head,
        .ne2 = 1,
        .ne3 = 1,
        .scale = 1.0f / sqrtf((float)head_dim),
        .max_bias = 0.0f,
        .m0 = 0.0f,
        .m1 = 0.0f,
        .n_head_log2 = 0,
        .logit_softcap = 0.0f,
    };
    const NSUInteger shared_elems =
        (2u * ds4_gpu_align_up_ns(head_dim, 128u) + 4u * ncpsg +
         2u * ds4_gpu_align_up_ns(head_dim, 128u)) * nsg;
    const NSUInteger shared_bytes =
        ds4_gpu_align_up_ns(shared_elems * (sizeof(float) / 2u), 16u);
    id<MTLComputeCommandEncoder> enc = ds4_gpu_compute_encoder(cb);
    [enc setComputePipelineState:vec_pipeline];
    [enc setBytes:&vec_args length:sizeof(vec_args) atIndex:0];
    [enc setBuffer:qbuf offset:ds4_gpu_tensor_offset(q) atIndex:1];
    [enc setBuffer:keybuf offset:ds4_gpu_tensor_offset(key_cache) atIndex:2];
    [enc setBuffer:valbuf offset:ds4_gpu_tensor_offset(value_cache) atIndex:3];
    [enc setBuffer:qbuf offset:ds4_gpu_tensor_offset(q) atIndex:4];
    [enc setBuffer:qbuf offset:ds4_gpu_tensor_offset(q) atIndex:5];
    [enc setBuffer:g_flash_attn_pad_buffer offset:0 atIndex:6];
    [enc setBuffer:nwg > 1u ? g_flash_attn_tmp_buffer : outbuf
            offset:nwg > 1u ? 0 : ds4_gpu_tensor_offset(out) atIndex:7];
    [enc setThreadgroupMemoryLength:shared_bytes atIndex:0];
    [enc dispatchThreadgroups:MTLSizeMake(1, n_head, nwg)
         threadsPerThreadgroup:MTLSizeMake(32, nsg, 1)];
    ds4_gpu_end_compute_encoder(cb, enc);

    if (nwg > 1u) {
        ds4_gpu_flash_attn_reduce_args reduce_args = {
            .nrows = (int32_t)nrows,
        };
        enc = ds4_gpu_compute_encoder(cb);
        [enc setComputePipelineState:reduce_pipeline];
        [enc setBytes:&reduce_args length:sizeof(reduce_args) atIndex:0];
        [enc setBuffer:g_flash_attn_tmp_buffer offset:0 atIndex:1];
        [enc setBuffer:outbuf offset:ds4_gpu_tensor_offset(out) atIndex:2];
        [enc dispatchThreadgroups:MTLSizeMake(nrows, 1, 1)
             threadsPerThreadgroup:MTLSizeMake(32u * nwg, 1, 1)];
        ds4_gpu_end_compute_encoder(cb, enc);
    }

    ds4_gpu_qwen35_attention_args gate_args = {
        .pos0 = cache_len - 1u,
        .n_tokens = 1u,
        .cache_cap = cache_cap,
        .n_head = n_head,
        .n_head_kv = n_head_kv,
        .head_dim = head_dim,
        .scale = vec_args.scale,
    };
    enc = ds4_gpu_compute_encoder(cb);
    [enc setComputePipelineState:gate_pipeline];
    [enc setBytes:&gate_args length:sizeof(gate_args) atIndex:0];
    [enc setBuffer:gatebuf offset:ds4_gpu_tensor_offset(gate) atIndex:1];
    [enc setBuffer:outbuf offset:ds4_gpu_tensor_offset(out) atIndex:2];
    [enc dispatchThreadgroups:MTLSizeMake((nrows * head_dim + 255u) / 256u,
                                          1, 1)
         threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
    ds4_gpu_end_compute_encoder(cb, enc);
    return ds4_gpu_finish_command_buffer(
        cb, owned, "Qwen3.6 native FlashAttention decode");
}

int ds4_gpu_qwen35_attention_tensor(
        ds4_gpu_tensor       *out,
        ds4_gpu_tensor       *scores,
        const ds4_gpu_tensor *q,
        const ds4_gpu_tensor *gate,
        const ds4_gpu_tensor *key_cache,
        const ds4_gpu_tensor *value_cache,
        uint32_t              cache_len,
        uint32_t              cache_cap,
        uint32_t              n_head,
        uint32_t              n_head_kv,
        uint32_t              head_dim) {
    if (!out || !scores || !q || !gate || !key_cache || !value_cache ||
        cache_len == 0u || cache_len > cache_cap || n_head == 0u ||
        n_head_kv == 0u || n_head % n_head_kv != 0u || head_dim == 0u ||
        head_dim > 256u || (head_dim & (head_dim - 1u)) != 0u) {
        return 0;
    }
    const uint32_t flash_splits =
        ds4_gpu_qwen35_flash_decode_splits(cache_len);
    /*
     * The generic tail-pad copier reads a complete token stride for each KV
     * head.  A following cache row makes that harmless for ordinary decode,
     * but a partial final row has no such storage.  The legacy kernel already
     * handles that one boundary case without padding.
     */
    const bool flash_tail_safe =
        cache_len < cache_cap || cache_len % 32u == 0u;
    if (flash_splits != 0u && flash_tail_safe && head_dim == 256u) {
        return ds4_gpu_qwen35_attention_flash_decode_tensor(
            out, q, gate, key_cache, value_cache, cache_len, cache_cap,
            n_head, n_head_kv, head_dim, flash_splits);
    }
    return ds4_gpu_qwen35_attention_batch_tensor(
        out, scores, q, gate, key_cache, value_cache,
        cache_len - 1u, 1u, cache_cap,
        n_head, n_head_kv, head_dim);
}

int ds4_gpu_qwen35_attention_batch_tensor(
        ds4_gpu_tensor       *out,
        ds4_gpu_tensor       *scores,
        const ds4_gpu_tensor *q,
        const ds4_gpu_tensor *gate,
        const ds4_gpu_tensor *key_cache,
        const ds4_gpu_tensor *value_cache,
        uint32_t              pos0,
        uint32_t              n_tokens,
        uint32_t              cache_cap,
        uint32_t              n_head,
        uint32_t              n_head_kv,
        uint32_t              head_dim) {
    if (!g_initialized && !ds4_gpu_init()) return 0;
    if (!out || !scores || !q || !gate || !key_cache || !value_cache ||
        n_tokens == 0 || pos0 >= cache_cap || n_tokens > cache_cap - pos0 ||
        n_head == 0 || n_head_kv == 0 ||
        n_head % n_head_kv != 0 || head_dim == 0 || head_dim > 256u ||
        (head_dim & (head_dim - 1u)) != 0) return 0;
    @autoreleasepool {
        const uint64_t q_row_bytes =
            (uint64_t)n_head * head_dim * sizeof(float);
        const uint64_t q_bytes = (uint64_t)n_tokens * q_row_bytes;
        const uint64_t score_bytes =
            (uint64_t)n_tokens * n_head * cache_cap * sizeof(float);
        const uint64_t cache_bytes = (uint64_t)cache_cap * n_head_kv * head_dim * sizeof(uint16_t);
        if (ds4_gpu_tensor_bytes(out) < q_bytes || ds4_gpu_tensor_bytes(q) < q_bytes ||
            ds4_gpu_tensor_bytes(gate) < q_bytes || ds4_gpu_tensor_bytes(scores) < score_bytes ||
            ds4_gpu_tensor_bytes(key_cache) < cache_bytes ||
            ds4_gpu_tensor_bytes(value_cache) < cache_bytes) return 0;
        id<MTLComputePipelineState> score_pipeline =
            ds4_gpu_get_pipeline("kernel_qwen35_attention_scores");
        id<MTLComputePipelineState> out_pipeline =
            ds4_gpu_get_pipeline("kernel_qwen35_attention_output");
        if (!score_pipeline || !out_pipeline) return 0;
        int owned = 0;
        id<MTLCommandBuffer> cb = ds4_gpu_command_buffer(&owned);
        if (!cb) return 0;
        ds4_gpu_qwen35_attention_args args = {
            .pos0 = pos0, .n_tokens = n_tokens, .cache_cap = cache_cap,
            .n_head = n_head, .n_head_kv = n_head_kv,
            .head_dim = head_dim,
            .scale = 1.0f / sqrtf((float)head_dim),
        };
        const uint32_t cache_len = pos0 + n_tokens;
        id<MTLComputeCommandEncoder> enc = ds4_gpu_compute_encoder(cb);
        [enc setComputePipelineState:score_pipeline];
        [enc setBytes:&args length:sizeof(args) atIndex:0];
        [enc setBuffer:ds4_gpu_tensor_buffer(q)
                offset:ds4_gpu_tensor_offset(q) atIndex:1];
        [enc setBuffer:ds4_gpu_tensor_buffer(key_cache)
                offset:ds4_gpu_tensor_offset(key_cache) atIndex:2];
        [enc setBuffer:ds4_gpu_tensor_buffer(scores)
                offset:ds4_gpu_tensor_offset(scores) atIndex:3];
        [enc dispatchThreadgroups:MTLSizeMake(
                ((NSUInteger)cache_len + 255u) / 256u, n_head, 1)
             threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        ds4_gpu_end_compute_encoder(cb, enc);

        enc = ds4_gpu_compute_encoder(cb);
        [enc setComputePipelineState:out_pipeline];
        [enc setBytes:&args length:sizeof(args) atIndex:0];
        [enc setBuffer:ds4_gpu_tensor_buffer(gate)
                offset:ds4_gpu_tensor_offset(gate) atIndex:1];
        [enc setBuffer:ds4_gpu_tensor_buffer(value_cache)
                offset:ds4_gpu_tensor_offset(value_cache) atIndex:2];
        [enc setBuffer:ds4_gpu_tensor_buffer(scores)
                offset:ds4_gpu_tensor_offset(scores) atIndex:3];
        [enc setBuffer:ds4_gpu_tensor_buffer(out)
                offset:ds4_gpu_tensor_offset(out) atIndex:4];
        [enc setThreadgroupMemoryLength:
                (NSUInteger)head_dim * sizeof(float) atIndex:0];
        [enc dispatchThreadgroups:MTLSizeMake(n_head, 1, 1)
             threadsPerThreadgroup:MTLSizeMake(head_dim, 1, 1)];
        ds4_gpu_end_compute_encoder(cb, enc);
        return ds4_gpu_finish_command_buffer(cb, owned, "Qwen3.6 grouped-query attention");
    }
}

int ds4_gpu_qwen35_gdn_tensor(
        ds4_gpu_tensor       *out,
        ds4_gpu_tensor       *prepared,
        ds4_gpu_tensor       *g,
        ds4_gpu_tensor       *b,
        ds4_gpu_tensor       *recurrent_out,
        ds4_gpu_tensor       *conv_state,
        ds4_gpu_tensor       *ssm_state,
        const ds4_gpu_tensor *qkv,
        const ds4_gpu_tensor *z,
        const ds4_gpu_tensor *alpha,
        const ds4_gpu_tensor *beta,
        const void           *model_map,
        uint64_t              model_size,
        uint64_t              conv_weight_offset,
        uint64_t              dt_bias_offset,
        uint64_t              a_offset,
        uint64_t              norm_offset,
        uint32_t              channels,
        uint32_t              qk_heads,
        uint32_t              v_heads,
        uint32_t              state_dim,
        uint32_t              conv_width,
        float                 eps) {
    return ds4_gpu_qwen35_gdn_batch_tensor(
        out, prepared, g, b, recurrent_out, conv_state, ssm_state,
        qkv, z, alpha, beta, model_map, model_size,
        conv_weight_offset, dt_bias_offset, a_offset, norm_offset,
        1u, channels, qk_heads, v_heads, state_dim, conv_width, eps);
}

static int ds4_gpu_qwen35_gdn_batch_impl(
        ds4_gpu_tensor       *out,
        ds4_gpu_tensor       *prepared,
        ds4_gpu_tensor       *g,
        ds4_gpu_tensor       *b,
        ds4_gpu_tensor       *recurrent_out,
        ds4_gpu_tensor       *conv_state,
        ds4_gpu_tensor       *ssm_state,
        ds4_gpu_tensor       *middle_conv_state,
        ds4_gpu_tensor       *middle_ssm_state,
        ds4_gpu_tensor       *penultimate_conv_state,
        ds4_gpu_tensor       *penultimate_ssm_state,
        ds4_gpu_tensor       *final_conv_state,
        ds4_gpu_tensor       *final_ssm_state,
        const ds4_gpu_tensor *qkv,
        const ds4_gpu_tensor *z,
        const ds4_gpu_tensor *alpha,
        const ds4_gpu_tensor *beta,
        const void           *model_map,
        uint64_t              model_size,
        uint64_t              conv_weight_offset,
        uint64_t              dt_bias_offset,
        uint64_t              a_offset,
        uint64_t              norm_offset,
        uint32_t              n_tokens,
        uint32_t              channels,
        uint32_t              qk_heads,
        uint32_t              v_heads,
        uint32_t              state_dim,
        uint32_t              conv_width,
        float                 eps,
        ds4_gpu_tensor       *chunk_w,
        ds4_gpu_tensor       *chunk_u,
        ds4_gpu_tensor       *chunk_qk,
        ds4_gpu_tensor       *chunk_cumulative_g,
        int                   chunkwise,
        uint32_t              row_offset) {
    if (!g_initialized && !ds4_gpu_init()) return 0;
    const bool preserve_four =
        penultimate_conv_state != NULL || penultimate_ssm_state != NULL;
    const bool preserve_first = !preserve_four &&
        (middle_conv_state != NULL || middle_ssm_state != NULL ||
         final_conv_state != NULL || final_ssm_state != NULL);
    const bool preserve_state = preserve_first || preserve_four;
    if (!out || !prepared || !g || !b || !recurrent_out || !conv_state ||
        !ssm_state || !qkv || !z || !alpha || !beta || !model_map ||
        n_tokens == 0 || channels == 0 || qk_heads == 0 || v_heads == 0 ||
        state_dim != 128u ||
        conv_width < 2u || v_heads != channels / state_dim - 2u * qk_heads ||
        (preserve_first && (!middle_conv_state || !middle_ssm_state ||
                            !final_conv_state || !final_ssm_state ||
                            n_tokens < 2u || n_tokens > 3u || chunkwise)) ||
        (preserve_four && (!middle_conv_state || !middle_ssm_state ||
                           !penultimate_conv_state ||
                           !penultimate_ssm_state ||
                           !final_conv_state || !final_ssm_state ||
                           n_tokens != 4u || chunkwise)) ||
        eps <= 0.0f) return 0;
    /* The matrix path consumes complete 8x8 tiles.  Short/ragged tails use
     * the recurrent kernel transparently; full prefill chunks use WY. */
    const bool wy_chunkwise = chunkwise && n_tokens >= 16u &&
        n_tokens <= 64u && (n_tokens & 7u) == 0u &&
        v_heads == 3u * qk_heads;
    if (wy_chunkwise &&
        (!chunk_w || !chunk_u || !chunk_qk || !chunk_cumulative_g)) return 0;
    @autoreleasepool {
        const uint64_t channels_row_offset =
            (uint64_t)row_offset * channels * sizeof(float);
        const uint64_t inner_row_offset =
            (uint64_t)row_offset * v_heads * state_dim * sizeof(float);
        const uint64_t param_row_offset =
            (uint64_t)row_offset * v_heads * sizeof(float);
        const uint64_t chunk_inner_offset = 0;
        const uint64_t chunk_param_offset = 0;
        const uint64_t channels_bytes =
            (uint64_t)n_tokens * channels * sizeof(float);
        const uint64_t inner_bytes =
            (uint64_t)n_tokens * v_heads * state_dim * sizeof(float);
        const uint64_t param_rows_bytes =
            (uint64_t)n_tokens * v_heads * sizeof(float);
        const uint64_t chunk_qk_bytes =
            (uint64_t)n_tokens * n_tokens * v_heads * sizeof(float);
        const uint64_t param_bytes = (uint64_t)v_heads * sizeof(float);
        const uint64_t conv_state_bytes = (uint64_t)channels * (conv_width - 1u) * sizeof(float);
        const uint64_t ssm_state_bytes = (uint64_t)v_heads * state_dim * state_dim * sizeof(float);
        const uint64_t conv_weight_bytes = (uint64_t)channels * conv_width * sizeof(float);
        const uint64_t norm_bytes = (uint64_t)state_dim * sizeof(float);
        if (ds4_gpu_tensor_bytes(prepared) < channels_row_offset + channels_bytes ||
            ds4_gpu_tensor_bytes(qkv) < channels_row_offset + channels_bytes ||
            ds4_gpu_tensor_bytes(z) < inner_row_offset + inner_bytes ||
            ds4_gpu_tensor_bytes(out) < inner_row_offset + inner_bytes ||
            ds4_gpu_tensor_bytes(recurrent_out) < inner_row_offset + inner_bytes ||
            ds4_gpu_tensor_bytes(alpha) < param_row_offset + param_rows_bytes ||
            ds4_gpu_tensor_bytes(beta) < param_row_offset + param_rows_bytes ||
            ds4_gpu_tensor_bytes(g) < param_row_offset + param_rows_bytes ||
            ds4_gpu_tensor_bytes(b) < param_row_offset + param_rows_bytes ||
            ds4_gpu_tensor_bytes(conv_state) < conv_state_bytes ||
            ds4_gpu_tensor_bytes(ssm_state) < ssm_state_bytes ||
            (preserve_state &&
             (ds4_gpu_tensor_bytes(middle_conv_state) < conv_state_bytes ||
              ds4_gpu_tensor_bytes(middle_ssm_state) < ssm_state_bytes ||
              ds4_gpu_tensor_bytes(final_conv_state) < conv_state_bytes ||
              ds4_gpu_tensor_bytes(final_ssm_state) < ssm_state_bytes)) ||
            (preserve_four &&
             (ds4_gpu_tensor_bytes(penultimate_conv_state) < conv_state_bytes ||
              ds4_gpu_tensor_bytes(penultimate_ssm_state) < ssm_state_bytes)) ||
            (wy_chunkwise &&
             (ds4_gpu_tensor_bytes(chunk_w) < inner_bytes ||
              ds4_gpu_tensor_bytes(chunk_u) < inner_bytes ||
              ds4_gpu_tensor_bytes(chunk_qk) < chunk_qk_bytes ||
              ds4_gpu_tensor_bytes(chunk_cumulative_g) < param_rows_bytes)) ||
            conv_weight_offset > model_size || conv_weight_bytes > model_size - conv_weight_offset ||
            dt_bias_offset > model_size || param_bytes > model_size - dt_bias_offset ||
            a_offset > model_size || param_bytes > model_size - a_offset ||
            norm_offset > model_size || norm_bytes > model_size - norm_offset) return 0;

        uint64_t conv_inner = 0, dt_inner = 0, a_inner = 0, norm_inner = 0;
        id<MTLBuffer> convbuf = ds4_gpu_wrap_model_range(model_map, model_size,
            conv_weight_offset, conv_weight_bytes, &conv_inner);
        id<MTLBuffer> dtbuf = ds4_gpu_wrap_model_range(model_map, model_size,
            dt_bias_offset, param_bytes, &dt_inner);
        id<MTLBuffer> abuf = ds4_gpu_wrap_model_range(model_map, model_size,
            a_offset, param_bytes, &a_inner);
        id<MTLBuffer> normbuf = ds4_gpu_wrap_model_range(model_map, model_size,
            norm_offset, norm_bytes, &norm_inner);
        const bool fused_decode_params =
            n_tokens == 1u && !chunkwise &&
            (getenv("DS4_METAL_QWEN_DECODE_FUSIONS") != NULL ||
             getenv("DS4_METAL_QWEN_DECODE_GDN_PARAM_FUSION") != NULL);
        id<MTLComputePipelineState> convp = ds4_gpu_get_pipeline(
            preserve_four ? "kernel_qwen35_gdn_conv_preserve_four" :
            preserve_first ? "kernel_qwen35_gdn_conv_preserve_first" :
            fused_decode_params ? "kernel_qwen35_gdn_conv_params" :
                                  "kernel_qwen35_gdn_conv");
        id<MTLComputePipelineState> normp = ds4_gpu_get_pipeline("kernel_qwen35_gdn_qk_norm");
        id<MTLComputePipelineState> paramp = fused_decode_params ? nil :
            ds4_gpu_get_pipeline("kernel_qwen35_gdn_params");
        id<MTLComputePipelineState> recp = wy_chunkwise ? nil :
            ds4_gpu_get_pipeline(preserve_first ?
                "kernel_qwen35_gdn_recurrent_preserve_first" :
                preserve_four ? "kernel_qwen35_gdn_recurrent_preserve_four" :
                "kernel_qwen35_gdn_recurrent");
        id<MTLComputePipelineState> wy_packp = wy_chunkwise ?
            ds4_gpu_get_pipeline("kernel_qwen35_gdn_wy_pack") : nil;
        id<MTLComputePipelineState> wy_kkt_qkp = wy_chunkwise ?
            ds4_gpu_get_pipeline("kernel_qwen35_gdn_wy_kkt_qk_raw") : nil;
        id<MTLComputePipelineState> wy_solvep = wy_chunkwise ?
            ds4_gpu_get_pipeline("kernel_qwen35_gdn_wy_solve") : nil;
        id<MTLComputePipelineState> wy_valuesp = wy_chunkwise ?
            ds4_gpu_get_pipeline("kernel_qwen35_gdn_wy_values_output_32") : nil;
        id<MTLComputePipelineState> postp = ds4_gpu_get_pipeline("kernel_qwen35_gdn_post");
        if (!convbuf || !dtbuf || !abuf || !normbuf || !convp || !normp ||
            (!fused_decode_params && !paramp) || !postp ||
            (!wy_chunkwise && !recp) ||
            (wy_chunkwise && (!wy_packp || !wy_kkt_qkp ||
                              !wy_solvep || !wy_valuesp))) return 0;
        ds4_gpu_qwen35_gdn_args args = {
            .n_tokens = n_tokens, .channels = channels,
            .qk_heads = qk_heads, .v_heads = v_heads,
            .state_dim = state_dim, .conv_width = conv_width, .eps = eps,
            .scale = 1.0f / sqrtf((float)state_dim),
        };
        int owned = 0;
        id<MTLCommandBuffer> cb = ds4_gpu_command_buffer(&owned);
        if (!cb) return 0;
        id<MTLComputeCommandEncoder> enc = ds4_gpu_compute_encoder(cb);
        [enc setComputePipelineState:convp];
        [enc setBytes:&args length:sizeof(args) atIndex:0];
        [enc setBuffer:ds4_gpu_tensor_buffer(qkv)
                offset:ds4_gpu_tensor_offset(qkv) + channels_row_offset atIndex:1];
        [enc setBuffer:convbuf offset:(NSUInteger)conv_inner atIndex:2];
        [enc setBuffer:ds4_gpu_tensor_buffer(conv_state) offset:ds4_gpu_tensor_offset(conv_state) atIndex:3];
        if (preserve_state) {
            [enc setBuffer:ds4_gpu_tensor_buffer(middle_conv_state)
                    offset:ds4_gpu_tensor_offset(middle_conv_state) atIndex:4];
            if (preserve_four) {
                [enc setBuffer:ds4_gpu_tensor_buffer(penultimate_conv_state)
                        offset:ds4_gpu_tensor_offset(penultimate_conv_state) atIndex:5];
                [enc setBuffer:ds4_gpu_tensor_buffer(final_conv_state)
                        offset:ds4_gpu_tensor_offset(final_conv_state) atIndex:6];
                [enc setBuffer:ds4_gpu_tensor_buffer(prepared)
                        offset:ds4_gpu_tensor_offset(prepared) atIndex:7];
            } else {
                [enc setBuffer:ds4_gpu_tensor_buffer(final_conv_state)
                        offset:ds4_gpu_tensor_offset(final_conv_state) atIndex:5];
                [enc setBuffer:ds4_gpu_tensor_buffer(prepared)
                        offset:ds4_gpu_tensor_offset(prepared) atIndex:6];
            }
        } else {
            [enc setBuffer:ds4_gpu_tensor_buffer(prepared)
                    offset:ds4_gpu_tensor_offset(prepared) + channels_row_offset atIndex:4];
        }
        if (fused_decode_params) {
            [enc setBuffer:ds4_gpu_tensor_buffer(alpha)
                    offset:ds4_gpu_tensor_offset(alpha) + param_row_offset atIndex:5];
            [enc setBuffer:ds4_gpu_tensor_buffer(beta)
                    offset:ds4_gpu_tensor_offset(beta) + param_row_offset atIndex:6];
            [enc setBuffer:dtbuf offset:(NSUInteger)dt_inner atIndex:7];
            [enc setBuffer:abuf offset:(NSUInteger)a_inner atIndex:8];
            [enc setBuffer:ds4_gpu_tensor_buffer(g)
                    offset:ds4_gpu_tensor_offset(g) + param_row_offset atIndex:9];
            [enc setBuffer:ds4_gpu_tensor_buffer(b)
                    offset:ds4_gpu_tensor_offset(b) + param_row_offset atIndex:10];
        }
        [enc dispatchThreadgroups:MTLSizeMake(((NSUInteger)channels + 255u) / 256u, 1, 1)
             threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        ds4_gpu_end_compute_encoder(cb, enc);

        enc = ds4_gpu_compute_encoder(cb);
        [enc setComputePipelineState:normp];
        [enc setBytes:&args length:sizeof(args) atIndex:0];
        [enc setBuffer:ds4_gpu_tensor_buffer(prepared)
                offset:ds4_gpu_tensor_offset(prepared) + channels_row_offset atIndex:1];
        [enc setThreadgroupMemoryLength:(NSUInteger)state_dim * sizeof(float) atIndex:0];
        [enc dispatchThreadgroups:MTLSizeMake(2u * qk_heads, n_tokens, 1)
             threadsPerThreadgroup:MTLSizeMake(state_dim, 1, 1)];
        ds4_gpu_end_compute_encoder(cb, enc);

        if (!fused_decode_params) {
            enc = ds4_gpu_compute_encoder(cb);
            [enc setComputePipelineState:paramp];
            [enc setBytes:&args length:sizeof(args) atIndex:0];
            [enc setBuffer:ds4_gpu_tensor_buffer(alpha)
                    offset:ds4_gpu_tensor_offset(alpha) + param_row_offset atIndex:1];
            [enc setBuffer:ds4_gpu_tensor_buffer(beta)
                    offset:ds4_gpu_tensor_offset(beta) + param_row_offset atIndex:2];
            [enc setBuffer:dtbuf offset:(NSUInteger)dt_inner atIndex:3];
            [enc setBuffer:abuf offset:(NSUInteger)a_inner atIndex:4];
            [enc setBuffer:ds4_gpu_tensor_buffer(g)
                    offset:ds4_gpu_tensor_offset(g) + param_row_offset atIndex:5];
            [enc setBuffer:ds4_gpu_tensor_buffer(b)
                    offset:ds4_gpu_tensor_offset(b) + param_row_offset atIndex:6];
            const NSUInteger param_total = (NSUInteger)n_tokens * v_heads;
            const NSUInteger param_threads = n_tokens == 1u ? v_heads : 256u;
            [enc dispatchThreadgroups:MTLSizeMake(
                    (param_total + param_threads - 1u) / param_threads, 1, 1)
                 threadsPerThreadgroup:MTLSizeMake(param_threads, 1, 1)];
            ds4_gpu_end_compute_encoder(cb, enc);
        }

        if (!wy_chunkwise) {
            enc = ds4_gpu_compute_encoder(cb);
            [enc setComputePipelineState:recp];
            [enc setBytes:&args length:sizeof(args) atIndex:0];
            [enc setBuffer:ds4_gpu_tensor_buffer(prepared)
                    offset:ds4_gpu_tensor_offset(prepared) + channels_row_offset atIndex:1];
            [enc setBuffer:ds4_gpu_tensor_buffer(g)
                    offset:ds4_gpu_tensor_offset(g) + param_row_offset atIndex:2];
            [enc setBuffer:ds4_gpu_tensor_buffer(b)
                    offset:ds4_gpu_tensor_offset(b) + param_row_offset atIndex:3];
            [enc setBuffer:ds4_gpu_tensor_buffer(ssm_state) offset:ds4_gpu_tensor_offset(ssm_state) atIndex:4];
            if (preserve_state) {
                [enc setBuffer:ds4_gpu_tensor_buffer(middle_ssm_state)
                        offset:ds4_gpu_tensor_offset(middle_ssm_state) atIndex:5];
                if (preserve_four) {
                    [enc setBuffer:ds4_gpu_tensor_buffer(penultimate_ssm_state)
                            offset:ds4_gpu_tensor_offset(penultimate_ssm_state) atIndex:6];
                    [enc setBuffer:ds4_gpu_tensor_buffer(final_ssm_state)
                            offset:ds4_gpu_tensor_offset(final_ssm_state) atIndex:7];
                    [enc setBuffer:ds4_gpu_tensor_buffer(recurrent_out)
                            offset:ds4_gpu_tensor_offset(recurrent_out) atIndex:8];
                } else {
                    [enc setBuffer:ds4_gpu_tensor_buffer(final_ssm_state)
                            offset:ds4_gpu_tensor_offset(final_ssm_state) atIndex:6];
                    [enc setBuffer:ds4_gpu_tensor_buffer(recurrent_out)
                            offset:ds4_gpu_tensor_offset(recurrent_out) atIndex:7];
                }
            } else {
                [enc setBuffer:ds4_gpu_tensor_buffer(recurrent_out)
                        offset:ds4_gpu_tensor_offset(recurrent_out) +
                               inner_row_offset atIndex:5];
            }
            [enc dispatchThreadgroups:MTLSizeMake(state_dim / 4u, v_heads, 1)
                 threadsPerThreadgroup:MTLSizeMake(32, 4, 1)];
            ds4_gpu_end_compute_encoder(cb, enc);
        } else {
            const NSUInteger tiles = n_tokens / 8u;
            enc = ds4_gpu_compute_encoder(cb);
            [enc setComputePipelineState:wy_packp];
            [enc setBytes:&args length:sizeof(args) atIndex:0];
            [enc setBuffer:ds4_gpu_tensor_buffer(prepared)
                    offset:ds4_gpu_tensor_offset(prepared) + channels_row_offset atIndex:1];
            [enc setBuffer:ds4_gpu_tensor_buffer(g)
                    offset:ds4_gpu_tensor_offset(g) + param_row_offset atIndex:2];
            [enc setBuffer:ds4_gpu_tensor_buffer(b)
                    offset:ds4_gpu_tensor_offset(b) + param_row_offset atIndex:3];
            [enc setBuffer:ds4_gpu_tensor_buffer(chunk_w)
                    offset:ds4_gpu_tensor_offset(chunk_w) + chunk_inner_offset atIndex:4];
            [enc setBuffer:ds4_gpu_tensor_buffer(chunk_u)
                    offset:ds4_gpu_tensor_offset(chunk_u) + chunk_inner_offset atIndex:5];
            [enc setBuffer:ds4_gpu_tensor_buffer(chunk_cumulative_g)
                    offset:ds4_gpu_tensor_offset(chunk_cumulative_g) + chunk_param_offset atIndex:6];
            [enc setThreadgroupMemoryLength:(NSUInteger)n_tokens * sizeof(float) atIndex:0];
            [enc dispatchThreadgroups:MTLSizeMake(v_heads, 1, 1)
                 threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
            ds4_gpu_end_compute_encoder(cb, enc);

            enc = ds4_gpu_compute_encoder(cb);
            [enc setComputePipelineState:wy_kkt_qkp];
            [enc setBytes:&args length:sizeof(args) atIndex:0];
            [enc setBuffer:ds4_gpu_tensor_buffer(chunk_u)
                    offset:ds4_gpu_tensor_offset(chunk_u) + chunk_inner_offset atIndex:1];
            [enc setBuffer:ds4_gpu_tensor_buffer(recurrent_out)
                    offset:ds4_gpu_tensor_offset(recurrent_out) + inner_row_offset atIndex:2];
            [enc dispatchThreadgroups:MTLSizeMake(tiles * (tiles + 1u) / 2u,
                                                  1u, qk_heads)
                 threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
            ds4_gpu_end_compute_encoder(cb, enc);

            enc = ds4_gpu_compute_encoder(cb);
            [enc setComputePipelineState:wy_solvep];
            [enc setBytes:&args length:sizeof(args) atIndex:0];
            [enc setBuffer:ds4_gpu_tensor_buffer(b)
                    offset:ds4_gpu_tensor_offset(b) + param_row_offset atIndex:1];
            [enc setBuffer:ds4_gpu_tensor_buffer(chunk_cumulative_g)
                    offset:ds4_gpu_tensor_offset(chunk_cumulative_g) + chunk_param_offset atIndex:2];
            [enc setBuffer:ds4_gpu_tensor_buffer(chunk_qk) offset:ds4_gpu_tensor_offset(chunk_qk) atIndex:3];
            [enc setBuffer:ds4_gpu_tensor_buffer(recurrent_out)
                    offset:ds4_gpu_tensor_offset(recurrent_out) + inner_row_offset atIndex:4];
            const NSUInteger solve_simdgroups = 8u;
            const NSUInteger solve_bytes =
                (2u * (NSUInteger)n_tokens * n_tokens +
                 solve_simdgroups * 64u) * sizeof(uint16_t) +
                solve_simdgroups * 64u * sizeof(float);
            [enc setThreadgroupMemoryLength:solve_bytes atIndex:0];
            [enc dispatchThreadgroups:MTLSizeMake(v_heads, 1, 1)
                 threadsPerThreadgroup:MTLSizeMake(solve_simdgroups * 32u, 1, 1)];
            ds4_gpu_end_compute_encoder(cb, enc);

            enc = ds4_gpu_compute_encoder(cb);
            [enc setComputePipelineState:wy_valuesp];
            [enc setBytes:&args length:sizeof(args) atIndex:0];
            [enc setBuffer:ds4_gpu_tensor_buffer(chunk_qk) offset:ds4_gpu_tensor_offset(chunk_qk) atIndex:1];
            [enc setBuffer:ds4_gpu_tensor_buffer(chunk_w)
                    offset:ds4_gpu_tensor_offset(chunk_w) + chunk_inner_offset atIndex:2];
            [enc setBuffer:ds4_gpu_tensor_buffer(chunk_u)
                    offset:ds4_gpu_tensor_offset(chunk_u) + chunk_inner_offset atIndex:3];
            [enc setBuffer:ds4_gpu_tensor_buffer(ssm_state) offset:ds4_gpu_tensor_offset(ssm_state) atIndex:4];
            [enc setBuffer:ds4_gpu_tensor_buffer(recurrent_out)
                    offset:ds4_gpu_tensor_offset(recurrent_out) + inner_row_offset atIndex:5];
            [enc setBuffer:ds4_gpu_tensor_buffer(out)
                    offset:ds4_gpu_tensor_offset(out) + inner_row_offset atIndex:6];
            [enc setBuffer:ds4_gpu_tensor_buffer(chunk_cumulative_g)
                    offset:ds4_gpu_tensor_offset(chunk_cumulative_g) + chunk_param_offset atIndex:7];
            const NSUInteger direct_value_bytes =
                MAX(8u * 32u, (NSUInteger)n_tokens * 8u) *
                    sizeof(uint16_t) +
                (NSUInteger)n_tokens * 32u *
                    (sizeof(float) + sizeof(uint16_t));
            [enc setThreadgroupMemoryLength:direct_value_bytes atIndex:0];
            [enc dispatchThreadgroups:MTLSizeMake(state_dim / 32u, v_heads, 1)
                 threadsPerThreadgroup:MTLSizeMake(tiles * 32u, 1, 1)];
            ds4_gpu_end_compute_encoder(cb, enc);
        }

        enc = ds4_gpu_compute_encoder(cb);
        [enc setComputePipelineState:postp];
        [enc setBytes:&args length:sizeof(args) atIndex:0];
        [enc setBuffer:ds4_gpu_tensor_buffer(wy_chunkwise ? out : recurrent_out)
                offset:ds4_gpu_tensor_offset(wy_chunkwise ? out : recurrent_out) +
                       inner_row_offset atIndex:1];
        [enc setBuffer:ds4_gpu_tensor_buffer(z)
                offset:ds4_gpu_tensor_offset(z) + inner_row_offset atIndex:2];
        [enc setBuffer:normbuf offset:(NSUInteger)norm_inner atIndex:3];
        [enc setBuffer:ds4_gpu_tensor_buffer(out)
                offset:ds4_gpu_tensor_offset(out) + inner_row_offset atIndex:4];
        [enc setThreadgroupMemoryLength:(NSUInteger)state_dim * sizeof(float) atIndex:0];
        [enc dispatchThreadgroups:MTLSizeMake(v_heads, n_tokens, 1)
             threadsPerThreadgroup:MTLSizeMake(state_dim, 1, 1)];
        ds4_gpu_end_compute_encoder(cb, enc);
        return ds4_gpu_finish_command_buffer(cb, owned, "Qwen3.6 Gated DeltaNet");
    }
}

int ds4_gpu_qwen35_gdn_batch_tensor(
        ds4_gpu_tensor       *out,
        ds4_gpu_tensor       *prepared,
        ds4_gpu_tensor       *g,
        ds4_gpu_tensor       *b,
        ds4_gpu_tensor       *recurrent_out,
        ds4_gpu_tensor       *conv_state,
        ds4_gpu_tensor       *ssm_state,
        const ds4_gpu_tensor *qkv,
        const ds4_gpu_tensor *z,
        const ds4_gpu_tensor *alpha,
        const ds4_gpu_tensor *beta,
        const void           *model_map,
        uint64_t              model_size,
        uint64_t              conv_weight_offset,
        uint64_t              dt_bias_offset,
        uint64_t              a_offset,
        uint64_t              norm_offset,
        uint32_t              n_tokens,
        uint32_t              channels,
        uint32_t              qk_heads,
        uint32_t              v_heads,
        uint32_t              state_dim,
        uint32_t              conv_width,
        float                 eps) {
    return ds4_gpu_qwen35_gdn_batch_impl(
        out, prepared, g, b, recurrent_out, conv_state, ssm_state,
        NULL, NULL, NULL, NULL, NULL, NULL,
        qkv, z, alpha, beta, model_map, model_size,
        conv_weight_offset, dt_bias_offset, a_offset, norm_offset,
        n_tokens, channels, qk_heads, v_heads, state_dim, conv_width, eps,
        NULL, NULL, NULL, NULL, 0, 0u);
}

int ds4_gpu_qwen35_gdn_batch_preserve_first_tensor(
        ds4_gpu_tensor       *out,
        ds4_gpu_tensor       *prepared,
        ds4_gpu_tensor       *g,
        ds4_gpu_tensor       *b,
        ds4_gpu_tensor       *recurrent_out,
        ds4_gpu_tensor       *conv_state,
        ds4_gpu_tensor       *ssm_state,
        ds4_gpu_tensor       *middle_conv_state,
        ds4_gpu_tensor       *middle_ssm_state,
        ds4_gpu_tensor       *final_conv_state,
        ds4_gpu_tensor       *final_ssm_state,
        const ds4_gpu_tensor *qkv,
        const ds4_gpu_tensor *z,
        const ds4_gpu_tensor *alpha,
        const ds4_gpu_tensor *beta,
        const void           *model_map,
        uint64_t              model_size,
        uint64_t              conv_weight_offset,
        uint64_t              dt_bias_offset,
        uint64_t              a_offset,
        uint64_t              norm_offset,
        uint32_t              n_tokens,
        uint32_t              channels,
        uint32_t              qk_heads,
        uint32_t              v_heads,
        uint32_t              state_dim,
        uint32_t              conv_width,
        float                 eps) {
    return ds4_gpu_qwen35_gdn_batch_impl(
        out, prepared, g, b, recurrent_out, conv_state, ssm_state,
        middle_conv_state, middle_ssm_state,
        NULL, NULL,
        final_conv_state, final_ssm_state,
        qkv, z, alpha, beta, model_map, model_size,
        conv_weight_offset, dt_bias_offset, a_offset, norm_offset,
        n_tokens, channels, qk_heads, v_heads, state_dim, conv_width, eps,
        NULL, NULL, NULL, NULL, 0, 0u);
}

int ds4_gpu_qwen35_gdn_batch_preserve_four_tensor(
        ds4_gpu_tensor *out, ds4_gpu_tensor *prepared,
        ds4_gpu_tensor *g, ds4_gpu_tensor *b,
        ds4_gpu_tensor *recurrent_out,
        ds4_gpu_tensor *conv_state, ds4_gpu_tensor *ssm_state,
        ds4_gpu_tensor *middle_conv_state,
        ds4_gpu_tensor *middle_ssm_state,
        ds4_gpu_tensor *penultimate_conv_state,
        ds4_gpu_tensor *penultimate_ssm_state,
        ds4_gpu_tensor *final_conv_state,
        ds4_gpu_tensor *final_ssm_state,
        const ds4_gpu_tensor *qkv, const ds4_gpu_tensor *z,
        const ds4_gpu_tensor *alpha, const ds4_gpu_tensor *beta,
        const void *model_map, uint64_t model_size,
        uint64_t conv_weight_offset, uint64_t dt_bias_offset,
        uint64_t a_offset, uint64_t norm_offset,
        uint32_t channels, uint32_t qk_heads, uint32_t v_heads,
        uint32_t state_dim, uint32_t conv_width, float eps) {
    return ds4_gpu_qwen35_gdn_batch_impl(
        out, prepared, g, b, recurrent_out, conv_state, ssm_state,
        middle_conv_state, middle_ssm_state,
        penultimate_conv_state, penultimate_ssm_state,
        final_conv_state, final_ssm_state,
        qkv, z, alpha, beta, model_map, model_size,
        conv_weight_offset, dt_bias_offset, a_offset, norm_offset,
        4u, channels, qk_heads, v_heads, state_dim, conv_width, eps,
        NULL, NULL, NULL, NULL, 0, 0u);
}

int ds4_gpu_qwen35_gdn_chunk_offset_tensor(
        ds4_gpu_tensor       *out,
        ds4_gpu_tensor       *prepared,
        ds4_gpu_tensor       *g,
        ds4_gpu_tensor       *b,
        ds4_gpu_tensor       *values,
        ds4_gpu_tensor       *chunk_w,
        ds4_gpu_tensor       *chunk_u,
        ds4_gpu_tensor       *chunk_qk,
        ds4_gpu_tensor       *chunk_cumulative_g,
        ds4_gpu_tensor       *conv_state,
        ds4_gpu_tensor       *ssm_state,
        const ds4_gpu_tensor *qkv,
        const ds4_gpu_tensor *z,
        const ds4_gpu_tensor *alpha,
        const ds4_gpu_tensor *beta,
        const void           *model_map,
        uint64_t              model_size,
        uint64_t              conv_weight_offset,
        uint64_t              dt_bias_offset,
        uint64_t              a_offset,
        uint64_t              norm_offset,
        uint32_t              row_offset,
        uint32_t              n_tokens,
        uint32_t              channels,
        uint32_t              qk_heads,
        uint32_t              v_heads,
        uint32_t              state_dim,
        uint32_t              conv_width,
        float                 eps) {
    return ds4_gpu_qwen35_gdn_batch_impl(
        out, prepared, g, b, values, conv_state, ssm_state,
        NULL, NULL, NULL, NULL, NULL, NULL,
        qkv, z, alpha, beta, model_map, model_size,
        conv_weight_offset, dt_bias_offset, a_offset, norm_offset,
        n_tokens, channels, qk_heads, v_heads, state_dim, conv_width, eps,
        chunk_w, chunk_u, chunk_qk, chunk_cumulative_g, 1, row_offset);
}

int ds4_gpu_qwen35_gdn_batch_offset_tensor(
        ds4_gpu_tensor       *out,
        ds4_gpu_tensor       *prepared,
        ds4_gpu_tensor       *g,
        ds4_gpu_tensor       *b,
        ds4_gpu_tensor       *values,
        ds4_gpu_tensor       *conv_state,
        ds4_gpu_tensor       *ssm_state,
        const ds4_gpu_tensor *qkv,
        const ds4_gpu_tensor *z,
        const ds4_gpu_tensor *alpha,
        const ds4_gpu_tensor *beta,
        const void           *model_map,
        uint64_t              model_size,
        uint64_t              conv_weight_offset,
        uint64_t              dt_bias_offset,
        uint64_t              a_offset,
        uint64_t              norm_offset,
        uint32_t              row_offset,
        uint32_t              n_tokens,
        uint32_t              channels,
        uint32_t              qk_heads,
        uint32_t              v_heads,
        uint32_t              state_dim,
        uint32_t              conv_width,
        float                 eps) {
    return ds4_gpu_qwen35_gdn_batch_impl(
        out, prepared, g, b, values, conv_state, ssm_state,
        NULL, NULL, NULL, NULL, NULL, NULL,
        qkv, z, alpha, beta, model_map, model_size,
        conv_weight_offset, dt_bias_offset, a_offset, norm_offset,
        n_tokens, channels, qk_heads, v_heads, state_dim, conv_width, eps,
        NULL, NULL, NULL, NULL, 0, row_offset);
}

/* TensorOps routed-MoE prefill uses bits 0/1/2 for gate/up/down. Unsupported
 * tensor types keep their established kernels. --quality and the global
 * Metal4 comparison switch retain the reference path. */

static int ds4_gpu_glm_attention_flash_tensor_impl(
        ds4_gpu_tensor       *heads,
        const ds4_gpu_tensor *q,
        const ds4_gpu_tensor *key_cache,
        const ds4_gpu_tensor *value_cache,
        uint32_t              pos0,
        uint32_t              n_tokens,
        uint32_t              cache_len,
        uint32_t              cache_cap,
        uint32_t              n_head,
        uint32_t              n_head_kv,
        uint32_t              qk_dim,
        uint32_t              value_dim,
        bool                  cache_f16,
        int                   kv_pre_staged) {
    if (!g_initialized && !ds4_gpu_init()) return 0;
    if (!heads || !q || !key_cache || !value_cache ||
        n_tokens == 0 || cache_len == 0 || cache_cap == 0 ||
        n_head == 0 || n_head_kv == 0 || n_head % n_head_kv != 0 ||
        qk_dim != 256u || value_dim != 256u ||
        cache_len > cache_cap ||
        pos0 > cache_len || n_tokens > cache_len - pos0) {
        return 0;
    }

    @autoreleasepool {
        id<MTLBuffer> headsbuf = ds4_gpu_tensor_buffer(heads);
        id<MTLBuffer> qbuf = ds4_gpu_tensor_buffer(q);
        id<MTLBuffer> keybuf = ds4_gpu_tensor_buffer(key_cache);
        id<MTLBuffer> valbuf = ds4_gpu_tensor_buffer(value_cache);
        const uint64_t heads_bytes = (uint64_t)n_tokens * n_head * value_dim * sizeof(float);
        const uint64_t q_bytes = (uint64_t)n_tokens * n_head * qk_dim * sizeof(float);
        const uint64_t cache_elem_bytes = cache_f16 ? sizeof(uint16_t) : sizeof(float);
        const uint64_t key_bytes = (uint64_t)cache_cap * n_head_kv * qk_dim * cache_elem_bytes;
        const uint64_t value_bytes = (uint64_t)cache_cap * n_head_kv * value_dim * cache_elem_bytes;
        if (!headsbuf || !qbuf || !keybuf || !valbuf ||
            ds4_gpu_tensor_bytes(heads) < heads_bytes ||
            ds4_gpu_tensor_bytes(q) < q_bytes ||
            ds4_gpu_tensor_bytes(key_cache) < key_bytes ||
            ds4_gpu_tensor_bytes(value_cache) < value_bytes) {
            fprintf(stderr, "ds4: Metal GLM FlashAttention received undersized buffers\n");
            return 0;
        }
        const uint64_t key_elems = (uint64_t)cache_len * n_head_kv * qk_dim;
        const uint64_t value_elems = (uint64_t)cache_len * n_head_kv * value_dim;
        if (key_elems > UINT32_MAX || value_elems > UINT32_MAX) {
            return 0;
        }

        const uint32_t nqptg = 8;
        const uint32_t ncpsg = 64;
        const uint32_t nsg = 4;
        const bool has_kvpad = (cache_len % ncpsg) != 0;
        const bool bc_mask = (n_tokens % nqptg) != 0;
        const NSUInteger q_row_bytes = (NSUInteger)qk_dim * sizeof(float);
        const NSUInteger q_row_bytes_f16 = (NSUInteger)qk_dim * sizeof(uint16_t);
        const NSUInteger v_row_bytes = (NSUInteger)value_dim * sizeof(float);
        const NSUInteger v_row_bytes_f16 = (NSUInteger)value_dim * sizeof(uint16_t);
        const NSUInteger mask_bytes = (NSUInteger)n_tokens * (NSUInteger)cache_len * sizeof(uint16_t);
        const NSUInteger key_f16_offset = 0;
        const NSUInteger key_f16_bytes =
            (NSUInteger)cache_len * (NSUInteger)n_head_kv * q_row_bytes_f16;
        const NSUInteger value_f16_offset = key_f16_bytes;
        const NSUInteger value_f16_bytes =
            (NSUInteger)cache_len * (NSUInteger)n_head_kv * v_row_bytes_f16;
        const NSUInteger kv_f16_bytes = key_f16_bytes + value_f16_bytes;
        const NSUInteger pad_bytes = has_kvpad
            ? (NSUInteger)ncpsg * ((NSUInteger)n_head_kv * (q_row_bytes_f16 + v_row_bytes_f16) +
                                   (NSUInteger)n_tokens * sizeof(uint16_t))
            : 1u;
        const NSUInteger nblk0 = ((NSUInteger)cache_len + ncpsg - 1u) / ncpsg;
        const NSUInteger nblk1 = ((NSUInteger)n_tokens + nqptg - 1u) / nqptg;
        const NSUInteger blk_bytes = ds4_gpu_align_up_ns(nblk0 * nblk1, 32u);

        id<MTLBuffer> mask_buffer =
            ds4_gpu_glm_prefill_mask_buffer(pos0, n_tokens, cache_len, mask_bytes);
        if (!mask_buffer) return 0;
        if (kv_pre_staged) {
            if (!g_flash_attn_kv_buffer || g_flash_attn_kv_bytes < kv_f16_bytes) {
                fprintf(stderr, "ds4: GLM staged FlashAttention KV scratch is missing\n");
                return 0;
            }
        } else if (!ds4_gpu_ensure_scratch_buffer(&g_flash_attn_kv_buffer,
                                                  &g_flash_attn_kv_bytes,
                                                  kv_f16_bytes,
                                                  "ds4_glm_flash_attn_kv_f16")) {
            return 0;
        }
        if (!ds4_gpu_ensure_scratch_buffer(&g_flash_attn_pad_buffer,
                                             &g_flash_attn_pad_bytes,
                                             pad_bytes,
                                             "ds4_glm_flash_attn_pad") ||
            !ds4_gpu_ensure_scratch_buffer(&g_flash_attn_blk_buffer,
                                             &g_flash_attn_blk_bytes,
                                             blk_bytes,
                                             "ds4_glm_flash_attn_blk")) {
            return 0;
        }

        id<MTLComputePipelineState> pad_pipeline = nil;
        if (has_kvpad) {
            pad_pipeline = ds4_gpu_get_flash_attn_pad_pipeline(true, (int32_t)ncpsg);
            if (!pad_pipeline) return 0;
        }
        id<MTLComputePipelineState> blk_pipeline =
            ds4_gpu_get_flash_attn_blk_pipeline((int32_t)nqptg, (int32_t)ncpsg);
        id<MTLComputePipelineState> attn_pipeline =
            ds4_gpu_get_flash_attn_pipeline("kernel_flash_attn_ext_f16_dk256_dv256",
                                              true, false, false, false, has_kvpad, bc_mask,
                                              (int32_t)qk_dim,
                                              (int32_t)value_dim,
                                              (int32_t)nsg);
        if (!blk_pipeline || !attn_pipeline) return 0;

        int owned = 0;
        id<MTLCommandBuffer> cb = ds4_gpu_command_buffer(&owned);
        if (!cb) return 0;

        if (!kv_pre_staged) {
            const bool copied = cache_f16 ?
                (ds4_gpu_encode_cpy_f16_f16_3d(cb,
                                               keybuf,
                                               ds4_gpu_tensor_offset(key_cache),
                                               g_flash_attn_kv_buffer,
                                               key_f16_offset,
                                               qk_dim,
                                               cache_len,
                                               n_head_kv,
                                               (uint64_t)n_head_kv * q_row_bytes_f16,
                                               q_row_bytes_f16,
                                               q_row_bytes_f16,
                                               (uint64_t)cache_len * q_row_bytes_f16) &&
                 ds4_gpu_encode_cpy_f16_f16_3d(cb,
                                               valbuf,
                                               ds4_gpu_tensor_offset(value_cache),
                                               g_flash_attn_kv_buffer,
                                               value_f16_offset,
                                               value_dim,
                                               cache_len,
                                               n_head_kv,
                                               (uint64_t)n_head_kv * v_row_bytes_f16,
                                               v_row_bytes_f16,
                                               v_row_bytes_f16,
                                               (uint64_t)cache_len * v_row_bytes_f16)) :
                (ds4_gpu_encode_cpy_f32_f16_3d(cb,
                                               keybuf,
                                               ds4_gpu_tensor_offset(key_cache),
                                               g_flash_attn_kv_buffer,
                                               key_f16_offset,
                                               qk_dim,
                                               cache_len,
                                               n_head_kv,
                                               (uint64_t)n_head_kv * q_row_bytes,
                                               q_row_bytes,
                                               q_row_bytes_f16,
                                               (uint64_t)cache_len * q_row_bytes_f16) &&
                 ds4_gpu_encode_cpy_f32_f16_3d(cb,
                                               valbuf,
                                               ds4_gpu_tensor_offset(value_cache),
                                               g_flash_attn_kv_buffer,
                                               value_f16_offset,
                                               value_dim,
                                               cache_len,
                                               n_head_kv,
                                               (uint64_t)n_head_kv * v_row_bytes,
                                               v_row_bytes,
                                               v_row_bytes_f16,
                                               (uint64_t)cache_len * v_row_bytes_f16));
            if (!copied) {
                return 0;
            }
        }

        if (has_kvpad) {
            ds4_gpu_flash_attn_pad_args pad_args = {
                .ne11 = (int32_t)cache_len,
                .ne_12_2 = (int32_t)n_head_kv,
                .ne_12_3 = 1,
                .nb11 = q_row_bytes_f16,
                .nb12 = (uint64_t)cache_len * q_row_bytes_f16,
                .nb13 = (uint64_t)cache_len * (uint64_t)n_head_kv * q_row_bytes_f16,
                .nb21 = v_row_bytes_f16,
                .nb22 = (uint64_t)cache_len * v_row_bytes_f16,
                .nb23 = (uint64_t)cache_len * (uint64_t)n_head_kv * v_row_bytes_f16,
                .ne31 = (int32_t)n_tokens,
                .ne32 = 1,
                .ne33 = 1,
                .nb31 = (uint64_t)cache_len * sizeof(uint16_t),
                .nb32 = mask_bytes,
                .nb33 = mask_bytes,
            };

            id<MTLComputeCommandEncoder> enc = ds4_gpu_compute_encoder(cb);
            [enc setComputePipelineState:pad_pipeline];
            [enc setBytes:&pad_args length:sizeof(pad_args) atIndex:0];
            [enc setBuffer:g_flash_attn_kv_buffer offset:key_f16_offset atIndex:1];
            [enc setBuffer:g_flash_attn_kv_buffer offset:value_f16_offset atIndex:2];
            [enc setBuffer:mask_buffer offset:0 atIndex:3];
            [enc setBuffer:g_flash_attn_pad_buffer offset:0 atIndex:4];
            [enc dispatchThreadgroups:MTLSizeMake(ncpsg, n_head_kv, 1)
                 threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
            ds4_gpu_end_compute_encoder(cb, enc);
        }

        ds4_gpu_flash_attn_blk_args blk_args = {
            .ne01 = (int32_t)n_tokens,
            .ne30 = (int32_t)cache_len,
            .ne31 = (int32_t)n_tokens,
            .ne32 = 1,
            .ne33 = 1,
            .nb31 = (uint64_t)cache_len * sizeof(uint16_t),
            .nb32 = mask_bytes,
            .nb33 = mask_bytes,
        };

        id<MTLComputeCommandEncoder> enc = ds4_gpu_compute_encoder(cb);
        [enc setComputePipelineState:blk_pipeline];
        [enc setBytes:&blk_args length:sizeof(blk_args) atIndex:0];
        [enc setBuffer:mask_buffer offset:0 atIndex:1];
        [enc setBuffer:g_flash_attn_blk_buffer offset:0 atIndex:2];
        [enc dispatchThreadgroups:MTLSizeMake(nblk0, nblk1, 1)
             threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
        ds4_gpu_end_compute_encoder(cb, enc);

        ds4_gpu_flash_attn_vec_args args = {
            .ne01 = (int32_t)n_tokens,
            .ne02 = (int32_t)n_head,
            .ne03 = 1,
            .nb01 = (uint64_t)n_head * q_row_bytes,
            .nb02 = q_row_bytes,
            .nb03 = (uint64_t)n_tokens * n_head * q_row_bytes,
            .ne11 = (int32_t)cache_len,
            .ne_12_2 = (int32_t)n_head_kv,
            .ne_12_3 = 1,
            .ns10 = (int32_t)qk_dim,
            .nb11 = q_row_bytes_f16,
            .nb12 = (uint64_t)cache_len * q_row_bytes_f16,
            .nb13 = (uint64_t)cache_len * (uint64_t)n_head_kv * q_row_bytes_f16,
            .ns20 = (int32_t)value_dim,
            .nb21 = v_row_bytes_f16,
            .nb22 = (uint64_t)cache_len * v_row_bytes_f16,
            .nb23 = (uint64_t)cache_len * (uint64_t)n_head_kv * v_row_bytes_f16,
            .ne31 = (int32_t)n_tokens,
            .ne32 = 1,
            .ne33 = 1,
            .nb31 = (uint64_t)cache_len * sizeof(uint16_t),
            .nb32 = mask_bytes,
            .nb33 = mask_bytes,
            .ne1 = (int32_t)n_head,
            .ne2 = (int32_t)n_tokens,
            .ne3 = 1,
            .scale = 1.0f / sqrtf((float)qk_dim),
            .max_bias = 0.0f,
            .m0 = 0.0f,
            .m1 = 0.0f,
            .n_head_log2 = 0,
            .logit_softcap = 0.0f,
        };

        const NSUInteger padded_v = ds4_gpu_align_up_ns(value_dim, 64u);
        const NSUInteger shared_elems = (NSUInteger)nqptg *
            ((NSUInteger)qk_dim + 2u * padded_v + 2u * (2u * (NSUInteger)ncpsg));
        const NSUInteger shared_bytes = ds4_gpu_align_up_ns(shared_elems * (sizeof(float) / 2u), 16u);

        enc = ds4_gpu_compute_encoder(cb);
        [enc setComputePipelineState:attn_pipeline];
        [enc setBytes:&args length:sizeof(args) atIndex:0];
        [enc setBuffer:qbuf offset:ds4_gpu_tensor_offset(q) atIndex:1];
        [enc setBuffer:g_flash_attn_kv_buffer offset:key_f16_offset atIndex:2];
        [enc setBuffer:g_flash_attn_kv_buffer offset:value_f16_offset atIndex:3];
        [enc setBuffer:mask_buffer offset:0 atIndex:4];
        [enc setBuffer:qbuf offset:ds4_gpu_tensor_offset(q) atIndex:5];
        [enc setBuffer:g_flash_attn_pad_buffer offset:0 atIndex:6];
        [enc setBuffer:g_flash_attn_blk_buffer offset:0 atIndex:7];
        [enc setBuffer:headsbuf offset:ds4_gpu_tensor_offset(heads) atIndex:8];
        [enc setThreadgroupMemoryLength:shared_bytes atIndex:0];
        [enc dispatchThreadgroups:MTLSizeMake(nblk1, n_head, 1)
             threadsPerThreadgroup:MTLSizeMake(32, nsg, 1)];
        ds4_gpu_end_compute_encoder(cb, enc);

        if (!ds4_gpu_finish_command_buffer(cb, owned, "GLM FlashAttention")) return 0;
    }

    return 1;
}

int ds4_gpu_qwen35_attention_flash_batch_tensor(
        ds4_gpu_tensor       *out,
        const ds4_gpu_tensor *q,
        const ds4_gpu_tensor *gate,
        const ds4_gpu_tensor *key_cache,
        const ds4_gpu_tensor *value_cache,
        uint32_t              pos0,
        uint32_t              n_tokens,
        uint32_t              cache_cap,
        uint32_t              n_head,
        uint32_t              n_head_kv,
        uint32_t              head_dim) {
    if (!out || !q || !gate || !key_cache || !value_cache ||
        n_tokens == 0 || n_head == 0 || n_head_kv == 0 ||
        n_head % n_head_kv != 0 || head_dim != 256u ||
        pos0 >= cache_cap || n_tokens > cache_cap - pos0) {
        return 0;
    }
    const uint64_t elems =
        (uint64_t)n_tokens * n_head * head_dim;
    const uint64_t bytes = elems * sizeof(float);
    if (elems > UINT32_MAX || ds4_gpu_tensor_bytes(out) < bytes ||
        ds4_gpu_tensor_bytes(gate) < bytes) {
        return 0;
    }
    id<MTLComputePipelineState> gate_pipeline =
        ds4_gpu_get_pipeline("kernel_qwen35_attention_gate");
    if (!gate_pipeline) return 0;

    if (!ds4_gpu_glm_attention_flash_tensor_impl(
            out, q, key_cache, value_cache,
            pos0, n_tokens, pos0 + n_tokens, cache_cap,
            n_head, n_head_kv, head_dim, head_dim, true, 0)) {
        return 0;
    }

    ds4_gpu_qwen35_attention_args args = {
        .pos0 = pos0, .n_tokens = n_tokens, .cache_cap = cache_cap,
        .n_head = n_head, .n_head_kv = n_head_kv,
        .head_dim = head_dim,
        .scale = 1.0f / sqrtf((float)head_dim),
    };
    int owned = 0;
    id<MTLCommandBuffer> cb = ds4_gpu_command_buffer(&owned);
    if (!cb) return 0;
    id<MTLComputeCommandEncoder> enc = ds4_gpu_compute_encoder(cb);
    [enc setComputePipelineState:gate_pipeline];
    [enc setBytes:&args length:sizeof(args) atIndex:0];
    [enc setBuffer:ds4_gpu_tensor_buffer(gate)
            offset:ds4_gpu_tensor_offset(gate) atIndex:1];
    [enc setBuffer:ds4_gpu_tensor_buffer(out)
            offset:ds4_gpu_tensor_offset(out) atIndex:2];
    [enc dispatchThreadgroups:MTLSizeMake(((NSUInteger)elems + 255u) / 256u,
                                          1, 1)
         threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
    ds4_gpu_end_compute_encoder(cb, enc);
    return ds4_gpu_finish_command_buffer(
        cb, owned, "Qwen3.6 FlashAttention gate");
}
