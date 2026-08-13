// Qwen3.6-27B hybrid attention kernels.  The dense Q8 projections use the
// engine's existing matvecs; these kernels implement the model-specific
// Gated DeltaNet recurrence and gated grouped-query attention plumbing.

struct ds4_metal_args_qwen35_full_prepare {
    uint32_t pos0;
    uint32_t n_tokens;
    uint32_t cache_cap;
    uint32_t n_head;
    uint32_t n_head_kv;
    uint32_t head_dim;
    uint32_t rot_dim;
    float eps;
    float freq_base;
};

struct ds4_metal_args_qwen35_attention {
    uint32_t pos0;
    uint32_t n_tokens;
    uint32_t cache_cap;
    uint32_t n_head;
    uint32_t n_head_kv;
    uint32_t head_dim;
    float scale;
};

struct ds4_metal_args_qwen35_gdn {
    uint32_t n_tokens;
    uint32_t channels;
    uint32_t qk_heads;
    uint32_t v_heads;
    uint32_t state_dim;
    uint32_t conv_width;
    float eps;
    float scale;
};

static inline float qwen35_silu(float x) {
    return x / (1.0f + exp(-x));
}

static inline float qwen35_softplus(float x) {
    return max(x, 0.0f) + log(1.0f + exp(-fabs(x)));
}

kernel void kernel_qwen35_full_prepare(
        constant ds4_metal_args_qwen35_full_prepare &args,
        device const float *qg,
        device const float *k,
        device const float *v,
        device const float *q_norm,
        device const float *k_norm,
        device float *q_out,
        device float *gate_out,
        device half *k_cache,
        device half *v_cache,
        threadgroup float *scratch [[threadgroup(0)]],
        uint tid [[thread_index_in_threadgroup]],
        uint3 tgpig [[threadgroup_position_in_grid]]) {
    const uint h = tgpig.x;
    const uint token = tgpig.y;
    if (h >= args.n_head || token >= args.n_tokens || tid >= args.head_dim) return;

    const uint qg_stride = 2u * args.n_head * args.head_dim;
    const uint q_stride = args.n_head * args.head_dim;
    const uint kv_stride = args.n_head_kv * args.head_dim;
    const uint qg_base = token * qg_stride + h * 2u * args.head_dim;
    const uint q_base = token * q_stride + h * args.head_dim;
    const float qx = qg[qg_base + tid];
    scratch[tid] = qx * qx;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint step = args.head_dim >> 1; step != 0; step >>= 1) {
        if (tid < step) scratch[tid] += scratch[tid + step];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    const float q_scale = rsqrt(scratch[0] / (float)args.head_dim + args.eps);
    float qy = qg[qg_base + tid] * q_scale * q_norm[tid];
    if (tid < args.rot_dim) {
        const uint rot_half = args.rot_dim / 2u;
        const uint freq = tid % rot_half;
        const uint other = tid < rot_half ? tid + rot_half : tid - rot_half;
        const float qa = qg[qg_base + tid] * q_scale * q_norm[tid];
        const float qb = qg[qg_base + other] * q_scale * q_norm[other];
        const float theta = (float)(args.pos0 + token) *
            pow(args.freq_base,
                -(2.0f * (float)freq) / (float)args.rot_dim);
        const float c = cos(theta);
        const float s = sin(theta);
        qy = tid < rot_half ? qa * c - qb * s : qa * c + qb * s;
    }
    q_out[q_base + tid] = qy;
    gate_out[q_base + tid] = qg[qg_base + args.head_dim + tid];

    if (h >= args.n_head_kv) return;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const uint kv_base = token * kv_stride + h * args.head_dim;
    const float kx = k[kv_base + tid];
    scratch[tid] = kx * kx;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint step = args.head_dim >> 1; step != 0; step >>= 1) {
        if (tid < step) scratch[tid] += scratch[tid + step];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    const float k_scale = rsqrt(scratch[0] / (float)args.head_dim + args.eps);
    float ky = k[kv_base + tid] * k_scale * k_norm[tid];
    if (tid < args.rot_dim) {
        const uint rot_half = args.rot_dim / 2u;
        const uint freq = tid % rot_half;
        const uint other = tid < rot_half ? tid + rot_half : tid - rot_half;
        const float ka = k[kv_base + tid] * k_scale * k_norm[tid];
        const float kb = k[kv_base + other] * k_scale * k_norm[other];
        const float theta = (float)(args.pos0 + token) *
            pow(args.freq_base,
                -(2.0f * (float)freq) / (float)args.rot_dim);
        const float c = cos(theta);
        const float s = sin(theta);
        ky = tid < rot_half ? ka * c - kb * s : ka * c + kb * s;
    }
    const ulong cache_base =
        ((ulong)(args.pos0 + token) * args.n_head_kv + h) * args.head_dim;
    k_cache[cache_base + tid] = (half)ky;
    v_cache[cache_base + tid] = (half)v[kv_base + tid];
}

kernel void kernel_qwen35_attention_scores(
        constant ds4_metal_args_qwen35_attention &args,
        device const float *q,
        device const half *k_cache,
        device float *scores,
        uint tid [[thread_index_in_threadgroup]],
        uint3 tgpig [[threadgroup_position_in_grid]]) {
    const uint pos = tgpig.x * 256u + tid;
    const uint h = tgpig.y;
    if (h >= args.n_head) return;
    const uint kvh = h / (args.n_head / args.n_head_kv);
    const ulong kbase = ((ulong)pos * args.n_head_kv + kvh) * args.head_dim;
    const device half4 *k4 = (device const half4 *)(k_cache + kbase);
    for (uint token = 0; token < args.n_tokens; token++) {
        const uint cache_len = args.pos0 + token + 1u;
        if (pos >= cache_len) continue;
        const uint qbase = (token * args.n_head + h) * args.head_dim;
        const device float4 *q4 = (device const float4 *)(q + qbase);
        float sum = 0.0f;
        for (uint i = 0; i < args.head_dim / 4u; i++) {
            sum += dot(q4[i], float4(k4[i]));
        }
        const ulong score_row =
            (ulong)(token * args.n_head + h) * args.cache_cap;
        scores[score_row + pos] = sum * args.scale;
    }
}

kernel void kernel_qwen35_attention_output(
        constant ds4_metal_args_qwen35_attention &args,
        device const float *gate,
        device const half *v_cache,
        device const float *scores,
        device float *out,
        threadgroup float *scratch [[threadgroup(0)]],
        uint tid [[thread_index_in_threadgroup]],
        uint3 tgpig [[threadgroup_position_in_grid]]) {
    const uint h = tgpig.x;
    if (h >= args.n_head || tid >= args.head_dim) return;
    const uint kvh = h / (args.n_head / args.n_head_kv);
    for (uint token = 0; token < args.n_tokens; token++) {
        const uint cache_len = args.pos0 + token + 1u;
        const device float *hs = scores +
            (ulong)(token * args.n_head + h) * args.cache_cap;
        float local_max = -INFINITY;
        for (uint p = tid; p < cache_len; p += args.head_dim) {
            local_max = max(local_max, hs[p]);
        }
        scratch[tid] = local_max;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint step = args.head_dim >> 1; step != 0; step >>= 1) {
            if (tid < step) {
                scratch[tid] = max(scratch[tid], scratch[tid + step]);
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        const float max_score = scratch[0];
        float local_sum = 0.0f;
        for (uint p = tid; p < cache_len; p += args.head_dim) {
            local_sum += exp(hs[p] - max_score);
        }
        scratch[tid] = local_sum;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint step = args.head_dim >> 1; step != 0; step >>= 1) {
            if (tid < step) scratch[tid] += scratch[tid + step];
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        const float inv_sum = 1.0f / max(scratch[0], 1.0e-20f);
        float acc = 0.0f;
        for (uint p = 0; p < cache_len; p++) {
            const ulong vbase =
                ((ulong)p * args.n_head_kv + kvh) * args.head_dim;
            acc += exp(hs[p] - max_score) * (float)v_cache[vbase + tid];
        }
        const uint o =
            (token * args.n_head + h) * args.head_dim + tid;
        const float sigmoid_gate = 1.0f / (1.0f + exp(-gate[o]));
        out[o] = acc * inv_sum * sigmoid_gate;
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
}

kernel void kernel_qwen35_attention_gate(
        constant ds4_metal_args_qwen35_attention &args,
        device const float *gate,
        device float *out,
        uint gid [[thread_position_in_grid]]) {
    const uint total =
        args.n_tokens * args.n_head * args.head_dim;
    if (gid >= total) return;
    out[gid] *= 1.0f / (1.0f + exp(-gate[gid]));
}

kernel void kernel_qwen35_gdn_conv(
        constant ds4_metal_args_qwen35_gdn &args,
        device const float *qkv,
        device const float *conv_weight,
        device float *conv_state,
        device float *prepared,
        uint gid [[thread_position_in_grid]]) {
    if (gid >= args.channels) return;
    const uint sb = gid * (args.conv_width - 1u);
    const uint wb = gid * args.conv_width;
    for (uint token = 0; token < args.n_tokens; token++) {
        const uint off = token * args.channels + gid;
        float y = qkv[off] * conv_weight[wb + args.conv_width - 1u];
        for (uint i = 0; i + 1u < args.conv_width; i++) {
            y += conv_state[sb + i] * conv_weight[wb + i];
        }
        for (uint i = 0; i + 2u < args.conv_width; i++) {
            conv_state[sb + i] = conv_state[sb + i + 1u];
        }
        conv_state[sb + args.conv_width - 2u] = qkv[off];
        prepared[off] = qwen35_silu(y);
    }
}

// Two-row speculative verification keeps the state after the committed first
// row in conv_state while materializing the state after the speculative second
// row in final_state. Both rows still produce the same prepared activations as
// the ordinary in-place batch kernel.
kernel void kernel_qwen35_gdn_conv_preserve_first(
        constant ds4_metal_args_qwen35_gdn &args,
        device const float *qkv,
        device const float *conv_weight,
        device float *conv_state,
        device float *middle_state,
        device float *final_state,
        device float *prepared,
        uint gid [[thread_position_in_grid]]) {
    if (gid >= args.channels || args.n_tokens < 2u ||
        args.n_tokens > 3u) return;
    const uint sb = gid * (args.conv_width - 1u);
    const uint wb = gid * args.conv_width;
    for (uint token = 0; token < args.n_tokens; token++) {
        device float *state = token == 0u ? conv_state :
            (token + 1u == args.n_tokens ? final_state : middle_state);
        if (token != 0u) {
            device const float *previous = token == 1u ?
                conv_state : middle_state;
            for (uint i = 0; i + 1u < args.conv_width; i++) {
                state[sb + i] = previous[sb + i];
            }
        }
        const uint off = token * args.channels + gid;
        float y = qkv[off] * conv_weight[wb + args.conv_width - 1u];
        for (uint i = 0; i + 1u < args.conv_width; i++) {
            y += state[sb + i] * conv_weight[wb + i];
        }
        for (uint i = 0; i + 2u < args.conv_width; i++) {
            state[sb + i] = state[sb + i + 1u];
        }
        state[sb + args.conv_width - 2u] = qkv[off];
        prepared[off] = qwen35_silu(y);
    }
}

kernel void kernel_qwen35_gdn_conv_preserve_four(
        constant ds4_metal_args_qwen35_gdn &args,
        device const float *qkv,
        device const float *conv_weight,
        device float *state0,
        device float *state1,
        device float *state2,
        device float *state3,
        device float *prepared,
        uint gid [[thread_position_in_grid]]) {
    if (gid >= args.channels || args.n_tokens != 4u) return;
    const uint sb = gid * (args.conv_width - 1u);
    const uint wb = gid * args.conv_width;
    for (uint token = 0; token < 4u; token++) {
        device float *state = token == 0u ? state0 :
            token == 1u ? state1 : token == 2u ? state2 : state3;
        if (token != 0u) {
            device const float *previous = token == 1u ? state0 :
                token == 2u ? state1 : state2;
            for (uint i = 0; i + 1u < args.conv_width; i++) {
                state[sb + i] = previous[sb + i];
            }
        }
        const uint off = token * args.channels + gid;
        float y = qkv[off] * conv_weight[wb + args.conv_width - 1u];
        for (uint i = 0; i + 1u < args.conv_width; i++) {
            y += state[sb + i] * conv_weight[wb + i];
        }
        for (uint i = 0; i + 2u < args.conv_width; i++) {
            state[sb + i] = state[sb + i + 1u];
        }
        state[sb + args.conv_width - 2u] = qkv[off];
        prepared[off] = qwen35_silu(y);
    }
}

// Decode-only dispatch fusion.  Parameter transforms are independent of the
// depthwise convolution, and their 48 scalar rows fit inside the convolution's
// existing grid.  Keeping each expression unchanged preserves the standalone
// kernels' arithmetic while removing one launch per GDN layer.
kernel void kernel_qwen35_gdn_conv_params(
        constant ds4_metal_args_qwen35_gdn &args,
        device const float *qkv,
        device const float *conv_weight,
        device float *conv_state,
        device float *prepared,
        device const float *alpha,
        device const float *beta,
        device const float *dt_bias,
        device const float *a,
        device float *g,
        device float *b,
        uint gid [[thread_position_in_grid]]) {
    if (gid < args.channels) {
        const uint sb = gid * (args.conv_width - 1u);
        const uint wb = gid * args.conv_width;
        for (uint token = 0; token < args.n_tokens; token++) {
            const uint off = token * args.channels + gid;
            float y = qkv[off] * conv_weight[wb + args.conv_width - 1u];
            for (uint i = 0; i + 1u < args.conv_width; i++) {
                y += conv_state[sb + i] * conv_weight[wb + i];
            }
            for (uint i = 0; i + 2u < args.conv_width; i++) {
                conv_state[sb + i] = conv_state[sb + i + 1u];
            }
            conv_state[sb + args.conv_width - 2u] = qkv[off];
            prepared[off] = qwen35_silu(y);
        }
    }

    const uint param_total = args.n_tokens * args.v_heads;
    if (gid < param_total) {
        const uint h = gid % args.v_heads;
        g[gid] = qwen35_softplus(alpha[gid] + dt_bias[h]) * a[h];
        b[gid] = 1.0f / (1.0f + exp(-beta[gid]));
    }
}

// Decode-only preparation fusion.  With one token, each 256-thread
// convolution group covers exactly two 128-wide Q/K heads.  Reuse that grid
// for the same reduction tree as kernel_qwen35_gdn_qk_norm while also folding
// in the independent parameter transform.  This removes both small follow-up
// dispatches without allocating another intermediate.
kernel void kernel_qwen35_gdn_qk_norm(
        constant ds4_metal_args_qwen35_gdn &args,
        device float *prepared,
        threadgroup float *scratch [[threadgroup(0)]],
        uint tid [[thread_index_in_threadgroup]],
        uint3 tgpig [[threadgroup_position_in_grid]]) {
    const uint hk = tgpig.x;
    const uint token = tgpig.y;
    if (hk >= 2u * args.qk_heads || token >= args.n_tokens ||
        tid >= args.state_dim) return;
    const uint base = token * args.channels + hk * args.state_dim;
    const float x = prepared[base + tid];
    scratch[tid] = x * x;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint step = args.state_dim >> 1; step != 0; step >>= 1) {
        if (tid < step) scratch[tid] += scratch[tid + step];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    prepared[base + tid] = x * rsqrt(scratch[0] + args.eps);
}

kernel void kernel_qwen35_gdn_params(
        constant ds4_metal_args_qwen35_gdn &args,
        device const float *alpha,
        device const float *beta,
        device const float *dt_bias,
        device const float *a,
        device float *g,
        device float *b,
        uint gid [[thread_position_in_grid]]) {
    const uint total = args.n_tokens * args.v_heads;
    if (gid >= total) return;
    const uint h = gid % args.v_heads;
    g[gid] = qwen35_softplus(alpha[gid] + dt_bias[h]) * a[h];
    b[gid] = 1.0f / (1.0f + exp(-beta[gid]));
}

kernel void kernel_qwen35_gdn_recurrent(
        constant ds4_metal_args_qwen35_gdn &args,
        device const float *prepared,
        device const float *g,
        device const float *b,
        device float *state,
        device float *out,
        uint3 tid [[thread_position_in_threadgroup]],
        uint3 tgpig [[threadgroup_position_in_grid]]) {
    const uint tx = tid.x;
    const uint ty = tid.y;
    const uint row = tgpig.x * 4u + ty;
    const uint h = tgpig.y;
    if (h >= args.v_heads || row >= args.state_dim) return;
    // Qwen3.5 GGUF stores value-side heads in tiled GGML broadcast order.
    const uint qkh = h % args.qk_heads;
    device float *s = state + ((ulong)h * args.state_dim + row) * args.state_dim;
    for (uint token = 0; token < args.n_tokens; token++) {
        const device float *token_prepared = prepared + token * args.channels;
        const device float *q = token_prepared + qkh * args.state_dim;
        const device float *k = token_prepared +
            (args.qk_heads + qkh) * args.state_dim;
        const device float *v = token_prepared +
            2u * args.qk_heads * args.state_dim + h * args.state_dim;
        const uint param_off = token * args.v_heads + h;
        const float decay = exp(g[param_off]);
        float sv[4];
        float sk = 0.0f;
        for (uint j = 0; j < 4u; j++) {
            const uint col = tx * 4u + j;
            sv[j] = s[col] * decay;
            sk += sv[j] * k[col];
        }
        sk = simd_sum(sk);
        const float delta = (v[row] - sk) * b[param_off];
        float y = 0.0f;
        for (uint j = 0; j < 4u; j++) {
            const uint col = tx * 4u + j;
            sv[j] += k[col] * delta;
            s[col] = sv[j];
            y += sv[j] * q[col];
        }
        y = simd_sum(y);
        if (tx == 0u) {
            out[(token * args.v_heads + h) * args.state_dim + row] =
                y * args.scale;
        }
    }
}

// Recurrent counterpart to kernel_qwen35_gdn_conv_preserve_first. Row zero
// advances the live state; row one starts from that state but writes its final
// result to final_state so a verifier rejection requires no target replay.
kernel void kernel_qwen35_gdn_recurrent_preserve_first(
        constant ds4_metal_args_qwen35_gdn &args,
        device const float *prepared,
        device const float *g,
        device const float *b,
        device float *state,
        device float *middle_state,
        device float *final_state,
        device float *out,
        uint3 tid [[thread_position_in_threadgroup]],
        uint3 tgpig [[threadgroup_position_in_grid]]) {
    const uint tx = tid.x;
    const uint ty = tid.y;
    const uint row = tgpig.x * 4u + ty;
    const uint h = tgpig.y;
    if (h >= args.v_heads || row >= args.state_dim ||
        args.n_tokens < 2u || args.n_tokens > 3u) return;
    const uint qkh = h % args.qk_heads;
    const ulong state_base =
        ((ulong)h * args.state_dim + row) * args.state_dim;
    device float *first = state + state_base;
    device float *middle = middle_state + state_base;
    device float *last = final_state + state_base;
    for (uint token = 0; token < args.n_tokens; token++) {
        device float *s = token == 0u ? first :
            (token + 1u == args.n_tokens ? last : middle);
        if (token != 0u) {
            device const float *previous = token == 1u ? first : middle;
            for (uint j = 0; j < 4u; j++) {
                const uint col = tx * 4u + j;
                s[col] = previous[col];
            }
        }
        const device float *token_prepared = prepared + token * args.channels;
        const device float *q = token_prepared + qkh * args.state_dim;
        const device float *k = token_prepared +
            (args.qk_heads + qkh) * args.state_dim;
        const device float *v = token_prepared +
            2u * args.qk_heads * args.state_dim + h * args.state_dim;
        const uint param_off = token * args.v_heads + h;
        const float decay = exp(g[param_off]);
        float sv[4];
        float sk = 0.0f;
        for (uint j = 0; j < 4u; j++) {
            const uint col = tx * 4u + j;
            sv[j] = s[col] * decay;
            sk += sv[j] * k[col];
        }
        sk = simd_sum(sk);
        const float delta = (v[row] - sk) * b[param_off];
        float y = 0.0f;
        for (uint j = 0; j < 4u; j++) {
            const uint col = tx * 4u + j;
            sv[j] += k[col] * delta;
            s[col] = sv[j];
            y += sv[j] * q[col];
        }
        y = simd_sum(y);
        if (tx == 0u) {
            out[(token * args.v_heads + h) * args.state_dim + row] =
                y * args.scale;
        }
    }
}

kernel void kernel_qwen35_gdn_recurrent_preserve_four(
        constant ds4_metal_args_qwen35_gdn &args,
        device const float *prepared,
        device const float *g,
        device const float *b,
        device float *state0,
        device float *state1,
        device float *state2,
        device float *state3,
        device float *out,
        uint3 tid [[thread_position_in_threadgroup]],
        uint3 tgpig [[threadgroup_position_in_grid]]) {
    const uint tx = tid.x;
    const uint ty = tid.y;
    const uint row = tgpig.x * 4u + ty;
    const uint h = tgpig.y;
    if (h >= args.v_heads || row >= args.state_dim ||
        args.n_tokens != 4u) return;
    const uint qkh = h % args.qk_heads;
    const ulong state_base =
        ((ulong)h * args.state_dim + row) * args.state_dim;
    device float *frontier0 = state0 + state_base;
    device float *frontier1 = state1 + state_base;
    device float *frontier2 = state2 + state_base;
    device float *frontier3 = state3 + state_base;
    for (uint token = 0; token < 4u; token++) {
        device float *s = token == 0u ? frontier0 :
            token == 1u ? frontier1 : token == 2u ? frontier2 : frontier3;
        if (token != 0u) {
            device const float *previous = token == 1u ? frontier0 :
                token == 2u ? frontier1 : frontier2;
            for (uint j = 0; j < 4u; j++) {
                const uint col = tx * 4u + j;
                s[col] = previous[col];
            }
        }
        const device float *token_prepared = prepared + token * args.channels;
        const device float *q = token_prepared + qkh * args.state_dim;
        const device float *k = token_prepared +
            (args.qk_heads + qkh) * args.state_dim;
        const device float *v = token_prepared +
            2u * args.qk_heads * args.state_dim + h * args.state_dim;
        const uint param_off = token * args.v_heads + h;
        const float decay = exp(g[param_off]);
        float sv[4];
        float sk = 0.0f;
        for (uint j = 0; j < 4u; j++) {
            const uint col = tx * 4u + j;
            sv[j] = s[col] * decay;
            sk += sv[j] * k[col];
        }
        sk = simd_sum(sk);
        const float delta = (v[row] - sk) * b[param_off];
        float y = 0.0f;
        for (uint j = 0; j < 4u; j++) {
            const uint col = tx * 4u + j;
            sv[j] += k[col] * delta;
            s[col] = sv[j];
            y += sv[j] * q[col];
        }
        y = simd_sum(y);
        if (tx == 0u) {
            out[(token * args.v_heads + h) * args.state_dim + row] =
                y * args.scale;
        }
    }
}

// Matrix-oriented extended-WY prefill.  This follows FLA's chunkwise split:
// form the causal K K^T system, solve its block lower-triangular inverse,
// build W/U, then express the state correction, output, and final state as
// matrix products.  Half inputs with float accumulation feed Apple's 8x8
// simdgroup matrix hardware; all persistent model state remains float.
//
// The existing float scratch tensors hold packed half intermediates. W stores
// scaled K and gated Q; U stores scaled V plus shared raw Q/K; qk stores A^-1;
// values stores corrected values, causal QK, and shared raw KKT/QK. Thus a
// 64-token chunk remains bounded without another activation allocation.
kernel void kernel_qwen35_gdn_wy_pack(
        constant ds4_metal_args_qwen35_gdn &args,
        device const float *prepared,
        device const float *g,
        device const float *b,
        device float *w,
        device float *u,
        device float *cumulative_g,
        threadgroup float *cg [[threadgroup(0)]],
        uint tid [[thread_index_in_threadgroup]],
        ushort3 ntg [[threads_per_threadgroup]],
        uint3 tgpig [[threadgroup_position_in_grid]]) {
    const uint h = tgpig.x;
    const uint n = args.n_tokens;
    const uint d = args.state_dim;
    const uint qkh = h % args.qk_heads;
    device half *scaled_k = (device half *)w;
    const ulong elems = (ulong)n * args.v_heads * d;
    device half *gated_q = scaled_k + elems;
    device half *scaled_v = (device half *)u;
    device half *raw_q = scaled_v + elems;
    device half *raw_k = raw_q + (ulong)args.qk_heads * n * d;

    if (tid == 0u) {
        float sum = 0.0f;
        for (uint i = 0; i < n; i++) {
            sum += g[i * args.v_heads + h];
            cumulative_g[i * args.v_heads + h] = sum;
            cg[i] = exp(sum);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint p = tid; p < n * d; p += ntg.x) {
        const uint token = p / d;
        const uint col = p - token * d;
        const device float *src = prepared + (ulong)token * args.channels;
        const float beta = b[token * args.v_heads + h];
        const float gate = cg[token];
        const ulong off = ((ulong)h * n + token) * d + col;
        scaled_k[off] = half(beta * gate *
            src[(args.qk_heads + qkh) * d + col]);
        gated_q[off] = half(gate * src[qkh * d + col]);
        scaled_v[off] = half(beta *
            src[2u * args.qk_heads * d + h * d + col]);
        if (h < args.qk_heads) {
            const ulong qk_off = ((ulong)h * n + token) * d + col;
            raw_q[qk_off] = half(src[h * d + col]);
            raw_k[qk_off] = half(src[(args.qk_heads + h) * d + col]);
        }
    }
}

kernel void kernel_qwen35_gdn_wy_kkt_qk_raw(
        constant ds4_metal_args_qwen35_gdn &args,
        device const float *u_storage,
        device float *values,
        uint tid [[thread_index_in_threadgroup]],
        uint3 tgpig [[threadgroup_position_in_grid]]) {
    const uint n = args.n_tokens;
    const uint d = args.state_dim;
    uint packed_tile = tgpig.x;
    uint row_tile = 0u;
    while (packed_tile >= row_tile + 1u) {
        packed_tile -= row_tile + 1u;
        row_tile++;
    }
    const uint col0 = packed_tile * 8u;
    const uint row0 = row_tile * 8u;
    const uint qkh = tgpig.z;
    const ulong nn = (ulong)n * n;
    const ulong elems = (ulong)n * args.v_heads * d;
    device const half *scaled_v = (device const half *)u_storage;
    device const half *raw_q = scaled_v + elems;
    device const half *raw_k = raw_q + (ulong)args.qk_heads * n * d;
    device half *raw_kkt = (device half *)values +
        elems + (ulong)args.v_heads * nn;
    device half *raw_qk = raw_kkt + (ulong)args.qk_heads * nn;
    threadgroup float kkt[64];
    threadgroup float qkt[64];

    simdgroup_float8x8 mkkt =
        make_filled_simdgroup_matrix<float, 8>(0.0f);
    simdgroup_float8x8 mqkt =
        make_filled_simdgroup_matrix<float, 8>(0.0f);
    for (uint db = 0u; db < d / 8u; db++) {
        simdgroup_half8x8 mka;
        simdgroup_half8x8 mqa;
        simdgroup_half8x8 mkb;
        simdgroup_load(mka, raw_k + ((ulong)qkh * n + row0) * d + db * 8u,
                       d, 0u, false);
        simdgroup_load(mqa, raw_q + ((ulong)qkh * n + row0) * d + db * 8u,
                       d, 0u, false);
        simdgroup_load(mkb, raw_k + ((ulong)qkh * n + col0) * d + db * 8u,
                       d, 0u, true);
        simdgroup_multiply_accumulate(mkkt, mka, mkb, mkkt);
        simdgroup_multiply_accumulate(mqkt, mqa, mkb, mqkt);
    }
    simdgroup_store(mkkt, kkt, 8u, 0u, false);
    simdgroup_store(mqkt, qkt, 8u, 0u, false);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint p = tid; p < 64u; p += 32u) {
        const uint row = row0 + p / 8u;
        const uint col = col0 + p % 8u;
        if (row >= n || col >= n) continue;
        const ulong off = ((ulong)qkh * n + row) * n + col;
        raw_kkt[off] = half(kkt[p]);
        raw_qk[off] = half(qkt[p]);
    }
}

kernel void kernel_qwen35_gdn_wy_solve(
        constant ds4_metal_args_qwen35_gdn &args,
        device const float *b,
        device const float *cumulative_g,
        device float *qk_storage,
        device float *values,
        threadgroup uchar *raw [[threadgroup(0)]],
        uint tid [[thread_index_in_threadgroup]],
        ushort lane [[thread_index_in_simdgroup]],
        ushort sg [[simdgroup_index_in_threadgroup]],
        ushort3 ntg [[threads_per_threadgroup]],
        uint3 tgpig [[threadgroup_position_in_grid]]) {
    const uint n = args.n_tokens;
    const uint nb = n / 8u;
    const uint h = tgpig.x;
    const uint qkh = h % args.qk_heads;
    const ulong nn = (ulong)n * n;
    const ulong matrices = (ulong)args.v_heads * nn;
    const ulong elems = (ulong)n * args.v_heads * args.state_dim;
    const uint nsg = ntg.x / 32u;
    device const half *raw_kkt = (device const half *)values +
        elems + (ulong)args.v_heads * nn + (ulong)qkh * nn;
    device const half *raw_qk = (device const half *)values +
        elems + (ulong)args.v_heads * nn +
        (ulong)args.qk_heads * nn + (ulong)qkh * nn;
    device half *causal_qk = (device half *)values + elems + (ulong)h * nn;
    device half *x_device = (device half *)qk_storage + matrices + (ulong)h * nn;
    threadgroup half *a = (threadgroup half *)raw;
    threadgroup half *x = a + nn;
    threadgroup half *tmp_h = x + nn;
    threadgroup float *tmp_f =
        (threadgroup float *)(tmp_h + (ulong)nsg * 64u);
    threadgroup half *sg_h = tmp_h + (ulong)sg * 64u;
    threadgroup float *sg_f = tmp_f + (ulong)sg * 64u;

    for (uint p = tid; p < nn; p += ntg.x) {
        const uint row = p / n;
        const uint col = p - row * n;
        if (row < col) {
            a[p] = half(0.0f);
            causal_qk[p] = half(0.0f);
        } else if (row == col) {
            a[p] = half(1.0f);
            causal_qk[p] = raw_qk[p];
        } else {
            const float gate = exp(
                cumulative_g[row * args.v_heads + h] -
                cumulative_g[col * args.v_heads + h]);
            a[p] = half(b[row * args.v_heads + h] * gate *
                        float(raw_kkt[p]));
            causal_qk[p] = half(gate * float(raw_qk[p]));
        }
        x[p] = half(0.0f);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Diagonal blocks and independent columns of each block row are spread
    // across SIMD groups.  Only each small 8x8 diagonal solve is scalar.
    for (uint bi = sg; bi < nb; bi += nsg) {
        for (uint p = lane; p < 64u; p += 32u) {
            sg_f[p] = float(a[(bi * 8u + p / 8u) * n + bi * 8u + p % 8u]);
        }
        simdgroup_barrier(mem_flags::mem_threadgroup);
        if (lane == 0u) {
            for (uint i = 1u; i < 8u; i++) {
                for (uint j = 0u; j < i; j++) {
                    float sum = 0.0f;
                    for (uint k = j; k < i; k++) {
                        sum += sg_f[i * 8u + k] * sg_f[k * 8u + j];
                    }
                    sg_f[i * 8u + j] = -sum;
                }
            }
        }
        simdgroup_barrier(mem_flags::mem_threadgroup);
        for (uint p = lane; p < 64u; p += 32u) {
            x[(bi * 8u + p / 8u) * n + bi * 8u + p % 8u] = half(sg_f[p]);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint bi = 1u; bi < nb; bi++) {
        for (uint bj = sg; bj < bi; bj += nsg) {
            simdgroup_float8x8 sum =
                make_filled_simdgroup_matrix<float, 8>(0.0f);
            for (uint bk = bj; bk < bi; bk++) {
                simdgroup_half8x8 ma;
                simdgroup_half8x8 mb;
                simdgroup_load(ma, a + bi * 8u * n + bk * 8u, n, 0u, false);
                simdgroup_load(mb, x + bk * 8u * n + bj * 8u, n, 0u, false);
                simdgroup_multiply_accumulate(sum, ma, mb, sum);
            }
            simdgroup_store(sum, sg_f, 8u, 0u, false);
            simdgroup_barrier(mem_flags::mem_threadgroup);
            for (uint p = lane; p < 64u; p += 32u) sg_h[p] = half(-sg_f[p]);
            simdgroup_barrier(mem_flags::mem_threadgroup);

            simdgroup_half8x8 md;
            simdgroup_half8x8 ms;
            simdgroup_float8x8 product =
                make_filled_simdgroup_matrix<float, 8>(0.0f);
            simdgroup_load(md, x + bi * 8u * n + bi * 8u, n, 0u, false);
            simdgroup_load(ms, sg_h, 8u, 0u, false);
            simdgroup_multiply_accumulate(product, md, ms, product);
            simdgroup_store(product, sg_f, 8u, 0u, false);
            simdgroup_barrier(mem_flags::mem_threadgroup);
            for (uint p = lane; p < 64u; p += 32u) {
                x[(bi * 8u + p / 8u) * n + bj * 8u + p % 8u] = half(sg_f[p]);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    for (uint p = tid; p < nn; p += ntg.x) x_device[p] = x[p];
}

kernel void kernel_qwen35_gdn_wy_values(
        constant ds4_metal_args_qwen35_gdn &args,
        device const float *w_storage,
        device const float *u_storage,
        device const float *state,
        device float *values_storage,
        uint tid [[thread_index_in_threadgroup]],
        uint3 tgpig [[threadgroup_position_in_grid]]) {
    const uint n = args.n_tokens;
    const uint d = args.state_dim;
    const uint col0 = tgpig.x * 8u;
    const uint row0 = tgpig.y * 8u;
    const uint h = tgpig.z;
    const ulong elems = (ulong)n * args.v_heads * d;
    device const half *wy_w = (device const half *)w_storage + elems;
    device const half *wy_u = (device const half *)u_storage + elems;
    device half *values = (device half *)values_storage;
    threadgroup half st[64];
    threadgroup float product[64];
    simdgroup_float8x8 mv = make_filled_simdgroup_matrix<float, 8>(0.0f);
    for (uint kb = 0u; kb < d; kb += 8u) {
        for (uint p = tid; p < 64u; p += 32u) {
            const uint k = kb + p / 8u;
            const uint v = col0 + p % 8u;
            st[p] = half(state[((ulong)h * d + v) * d + k]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        simdgroup_half8x8 mw;
        simdgroup_half8x8 ms;
        simdgroup_load(mw, wy_w + ((ulong)h * n + row0) * d + kb,
                       d, 0u, false);
        simdgroup_load(ms, st, 8u, 0u, false);
        simdgroup_multiply_accumulate(mv, mw, ms, mv);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    simdgroup_store(mv, product, 8u, 0u, false);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint p = tid; p < 64u; p += 32u) {
        const uint token = row0 + p / 8u;
        const uint v = col0 + p % 8u;
        const ulong off = ((ulong)h * n + token) * d + v;
        values[off] = half(float(wy_u[off]) - product[p]);
    }
}

// Direct FlashQLA-style value correction:
//   Vd = A^-1 (diag(beta)V - diag(beta exp(G))K S^T).
// This avoids materializing W=A^-1 Kb and U=A^-1 Vb and replaces the three
// GEMMs in wy_wu + wy_values with two.  One simdgroup owns each 8-token row
// tile while all row tiles share the same state/value-column tile.
kernel void kernel_qwen35_gdn_wy_values_direct(
        constant ds4_metal_args_qwen35_gdn &args,
        device const float *qk_storage,
        device const float *w_storage,
        device const float *u_storage,
        device const float *state,
        device float *values_storage,
        threadgroup uchar *raw [[threadgroup(0)]],
        uint tid [[thread_index_in_threadgroup]],
        ushort sg [[simdgroup_index_in_threadgroup]],
        ushort3 ntg [[threads_per_threadgroup]],
        uint3 tgpig [[threadgroup_position_in_grid]]) {
    const uint n = args.n_tokens;
    const uint d = args.state_dim;
    const uint h = tgpig.y;
    const uint col0 = tgpig.x * 8u;
    const uint row0 = (uint)sg * 8u;
    const ulong elems = (ulong)n * args.v_heads * d;
    const ulong matrices = (ulong)args.v_heads * n * n;
    device const half *ainv = (device const half *)qk_storage +
        matrices + (ulong)h * n * n;
    device const half *scaled_k = (device const half *)w_storage;
    device const half *scaled_v = (device const half *)u_storage;
    device half *values = (device half *)values_storage;
    threadgroup half *state_tile = (threadgroup half *)raw;
    threadgroup float *ks = (threadgroup float *)(state_tile + 64u);
    threadgroup half *residual =
        (threadgroup half *)(ks + (ulong)n * 8u);

    simdgroup_float8x8 mks =
        make_filled_simdgroup_matrix<float, 8>(0.0f);
    for (uint kb = 0u; kb < d; kb += 8u) {
        for (uint p = tid; p < 64u; p += ntg.x) {
            const uint k = kb + p / 8u;
            const uint v = col0 + p % 8u;
            state_tile[p] = half(state[((ulong)h * d + v) * d + k]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        simdgroup_half8x8 mk;
        simdgroup_half8x8 ms;
        simdgroup_load(mk, scaled_k + ((ulong)h * n + row0) * d + kb,
                       d, 0u, false);
        simdgroup_load(ms, state_tile, 8u, 0u, false);
        simdgroup_multiply_accumulate(mks, mk, ms, mks);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    simdgroup_store(mks, ks + row0 * 8u, 8u, 0u, false);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint p = tid; p < n * 8u; p += ntg.x) {
        const uint token = p / 8u;
        const uint v = col0 + p % 8u;
        const ulong off = ((ulong)h * n + token) * d + v;
        residual[p] = half(float(scaled_v[off]) - ks[p]);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    simdgroup_float8x8 mvalues =
        make_filled_simdgroup_matrix<float, 8>(0.0f);
    for (uint kb = 0u; kb < n; kb += 8u) {
        simdgroup_half8x8 ma;
        simdgroup_half8x8 mr;
        simdgroup_load(ma, ainv + row0 * n + kb, n, 0u, false);
        simdgroup_load(mr, residual + kb * 8u, 8u, 0u, false);
        simdgroup_multiply_accumulate(mvalues, ma, mr, mvalues);
    }
    simdgroup_store(mvalues, ks + row0 * 8u, 8u, 0u, false);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint p = tid; p < n * 8u; p += ntg.x) {
        const uint token = p / 8u;
        const uint v = col0 + p % 8u;
        values[((ulong)h * n + token) * d + v] = half(ks[p]);
    }
}

// Direct value correction plus output.  The state tile is shared by KS and
// QS, and the corrected values remain in threadgroup memory for causal QK*V.
kernel void kernel_qwen35_gdn_wy_values_output(
        constant ds4_metal_args_qwen35_gdn &args,
        device const float *qk_storage,
        device const float *w_storage,
        device const float *u_storage,
        device const float *state,
        device float *values_storage,
        device float *out,
        threadgroup uchar *raw [[threadgroup(0)]],
        uint tid [[thread_index_in_threadgroup]],
        ushort sg [[simdgroup_index_in_threadgroup]],
        ushort3 ntg [[threads_per_threadgroup]],
        uint3 tgpig [[threadgroup_position_in_grid]]) {
    const uint n = args.n_tokens;
    const uint d = args.state_dim;
    const uint h = tgpig.y;
    const uint col0 = tgpig.x * 8u;
    const uint row0 = (uint)sg * 8u;
    const ulong elems = (ulong)n * args.v_heads * d;
    const ulong matrices = (ulong)args.v_heads * n * n;
    device const half *ainv = (device const half *)qk_storage +
        matrices + (ulong)h * n * n;
    device const half *scaled_k = (device const half *)w_storage;
    device const half *gated_q = scaled_k + elems;
    device const half *scaled_v = (device const half *)u_storage;
    device half *values = (device half *)values_storage;
    device const half *causal_qk = values + elems + (ulong)h * n * n;
    threadgroup half *state_tile = (threadgroup half *)raw;
    threadgroup float *ks = (threadgroup float *)(state_tile + 64u);
    threadgroup float *qs = ks + (ulong)n * 8u;
    threadgroup half *local_values =
        (threadgroup half *)(qs + (ulong)n * 8u);

    simdgroup_float8x8 mks =
        make_filled_simdgroup_matrix<float, 8>(0.0f);
    simdgroup_float8x8 mqs =
        make_filled_simdgroup_matrix<float, 8>(0.0f);
    for (uint kb = 0u; kb < d; kb += 8u) {
        for (uint p = tid; p < 64u; p += ntg.x) {
            const uint k = kb + p / 8u;
            const uint v = col0 + p % 8u;
            state_tile[p] = half(state[((ulong)h * d + v) * d + k]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        simdgroup_half8x8 mk;
        simdgroup_half8x8 mq;
        simdgroup_half8x8 ms;
        simdgroup_load(mk, scaled_k + ((ulong)h * n + row0) * d + kb,
                       d, 0u, false);
        simdgroup_load(mq, gated_q + ((ulong)h * n + row0) * d + kb,
                       d, 0u, false);
        simdgroup_load(ms, state_tile, 8u, 0u, false);
        simdgroup_multiply_accumulate(mks, mk, ms, mks);
        simdgroup_multiply_accumulate(mqs, mq, ms, mqs);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    simdgroup_store(mks, ks + row0 * 8u, 8u, 0u, false);
    simdgroup_store(mqs, qs + row0 * 8u, 8u, 0u, false);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint p = tid; p < n * 8u; p += ntg.x) {
        const uint token = p / 8u;
        const uint v = col0 + p % 8u;
        const ulong off = ((ulong)h * n + token) * d + v;
        local_values[p] = half(float(scaled_v[off]) - ks[p]);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    simdgroup_float8x8 mv =
        make_filled_simdgroup_matrix<float, 8>(0.0f);
    for (uint kb = 0u; kb < n; kb += 8u) {
        simdgroup_half8x8 ma;
        simdgroup_half8x8 mr;
        simdgroup_load(ma, ainv + row0 * n + kb, n, 0u, false);
        simdgroup_load(mr, local_values + kb * 8u, 8u, 0u, false);
        simdgroup_multiply_accumulate(mv, ma, mr, mv);
    }
    simdgroup_store(mv, ks + row0 * 8u, 8u, 0u, false);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint p = tid; p < n * 8u; p += ntg.x) {
        local_values[p] = half(ks[p]);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    simdgroup_float8x8 mo = mqs;
    for (uint kb = 0u; kb < n; kb += 8u) {
        simdgroup_half8x8 mqk;
        simdgroup_half8x8 mvalues;
        simdgroup_load(mqk, causal_qk + row0 * n + kb, n, 0u, false);
        simdgroup_load(mvalues, local_values + kb * 8u, 8u, 0u, false);
        simdgroup_multiply_accumulate(mo, mqk, mvalues, mo);
    }
    simdgroup_store(mo, qs + row0 * 8u, 8u, 0u, false);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint p = tid; p < n * 8u; p += ntg.x) {
        const uint token = p / 8u;
        const uint v = col0 + p % 8u;
        const ulong off = ((ulong)h * n + token) * d + v;
        values[off] = local_values[p];
        out[(token * args.v_heads + h) * d + v] = qs[p] * args.scale;
    }
}

// Four adjacent value tiles share each K/Q, A^-1, and causal-QK tile.  The
// 32-column macrotile raises operand reuse while keeping the complete
// corrected-value slab in bounded threadgroup memory (about 21 KiB at N=64).
kernel void kernel_qwen35_gdn_wy_values_output_32(
        constant ds4_metal_args_qwen35_gdn &args,
        device const float *qk_storage,
        device const float *w_storage,
        device const float *u_storage,
        device float *state,
        device float *values_storage,
        device float *out,
        device const float *cumulative_g,
        threadgroup uchar *raw [[threadgroup(0)]],
        uint tid [[thread_index_in_threadgroup]],
        ushort sg [[simdgroup_index_in_threadgroup]],
        ushort3 ntg [[threads_per_threadgroup]],
        uint3 tgpig [[threadgroup_position_in_grid]]) {
    const uint n = args.n_tokens;
    const uint d = args.state_dim;
    const uint h = tgpig.y;
    const uint col0 = tgpig.x * 32u;
    const uint row0 = (uint)sg * 8u;
    const ulong elems = (ulong)n * args.v_heads * d;
    const ulong matrices = (ulong)args.v_heads * n * n;
    device const half *ainv = (device const half *)qk_storage +
        matrices + (ulong)h * n * n;
    device const half *scaled_k = (device const half *)w_storage;
    device const half *gated_q = scaled_k + elems;
    device const half *scaled_v = (device const half *)u_storage;
    device const half *raw_q = scaled_v + elems;
    device const half *raw_k = raw_q +
        (ulong)args.qk_heads * n * d;
    device half *values = (device half *)values_storage;
    device const half *causal_qk = values + elems + (ulong)h * n * n;
    threadgroup half *state_tile = (threadgroup half *)raw;
    threadgroup float *ks = (threadgroup float *)(state_tile +
        max(8u * 32u, n * 8u));
    threadgroup half *local_values =
        (threadgroup half *)(ks + (ulong)n * 32u);

    simdgroup_float8x8 mks0 =
        make_filled_simdgroup_matrix<float, 8>(0.0f);
    simdgroup_float8x8 mks1 = mks0;
    simdgroup_float8x8 mks2 = mks0;
    simdgroup_float8x8 mks3 = mks0;
    simdgroup_float8x8 mqs0 = mks0;
    simdgroup_float8x8 mqs1 = mks0;
    simdgroup_float8x8 mqs2 = mks0;
    simdgroup_float8x8 mqs3 = mks0;
    for (uint kb = 0u; kb < d; kb += 8u) {
        for (uint p = tid; p < 8u * 32u; p += ntg.x) {
            const uint k = kb + p / 32u;
            const uint v = col0 + p % 32u;
            state_tile[p] = half(state[((ulong)h * d + v) * d + k]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        simdgroup_half8x8 mk;
        simdgroup_half8x8 mq;
        simdgroup_half8x8 ms0;
        simdgroup_half8x8 ms1;
        simdgroup_half8x8 ms2;
        simdgroup_half8x8 ms3;
        simdgroup_load(mk, scaled_k + ((ulong)h * n + row0) * d + kb,
                       d, 0u, false);
        simdgroup_load(mq, gated_q + ((ulong)h * n + row0) * d + kb,
                       d, 0u, false);
        simdgroup_load(ms0, state_tile, 32u, 0u, false);
        simdgroup_load(ms1, state_tile + 8u, 32u, 0u, false);
        simdgroup_load(ms2, state_tile + 16u, 32u, 0u, false);
        simdgroup_load(ms3, state_tile + 24u, 32u, 0u, false);
        simdgroup_multiply_accumulate(mks0, mk, ms0, mks0);
        simdgroup_multiply_accumulate(mks1, mk, ms1, mks1);
        simdgroup_multiply_accumulate(mks2, mk, ms2, mks2);
        simdgroup_multiply_accumulate(mks3, mk, ms3, mks3);
        simdgroup_multiply_accumulate(mqs0, mq, ms0, mqs0);
        simdgroup_multiply_accumulate(mqs1, mq, ms1, mqs1);
        simdgroup_multiply_accumulate(mqs2, mq, ms2, mqs2);
        simdgroup_multiply_accumulate(mqs3, mq, ms3, mqs3);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    simdgroup_store(mks0, ks + row0 * 32u, 32u, 0u, false);
    simdgroup_store(mks1, ks + row0 * 32u + 8u, 32u, 0u, false);
    simdgroup_store(mks2, ks + row0 * 32u + 16u, 32u, 0u, false);
    simdgroup_store(mks3, ks + row0 * 32u + 24u, 32u, 0u, false);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint p = tid; p < n * 32u; p += ntg.x) {
        const uint token = p / 32u;
        const uint v = col0 + p % 32u;
        const ulong off = ((ulong)h * n + token) * d + v;
        local_values[p] = half(float(scaled_v[off]) - ks[p]);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    simdgroup_float8x8 mv0 =
        make_filled_simdgroup_matrix<float, 8>(0.0f);
    simdgroup_float8x8 mv1 = mv0;
    simdgroup_float8x8 mv2 = mv0;
    simdgroup_float8x8 mv3 = mv0;
    for (uint kb = 0u; kb < n; kb += 8u) {
        simdgroup_half8x8 ma;
        simdgroup_half8x8 mr0;
        simdgroup_half8x8 mr1;
        simdgroup_half8x8 mr2;
        simdgroup_half8x8 mr3;
        simdgroup_load(ma, ainv + row0 * n + kb, n, 0u, false);
        simdgroup_load(mr0, local_values + kb * 32u, 32u, 0u, false);
        simdgroup_load(mr1, local_values + kb * 32u + 8u,
                       32u, 0u, false);
        simdgroup_load(mr2, local_values + kb * 32u + 16u,
                       32u, 0u, false);
        simdgroup_load(mr3, local_values + kb * 32u + 24u,
                       32u, 0u, false);
        simdgroup_multiply_accumulate(mv0, ma, mr0, mv0);
        simdgroup_multiply_accumulate(mv1, ma, mr1, mv1);
        simdgroup_multiply_accumulate(mv2, ma, mr2, mv2);
        simdgroup_multiply_accumulate(mv3, ma, mr3, mv3);
    }
    simdgroup_store(mv0, ks + row0 * 32u, 32u, 0u, false);
    simdgroup_store(mv1, ks + row0 * 32u + 8u, 32u, 0u, false);
    simdgroup_store(mv2, ks + row0 * 32u + 16u, 32u, 0u, false);
    simdgroup_store(mv3, ks + row0 * 32u + 24u, 32u, 0u, false);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint p = tid; p < n * 32u; p += ntg.x) {
        local_values[p] = half(ks[p]);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    simdgroup_float8x8 mo0 = mqs0;
    simdgroup_float8x8 mo1 = mqs1;
    simdgroup_float8x8 mo2 = mqs2;
    simdgroup_float8x8 mo3 = mqs3;
    for (uint kb = 0u; kb < n; kb += 8u) {
        simdgroup_half8x8 mqk;
        simdgroup_half8x8 mvalues0;
        simdgroup_half8x8 mvalues1;
        simdgroup_half8x8 mvalues2;
        simdgroup_half8x8 mvalues3;
        simdgroup_load(mqk, causal_qk + row0 * n + kb, n, 0u, false);
        simdgroup_load(mvalues0, local_values + kb * 32u,
                       32u, 0u, false);
        simdgroup_load(mvalues1, local_values + kb * 32u + 8u,
                       32u, 0u, false);
        simdgroup_load(mvalues2, local_values + kb * 32u + 16u,
                       32u, 0u, false);
        simdgroup_load(mvalues3, local_values + kb * 32u + 24u,
                       32u, 0u, false);
        simdgroup_multiply_accumulate(mo0, mqk, mvalues0, mo0);
        simdgroup_multiply_accumulate(mo1, mqk, mvalues1, mo1);
        simdgroup_multiply_accumulate(mo2, mqk, mvalues2, mo2);
        simdgroup_multiply_accumulate(mo3, mqk, mvalues3, mo3);
    }
    simdgroup_store(mo0, ks + row0 * 32u, 32u, 0u, false);
    simdgroup_store(mo1, ks + row0 * 32u + 8u, 32u, 0u, false);
    simdgroup_store(mo2, ks + row0 * 32u + 16u, 32u, 0u, false);
    simdgroup_store(mo3, ks + row0 * 32u + 24u, 32u, 0u, false);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint p = tid; p < n * 32u; p += ntg.x) {
        const uint token = p / 32u;
        const uint v = col0 + p % 32u;
        const ulong off = ((ulong)h * n + token) * d + v;
        out[(token * args.v_heads + h) * d + v] = ks[p] * args.scale;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Reuse the corrected values while they are still on chip to form the
    // final state.  Each workgroup owns disjoint state value columns, so the
    // update is race-free and can replace the separate state pass.
    const uint tiles = n / 8u;
    const uint state_tiles = d / 8u;
    const uint qkh = h % args.qk_heads;
    const float glast = cumulative_g[(n - 1u) * args.v_heads + h];
    const float decay = exp(glast);
    threadgroup half *kt = state_tile;
    for (uint round = 0u; round * tiles < state_tiles; round++) {
        const uint state_tile_index = round * tiles + (uint)sg;
        simdgroup_float8x8 ms0 =
            make_filled_simdgroup_matrix<float, 8>(0.0f);
        simdgroup_float8x8 ms1 = ms0;
        simdgroup_float8x8 ms2 = ms0;
        simdgroup_float8x8 ms3 = ms0;
        for (uint kb = 0u; kb < n; kb += 8u) {
            for (uint p = tid; p < tiles * 64u; p += ntg.x) {
                const uint local_tile = p / 64u;
                const uint q = p - local_tile * 64u;
                const uint k = (round * tiles + local_tile) * 8u + q / 8u;
                const uint token = kb + q % 8u;
                kt[p] = k < d ? half(exp(glast -
                    cumulative_g[token * args.v_heads + h]) *
                    float(raw_k[((ulong)qkh * n + token) * d + k])) :
                    half(0.0f);
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            simdgroup_half8x8 mk;
            simdgroup_half8x8 mv0;
            simdgroup_half8x8 mv1;
            simdgroup_half8x8 mv2;
            simdgroup_half8x8 mv3;
            simdgroup_load(mk, kt + (uint)sg * 64u, 8u, 0u, false);
            simdgroup_load(mv0, local_values + kb * 32u,
                           32u, 0u, false);
            simdgroup_load(mv1, local_values + kb * 32u + 8u,
                           32u, 0u, false);
            simdgroup_load(mv2, local_values + kb * 32u + 16u,
                           32u, 0u, false);
            simdgroup_load(mv3, local_values + kb * 32u + 24u,
                           32u, 0u, false);
            simdgroup_multiply_accumulate(ms0, mk, mv0, ms0);
            simdgroup_multiply_accumulate(ms1, mk, mv1, ms1);
            simdgroup_multiply_accumulate(ms2, mk, mv2, ms2);
            simdgroup_multiply_accumulate(ms3, mk, mv3, ms3);
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        if (state_tile_index < state_tiles) {
            simdgroup_store(ms0, ks + (uint)sg * 4u * 64u,
                            8u, 0u, false);
            simdgroup_store(ms1, ks + ((uint)sg * 4u + 1u) * 64u,
                            8u, 0u, false);
            simdgroup_store(ms2, ks + ((uint)sg * 4u + 2u) * 64u,
                            8u, 0u, false);
            simdgroup_store(ms3, ks + ((uint)sg * 4u + 3u) * 64u,
                            8u, 0u, false);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint p = tid; p < tiles * 8u * 32u; p += ntg.x) {
            const uint local_tile = p / (8u * 32u);
            const uint q = p - local_tile * 8u * 32u;
            const uint k = (round * tiles + local_tile) * 8u + q / 32u;
            const uint local_v = q % 32u;
            if (k < d) {
                const uint block = local_v / 8u;
                const uint product_off = local_tile * 4u * 64u +
                    block * 64u + (q / 32u) * 8u + local_v % 8u;
                const uint v = col0 + local_v;
                const ulong off = ((ulong)h * d + v) * d + k;
                state[off] = decay * state[off] + ks[product_off];
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
}

kernel void kernel_qwen35_gdn_wy_output(
        constant ds4_metal_args_qwen35_gdn &args,
        device const float *prepared,
        device const float *cumulative_g,
        device const float *state,
        device const float *values_storage,
        device float *out,
        uint tid [[thread_index_in_threadgroup]],
        uint3 tgpig [[threadgroup_position_in_grid]]) {
    const uint n = args.n_tokens;
    const uint d = args.state_dim;
    const uint col0 = tgpig.x * 8u;
    const uint row0 = tgpig.y * 8u;
    const uint h = tgpig.z;
    const uint qkh = h % args.qk_heads;
    const ulong elems = (ulong)n * args.v_heads * d;
    device const half *values = (device const half *)values_storage;
    device const half *causal_qk = values + elems + (ulong)h * n * n;
    threadgroup half qt[64];
    threadgroup half st[64];
    threadgroup float result[64];
    simdgroup_float8x8 mo = make_filled_simdgroup_matrix<float, 8>(0.0f);

    for (uint kb = 0u; kb < d; kb += 8u) {
        for (uint p = tid; p < 64u; p += 32u) {
            const uint token = row0 + p / 8u;
            const uint k = kb + p % 8u;
            const uint sk = kb + p / 8u;
            const uint v = col0 + p % 8u;
            qt[p] = half(exp(cumulative_g[token * args.v_heads + h]) *
                prepared[(ulong)token * args.channels + qkh * d + k]);
            st[p] = half(state[((ulong)h * d + v) * d + sk]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        simdgroup_half8x8 mq;
        simdgroup_half8x8 ms;
        simdgroup_load(mq, qt, 8u, 0u, false);
        simdgroup_load(ms, st, 8u, 0u, false);
        simdgroup_multiply_accumulate(mo, mq, ms, mo);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    for (uint kb = 0u; kb < n; kb += 8u) {
        simdgroup_half8x8 mqk;
        simdgroup_half8x8 mvalues;
        simdgroup_load(mqk, causal_qk + row0 * n + kb, n, 0u, false);
        simdgroup_load(mvalues, values + ((ulong)h * n + kb) * d + col0,
                       d, 0u, false);
        simdgroup_multiply_accumulate(mo, mqk, mvalues, mo);
    }
    simdgroup_store(mo, result, 8u, 0u, false);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint p = tid; p < 64u; p += 32u) {
        const uint token = row0 + p / 8u;
        const uint v = col0 + p % 8u;
        out[((ulong)token * args.v_heads + h) * d + v] = result[p] * args.scale;
    }
}

kernel void kernel_qwen35_gdn_wy_state(
        constant ds4_metal_args_qwen35_gdn &args,
        device const float *prepared,
        device const float *cumulative_g,
        device const float *values_storage,
        device float *state,
        uint tid [[thread_index_in_threadgroup]],
        uint3 tgpig [[threadgroup_position_in_grid]]) {
    const uint n = args.n_tokens;
    const uint d = args.state_dim;
    const uint col0 = tgpig.x * 8u;
    const uint row0 = tgpig.y * 8u;
    const uint h = tgpig.z;
    const uint qkh = h % args.qk_heads;
    const float glast = cumulative_g[(n - 1u) * args.v_heads + h];
    device const half *values = (device const half *)values_storage;
    threadgroup half kt[64];
    threadgroup float product[64];
    simdgroup_float8x8 ms = make_filled_simdgroup_matrix<float, 8>(0.0f);
    for (uint kb = 0u; kb < n; kb += 8u) {
        for (uint p = tid; p < 64u; p += 32u) {
            const uint k = row0 + p / 8u;
            const uint token = kb + p % 8u;
            kt[p] = half(exp(glast - cumulative_g[token * args.v_heads + h]) *
                prepared[(ulong)token * args.channels +
                    (args.qk_heads + qkh) * d + k]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        simdgroup_half8x8 mk;
        simdgroup_half8x8 mv;
        simdgroup_load(mk, kt, 8u, 0u, false);
        simdgroup_load(mv, values + ((ulong)h * n + kb) * d + col0,
                       d, 0u, false);
        simdgroup_multiply_accumulate(ms, mk, mv, ms);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    simdgroup_store(ms, product, 8u, 0u, false);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const float decay = exp(glast);
    for (uint p = tid; p < 64u; p += 32u) {
        const uint k = row0 + p / 8u;
        const uint v = col0 + p % 8u;
        const ulong off = ((ulong)h * d + v) * d + k;
        state[off] = decay * state[off] + product[p];
    }
}

// Share each gated-K tile across four adjacent value tiles.
kernel void kernel_qwen35_gdn_wy_state_32(
        constant ds4_metal_args_qwen35_gdn &args,
        device const float *prepared,
        device const float *cumulative_g,
        device const float *values_storage,
        device float *state,
        uint tid [[thread_index_in_threadgroup]],
        ushort sg [[simdgroup_index_in_threadgroup]],
        uint3 tgpig [[threadgroup_position_in_grid]]) {
    const uint n = args.n_tokens;
    const uint d = args.state_dim;
    const uint col0 = tgpig.x * 32u;
    const uint row0 = tgpig.y * 8u;
    const uint h = tgpig.z;
    const uint qkh = h % args.qk_heads;
    const float glast = cumulative_g[(n - 1u) * args.v_heads + h];
    device const half *values = (device const half *)values_storage;
    threadgroup half kt[64];
    threadgroup float product[4u * 64u];
    simdgroup_float8x8 ms =
        make_filled_simdgroup_matrix<float, 8>(0.0f);
    for (uint kb = 0u; kb < n; kb += 8u) {
        for (uint p = tid; p < 64u; p += 128u) {
            const uint k = row0 + p / 8u;
            const uint token = kb + p % 8u;
            kt[p] = half(exp(glast -
                cumulative_g[token * args.v_heads + h]) *
                prepared[(ulong)token * args.channels +
                    (args.qk_heads + qkh) * d + k]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        simdgroup_half8x8 mk;
        simdgroup_half8x8 mv;
        simdgroup_load(mk, kt, 8u, 0u, false);
        simdgroup_load(mv, values + ((ulong)h * n + kb) * d +
                       col0 + (uint)sg * 8u, d, 0u, false);
        simdgroup_multiply_accumulate(ms, mk, mv, ms);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    simdgroup_store(ms, product + (uint)sg * 64u, 8u, 0u, false);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const float decay = exp(glast);
    for (uint p = tid; p < 8u * 32u; p += 128u) {
        const uint k = row0 + p / 32u;
        const uint v = col0 + p % 32u;
        const uint block = (p % 32u) / 8u;
        const uint product_off = block * 64u +
            (p / 32u) * 8u + p % 8u;
        const ulong off = ((ulong)h * d + v) * d + k;
        state[off] = decay * state[off] + product[product_off];
    }
}

// The final state has enough independent value columns for one workgroup to
// cover the full 128-wide row.  Sixteen SIMDgroups then share a single gated-K
// tile and its exponential evaluation.
kernel void kernel_qwen35_gdn_wy_state_128(
        constant ds4_metal_args_qwen35_gdn &args,
        device const float *prepared,
        device const float *cumulative_g,
        device const float *values_storage,
        device float *state,
        uint tid [[thread_index_in_threadgroup]],
        ushort sg [[simdgroup_index_in_threadgroup]],
        uint3 tgpig [[threadgroup_position_in_grid]]) {
    const uint n = args.n_tokens;
    const uint d = args.state_dim;
    const uint row0 = tgpig.y * 8u;
    const uint h = tgpig.z;
    const uint qkh = h % args.qk_heads;
    const float glast = cumulative_g[(n - 1u) * args.v_heads + h];
    device const half *values = (device const half *)values_storage;
    threadgroup half kt[64];
    threadgroup float product[16u * 64u];
    simdgroup_float8x8 ms =
        make_filled_simdgroup_matrix<float, 8>(0.0f);
    for (uint kb = 0u; kb < n; kb += 8u) {
        if (tid < 64u) {
            const uint k = row0 + tid / 8u;
            const uint token = kb + tid % 8u;
            kt[tid] = half(exp(glast -
                cumulative_g[token * args.v_heads + h]) *
                prepared[(ulong)token * args.channels +
                    (args.qk_heads + qkh) * d + k]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        simdgroup_half8x8 mk;
        simdgroup_half8x8 mv;
        simdgroup_load(mk, kt, 8u, 0u, false);
        simdgroup_load(mv, values + ((ulong)h * n + kb) * d +
                       (uint)sg * 8u, d, 0u, false);
        simdgroup_multiply_accumulate(ms, mk, mv, ms);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    simdgroup_store(ms, product + (uint)sg * 64u, 8u, 0u, false);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const float decay = exp(glast);
    for (uint p = tid; p < 8u * 128u; p += 512u) {
        const uint k = row0 + p / 128u;
        const uint v = p % 128u;
        const uint block = v / 8u;
        const uint product_off = block * 64u +
            (p / 128u) * 8u + v % 8u;
        const ulong off = ((ulong)h * d + v) * d + k;
        state[off] = decay * state[off] + product[product_off];
    }
}

kernel void kernel_qwen35_gdn_post(
        constant ds4_metal_args_qwen35_gdn &args,
        device const float *in,
        device const float *z,
        device const float *norm_weight,
        device float *out,
        threadgroup float *scratch [[threadgroup(0)]],
        uint tid [[thread_index_in_threadgroup]],
        uint3 tgpig [[threadgroup_position_in_grid]]) {
    const uint h = tgpig.x;
    const uint token = tgpig.y;
    if (h >= args.v_heads || token >= args.n_tokens ||
        tid >= args.state_dim) return;
    const uint off = (token * args.v_heads + h) * args.state_dim + tid;
    const float x = in[off];
    scratch[tid] = x * x;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint step = args.state_dim >> 1; step != 0; step >>= 1) {
        if (tid < step) scratch[tid] += scratch[tid + step];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    const float scale = rsqrt(scratch[0] / (float)args.state_dim + args.eps);
    out[off] = x * scale * norm_weight[tid] * qwen35_silu(z[off]);
}
