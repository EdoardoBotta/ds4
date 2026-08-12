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

// Extended-WY form of the gated delta recurrence (Yang et al., 2024).
// Qwen prefill is capped at 16 tokens, so the small causal system is formed
// and inverted once per value head in threadgroup memory.  `cumulative_g`
// stores log cumulative decays; differences are exponentiated instead of
// dividing products of decays, which avoids avoidable underflow.
kernel void kernel_qwen35_gdn_chunk_prepare(
        constant ds4_metal_args_qwen35_gdn &args,
        device const float *prepared,
        device const float *g,
        device const float *b,
        device float *w,
        device float *u,
        device float *qk,
        device float *cumulative_g,
        threadgroup float *scratch [[threadgroup(0)]],
        uint tid [[thread_index_in_threadgroup]],
        uint3 tgpig [[threadgroup_position_in_grid]]) {
    constexpr uint chunk_max = 16u;
    const uint h = tgpig.x;
    const uint n = args.n_tokens;
    const uint d = args.state_dim;
    const uint qkh = h % args.qk_heads;
    threadgroup float *cg = scratch;
    threadgroup float *tri = scratch + chunk_max;

    if (tid == 0u) {
        float sum = 0.0f;
        for (uint i = 0; i < n; i++) {
            sum += g[i * args.v_heads + h];
            cg[i] = sum;
            cumulative_g[i * args.v_heads + h] = sum;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // A = I + strictly_lower(beta_i exp(G_i-G_j) K_i K_j^T).
    for (uint p = tid; p < n * n; p += 128u) {
        const uint i = p / n;
        const uint j = p - i * n;
        float x = i == j ? 1.0f : 0.0f;
        if (i > j) {
            const device float *ki = prepared +
                (ulong)i * args.channels + (args.qk_heads + qkh) * d;
            const device float *kj = prepared +
                (ulong)j * args.channels + (args.qk_heads + qkh) * d;
            float dot = 0.0f;
            for (uint col = 0; col < d; col++) dot += ki[col] * kj[col];
            x = b[i * args.v_heads + h] * exp(cg[i] - cg[j]) * dot;
        }
        tri[i * chunk_max + j] = x;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Invert the unit lower-triangular matrix in place.  Columns are visited
    // in ascending order so entries of A needed later in the row are intact.
    if (tid == 0u) {
        for (uint i = 1u; i < n; i++) {
            for (uint j = 0u; j < i; j++) {
                float x = 0.0f;
                for (uint k = j; k < i; k++) {
                    x += tri[i * chunk_max + k] *
                         tri[k * chunk_max + j];
                }
                tri[i * chunk_max + j] = -x;
            }
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // W = A^-1 diag(beta exp(G)) K; U = A^-1 diag(beta) V.
    for (uint p = tid; p < n * d; p += 128u) {
        const uint i = p / d;
        const uint col = p - i * d;
        float wx = 0.0f;
        float ux = 0.0f;
        for (uint j = 0u; j <= i; j++) {
            const float tij = tri[i * chunk_max + j];
            const float bj = b[j * args.v_heads + h];
            const device float *token = prepared + (ulong)j * args.channels;
            const device float *kj = token +
                (args.qk_heads + qkh) * d;
            const device float *vj = token +
                2u * args.qk_heads * d + h * d;
            wx += tij * bj * exp(cg[j]) * kj[col];
            ux += tij * bj * vj[col];
        }
        const ulong off = ((ulong)i * args.v_heads + h) * d + col;
        w[off] = wx;
        u[off] = ux;
    }

    // Causal QK coefficients used by every value row in the output kernel.
    for (uint p = tid; p < n * n; p += 128u) {
        const uint i = p / n;
        const uint j = p - i * n;
        float x = 0.0f;
        if (j <= i) {
            const device float *qi = prepared +
                (ulong)i * args.channels + qkh * d;
            const device float *kj = prepared +
                (ulong)j * args.channels + (args.qk_heads + qkh) * d;
            for (uint col = 0; col < d; col++) x += qi[col] * kj[col];
            x *= exp(cg[i] - cg[j]);
        }
        qk[((ulong)i * args.v_heads + h) * n + j] = x;
    }
}

kernel void kernel_qwen35_gdn_chunk_values(
        constant ds4_metal_args_qwen35_gdn &args,
        device const float *w,
        device const float *u,
        device const float *state,
        device float *values,
        uint gid [[thread_position_in_grid]]) {
    const uint inner = args.v_heads * args.state_dim;
    const uint total = args.n_tokens * inner;
    if (gid >= total) return;
    const uint token = gid / inner;
    const uint rem = gid - token * inner;
    const uint h = rem / args.state_dim;
    const uint row = rem - h * args.state_dim;
    const device float *wi = w + ((ulong)token * args.v_heads + h) * args.state_dim;
    const device float *s = state +
        ((ulong)h * args.state_dim + row) * args.state_dim;
    float x = u[gid];
    for (uint col = 0; col < args.state_dim; col++) x -= wi[col] * s[col];
    values[gid] = x;
}

kernel void kernel_qwen35_gdn_chunk_output(
        constant ds4_metal_args_qwen35_gdn &args,
        device const float *prepared,
        device const float *qk,
        device const float *cumulative_g,
        device const float *state,
        device const float *values,
        device float *out,
        uint gid [[thread_position_in_grid]]) {
    const uint inner = args.v_heads * args.state_dim;
    const uint total = args.n_tokens * inner;
    if (gid >= total) return;
    const uint token = gid / inner;
    const uint rem = gid - token * inner;
    const uint h = rem / args.state_dim;
    const uint row = rem - h * args.state_dim;
    const uint qkh = h % args.qk_heads;
    const device float *q = prepared +
        (ulong)token * args.channels + qkh * args.state_dim;
    const device float *s = state +
        ((ulong)h * args.state_dim + row) * args.state_dim;
    float x = 0.0f;
    for (uint col = 0; col < args.state_dim; col++) x += s[col] * q[col];
    x *= exp(cumulative_g[token * args.v_heads + h]);
    const device float *coeff = qk +
        ((ulong)token * args.v_heads + h) * args.n_tokens;
    for (uint j = 0; j <= token; j++) {
        x += coeff[j] * values[((ulong)j * args.v_heads + h) *
                               args.state_dim + row];
    }
    out[gid] = x * args.scale;
}

kernel void kernel_qwen35_gdn_chunk_state(
        constant ds4_metal_args_qwen35_gdn &args,
        device const float *prepared,
        device const float *cumulative_g,
        device const float *values,
        device float *state,
        uint gid [[thread_position_in_grid]]) {
    const uint plane = args.state_dim * args.state_dim;
    const uint total = args.v_heads * plane;
    if (gid >= total) return;
    const uint h = gid / plane;
    const uint rem = gid - h * plane;
    const uint row = rem / args.state_dim;
    const uint col = rem - row * args.state_dim;
    const uint qkh = h % args.qk_heads;
    const uint last = args.n_tokens - 1u;
    const float glast = cumulative_g[last * args.v_heads + h];
    float x = exp(glast) * state[gid];
    for (uint j = 0; j < args.n_tokens; j++) {
        const device float *kj = prepared +
            (ulong)j * args.channels + (args.qk_heads + qkh) * args.state_dim;
        const float vj = values[((ulong)j * args.v_heads + h) *
                                args.state_dim + row];
        x += exp(glast - cumulative_g[j * args.v_heads + h]) * vj * kj[col];
    }
    state[gid] = x;
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
