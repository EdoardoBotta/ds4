// DS4 Metal matvec kernels used by generation.

constant short FC_mul_mv_nsg   [[function_constant(FC_MUL_MV + 0)]];
constant short FC_mul_mv_nxpsg [[function_constant(FC_MUL_MV + 1)]];

struct ds4_metal_args_mul_mv {
    int ne00;
    int ne01;
    int ne02;
    ulong nb00;
    ulong nb01;
    ulong nb02;
    ulong nb03;
    int ne10;
    int ne11;
    int ne12;
    ulong nb10;
    ulong nb11;
    ulong nb12;
    ulong nb13;
    int ne0;
    int ne1;
    int nr0;
    short r2;
    short r3;
};

struct ds4_metal_args_mul_mm {
    int32_t ne00;
    int32_t ne02;
    uint64_t nb01;
    uint64_t nb02;
    uint64_t nb03;
    int32_t ne12;
    uint64_t nb10;
    uint64_t nb11;
    uint64_t nb12;
    uint64_t nb13;
    int32_t ne0;
    int32_t ne1;
    int16_t r2;
    int16_t r3;
};

struct ds4_metal_args_mul_mv_ext {
    int32_t ne00;
    int32_t ne01;
    int32_t ne02;
    uint64_t nb00;
    uint64_t nb01;
    uint64_t nb02;
    uint64_t nb03;
    int32_t ne10;
    int32_t ne11;
    int32_t ne12;
    uint64_t nb10;
    uint64_t nb11;
    uint64_t nb12;
    uint64_t nb13;
    int32_t ne0;
    int32_t ne1;
    int16_t r2;
    int16_t r3;
};

template<short NR0>
static inline void helper_mv_reduce_and_write(
        device float * dst_f32,
        float sumf[NR0],
        const int r0,
        const int ne01,
        ushort tiisg,
        ushort sgitg,
        threadgroup char * shmem) {
    constexpr short NW = N_SIMDWIDTH;

    threadgroup float * shmem_f32[NR0];

    for (short row = 0; row < NR0; ++row) {
        shmem_f32[row] = (threadgroup float *) shmem + NW*row;

        if (sgitg == 0) {
            shmem_f32[row][tiisg] = 0.0f;
        }

        sumf[row] = simd_sum(sumf[row]);
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (short row = 0; row < NR0; ++row) {
        if (tiisg == 0) {
            shmem_f32[row][sgitg] = sumf[row];
        }
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (short row = 0; row < NR0 && r0 + row < ne01; ++row) {
        float tot = simd_sum(shmem_f32[row][tiisg]);

        if (tiisg == 0 && sgitg == 0) {
            dst_f32[r0 + row] = tot;
        }
    }
}

template<short NR0, typename args_t>
void kernel_mul_mv_q8_0_f32_impl(
        args_t args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        threadgroup  char * shmem,
        uint3  tgpig,
        ushort tiisg,
        ushort sgitg) {
    const short NSG = FC_mul_mv_nsg;

    constexpr short NW = N_SIMDWIDTH;
    constexpr short NQ = 8;

    const int nb = args.ne00/QK8_0;

    const int r0 = tgpig.x*NR0;
    const int r1 = tgpig.y;
    const int im = tgpig.z;

    const uint i12 = im%args.ne12;
    const uint i13 = im/args.ne12;

    const uint64_t offset1 = r1*args.nb11 + (i12)*args.nb12 + (i13)*args.nb13;

    device const float * y = (device const float *) (src1 + offset1);

    device const block_q8_0 * ax[NR0];
    FOR_UNROLL (short row = 0; row < NR0; ++row) {
        const uint64_t offset0 = (r0 + row)*args.nb01 + (i12/args.r2)*args.nb02 + (i13/args.r3)*args.nb03;

        ax[row] = (device const block_q8_0 *) ((device char *) src0 + offset0);
    }

    float sumf[NR0] = { 0.f };

    const short ix = tiisg/(NW/NQ);
    const short il = tiisg%(NW/NQ);

    const int ib0 = sgitg*NQ + ix;

    float yl[NQ];

    device const float * yb = y + ib0*QK8_0 + il*NQ;

    for (int ib = ib0; ib < nb; ib += NSG*NQ) {
        for (short i = 0; i < NQ; ++i) {
            yl[i] = yb[i];
        }

        for (short row = 0; row < NR0; row++) {
            device const int8_t * qs = ax[row][ib].qs + il*NQ;

            float sumq = 0.f;
            FOR_UNROLL (short i = 0; i < NQ; ++i) {
                sumq += qs[i] * yl[i];
            }

            sumf[row] += sumq*ax[row][ib].d;
        }

        yb += NSG*NQ*QK8_0;
    }

    device float * dst_f32 = (device float *) dst + (uint64_t)im*args.ne0*args.ne1 + (uint64_t)r1*args.ne0;

    helper_mv_reduce_and_write<NR0>(dst_f32, sumf, r0, args.ne01, tiisg, sgitg, shmem);
}

// Decode-time Q8_0 matrix-vector multiply. DS4 uses this for Q8_0 dense
// projections such as shared experts and output-side small matvecs.
[[host_name("kernel_mul_mv_q8_0_f32")]]
kernel void kernel_mul_mv_q8_0_f32(
        constant ds4_metal_args_mul_mv & args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        threadgroup  char * shmem [[threadgroup(0)]],
        uint3  tgpig[[threadgroup_position_in_grid]],
        ushort tiisg[[thread_index_in_simdgroup]],
        ushort sgitg[[simdgroup_index_in_threadgroup]]) {
    kernel_mul_mv_q8_0_f32_impl<N_R0_Q8_0, constant ds4_metal_args_mul_mv &>(args, src0, src1, dst, shmem, tgpig, tiisg, sgitg);
}

struct ds4_metal_args_output_sample {
    uint32_t in_dim;
    uint32_t out_dim;
    uint32_t n_blocks;
    uint32_t top_k;
    float    temperature;
    uint64_t seed;
    uint32_t mode;
    uint32_t n_groups;
};

static inline uint64_t ds4_output_sample_mix64(uint64_t x) {
    x ^= x >> 30;
    x *= 0xbf58476d1ce4e5b9ul;
    x ^= x >> 27;
    x *= 0x94d049bb133111ebul;
    x ^= x >> 31;
    return x;
}

static inline float ds4_output_sample_gumbel(uint64_t seed, uint32_t id) {
    const uint64_t x = ds4_output_sample_mix64(seed ^ ((uint64_t)id * 0x9e3779b97f4a7c15ul));
    float u = (float)((x >> 40) & 0xffffffu) * (1.0f / 16777216.0f);
    u = clamp(u, 5.9604644775390625e-8f, 0.9999999403953552f);
    return -log(-log(u));
}

static inline bool ds4_output_sample_better(float av, uint32_t ai, float bv, uint32_t bi) {
    return av > bv || (av == bv && ai < bi);
}

static inline uint32_t ds4_output_sample_ordered_float(float v) {
    if (isnan(v)) v = -INFINITY;
    const uint32_t bits = as_type<uint32_t>(v);
    return (bits & 0x80000000u) != 0u ? ~bits : bits ^ 0x80000000u;
}

// Each threadgroup projects a grid-stride set of Q8_0 vocabulary-row pairs,
// retains its local winner, then contributes once to the global score winner.
kernel void kernel_dsv4_output_sample_atomic_q8_0(
        constant ds4_metal_args_output_sample &args,
        device const char  *weight,
        device const float *x,
        device atomic_uint *winner_score,
        device uint32_t    *group_scores,
        device uint32_t    *group_ids,
        threadgroup char   *shmem [[threadgroup(0)]],
        uint group [[threadgroup_position_in_grid]],
        ushort tiisg [[thread_index_in_simdgroup]],
        ushort sgitg [[simdgroup_index_in_threadgroup]]) {
    constexpr short NW = N_SIMDWIDTH;
    constexpr short NQ = 8;
    constexpr short NSG = N_SG_Q8_0;
    const uint64_t row_bytes = (uint64_t)args.n_blocks * sizeof(block_q8_0);
    const short ix = tiisg / (NW / NQ);
    const short il = tiisg % (NW / NQ);
    const int ib0 = sgitg * NQ + ix;
    threadgroup float *s0 = (threadgroup float *)shmem;
    threadgroup float *s1 = s0 + NW;
    float group_best = -INFINITY;
    uint32_t group_best_id = 0xffffffffu;
    const uint32_t pair_count = (args.out_dim + 1u) >> 1;
    const float inv_t = 1.0f / max(args.temperature, 1.0e-20f);

    for (uint32_t pair = group; pair < pair_count; pair += args.n_groups) {
        const uint32_t row0 = pair * 2u;
        const uint32_t row1 = row0 + 1u;
        device const block_q8_0 *w0 =
            (device const block_q8_0 *)(weight + (uint64_t)row0 * row_bytes);
        device const block_q8_0 *w1 = row1 < args.out_dim
            ? (device const block_q8_0 *)(weight + (uint64_t)row1 * row_bytes)
            : (device const block_q8_0 *)weight;

        float sumf[2] = {0.0f, 0.0f};
        device const float *yb = x + (uint64_t)ib0 * QK8_0 + il * NQ;
        for (uint32_t ib = ib0; ib < args.n_blocks; ib += NSG * NQ) {
            float yl[NQ];
            for (short i = 0; i < NQ; i++) yl[i] = yb[i];
            {
                device const int8_t *qs = w0[ib].qs + il * NQ;
                float sumq = 0.0f;
                for (short i = 0; i < NQ; i++) sumq += qs[i] * yl[i];
                sumf[0] += sumq * (float)w0[ib].d;
            }
            if (row1 < args.out_dim) {
                device const int8_t *qs = w1[ib].qs + il * NQ;
                float sumq = 0.0f;
                for (short i = 0; i < NQ; i++) sumq += qs[i] * yl[i];
                sumf[1] += sumq * (float)w1[ib].d;
            }
            yb += NSG * NQ * QK8_0;
        }

        if (sgitg == 0) {
            s0[tiisg] = 0.0f;
            s1[tiisg] = 0.0f;
        }
        sumf[0] = simd_sum(sumf[0]);
        sumf[1] = simd_sum(sumf[1]);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (tiisg == 0) {
            s0[sgitg] = sumf[0];
            s1[sgitg] = sumf[1];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        float v0 = simd_sum(s0[tiisg]);
        float v1 = simd_sum(s1[tiisg]);
        if (tiisg == 0 && sgitg == 0) {
            if (args.mode == 1u) {
                v0 = v0 * inv_t + ds4_output_sample_gumbel(args.seed, row0);
                v1 = row1 < args.out_dim
                    ? v1 * inv_t + ds4_output_sample_gumbel(args.seed, row1)
                    : -INFINITY;
            }
            if (isnan(v0)) v0 = -INFINITY;
            if (isnan(v1)) v1 = -INFINITY;
            float pair_best = v0;
            uint32_t pair_best_id = row0;
            if (row1 < args.out_dim &&
                ds4_output_sample_better(v1, row1, pair_best, pair_best_id)) {
                pair_best = v1;
                pair_best_id = row1;
            }
            if (ds4_output_sample_better(pair_best, pair_best_id,
                                         group_best, group_best_id)) {
                group_best = pair_best;
                group_best_id = pair_best_id;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (tiisg == 0 && sgitg == 0 && group_best_id != 0xffffffffu) {
        const uint32_t ordered_score = ds4_output_sample_ordered_float(group_best);
        group_scores[group] = ordered_score;
        group_ids[group] = group_best_id;
        atomic_fetch_max_explicit(winner_score, ordered_score, memory_order_relaxed);
    }
}

// Computes two Q8_0 output rows per threadgroup and writes raw logits for the
// top-k reducer. This is still private scratch, not the public logits tensor.
kernel void kernel_dsv4_output_sample_scores2_q8_0(
        constant ds4_metal_args_output_sample &args,
        device const char  *weight,
        device const float *x,
        device float       *scores,
        threadgroup char   *shmem [[threadgroup(0)]],
        uint pair [[threadgroup_position_in_grid]],
        ushort tiisg [[thread_index_in_simdgroup]],
        ushort sgitg [[simdgroup_index_in_threadgroup]]) {
    constexpr short NW = N_SIMDWIDTH;
    constexpr short NQ = 8;
    constexpr short NSG = N_SG_Q8_0;
    const uint32_t row0 = pair * 2u;
    const uint32_t row1 = row0 + 1u;
    const uint64_t row_bytes = (uint64_t)args.n_blocks * sizeof(block_q8_0);
    device const block_q8_0 *w0 = row0 < args.out_dim
        ? (device const block_q8_0 *)(weight + (uint64_t)row0 * row_bytes)
        : (device const block_q8_0 *)weight;
    device const block_q8_0 *w1 = row1 < args.out_dim
        ? (device const block_q8_0 *)(weight + (uint64_t)row1 * row_bytes)
        : (device const block_q8_0 *)weight;

    float sumf[2] = {0.0f, 0.0f};
    const short ix = tiisg / (NW / NQ);
    const short il = tiisg % (NW / NQ);
    const int ib0 = sgitg * NQ + ix;
    device const float *yb = x + (uint64_t)ib0 * QK8_0 + il * NQ;
    for (uint32_t ib = ib0; ib < args.n_blocks; ib += NSG * NQ) {
        float yl[NQ];
        for (short i = 0; i < NQ; i++) yl[i] = yb[i];
        if (row0 < args.out_dim) {
            device const int8_t *qs = w0[ib].qs + il * NQ;
            float sumq = 0.0f;
            for (short i = 0; i < NQ; i++) sumq += qs[i] * yl[i];
            sumf[0] += sumq * (float)w0[ib].d;
        }
        if (row1 < args.out_dim) {
            device const int8_t *qs = w1[ib].qs + il * NQ;
            float sumq = 0.0f;
            for (short i = 0; i < NQ; i++) sumq += qs[i] * yl[i];
            sumf[1] += sumq * (float)w1[ib].d;
        }
        yb += NSG * NQ * QK8_0;
    }

    threadgroup float *s0 = (threadgroup float *)shmem;
    threadgroup float *s1 = s0 + NW;
    if (sgitg == 0) {
        s0[tiisg] = 0.0f;
        s1[tiisg] = 0.0f;
    }
    sumf[0] = simd_sum(sumf[0]);
    sumf[1] = simd_sum(sumf[1]);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tiisg == 0) {
        s0[sgitg] = sumf[0];
        s1[sgitg] = sumf[1];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float v0 = simd_sum(s0[tiisg]);
    float v1 = simd_sum(s1[tiisg]);
    if (tiisg == 0 && sgitg == 0) {
        if (row0 < args.out_dim) scores[row0] = v0;
        if (row1 < args.out_dim) scores[row1] = v1;
    }
}

kernel void kernel_dsv4_output_sample_topk(
        constant ds4_metal_args_output_sample &args,
        device const float *scores,
        device int32_t     *out_idx,
        threadgroup float    *score_mem [[threadgroup(0)]],
        threadgroup uint32_t *id_mem [[threadgroup(1)]],
        threadgroup float    *top_vals [[threadgroup(2)]],
        threadgroup uint32_t *top_ids [[threadgroup(3)]],
        uint tid [[thread_position_in_threadgroup]],
        uint ntg [[threads_per_threadgroup]]) {
    uint32_t k = min(args.top_k, args.out_dim);
    k = min(k, 1024u);
    if (k == 0u) {
        if (tid == 0) out_idx[0] = 0;
        return;
    }

    for (uint32_t i = tid; i < k; i += ntg) {
        top_vals[i] = -INFINITY;
        top_ids[i] = 0xffffffffu;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float prev_v = INFINITY;
    uint32_t prev_id = 0u;
    for (uint32_t rank = 0; rank < k; rank++) {
        float best = -INFINITY;
        uint32_t best_id = 0xffffffffu;
        for (uint32_t id = tid; id < args.out_dim; id += ntg) {
            const float v = scores[id];
            if (!isfinite(v)) continue;
            if (rank != 0u && !(v < prev_v || (v == prev_v && id > prev_id))) continue;
            if (ds4_output_sample_better(v, id, best, best_id)) {
                best = v;
                best_id = id;
            }
        }

        score_mem[tid] = best;
        id_mem[tid] = best_id;
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint stride = ntg >> 1; stride > 0; stride >>= 1) {
            if (tid < stride &&
                ds4_output_sample_better(score_mem[tid + stride],
                                         id_mem[tid + stride],
                                         score_mem[tid],
                                         id_mem[tid])) {
                score_mem[tid] = score_mem[tid + stride];
                id_mem[tid] = id_mem[tid + stride];
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }

        if (tid == 0) {
            top_vals[rank] = score_mem[0];
            top_ids[rank] = id_mem[0];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (top_ids[rank] == 0xffffffffu) break;
        prev_v = top_vals[rank];
        prev_id = top_ids[rank];
    }

    const float inv_t = 1.0f / max(args.temperature, 1.0e-20f);
    float best = -INFINITY;
    uint32_t best_id = 0xffffffffu;
    for (uint32_t i = tid; i < k; i += ntg) {
        const uint32_t id = top_ids[i];
        if (id == 0xffffffffu) continue;
        const float v = top_vals[i] * inv_t + ds4_output_sample_gumbel(args.seed, id);
        if (ds4_output_sample_better(v, id, best, best_id)) {
            best = v;
            best_id = id;
        }
    }

    score_mem[tid] = best;
    id_mem[tid] = best_id;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint stride = ntg >> 1; stride > 0; stride >>= 1) {
        if (tid < stride &&
            ds4_output_sample_better(score_mem[tid + stride],
                                     id_mem[tid + stride],
                                     score_mem[tid],
                                     id_mem[tid])) {
            score_mem[tid] = score_mem[tid + stride];
            id_mem[tid] = id_mem[tid + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (tid == 0) out_idx[0] = id_mem[0] == 0xffffffffu ? 0 : (int32_t)id_mem[0];
}

// Decode shared-expert gate/up projections followed by SwiGLU:
//
//     mid = silu(min(gate, limit)) * clamp(up, -limit, limit)
//
// DS4's shared expert uses two Q8_0 matrices with the same input row.  This
// kernel preserves the exact Q8_0 dot-product reduction shape for both
// projections, still writes gate/up for diagnostics, and derives `mid` in the
// same lane that owns the reduced output row.  The point is not to fuse two
// independent weight streams into one matmul; it is to remove the separate
// activation pass and its reread of the two 2048-wide rows.
[[host_name("kernel_dsv4_shared_gate_up_swiglu_q8_0")]]
kernel void kernel_dsv4_shared_gate_up_swiglu_q8_0(
        constant ds4_metal_args_mul_mv & args,
        device const char * src0_gate,
        device const char * src0_up,
        device const char * src1,
        device       char * dst_gate,
        device       char * dst_up,
        device       char * dst_mid,
        constant     float &clamp_value,
        threadgroup  char * shmem [[threadgroup(0)]],
        uint3  tgpig[[threadgroup_position_in_grid]],
        ushort tiisg[[thread_index_in_simdgroup]],
        ushort sgitg[[simdgroup_index_in_threadgroup]]) {
    const short NSG = FC_mul_mv_nsg;
    constexpr short NW = N_SIMDWIDTH;
    constexpr short NQ = 8;
    constexpr short NR0 = N_R0_Q8_0;

    const int nb = args.ne00 / QK8_0;
    const int r0 = tgpig.x * NR0;
    const int r1 = tgpig.y;
    const int im = tgpig.z;

    const uint i12 = im % args.ne12;
    const uint i13 = im / args.ne12;
    const uint64_t offset1 = r1 * args.nb11 + i12 * args.nb12 + i13 * args.nb13;
    device const float *y = (device const float *)(src1 + offset1);

    device const block_q8_0 *ag[NR0];
    device const block_q8_0 *au[NR0];
    FOR_UNROLL (short row = 0; row < NR0; ++row) {
        const uint64_t offset0 = (r0 + row) * args.nb01 +
                                 (i12 / args.r2) * args.nb02 +
                                 (i13 / args.r3) * args.nb03;
        ag[row] = (device const block_q8_0 *)((device const char *)src0_gate + offset0);
        au[row] = (device const block_q8_0 *)((device const char *)src0_up   + offset0);
    }

    float sumg[NR0] = { 0.f };
    float sumu[NR0] = { 0.f };

    const short ix = tiisg / (NW / NQ);
    const short il = tiisg % (NW / NQ);
    const int ib0 = sgitg * NQ + ix;
    float yl[NQ];
    device const float *yb = y + ib0 * QK8_0 + il * NQ;

    for (int ib = ib0; ib < nb; ib += NSG * NQ) {
        FOR_UNROLL (short i = 0; i < NQ; ++i) {
            yl[i] = yb[i];
        }

        FOR_UNROLL (short row = 0; row < NR0; ++row) {
            device const int8_t *qg = ag[row][ib].qs + il * NQ;
            device const int8_t *qu = au[row][ib].qs + il * NQ;

            float sg = 0.f;
            float su = 0.f;
            FOR_UNROLL (short i = 0; i < NQ; ++i) {
                sg += qg[i] * yl[i];
                su += qu[i] * yl[i];
            }

            sumg[row] += sg * ag[row][ib].d;
            sumu[row] += su * au[row][ib].d;
        }

        yb += NSG * NQ * QK8_0;
    }

    threadgroup float *shmem_f32 = (threadgroup float *)shmem;
    threadgroup float *sh_gate[NR0];
    threadgroup float *sh_up[NR0];
    FOR_UNROLL (short row = 0; row < NR0; ++row) {
        sh_gate[row] = shmem_f32 + NW * row;
        sh_up[row]   = shmem_f32 + NW * (NR0 + row);
        if (sgitg == 0) {
            sh_gate[row][tiisg] = 0.0f;
            sh_up[row][tiisg] = 0.0f;
        }
        sumg[row] = simd_sum(sumg[row]);
        sumu[row] = simd_sum(sumu[row]);
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    FOR_UNROLL (short row = 0; row < NR0; ++row) {
        if (tiisg == 0) {
            sh_gate[row][sgitg] = sumg[row];
            sh_up[row][sgitg] = sumu[row];
        }
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    device float *gate_f32 = (device float *)dst_gate +
        (uint64_t)im * args.ne0 * args.ne1 + (uint64_t)r1 * args.ne0;
    device float *up_f32 = (device float *)dst_up +
        (uint64_t)im * args.ne0 * args.ne1 + (uint64_t)r1 * args.ne0;
    device float *mid_f32 = (device float *)dst_mid +
        (uint64_t)im * args.ne0 * args.ne1 + (uint64_t)r1 * args.ne0;

    FOR_UNROLL (short row = 0; row < NR0 && r0 + row < args.ne01; ++row) {
        const float gate = simd_sum(sh_gate[row][tiisg]);
        const float up = simd_sum(sh_up[row][tiisg]);
        if (tiisg == 0 && sgitg == 0) {
            const uint out_row = r0 + row;
            gate_f32[out_row] = gate;
            up_f32[out_row] = up;
            float g = gate;
            float u = up;
            if (clamp_value > 1.0e-6f) {
                g = min(g, clamp_value);
                u = clamp(u, -clamp_value, clamp_value);
            }
            const float silu = g / (1.0f + exp(-g));
            mid_f32[out_row] = silu * u;
        }
    }
}

template<typename T0, typename T1, short NR0, typename args_t>
void kernel_mul_mv_t_t_impl(
        args_t args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        threadgroup  char * shmem,
        uint3  tgpig,
        ushort tiisg,
        ushort sgitg) {
    const short NSG = FC_mul_mv_nsg;

    constexpr short NW = N_SIMDWIDTH;
    constexpr short NB = 32;
    constexpr short NF = 8;

    const int nb = args.ne00/NB;

    const int r0 = tgpig.x*NR0;
    const int r1 = tgpig.y;
    const int im = tgpig.z;

    const uint i12 = im%args.ne12;
    const uint i13 = im/args.ne12;

    const uint64_t offset1 = r1*args.nb11 + (i12)*args.nb12 + (i13)*args.nb13;

    device const T1 * y = (device const T1 *) (src1 + offset1);

    device const T0 * ax[NR0];
    FOR_UNROLL (short row = 0; row < NR0; ++row) {
        const uint64_t offset0 = (r0 + row)*args.nb01 + (i12/args.r2)*args.nb02 + (i13/args.r3)*args.nb03;

        ax[row] = (device const T0 *) ((device char *) src0 + offset0);
    }

    float sumf[NR0] = { 0.f };

    const short ix = tiisg/(NW/NF);
    const short il = tiisg%(NW/NF);

    const int ib0 = sgitg*NF + ix;

    T1 yl[NF];

    device const T1 * yb = y + (ib0*NB + il*NF);

    for (int ib = ib0; ib < nb; ib += NSG*NF) {
        for (short i = 0; i < NF; ++i) {
            yl[i] = yb[i];
        }

        for (short row = 0; row < NR0; row++) {
            device const T0 * xb = ax[row] + (ib*NB + il*NF);

            float sumq = 0.f;
            FOR_UNROLL (short i = 0; i < NF; ++i) {
                sumq += xb[i] * yl[i];
            }

            sumf[row] += sumq;
        }

        yb += NSG*NF*NW;
    }

    for (int i = nb*NB + sgitg*NW + tiisg; i < args.ne00; i += NW*NSG) {
        for (short row = 0; row < NR0; row++) {
            sumf[row] += ax[row][i] * y[i];
        }
    }

    device float * dst_f32 = (device float *) dst + (uint64_t)im*args.ne0*args.ne1 + (uint64_t)r1*args.ne0;

    helper_mv_reduce_and_write<NR0>(dst_f32, sumf, r0, args.ne01, tiisg, sgitg, shmem);
}

template<typename T0, typename T1, typename args_t>
void kernel_mul_mv_t_t_disp(
        args_t args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        threadgroup  char * shmem,
        uint3  tgpig,
        ushort tiisg,
        ushort sgitg) {
    switch (args.nr0) {
        case 2: kernel_mul_mv_t_t_impl<T0, T1, 2, args_t>(args, src0, src1, dst, shmem, tgpig, tiisg, sgitg); break;
        case 4: kernel_mul_mv_t_t_impl<T0, T1, 4, args_t>(args, src0, src1, dst, shmem, tgpig, tiisg, sgitg); break;
    }
}

// Decode-time dense F32/F16 matrix-vector multiply. The instantiated kernels
// handle unquantized DS4 weights and activations that are already float rows.
template<typename T0, typename T1>
kernel void kernel_mul_mv_t_t(
        constant ds4_metal_args_mul_mv & args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        threadgroup  char * shmem [[threadgroup(0)]],
        uint3  tgpig[[threadgroup_position_in_grid]],
        ushort tiisg[[thread_index_in_simdgroup]],
        ushort sgitg[[simdgroup_index_in_threadgroup]]) {
    kernel_mul_mv_t_t_disp<T0, T1, constant ds4_metal_args_mul_mv &>(args, src0, src1, dst, shmem, tgpig, tiisg, sgitg);
}

typedef decltype(kernel_mul_mv_t_t<half, half>) mul_mv_t_t;

// Host-visible dense matvec variants used by the graph for F32 and F16 weights.
template [[host_name("kernel_mul_mv_f32_f32")]] kernel mul_mv_t_t kernel_mul_mv_t_t<float, float>;
template [[host_name("kernel_mul_mv_f16_f32")]] kernel mul_mv_t_t kernel_mul_mv_t_t<half,  float>;

template<typename T0, typename T04, typename T1, typename T14, short NR0, typename args_t>
void kernel_mul_mv_t_t_4_impl(
        args_t args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        threadgroup  char * shmem,
        uint3  tgpig,
        ushort tiisg,
        ushort sgitg) {
    const short NSG = FC_mul_mv_nsg;

    constexpr short NW = N_SIMDWIDTH;
    constexpr short NB  = 32;
    constexpr short NF  = 16;
    constexpr short NF4 = NF/4;

    const int nb = args.ne00/NB;

    const int r0 = tgpig.x*NR0;
    const int r1 = tgpig.y;
    const int im = tgpig.z;

    const uint i12 = im%args.ne12;
    const uint i13 = im/args.ne12;

    const uint64_t offset1 = r1*args.nb11 + (i12)*args.nb12 + (i13)*args.nb13;

    device const T1  * y  = (device const T1  *) (src1 + offset1);
    device const T14 * y4 = (device const T14 *) (src1 + offset1);

    device const T0  * ax [NR0];
    device const T04 * ax4[NR0];
    FOR_UNROLL (short row = 0; row < NR0; ++row) {
        const uint64_t offset0 = (r0 + row)*args.nb01 + (i12/args.r2)*args.nb02 + (i13/args.r3)*args.nb03;

        ax [row] = (device const T0  *) ((device char *) src0 + offset0);
        ax4[row] = (device const T04 *) ((device char *) src0 + offset0);
    }

    float sumf[NR0] = { 0.f };

    const short ix = tiisg/(NW/NF);
    const short il = tiisg%(NW/NF);

    const int ib0 = sgitg*NF + ix;

    T14 yl4[NF4];

    device const T14 * yb4 = y4 + (ib0*NB + il*NF)/4;

    for (int ib = ib0; ib < nb; ib += NSG*NF) {
        for (short i = 0; i < NF4; ++i) {
            yl4[i] = yb4[i];
        }

        for (short row = 0; row < NR0; row++) {
            device const T04 * xb4 = ax4[row] + (ib*NB + il*NF)/4;

            float sumq = 0.f;
            FOR_UNROLL (short i = 0; i < NF4; ++i) {
                sumq += dot(float4(xb4[i]), float4(yl4[i]));
            }

            sumf[row] += sumq;
        }

        yb4 += NSG*NF*NW/4;
    }

    for (int i = nb*NB + sgitg*NW + tiisg; i < args.ne00; i += NW*NSG) {
        for (short row = 0; row < NR0; row++) {
            sumf[row] += ax[row][i] * y[i];
        }
    }

    device float * dst_f32 = (device float *) dst + (uint64_t)im*args.ne0*args.ne1 + (uint64_t)r1*args.ne0;

    helper_mv_reduce_and_write<NR0>(dst_f32, sumf, r0, args.ne01, tiisg, sgitg, shmem);
}

template<typename T0, typename T04, typename T1, typename T14, typename args_t>
void kernel_mul_mv_t_t_4_disp(
        args_t args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        threadgroup  char * shmem,
        uint3  tgpig,
        ushort tiisg,
        ushort sgitg) {
    switch (args.nr0) {
        case 2: kernel_mul_mv_t_t_4_impl<T0, T04, T1, T14, 2, args_t>(args, src0, src1, dst, shmem, tgpig, tiisg, sgitg); break;
        case 4: kernel_mul_mv_t_t_4_impl<T0, T04, T1, T14, 4, args_t>(args, src0, src1, dst, shmem, tgpig, tiisg, sgitg); break;
    };
}

// Vectorized dense matvec using float4/half4 loads. DS4 uses this where the
// inner dimension and alignment make vector loads cheaper than scalar lanes.
template<typename T0, typename T04, typename T1, typename T14>
kernel void kernel_mul_mv_t_t_4(
        constant ds4_metal_args_mul_mv & args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        threadgroup  char * shmem [[threadgroup(0)]],
        uint3  tgpig[[threadgroup_position_in_grid]],
        ushort tiisg[[thread_index_in_simdgroup]],
        ushort sgitg[[simdgroup_index_in_threadgroup]]) {
    kernel_mul_mv_t_t_4_disp<T0, T04, T1, T14, constant ds4_metal_args_mul_mv &>(args, src0, src1, dst, shmem, tgpig, tiisg, sgitg);
}

typedef decltype(kernel_mul_mv_t_t_4<half, half4, half, half4>) mul_mv_t_t_4;

// Host-visible vectorized dense matvec variants for F32 and F16 weights.
template [[host_name("kernel_mul_mv_f32_f32_4")]] kernel mul_mv_t_t_4 kernel_mul_mv_t_t_4<float, float4, float, float4>;
template [[host_name("kernel_mul_mv_f16_f32_4")]] kernel mul_mv_t_t_4 kernel_mul_mv_t_t_4<half,  half4,  float, float4>;

// DS4 compressor projections always compute two same-shaped F16 matvecs from
// the same normalized activation: one for projected KV and one for pooling
// scores.  This paired variant keeps the exact dense F16 row-reduction shape
// for each matrix, but shares one dispatch and one activation stream.
template<short NR0, typename args_t>
void kernel_mul_mv_f16_f32_pair_4_impl(
        args_t args,
        device const char * src0_a,
        device const char * src0_b,
        device const char * src1,
        device       char * dst_a,
        device       char * dst_b,
        threadgroup  char * shmem,
        uint3  tgpig,
        ushort tiisg,
        ushort sgitg) {
    const short NSG = FC_mul_mv_nsg;

    constexpr short NW = N_SIMDWIDTH;
    constexpr short NB  = 32;
    constexpr short NF  = 16;
    constexpr short NF4 = NF/4;

    const int nb = args.ne00/NB;

    const int r0 = tgpig.x*NR0;
    const int r1 = tgpig.y;
    const int im = tgpig.z;

    const uint i12 = im%args.ne12;
    const uint i13 = im/args.ne12;

    const uint64_t offset1 = r1*args.nb11 + (i12)*args.nb12 + (i13)*args.nb13;

    device const float  * y  = (device const float  *) (src1 + offset1);
    device const float4 * y4 = (device const float4 *) (src1 + offset1);

    device const half  * ax_a [NR0];
    device const half4 * ax4_a[NR0];
    device const half  * ax_b [NR0];
    device const half4 * ax4_b[NR0];
    FOR_UNROLL (short row = 0; row < NR0; ++row) {
        const uint64_t offset0 = (r0 + row)*args.nb01 + (i12/args.r2)*args.nb02 + (i13/args.r3)*args.nb03;

        ax_a [row] = (device const half  *) ((device char *) src0_a + offset0);
        ax4_a[row] = (device const half4 *) ((device char *) src0_a + offset0);
        ax_b [row] = (device const half  *) ((device char *) src0_b + offset0);
        ax4_b[row] = (device const half4 *) ((device char *) src0_b + offset0);
    }

    float sum_a[NR0] = { 0.f };
    float sum_b[NR0] = { 0.f };

    const short ix = tiisg/(NW/NF);
    const short il = tiisg%(NW/NF);

    const int ib0 = sgitg*NF + ix;

    float4 yl4[NF4];

    device const float4 * yb4 = y4 + (ib0*NB + il*NF)/4;

    for (int ib = ib0; ib < nb; ib += NSG*NF) {
        for (short i = 0; i < NF4; ++i) {
            yl4[i] = yb4[i];
        }

        for (short row = 0; row < NR0; row++) {
            device const half4 * xb4_a = ax4_a[row] + (ib*NB + il*NF)/4;
            device const half4 * xb4_b = ax4_b[row] + (ib*NB + il*NF)/4;

            float suma = 0.f;
            float sumb = 0.f;
            FOR_UNROLL (short i = 0; i < NF4; ++i) {
                const float4 yv = float4(yl4[i]);
                suma += dot(float4(xb4_a[i]), yv);
                sumb += dot(float4(xb4_b[i]), yv);
            }

            sum_a[row] += suma;
            sum_b[row] += sumb;
        }

        yb4 += NSG*NF*NW/4;
    }

    for (int i = nb*NB + sgitg*NW + tiisg; i < args.ne00; i += NW*NSG) {
        for (short row = 0; row < NR0; row++) {
            const float yi = y[i];
            sum_a[row] += ax_a[row][i] * yi;
            sum_b[row] += ax_b[row][i] * yi;
        }
    }

    device float * dst_a_f32 = (device float *) dst_a + (uint64_t)im*args.ne0*args.ne1 + (uint64_t)r1*args.ne0;
    device float * dst_b_f32 = (device float *) dst_b + (uint64_t)im*args.ne0*args.ne1 + (uint64_t)r1*args.ne0;

    helper_mv_reduce_and_write<NR0>(dst_a_f32, sum_a, r0, args.ne01, tiisg, sgitg, shmem);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    helper_mv_reduce_and_write<NR0>(dst_b_f32, sum_b, r0, args.ne01, tiisg, sgitg, shmem);
}

template<typename args_t>
void kernel_mul_mv_f16_f32_pair_4_disp(
        args_t args,
        device const char * src0_a,
        device const char * src0_b,
        device const char * src1,
        device       char * dst_a,
        device       char * dst_b,
        threadgroup  char * shmem,
        uint3  tgpig,
        ushort tiisg,
        ushort sgitg) {
    switch (args.nr0) {
        case 2: kernel_mul_mv_f16_f32_pair_4_impl<2>(args, src0_a, src0_b, src1, dst_a, dst_b, shmem, tgpig, tiisg, sgitg); break;
        case 4: kernel_mul_mv_f16_f32_pair_4_impl<4>(args, src0_a, src0_b, src1, dst_a, dst_b, shmem, tgpig, tiisg, sgitg); break;
    }
}

kernel void kernel_mul_mv_f16_f32_pair_4(
        constant ds4_metal_args_mul_mv & args,
        device const char * src0_a,
        device const char * src0_b,
        device const char * src1,
        device       char * dst_a,
        device       char * dst_b,
        threadgroup  char * shmem [[threadgroup(0)]],
        uint3  tgpig[[threadgroup_position_in_grid]],
        ushort tiisg[[thread_index_in_simdgroup]],
        ushort sgitg[[simdgroup_index_in_threadgroup]]) {
    kernel_mul_mv_f16_f32_pair_4_disp<constant ds4_metal_args_mul_mv &>(
            args, src0_a, src0_b, src1, dst_a, dst_b, shmem, tgpig, tiisg, sgitg);
}

template<typename T0, typename T1, typename args_t>
void kernel_mul_mv_t_t_short_impl(
        args_t args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        uint3  tgpig,
        ushort tiisg) {
    const int r0 = tgpig.x*32 + tiisg;
    const int r1 = tgpig.y;
    const int im = tgpig.z;

    if (r0 >= args.ne01) {
        return;
    }

    const uint i12 = im%args.ne12;
    const uint i13 = im/args.ne12;

    const uint64_t offset0 = r0*args.nb01 + (i12/args.r2)*args.nb02 + (i13/args.r3)*args.nb03;

    device const T0 * x = (device const T0 *) (src0 + offset0);

    device float * dst_f32 = (device float *) dst + (uint64_t)im*args.ne0*args.ne1;

    const uint64_t offset1 = r1*args.nb11 + (i12)*args.nb12 + (i13)*args.nb13;

    device const T1 * y = (device const T1 *) (src1 + offset1);

    float res = 0.0f;

    for (int i = 0; i < args.ne00; ++i) {
        res += (float) x[i] * (float) y[i];
    }

    dst_f32[(uint64_t)r1*args.ne0 + r0] = res;
}

// Scalar fallback for short rows. It trades parallelism for lower dispatch and
// reduction overhead when DS4 asks for tiny dense matvecs.
template<typename T0, typename T1>
kernel void kernel_mul_mv_t_t_short(
        constant ds4_metal_args_mul_mv & args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        uint3  tgpig[[threadgroup_position_in_grid]],
        ushort tiisg[[thread_index_in_simdgroup]]) {
    kernel_mul_mv_t_t_short_impl<T0, T1, constant ds4_metal_args_mul_mv &>(
        args,
        src0,
        src1,
        dst,
        tgpig,
        tiisg);
}

typedef decltype(kernel_mul_mv_t_t_short<half, half>) mul_mv_t_t_short_t;

// Host-visible short-row dense matvec variants.
template [[host_name("kernel_mul_mv_f32_f32_short")]] kernel mul_mv_t_t_short_t kernel_mul_mv_t_t_short<float, float>;
template [[host_name("kernel_mul_mv_f16_f32_short")]] kernel mul_mv_t_t_short_t kernel_mul_mv_t_t_short<half,  float>;

template <typename type4x4>
void dequantize_f32(device const float4x4 * src, short il, thread type4x4 & reg) {
    reg = (type4x4)(*src);
}

template <typename type4x4>
void dequantize_f16(device const half4x4 * src, short il, thread type4x4 & reg) {
    reg = (type4x4)(*src);
}

template <typename type4x4>
void dequantize_q8_0(device const block_q8_0 *xb, short il, thread type4x4 & reg) {
    device const int8_t * qs = ((device const int8_t *)xb->qs);
    const float d = xb->d;

    float4x4 reg_f;

    for (int i = 0; i < 16; i++) {
        reg_f[i/4][i%4] = (qs[i + 16*il] * d);
    }

    reg = (type4x4) reg_f;
}

template <typename type4>
void dequantize_q8_0_t4(device const block_q8_0 *xb, short il, thread type4 & reg) {
    device const int8_t * qs = ((device const int8_t *)xb->qs);
    const float d = xb->d;

    for (int i = 0; i < 4; i++) {
        reg[i] = (qs[4*(il%4) + i + 16*(il/4)] * d);
    }
}

// DS4 small-batch mat-vec kernel used for 2..8 prompt tokens.
template<short r1ptg, typename q_t, short chpb, void (*deq_t4)(device const q_t *, short, thread float4 &) >
void kernel_mul_mv_ext_q4_f32_impl(
        constant ds4_metal_args_mul_mv_ext & args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        uint3   tgpig[[threadgroup_position_in_grid]],
        ushort  tiisg[[thread_index_in_simdgroup]],
        ushort  sgitg[[simdgroup_index_in_threadgroup]]) {
    const short NSG   = FC_mul_mv_nsg;
    const short nxpsg = FC_mul_mv_nxpsg;

    const short chpt = 4; // chunks per thread

    const short nypsg = (32/nxpsg);

    const short tx = tiisg%nxpsg;
    const short ty = tiisg/nxpsg;

    const int i01 = tgpig.x*(nypsg*NSG) + nypsg*sgitg + ty;
    const int i11 = tgpig.y*r1ptg;
    const int i1m = tgpig.z;

    const int i12 = i1m%args.ne12;
    const int i13 = i1m/args.ne12;

    const uint64_t offset0 = i01*args.nb01 + (i12/args.r2)*args.nb02 + (i13/args.r3)*args.nb03;
    const uint64_t offset1 = i11*args.nb11 + (i12        )*args.nb12 + (i13        )*args.nb13;

    device const q_t * xq = (i01 < args.ne01) ? (device const q_t *) (src0 + offset0) + tx/chpb : (device const q_t *) src0;

    device const float4 * y4[r1ptg];

    for (int ir1 = 0; ir1 < r1ptg; ++ir1) {
        y4[ir1] = (i11 + ir1 < args.ne11) ? (device const float4 *) (src1 + offset1 + ir1*args.nb11) + tx : (device const float4 *) src1;
    }

    float sumf[r1ptg] = { [ 0 ... r1ptg - 1 ] = 0.0f };

    short cch = tx%chpb; // current chunk index

    for (int ich = tx; 4*ich < args.ne00; ich += chpt*nxpsg) {
        float4 lx[chpt];

#pragma unroll(chpt)
        for (short ch = 0; ch < chpt; ++ch) {
            deq_t4(xq, cch, lx[ch]);

            cch += nxpsg;
            if (cch >= chpb) {
                xq  += cch/chpb;
                cch %= chpb;
            }
        }

#pragma unroll(chpt)
        for (short ch = 0; ch < chpt; ++ch) {
#pragma unroll(r1ptg)
            for (short ir1 = 0; ir1 < r1ptg; ++ir1) {
                sumf[ir1] += dot(lx[ch], y4[ir1][ch*nxpsg]);
            }
        }

#pragma unroll(r1ptg)
        for (short ir1 = 0; ir1 < r1ptg; ++ir1) {
            y4[ir1] += chpt*nxpsg;
        }
    }

    // reduce only the threads in each row
    for (short ir1 = 0; ir1 < r1ptg; ++ir1) {
        if (nxpsg >= 32) {
            sumf[ir1] += simd_shuffle_down(sumf[ir1], 16);
        }
        if (nxpsg >= 16) {
            sumf[ir1] += simd_shuffle_down(sumf[ir1],  8);
        }
        if (nxpsg >= 8) {
            sumf[ir1] += simd_shuffle_down(sumf[ir1],  4);
        }
        if (nxpsg >= 4) {
            sumf[ir1] += simd_shuffle_down(sumf[ir1],  2);
        }
        if (nxpsg >= 2) {
            sumf[ir1] += simd_shuffle_down(sumf[ir1],  1);
        }
    }

    if (tx == 0) {
        for (short ir1 = 0; ir1 < r1ptg && i11 + ir1 < args.ne11; ++ir1) {
            device float * dst_f32 = (device float *) dst + (uint64_t)i1m*args.ne0*args.ne1 + (uint64_t)(i11 + ir1)*args.ne0;

            if (i01 < args.ne01) {
                dst_f32[i01] = sumf[ir1];
            }
        }
    }
}

// Small-batch prompt matvec for 2..5 tokens. It bridges decode-style matvec and
// full matmul when DS4 prefill chunks are too small to amortize matrix tiles.
template<short r1ptg, typename q_t, short epb, void (*deq_t4)(device const q_t *, short, thread float4 &)>
kernel void kernel_mul_mv_ext_q4_f32_disp(
        constant ds4_metal_args_mul_mv_ext & args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        uint3   tgpig[[threadgroup_position_in_grid]],
        ushort  tiisg[[thread_index_in_simdgroup]],
        ushort  sgitg[[simdgroup_index_in_threadgroup]]) {
    kernel_mul_mv_ext_q4_f32_impl<r1ptg, q_t, epb/4, deq_t4>(args, src0, src1, dst, tgpig, tiisg, sgitg);
}

typedef decltype(kernel_mul_mv_ext_q4_f32_disp<2, block_q8_0, 32, dequantize_q8_0_t4>) mul_mv_ext_q4_f32_t;

// Host-visible small-batch variants. DS4 currently needs F16 and Q8_0 weights
// for r1=2..5 during the prompt path.
template [[host_name("kernel_mul_mv_ext_f16_f32_r1_2")]]  kernel mul_mv_ext_q4_f32_t kernel_mul_mv_ext_q4_f32_disp<2, half4,      4,  dequantize_f16_t4>;
template [[host_name("kernel_mul_mv_ext_f16_f32_r1_3")]]  kernel mul_mv_ext_q4_f32_t kernel_mul_mv_ext_q4_f32_disp<3, half4,      4,  dequantize_f16_t4>;
template [[host_name("kernel_mul_mv_ext_f16_f32_r1_4")]]  kernel mul_mv_ext_q4_f32_t kernel_mul_mv_ext_q4_f32_disp<4, half4,      4,  dequantize_f16_t4>;
template [[host_name("kernel_mul_mv_ext_f16_f32_r1_5")]]  kernel mul_mv_ext_q4_f32_t kernel_mul_mv_ext_q4_f32_disp<5, half4,      4,  dequantize_f16_t4>;

template [[host_name("kernel_mul_mv_ext_q8_0_f32_r1_2")]] kernel mul_mv_ext_q4_f32_t kernel_mul_mv_ext_q4_f32_disp<2, block_q8_0, 32, dequantize_q8_0_t4>;
template [[host_name("kernel_mul_mv_ext_q8_0_f32_r1_3")]] kernel mul_mv_ext_q4_f32_t kernel_mul_mv_ext_q4_f32_disp<3, block_q8_0, 32, dequantize_q8_0_t4>;
template [[host_name("kernel_mul_mv_ext_q8_0_f32_r1_4")]] kernel mul_mv_ext_q4_f32_t kernel_mul_mv_ext_q4_f32_disp<4, block_q8_0, 32, dequantize_q8_0_t4>;
template [[host_name("kernel_mul_mv_ext_q8_0_f32_r1_5")]] kernel mul_mv_ext_q4_f32_t kernel_mul_mv_ext_q4_f32_disp<5, block_q8_0, 32, dequantize_q8_0_t4>;

constant bool FC_mul_mm_bc_inp [[function_constant(FC_MUL_MM + 0)]];
constant bool FC_mul_mm_bc_out [[function_constant(FC_MUL_MM + 1)]];

#ifdef DS4_METAL_HAS_TENSOR
template<
    short NR0, short NR1,
    typename SA, typename SA_4x4, typename block_q, short nl,
    void (*dequantize_func)(device const block_q *, short, thread SA_4x4 &),
    typename T0, typename T0_4x4, typename T1>
kernel void kernel_mul_mm_mpp(
        constant ds4_metal_args_mul_mm & args,
        device const char * srcA,
        device const char * srcB,
        device       char * dst,
        threadgroup  char * shmem [[threadgroup(0)]],
        uint3  tgpig [[threadgroup_position_in_grid]],
        ushort tiitg [[thread_index_in_threadgroup]],
        ushort sgitg [[simdgroup_index_in_threadgroup]]) {
    (void) sgitg;

    constexpr int NK  = 32;
    constexpr int NL  = NK/16;
    constexpr int NUM_THREADS = 128;

    const int K = args.ne00;
    const int M = args.ne0;
    const int N = args.ne1;
    const int im = tgpig.z;
    const int i12 = im%args.ne12;
    const int i13 = im/args.ne12;
    const int r0 = tgpig.y*NR0;
    const int r1 = tgpig.x*NR1;

    const uint64_t offset0 = (i12/args.r2)*args.nb02 + (i13/args.r3)*args.nb03;

    threadgroup SA *sa = (threadgroup SA *)shmem;
    threadgroup SA *sb = sa + NR0*NK;
    auto tA = tensor(sa, dextents<int32_t, 2>(NK, NR0));
    auto tB = tensor(sb, dextents<int32_t, 2>(NK, NR1));

    device const T1 *ptrB = (device const T1 *)(srcB + args.nb12*i12 + args.nb13*i13);
    const int strideB = args.nb11/sizeof(T1);

    matmul2d<
        matmul2d_descriptor(NR1, NR0, NK, false, true, false,
            matmul2d_descriptor::mode::multiply_accumulate),
        execution_simdgroups<4>> mm;

    auto cT = mm.template get_destination_cooperative_tensor<decltype(tB), decltype(tA), float>();

    #pragma unroll
    for (uint16_t i = 0; i < cT.get_capacity(); ++i) {
        if (cT.is_valid_element(i)) {
            cT[i] = 0.0f;
        }
    }

    for (int loop_k = 0; loop_k < K; loop_k += NK) {
        for (int work = tiitg; work < NR0*NL; work += NUM_THREADS) {
            const int row = work/NL;
            const int k_chunk = work%NL;
            const int k_pos = loop_k + k_chunk*16;
            const short k_base = k_chunk*16;

            if (!FC_mul_mm_bc_out || r0 + row < M) {
                if (is_same<T0_4x4, block_q>::value && FC_mul_mm_bc_inp) {
                    device const T0 *row_ptr = (device const T0 *)(srcA + args.nb01*(r0 + row) + offset0);
                    FOR_UNROLL (short i = 0; i < 16; i++) {
                        sa[row*NK + k_base + i] = (k_pos + i < K) ? (SA)row_ptr[k_pos + i] : (SA)0;
                    }
                } else {
                    const int block_idx = k_pos/(16*nl);
                    const short il = (k_pos/16)%nl;
                    device const block_q *row_ptr = (device const block_q *)(srcA + args.nb01*(r0 + row) + offset0);

                    SA_4x4 temp_a;
                    dequantize_func(row_ptr + block_idx, il, temp_a);
                    FOR_UNROLL (short i = 0; i < 16; i++) {
                        sa[row*NK + k_base + i] = (k_pos + i < K) ? temp_a[i/4][i%4] : (SA)0;
                    }
                }
            } else {
                FOR_UNROLL (short i = 0; i < 16; i++) {
                    sa[row*NK + k_base + i] = (SA)0;
                }
            }
        }
        for (int work = tiitg; work < NK*NR1; work += NUM_THREADS) {
            const int col = work/NK;
            const int k = work%NK;
            if ((!FC_mul_mm_bc_out && !FC_mul_mm_bc_inp) ||
                (r1 + col < N && loop_k + k < K)) {
                sb[col*NK + k] = (SA)ptrB[(uint64_t)(r1 + col)*strideB + loop_k + k];
            } else {
                sb[col*NK + k] = (SA)0;
            }
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        auto mA = tA.slice(0, 0);
        auto mB = tB.slice(0, 0);
        mm.run(mB, mA, cT);

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    device float *dst_batch = (device float *)dst + im*N*M;
    if (!FC_mul_mm_bc_out) {
        device float *dst_tile = dst_batch + r0 + (uint64_t)r1*M;
        auto tD = tensor(dst_tile, dextents<int32_t, 2>(NR0, NR1), array<int, 2>({1, M}));
        cT.store(tD);
    } else {
        auto tD = tensor(dst_batch, dextents<int32_t, 2>(M, N), array<int, 2>({1, M}));
        auto mD = tD.slice(r0, r1);
        cT.store(mD);
    }
}

typedef decltype(kernel_mul_mm_mpp<64, 32, half, half4x4, float4x4, 1, dequantize_f32, float, float4x4, float>) mul_mm_mpp_t;

template [[host_name("kernel_mul_mm_f16_f32_mpp")]]  kernel mul_mm_mpp_t kernel_mul_mm_mpp<64, 32, half, half4x4, half4x4, 1, dequantize_f16,  half,  half4x4,  float>;

// Retained Metal4/TensorOps dense prefill kernel.  The legacy MPP prototype
// staged both operands in threadgroup memory; this version stages only the
// model weight tile and lets MPP read the dense RHS activation matrix directly
// from device memory.  That direct-RHS shape was the clear win for DS4's large
// aligned F16/Q8_0 prompt matmuls.  The host selects the widest token tile that
// evenly divides the batch, with 128-token tiles retained after the 64-token
// retest was neutral or slower.
template<
    short NR1,
    typename SA, typename SA_4x4, typename block_q, short nl,
    void (*dequantize_func)(device const block_q *, short, thread SA_4x4 &),
    typename T0, typename T0_4x4, typename T1>
kernel void kernel_mul_mm_mpp_direct_rhs(
        constant ds4_metal_args_mul_mm & args,
        device const char * srcA,
        device const char * srcB,
        device       char * dst,
        threadgroup  char * shmem [[threadgroup(0)]],
        uint3  tgpig [[threadgroup_position_in_grid]],
        ushort tiitg [[thread_index_in_threadgroup]],
        ushort sgitg [[simdgroup_index_in_threadgroup]]) {
    (void) sgitg;

    constexpr int NR0 = 64;
    constexpr int NK  = 32;
    constexpr int NL  = NK/16;
    constexpr int NUM_THREADS = 128;

    const int K = args.ne00;
    const int M = args.ne0;
    const int N = args.ne1;
    const int im = tgpig.z;
    const int i12 = im%args.ne12;
    const int i13 = im/args.ne12;
    const int r0 = tgpig.y*NR0;
    const int r1 = tgpig.x*NR1;

    const uint64_t offset0 = (i12/args.r2)*args.nb02 + (i13/args.r3)*args.nb03;

    threadgroup SA *sa = (threadgroup SA *)shmem;
    auto tA = tensor(sa, dextents<int32_t, 2>(NK, NR0));

    device T1 *ptrB = (device T1 *)(srcB + args.nb12*i12 + args.nb13*i13);
    const int strideB = args.nb11/sizeof(T1);
    auto tB = tensor(ptrB, dextents<int32_t, 2>(K, N), array<int, 2>({1, strideB}));

    matmul2d<
        matmul2d_descriptor(NR1, NR0, NK, false, true, true,
            matmul2d_descriptor::mode::multiply_accumulate),
        execution_simdgroups<4>> mm;

    auto cT = mm.template get_destination_cooperative_tensor<decltype(tB), decltype(tA), float>();

    #pragma unroll
    for (uint16_t i = 0; i < cT.get_capacity(); ++i) {
        if (cT.is_valid_element(i)) {
            cT[i] = 0.0f;
        }
    }

    for (int loop_k = 0; loop_k < K; loop_k += NK) {
        for (int work = tiitg; work < NR0*NL; work += NUM_THREADS) {
            const int row = work/NL;
            const int k_chunk = work%NL;
            const int k_pos = loop_k + k_chunk*16;
            const short k_base = k_chunk*16;

            if (r0 + row < M) {
                if (is_same<T0_4x4, block_q>::value && FC_mul_mm_bc_inp) {
                    device const T0 *row_ptr = (device const T0 *)(srcA + args.nb01*(r0 + row) + offset0);
                    FOR_UNROLL (short i = 0; i < 16; i++) {
                        sa[row*NK + k_base + i] = (k_pos + i < K) ? (SA)row_ptr[k_pos + i] : (SA)0;
                    }
                } else {
                    const int block_idx = k_pos/(16*nl);
                    const short il = (k_pos/16)%nl;
                    device const block_q *row_ptr = (device const block_q *)(srcA + args.nb01*(r0 + row) + offset0);

                    SA_4x4 temp_a;
                    dequantize_func(row_ptr + block_idx, il, temp_a);
                    FOR_UNROLL (short i = 0; i < 16; i++) {
                        sa[row*NK + k_base + i] = (k_pos + i < K) ? temp_a[i/4][i%4] : (SA)0;
                    }
                }
            } else {
                FOR_UNROLL (short i = 0; i < 16; i++) {
                    sa[row*NK + k_base + i] = (SA)0;
                }
            }
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        auto mA = tA.slice(0, 0);
        auto mB = tB.slice(loop_k, r1);
        mm.run(mB, mA, cT);

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    device float *dst_batch = (device float *)dst + im*N*M;
    auto tD = tensor(dst_batch, dextents<int32_t, 2>(M, N), array<int, 2>({1, M}));
    auto mD = tD.slice(r0, r1);
    cT.store(mD);
}

typedef decltype(kernel_mul_mm_mpp_direct_rhs<32, half, half4x4, float4x4, 1, dequantize_f32, float, float4x4, float>) mul_mm_mpp_direct_rhs_t;

template [[host_name("kernel_mul_mm_f16_f32_mpp_direct_rhs")]]  kernel mul_mm_mpp_direct_rhs_t kernel_mul_mm_mpp_direct_rhs<32, half, half4x4, half4x4, 1, dequantize_f16,  half,  half4x4,  float>;
template [[host_name("kernel_mul_mm_f16_f32_mpp_direct_rhs_n64")]]  kernel mul_mm_mpp_direct_rhs_t kernel_mul_mm_mpp_direct_rhs<64, half, half4x4, half4x4, 1, dequantize_f16,  half,  half4x4,  float>;
template [[host_name("kernel_mul_mm_f16_f32_mpp_direct_rhs_n128")]]  kernel mul_mm_mpp_direct_rhs_t kernel_mul_mm_mpp_direct_rhs<128, half, half4x4, half4x4, 1, dequantize_f16,  half,  half4x4,  float>;
template [[host_name("kernel_mul_mm_q8_0_f32_nax_direct_rhs")]] kernel mul_mm_mpp_direct_rhs_t kernel_mul_mm_mpp_direct_rhs<32, half, half4x4, block_q8_0, 2, dequantize_q8_0, float, float4x4, float>;
template [[host_name("kernel_mul_mm_q8_0_f32_nax_direct_rhs_n64")]] kernel mul_mm_mpp_direct_rhs_t kernel_mul_mm_mpp_direct_rhs<64, half, half4x4, block_q8_0, 2, dequantize_q8_0, float, float4x4, float>;
template [[host_name("kernel_mul_mm_q8_0_f32_nax_direct_rhs_n128")]] kernel mul_mm_mpp_direct_rhs_t kernel_mul_mm_mpp_direct_rhs<128, half, half4x4, block_q8_0, 2, dequantize_q8_0, float, float4x4, float>;
#endif

// Tiled matrix-matrix kernel used for prompt batches larger than 8. DS4 uses
// this to turn prefill into large simdgroup matrix operations; each block_q
// contains 16*nl weights.
template<typename S0, typename S0_4x4, typename S0_8x8, typename S1, typename S1_2x4, typename S1_8x8, typename block_q, short nl, void (*dequantize_func)(device const block_q *, short, thread S0_4x4 &), typename T0, typename T0_4x4, typename T1, typename T1_2x4>
kernel void kernel_mul_mm(
        constant ds4_metal_args_mul_mm & args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        threadgroup  char * shmem [[threadgroup(0)]],
        uint3  tgpig[[threadgroup_position_in_grid]],
        ushort tiitg[[thread_index_in_threadgroup]],
        ushort sgitg[[simdgroup_index_in_threadgroup]]) {

    threadgroup S0 * sa = (threadgroup S0 *)(shmem);
    threadgroup S1 * sb = (threadgroup S1 *)(shmem + 4096);

    constexpr int NR0 = 64;
    constexpr int NR1 = 32;

    constexpr int NK  = 32;
    constexpr int NL0 = NK/16;
    constexpr int NL1 = NK/8;

    const int im = tgpig.z;
    const int r0 = tgpig.y*NR0;
    const int r1 = tgpig.x*NR1;

    // if this block is of 64x32 shape or smaller
    const short nr0 = (args.ne0 - r0 < NR0) ? (args.ne0 - r0) : NR0;
    const short nr1 = (args.ne1 - r1 < NR1) ? (args.ne1 - r1) : NR1;

    // a thread shouldn't load data outside of the matrix
    const short lr0 = ((short)tiitg/NL0) < nr0 ? ((short)tiitg/NL0) : nr0 - 1; // 0 .. 63
    const short lr1 = ((short)tiitg/NL1) < nr1 ? ((short)tiitg/NL1) : nr1 - 1; // 0 .. 31

    const short il0 = (tiitg % NL0);

    short il = il0;

    const int i12 = im%args.ne12;
    const int i13 = im/args.ne12;

    const uint64_t offset0 = (i12/args.r2)*args.nb02 + (i13/args.r3)*args.nb03;
    const short    offset1 = il0/nl;

    device const block_q * x = (device const block_q *)(src0 + args.nb01*(r0 + lr0) + offset0) + offset1;

    const short iy = 8*(tiitg % NL1);

    device const T1 * y = (device const T1 *)(src1
        + args.nb13*i13
        + args.nb12*i12
        + args.nb11*(r1 + lr1)
        + args.nb10*iy);

    S0_8x8 ma[4];
    S1_8x8 mb[2];

    simdgroup_float8x8 mc[8];

    for (short i = 0; i < 8; i++){
        mc[i] = make_filled_simdgroup_matrix<float, 8>(0.f);
    }

    for (int loop_k = 0; loop_k < args.ne00; loop_k += NK) {
        // load data and store to threadgroup memory
        if (is_same<T0_4x4, block_q>::value && FC_mul_mm_bc_inp) {
            threadgroup_barrier(mem_flags::mem_threadgroup);

            // no need for dequantization
            for (short i = 0; i < 16; i++) {
                const short sx = 2*il0 + i/8;
                const short sy = (tiitg/NL0)/8;

                const short lx = (tiitg/NL0)%8;
                const short ly = i%8;

                const short ib = 8*sx + sy;

                *(sa + 64*ib + 8*ly + lx) = loop_k + 16*il + i < args.ne00 ? *((device T0 *) x + i) : 0;
            }
        } else {
            S0_4x4 temp_a;
            dequantize_func(x, il, temp_a);

            threadgroup_barrier(mem_flags::mem_threadgroup);

            FOR_UNROLL (short i = 0; i < 16; i++) {
                const short sx = 2*il0 + i/8;
                const short sy = (tiitg/NL0)/8;

                const short lx = (tiitg/NL0)%8;
                const short ly = i%8;

                const short ib = 8*sx + sy;

                // Pointer-form store avoids a slower address-lowering path in
                // current Apple Metal compilers for this dequantized tile write.
                *(sa + 64*ib + 8*ly + lx) = temp_a[i/4][i%4];
            }
        }

        if (FC_mul_mm_bc_inp) {
            for (short i = 0; i < 8; ++i) {
                const short sx = (tiitg%NL1);
                const short sy = (tiitg/NL1)/8;

                const short lx = i;
                const short ly = (tiitg/NL1)%8;

                const short ib = 4*sx + sy;

                *(sb + 64*ib + 8*ly + lx) = loop_k + iy + i < args.ne00 ? (S1) *((device T1 *) y + i) : 0;
            }
        } else {
            const short sx = (tiitg%NL1);
            const short sy = (tiitg/NL1)/8;

            const short ly = (tiitg/NL1)%8;

            const short ib = 4*sx + sy;

            *(threadgroup S1_2x4 *)(sb + 64*ib + 8*ly) = (S1_2x4)(*((device T1_2x4 *) y));
        }

        il = (il + 2 < nl) ? il + 2 : il % 2;
        x  = (il < 2) ? x + (2 + nl - 1)/nl : x;

        y += NK;

        threadgroup_barrier(mem_flags::mem_threadgroup);

        // load matrices from threadgroup memory and conduct outer products
        threadgroup const S0 * lsma = (sa + 4*64*(sgitg%2));
        threadgroup const S1 * lsmb = (sb + 2*64*(sgitg/2));

        FOR_UNROLL (short ik = 0; ik < NK/8; ik++) {
            simdgroup_barrier(mem_flags::mem_none);

            FOR_UNROLL (short i = 0; i < 4; i++) {
                simdgroup_load(ma[i], lsma + 64*i, 8, 0, false);
            }

            simdgroup_barrier(mem_flags::mem_none);

            FOR_UNROLL (short i = 0; i < 2; i++) {
                simdgroup_load(mb[i], lsmb + 64*i, 8, 0, false);
            }

            simdgroup_barrier(mem_flags::mem_none);

            FOR_UNROLL (short i = 0; i < 8; i++){
                simdgroup_multiply_accumulate(mc[i], mb[i/4], ma[i%4], mc[i]);
            }

            lsma += 8*64;
            lsmb += 4*64;
        }
    }

    if (!FC_mul_mm_bc_out || (r0 + NR0 <= args.ne0 && r1 + NR1 <= args.ne1)) {
        // if no bounds checks on the output are needed, we can directly write to device memory
        device float * C = (device float *) dst +
            (r0 + 32*(sgitg &  1)) + \
            (r1 + 16*(sgitg >> 1)) * args.ne0 + im*args.ne1*args.ne0;

        for (short i = 0; i < 8; i++) {
            simdgroup_store(mc[i], C + 8*(i%4) + 8*args.ne0*(i/4), args.ne0, 0, false);
        }
    } else {
        // block is smaller than 64x32, we should avoid writing data outside of the matrix
        threadgroup_barrier(mem_flags::mem_threadgroup);

        threadgroup float * temp_str = ((threadgroup float *) shmem) + 32*(sgitg&1) + (16*(sgitg >> 1))*NR0;

        for (short i = 0; i < 8; i++) {
            simdgroup_store(mc[i], temp_str + 8*(i%4) + 8*NR0*(i/4), NR0, 0, false);
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (sgitg == 0) {
            for (int j = tiitg; j < nr1; j += NR1) {
                device float  * D  = (device float  *) dst + r0 + (r1 + j)*args.ne0 + im*args.ne1*args.ne0;
                device float4 * D4 = (device float4 *) D;

                threadgroup float  * C  = temp_str + (j*NR0);
                threadgroup float4 * C4 = (threadgroup float4 *) C;

                int i = 0;
                for (; i < nr0/4; i++) {
                    *(D4 + i) = *(C4 + i);
                }

                i *= 4;
                for (; i < nr0; i++) {
                    *(D + i) = *(C + i);
                }
            }
        }
    }
}

kernel void kernel_mul_mm_f16_f32_pair(
        constant ds4_metal_args_mul_mm & args,
        device const char * src0_a,
        device const char * src0_b,
        device const char * src1,
        device       char * dst_a,
        device       char * dst_b,
        threadgroup  char * shmem [[threadgroup(0)]],
        uint3  tgpig[[threadgroup_position_in_grid]],
        ushort tiitg[[thread_index_in_threadgroup]],
        ushort sgitg[[simdgroup_index_in_threadgroup]]) {
    threadgroup half * sa_a = (threadgroup half *)(shmem);
    threadgroup half * sa_b = (threadgroup half *)(shmem + 4096);
    threadgroup half * sb   = (threadgroup half *)(shmem + 8192);

    constexpr int NR0 = 64;
    constexpr int NR1 = 32;
    constexpr int NK  = 32;
    constexpr int NL0 = NK/16;
    constexpr int NL1 = NK/8;

    const int im = tgpig.z;
    const int r0 = tgpig.y*NR0;
    const int r1 = tgpig.x*NR1;

    const short nr0 = (args.ne0 - r0 < NR0) ? (args.ne0 - r0) : NR0;
    const short nr1 = (args.ne1 - r1 < NR1) ? (args.ne1 - r1) : NR1;

    const short lr0 = ((short)tiitg/NL0) < nr0 ? ((short)tiitg/NL0) : nr0 - 1;
    const short lr1 = ((short)tiitg/NL1) < nr1 ? ((short)tiitg/NL1) : nr1 - 1;

    const short il0 = (tiitg % NL0);
    short il = il0;

    const int i12 = im%args.ne12;
    const int i13 = im/args.ne12;

    const uint64_t offset0 = (i12/args.r2)*args.nb02 + (i13/args.r3)*args.nb03;
    const short    offset1 = il0;

    device const half4x4 * xa = (device const half4x4 *)(src0_a + args.nb01*(r0 + lr0) + offset0) + offset1;
    device const half4x4 * xb = (device const half4x4 *)(src0_b + args.nb01*(r0 + lr0) + offset0) + offset1;

    const short iy = 8*(tiitg % NL1);

    device const float * y = (device const float *)(src1
        + args.nb13*i13
        + args.nb12*i12
        + args.nb11*(r1 + lr1)
        + args.nb10*iy);

    simdgroup_half8x8 ma[4];
    simdgroup_half8x8 mb[2];

    simdgroup_float8x8 mc_a[8];
    simdgroup_float8x8 mc_b[8];

    for (short i = 0; i < 8; i++) {
        mc_a[i] = make_filled_simdgroup_matrix<float, 8>(0.f);
        mc_b[i] = make_filled_simdgroup_matrix<float, 8>(0.f);
    }

    for (int loop_k = 0; loop_k < args.ne00; loop_k += NK) {
        half4x4 temp_a;
        half4x4 temp_b;
        dequantize_f16(xa, il, temp_a);
        dequantize_f16(xb, il, temp_b);

        threadgroup_barrier(mem_flags::mem_threadgroup);

        FOR_UNROLL (short i = 0; i < 16; i++) {
            const short sx = 2*il0 + i/8;
            const short sy = (tiitg/NL0)/8;

            const short lx = (tiitg/NL0)%8;
            const short ly = i%8;

            const short ib = 8*sx + sy;

            *(sa_a + 64*ib + 8*ly + lx) = temp_a[i/4][i%4];
            *(sa_b + 64*ib + 8*ly + lx) = temp_b[i/4][i%4];
        }

        if (FC_mul_mm_bc_inp) {
            for (short i = 0; i < 8; ++i) {
                const short sx = (tiitg%NL1);
                const short sy = (tiitg/NL1)/8;

                const short lx = i;
                const short ly = (tiitg/NL1)%8;

                const short ib = 4*sx + sy;

                *(sb + 64*ib + 8*ly + lx) = loop_k + iy + i < args.ne00 ? (half) *((device float *) y + i) : 0;
            }
        } else {
            const short sx = (tiitg%NL1);
            const short sy = (tiitg/NL1)/8;

            const short ly = (tiitg/NL1)%8;

            const short ib = 4*sx + sy;

            *(threadgroup half2x4 *)(sb + 64*ib + 8*ly) = (half2x4)(*((device float2x4 *) y));
        }

        il = (il + 2 < 1) ? il + 2 : il % 2;
        xa = (il < 2) ? xa + 2 : xa;
        xb = (il < 2) ? xb + 2 : xb;

        y += NK;

        threadgroup_barrier(mem_flags::mem_threadgroup);

        threadgroup const half * lsma_a = (sa_a + 4*64*(sgitg%2));
        threadgroup const half * lsma_b = (sa_b + 4*64*(sgitg%2));
        threadgroup const half * lsmb   = (sb   + 2*64*(sgitg/2));

        FOR_UNROLL (short ik = 0; ik < NK/8; ik++) {
            simdgroup_barrier(mem_flags::mem_none);

            FOR_UNROLL (short i = 0; i < 2; i++) {
                simdgroup_load(mb[i], lsmb + 64*i, 8, 0, false);
            }

            simdgroup_barrier(mem_flags::mem_none);

            FOR_UNROLL (short i = 0; i < 4; i++) {
                simdgroup_load(ma[i], lsma_a + 64*i, 8, 0, false);
            }

            simdgroup_barrier(mem_flags::mem_none);

            FOR_UNROLL (short i = 0; i < 8; i++) {
                simdgroup_multiply_accumulate(mc_a[i], mb[i/4], ma[i%4], mc_a[i]);
            }

            simdgroup_barrier(mem_flags::mem_none);

            FOR_UNROLL (short i = 0; i < 4; i++) {
                simdgroup_load(ma[i], lsma_b + 64*i, 8, 0, false);
            }

            simdgroup_barrier(mem_flags::mem_none);

            FOR_UNROLL (short i = 0; i < 8; i++) {
                simdgroup_multiply_accumulate(mc_b[i], mb[i/4], ma[i%4], mc_b[i]);
            }

            lsma_a += 8*64;
            lsma_b += 8*64;
            lsmb   += 4*64;
        }
    }

    if (!FC_mul_mm_bc_out || (r0 + NR0 <= args.ne0 && r1 + NR1 <= args.ne1)) {
        device float * C_a = (device float *) dst_a +
            (r0 + 32*(sgitg &  1)) +
            (r1 + 16*(sgitg >> 1)) * args.ne0 + im*args.ne1*args.ne0;
        device float * C_b = (device float *) dst_b +
            (r0 + 32*(sgitg &  1)) +
            (r1 + 16*(sgitg >> 1)) * args.ne0 + im*args.ne1*args.ne0;

        for (short i = 0; i < 8; i++) {
            simdgroup_store(mc_a[i], C_a + 8*(i%4) + 8*args.ne0*(i/4), args.ne0, 0, false);
            simdgroup_store(mc_b[i], C_b + 8*(i%4) + 8*args.ne0*(i/4), args.ne0, 0, false);
        }
    } else {
        threadgroup_barrier(mem_flags::mem_threadgroup);

        threadgroup float * temp_str = (threadgroup float *) shmem;

        for (short i = 0; i < 8; i++) {
            simdgroup_store(mc_a[i],
                            temp_str + 32*(sgitg&1) + (16*(sgitg >> 1))*NR0 + 8*(i%4) + 8*NR0*(i/4),
                            NR0,
                            0,
                            false);
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (sgitg == 0) {
            for (int j = tiitg; j < nr1; j += NR1) {
                device float  * D  = (device float *) dst_a + r0 + (r1 + j)*args.ne0 + im*args.ne1*args.ne0;
                device float4 * D4 = (device float4 *) D;

                threadgroup float  * C  = temp_str + (j*NR0);
                threadgroup float4 * C4 = (threadgroup float4 *) C;

                int i = 0;
                for (; i < nr0/4; i++) {
                    *(D4 + i) = *(C4 + i);
                }

                i *= 4;
                for (; i < nr0; i++) {
                    *(D + i) = *(C + i);
                }
            }
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (short i = 0; i < 8; i++) {
            simdgroup_store(mc_b[i],
                            temp_str + 32*(sgitg&1) + (16*(sgitg >> 1))*NR0 + 8*(i%4) + 8*NR0*(i/4),
                            NR0,
                            0,
                            false);
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (sgitg == 0) {
            for (int j = tiitg; j < nr1; j += NR1) {
                device float  * D  = (device float *) dst_b + r0 + (r1 + j)*args.ne0 + im*args.ne1*args.ne0;
                device float4 * D4 = (device float4 *) D;

                threadgroup float  * C  = temp_str + (j*NR0);
                threadgroup float4 * C4 = (threadgroup float4 *) C;

                int i = 0;
                for (; i < nr0/4; i++) {
                    *(D4 + i) = *(C4 + i);
                }

                i *= 4;
                for (; i < nr0; i++) {
                    *(D + i) = *(C + i);
                }
            }
        }
    }
}

typedef decltype(kernel_mul_mm<half, half4x4, simdgroup_half8x8, half, half2x4, simdgroup_half8x8, float4x4, 1, dequantize_f32, float, float4x4, float, float2x4>) mul_mm_t;

// Host-visible prefill matmul variants for F16 and Q8_0 weights.
template [[host_name("kernel_mul_mm_f16_f32")]]  kernel mul_mm_t kernel_mul_mm<half, half4x4, simdgroup_half8x8, half, half2x4, simdgroup_half8x8, half4x4, 1, dequantize_f16,  half,  half4x4,  float, float2x4>;
template [[host_name("kernel_mul_mm_q8_0_f32")]] kernel mul_mm_t kernel_mul_mm<half, half4x4, simdgroup_half8x8, half, half2x4, simdgroup_half8x8, block_q8_0, 2, dequantize_q8_0, float, float4x4, float, float2x4>;
