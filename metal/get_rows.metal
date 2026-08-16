struct ds4_metal_args_get_rows_q8_0 {
    int32_t  n_embd;
    int32_t  n_vocab;
    int32_t  n_tokens;
    uint64_t src_row_bytes;
    uint64_t dst_row_bytes;
    uint64_t token_stride;
};
kernel void kernel_get_rows_q8_0_f32(
        constant ds4_metal_args_get_rows_q8_0 & args,
        device const char    * src0,
        device const char    * src1,
        device       char    * dst,
        uint3                  tgpig[[threadgroup_position_in_grid]],
        ushort                 tiitg[[thread_index_in_threadgroup]],
        ushort3                ntg [[threads_per_threadgroup]]) {
    const int32_t block = (int32_t)tgpig.x;
    const int32_t tok_i = (int32_t)tgpig.y;
    if (tok_i >= args.n_tokens) return;

    const int32_t token =
        ((const device int32_t *)(src1 + (uint64_t)tok_i*args.token_stride))[0];
    if (token < 0 || token >= args.n_vocab) return;

    const device block_q8_0 *row =
        (const device block_q8_0 *)(src0 + (uint64_t)token * args.src_row_bytes);
    const device block_q8_0 *qb = row + block;
    device float *out =
        (device float *)(dst + (uint64_t)tok_i * args.dst_row_bytes);

    const int32_t i0 = block * QK8_0;
    const float d = (float)qb->d;
    for (int32_t i = (int32_t)tiitg; i < QK8_0; i += (int32_t)ntg.x) {
        const int32_t idx = i0 + i;
        if (idx < args.n_embd) {
            out[idx] = d * (float)qb->qs[i];
        }
    }
}
