struct ds4_metal_args_add3 {
    uint32_t n;
};
kernel void kernel_add2_f32(
        constant ds4_metal_args_add3 &args,
        device const float *a,
        device const float *b,
        device float *out,
        uint i [[thread_position_in_grid]]) {
    if (i >= args.n) return;
    out[i] = a[i] + b[i];
}
