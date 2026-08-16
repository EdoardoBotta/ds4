#ifndef DS4_GPU_H
#define DS4_GPU_H

#include <stdint.h>

typedef struct ds4_gpu_tensor ds4_gpu_tensor;

int ds4_gpu_init(void);

void ds4_gpu_cleanup(void);

ds4_gpu_tensor *ds4_gpu_tensor_alloc(uint64_t bytes);

ds4_gpu_tensor *ds4_gpu_tensor_view(const ds4_gpu_tensor *base, uint64_t offset, uint64_t bytes);

void ds4_gpu_tensor_free(ds4_gpu_tensor *tensor);

uint64_t ds4_gpu_tensor_bytes(const ds4_gpu_tensor *tensor);

void *ds4_gpu_tensor_contents(ds4_gpu_tensor *tensor);

int ds4_gpu_tensor_fill_f32(ds4_gpu_tensor *tensor, float value, uint64_t count);

int ds4_gpu_tensor_write(ds4_gpu_tensor *tensor, uint64_t offset, const void *data, uint64_t bytes);

int ds4_gpu_tensor_read(const ds4_gpu_tensor *tensor, uint64_t offset, void *data, uint64_t bytes);

int ds4_gpu_tensor_copy(ds4_gpu_tensor *dst, uint64_t dst_offset,
                          const ds4_gpu_tensor *src, uint64_t src_offset,
                          uint64_t bytes);

int ds4_gpu_begin_commands(void);

int ds4_gpu_commands_active(void);

int ds4_gpu_end_commands(void);

int ds4_gpu_synchronize(void);

int ds4_gpu_set_model_fd(int fd);

int ds4_gpu_set_model_fd_for_map(int fd, const void *model_map);

int ds4_gpu_set_model_map_range(const void *model_map, uint64_t model_size, uint64_t map_offset, uint64_t map_size, uint64_t max_tensor_bytes);

int ds4_gpu_device_is_pre_m5_apple_silicon(void);


uint32_t ds4_gpu_stream_expert_cache_configured_count(void);

void ds4_gpu_print_memory_report(const char *label);

int ds4_gpu_embed_token_q8_0_tensor(
        ds4_gpu_tensor *out,
        const void       *model_map,
        uint64_t          model_size,
        uint64_t          weight_offset,
        uint32_t          n_vocab,
        uint32_t          token,
        uint32_t          n_embd);

int ds4_gpu_embed_tokens_q8_0_tensor(
        ds4_gpu_tensor       *out,
        const ds4_gpu_tensor *tokens,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                weight_offset,
        uint32_t                n_vocab,
        uint32_t                n_tokens,
        uint32_t                n_embd);

int ds4_gpu_indexer_topk_tensor(
        ds4_gpu_tensor       *selected,
        const ds4_gpu_tensor *scores,
        uint32_t                n_comp,
        uint32_t                n_tokens,
        uint32_t                top_k);

int ds4_gpu_matmul_q8_0_tensor(
        ds4_gpu_tensor       *out,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                weight_offset,
        uint64_t                in_dim,
        uint64_t                out_dim,
        const ds4_gpu_tensor *x,
        uint64_t                n_tok);

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
        uint64_t                n_tok);

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
        float                   clamp);

int ds4_gpu_rms_norm_weight_tensor(
        ds4_gpu_tensor       *out,
        const ds4_gpu_tensor *x,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                weight_offset,
        uint32_t                n,
        float                   eps);

int ds4_gpu_rms_norm_weight_rows_tensor(
        ds4_gpu_tensor       *out,
        const ds4_gpu_tensor *x,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                weight_offset,
        uint32_t                n,
        uint32_t                rows,
        float                   eps);

int ds4_gpu_add_rms_norm_weight_tensor(
        ds4_gpu_tensor       *norm_out,
        ds4_gpu_tensor       *sum_out,
        const ds4_gpu_tensor *a,
        const ds4_gpu_tensor *b,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                weight_offset,
        uint32_t                n,
        float                   eps);

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
        float                   eps);

int ds4_gpu_swiglu_tensor(
        ds4_gpu_tensor       *out,
        const ds4_gpu_tensor *gate,
        const ds4_gpu_tensor *up,
        uint32_t                n,
        float                   clamp,
        float                   weight);

int ds4_gpu_swiglu_f16_tensor(
        ds4_gpu_tensor       *out,
        const ds4_gpu_tensor *gate,
        const ds4_gpu_tensor *up,
        uint32_t              n,
        float                 clamp,
        float                 weight);

int ds4_gpu_matmul_q8_0_f16_rhs_tensor(
        ds4_gpu_tensor       *out,
        const void           *model_map,
        uint64_t              model_size,
        uint64_t              weight_offset,
        uint64_t              in_dim,
        uint64_t              out_dim,
        const ds4_gpu_tensor *x_f16,
        uint64_t              n_tok);

int ds4_gpu_add_tensor(
        ds4_gpu_tensor       *out,
        const ds4_gpu_tensor *a,
        const ds4_gpu_tensor *b,
        uint32_t                n);

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
        float                 freq_base);

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
        float                 freq_base);

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
        uint32_t              head_dim);

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
        uint32_t              head_dim);

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
        uint32_t              head_dim);

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
        float                 eps);

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
        float                 eps);

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
        float                 eps);

int ds4_gpu_qwen35_gdn_batch_preserve_four_tensor(
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
        uint32_t              channels,
        uint32_t              qk_heads,
        uint32_t              v_heads,
        uint32_t              state_dim,
        uint32_t              conv_width,
        float                 eps);

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
        float                 eps);

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
        float                 eps);

#endif
