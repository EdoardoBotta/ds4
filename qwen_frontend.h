#ifndef QWEN_FRONTEND_H
#define QWEN_FRONTEND_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

typedef struct {
    const char *role;
    const char *content;
} qwen_chat_message;

typedef void (*qwen_text_emit_fn)(void *ud, const char *text, size_t len);

typedef struct {
    int prompt_tokens;
    int completion_tokens;
    bool stopped;
} qwen_http_stats;

typedef int (*qwen_http_generate_fn)(
    void *ud, const qwen_chat_message *messages, size_t n_messages,
    int max_tokens, bool think, float temperature, uint64_t seed,
    qwen_text_emit_fn emit, void *emit_ud, qwen_http_stats *stats);

int qwen_http_serve(const char *host, int port, int default_tokens,
                    float default_temperature, uint64_t default_seed,
                    bool default_think, bool cors,
                    qwen_http_generate_fn generate, void *generate_ud);

#endif
