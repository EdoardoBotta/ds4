#include "ds4.h"

#include <ctype.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static void trim_ascii(char *text) {
    size_t start = 0;
    size_t len = strlen(text);
    while (start < len && isspace((unsigned char)text[start])) start++;
    while (len > start && isspace((unsigned char)text[len - 1u])) len--;
    if (start != 0) memmove(text, text + start, len - start);
    text[len - start] = '\0';
}

static int run_case(ds4_engine *engine,
                    ds4_session *session,
                    const char *prompt_text,
                    const char *expected) {
    ds4_tokens prompt = {0};
    ds4_encode_chat_prompt(engine, NULL, prompt_text, DS4_THINK_NONE, &prompt);

    char err[256] = "";
    if (ds4_session_sync(session, &prompt, err, sizeof(err)) != 0) {
        fprintf(stderr, "qwen_session_smoke: sync failed: %s\n", err);
        ds4_tokens_free(&prompt);
        return 1;
    }

    char output[4096] = "";
    size_t output_len = 0;
    for (int i = 0; i < 16; i++) {
        const int token = ds4_session_argmax(session);
        if (token < 0 || ds4_token_is_stop(engine, token)) break;
        size_t piece_len = 0;
        char *piece = ds4_token_text(engine, token, &piece_len);
        if (piece && piece_len < sizeof(output) - output_len) {
            memcpy(output + output_len, piece, piece_len);
            output_len += piece_len;
            output[output_len] = '\0';
        }
        free(piece);
        if (ds4_session_eval(session, token, err, sizeof(err)) != 0) {
            fprintf(stderr, "qwen_session_smoke: decode failed: %s\n", err);
            ds4_tokens_free(&prompt);
            return 1;
        }
    }

    trim_ascii(output);
    printf("%s\n", output);
    const int ok = strcmp(output, expected) == 0;
    if (!ok) {
        fprintf(stderr, "qwen_session_smoke: expected '%s', got '%s'\n",
                expected, output);
    }
    ds4_tokens_free(&prompt);
    return ok ? 0 : 1;
}

int main(int argc, char **argv) {
    if (argc < 4 || (argc & 1) != 0) {
        fprintf(stderr,
                "usage: %s MODEL PROMPT EXPECTED [PROMPT EXPECTED ...]\n",
                argv[0]);
        return 2;
    }

    ds4_engine_options opt;
    memset(&opt, 0, sizeof(opt));
    opt.model_path = argv[1];
    opt.backend = DS4_BACKEND_CPU;
    opt.context_size = 64;
    opt.placement_ctx_hint = 64;
    opt.placement_session_count_hint = 1;
    opt.power_percent = 100;

    ds4_engine *engine = NULL;
    if (ds4_engine_open(&engine, &opt) != 0 || !engine) {
        fprintf(stderr, "qwen_session_smoke: engine open failed\n");
        return 1;
    }
    ds4_session *session = NULL;
    if (ds4_session_create(&session, engine, 64) != 0 || !session) {
        fprintf(stderr, "qwen_session_smoke: session create failed\n");
        ds4_engine_close(engine);
        return 1;
    }

    int rc = 0;
    for (int i = 2; i + 1 < argc; i += 2) {
        if (run_case(engine, session, argv[i], argv[i + 1]) != 0) {
            rc = 1;
            break;
        }
    }
    ds4_session_free(session);
    ds4_engine_close(engine);
    return rc;
}
