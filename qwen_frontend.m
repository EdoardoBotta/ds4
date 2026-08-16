#import <Foundation/Foundation.h>

#include <arpa/inet.h>
#include <errno.h>
#include <math.h>
#include <netdb.h>
#include <signal.h>
#include <sys/socket.h>
#include <unistd.h>

#include "qwen_frontend.h"

static uint64_t request_seed_nonce;

static bool send_all(int fd, const void *data, size_t len) {
    const uint8_t *p = data;
    while (len) {
        ssize_t n = send(fd, p, len, MSG_NOSIGNAL);
        if (n <= 0) return false;
        p += n;
        len -= (size_t)n;
    }
    return true;
}

static NSData *json_data(id object) {
    return [NSJSONSerialization dataWithJSONObject:object options:0 error:nil];
}

static void http_reply(int fd, int status, NSString *type, NSData *body,
                       bool cors) {
    NSString *reason = status == 200 ? @"OK" :
        (status == 400 ? @"Bad Request" :
         (status == 404 ? @"Not Found" : @"Internal Server Error"));
    NSString *headers = [NSString stringWithFormat:
        @"HTTP/1.1 %d %@\r\nContent-Type: %@\r\nContent-Length: %lu\r\n"
         "Connection: close\r\n%@\r\n",
        status, reason, type, (unsigned long)body.length,
        cors ? @"Access-Control-Allow-Origin: *\r\nAccess-Control-Allow-Headers: Content-Type, Authorization\r\nAccess-Control-Allow-Methods: GET, POST, OPTIONS\r\n" : @""];
    NSData *head = [headers dataUsingEncoding:NSUTF8StringEncoding];
    send_all(fd, head.bytes, head.length);
    send_all(fd, body.bytes, body.length);
}

static void json_reply(int fd, int status, id object, bool cors) {
    http_reply(fd, status, @"application/json", json_data(object), cors);
}

static void error_reply(int fd, int status, NSString *message, bool cors) {
    json_reply(fd, status, @{ @"error": @{ @"message": message,
                                           @"type": @"invalid_request_error" } },
               cors);
}

static NSString *message_text(id content) {
    if ([content isKindOfClass:NSString.class]) return content;
    if (![content isKindOfClass:NSArray.class]) return nil;
    NSMutableString *text = [NSMutableString string];
    for (id part in content) {
        if (![part isKindOfClass:NSDictionary.class]) continue;
        id value = part[@"text"] ?: part[@"content"];
        if ([value isKindOfClass:NSString.class]) [text appendString:value];
    }
    return text;
}

typedef struct {
    int fd;
    bool stream;
    bool chat;
    bool cors;
    NSString *request_id;
    NSMutableData *output;
} response_sink;

static NSDictionary *chat_chunk(response_sink *sink, NSDictionary *delta,
                                id finish) {
    return @{ @"id": sink->request_id,
              @"object": @"chat.completion.chunk",
              @"created": @((long long)time(NULL)),
              @"model": @"qwen3.6-27b",
              @"choices": @[ @{ @"index": @0, @"delta": delta,
                                  @"finish_reason": finish ?: NSNull.null } ] };
}

static NSDictionary *completion_chunk(response_sink *sink, NSString *text,
                                      id finish) {
    return @{ @"id": sink->request_id,
              @"object": @"text_completion",
              @"created": @((long long)time(NULL)),
              @"model": @"qwen3.6-27b",
              @"choices": @[ @{ @"index": @0, @"text": text,
                                  @"finish_reason": finish ?: NSNull.null } ] };
}

static void sse_send(response_sink *sink, NSDictionary *event) {
    NSData *json = json_data(event);
    send_all(sink->fd, "data: ", 6);
    send_all(sink->fd, json.bytes, json.length);
    send_all(sink->fd, "\n\n", 2);
}

static void stream_headers(response_sink *sink) {
    NSString *headers = [NSString stringWithFormat:
        @"HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\n"
         "Cache-Control: no-cache\r\nConnection: close\r\n%@\r\n",
        sink->cors ? @"Access-Control-Allow-Origin: *\r\n" : @""];
    NSData *data = [headers dataUsingEncoding:NSUTF8StringEncoding];
    send_all(sink->fd, data.bytes, data.length);
    if (sink->chat) {
        sse_send(sink, chat_chunk(sink, @{ @"role": @"assistant" }, nil));
    }
}

/* Byte-BPE tokens can split UTF-8. Hold an incomplete suffix until it forms a
 * valid string, then stream it as one OpenAI chunk. */
static void emit_text(void *ud, const char *text, size_t len) {
    response_sink *sink = ud;
    [sink->output appendBytes:text length:len];
    if (!sink->stream || sink->output.length == 0) return;
    NSString *chunk = [[NSString alloc] initWithData:sink->output
                                             encoding:NSUTF8StringEncoding];
    if (!chunk) return;
    [sink->output setLength:0];
    sse_send(sink, sink->chat
        ? chat_chunk(sink, @{ @"content": chunk }, nil)
        : completion_chunk(sink, chunk, nil));
}

static int listen_socket(const char *host, int port) {
    struct addrinfo hints = {0}, *addresses = NULL;
    hints.ai_family = AF_UNSPEC;
    hints.ai_socktype = SOCK_STREAM;
    hints.ai_flags = AI_PASSIVE;
    char service[16];
    snprintf(service, sizeof(service), "%d", port);
    if (getaddrinfo(host, service, &hints, &addresses) != 0) return -1;
    int fd = -1;
    for (struct addrinfo *a = addresses; a; a = a->ai_next) {
        fd = socket(a->ai_family, a->ai_socktype, a->ai_protocol);
        if (fd < 0) continue;
        int yes = 1;
        setsockopt(fd, SOL_SOCKET, SO_REUSEADDR, &yes, sizeof(yes));
        if (bind(fd, a->ai_addr, a->ai_addrlen) == 0 && listen(fd, 16) == 0) {
            break;
        }
        close(fd);
        fd = -1;
    }
    freeaddrinfo(addresses);
    return fd;
}

static bool read_request(int fd, NSString **method, NSString **path,
                         NSData **body) {
    NSMutableData *data = [NSMutableData data];
    NSRange end = NSMakeRange(NSNotFound, 0);
    uint8_t buf[8192];
    while (data.length < 1024 * 1024) {
        ssize_t n = recv(fd, buf, sizeof(buf), 0);
        if (n <= 0) return false;
        [data appendBytes:buf length:(NSUInteger)n];
        end = [data rangeOfData:[@"\r\n\r\n"
            dataUsingEncoding:NSUTF8StringEncoding]
                           options:0 range:NSMakeRange(0, data.length)];
        if (end.location != NSNotFound) break;
    }
    if (end.location == NSNotFound) return false;
    NSUInteger body_at = NSMaxRange(end);
    NSData *head_data = [data subdataWithRange:NSMakeRange(0, end.location)];
    NSString *head = [[NSString alloc] initWithData:head_data
                                           encoding:NSUTF8StringEncoding];
    NSArray<NSString *> *lines = [head componentsSeparatedByString:@"\r\n"];
    NSArray<NSString *> *first = [lines.firstObject componentsSeparatedByString:@" "];
    if (first.count < 2) return false;
    *method = first[0];
    *path = first[1];
    NSInteger content_length = 0;
    for (NSString *line in lines) {
        NSRange colon = [line rangeOfString:@":"];
        if (colon.location == NSNotFound) continue;
        NSString *key = [[line substringToIndex:colon.location] lowercaseString];
        if ([key isEqualToString:@"content-length"]) {
            content_length = [[line substringFromIndex:colon.location + 1]
                integerValue];
        }
    }
    if (content_length < 0 || content_length > 1024 * 1024) return false;
    while (data.length - body_at < (NSUInteger)content_length) {
        ssize_t n = recv(fd, buf, sizeof(buf), 0);
        if (n <= 0) return false;
        [data appendBytes:buf length:(NSUInteger)n];
    }
    *body = [data subdataWithRange:NSMakeRange(body_at,
                                               (NSUInteger)content_length)];
    return true;
}

static void handle_completion(
        int fd, NSString *path, NSData *body, int default_tokens,
        float default_temperature, uint64_t default_seed, bool default_think,
        bool cors, qwen_http_generate_fn generate, void *generate_ud) {
    id root = [NSJSONSerialization JSONObjectWithData:body options:0 error:nil];
    if (![root isKindOfClass:NSDictionary.class]) {
        error_reply(fd, 400, @"request body must be a JSON object", cors);
        return;
    }
    NSDictionary *request = root;
    NSMutableArray<NSDictionary *> *messages = [NSMutableArray array];
    if ([path isEqualToString:@"/v1/completions"]) {
        NSString *prompt = message_text(request[@"prompt"]);
        if (prompt) [messages addObject:@{ @"role": @"user", @"content": prompt }];
    } else {
        id array = request[@"messages"];
        if ([array isKindOfClass:NSArray.class]) {
            for (id item in array) {
                if (![item isKindOfClass:NSDictionary.class]) continue;
                NSString *role = item[@"role"];
                NSString *content = message_text(item[@"content"]);
                if (role && content) {
                    [messages addObject:@{ @"role": role, @"content": content }];
                }
            }
        }
    }
    if (!messages.count) {
        error_reply(fd, 400, @"a prompt or messages array is required", cors);
        return;
    }
    NSNumber *limit = request[@"max_completion_tokens"] ?: request[@"max_tokens"];
    int max_tokens = [limit isKindOfClass:NSNumber.class] ? limit.intValue :
        default_tokens;
    if (max_tokens < 1) max_tokens = default_tokens;
    bool stream = [request[@"stream"] boolValue];
    bool think = request[@"think"] ? [request[@"think"] boolValue] :
        default_think;
    id temperature_value = request[@"temperature"];
    double temperature = temperature_value ?
        [temperature_value doubleValue] : default_temperature;
    if ((temperature_value &&
         ![temperature_value isKindOfClass:NSNumber.class]) ||
        !isfinite(temperature) || temperature < 0.0 || temperature > 100.0) {
        error_reply(fd, 400, @"temperature must be between 0 and 100", cors);
        return;
    }
    id seed_value = request[@"seed"];
    long long signed_seed = seed_value ? [seed_value longLongValue] : 0;
    if (seed_value &&
        (![seed_value isKindOfClass:NSNumber.class] || signed_seed < 0 ||
         [seed_value doubleValue] != (double)signed_seed)) {
        error_reply(fd, 400, @"seed must be a non-negative integer", cors);
        return;
    }
    uint64_t seed = seed_value ? (uint64_t)signed_seed :
        default_seed + request_seed_nonce++;

    qwen_chat_message *native = calloc(messages.count, sizeof(*native));
    for (NSUInteger i = 0; i < messages.count; i++) {
        native[i].role = [messages[i][@"role"] UTF8String];
        native[i].content = [messages[i][@"content"] UTF8String];
    }
    response_sink sink = {
        .fd = fd, .stream = stream,
        .chat = ![path isEqualToString:@"/v1/completions"], .cors = cors,
        .request_id = [NSString stringWithFormat:@"chatcmpl-%lld-%d",
            (long long)time(NULL), getpid()],
        .output = [NSMutableData data],
    };
    if (stream) stream_headers(&sink);
    qwen_http_stats stats = {0};
    int rc = generate(generate_ud, native, messages.count, max_tokens, think,
                      (float)temperature, seed, emit_text, &sink, &stats);
    free(native);
    if (rc != 0) {
        if (!stream) error_reply(fd, 500, @"generation failed", cors);
        return;
    }
    NSString *finish = stats.stopped ? @"stop" : @"length";
    if (stream) {
        emit_text(&sink, "", 0);
        sse_send(&sink, sink.chat
            ? chat_chunk(&sink, @{}, finish)
            : completion_chunk(&sink, @"", finish));
        send_all(fd, "data: [DONE]\n\n", 14);
        return;
    }
    NSString *text = [[NSString alloc] initWithData:sink.output
                                            encoding:NSUTF8StringEncoding] ?: @"";
    if ([path isEqualToString:@"/v1/completions"]) {
        json_reply(fd, 200,
            @{ @"id": sink.request_id, @"object": @"text_completion",
               @"created": @((long long)time(NULL)), @"model": @"qwen3.6-27b",
               @"choices": @[ @{ @"index": @0, @"text": text,
                                   @"finish_reason": finish } ],
               @"usage": @{ @"prompt_tokens": @(stats.prompt_tokens),
                              @"completion_tokens": @(stats.completion_tokens),
                              @"total_tokens": @(stats.prompt_tokens +
                                                  stats.completion_tokens) } }, cors);
    } else {
        json_reply(fd, 200,
            @{ @"id": sink.request_id, @"object": @"chat.completion",
               @"created": @((long long)time(NULL)), @"model": @"qwen3.6-27b",
               @"choices": @[ @{ @"index": @0,
                    @"message": @{ @"role": @"assistant", @"content": text },
                    @"finish_reason": finish } ],
               @"usage": @{ @"prompt_tokens": @(stats.prompt_tokens),
                              @"completion_tokens": @(stats.completion_tokens),
                              @"total_tokens": @(stats.prompt_tokens +
                                                  stats.completion_tokens) } }, cors);
    }
}

int qwen_http_serve(const char *host, int port, int default_tokens,
                    float default_temperature, uint64_t default_seed,
                    bool default_think, bool cors,
                    qwen_http_generate_fn generate, void *generate_ud) {
    signal(SIGPIPE, SIG_IGN);
    int server = listen_socket(host, port);
    if (server < 0) {
        fprintf(stderr, "ds4-server: cannot listen on %s:%d: %s\n",
                host, port, strerror(errno));
        return 1;
    }
    fprintf(stderr, "ds4-server: listening on http://%s:%d\n", host, port);
    for (;;) {
        int client = accept(server, NULL, NULL);
        if (client < 0) {
            if (errno == EINTR) continue;
            break;
        }
        @autoreleasepool {
            NSString *method = nil, *path = nil;
            NSData *body = nil;
            if (!read_request(client, &method, &path, &body)) {
                error_reply(client, 400, @"invalid HTTP request", cors);
            } else if ([method isEqualToString:@"OPTIONS"]) {
                http_reply(client, 200, @"text/plain", [NSData data], cors);
            } else if ([method isEqualToString:@"GET"] &&
                       ([path isEqualToString:@"/health"] ||
                        [path isEqualToString:@"/healthz"])) {
                json_reply(client, 200, @{ @"status": @"ok" }, cors);
            } else if ([method isEqualToString:@"GET"] &&
                       [path isEqualToString:@"/v1/models"]) {
                json_reply(client, 200,
                    @{ @"object": @"list", @"data": @[ @{
                        @"id": @"qwen3.6-27b", @"object": @"model" } ] }, cors);
            } else if ([method isEqualToString:@"POST"] &&
                       ([path isEqualToString:@"/v1/chat/completions"] ||
                        [path isEqualToString:@"/v1/completions"])) {
                handle_completion(client, path, body, default_tokens,
                                  default_temperature, default_seed,
                                  default_think, cors, generate, generate_ud);
            } else {
                error_reply(client, 404, @"endpoint not found", cors);
            }
        }
        close(client);
    }
    close(server);
    return 1;
}
