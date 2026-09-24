/* Minimal HTTP/1.1 + JSON scoring daemon for the cmd-risk single binary.
 *
 * Design constraints (see reports / README):
 *   - no third-party libraries: a few hundred lines of C implement the
 *     HTTP/1.1 request loop and a small, bounds-checked, depth-limited
 *     recursive-descent JSON parser.
 *   - single-threaded and serial: accept -> read -> parse -> score ->
 *     respond -> close.  `Connection: close` on every response.
 *   - default bind 127.0.0.1, no authentication.  Binding to a
 *     non-loopback address prints a warning on startup.
 *   - inference is delegated to the exact same scoring path as the batch
 *     CLI (tools/cmdrisk_single.c) so probabilities are bit-identical.
 */
#define _GNU_SOURCE
#include "cmdrisk_infer.h"

#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <errno.h>
#include <math.h>
#include <ctype.h>
#include <signal.h>

#include <unistd.h>
#include <sys/types.h>
#include <sys/select.h>
#include <sys/socket.h>
#include <sys/time.h>
#include <netinet/in.h>
#include <arpa/inet.h>

#define HEADER_CAP 65536     /* max header block (without body)      */
#define MAX_COMMANDS 1024    /* hard cap on commands per session     */
#define MAX_JSON_DEPTH 32    /* hard cap on JSON nesting             */

/* ------------------------------------------------------------------ */
/* arena + string-builder plumbing                                     */
/* ------------------------------------------------------------------ */

typedef struct {
    char *mem;
    size_t len;
    size_t cap;
} Arena;

typedef struct {
    char *s;
    size_t len;
    size_t cap;
    int oom;
} SB;

static void *arena_alloc(Arena *a, size_t n) {
    n = (n + 15u) & ~(size_t)15u;
    if (n > (size_t)-16 - a->len) return NULL;
    if (n > a->cap - a->len) {
        size_t ncap = a->cap ? a->cap * 2 : 4096;
        while (ncap < a->len + n) {
            if (ncap > ((size_t)-1) / 2) return NULL;
            ncap *= 2;
        }
        char *m = (char *)realloc(a->mem, ncap);
        if (!m) return NULL;
        a->mem = m;
        a->cap = ncap;
    }
    void *p = a->mem + a->len;
    a->len += n;
    return p;
}

static void sb_reserve(SB *sb, size_t extra) {
    if (sb->oom) return;
    if (extra > sb->cap - sb->len) {
        size_t ncap = sb->cap ? sb->cap * 2 : 256;
        size_t need = sb->len + extra + 1;
        while (ncap < need) ncap *= 2;
        char *m = (char *)realloc(sb->s, ncap);
        if (!m) { sb->oom = 1; return; }
        sb->s = m;
        sb->cap = ncap;
    }
}

static void sb_puts(SB *sb, const char *p, size_t n) {
    sb_reserve(sb, n);
    if (sb->oom) return;
    memcpy(sb->s + sb->len, p, n);
    sb->len += n;
}

static void sb_str(SB *sb, const char *p) { sb_puts(sb, p, strlen(p)); }

static void sb_putc(SB *sb, char c) { sb_puts(sb, &c, 1); }

static void sb_fmt(SB *sb, const char *fmt, ...) {
    char tmp[320];
    va_list ap;
    va_start(ap, fmt);
    int n = vsnprintf(tmp, sizeof tmp, fmt, ap);
    va_end(ap);
    if (n < 0) return;
    if ((size_t)n < sizeof tmp) { sb_puts(sb, tmp, (size_t)n); return; }
    sb_reserve(sb, (size_t)n);
    if (sb->oom) return;
    va_start(ap, fmt);
    vsnprintf(sb->s + sb->len, (size_t)n + 1, fmt, ap);
    va_end(ap);
    sb->len += (size_t)n;
}


static void sb_term(SB *sb) {
    if (sb->oom || !sb->s) return;
    sb_reserve(sb, 1);
    if (!sb->oom) sb->s[sb->len] = 0;
}

static void sb_jsonstr(SB *sb, const char *p, size_t n) {
    sb_putc(sb, '"');
    for (size_t i = 0; i < n; i++) {
        unsigned char c = (unsigned char)p[i];
        switch (c) {
        case '"':  sb_str(sb, "\\\""); break;
        case '\\': sb_str(sb, "\\\\"); break;
        case '\b': sb_str(sb, "\\b"); break;
        case '\f': sb_str(sb, "\\f"); break;
        case '\n': sb_str(sb, "\\n"); break;
        case '\r': sb_str(sb, "\\r"); break;
        case '\t': sb_str(sb, "\\t"); break;
        default:
            if (c < 0x20) sb_fmt(sb, "\\u%04x", c);
            else sb_putc(sb, (char)c);
        }
    }
    sb_putc(sb, '"');
}

/* ------------------------------------------------------------------ */
/* JSON parser (recursive descent, arena-backed, end-bounded)          */
/* ------------------------------------------------------------------ */

typedef enum { J_NULL = 0, J_BOOL, J_NUM, J_STR, J_ARR, J_OBJ } JType;

typedef struct JValue JValue;
struct JValue {
    JType type;
    double num;        /* J_NUM */
    int boolean;       /* J_BOOL */
    const char *str;   /* J_STR (NUL-terminated view into arena) */
    size_t slen;
    JValue **items;    /* J_ARR values / J_OBJ values */
    const char **keys; /* J_OBJ keys */
    size_t count;
};

/* `end` points one past the last body byte (at the NUL terminator). */
static void jws(const char **p, const char *end) {
    while (*p < end && (**p == ' ' || **p == '\t' || **p == '\n' || **p == '\r')) (*p)++;
}

static int put_utf8(char *dst, unsigned long cp) {
    if (cp <= 0x7f) { dst[0] = (char)cp; return 1; }
    if (cp <= 0x7ff) {
        dst[0] = (char)(0xc0 | (cp >> 6));
        dst[1] = (char)(0x80 | (cp & 0x3f));
        return 2;
    }
    if (cp <= 0xffff) {
        dst[0] = (char)(0xe0 | (cp >> 12));
        dst[1] = (char)(0x80 | ((cp >> 6) & 0x3f));
        dst[2] = (char)(0x80 | (cp & 0x3f));
        return 3;
    }
    dst[0] = (char)(0xf0 | (cp >> 18));
    dst[1] = (char)(0x80 | ((cp >> 12) & 0x3f));
    dst[2] = (char)(0x80 | ((cp >> 6) & 0x3f));
    dst[3] = (char)(0x80 | (cp & 0x3f));
    return 4;
}

static int hex4(const char *s, unsigned long *out) {
    unsigned long v = 0;
    for (int i = 0; i < 4; i++) {
        char c = s[i];
        v <<= 4;
        if (c >= '0' && c <= '9') v += (unsigned long)(c - '0');
        else if (c >= 'a' && c <= 'f') v += (unsigned long)(c - 'a' + 10);
        else if (c >= 'A' && c <= 'F') v += (unsigned long)(c - 'A' + 10);
        else return -1;
    }
    *out = v;
    return 0;
}

static int jparse_string(Arena *a, const char **p, const char *end,
                         const char **out, size_t *outlen, char *err, size_t errcap) {
    if (*p >= end || **p != '"') { snprintf(err, errcap, "expected string"); return -1; }
    const char *q = *p + 1;
    while (q < end) {
        if (*q == '"') break;
        if (*q == '\\') {
            q++;
            if (q >= end) { snprintf(err, errcap, "unterminated escape"); return -1; }
            if (*q == 'u') {
                if (q + 4 >= end) { snprintf(err, errcap, "bad unicode escape"); return -1; }
                q += 5;
            } else q++;
            continue;
        }
        q++;
    }
    if (q >= end) { snprintf(err, errcap, "unterminated string"); return -1; }
    size_t raw = (size_t)(q - (*p + 1));
    char *dst = (char *)arena_alloc(a, raw + 1);
    if (!dst) { snprintf(err, errcap, "out of memory"); return -1; }
    const char *s = *p + 1;
    size_t n = 0;
    for (size_t i = 0; i < raw; i++) {
        char c = s[i];
        if (c != '\\') { dst[n++] = c; continue; }
        i++;
        char e = s[i];
        switch (e) {
        case '"':  dst[n++] = '"'; break;
        case '\\': dst[n++] = '\\'; break;
        case '/':  dst[n++] = '/'; break;
        case 'b':  dst[n++] = '\b'; break;
        case 'f':  dst[n++] = '\f'; break;
        case 'n':  dst[n++] = '\n'; break;
        case 'r':  dst[n++] = '\r'; break;
        case 't':  dst[n++] = '\t'; break;
        case 'u': {
            unsigned long cp = 0xfffdUL;
            if (i + 5 <= raw && hex4(s + i + 1, &cp) == 0) {
                i += 4;
                if (cp >= 0xd800UL && cp <= 0xdbffUL && i + 7 <= raw &&
                    s[i + 1] == '\\' && s[i + 2] == 'u') {
                    unsigned long lo = 0;
                    if (hex4(s + i + 3, &lo) == 0 && lo >= 0xdc00UL && lo <= 0xdfffUL) {
                        cp = 0x10000UL + ((cp - 0xd800UL) << 10) + (lo - 0xdc00UL);
                        i += 6;
                    }
                }
            }
            n += (size_t)put_utf8(dst + n, cp);
            break;
        }
        default:
            break; /* invalid escape: drop it */
        }
    }
    dst[n] = 0;
    *p = q + 1;
    *out = dst;
    *outlen = n;
    return 0;
}

static int jparse_number(Arena *a, const char **p, const char *end, JValue *v,
                         char *err, size_t errcap) {
    (void)a;
    const char *s = *p;
    if (s < end && *s == '-') s++;
    if (s >= end || !isdigit((unsigned char)*s)) { snprintf(err, errcap, "invalid number"); return -1; }
    if (*s == '0') s++;
    else while (s < end && isdigit((unsigned char)*s)) s++;
    if (s < end && *s == '.') {
        s++;
        if (s >= end || !isdigit((unsigned char)*s)) { snprintf(err, errcap, "invalid number"); return -1; }
        while (s < end && isdigit((unsigned char)*s)) s++;
    }
    if (s < end && (*s == 'e' || *s == 'E')) {
        s++;
        if (s < end && (*s == '+' || *s == '-')) s++;
        if (s >= end || !isdigit((unsigned char)*s)) { snprintf(err, errcap, "invalid number"); return -1; }
        while (s < end && isdigit((unsigned char)*s)) s++;
    }
    char *ep = NULL;
    double val = strtod(*p, &ep);
    if (ep != s || !isfinite(val)) { snprintf(err, errcap, "invalid number"); return -1; }
    *p = s;
    v->type = J_NUM;
    v->num = val;
    return 0;
}

static int jparse_value(Arena *a, const char **p, const char *end, JValue *v, int depth,
                        char *err, size_t errcap);

static int jparse_array(Arena *a, const char **p, const char *end, JValue *v, int depth,
                        char *err, size_t errcap) {
    (*p)++;
    v->type = J_ARR;
    v->items = NULL;
    v->count = 0;
    jws(p, end);
    if (*p >= end) { snprintf(err, errcap, "unterminated array"); return -1; }
    if (**p == ']') { (*p)++; return 0; }
    size_t cap = 0, count = 0;
    JValue **items = NULL;
    for (;;) {
        if (count == cap) {
            size_t ncap = cap ? cap * 2 : 8;
            if (ncap > 1024 * 1024) { snprintf(err, errcap, "array too large"); return -1; }
            JValue **nitems = (JValue **)arena_alloc(a, ncap * sizeof(JValue *));
            if (!nitems) { snprintf(err, errcap, "out of memory"); return -1; }
            if (cap) memcpy(nitems, items, cap * sizeof(JValue *));
            items = nitems;
            cap = ncap;
        }
        JValue *item = (JValue *)arena_alloc(a, sizeof *item);
        if (!item) { snprintf(err, errcap, "out of memory"); return -1; }
        memset(item, 0, sizeof *item);
        if (jparse_value(a, p, end, item, depth, err, errcap) != 0) return -1;
        items[count++] = item;
        jws(p, end);
        if (*p >= end) { snprintf(err, errcap, "unterminated array"); return -1; }
        if (**p == ',') { (*p)++; jws(p, end); continue; }
        if (**p == ']') { (*p)++; break; }
        snprintf(err, errcap, "expected ',' or ']'");
        return -1;
    }
    v->items = items;
    v->count = count;
    return 0;
}

static int jparse_object(Arena *a, const char **p, const char *end, JValue *v, int depth,
                         char *err, size_t errcap) {
    (*p)++;
    v->type = J_OBJ;
    v->items = NULL;
    v->keys = NULL;
    v->count = 0;
    jws(p, end);
    if (*p >= end) { snprintf(err, errcap, "unterminated object"); return -1; }
    if (**p == '}') { (*p)++; return 0; }
    size_t cap = 0, count = 0;
    JValue **items = NULL;
    const char **keys = NULL;
    for (;;) {
        if (count == cap) {
            size_t ncap = cap ? cap * 2 : 8;
            if (ncap > 1024 * 1024) { snprintf(err, errcap, "object too large"); return -1; }
            JValue **nitems = (JValue **)arena_alloc(a, ncap * sizeof(JValue *));
            const char **nkeys = (const char **)arena_alloc(a, ncap * sizeof(char *));
            if (!nitems || !nkeys) { snprintf(err, errcap, "out of memory"); return -1; }
            if (cap) {
                memcpy(nitems, items, cap * sizeof(JValue *));
                memcpy(nkeys, keys, cap * sizeof(char *));
            }
            items = nitems;
            keys = nkeys;
            cap = ncap;
        }
        const char *key = NULL;
        size_t keylen = 0;
        if (jparse_string(a, p, end, &key, &keylen, err, errcap) != 0) return -1;
        jws(p, end);
        if (*p >= end || **p != ':') { snprintf(err, errcap, "expected ':'"); return -1; }
        (*p)++;
        jws(p, end);
        JValue *item = (JValue *)arena_alloc(a, sizeof *item);
        if (!item) { snprintf(err, errcap, "out of memory"); return -1; }
        memset(item, 0, sizeof *item);
        if (jparse_value(a, p, end, item, depth, err, errcap) != 0) return -1;
        keys[count] = key;
        items[count] = item;
        count++;
        jws(p, end);
        if (*p >= end) { snprintf(err, errcap, "unterminated object"); return -1; }
        if (**p == ',') { (*p)++; jws(p, end); continue; }
        if (**p == '}') { (*p)++; break; }
        snprintf(err, errcap, "expected ',' or '}'");
        return -1;
    }
    v->items = items;
    v->keys = keys;
    v->count = count;
    return 0;
}

static int jparse_value(Arena *a, const char **p, const char *end, JValue *v, int depth,
                        char *err, size_t errcap) {
    jws(p, end);
    if (*p >= end) { snprintf(err, errcap, "unexpected end of JSON"); return -1; }
    if (depth > MAX_JSON_DEPTH) { snprintf(err, errcap, "JSON nesting too deep"); return -1; }
    char c = **p;
    switch (c) {
    case '{': return jparse_object(a, p, end, v, depth + 1, err, errcap);
    case '[': return jparse_array(a, p, end, v, depth + 1, err, errcap);
    case '"':
        v->type = J_STR;
        return jparse_string(a, p, end, &v->str, &v->slen, err, errcap);
    case 't':
        if (end - *p >= 4 && strncmp(*p, "true", 4) == 0) {
            *p += 4; v->type = J_BOOL; v->boolean = 1; return 0;
        }
        snprintf(err, errcap, "invalid literal");
        return -1;
    case 'f':
        if (end - *p >= 5 && strncmp(*p, "false", 5) == 0) {
            *p += 5; v->type = J_BOOL; v->boolean = 0; return 0;
        }
        snprintf(err, errcap, "invalid literal");
        return -1;
    case 'n':
        if (end - *p >= 4 && strncmp(*p, "null", 4) == 0) {
            *p += 4; v->type = J_NULL; return 0;
        }
        snprintf(err, errcap, "invalid literal");
        return -1;
    default:
        if (c == '-' || (c >= '0' && c <= '9'))
            return jparse_number(a, p, end, v, err, errcap);
        snprintf(err, errcap, "unexpected character 0x%02x", (unsigned char)c);
        return -1;
    }
}

static const JValue *jget(const JValue *obj, const char *key) {
    if (!obj || obj->type != J_OBJ) return NULL;
    for (size_t i = 0; i < obj->count; i++)
        if (obj->keys[i] && strcmp(obj->keys[i], key) == 0) return obj->items[i];
    return NULL;
}

/* ------------------------------------------------------------------ */
/* HTTP plumbing                                                       */
/* ------------------------------------------------------------------ */

static const char *status_text(int code) {
    switch (code) {
    case 200: return "OK";
    case 400: return "Bad Request";
    case 404: return "Not Found";
    case 405: return "Method Not Allowed";
    case 408: return "Request Timeout";
    case 413: return "Payload Too Large";
    case 431: return "Request Header Fields Too Large";
    case 500: return "Internal Server Error";
    case 505: return "HTTP Version Not Supported";
    default: return "Error";
    }
}

static const char *err_name_for(int code) {
    switch (code) {
    case 400: return "invalid_request";
    case 404: return "not_found";
    case 405: return "method_not_allowed";
    case 408: return "request_timeout";
    case 413: return "payload_too_large";
    case 431: return "headers_too_large";
    case 500: return "inference_error";
    case 505: return "http_version_not_supported";
    default: return "error";
    }
}

static int send_all(int fd, const char *buf, size_t n) {
    size_t off = 0;
    while (off < n) {
        ssize_t r = send(fd, buf + off, n - off, MSG_NOSIGNAL);
        if (r < 0) {
            if (errno == EINTR) continue;
            return -1;
        }
        off += (size_t)r;
    }
    return 0;
}

static void respond_full(int fd, int code, int is_head, const char *extra_header, const char *body) {
    char hdr[512];
    size_t blen = strlen(body);
    snprintf(hdr, sizeof hdr,
             "HTTP/1.1 %d %s\r\nContent-Type: application/json; charset=utf-8\r\n"
             "Content-Length: %zu\r\nConnection: close\r\n%s\r\n",
             code, status_text(code), blen, extra_header ? extra_header : "");
    send_all(fd, hdr, strlen(hdr));
    if (!is_head) send_all(fd, body, blen);
}

static void respond_error_full(int fd, int code, const char *err_code, const char *msg,
                               int is_head, const char *allow) {
    SB sb = {0};
    sb_str(&sb, "{\"error\":{\"code\":");
    sb_jsonstr(&sb, err_code, strlen(err_code));
    sb_str(&sb, ",\"message\":");
    sb_jsonstr(&sb, msg, strlen(msg));
    sb_str(&sb, "}}");
    sb_term(&sb);
    respond_full(fd, code, is_head, allow, sb.s ? sb.s : "{}");
    free(sb.s);
}

static void respond_error(int fd, int code, const char *err_code, const char *msg, int is_head) {
    respond_error_full(fd, code, err_code, msg, is_head, NULL);
}

/* ------------------------------------------------------------------ */
/* request reading                                                     */
/* ------------------------------------------------------------------ */

typedef struct {
    char method[16];
    char path[2048];
    int http_minor;   /* 0 or 1; -1 = unsupported */
    char *buf;        /* owner of the whole request buffer */
    char *body;       /* NUL-terminated view into buf     */
    size_t body_len;
} HttpReq;

static int recv_some(int fd, char *buf, size_t *len, size_t cap,
                     int *errcode, const char **errmsg) {
    if (*len >= cap) { *errcode = 413; *errmsg = "request body too large"; return -1; }
    ssize_t r = recv(fd, buf + *len, cap - *len, 0);
    if (r < 0) {
        if (errno == EINTR) return 0;
        if (errno == EAGAIN || errno == EWOULDBLOCK) {
            *errcode = 408;
            *errmsg = "request timed out";
            return -1;
        }
        *errcode = 400;
        *errmsg = "read error";
        return -1;
    }
    if (r == 0) {
        if (*len == 0) { *errcode = 400; *errmsg = "connection closed before request"; }
        else { *errcode = 400; *errmsg = "connection closed while reading request"; }
        return -1;
    }
    *len += (size_t)r;
    return 1;
}

/* Returns 0 on success; on failure sets *errcode and *errmsg and returns -1. */
static int read_request(int fd, size_t max_body, HttpReq *req,
                        int *errcode, const char **errmsg) {
    memset(req, 0, sizeof *req);
    req->http_minor = 1;
    size_t cap = HEADER_CAP + max_body + 1;
    req->buf = (char *)malloc(cap);
    if (!req->buf) { *errcode = 500; *errmsg = "out of memory"; return -1; }
    size_t len = 0;

    /* header block until CRLFCRLF */
    char *hend = NULL;
    for (;;) {
        int st = recv_some(fd, req->buf, &len, cap - 1, errcode, errmsg);
        if (st < 0) return -1;
        if (st == 0) continue;
        if (len >= 4 && (hend = (char *)memmem(req->buf, len, "\r\n\r\n", 4)) != NULL) break;
        if (len >= HEADER_CAP) { *errcode = 431; *errmsg = "request headers too large"; return -1; }
    }
    hend += 4;
    size_t hlen = (size_t)(hend - req->buf);

    /* request line: METHOD SP TARGET SP HTTP/1.x */
    const char *rp = req->buf;
    const char *rlend = memchr(rp, '\r', hlen);
    if (!rlend) { *errcode = 400; *errmsg = "malformed request line"; return -1; }
    if (rlend == rp) { *errcode = 400; *errmsg = "empty method"; return -1; }
    const char *sp1 = memchr(rp, ' ', (size_t)(rlend - rp));
    if (!sp1 || sp1 == rp) { *errcode = 400; *errmsg = "malformed request line"; return -1; }
    size_t mlen = (size_t)(sp1 - rp);
    if (mlen >= sizeof req->method) { *errcode = 400; *errmsg = "method too long"; return -1; }
    memcpy(req->method, rp, mlen);
    req->method[mlen] = 0;
    const char *tp = sp1 + 1;
    const char *sp2 = memchr(tp, ' ', (size_t)(rlend - tp));
    if (!sp2 || sp2 == tp) { *errcode = 400; *errmsg = "malformed request target"; return -1; }
    size_t tlen = (size_t)(sp2 - tp);
    if (tlen >= sizeof req->path) { *errcode = 400; *errmsg = "request target too long"; return -1; }
    memcpy(req->path, tp, tlen);
    req->path[tlen] = 0;
    if (sp2 + 9 > rlend || strncmp(sp2 + 1, "HTTP/1.", 7) != 0) {
        *errcode = 505; *errmsg = "only HTTP/1.0 and HTTP/1.1 are supported";
        return -1;
    }
    int minor = sp2[8] - '0';
    if (minor != 0 && minor != 1) {
        *errcode = 505; *errmsg = "only HTTP/1.0 and HTTP/1.1 are supported";
        return -1;
    }
    req->http_minor = minor;
    /* strip query string */
    char *qq = strchr(req->path, '?');
    if (qq) *qq = 0;

    /* headers: find Content-Length (case-insensitive, conflict-rejected) */
    int has_cl = 0;
    unsigned long long cl_value = 0;
    const char *hp = req->buf + (size_t)(rlend - req->buf) + 2;
    while (hp < hend - 2) {
        const char *le = (const char *)memmem(hp, (size_t)(hend - hp), "\r\n", 2);
        if (!le) break;
        if (le > hp) {
            const char *colon = memchr(hp, ':', (size_t)(le - hp));
            if (colon) {
                size_t hname = (size_t)(colon - hp);
                if (hname == 14 && strncasecmp(hp, "Content-Length", 14) == 0) {
                    const char *v = colon + 1;
                    while (v < le && (*v == ' ' || *v == '\t')) v++;
                    const char *ve = le;
                    while (ve > v && (ve[-1] == ' ' || ve[-1] == '\t')) ve--;
                    if (ve == v) { *errcode = 400; *errmsg = "empty Content-Length"; return -1; }
                    unsigned long long cl = 0;
                    for (const char *d = v; d < ve; d++) {
                        if (!isdigit((unsigned char)*d)) {
                            *errcode = 400; *errmsg = "invalid Content-Length";
                            return -1;
                        }
                        if (cl > (18446744073709551615ULL - (unsigned long long)(*d - '0')) / 10ULL) {
                            *errcode = 400; *errmsg = "Content-Length overflow";
                            return -1;
                        }
                        cl = cl * 10ULL + (unsigned long long)(*d - '0');
                    }
                    if (has_cl && cl != cl_value) {
                        *errcode = 400; *errmsg = "conflicting Content-Length";
                        return -1;
                    }
                    has_cl = 1;
                    cl_value = cl;
                }
            }
        }
        hp = le + 2;
    }

    /* body completeness */
    size_t bstart = hlen;
    int is_getish = (strcmp(req->method, "GET") == 0 || strcmp(req->method, "HEAD") == 0);
    if (!has_cl) {
        if (!is_getish) { *errcode = 400; *errmsg = "Content-Length required"; return -1; }
        cl_value = 0;
    }
    if (cl_value > max_body) { *errcode = 413; *errmsg = "request body too large"; return -1; }
    while (len < bstart + cl_value) {
        int st = recv_some(fd, req->buf, &len, cap - 1, errcode, errmsg);
        if (st < 0) return -1;
    }
    req->buf[bstart + cl_value] = 0;
    req->body = req->buf + bstart;
    req->body_len = (size_t)cl_value;
    return 0;
}

/* ------------------------------------------------------------------ */
/* route handlers                                                      */
/* ------------------------------------------------------------------ */

static void handle_healthz(int fd, int is_head, const char *bind_spec) {
    SB rb = {0};
    sb_str(&rb, "{\"ok\":true,\"model\":");
    sb_jsonstr(&rb, cmdrisk_model_version(), strlen(cmdrisk_model_version()));
    sb_str(&rb, ",\"version\":");
    sb_jsonstr(&rb, cmdrisk_server_version(), strlen(cmdrisk_server_version()));
    sb_str(&rb, ",\"bind\":");
    sb_jsonstr(&rb, bind_spec, strlen(bind_spec));
    sb_str(&rb, "}");
    sb_term(&rb);
    respond_full(fd, 200, is_head, NULL, rb.s ? rb.s : "{}");
    free(rb.s);
}

typedef struct {
    double mal_logit, prob, risk;
    int alert, top_tact, top_tech;
    char intent[256];
} Answer;

static void handle_score(int fd, const char *body, size_t body_len) {
    Arena arena = {0};
    char err[256] = {0};
    JValue root;
    memset(&root, 0, sizeof root);
    const char *p = body;
    const char *end = body + body_len;
    if (jparse_value(&arena, &p, end, &root, 0, err, sizeof err) != 0) {
        respond_error(fd, 400, "invalid_json", err, 0);
        free(arena.mem);
        return;
    }
    jws(&p, end);
    if (p != end) {
        respond_error(fd, 400, "invalid_json", "trailing data after JSON value", 0);
        free(arena.mem);
        return;
    }
    if (root.type != J_OBJ) {
        respond_error(fd, 400, "invalid_request", "request body must be a JSON object", 0);
        free(arena.mem);
        return;
    }

    /* session_id (optional, default generated) */
    static unsigned long g_seq = 0;
    char defsid[64];
    const char *sid = NULL;
    const JValue *sidv = jget(&root, "session_id");
    if (sidv && sidv->type == J_STR) {
        if (sidv->slen == 0) {
            respond_error(fd, 400, "invalid_request", "session_id must be a non-empty string", 0);
            free(arena.mem);
            return;
        }
        sid = sidv->str;
    } else if (sidv && sidv->type != J_NULL) {
        respond_error(fd, 400, "invalid_request", "session_id must be a string or null", 0);
        free(arena.mem);
        return;
    }
    if (!sid) {
        snprintf(defsid, sizeof defsid, "sess-%d-%lu", (int)getpid(), ++g_seq);
        sid = defsid;
    }

    /* commands (required, non-empty array) */
    const JValue *cmds = jget(&root, "commands");
    if (!cmds || cmds->type != J_ARR) {
        respond_error(fd, 400, "invalid_request", "commands must be a non-empty array", 0);
        free(arena.mem);
        return;
    }
    if (cmds->count == 0) {
        respond_error(fd, 400, "invalid_request", "commands must be a non-empty array", 0);
        free(arena.mem);
        return;
    }
    if (cmds->count > MAX_COMMANDS) {
        respond_error(fd, 400, "invalid_request", "too many commands", 0);
        free(arena.mem);
        return;
    }
    size_t ncmds = cmds->count;

    /* normalize each command item into a canonical \t-suffixed segment */
    const char **segs = (const char **)arena_alloc(&arena, sizeof(char *) * ncmds);
    size_t *seglens = (size_t *)arena_alloc(&arena, sizeof(size_t) * ncmds);
    if (!segs || !seglens) {
        respond_error(fd, 500, "inference_error", "out of memory", 0);
        free(arena.mem);
        return;
    }
    for (size_t idx = 0; idx < ncmds; idx++) {
        const JValue *item = cmds->items[idx];
        const char *cmd = NULL;
        size_t cmdlen = 0;
        double dt = -1.0;
        int has_dt = 0;
        if (item->type == J_STR) {
            cmd = item->str;
            cmdlen = item->slen;
        } else if (item->type == J_OBJ) {
            const JValue *c = jget(item, "cmd");
            if (!c || c->type != J_STR) {
                respond_error(fd, 400, "invalid_request", "command object requires a string 'cmd'", 0);
                free(arena.mem);
                return;
            }
            cmd = c->str;
            cmdlen = c->slen;
            const JValue *d = jget(item, "dt");
            if (d && d->type == J_NUM) {
                if (d->num < 0) {
                    respond_error(fd, 400, "invalid_request", "dt must be a non-negative number or null", 0);
                    free(arena.mem);
                    return;
                }
                dt = d->num;
                has_dt = 1;
            } else if (d && d->type != J_NULL) {
                respond_error(fd, 400, "invalid_request", "dt must be a non-negative number or null", 0);
                free(arena.mem);
                return;
            }
        } else {
            respond_error(fd, 400, "invalid_request", "each command must be a string or an object", 0);
            free(arena.mem);
            return;
        }
        int ok = 0;
        for (size_t i = 0; i < cmdlen; i++)
            if (!isspace((unsigned char)cmd[i])) { ok = 1; break; }
        if (!ok) {
            respond_error(fd, 400, "invalid_request", "command must contain a non-empty cmd", 0);
            free(arena.mem);
            return;
        }
        char *seg = (char *)arena_alloc(&arena, cmdlen + 64);
        if (!seg) {
            respond_error(fd, 500, "inference_error", "out of memory", 0);
            free(arena.mem);
            return;
        }
        size_t n = 0;
        for (size_t i = 0; i < cmdlen; i++) {
            unsigned char ch = (unsigned char)cmd[i];
            if (ch == '\t' || ch == '\x1f' || ch == '\n' || ch == '\r') ch = ' ';
            seg[n++] = (char)ch;
        }
        if (has_dt) {
            int got = snprintf(seg + n, cmdlen + 64 - n, "\t%.17g", dt);
            if (got > 0) n += (size_t)got;
        }
        seg[n] = 0;
        segs[idx] = seg;
        seglens[idx] = n;
    }

    /* score every prefix commands[0..k], mirroring the Python scorer */
    Answer *answers = (Answer *)arena_alloc(&arena, sizeof(Answer) * ncmds);
    if (!answers) {
        respond_error(fd, 500, "inference_error", "out of memory", 0);
        free(arena.mem);
        return;
    }
    double th = cmdrisk_threshold();
    SB line = {0};
    for (size_t k = 0; k < ncmds; k++) {
        line.len = 0;
        for (size_t j = 0; j <= k; j++) {
            if (j) sb_putc(&line, '\x1f');
            sb_puts(&line, segs[j], seglens[j]);
        }
        sb_putc(&line, '\0');
        if (line.oom || !line.s) {
            respond_error(fd, 500, "inference_error", "out of memory", 0);
            free(line.s);
            free(arena.mem);
            return;
        }
        Answer *a = &answers[k];
        CmdScoreOut out;
        if (cmdrisk_score_prefix(line.s, &out) != 0) {
            respond_error(fd, 500, "inference_error", "prefix failed to encode", 0);
            free(line.s);
            free(arena.mem);
            return;
        }
        a->mal_logit = out.mal_logit;
        a->prob = out.prob;
        a->alert = (out.prob >= th) || out.rule_hit;
        a->top_tact = out.top_tact;
        a->top_tech = out.top_tech;
        snprintf(a->intent, sizeof a->intent, "%s", out.intent);
        double sev = out.top_tact >= 0 ? cmdrisk_tact_severity(out.top_tact) : 0.0;
        double risk = sev > 0.0 ? out.prob * sev : 0.0;
        if (risk < 0.0) risk = 0.0;
        if (risk > 1.0) risk = 1.0;
        a->risk = risk;
    }
    free(line.s);

    double peak = 0.0;
    int alerted = 0;
    for (size_t k = 0; k < ncmds; k++) {
        if (answers[k].prob > peak) peak = answers[k].prob;
        if (answers[k].alert) alerted = 1;
    }

    SB rb = {0};
    sb_str(&rb, "{\"session_id\":");
    sb_jsonstr(&rb, sid, strlen(sid));
    sb_str(&rb, ",\"model_version\":");
    sb_jsonstr(&rb, cmdrisk_model_version(), strlen(cmdrisk_model_version()));
    sb_str(&rb, ",\"schema_version\":\"1.0\",\"peak_probability\":");
    sb_fmt(&rb, "%.9g", peak);
    sb_str(&rb, ",\"alerted\":");
    sb_str(&rb, alerted ? "true" : "false");
    sb_str(&rb, ",\"answers\":[");
    for (size_t k = 0; k < ncmds; k++) {
        const Answer *a = &answers[k];
        if (k) sb_putc(&rb, ',');
        sb_str(&rb, "{\"command_index\":");
        sb_fmt(&rb, "%zu", k);
        sb_str(&rb, ",\"malicious_probability\":");
        sb_fmt(&rb, "%.9g", a->prob);
        sb_str(&rb, ",\"risk_score\":");
        sb_fmt(&rb, "%.9g", a->risk);
        sb_str(&rb, ",\"attack_stage\":");
        const char *stage = (a->top_tact >= 0 && cmdrisk_tact_name(a->top_tact))
                                ? cmdrisk_tact_name(a->top_tact) : "unknown";
        sb_jsonstr(&rb, stage, strlen(stage));
        sb_str(&rb, ",\"intent_classes\":");
        if (a->intent[0]) sb_jsonstr(&rb, a->intent, strlen(a->intent));
        else sb_str(&rb, "\"-\"");
        sb_str(&rb, ",\"tactic\":");
        const char *tac = (a->alert && a->top_tact >= 0 && cmdrisk_tact_label(a->top_tact))
                              ? cmdrisk_tact_label(a->top_tact) : "-";
        sb_jsonstr(&rb, tac, strlen(tac));
        sb_str(&rb, ",\"technique\":");
        const char *tech = (a->alert && a->top_tech >= 0 && cmdrisk_tech_label(a->top_tech))
                               ? cmdrisk_tech_label(a->top_tech) : "-";
        sb_jsonstr(&rb, tech, strlen(tech));
        sb_str(&rb, ",\"alert\":");
        sb_str(&rb, a->alert ? "true" : "false");
        sb_str(&rb, "}");
    }
    sb_str(&rb, "]}");
    sb_term(&rb);
    respond_full(fd, 200, 0, NULL, rb.s ? rb.s : "{}");
    free(rb.s);
    free(arena.mem);
}

/* ------------------------------------------------------------------ */
/* per-connection dispatch                                              */
/* ------------------------------------------------------------------ */

static void handle_conn(int fd, size_t max_body, int timeout_sec, const char *bind_spec) {
    struct timeval tv = { timeout_sec, 0 };
    (void)setsockopt(fd, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof tv);
    (void)setsockopt(fd, SOL_SOCKET, SO_SNDTIMEO, &tv, sizeof tv);

    HttpReq req;
    int errcode = 0;
    const char *errmsg = NULL;
    if (read_request(fd, max_body, &req, &errcode, &errmsg) != 0) {
        respond_error(fd, errcode, err_name_for(errcode), errmsg, 0);
        free(req.buf);
        return;
    }
    int is_get = strcmp(req.method, "GET") == 0;
    int is_head = strcmp(req.method, "HEAD") == 0;
    int is_post = strcmp(req.method, "POST") == 0;

    if (strcmp(req.path, "/healthz") == 0) {
        if (is_get || is_head) handle_healthz(fd, is_head, bind_spec);
        else respond_error_full(fd, 405, "method_not_allowed", "use GET or HEAD",
                                is_head, "Allow: GET, HEAD\r\n");
    } else if (strcmp(req.path, "/v1/score") == 0) {
        if (is_post) handle_score(fd, req.body, req.body_len);
        else respond_error_full(fd, 405, "method_not_allowed", "use POST",
                                is_head, "Allow: POST\r\n");
    } else {
        respond_error(fd, 404, "not_found", "unknown path", is_head);
    }
    free(req.buf);
}

/* ------------------------------------------------------------------ */
/* daemon entrypoint                                                   */
/* ------------------------------------------------------------------ */

static volatile sig_atomic_t g_stop = 0;

static void on_signal(int sig) {
    (void)sig;
    g_stop = 1;
}

static int parse_bind(const char *spec, char *host, size_t hostcap, int *port) {
    const char *colon = strrchr(spec, ':');
    if (!colon || colon == spec) return -1;
    size_t hl = (size_t)(colon - spec);
    if (hl == 0 || hl >= hostcap) return -1;
    memcpy(host, spec, hl);
    host[hl] = 0;
    if (strchr(host, ':')) return -1; /* IPv6 not supported in v1 */
    char *end = NULL;
    long p = strtol(colon + 1, &end, 10);
    if (!end || *end || p < 1 || p > 65535) return -1;
    *port = (int)p;
    return 0;
}

static int open_listen(const char *host, int port, char *err, size_t errcap) {
    int fd = socket(AF_INET, SOCK_STREAM, 0);
    if (fd < 0) { snprintf(err, errcap, "socket: %s", strerror(errno)); return -1; }
    int one = 1;
    (void)setsockopt(fd, SOL_SOCKET, SO_REUSEADDR, &one, sizeof one);
    struct sockaddr_in sa;
    memset(&sa, 0, sizeof sa);
    sa.sin_family = AF_INET;
    sa.sin_port = htons((uint16_t)port);
    if (inet_pton(AF_INET, host, &sa.sin_addr) != 1) {
        snprintf(err, errcap, "cannot resolve host '%s' (use a numeric IPv4 address)", host);
        close(fd);
        return -1;
    }
    if (bind(fd, (struct sockaddr *)&sa, sizeof sa) != 0) {
        snprintf(err, errcap, "bind %s:%d: %s", host, port, strerror(errno));
        close(fd);
        return -1;
    }
    if (listen(fd, 64) != 0) {
        snprintf(err, errcap, "listen: %s", strerror(errno));
        close(fd);
        return -1;
    }
    return fd;
}

int cmdrisk_serve(const char *bind_spec, size_t max_body, int timeout_sec) {
    char host[64];
    int port = 0;
    if (parse_bind(bind_spec, host, sizeof host, &port) != 0) {
        fprintf(stderr, "serve: invalid --bind '%s' (expected IPv4 host:port, e.g. 127.0.0.1:8080)\n",
                bind_spec);
        return 2;
    }
    if (cmdrisk_calibration_setup() != 0) return 1;
    fprintf(stderr, "operating-point: threshold=%.9g temperature=%.9g\n",
            cmdrisk_threshold(), cmdrisk_temperature());
    if (cmdrisk_model_setup() != 0) return 1;
    if (cmdrisk_workspace_alloc() != 0) return 1;

    char err[256];
    int listen_fd = open_listen(host, port, err, sizeof err);
    if (listen_fd < 0) {
        fprintf(stderr, "serve: %s\n", err);
        return 1;
    }

    (void)signal(SIGPIPE, SIG_IGN);
    struct sigaction sa;
    memset(&sa, 0, sizeof sa);
    sa.sa_handler = on_signal;
    sigemptyset(&sa.sa_mask);
    sigaction(SIGINT, &sa, NULL);
    sigaction(SIGTERM, &sa, NULL);

    if (strcmp(host, "127.0.0.1") != 0)
        fprintf(stderr,
                "warning: binding to %s exposes the scorer on the network and "
                "no authentication is configured\n", host);
    fprintf(stderr, "listening on %s\n", bind_spec);

    for (;;) {
        fd_set rfds;
        FD_ZERO(&rfds);
        FD_SET(listen_fd, &rfds);
        struct timeval tv = { 0, 500000 };  /* 500 ms poll for g_stop */
        int rc = select(listen_fd + 1, &rfds, NULL, NULL, &tv);
        if (rc < 0) {
            if (errno == EINTR) continue; /* handler only sets g_stop */
            fprintf(stderr, "select: %s\n", strerror(errno));
            continue;
        }
        if (rc == 0) {
            if (g_stop) break;
            continue;
        }
        int fd = accept(listen_fd, NULL, NULL);
        if (fd < 0) {
            if (g_stop) break;
            fprintf(stderr, "accept: %s\n", strerror(errno));
            continue;
        }
        handle_conn(fd, max_body, timeout_sec, bind_spec);
        close(fd);
    }
    close(listen_fd);
    fprintf(stderr, "shutting down\n");
    return 0;
}
