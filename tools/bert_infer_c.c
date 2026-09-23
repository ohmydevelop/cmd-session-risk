/* cmd-session-risk C BERT inference prototype (token-id input).
 * build: cc -O3 -march=native tools/bert_infer_c.c -lm -o tools/bert_infer_c
 */
#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <unistd.h>
#include <errno.h>

typedef float f32;

struct Cfg {
    int vocab, hidden, layers, heads, intermediate, max_seq;
    float eps;
    int ntact, ntech;
};

static unsigned char *gblob;
static size_t gblob_len;
static size_t goff;

static const f32 *take(size_t count) {
    size_t bytes = count * sizeof(f32);
    if (goff + bytes > gblob_len) {
        fprintf(stderr, "weight blob overrun at %zu (need %zu, have %zu)\n", goff, bytes, gblob_len);
        exit(2);
    }
    const f32 *p = (const f32 *)(gblob + goff);
    goff += bytes;
    return p;
}

static int load_blob(const char *path) {
    FILE *fp = fopen(path, "rb");
    if (!fp) { fprintf(stderr, "cannot open %s: %s\n", path, strerror(errno)); return -1; }
    if (fseek(fp, 0, SEEK_END) != 0) return -1;
    long sz = ftell(fp);
    if (sz < 0) return -1;
    rewind(fp);
    gblob_len = (size_t)sz;
    gblob = (unsigned char *)malloc(gblob_len ? gblob_len : 1);
    if (!gblob) return -1;
    if (fread(gblob, 1, gblob_len, fp) != gblob_len) { fclose(fp); return -1; }
    fclose(fp);
    return 0;
}

struct Model {
    struct Cfg cfg;
    const f32 *emb_words;
    const f32 *emb_pos;
    const f32 *emb_ttype;
    const f32 *emb_ln_w;
    const f32 *emb_ln_b;
    const f32 *pooler_w;
    const f32 *pooler_b;
    const f32 *mal_w;
    const f32 *mal_b;
    const f32 *tact_w;
    const f32 *tact_b;
    const f32 *tech_w;
    const f32 *tech_b;
    const f32 *q_w[8], *q_b[8], *k_w[8], *k_b[8], *v_w[8], *v_b[8];
    const f32 *attn_o_w[8], *attn_o_b[8], *attn_ln_w[8], *attn_ln_b[8];
    const f32 *ffn1_w[8], *ffn1_b[8];
    const f32 *ffn2_w[8], *ffn2_b[8], *ffn_ln_w[8], *ffn_ln_b[8];
};

static int load_model(const char *path, struct Model *m) {
    if (load_blob(path) != 0) return -1;
    struct Cfg *c = &m->cfg;
    c->vocab = 30534; c->hidden = 256; c->layers = 8; c->heads = 4;
    c->intermediate = 1024; c->max_seq = 256; c->eps = 1e-12f;
    c->ntact = 14; c->ntech = 12;
    m->emb_words = take((size_t)c->vocab * c->hidden);
    m->emb_pos   = take((size_t)512 * c->hidden);
    m->emb_ttype = take((size_t)2 * c->hidden);
    m->emb_ln_w  = take((size_t)c->hidden);
    m->emb_ln_b  = take((size_t)c->hidden);
    for (int l = 0; l < c->layers; ++l) {
        m->q_w[l] = take((size_t)c->hidden*c->hidden); m->q_b[l] = take((size_t)c->hidden);
        m->k_w[l] = take((size_t)c->hidden*c->hidden); m->k_b[l] = take((size_t)c->hidden);
        m->v_w[l] = take((size_t)c->hidden*c->hidden); m->v_b[l] = take((size_t)c->hidden);
        m->attn_o_w[l] = take((size_t)c->hidden*c->hidden); m->attn_o_b[l] = take((size_t)c->hidden);
        m->attn_ln_w[l] = take((size_t)c->hidden); m->attn_ln_b[l] = take((size_t)c->hidden);
        m->ffn1_w[l] = take((size_t)c->intermediate*c->hidden); m->ffn1_b[l] = take((size_t)c->intermediate);
        m->ffn2_w[l] = take((size_t)c->hidden*c->intermediate); m->ffn2_b[l] = take((size_t)c->hidden);
        m->ffn_ln_w[l] = take((size_t)c->hidden); m->ffn_ln_b[l] = take((size_t)c->hidden);
    }
    m->pooler_w = take((size_t)c->hidden*c->hidden); m->pooler_b = take((size_t)c->hidden);
    m->mal_w = take((size_t)1*c->hidden); m->mal_b = take(1);
    m->tact_w = take((size_t)14*c->hidden); m->tact_b = take(14);
    m->tech_w = take((size_t)12*c->hidden); m->tech_b = take(12);
    return 0;
}

static inline float gelu_erf(float x) {
    /* PyTorch F.gelu default (approximate='none') */
    return 0.5f * x * (1.0f + erff(x * 0.70710678118654752440f));
}

static void mm(const float *in, const float *W, const float *bias, int S, int K, int N, float *out) {
    for (int s = 0; s < S; ++s) {
        const float *ins = in + (size_t)s * K;
        float *outs = out + (size_t)s * N;
        for (int n = 0; n < N; ++n) {
            double sum = bias ? (double)bias[n] : 0.0;
            const float *wrow = W + (size_t)n * K;
            for (int k = 0; k < K; ++k) sum += (double)ins[k] * (double)wrow[k];
            outs[n] = (float)sum;
        }
    }
}

static void ln_bias(float *h, const float *w, const float *b, int S, int H, float eps) {
    for (int s = 0; s < S; ++s) {
        float *row = h + (size_t)s * H;
        double mean = 0.0, var = 0.0;
        for (int i = 0; i < H; ++i) mean += row[i];
        mean /= H;
        for (int i = 0; i < H; ++i) { double d = row[i] - mean; var += d * d; }
        var /= H;
        float inv = (float)(1.0 / sqrt(var + (double)eps));
        for (int i = 0; i < H; ++i) row[i] = (float)((row[i] - mean) * inv) * w[i] + b[i];
    }
}

static void add_res(float *out, const float *in, int n) {
    for (int i = 0; i < n; ++i) out[i] += in[i];
}

static int parse_line(char *buf, int *tokens, int max) {
    char *p = buf;
    long nn = strtol(p, &p, 10);
    if (nn <= 1 || nn > max) return 0;
    for (int i = 0; i < nn; ++i) tokens[i] = (int)strtol(p, &p, 10);
    return (int)nn;
}

int main(int argc, char **argv) {
    if (argc < 2) { fprintf(stderr, "usage: %s model_f32.bin [inputs.tokenids.txt]\n", argv[0]); return 1; }
    struct Model M;
    if (load_model(argv[1], &M) != 0) return 1;
    struct Cfg *c = &M.cfg;
    FILE *in = stdin;
    if (argc > 2) { in = fopen(argv[2], "r"); if (!in) { perror("input"); return 1; } }

    const int S = c->max_seq, H = c->hidden, L = c->layers, NH = c->heads, HD = H / NH, MID = c->intermediate;

    int *tokens = (int *)malloc(sizeof(int) * S);
    float *hidden = (float *)malloc(sizeof(float) * S * H * 2);
    float *buf2 = hidden + (size_t)S * H;
    float *tmp = (float *)malloc(sizeof(float) * S * H);
    float *q = (float *)malloc(sizeof(float) * S * H);
    float *k = (float *)malloc(sizeof(float) * S * H);
    float *v = (float *)malloc(sizeof(float) * S * H);
    float *scores = (float *)malloc(sizeof(float) * NH * S * S);
    float *ctx = (float *)malloc(sizeof(float) * NH * S * HD);
    float *inter = (float *)malloc(sizeof(float) * S * MID);
    if (!tokens || !hidden || !tmp || !q || !k || !v || !scores || !ctx || !inter) { fprintf(stderr,"alloc failed\n"); return 1; }

    struct timespec t0, t1;
    clock_gettime(CLOCK_MONOTONIC, &t0);

    int line = 0;
    char *buf = NULL; size_t cap = 0;
    while (getline(&buf, &cap, in) != -1) {
        int nn = parse_line(buf, tokens, S);
        if (!nn) continue;
        ++line;
#ifdef DEBUG
        fprintf(stderr, "line %d nn=%d first=%d\n", line, nn, tokens[0]);
#endif

        for (int i = 0; i < nn; ++i) {
            const float *row = M.emb_words + (size_t)tokens[i] * H;
            memcpy(hidden + (size_t)i * H, row, sizeof(float) * H);
        }
        for (int i = 0; i < nn; ++i) {
            int pidx = i < 512 ? i : 511;
            const float *prow = M.emb_pos + (size_t)pidx * H;
            const float *trow = M.emb_ttype;
            float *out = hidden + (size_t)i * H;
            for (int j = 0; j < H; ++j) out[j] = out[j] + prow[j] + trow[j];
        }
        ln_bias(hidden, M.emb_ln_w, M.emb_ln_b, nn, H, c->eps);
#ifdef DEBUG
        fprintf(stderr, "line%d embed_h0 %.6g %.6g %.6g\n", line, hidden[0],hidden[1],hidden[2]);
#endif
#ifdef DEBUG
        if (line==1) { fprintf(stderr,"emb[0..3] %.6g %.6g %.6g\n", hidden[0],hidden[1],hidden[2]); }
#endif

        for (int l = 0; l < L; ++l) {
            mm(hidden, M.q_w[l], M.q_b[l], nn, H, H, q);
            mm(hidden, M.k_w[l], M.k_b[l], nn, H, H, k);
            mm(hidden, M.v_w[l], M.v_b[l], nn, H, H, v);

#ifdef DEBUG
            if(line==1&&l==0) fprintf(stderr,"C qkvv %.6g %.6g %.6g  | %.6g %.6g %.6g  | %.6g %.6g %.6g\n", q[0],q[1],q[2], k[0],k[1],k[2], v[0],v[1],v[2]);
#endif
            for (int h = 0; h < NH; ++h) {
                float *sc = scores + (size_t)h * nn * nn;
                for (int i = 0; i < nn; ++i) {
                    const float *qi = q + (size_t)i * H + h * HD;
                    float *si = sc + (size_t)i * nn;
                    double mx = -1e30;
                    for (int j = 0; j < nn; ++j) {
                        const float *kj = k + (size_t)j * H + h * HD;
                        double d = 0.0;
                        for (int d0 = 0; d0 < HD; ++d0) d += (double)qi[d0] * (double)kj[d0];
                        d *= 0.125;
                        si[j] = (float)d;
                        if (d > mx) mx = d;
                    }
                    double sm = 0.0;
                    for (int j = 0; j < nn; ++j) { double e = exp((double)si[j] - (double)mx); si[j] = (float)e; sm += e; }
                    float inv = (float)(1.0 / sm);
                    for (int j = 0; j < nn; ++j) si[j] *= inv;
#ifdef DEBUG
                    if(line==1&&l==0&&h==0&&i==0){ fprintf(stderr,"C scores-row0: "); for(int j=0;j<nn;j++)fprintf(stderr,"%.9g ",si[j]); fprintf(stderr,"\n"); }
#endif
                }
                for (int i = 0; i < nn; ++i) {
                    float *ci = ctx + (size_t)h * nn * HD + (size_t)i * HD;
                    for (int d0 = 0; d0 < HD; ++d0) ci[d0] = 0.0f;
                    for (int j = 0; j < nn; ++j) {
                        double w = (double)scores[(size_t)h * nn * nn + (size_t)i * nn + j];
                        const float *vj = v + (size_t)j * H + h * HD;
                        for (int d0 = 0; d0 < HD; ++d0) ci[d0] += (float)(w * (double)vj[d0]);
                    }
                }
                for (int i = 0; i < nn; ++i)
                    memcpy(tmp + (size_t)i * H + h * HD,
                           ctx + (size_t)h * nn * HD + (size_t)i * HD,
                           sizeof(float) * HD);
#ifdef DEBUG
                if(line==1&&l==0&&h==0){ fprintf(stderr,"C ctx-row0: "); for(int d0=0;d0<6;d0++)fprintf(stderr,"%.6g ", ctx[d0]); fprintf(stderr,"\n"); }
#endif
            }


#ifdef DEBUG
            if(line==1&&l==0){ fprintf(stderr,"C tmp-row0 h0 %.6g %.6g | h1 %.6g %.6g | h2 %.6g %.6g | h3 %.6g %.6g\n", tmp[0],tmp[1], tmp[64],tmp[65], tmp[128],tmp[129], tmp[192],tmp[193]); }
#endif
            mm(tmp, M.attn_o_w[l], M.attn_o_b[l], nn, H, H, buf2);
#ifdef DEBUG
            if(line==1&&l==0) fprintf(stderr,"C off_attno=%ld off_attnln=%ld off_ffn1=%ld\n",(long)((const unsigned char*)M.attn_o_w[l]-gblob)/4, (long)((const unsigned char*)M.attn_ln_w[l]-gblob)/4, (long)((const unsigned char*)M.ffn1_w[l]-gblob)/4);
#endif
#ifdef DEBUG
            if (line==1 && l==0) fprintf(stderr, "attnout0 %.6g %.6g %.6g\n", buf2[0], buf2[1], buf2[2]);
#endif
            add_res(buf2, hidden, nn * H);
            ln_bias(buf2, M.attn_ln_w[l], M.attn_ln_b[l], nn, H, c->eps);
#ifdef DEBUG
            if (line==1 && l==0) fprintf(stderr, "attnLN0 %.6g %.6g %.6g\n", buf2[0], buf2[1], buf2[2]);
#endif

            mm(buf2, M.ffn1_w[l], M.ffn1_b[l], nn, H, MID, inter);
            for (int i = 0; i < nn * MID; ++i) inter[i] = gelu_erf(inter[i]);
            mm(inter, M.ffn2_w[l], M.ffn2_b[l], nn, MID, H, hidden);
            add_res(hidden, buf2, nn * H);
#ifdef DEBUG
            if (line>=1) fprintf(stderr,"line%d layer %d h0 %.6g\n", line, l, hidden[0]);
#endif
        }

        const float *cls_vec = hidden; /* first token, same as PyTorch forward */
        float mal_logit = M.mal_b[0];
        for (int k = 0; k < H; ++k) mal_logit += cls_vec[k] * M.mal_w[k];
        float tact_logit[14], tech_logit[12];
        for (int i = 0; i < 14; ++i) {
            float s = M.tact_b[i];
            for (int k = 0; k < H; ++k) s += cls_vec[k] * M.tact_w[(size_t)i * H + k];
            tact_logit[i] = s;
        }
        for (int i = 0; i < 12; ++i) {
            float s = M.tech_b[i];
            for (int k = 0; k < H; ++k) s += cls_vec[k] * M.tech_w[(size_t)i * H + k];
            tech_logit[i] = s;
        }

        printf("LINE %d mal_logit %.9g", line, mal_logit);
        printf(" tact_logit");
        for (int i = 0; i < 14; ++i) printf(" %.9g", tact_logit[i]);
        printf(" tech_logit");
        for (int i = 0; i < 12; ++i) printf(" %.9g", tech_logit[i]);
        printf(" mal_prob %.9g", 1.0f / (1.0f + expf(-mal_logit)));
        printf("\n");
        fflush(stdout);
    }
    clock_gettime(CLOCK_MONOTONIC, &t1);
    double secs = (double)(t1.tv_sec - t0.tv_sec) + (double)(t1.tv_nsec - t0.tv_nsec) / 1e9;

    long vmhwm_kb = -1;
    FILE *st = fopen("/proc/self/status", "r");
    if (st) {
        char linebuf[256];
        while (fgets(linebuf, sizeof(linebuf), st)) {
            if (sscanf(linebuf, "VmHWM: %ld kB", &vmhwm_kb) == 1) break;
        }
        fclose(st);
    }
    fprintf(stderr, "lines=%d inferred_seconds=%.6f VmHWM_KiB=%ld VmHWM_MiB=%.1f weights_MiB=%.1f\n",
            line, secs, vmhwm_kb, vmhwm_kb / 1024.0, gblob_len / 1048576.0);

    if (argc > 2) fclose(in);
    free(buf);
    free(tokens); free(hidden); free(tmp); free(q); free(k); free(v); free(scores); free(ctx); free(inter);
    free(gblob);
    return 0;
}
