/* cmd-session-risk C BERT inference prototype (token-id input).
 * build: cc -O3 -march=native tools/bert_infer_c.c -lm -o tools/bert_infer_c
 */
#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "cmdrisk_calibration.h"
#include "cmdrisk_intent.h"
#include <time.h>
#include <unistd.h>
#include <errno.h>
#include <ctype.h>

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

#define MAX_VOCAB 40000
#define MAXW 200
typedef struct { char *w; int id; } VEntry;
static char **vocab_tokens;
static VEntry *vocab_entries;
static int vcmp(const void*, const void*);
static int vocab_count = 0;
#define SID_SESSION 30522
#define SID_EOS     30523
#define SID_CMD_START 30524
#define SID_CMD_END   30525
#define SID_TRUNC     30526
#define SID_DT_0    30527
#define SID_DT_1    30528
#define SID_DT_10   30529
#define SID_DT_1M   30530
#define SID_DT_10M  30531
#define SID_DT_GT   30532
#define SID_DT_UNK  30533

static const char *TACT_LABELS[14] = {"TA0043","TA0042","TA0001","TA0002","TA0003","TA0004","TA0005","TA0006","TA0007","TA0008","TA0009","TA0011","TA0010","TA0040"};
static const char *TECH_LABELS[12] = {"T1005","T1036","T1053","T1057","T1059","T1070","T1082","T1083","T1087","T1098","T1105","T1222"};

static int load_vocab(const char*path) {
    FILE*f=fopen(path,"r"); if(!f){perror("vocab");return -1;}
    vocab_tokens=(char**)malloc(sizeof(char*)*MAX_VOCAB);
    vocab_entries=(VEntry*)malloc(sizeof(VEntry)*MAX_VOCAB);
    char line[MAXW];
    while(fgets(line,sizeof line,f)&&vocab_count<MAX_VOCAB){
        size_t n=strlen(line); while(n&&isspace((unsigned char)line[n-1]))line[--n]=0;
        size_t w=n+2; char* slot=(char*)malloc(w); memcpy(slot,line,n+1);
        vocab_tokens[vocab_count]=slot; vocab_entries[vocab_count].w=slot; vocab_entries[vocab_count].id=vocab_count; vocab_count++;
    }
    fclose(f);
    qsort(vocab_entries, vocab_count, sizeof(VEntry), vcmp);
    return 0;
}
static int vcmp(const void*a,const void*b){
    const VEntry*x=(const VEntry*)a; const VEntry*y=(const VEntry*)b; return strcmp(x->w, y->w);
}
static int voc_id(const char*word){
    char full[MAXW];
    snprintf(full,sizeof full,"%s",word);
    VEntry key;
    key.w = full; key.id = -1;
    VEntry *hit = bsearch(&key, vocab_entries, vocab_count, sizeof(VEntry), vcmp);
    return hit ? hit->id : -1;
}
static int is_punct_ascii(char c){
    const char* p="!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~";
    return strchr(p,c)!=NULL;
}
static int emit_wordpiece(const char *buf, int bn, int *out, int n, int cap){
    int id = voc_id(buf);
    if(id>=0 && n<cap){ out[n++] = id; return n; }
    int off=0, fail=0;
    while(off<bn && n<cap){
        int best=-1, bl=0;
        for(int l=bn-off; l>0; --l){
            char probe[MAXW];
            if(off==0) snprintf(probe,sizeof probe,"%.*s", l, buf+off);
            else snprintf(probe,sizeof probe,"##%.*s", l, buf+off);
            int wpid = voc_id(probe);
            if(wpid>=0){ best=wpid; bl=l; break; }
        }
        if(best<0){ fail=1; break; }
        out[n++]=best; off+=bl;
    }
    if(fail && n<cap) out[n++] = voc_id("[UNK]");
    return n;
}

static int tok_cmd(const char* text, int* out, int cap){
    int n=0;
    const char*p=text;
    char buf[MAXW]; int bn=0;
    while(*p){
        char c=*p;
        if((unsigned char)c>=0x80){ /* non-ascii: single-char token or [UNK] */
            if(bn){ buf[bn]=0; n = emit_wordpiece(buf, bn, out, n, cap); bn=0; }
            char sc[2]={c,0}; int id=voc_id(sc);
            if(id>=0&&n<cap)out[n++]=id; else if(n<cap)out[n++]=voc_id("[UNK]");
            p++; continue;
        }
        if(isspace((unsigned char)c)||is_punct_ascii(c)){
            if(bn){ buf[bn]=0; n = emit_wordpiece(buf, bn, out, n, cap); bn=0; }
            if(is_punct_ascii(c)&&c!=' '){ char p1[2]={c,0}; int id=voc_id(p1); if(id>=0&&n<cap)out[n++]=id; }
            p++; continue;
        }
        if(bn<MAXW-1){ if(c>='A'&&c<='Z')c=c-'A'+'a'; buf[bn++]=c; } p++;
    }
    if(bn){ buf[bn]=0; n = emit_wordpiece(buf, bn, out, n, cap); }
    return n;
}

#define BSEP '\x1f'
static int encode_prefix(const char*line, int64_t* ids, int cap){
    const char*p=line; int n=0; int64_t ids0[4096];
    ids0[n++]=SID_SESSION;
    while(*p){
        /* command until separator or end */
        const char*start=p;
        while(*p && *p!=BSEP) p++;
        int clen=(int)(p-start);
        char cmd[MAXW]; int cn=0;
        const char*q=start;
        while(q<start+clen && *q!='\t'){ if(cn<MAXW-1)cmd[cn++]=*q; q++; }
        cmd[cn]=0;
        double dt=-1;
        if(*q=='\t'){ char num[64]; int dn=0; q++;
            while(q<start+clen && dn<63){ num[dn++]=*q++; }
            num[dn]=0; dt=atof(num);
        }
        int tok[128];
        int tn=tok_cmd(cmd,tok,128);
        if(tn){ if(tn>128)tn=128; }
        if(n<4096) ids0[n++]=SID_CMD_START;
        for(int i=0;i<tn&&n<4096;i++) ids0[n++]=tok[i];
        if(n<4096) ids0[n++]=SID_CMD_END;
        int dtid;
        if(dt<0) dtid=SID_DT_UNK; else if(dt==0)dtid=SID_DT_0;
        else if(dt<=1)dtid=SID_DT_1; else if(dt<=10)dtid=SID_DT_10;
        else if(dt<=60)dtid=SID_DT_1M; else if(dt<=600)dtid=SID_DT_10M;
        else dtid=SID_DT_GT;
        if(n<4096) ids0[n++]=dtid;
        if(*p) p++;
    }
    if(n<4096) ids0[n++]=SID_EOS;
    if(n>cap){
        for(int i=0;i<cap-1;i++) ids[i]=ids0[i];
        ids[cap-1]=SID_TRUNC;
        return cap;
    }
    for(int i=0;i<n;i++) ids[i]=ids0[i];
    return n;
}

extern const unsigned char _binary_ci_assets_model_f32_bin_start[];
extern const unsigned char _binary_ci_assets_model_f32_bin_end[];
extern const unsigned char _binary_ci_assets_vocab_txt_start[];
extern const unsigned char _binary_ci_assets_vocab_txt_end[];

extern const unsigned char _binary_ci_assets_calibration_json_start[];
extern const unsigned char _binary_ci_assets_calibration_json_end[];
extern const unsigned char _binary_ci_assets_model_config_json_start[];
extern const unsigned char _binary_ci_assets_model_config_json_end[];

/* micro JSON number reader: find "key" then the next number.  Returns
   1 on success.  Deliberately tiny: our embedded files are simple json. */
static int json_num(const unsigned char *s, const unsigned char *e,
                    const char *key, double *out){
    size_t klen = strlen(key);
    for (const unsigned char *p = s; p + klen < e; p++){
        if (*p != '"') continue;
        if (memcmp(p+1, key, klen) != 0 || p[1+klen] != '"') continue;
        const unsigned char *q = p + klen + 2;
        while (q < e && *q != ':') q++;
        q++;
        while (q < e && (*q==' '||*q=='\t'||*q=='\n')) q++;
        char num[64]; size_t n=0;
        while (q < e && n+1 < sizeof(num) &&
               (isdigit((unsigned char)*q) || *q=='-' || *q=='.' || *q=='e' || *q=='E' || *q=='+')){
            num[n++] = (char)*q; q++;
        }
        if (!n) return 0;
        num[n] = 0;
        *out = strtod(num, NULL);
        return 1;
    }
    return 0;
}

static int load_vocab_mem(const unsigned char *data, size_t len) {
    (void)len;
    FILE *f = fmemopen((void*)data, len, "rb");
    if (!f) return -1;
    vocab_tokens=(char**)malloc(sizeof(char*)*MAX_VOCAB);
    vocab_entries=(VEntry*)malloc(sizeof(VEntry)*MAX_VOCAB);
    char line[MAXW];
    while (fgets(line, sizeof line, f) && vocab_count < MAX_VOCAB) {
        size_t n=strlen(line); while(n&&isspace((unsigned char)line[n-1])) line[--n]=0;
        char* slot=(char*)malloc(n+2); memcpy(slot,line,n+1); slot[n]=0;
        vocab_tokens[vocab_count]=slot;
        vocab_entries[vocab_count].w=slot; vocab_entries[vocab_count].id=vocab_count;
        vocab_count++;
    }
    fclose(f);
    qsort(vocab_entries, vocab_count, sizeof(VEntry), vcmp);
    return 0;
}

static int load_model_mem(const unsigned char *data, size_t len, struct Model *m) {
    gblob = (unsigned char*)data; gblob_len = len; goff = 0;
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

int main(int argc, char **argv) {
    if (argc < 2) { fprintf(stderr, "usage: %s input_text.txt [threshold]\n", argv[0]); return 1; }
    const char* input_path = argv[1];
    float TH = 0.00011f; /* fallback only; normally overridden by the embedded calibration */
    double cal_th = 0.0, cal_temp = 0.0;
    if (json_num(_binary_ci_assets_calibration_json_start, _binary_ci_assets_calibration_json_end,
                 "malicious_threshold", &cal_th) && cal_th > 0.0)
        TH = (float)cal_th;
    if (argc > 2) TH = (float)atof(argv[2]);   /* explicit CLI arg always wins */
    double MALT = 0.0;
    if (json_num(_binary_ci_assets_calibration_json_start, _binary_ci_assets_calibration_json_end,
                 "malicious_temperature", &MALT) && MALT > 0.0) {
        /* runtime temperature from the embedded calibration */
    } else {
        MALT = (double)MAL_TEMP;               /* compiled fallback */
    }

    /* compile-time structure constants must match the embedded model config,
       otherwise the weight blob layout is wrong; fail loudly instead of
       silently mis-reading the tensor table. */
    {
        static const struct { const char *key; double want; } expect[] = {
            {"vocab_size", 30534}, {"hidden_size", 256}, {"num_layers", 8},
            {"num_heads", 4}, {"intermediate_size", 1024}, {"max_seq", 256},
        };
        for (size_t i = 0; i < sizeof(expect)/sizeof(expect[0]); i++){
            double got = -1.0;
            if (!json_num(_binary_ci_assets_model_config_json_start, _binary_ci_assets_model_config_json_end,
                          expect[i].key, &got) || got != expect[i].want){
                fprintf(stderr, "embedded model_config mismatch: %s=%g (code expects %g)\n",
                        expect[i].key, got, expect[i].want);
                return 1;
            }
        }
    }
    fprintf(stderr, "operating-point: threshold=%.9g temperature=%.9g\n", TH, MALT);

    load_vocab_mem(_binary_ci_assets_vocab_txt_start, _binary_ci_assets_vocab_txt_end - _binary_ci_assets_vocab_txt_start);
    struct Model M;
    if (load_model_mem(_binary_ci_assets_model_f32_bin_start, _binary_ci_assets_model_f32_bin_end - _binary_ci_assets_model_f32_bin_start, &M) != 0) return 1;
    struct Cfg *c = &M.cfg;
    FILE *in = fopen(input_path, "r");
    if (!in) { perror("input"); return 1; }

    const int S = c->max_seq, H = c->hidden, L = c->layers, NH = c->heads, HD = H / NH, MID = c->intermediate;
    int64_t *tokens = (int64_t *)malloc(sizeof(int64_t) * S);
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

    struct timespec t0, t1; clock_gettime(CLOCK_MONOTONIC, &t0);
    char *lb = NULL; size_t lc = 0; int line = 0;
    while (getline(&lb, &lc, in) != -1) {
        size_t len = strlen(lb); while (len && (lb[len-1]=='\n' || lb[len-1]=='\r')) lb[--len]=0;
        if (!len) continue;
        int nn = encode_prefix(lb, tokens, S);
        if (nn < 2) continue;
        ++line;
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
        for (int l = 0; l < L; ++l) {
            mm(hidden, M.q_w[l], M.q_b[l], nn, H, H, q);
            mm(hidden, M.k_w[l], M.k_b[l], nn, H, H, k);
            mm(hidden, M.v_w[l], M.v_b[l], nn, H, H, v);
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
                    for (int j = 0; j < nn; ++j) { double e = exp((double)si[j] - mx); si[j] = (float)e; sm += e; }
                    float inv = (float)(1.0 / sm);
                    for (int j = 0; j < nn; ++j) si[j] *= inv;
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
            }
            mm(tmp, M.attn_o_w[l], M.attn_o_b[l], nn, H, H, buf2);
            add_res(buf2, hidden, nn * H);
            ln_bias(buf2, M.attn_ln_w[l], M.attn_ln_b[l], nn, H, c->eps);
            mm(buf2, M.ffn1_w[l], M.ffn1_b[l], nn, H, MID, inter);
            for (int i = 0; i < nn * MID; ++i) inter[i] = gelu_erf(inter[i]);
            mm(inter, M.ffn2_w[l], M.ffn2_b[l], nn, MID, H, hidden);
            add_res(hidden, buf2, nn * H);
            ln_bias(hidden, M.ffn_ln_w[l], M.ffn_ln_b[l], nn, H, c->eps);
        }
        const float *cls_vec = hidden;
        float mal_logit = M.mal_b[0];
        for (int kdx = 0; kdx < H; ++kdx) mal_logit += cls_vec[kdx] * M.mal_w[kdx];
        float prob = 1.0f / (1.0f + expf(-(mal_logit / (float)MALT)));
        /* argmax interest labels (multi-label heads; thresholds not applied here) */
        float best_t = -1e30f; int bt = -1;
        for (int k = 0; k < c->ntact; k++){
            float s = M.tact_b[k];
            const float *row = M.tact_w + (size_t)k * H;
            for (int d = 0; d < H; ++d) s += cls_vec[d] * row[d];
            if (s > best_t){ best_t = s; bt = k; }
        }
        float best_e = -1e30f; int be = -1;
        for (int k = 0; k < c->ntech; k++){
            float s = M.tech_b[k];
            const float *row = M.tech_w + (size_t)k * H;
            for (int d = 0; d < H; ++d) s += cls_vec[d] * row[d];
            if (s > best_e){ best_e = s; be = k; }
        }
        char cat[256]; cat[0] = 0;
        int rule_hit = prefix_intent_alert_c(lb, cat, sizeof cat);
        int decis = (prob >= TH) || rule_hit;
        printf("%d\t%.9g\t%.9g\t%d\t%d\t%s\t%s\t%s\n",
               line, mal_logit, prob, decis, rule_hit,
               cat[0] ? cat : "-",
               (decis && bt >= 0) ? TACT_LABELS[bt] : "-",
               (decis && be >= 0) ? TECH_LABELS[be] : "-");
        if (rule_hit) fprintf(stderr, "line %d intent-layer force-alert [%s]\n", line, cat);
    }
    clock_gettime(CLOCK_MONOTONIC, &t1);
    double secs = (double)(t1.tv_sec - t0.tv_sec) + (double)(t1.tv_nsec - t0.tv_nsec) / 1e9;
    long vmhwm = -1;
    FILE *st = fopen("/proc/self/status", "r");
    if (st) { char lb2[256]; while (fgets(lb2, sizeof(lb2), st)) { if (sscanf(lb2, "VmHWM: %ld kB", &vmhwm) == 1) break; } fclose(st); }
    fprintf(stderr, "lines=%d seconds=%.3f VmHWM_MiB=%.1f\n", line, secs, vmhwm/1024.0);
    fclose(in); free(lb);
    return 0;
}
