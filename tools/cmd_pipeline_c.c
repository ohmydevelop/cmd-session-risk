/* C end-to-end runtime: raw command text prefix -> ONNX Runtime INT8 session -> decision.
 * build:
 *   cc -O2 -I tools/third_party tools/cmd_pipeline_c.c \
 *      "$PWD/.venv/lib/python3.11/site-packages/onnxruntime/capi/libonnxruntime.so.1.30.0" \
 *      -Wl,-rpath,"$PWD/tools/lib" -lm -o tools/cmd_pipeline_c
 * input line format: commands separated by \x1f; each command "<text>\t<dt>" (dt optional).
 */
#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <ctype.h>
#include <time.h>
#include <unistd.h>
#include "onnxruntime_c_api.h"

#define CHECK_STATUS(expr) do { OrtStatus* s_ = (expr); if (s_) { fprintf(stderr, "ORT err %d: %s\n", ort->GetErrorCode(s_), ort->GetErrorMessage(s_)); ort->ReleaseStatus(s_); return 1; } } while(0)

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
static int tok_cmd(const char* text, int* out, int cap){
    int n=0;
    const char*p=text;
    char buf[MAXW]; int bn=0;
    char word[MAXW];
    while(*p){
        char c=*p;
        if((unsigned char)c>=0x80){ /* non-ascii: emit as single-charunk if possible */
            if(bn){ buf[bn]=0; int id=voc_id(buf); if(id>=0&&n<cap)out[n++]=id; bn=0; }
            char sc[2]={c,0}; int id=voc_id(sc); if(id>=0&&n<cap)out[n++]=id; else if(n<cap)out[n++]=voc_id("[UNK]");
            p++; continue;
        }
        if(isspace((unsigned char)c)||is_punct_ascii(c)){
            if(bn){ buf[bn]=0; int id=voc_id(buf);
                if(id>=0&&n<cap)out[n++]=id; else { /* subword */
                    int off=0; int fail=0;
                    while(off<bn && n<cap){
                        int best=-1, bl=0;
                        for(int l=bn-off; l>0; --l){
                            buf[bn]=0; char probe[MAXW];
                            snprintf(probe,sizeof probe,"%s%s", off?"##":"", buf+off);
                            // emit with ## prefix each attempt
                            if(l==bn-off){ snprintf(probe,sizeof probe,"%s%.*s", off?"##":"", l, buf+off); } else { continue; }
                            int id=voc_id(probe);
                            if(id>=0){ best=id; bl=l; break; }
                        }
                        if(best<0){ fail=1; break; }
                        out[n++]=best; off+=bl;
                    }
                    if(fail&&n<cap)out[n++]=voc_id("[UNK]");
                }
                bn=0;
            }
            if(is_punct_ascii(c)&&c!=' '){ char p1[2]={c,0}; int id=voc_id(p1); if(id>=0&&n<cap)out[n++]=id; }
            p++; continue;
        }
        if(bn<MAXW-1){ if(c>='A'&&c<='Z')c=c-'A'+'a'; buf[bn++]=c; } p++;
    }
    if(bn){ buf[bn]=0; int id=voc_id(buf); if(id>=0&&n<cap)out[n++]=id; }
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
        ids0[n++]=SID_CMD_START;
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
    ids0[n++]=SID_EOS;
    if(n>cap){
        for(int i=0;i<cap-1;i++) ids[i]=ids0[i];
        ids[cap-1]=SID_TRUNC;
        return cap;
    }
    for(int i=0;i<n;i++) ids[i]=ids0[i];
    return n;
}
int main(int argc,char**argv){
    if(argc<4){ fprintf(stderr,"usage: %s model.int8.onnx vocab.txt input.txt threshold\n",argv[0]); return 1; }
    const OrtApi* ort=OrtGetApiBase()->GetApi(ORT_API_VERSION);
    load_vocab(argv[2]);
    const char*MOD=argv[1]; const char*INP=argv[3];
    float TH=(float)atof(argv[4]);
    OrtEnv*env=NULL;OrtSessionOptions*opts=NULL;OrtSession*sess=NULL;OrtMemoryInfo*mem=NULL;
    CHECK_STATUS(ort->CreateEnv(ORT_LOGGING_LEVEL_WARNING,"crisp", &env));
    CHECK_STATUS(ort->CreateSessionOptions(&opts));
    CHECK_STATUS(ort->SetSessionGraphOptimizationLevel(opts,ORT_ENABLE_ALL));
    CHECK_STATUS(ort->SetIntraOpNumThreads(opts,2));
    CHECK_STATUS(ort->DisableCpuMemArena(opts));
    CHECK_STATUS(ort->CreateSession(env,MOD,opts,&sess));
    CHECK_STATUS(ort->CreateCpuMemoryInfo(OrtArenaAllocator,OrtMemTypeDefault,&mem));
    FILE*f=fopen(INP,"r"); if(!f){perror("input");return 1;}
    const char*inames[]={"input_ids","attention_mask"};
    const char*onames[]={"malicious_logits","tactic_logits","technique_logits"};
    OrtValue*invals[2]={0}; OrtValue*outvals[3]={0};
    char *lb=NULL; size_t lc=0; int line=0;
    struct timespec t0,t1; clock_gettime(CLOCK_MONOTONIC,&t0);
    while(getline(&lb,&lc,f)!=-1){
        size_t len=strlen(lb); while(len&&(lb[len-1]=='\n'||lb[len-1]=='\r'))lb[--len]=0;
        if(!len)continue;
        static int64_t ids[256]; static int64_t mask[256];
        int nn=encode_prefix(lb,ids,256);
        if(nn<2)continue;
        for(int i=0;i<nn;i++)mask[i]=1;
        int64_t shape[2]={1,nn};

        CHECK_STATUS(ort->CreateTensorWithDataAsOrtValue(mem,ids,sizeof(int64_t)*nn,shape,2,ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64,&invals[0]));
        CHECK_STATUS(ort->CreateTensorWithDataAsOrtValue(mem,mask,sizeof(int64_t)*nn,shape,2,ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64,&invals[1]));
        CHECK_STATUS(ort->Run(sess,NULL,inames,(const OrtValue* const*)invals,2,onames,3,outvals));
        float*mal=NULL; float*tact=NULL; float*tech=NULL;
        CHECK_STATUS(ort->GetTensorMutableData(outvals[0],(void**)&mal));
        CHECK_STATUS(ort->GetTensorMutableData(outvals[1],(void**)&tact));
        CHECK_STATUS(ort->GetTensorMutableData(outvals[2],(void**)&tech));
        line++;
        float prob=1.0f/(1.0f+expf(-mal[0]));
        printf("%d\t%.9g\t%.9g\t%d\n",line,mal[0],prob,(int)(prob>=TH));
        fflush(stdout);
        ort->ReleaseValue(invals[0]); invals[0]=0; ort->ReleaseValue(invals[1]); invals[1]=0;
        ort->ReleaseValue(outvals[0]); outvals[0]=0; ort->ReleaseValue(outvals[1]); outvals[1]=0; ort->ReleaseValue(outvals[2]); outvals[2]=0;
    }
    clock_gettime(CLOCK_MONOTONIC,&t1);
    double secs=(double)(t1.tv_sec-t0.tv_sec)+(double)(t1.tv_nsec-t0.tv_nsec)/1e9;
    long vh=-1; FILE*st=fopen("/proc/self/status","r"); if(st){char b[256];while(fgets(b,sizeof b,st)){if(sscanf(b,"VmHWM: %ld kB",&vh)==1)break;}fclose(st);}
    fprintf(stderr,"lines=%d seconds=%.3f VmHWM_MiB=%.1f\n",line,secs,vh/1024.0);
    fclose(f); free(lb);
    return 0;
}
