/* C-only session1p deployment runtime using the ONNX Runtime C API.
 * build:
 *   cc -O2 -I tools/third_party -L .venv/lib/python3.11/site-packages/onnxruntime/capi \
 *      tools/onnx_int8_c.c -lonnxruntime -Wl,-rpath,$PWD/.venv/lib/python3.11/site-packages/onnxruntime/capi -lm -o tools/onnx_int8_c
 */
#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>
#include <math.h>
#include "onnxruntime_c_api.h"

#define CHECK_STATUS(expr, ort) do { OrtStatus* st_ = (expr); if (st_) { fprintf(stderr, "ORT %d: %s\n", ort->GetErrorCode(st_), ort->GetErrorMessage(st_)); ort->ReleaseStatus(st_); return 1; } } while(0)
#define CHK(expr) do { OrtStatus* st = (expr); if (st) { fprintf(stderr, "ORT %d: %s\n", ortGetErrorCode(st), ortGetErrorMessage(st)); ortReleaseStatus(st); return 1; } } while(0)

static const char* MOD;
static const char* INP;
static float TH = 0.0001179935351014905f;

int main(int argc, char** argv) {
    if (argc < 2) { fprintf(stderr, "usage: %s model.int8.onnx [inputs.tokenids.txt] [threshold]\n", argv[0]); return 1; }
    MOD = argv[1];
    INP = argc > 2 ? argv[2] : NULL;
    if (argc > 3) TH = (float)atof(argv[3]);

    const OrtApi* ort = OrtGetApiBase()->GetApi(ORT_API_VERSION);
    OrtEnv* env = NULL;
    OrtSessionOptions* opts = NULL;
    OrtSession* sess = NULL;
    OrtMemoryInfo* mem = NULL;
    CHECK_STATUS(ort->CreateEnv(ORT_LOGGING_LEVEL_WARNING, "crisp", &env), ort);
    CHECK_STATUS(ort->CreateSessionOptions(&opts), ort);
    CHECK_STATUS(ort->SetSessionGraphOptimizationLevel(opts, ORT_ENABLE_ALL), ort);
    CHECK_STATUS(ort->SetIntraOpNumThreads(opts, 2), ort);
    CHECK_STATUS(ort->DisableCpuMemArena(opts), ort);
    CHECK_STATUS(ort->CreateSession(env, MOD, opts, &sess), ort);
    CHECK_STATUS(ort->CreateCpuMemoryInfo(OrtArenaAllocator, OrtMemTypeDefault, &mem), ort);

    FILE* f = INP ? fopen(INP, "r") : stdin;
    if (!f) { fprintf(stderr, "cannot open input\n"); return 1; }

    const char* inames[] = {"input_ids", "attention_mask"};
    const char* onames[] = {"malicious_logits", "tactic_logits", "technique_logits"};
    OrtValue* inputs[2] = {0, 0};
    OrtValue* outputs[3] = {0, 0, 0};

    struct timespec t0, t1;
    clock_gettime(CLOCK_MONOTONIC, &t0);

    char *linebuf = NULL; size_t cap = 0;
    int line = 0;
    int64_t shape[2] = {1, 0};
    while (getline(&linebuf, &cap, f) != -1) {
        char* tokp = linebuf;
        long nn = strtol(tokp, &tokp, 10);
        if (nn < 1 || nn > 256) continue;
        if (tokp == linebuf) continue;
        int64_t* ids = (int64_t*)malloc(sizeof(int64_t)*(size_t)nn);
        int64_t* mask = (int64_t*)malloc(sizeof(int64_t)*(size_t)nn);
        for (int i = 0; i < nn; ++i) {
            ids[i] = (int64_t)strtol(tokp, &tokp, 10);
            mask[i] = 1;
        }
        shape[1] = nn;
        CHECK_STATUS(ort->CreateTensorWithDataAsOrtValue(mem, ids, sizeof(int64_t)*(size_t)nn, shape, 2, ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64, &inputs[0]), ort);
        CHECK_STATUS(ort->CreateTensorWithDataAsOrtValue(mem, mask, sizeof(int64_t)*(size_t)nn, shape, 2, ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64, &inputs[1]), ort);
        CHECK_STATUS(ort->Run(sess, NULL, inames, (const OrtValue* const*)inputs, 2, onames, 3, outputs), ort);

        float* mal = NULL; float* tact = NULL; float* tech = NULL;
        CHECK_STATUS(ort->GetTensorMutableData(outputs[0], (void**)&mal), ort);
        CHECK_STATUS(ort->GetTensorMutableData(outputs[1], (void**)&tact), ort);
        CHECK_STATUS(ort->GetTensorMutableData(outputs[2], (void**)&tech), ort);
        ++line;
        printf("LINE %d mal_logit %.9g", line, mal[0]);
        printf(" tact_logit");
        for (int i = 0; i < 14; ++i) printf(" %.9g", tact[i]);
        printf(" tech_logit");
        for (int i = 0; i < 12; ++i) printf(" %.9g", tech[i]);
        float prob = 1.0f / (1.0f + expf(-mal[0]));
        printf(" mal_prob %.9g", prob);
        printf(" over_thr %d", (int)(prob >= TH));
        printf("\n");
        fflush(stdout);
        ort->ReleaseValue(inputs[0]); inputs[0] = NULL;
        ort->ReleaseValue(inputs[1]); inputs[1] = NULL;
        ort->ReleaseValue(outputs[0]); outputs[0] = NULL;
        ort->ReleaseValue(outputs[1]); outputs[1] = NULL;
        ort->ReleaseValue(outputs[2]); outputs[2] = NULL;
        free(ids); free(mask);
    }
    clock_gettime(CLOCK_MONOTONIC, &t1);
    double secs = (double)(t1.tv_sec - t0.tv_sec) + (double)(t1.tv_nsec - t0.tv_nsec)/1e9;
    long vmhwm = -1;
    FILE* st = fopen("/proc/self/status", "r");
    if (st) { char lb[256]; while(fgets(lb,sizeof(lb),st)){ if(sscanf(lb,"VmHWM: %ld kB",&vmhwm)==1) break;} fclose(st); }
    fprintf(stderr, "lines=%d inferred_seconds=%.6f VmHWM_KiB=%ld VmHWM_MiB=%.1f\n", line, secs, vmhwm, vmhwm/1024.0);

    if (INP) fclose(f);
    free(linebuf);
    ort->ReleaseValue(outputs[0]); ort->ReleaseValue(outputs[1]); ort->ReleaseValue(outputs[2]);
    ort->ReleaseValue(inputs[0]); ort->ReleaseValue(inputs[1]);
    ort->ReleaseMemoryInfo(mem);
    ort->ReleaseSession(sess);
    ort->ReleaseSessionOptions(opts);
    ort->ReleaseEnv(env);
    return 0;
}
