/* Shared scoring contract between the batch CLI and the HTTP daemon.
 *
 * Both consumers live in the same single binary: the batch CLI scores one
 * \x1f-separated session line per row, and `cmdrisk serve` reuses the exact
 * same per-prefix inference path so probabilities stay bit-identical.
 */
#ifndef CMDRISK_INFER_H
#include <stddef.h>
#define CMDRISK_INFER_H

#ifdef __cplusplus
extern "C" {
#endif

/* Result of one forward pass over a command prefix line. */
typedef struct {
    double mal_logit;  /* raw malicious logit (pre-temperature)          */
    double prob;       /* temperature-scaled sigmoid probability          */
    int rule_hit;      /* 1 when the deterministic intent layer alerts    */
    int top_tact;      /* argmax tactic-head index, or -1                 */
    int top_tech;      /* argmax technique-head index, or -1              */
    char intent[256];  /* '+' joined intent classes, or empty             */
} CmdScoreOut;

/* ---- lifecycle (process-wide, single-use) ---- */

/* Read threshold/temperature from the embedded calibration.json and verify
 * that the embedded model_config matches the compile-time tensor layout.
 * Returns 0 on success, 1 on a hard mismatch. */
int cmdrisk_calibration_setup(void);

/* Load the embedded vocab + FP32 weight blob into process globals.
 * Returns 0 on success. */
int cmdrisk_model_setup(void);

/* Allocate/reuse the forward-pass workspace buffers. Returns 0 on success. */
int cmdrisk_workspace_alloc(void);

/* Release the workspace buffers (used by tests; process exit frees anyway). */
void cmdrisk_workspace_free(void);

/* ---- scoring ---- */

/* Score one \x1f-separated command-prefix line (same format as the batch
 * CLI input files). Returns 0 on success and fills *out; returns -1 when the
 * line does not encode into a valid prefix. */
int cmdrisk_score_prefix(const char *line, CmdScoreOut *out);

/* ---- operating point + labels ---- */

double cmdrisk_threshold(void);
double cmdrisk_temperature(void);
int cmdrisk_num_tactics(void);
int cmdrisk_num_techniques(void);

/* Head-order label accessors; return NULL for out-of-range indexes. */
const char *cmdrisk_tact_label(int idx);   /* "TAxxxx" */
const char *cmdrisk_tact_name(int idx);    /* display name, e.g. "Impact" */
double cmdrisk_tact_severity(int idx);     /* 0 for out-of-range indexes */
const char *cmdrisk_tech_label(int idx);   /* "Txxxx" */

/* ---- version metadata ---- */
const char *cmdrisk_server_version(void);  /* "v1.5.0" */
const char *cmdrisk_model_version(void);   /* "v6" */

/* ---- HTTP daemon entry (implemented in cmdrisk_http.c) ---- */
int cmdrisk_serve(const char *bind_spec, size_t max_body, int timeout_sec);

#ifdef __cplusplus
}
#endif

#endif /* CMDRISK_INFER_H */
