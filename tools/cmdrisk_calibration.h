/* Calibrated operating-point parameters shared by the C inference tools.
 *
 * Source of truth:
 *   artifacts/cmd-session-risk-v5-l8-h256-million-gpu-20260922-recal-prod-session1p-20260922/onnx/calibration.json
 *
 * The ONNX graph emits a RAW malicious logit.  The Python scoring runtime
 * applies temperature scaling before the logistic transform:
 *     scaled = logit / malicious_temperature
 *     prob   = sigmoid(scaled)
 *     alert  = prob >= malicious_threshold
 *
 * The C tools must apply exactly the same scaling.  Without it the printed
 * probability (and therefore the alert decision in the near-threshold band)
 * disagrees with the calibrated session-FPR <= 1% operating point.
 *
 * This bug shipped in release v1.0.0 (C runtime) and v1.1.0 (single binary)
 * and was fixed in v1.1.1; see reports/C_TEMP_FIX_v1.1.1_zh.md.
 */
#ifndef CMDRISK_CALIBRATION_H
#define CMDRISK_CALIBRATION_H

#define MAL_TEMP 0.9085309978652624f

#endif
