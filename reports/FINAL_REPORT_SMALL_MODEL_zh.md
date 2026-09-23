# Command Session Risk 端侧小模型最终报告

- 报告日期：2026-09-20
- 模型 ID：`cmd-session-risk-small-l4-20260920-2226`
- 基座模型：`google/bert_uncased_L-4-H-256_A-4`
- 本地 artifact：`artifacts/cmd-session-risk-small-l4-20260920-2226`
- ONNX 目录：`artifacts/cmd-session-risk-small-l4-20260920-2226/onnx`
- 远端训练目录：`/teamspace/studios/this_studio/cmd-session-risk-small-l4-20260920-2226`
- 交付状态：**训练、FP32/INT8 ONNX 导出、全量验证、CPU 延迟和流式冒烟均已完成；端侧可用性已验证，但未通过 fail-closed production promotion，不能据此直接上线自动阻断。**

## 0. 结论先行

本轮已经完成端侧小模型的训练与交付验证：

1. 使用 4 层 BERT 小模型完成 3 epoch 训练，训练耗时 `575.3 s`，train loss `0.3764`，eval loss `0.1396`。
2. 已导出动态 batch/sequence 的 FP32 ONNX 和 INT8 ONNX；FP32 parity 最大绝对误差 `2.1457672119140625e-06`（atol `1e-4`），INT8 parity 最大绝对误差 `0.018622159957885742`（atol `0.05`）。
3. FP32 已在 held-out test 上完成独立全量复测：`31,467/31,467` 个 prefix、`1,958/1,958` 个 session，无缺行、无未知 session、无 prefix 数量不匹配。
4. 独立 FP32 结果为 recall `0.989212`、FPR `0.010507`、PR-AUC `0.999806`、ROC-AUC `0.997931`；session 检测率 `0.992576`，漏报 13 个 session，中位检测延迟 `0` 条命令。
5. INT8 全量复测完成；prefix 决策一致率 `0.998792`，session 决策一致率 `0.997446`。INT8 比 FP32 更小：`11.65 MB`，相比 FP32 总计 `89.47 MB` 减少约 `86.98%`。
6. 同机 CPU 实测 INT8 512-token p95 `8.298 ms`、p99 `10.685 ms`，延迟门禁通过；相比旧 ModernBERT FP32，512-token p95 约快 `26.92x`。
7. `export_manifest.json` 的 promotion 仍为 `passed: false`、`eligible_for_promotion: false`。失败项是 ECE、tactic label coverage、proxy audit 和 OOD 2024 slice；因此本模型可进入端侧试点、影子模式或人工复核链路，但不能直接作为生产自动阻断模型。

## 1. 交付物与校验值

| 交付物 | 路径 | 大小 | SHA256 |
| --- | --- | ---: | --- |
| FP32 ONNX 图 | `artifacts/cmd-session-risk-small-l4-20260920-2226/onnx/model.onnx` | 44,964,039 B | `719402c5ba5ad57aaf2d2410bb47100b79da276a9e8dd342e6949c4182db6511` |
| FP32 外部权重 | `.../onnx/model.onnx.data` | 44,505,088 B | `79e4a4a81521ceca54da55be59440c974ae3db08bb0c7118d960efad06b43bed` |
| INT8 ONNX | `.../onnx/model.int8.onnx` | 11,652,743 B | `9f86b30229bc5276ecc45f6ac9a86e30d3586d5432c8458ebc80b83a51e22a2d` |
| 导出清单 | `.../onnx/export_manifest.json` | 5,271 B | `78c4e44ee538d36fd9f6fa6a1371b0ea274fcfc3caf16a449dba800b019f9768` |
| FP32 全量打分 | `.../validation_scores.jsonl` | 26,783,173 B | `c05188f54a594829157e5998bab3d47be429489a7467d60e8c74234f38f276a0` |
| 独立 FP32 指标 | `reports/cmd-session-risk-small-l4-20260920-2226_validation_metrics.json` | 33,022 B | `3ef59b43bb6550462ecb1d0bbc6d211cf2536797c5088b4fe4bee6234407a7cc` |
| INT8 全量对比 | `.../int8_validation.json` | 1,272 B | `ed1610558ab902753199bff8aebdf2bedebcc625b47ab9b93041b565c1d761b6` |
| CPU 延迟报告 | `.../onnx/latency_report.json` | 1,986 B | `e053f4849d10fa7ab79528a4cfd73642efd0314d4852ad5038a9937ca77293a3` |
| 流式输入 | `reports/cmd-session-risk-small-l4-20260920-2226_stream_input.jsonl` | 676 B | `33e77d3e1f2883fe2c81808af34e0d291035acff04fc7376408cb986965c951a` |
| 流式输出 | `reports/cmd-session-risk-small-l4-20260920-2226_stream_output.jsonl` | 1,952 B | `bc965cebc8ca8218a763cffb960dbd4ff9523e9c11b15c0102397e8e426aae4d` |

FP32 图与外部权重合计 `89,469,127 B`，约 `89.47 MB`；INT8 为 `11,652,743 B`，约 `11.65 MB`，体积减少约 `86.98%`。

ONNX 结构事实：

- opset：`18`
- 输入：`input_ids [batch, sequence]`、`attention_mask [batch, sequence]`
- 输出：`malicious_logits [batch]`、`tactic_logits [batch, 14]`、`technique_logits [batch, 12]`
- initializer：94 个，参数元素 `11,115,822`；其中 float initializer 参数元素 `11,114,785`
- checkpoint 参数量：`11,180,598`
- batch 和 sequence 均为动态维度

## 2. 训练配置与收敛

| 项 | 值 |
| --- | --- |
| 训练模式 | short phase，`skip_long_phase=true` |
| epoch | 3 |
| 最大长度 | session `512`，单条命令 `128`，训练 prefix 上限 32 |
| 学习率 | `2e-05`，warmup ratio `0.05`，weight decay `0.01` |
| dropout | `0.1` |
| seed | `42` |
| 训练 loss | `0.3764` |
| 最终 eval loss | `0.1396` |
| 各 epoch eval loss | `0.2634017765522003` / `0.1539575755596161` / `0.13958102464675903` |
| 训练运行时间 | `575.3 s` |
| 校准集 | 4,094 样本 / 174 family，目标 FPR 1% |
| 校准阈值 | `1.2257560118872811e-24` |
| malicious temperature | `0.058460563046299915` |
| tactic temperature | `0.49597769677161135` |
| technique temperature | `0.5983144652840321` |

该模型是短上下文端侧模型，训练和导出均只覆盖 `512` token session 上限；旧大模型使用的 2048/8192 长上下文阶段本次被明确跳过。

## 3. 数据切分与覆盖

| split | session | 命令数 | family | 训练/评估有效 prefix |
| --- | ---: | ---: | ---: | ---: |
| train | 4,000 | 68,917 | 3,184 | 56,696（prefix 上限 32 后） |
| dev | 500 | 10,389 | 347 | 7,456 |
| test | 1,958 | 31,467 | 1,749 | 28,678（训练期截断口径） |

训练数据来源：

| source kind | train | dev | test |
| --- | ---: | ---: | ---: |
| real malicious | 1,589 | 195 | 1,674 |
| real benign | 1,544 | 185 | 171 |
| llm variant | 593 | 80 | 77 |
| hard negative | 199 | 25 | 26 |
| script benign | 75 | 15 | 10 |

标签覆盖存在明显缺口：

- train/dev/test 的 tactic 标签都只覆盖 `TA0005`、`TA0007`、`TA0011` 三类；14 路 tactic 输出头中的其余 11 类没有训练监督。
- train 出现 17 个 technique，但模型输出头只覆盖其中 12 个。
- test 出现 24 个 technique；未进入输出词表的标签必然无法被模型预测。
- `real_plus_semi_real` 切片没有正样本，`ood_2024` 切片没有负样本，导致 recall/FPR/ROC-AUC 等切片指标退化或无定义。
- 已确认 test 中至少存在 honeypot banner `system` 被标为 malicious 的标签噪声；不能把第一命令告警全部解释为真实行为证据。

## 4. FP32 独立全量验证

验证链路为：ONNX Runtime 对 held-out test 全量 `--flat` 打分，生成 `validation_scores.jsonl`；随后由独立脚本 `scripts/evaluate_validation.py` 重新读取 scores 和原始 test 标签复算指标。不是直接引用训练期 PyTorch 输出。

覆盖度：

- prefix：`31,467 / 31,467`
- session：`1,958 / 1,958`
- missing sessions：0
- unknown sessions：0
- prefix count mismatch：0

### 4.1 prefix 级恶意概率指标

| 指标 | 独立 FP32 ONNX 全量结果 |
| --- | ---: |
| TP / TN / FP / FN | 29,527 / 1,601 / 17 / 322 |
| Recall | **0.989212368923582** |
| Recall 95% CI | [0.9879759016270394, 0.9903229326827755] |
| FPR | **0.010506798516687269** |
| FPR 95% CI | [0.0065702307741873495, 0.01676217241163956] |
| Precision | 0.9994245870565935 |
| PR-AUC | 0.999806252381459 |
| ROC-AUC | 0.9979309951560473 |
| Brier | 0.0547269681597377 |
| ECE | 0.054808729780529815 |
| Log loss | 1.4668868921607754 |

### 4.2 session 检测与延迟

| 指标 | 值 |
| --- | ---: |
| malicious sessions | 1,751 |
| detected sessions | 1,738 |
| missed sessions | 13 |
| detection rate | 0.9925756710451171 |
| false-positive sessions | 12 |
| median detection delay | 0 条命令 |
| p90 detection delay | 0 条命令 |
| max detection delay | 8 条命令 |

绝大多数检测在第一条命令即发生；但该结论仍受 honeypot banner 标签和其他数据噪声影响。

## 5. INT8 全量验证

INT8 已对相同 test 集的 `1,958` 个 session、`31,467` 个 prefix 全量执行，无跳过。

| FP32 vs INT8 对比项 | 值 |
| --- | ---: |
| prefix probability delta mean | 0.0004747840142882828 |
| prefix probability delta p95 | 5.2802207051172445e-12 |
| prefix probability delta p99 | 0.0002508624466326381 |
| prefix probability delta max | 0.9833338024342577 |
| prefix 决策一致率 | 0.9987923856738806 |
| session 决策一致率 | 0.9974463738508682 |
| 首告警漂移 session 数 | 10 |
| 首告警漂移 mean | 0.02167712492869367 |
| 首告警漂移 max | 22 |

该脚本自身 session 聚合口径下：

| 脚本口径 | FP32 | INT8 | INT8 - FP32 |
| --- | ---: | ---: | ---: |
| recall | 0.9920045688178184 | 0.9942889777270132 | +0.0022844089091947906 |
| FPR | 0.05314009661835749 | 0.057971014492753624 | +0.004830917874396136 |
| FN | 14 | 10 | -4 |
| FP | 11 | 12 | +1 |

这里必须区分口径：上表的 session recall/FPR 来自 `scripts/validate_int8.py` 的内部聚合方式，不能替代第 4 节独立 FP32 报告中的头条指标。INT8 决策一致性高，但存在少量非常大的单点概率偏差和首告警漂移；建议端侧上线前继续按目标设备与目标阈值复测。

## 6. parity、manifest、测试与流式冒烟

### 6.1 parity

| 模型 | 最大绝对误差 | atol | 判定 |
| --- | ---: | ---: | --- |
| FP32 PyTorch vs ONNX | 2.1457672119140625e-06 | 1e-4 | 通过 |
| INT8 ONNX vs FP32 ONNX | 0.018622159957885742 | 0.05 | 通过 |

### 6.2 manifest

`onnx/export_manifest.json` 当前声明 12 个交付文件，包含 FP32、INT8、tokenizer、label vocabulary、校准、指标、运行时配置和延迟报告。使用仓库 `verify_export_manifest` 校验返回 `ok: true`。`runtime_config.json` 和 manifest 的 `model_version` 已统一为 `cmd-session-risk-small-l4-20260920-2226`。

历史 `train.log` 的导出日志中曾出现 `model_version: artifacts`，那是一次导出时使用输出目录名产生的历史值；最终 manifest 和运行时配置已刷新为正式模型 ID，实际交付以当前 `export_manifest.json` 为准。

### 6.3 测试

仓库测试结果：

```text
232 passed, 1 skipped
```

仅有 ONNX `dynamic_axes` 弃用类 warning，没有测试失败。

### 6.4 流式冒烟

流式冒烟包含 8 条恶意 session 命令，输出 8 条：

- 8/8 均 `alert: true`
- 8/8 均 `malicious_probability: 1.0`
- 每行 `model_version` 都是 `cmd-session-risk-small-l4-20260920-2226`

本次留存文件只覆盖恶意样本，不能据此声称良性流式误报率；良性与端侧目标设备行为仍需单独验证。

## 7. CPU 延迟

环境：Intel Core Ultra 7 270K Plus、10 logical CPU、AVX2、x86_64、ONNX Runtime `1.30.0`、`CPUExecutionProvider`。latency launcher 在 INT8 存在时自动选择 `model.int8.onnx`。

| requested tokens | p50 ms | p95 ms | p99 ms |
| ---: | ---: | ---: | ---: |
| 128 | 1.6730 | 2.2165 | 3.4759 |
| 256 | 3.0332 | 3.9686 | 4.3340 |
| 512 | 6.3302 | 8.2984 | 10.6846 |

门禁为 CPU p95 不超过 50 ms、加 10 ms 容差；p99 不超过 70 ms、加 10 ms 容差。本次 `passed: true`。

同机同长度旧 ModernBERT FP32 对比：

| tokens | 旧 p50 | 旧 p95 | 旧 p99 | 512-token 加速（p50 / p95 / p99） |
| ---: | ---: | ---: | ---: | --- |
| 128 | 47.9962 | 59.7711 | 62.5580 | 28.69x / 26.97x / 18.00x |
| 256 | 90.9291 | 115.0204 | 120.9686 | 29.98x / 28.98x / 27.91x |
| 512 | 194.1312 | 223.3711 | 234.5077 | **30.67x / 26.92x / 21.95x** |

旧 ModernBERT latency 文件：`reports/modernbert_latency_same_lengths.json`，SHA256 `aa38d619ea5876ae98f1df102e21c360e45645e549eb2d1d6f41cdf836262f89`。

## 8. Promotion 门禁：未通过，端侧可用不等于可上线

`export_manifest.json` promotion 状态：

```json
{
  "passed": false,
  "required": false,
  "metrics_passed": false,
  "latency_passed": true,
  "eligible_for_promotion": false,
  "failures": [
    "promotion_gate_failed",
    "promotion_violations_present",
    "evaluation_slices_passed",
    "proxy_audit"
  ]
}
```

`onnx/metrics.json` 的门禁指标如下。该表是训练期校准模型/导出快照口径，不是第 4 节的独立全量复测口径；两种口径最终都支持“不要直接上线”的结论。

| 门禁 | 实测值 | 要求 | 判定 |
| --- | ---: | ---: | --- |
| Recall | 0.9873524451939292 | ≥ 0.90 | 通过 |
| FPR | 0.009285714285714286 | ≤ 0.02 | 通过 |
| PR-AUC | 0.9999020267841491 | ≥ 0.90 | 通过 |
| ROC-AUC | 0.9980881767620164 | ≥ 0.95 | 通过 |
| Brier | 0.06047797546652578 | ≤ 0.10 | 通过 |
| ECE | **0.0605717649662843** | ≤ 0.05 | **失败** |
| detection rate | 0.9937178754997145 | ≥ 0.90 | 通过 |
| median detection delay | 0 | ≤ 3 | 通过 |
| tactic macro-F1（support ≥ 50） | 0.6915997775368362 | ≥ 0.65 | 通过 |
| tactic label coverage | **7** | ≥ 10 | **失败** |
| technique macro-F1 | 0.7356037872217246 | ≥ 0.65 | 通过 |
| proxy audit | **不可用/未满足** | 至少 500 resolved 且无 blocker | **失败** |
| evaluation slices | **至少一个失败** | 全部通过 | **失败** |

独立 FP32 复测的 ECE 为 `0.054809`，虽然低于训练期 `0.060572`，但仍高于 `0.05`，因此不能把 ECE 失败解释成单一评测脚本误差。

### 8.1 切片结果

| 切片 | 正/负 prefix | Recall | FPR | ECE | Tactic F1（support ≥ 50） | 判定 |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| `real_only` | 4,518 / 1,063 | 1.0 | 0.007526 | 0.000220 | 0.956733 | 通过 |
| `synthetic_stress` | 1,801 / 220 | 1.0 | 0 | 1.93e-12 | 0.925573 | 通过 |
| `real_plus_semi_real` | **0** / 117 | — | — | 9.46e-25 | — | 门禁仅因 Brier/ECE 可定义而通过；**无正样本，不能当正向证据** |
| `ood_2024` | 20,959 / **0** | 0.983539 | 不可定义 | 0.082821 | 0.629468 | **失败**；无负样本，且 ECE、tactic F1 均未达标 |

因此 `evaluation_slices_passed: false` 不能仅归因于“标签为 0 的统计退化”：`ood_2024` 同时存在 ECE 和 tactic macro-F1 未达标；而 `real_plus_semi_real` 的通过也没有足够统计意义。

### 8.2 Proxy audit

独立审计文件 `reports/proxy_audit/report.json`：

- total cases：607
- resolved：569
- high consensus：483
- medium consensus：86
- unresolved：38
- Fleiss kappa：0.7341925484394832
- 与共识比较：accuracy `0.880492`、precision `0.972292`、recall `0.871332`、FPR `0.087302`
- blocker：`unresolved_sessions_present:38`
- eligible for promotion：false

该审计是 3 个外部 agent 经 `ProxyAuditor` 重放形成的共识，不是人工 gold set；provenance 也没有记录厂商/模型身份。因此它只能作为证据质量筛查，不能替代人工审计和生产 FPR 验证。

`metrics.json` 中的导出快照仍显示 `proxy_audit_missing`、resolved 0，这是因为模型导出时独立审计尚未完成；独立审计完成后仍有 38 个未决 case。两个快照的 blocker 原因不同，但都判定不通过。

## 9. 已知限制与风险

1. **ECE 未过线**：训练期口径 `0.060572 > 0.05`，独立 FP32 全量口径 `0.054809 > 0.05`，需要重新校准或在目标场景重建阈值。
2. **ATT&CK 标签覆盖不足**：数据只监督 3 个 tactic，门禁要求至少 10 个输出头有足够支持；tactic macro-F1 不应被宣传为完整 ATT&CK 分类能力。
3. **technique 词表不完整**：输出头 12 个，train 17 个、test 24 个；词表外标签无法预测，相关 macro-F1 天然偏低。
4. **切片退化**：`real_plus_semi_real` 无正样本，`ood_2024` 无负样本；当前门禁已 fail-closed，但数据补全后必须重跑。
5. **审计不是人工金标准**：38 个未决 session 和外部 agent 身份缺失，无法支撑高风险生产结论。
6. **INT8 有长尾偏差**：prefix 概率最大差 `0.983334`，10 个 session 的首告警位置发生漂移；平均一致性高不等于所有边界样本无风险。
7. **数据噪声**：test 的 honeypot banner `system` 被标为 malicious，模型对首命令告警的一部分可能来自数据集标签偏差。
8. **目标设备未覆盖**：延迟是在当前 Intel CPU profile 上测得，不是目标端侧设备的最终内存/热耗/批处理性能报告；流式冒烟只含恶意样本。
9. **长上下文被跳过**：本次专门交付短上下文小模型，`skip_long_phase=true`，不应期待其复现旧长上下文模型的全部能力。

## 10. 复现命令

设：

```bash
A=artifacts/cmd-session-risk-small-l4-20260920-2226
```

训练与 FP32/INT8 导出（示意，实际远端目录和 detached session 由训练脚本管理）：

```bash
scripts/train_remote.sh \
  --data-dir data/processed/v2 \
  --output-dir "$A" \
  --remote-root cmd-session-risk-small-l4-20260920-2226 \
  --detach -- \
  --model-name google/bert_uncased_L-4-H-256_A-4 \
  --max-epochs 3 \
  --short-max-length 512 \
  --long-max-length 512 \
  --command-max-tokens 128 \
  --max-prefixes 32 \
  --calibration-fpr 0.01 \
  --skip-long-phase
```

FP32 全量 ONNX 打分：

```bash
bash scripts/run_onnx_validation.sh \
  "$A/onnx" \
  data/processed/v2/test.jsonl \
  "$A/validation_scores.jsonl"
```

独立复算 FP32 指标：

```bash
.venv/bin/python scripts/evaluate_validation.py \
  --scores "$A/validation_scores.jsonl" \
  --test data/processed/v2/test.jsonl \
  --artifact-dir "$A/onnx" \
  --output reports/cmd-session-risk-small-l4-20260920-2226_validation_metrics.json
```

INT8 全量对比：

```bash
.venv/bin/python scripts/validate_int8.py \
  --artifact-dir "$A/onnx" \
  --input data/processed/v2/test.jsonl
```

CPU 延迟：

```bash
.venv/bin/python scripts/latency_bench.py \
  --artifact-dir "$A/onnx" \
  --lengths 128 256 512 \
  --warmup 10 \
  --iterations 50 \
  --output "$A/onnx/latency_report.json" \
  --update-manifest
```

manifest 校验：

```bash
.venv/bin/python - <<'PY'
import sys
sys.path.insert(0, "src")
from cmdrisk.artifact_manifest import verify_export_manifest

print(verify_export_manifest(
    "artifacts/cmd-session-risk-small-l4-20260920-2226/onnx"
))
PY
```

## 11. 最终建议

- **可以**：用于端侧离线评估、内部试点、影子模式、人工复核和告警排序。
- **不可以**：直接用于生产自动阻断、替代人工 gold set 审计，或宣称已覆盖完整 ATT&CK tactic/technique 体系。
- **下一步优先级**：补齐 tactic/technique 数据与输出词表 → 增加人工标注审计并消解 38 个未决 case → 修复或重新定义退化切片 → 重新校准 ECE → 在目标 CPU/NPU 设备复测 INT8 边界一致性 → 再决定是否进入生产 promotion。
