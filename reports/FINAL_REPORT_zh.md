# Command Session Risk 模型交付报告

- 版本 / 模型 ID：`cmd-session-risk-l4-20260919-0020`
- 交付日期：2026-09-20
- 任务：Command Sequence / Session 恶意概率 + ATT&CK 攻击阶段（tactic / technique）
- 形态：Prefix-aware Transformer Encoder（ModernBERT-base），导出为 FP32 ONNX
- 训练/导出环境：Lightning AI L4 23GB（8 vCPU），PyTorch + ONNX Runtime（GPU 打分，CPU 延迟基准）
- **结论先行：模型本身已可用——独立 ONNX 复测 recall 0.989 / FPR 0.0019 / 中位检测延迟 0 条命令；但它没有通过本仓库的 promotion 门禁，因此按"可用但未获准上生产"交付。门禁失败逐条列在第 7 节，全部是数据覆盖、审计、切片与延迟问题，不是恶意概率头的能力问题。**

---

## 1. 交付物清单

模型与运行时（`artifacts/cmd-session-risk-l4-20260919-0020/onnx/`）：

| 文件 | 大小 (bytes) | sha256（前 16 位） | 说明 |
| --- | --- | --- | --- |
| `model.onnx` | 598,618,975 | `8a4dcf34fc58a24b` | FP32 主图（opset 18，动态 batch） |
| `model.onnx.data` | 596,217,856 | `553f5b97c2c4903c` | 外部权重 |
| `tokenizer.json` | 3,585,462 | `61fa06f31c6bcce9` | 特化 tokenizer（128 个 added token：`<SESSION>`、`<CMD_START>`、`<CMD_END>`、`<DT_*>` 时间桶等） |
| `calibration.json` | 1,181 | `75f20bd3cfbbfbb1` | 温度缩放 + 恶意阈值 + 各 tactic 阈值 |
| `runtime_config.json` | 75 | `12da0346f0087e7e` | `model_version` 的单一事实来源 |
| `labels.json` | 412 | `26dde70fe6b035ac` | tactic / technique 输出头标签表 |
| `metrics.json` | 91,488 | `ae3bc9310d84337a` | 训练期指标 + 门禁判定（PyTorch 侧） |
| `command_tokenization.json` | 376 | `6403f8c306ccc210` | 命令级 tokenization 约定 |
| `README.md` | 92,124 | `91918dba44297750` | 模型卡 |
| `tokenizer_config.json` | 629 | `6f832f9227733a07` | tokenizer 配置 |
| `latency_report.json` | 1,644 | `874b1f97ec43627e` | 远程 CPU-only 延迟基准原始报告（见第 7 节） |

以上 11 个文件即 `export_manifest.json` 内声明的全部产物（`export_manifest.json` 自身是清单，不在声明列表内）。清单已通过 `verify_export_manifest` 全量校验：`model_version = cmd-session-risk-l4-20260919-0020`，清单自身 sha256 `94ddf0dfd875bb93…`。

证据文件：

| 文件 | 说明 |
| --- | --- |
| `artifacts/.../validation_scores.jsonl` | 31,467 行 prefix 级 ONNX 打分输出（26,109,259 B，sha256 `682f21eb…`） |
| `reports/validation_metrics.json` | 独立评估器基于上面 JSONL 复算的指标 |
| `reports/validation_run.log` | 全量打分的运行日志（含长 session 的显存重试证据） |
| `reports/proxy_audit/report.json` | 3 评审代理审计结果 |
| `reports/hard_negative_smoke.jsonl` | DevOps/安装类硬负样本冒烟测试输出 |
| `artifacts/.../onnx/metrics.json` | 训练期切片 / 门禁指标 |

推理方式：

```bash
# 会话级批量打分
cmdrisk score --artifact-dir <onnx 目录> --input sessions.jsonl --output out.jsonl
# prefix 级（实时预警链路），每行一个前缀
cmdrisk score --artifact-dir <onnx 目录> --input sessions.jsonl --flat --output out.jsonl
# 逐条命令流式打分（stdin: {"cmd": ..., "dt": ...}）
cmdrisk stream --artifact-dir <onnx 目录> --session-id <id>
```

`model_version` 由 `runtime_config.json` 驱动并写入每一行打分结果，避免评测结果与模型版本对不上号。

---

## 2. 数据与训练配置

| 项 | 值 |
| --- | --- |
| session 总数 | 6,458（real malicious 3,458 / real benign 1,900 / hard negative 250 / LLM 变体 750 / script benign 100） |
| 命令总数 | 110,773 |
| 攻击族（family）数 | 5,280 |
| 切分 | train 4,000 / dev 500 / test 1,958，**按 family 切分**（无同族泄漏） |
| prefix 展开 | train 56,696 / dev 7,456 / test（评估时未截断）31,467 |
| session 内 prefix 上限 | 32（训练期）；全量数据最长 session 134 条命令，test 集最长 128 条（即最大 prefix index 127） |
| 模型 | `answerdotai/ModernBERT-base`（ONNX initializer 参数 149,044,338） |
| 训练 | 2 epoch；short phase ≤2048 tok + long phase ≤8192 tok（`skip_long_phase: false`，长上下文阶段确实执行） |
| 校准 | 4,094 样本 / 174 family，目标 FPR 1% → 阈值 0.6498378329043767，温度 1.07396745992613 |
| 输出头 | 恶意二分类（0~1）+ 14 路 tactic 多标签 + 12 路 technique 多标签 |

prefix 训练是这套方案的关键：一条 session 的每个前缀都是独立样本，因此模型天然支持"边敲边判"，而不是等 session 结束才给结论。

---

## 3. 独立验证结果（ONNX 复测，本报告的核心证据）

训练期指标来自校准后的 PyTorch 模型；下表来自**另一条独立链路**：`cmdrisk score --flat` 用 ONNX Runtime 对 held-out test 的 1,958 个 session / 31,467 个 prefix 全量打分，再用 `scripts/evaluate_validation.py` 重新计算。该次打分由 `--provider CUDAExecutionProvider --provider CPUExecutionProvider` 启动，复现脚本为 `scripts/run_onnx_validation.sh`，原始日志见 `reports/validation_run.log`。全程耗时 1,904 s（04:21:54Z → 04:53:38Z），即 31,467 个 prefix 约 16.5 prefix/s（L4 上含长 session 的平均值）。

覆盖度干净：1958/1958 session 全部命中，无缺失、无未知、无 prefix 计数不匹配；输出行数 31,467 = 期望值。

### 3.1 恶意概率头（prefix 级）

| 指标 | 独立 ONNX 复测 | 训练期 PyTorch | 门禁 | 判定 |
| --- | --- | --- | --- | --- |
| Recall | **0.98898** [0.98773, 0.99010] | 0.98772 | ≥ 0.90 | ✅ |
| FPR | **0.001854** [0.000631, 0.005437] | 0.000714 | ≤ 0.02 | ✅ |
| Precision | 0.99990 | 0.99996 | — | — |
| PR-AUC | 0.99995 | 0.99995 | ≥ 0.90 | ✅ |
| ROC-AUC | 0.99901 | 0.99898 | ≥ 0.95 | ✅ |
| Brier | 0.00649 | 0.00719 | ≤ 0.10 | ✅ |
| ECE | 0.01064 | 0.01187 | ≤ 0.05 | ✅ |
| 混淆矩阵 | TP 29,520 / TN 1,615 / FP 3 / FN 329 | TP 26,943 / TN 1,399 / FP 1 / FN 335 | — | — |

两条链路口径不同：训练期指标在 `max_prefixes=32` 截断后的 28,678 个 prefix 上计算，独立复测覆盖全部 31,467 个 prefix。所以数字不完全相同属预期；独立复测的 FP 略多（3 vs 1），说明这些数字没有被挑过好看的截面。

### 3.2 检测延迟（实时预警最关键）

| 指标 | 值 | 门禁 |
| --- | --- | --- |
| Detection rate | 0.98744（1,751 个恶意 session 中漏报 22 个） | ≥ 0.90 ✅ |
| 中位（含删失）检测延迟 | **0 条命令** | ≤ 3 ✅ |
| 中位告警命令序号 | 第 1 条命令 | — |
| p90 告警命令序号 | 第 1 条命令 | — |
| 最大检测延迟 | 2 条命令 | — |
| 误报 session | 3 个（207 个良性 session 中） | — |

绝大多数攻击链在第 1 条命令就越过阈值；漏报的 22 个 session 按"观察到结尾都没告警"计入删失延迟，因此漏报不会让延迟指标变好看。

### 3.3 不是"复杂规则引擎"的证据

`reports/hard_negative_smoke.jsonl`（可复现）——一个完全正常的 DevOps 安装流程：

```
whoami
uname -a
curl -fsSL https://example.com/pkg/pkg-1.4.2.tar.gz -o /tmp/pkg-1.4.2.tar.gz
tar xf /tmp/pkg-1.4.2.tar.gz -C /tmp
cd /tmp/pkg-1.4.2 && chmod +x install.sh && ./install.sh
```

逐条恶意概率：0.000416 / 0.000952 / 0.000139 / 0.000127 / 0.000174，**峰值 0.00095**，阈值 0.6498，全程不告警。说明模型没有退化成"看到 download + chmod + execute 就加权报警"的规则引擎。

---

## 4. 切片结果（诚实版）

| 切片 | 正/负 | Recall | FPR | ROC-AUC | PR-AUC | Tactic F1 | Technique F1 | 切片判定 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `real_only` | 4,518 / 1,063 | 1.0000 | 0.00094 | 0.99999 | 1.0000 | 0.198 | 0.955 | ❌（tactic 门禁） |
| `synthetic_stress` | 1,801 / 220 | 1.0000 | 0.0000 | 1.0000 | 1.0000 | 0.189 | 0.894 | ❌（tactic 门禁） |
| `ood_2024` | 20,959 / **0** | 0.9840 | — | **0.0** | 1.0 | 0.199 | 0.826 | ❌（负样本为 0 → ROC=0） |
| `real_plus_semi_real` | **0** / 117 | — | 0.0 | 0.0 | 0.0 | — | — | ❌（正样本为 0 → 分类/排序/检测指标全 0；Brier 0.00205、ECE 0.00712 仍有值） |

**四个切片的 `passed` 全为 false。** 其中三个的失败与模型能力无关，而是样本分布退化（某类标签为 0）叠加 tactic 标签几乎为空（见第 7 节）：这类指标在统计上没有意义，门禁按 fail-closed 处理，本报告不做美化。

---

## 5. 代理审计（model-proxy）

- 抽样 607 个 case，3 个外部 agent 评审（经 `ProxyAuditor` 重放）独立标注：569 个形成决议，483 个高一致性，**38 个未决**。
- Fleiss κ = 0.7342（一致性中上）。
- 与共识比：accuracy 0.8805、precision 0.9723、recall 0.8713、FPR 0.0873、Brier 0.1148。
- 审计结论：`eligible_for_promotion: false`，唯一 blocker 为 `unresolved_sessions_present:38`。

必须明确的局限：**审计的"金标准"是 3 个外部 agent 的共识，不是人工标注**。`provenance.json` 只记录 `reviewer_kind: "external agent reviewers replayed through ProxyAuditor"`，没有记录模型身份/供应商，因此不能声称它们来自 3 个不同厂商的模型。它能筛掉明显错误、度量一致性，但不能替代人工 gold set；即使通过，也只说明"代理口径下没大问题"，不能直接外推成生产环境的真实 FPR。未决的 38 例正是分歧最大的样本（`ProxyAuditor` 目前硬性要求恰好 3 个 reviewer，消解需要引入第 4 位评审做 tie-break，属于代码改动）。

---

## 6. 导出与部署事实

- **只交付 FP32 ONNX，INT8 是主动跳过的**：`export_manifest.json` 中 `int8_model: null`、`int8_parity_*: null`。原因是没有完成量化后的精度/延迟双验证——宁可不交，也不交未验证的量化模型。
- FP32 parity：PyTorch vs ONNX 最大绝对误差 **5.7220458984375e-06**（atol 1e-04）。
- opset 18，动态 batch；按宽度排序 + `--max-batch-tokens` 控制 padding 浪费。
- 长 session 会撑爆显存：本次全量打分中 ONNX Runtime 报了 **4 次可恢复的分配失败**（单次申请 5.32 GB / 5.19 GB / 4.75 GB / 4.32 GB），`cmdrisk score` 的自动 batch 二分重试把这几批拆小后跑通，**输出仍是完整的 31,467 行**（原始日志：`reports/validation_run.log`）。部署建议：长 session 给更保守的 `--max-batch-tokens`（如 8192）。
- **CPU 延迟门禁不可达，且不是"差一点"**：在 8 vCPU / AVX2 的 CPU profile 下，2048 token（512 命令）p95 = 5.92 s，8192 token（2048 命令）p95 = 65.2 s，而门禁上限是 60 ms（50 ms + 10 ms 容差）——差距约 100~1000 倍。端侧 CPU 实时推理需要更小的模型或 INT8；GPU 部署则需同步修改门禁口径。

---

## 7. 门禁结论：**FAILED（未获准上生产）**

`export_manifest.json` 的 promotion 块与 `metrics.json` 一致给出 `passed: false`。

| 门禁 | 结果 | 失败根因 |
| --- | --- | --- |
| recall / FPR / PR-AUC / ROC-AUC / Brier / ECE | ✅ 通过 | — |
| detection rate / 中位检测延迟 | ✅ 通过 | — |
| technique macro-F1 | ✅ 0.843 | — |
| **tactic macro-F1** | ❌ 0.199（要求 ≥0.65） | **整个数据集只出现过 3 个 tactic：TA0005 防御规避（test 1,556）、TA0007 信息发现（1,476）、TA0011 命令与控制（596）**。14 路 tactic 头里 11 路没有任何正样本，macro-F1 在数学上不可能达标。这是数据覆盖问题，需要接入 Atomic Red Team 等多源数据补齐 tactic 标签后重训。 |
| **evaluation_slices_passed** | ❌ | 第 4 节：四个切片全部 false，其中两个因类别退化（0 正 / 0 负）导致指标无统计意义。 |
| **proxy_audit** | ❌ | 38 个评审分歧 case 未决（`unresolved_sessions_present:38`）。 |
| **latency** | ❌ | `latency_gate_failed` / `p95_exceeds_tolerance` / `p99_exceeds_tolerance`，见下。 |

> 口径说明：清单里 `proxy_audit` 这一条 blocker 来自 `metrics.json` 内嵌的审计状态快照（导出模型时审计尚未独立跑完，因此记为 `proxy_audit_missing`）；独立审计 `reports/proxy_audit/report.json` 给出的 blocker 是 `unresolved_sessions_present:38`。两者都判定"不通过"，只是快照时间点不同。

另外记录标签体系的两个偏移（属数据/建模口径问题，不是模型退化）：

1. **tactic 词表虚高**：14 路输出头，实际只有 3 个 tactic 有标签。
2. **technique 词表未对齐**：输出头是按支持度筛出的 12 个高频 technique（`T1005 T1036 T1053 T1057 T1059 T1070 T1082 T1083 T1087 T1098 T1105 T1222`）；但训练集实际出现 17 个（其中 5 个 `T1016 T1027 T1037 T1489 T1548` 无输出头），test 集出现 24 个（其中 12 个无输出头，如 T1003、T1033、T1069、T1071、T1424、T1562、T1564、T1590）。这些所谓"未见过"的标签 F1 恒为 0，是词表覆盖问题。

### 延迟门禁

门禁要求 CPU 部署 profile 下 p95 ≤ 50 ms（+10 ms 容差 = 60 ms），且显式要求 provider 恰为 `CPUExecutionProvider`——任何 GPU/加速数字都会被判 `latency_not_cpu_deployment_profile` 直接失败，防止用加速卡数字冒充端侧能力。

实测结果（`artifacts/.../onnx/latency_report.json`，sha256 `874b1f97ec43627e…`，与远程原始文件逐字节一致）：

| 环境 | 判定 |
| --- | --- |
| `deployment_profile` | `cpu`（provider 恰为 `CPUExecutionProvider`） |
| profile | Intel Xeon @2.20GHz，8 logical CPU，AVX2 ✔，x86_64，`profile_passed: true` |
| ONNX Runtime | 1.23.2 |

| 输入规模 | p50 | p95 | p99 | 门禁（≤60 ms） |
| --- | --- | --- | --- | --- |
| 512 命令 / 2,050 token | 5,209 ms | 5,915 ms | 5,930 ms | ❌ 超标约 100× |
| 2,048 命令 / 8,192 token | 56,832 ms | **65,198 ms** | 67,006 ms | ❌ 超标约 1,000× |

（表中 p50/p95 是 20 次测量的分位数，5 次预热。8192 token 时 20 次的 min ~55 s、max ~67 s，说明不是偶发抖动，而是 CPU 上长上下文编码的固有成本。）因此 `latency.failures = ["latency_gate_failed","p95_exceeds_tolerance","p99_exceeds_tolerance"]`。要在 CPU 上过门禁只有两条路——量化到 INT8（需重跑精度/延迟双验证）或把模型压到 ~30M 量级；否则必须把部署目标改成 GPU 并同步修改门禁口径。

---

## 8. 已知数据质量问题（不要当成模型的锅）

1. test 集第一条样本的 `cmd` 字段是 `system`——honeypot 登录 banner 被当成一条命令，且被标成 malicious；模型给它 0.9976554636808472 的恶意概率。这是标签/归一化产物，数据清洗里应剔除 banner 行。
2. 只有 3 个 tactic 标签可用（第 7 节）。
3. technique 词表 train/test 不一致（第 7 节）。
4. 代理审计是 agent 共识、不是人工 gold（第 5 节）。
5. `real_plus_semi_real`（0 正）与 `ood_2024`（0 负）切片类别退化（第 4 节）。

---

## 9. 复现命令

以 `A=artifacts/cmd-session-risk-l4-20260919-0020` 为例：

```bash
# 1) 全量 prefix 打分（GPU；长 session 触发自动二分重试）
#    等价于仓库内的复现脚本 scripts/run_onnx_validation.sh
bash scripts/run_onnx_validation.sh "$A/onnx" data/processed/v2/test.jsonl "$A/validation_scores.jsonl"

# 2) 独立复算指标
python scripts/evaluate_validation.py \
  --scores "$A/validation_scores.jsonl" \
  --test data/processed/v2/test.jsonl \
  --artifact-dir "$A/onnx" \
  --expected-line-count 31467 \
  --output reports/validation_metrics.json

# 3) CPU 延迟 / 部署 profile
python scripts/latency_bench.py --artifact-dir "$A/onnx" \
  --output "$A/onnx/latency_report.json" --format json --allow-failure

# 4) 刷新产物清单（model_version 由 runtime_config.json 推导）
python scripts/refresh_artifact_manifest.py --artifact-dir "$A/onnx"
```

---

## 10. 达到生产可用的最短路径（按性价比排序）

1. **补 tactic 标签**：接入 Atomic Red Team + 多源数据，把有标签的 tactic 从 3 类扩到 10+ 类，并保证每类 ≥50 session / ≥3 family（训练脚本已有同款约束参数），重训后 tactic macro-F1 才有意义。
2. **修切片**：为 `real_plus_semi_real` 补恶意样本、为 `ood_2024` 补负样本；或把"单类切片"从门禁显式降级为诊断项（需改门禁语义，不能偷偷放行）。
3. **代理审计第 4 位评审 + tie-break**，收敛 38 个未决 case。
4. **INT8 量化 + 双验证**：必须同时验证精度不塌（recall/FPR 对比 FP32）与 p95 ≤ 60 ms，否则延迟门禁依旧失败。
5. **数据清洗**：剔除 honeypot banner 类伪命令；对齐 technique 标签词表（或把输出头扩到全词表并按支持度加阈值）。
6. **人工 gold set**：从代理审计抽取 500 例人工复核，替换"agent 共识即金标准"的口径——这是让 FPR 声明可信的唯一办法。
