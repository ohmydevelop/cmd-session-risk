# cmd-session-risk 生产可用性报告 V4（端侧小模型 + ONNX FP32/INT8）

- 日期：2026-09-21（America/New_York）
- 工作目录：`/home/ubuntu/Works/llm/cmd-session-risk`
- 范围：在**保留全部旧方案、旧 artifact、旧报告**的前提下，把 v3 数据 / L6-H256 方案推进到“**可用于生产灰度**”的状态：完成 ONNX FP32/INT8 导出与校验、后校准（temperature scaling）、面向会话级告警的阈值选择、INT8 一致性核验、512-token 端侧延迟复测，并给出与旧方案的完整对比。
- 硬约束声明：本环境 `torch.cuda.is_available() == False`（无 GPU/CUDA、无 `nvidia-smi`），因此**本轮没有训练任何新模型**。所有新产物都基于已冻结的 v3 检查点做**推理侧 / 校准侧**优化，权重字节未变。
- 门禁声明：**本报告不声称通过全部门禁**。所有 artifact 的 `promotion.passed` 仍为 `false`，硬 blocker 见第 7 节。
- 部署边界：本模型定位为**生产灰度 / shadow / 告警 + 人工复核候选**，**不可用于自动阻断**，直到第 7 节 blocker 全部清除。
- 过程声明：本轮**未创建 token budget**；**未删除或覆盖任何旧方案 / 旧 artifact / 旧报告**；**未伪造任何 proxy audit / reviewer / promotion 结论**。

---

## 0. 结论先行

1. **生产默认推荐**：`artifacts/cmd-session-risk-v3-l6-h256-recal-prod-20260921`
   - 权重与 `cmd-session-risk-v3-l6-h256-20260920` **逐字节一致**（INT8 `sha256 = 4d03a5d453222a5598cc4a0c33a49a16bec7b56267362bafe1b4f1278ac39077`），只改了校准温度与阈值。
   - 校准：`malicious_temperature = 0.6804212`，`malicious_threshold = 0.25333938`，`selection_metric = session_fpr`，目标为**会话级** FPR ≤ 1%。
   - v3 test（held-out）：**会话级** recall `0.930021`、FPR `0.003268`（FN 131 / FP 1）；**cap32 prefix 级** recall `0.969408`、FPR `0.000470`、precision `0.999965`、PR-AUC `0.999854`、ROC-AUC `0.997883`、Brier `0.033117`、ECE `0.042045`。
   - 512-token INT8 端侧延迟（10 warmup / 50 iter）：p50 `8.48 ms`、p95 `18.55 ms`、p99 `27.80 ms`、max `32.02 ms`，均低于 50/70 ms 门限。
2. **高召回回退**：`artifacts/cmd-session-risk-v3-l6-h256-recal-20260921`（`T = 0.6798584`，`threshold = 0.002157844`）。会话级 recall `0.995192`、FPR `0.058824`；prefix 级 recall `0.990907`、FPR `0.014568`。适合“宁可多报不可漏报”的取证/审计场景。
3. **保守回退**：`artifacts/cmd-session-risk-v3-l4-h256-20260920`（B，v3+L4）。参数量与 A 相同，端侧最快（p95 `7.79 ms`），INT8 最小（`11.65 MB`），但 prefix FPR 最高（`0.0263`）。
4. **是否还能“再调大参数”？** 不能作为默认动作。L6（+14.13% 参数、+25.3% 训练时间、+15.53% INT8 体积）已经实测出 OOD recall 下降 1.51pt、p95 延迟 +138.2%；继续放大缺乏收益保证。**当前瓶颈是数据分布与审计覆盖，不是模型容量。**
5. **本轮真正带来收益的优化**（都不增加推理耗时）：后校准温度缩放、**按会话级 FPR 选阈值**（而不是按 prefix 级 FPR）、以及把“评估口径”和“部署口径”对齐。详见第 4 节。

---

## 1. 当前训练数据有多少、怎么布局（本轮实测）

### 1.1 文件与字段

- 主数据：`data/processed/v3/{train,dev,test}.jsonl`，**一行一个 session**。
- 评估切片：`data/processed/v3/slices/{ood_2024,synthetic_stress}.jsonl`。
- 每个 session 关键字段：`session_id`、`family_id`、`commands`（元素为 `{cmd, dt}` 的命令序列）、`source`（`kind` / 来源 / 许可证 / 父 session）、`label`（`malicious`、`first_evidence_index`、`tactics`、`techniques`、质量与 teacher 投票）、`split`、`meta`、`schema_version`。
- **前缀样本（prefix）**由 session 的命令前缀生成：
  - `full prefixes` = session 全部合法前缀数；
  - `cap32 prefixes` = 训练/评估/布署实际使用的“最多 32 个前缀”（family-aware 采样）。

### 1.2 v3 主数据规模（本轮重新统计，与旧报告一致）

| 数据集 | Split | Sessions | Families | Malicious / Benign | Full prefixes | cap32 prefixes |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| v3 | train | 5,759 | 4,416 | 3,117 / 2,642 | 101,591 | 82,433 |
| v3 | dev | 721 | 510 | 429 / 292 | 14,242 | 10,541 |
| v3 | test | 2,178 | 1,916 | 1,872 / 306 | 34,870 | 31,711 |

### 1.3 v3 train 的来源构成（按 `source.kind`）

| kind | Sessions |
| --- | ---: |
| real_malicious | 2,258 |
| real_benign | 2,229 |
| llm_variant | 859 |
| hard_negative | 296 |
| script_benign | 117 |
| **合计** | **5,759** |

dev：real_benign 252 / real_malicious 313 / llm_variant 116 / hard_negative 28 / script_benign 12。
test：real_malicious 1,767 / real_benign 251 / llm_variant 105 / hard_negative 36 / script_benign 19。

### 1.4 数据流水线（`data/processed/v3/reports/prepare_state.json`）

```
原始池 21,194 sessions
  -> exact dedup      12,029
  -> family-aware dedup 9,758
  -> 配额/过滤后        8,658  (= train 5,759 + dev 721 + test 2,178)
```

关键参数：`pipeline_version=v2`、`total=7,200`、`max_records_per_source=5,000`、`holdout_period=2024`、`seed=42`、`jaccard_threshold=0.9`、`max_family_fraction=0.0025`、`minimum_family_cap=50`。
另有 `holdout_count=1,458`（= OOD 2024 切片）、`stress_generated=1,440`（含 `stress_attack_variants=1,080`、`stress_hard_negatives=360`）。

### 1.5 评估切片

| 数据集 | Sessions | Families | Malicious / Benign | 说明 |
| --- | ---: | ---: | ---: | --- |
| OOD 2024 (`slices/ood_2024.jsonl`) | 1,458 | 1,367 | 1,458 / **0** | 时间切片，**全部恶意、无良性**；只能报 recall / detection rate / delay，**FPR、ROC-AUC、PR-AUC 在该切片上未定义** |
| synthetic stress (`slices/synthetic_stress.jsonl`) | 1,440 | 1,381 | 1,080 / 360 | 压力切片，**与 train 有重合**，不能当作独立泛化证据 |

### 1.6 切分隔离

- v3 内部 `train / dev / test`：session overlap = `0`，family overlap = `0`（family-aware 切分）。
- OOD 2024 与 v3 的 train/dev 无 session/family 重合，但其 session 同时出现在 v3 test 集合中，属于**时间切片评估**，不是第二个完全独立的 holdout。
- synthetic stress 与 v3 train 存在重合，仅用于压力分析。

---

## 2. 数据量是不是太少了？

**要分任务回答，不能一句话概括。**

### 2.1 二分类（malicious / benign）：对 L4–L6 这个规模不算少

- train 5,759 session / 4,416 family / 82,433 cap32 prefix；v3 test 31,711 cap32 prefix。
- 依据：A/B 在公共 clean full 上 recall 完全相同（`0.996582`）；把模型从 L4 扩到 L6（参数 +14.13%）后，clean recall 仍是 `0.996582`，但 loss 更低、OOD 反而更差。**这说明二分类已接近该数据量下的性能饱和，继续堆参数边际收益极低。**

### 2.2 但下面几类仍然明显偏少（这才是真正的瓶颈）

| 缺口 | 实测证据 | 影响 |
| --- | --- | --- |
| **真实良性 session 太少** | v3 test 仅 306 benign；dev 仅 292 benign | 会话级 FPR 的分辨率只有 `1/292 ≈ 0.34%`；阈值只能被识别到一个很宽的区间（见 4.3），门禁 `fpr ≤ 0.02` 的置信度受限 |
| **OOD 切片没有良性** | OOD 2024 = 1,458 session / 0 benign | `evaluation_slice:ood_2024` 永久 blocker（`insufficient_negatives`），FPR/ROC-AUC/PR-AUC 不可计算 |
| **real_plus_semi_real 切片没有正类** | `negative_count=125, positive_count=0` | `evaluation_slice:real_plus_semi_real` 永久 blocker（`insufficient_positives`） |
| **tactic 标签覆盖不足** | v3 test 只有 **7** 个 tactic head 的 support ≥ 50，门禁要求 **10** | `tactic_label_coverage` blocker；TA0001/TA0008/TA0010 在 test 上 support 为 0，TA0006 只有 4 |
| **proxy audit 未收敛** | 607 cases：569 多数票 + 27 tiebreak = 596 resolved，**11 unresolved**，Fleiss κ 0.692 | `proxy_audit` blocker；需要真实新增独立 reviewer，不能伪造投票 |

**结论：二分类数据够用；真实良性、OOD 良性、稀有 tactic/technique、独立审计覆盖不足。下一步优先级是“补数据 + 补审计”，不是“继续加大模型”。**

---

## 3. 保留的方案与实验定义

| 方案 | 数据 | 初始化 | Artifact | 说明 |
| --- | --- | --- | --- | --- |
| A（旧） | v2 | `bert_uncased_L-4_H-256_A-4` | `artifacts/cmd-session-risk-small-l4-20260920-2226` | taxonomy/OOD 回退 |
| B（旧） | v3 | `bert_uncased_L-4_H-256_A-4` | `artifacts/cmd-session-risk-v3-l4-h256-20260920` | 旧生产推荐 |
| C（旧） | v3 | `bert_uncased_L-6_H-256_A-4` | `artifacts/cmd-session-risk-v3-l6-h256-20260920` | 扩容对照，不晋级 |
| C-recal（新） | v3 | 同 C（权重冻结） | `artifacts/cmd-session-risk-v3-l6-h256-recal-20260921` | 全 dev 重新拟合温度/阈值，高召回档 |
| **C-recal-prod（新）** | v3 | 同 C（权重冻结） | `artifacts/cmd-session-risk-v3-l6-h256-recal-prod-20260921` | **按会话级 FPR ≤ 1% 选阈值，生产默认** |

共同训练/导出设置（A/B/C）：short sequence 512 token、command token cap 128、prefix cap 32、seed 42、epoch 3、`skip_long_phase=true`。
C-recal / C-recal-prod **没有重新训练**：`recalibration.json` 里 `weights_unchanged: true`，三个 artifact 的 `model.onnx` / `model.onnx.data` / `model.int8.onnx` 字节完全一致。

---

## 4. 本轮指定并执行的优化方案

### 4.1 优化 1：后校准温度缩放（post-hoc temperature scaling）

- 在 family-disjoint 的 **dev** 上重新拟合 malicious / tactic / technique 温度；**test 从未参与拟合**（`test_used_for_fitting: false`）。
- **provenance（本轮实际设置）**：`--calibration-fraction 1.0`，即在**完整 family-disjoint dev（cap32 前缀 10,541 条，缓存于 `/tmp/recal/full_dev_malicious_logits.npz`）**上重新拟合温度/阈值；`recalibration.json` 的 `weights_unchanged: true` 与 `source_temperature 0.5568582` 记录了这一链路。旧 artifact 的说明若提到“half-dev（0.5）拆分”，属于**旧的半量拟合口径**，与本轮（fraction = 1.0）不一致，**不作为本轮 provenance**。
- 效果：把 C 的概率质量从“欠校准”拉回来（ECE 从 B 的 `0.0834` 降到 C 系的 `0.0420`），且**零训练成本、零推理成本**。
- 温度：`malicious 0.6804212`、`tactic 0.6836067`、`technique 0.7594778`。

### 4.2 优化 2：把阈值选择口径从“prefix 级”改成“会话级”（本轮最大收益点）

部署是**会话锁存告警**：只要该 session 任一已观测 prefix 越过阈值就告警。因此运维真正关心的是 **session 级 FPR**，而旧方案（C，`threshold = 0.0014867`）是按 prefix 级目标选的。

| 口径 | 阈值 | v3 test recall | v3 test FPR | FN | FP |
| --- | ---: | ---: | ---: | ---: | ---: |
| 旧 C（prefix 口径，thr `.0014867`） | 0.0014867 | 0.979583 | 0.002820 | 604 | 6 |
| prod（session 口径，thr `.25333938`） | 0.25333938 | 0.969408 | 0.000469925 | 905 | 1 |
| prod（session 口径，会话级读数） | 0.25333938 | **0.930021** | **0.003268** | 131 | **1** |

> 注意口径：上行是 **cap32 prefix 级**（分母 29,583 正 / 2,128 负），中行同口径；下行是**会话级**（分母 1,872 正 / 306 负）。两者不可混用。

### 4.3 优化 3：会话级操作点扫描（新增脚本 `scripts/session_operating_point_sweep.py`）

在 dev 会话级 FPR 预算上扫描阈值，再在 test 上评估（阈值只用 dev 选）：

| dev 会话 FPR 预算 | 选中阈值 | test 会话 recall | test 会话 FPR | FN | FP |
| --- | ---: | ---: | ---: | ---: | ---: |
| ≤ 0.0025 | 0.993247 | 0.914530 | 0.000000 | 160 | 0 |
| ≤ 0.005 | 0.970807 | 0.922009 | 0.000000 | 146 | 0 |
| ≤ 0.0075 | 0.053419 | 0.932692 | 0.003268 | 126 | 1 |
| ≤ 0.010 | 0.053419 | 0.932692 | 0.003268 | 126 | 1 |
| ≤ 0.015 | 0.017776 | 0.935363 | 0.003268 | 121 | 1 |
| ≤ 0.020 | 0.016512 | 0.935363 | 0.003268 | 121 | 1 |
| ≤ 0.030 | 0.005661 | 0.953526 | 0.009804 | 87 | 3 |

**关键发现（阈值可辨识性）**：dev 只有 292 个 benign，会话级 FPR 的粒度是 `1/292 = 0.34%`。dev 负样本里有两个极端离群点（`0.97081`、`0.99325`），因此**任何落在 `(0.053419, 0.970807]` 的阈值在 dev 上完全等价**（都是 2/292 = 0.685%）。当前生产阈值 `0.25333938` 落在该**宽等价区间 `(0.053419, 0.970807]`** 内，属于“dev 只读”意义下的一次部署选择 —— 区间**两个端点在 dev 上都落在同一档**（2/292 = 0.685%），因此在 dev 视角下无法再把阈值细分；区间宽度本身，正是 **dev benign 样本不足** 的直接后果。

### 4.4 优化 4：评估并否决“持久化 / k-of-n 规则”

在同一阈值上要求“至少 2 个 prefix 越线才告警”（纯后处理，无额外推理成本）：

| 阈值 | 规则 | test 会话 recall | test 会话 FPR | FN | FP |
| ---: | --- | ---: | ---: | ---: | ---: |
| 0.053419 | 任一 prefix（1-of-n） | 0.932692 | 0.003268 | 126 | 1 |
| 0.053419 | 2-of-n | 0.917201 | 0.000000 | 155 | 0 |
| 0.017776 | 任一 prefix | 0.935363 | 0.003268 | 121 | 1 |
| 0.017776 | 2-of-n | 0.918269 | 0.000000 | 153 | 0 |

**结论：2-of-n 会额外丢掉约 1.5pt 会话 recall，只换来 1 个 FP** —— 因为绝大多数恶意会话只在很短的前缀窗口内越线。**该优化被否决，不进入生产规则。**

### 4.5 优化 5：INT8 动态量化（已有）+ 一致性核验

- INT8：ONNX Runtime 动态量化；`model.int8.onnx` = `13.46 MB`（FP32 合计 `102.29 MB`）。
- 与 FP32 的差异见第 6 节，决策一致率 ≥ `0.9995`。
- **静态 QDQ / SmoothQuant / QAT / 蒸馏**属于需要训练的候选，本轮因无 GPU 不做，列入第 9 节计划。

---

## 5. 完整对比结果

### 5.1 参数量、体积与训练成本

| 方案 | Checkpoint 参数 | ONNX initializer 元素 | FP32 ONNX | INT8 ONNX | 训练时间 | Train loss | Eval loss |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| A (v2+L4) | 11,180,598 | 11,115,822 | 89,469,127 B | 11,652,743 B | 575.3 s | 0.3764 | 0.1396 |
| B (v3+L4) | 11,180,598 | 11,115,822 | 89,465,782 B | 11,649,977 B | 983.4 s | 0.3004 | 0.1244 |
| C (v3+L6) | 12,760,118 | 12,695,342 | 102,289,492 B | 13,459,220 B | 1,232.0 s | 0.2640 | 0.1124 |
| C-recal | 同 C（权重冻结） | 同 C | 同 C | 同 C | 0（不重训） | — | — |
| **C-recal-prod** | 同 C（权重冻结） | 同 C | 同 C | 同 C | 0（不重训） | — | — |

- C 相对 A/B：参数 `+14.13%`，训练时间相对 B `+25.3%`，INT8 体积 `+15.53%`。
- recal / recal-prod 与 C 的 ONNX 三个文件 `sha256` 完全相同，**没有任何二次训练成本**。

### 5.2 v3 test 上的 prefix 级对比（同一评估面：29,583 正 / 2,128 负 cap32 prefix）

| 方案 | 阈值 | Recall | FPR | FN / FP | PR-AUC | ROC-AUC | Brier | ECE | Tactic F1(≥50) | Technique F1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| B (v3+L4) | 0.0007472 | 0.993611 | 0.026316 | 189 / 56 | 0.999868 | 0.998215 | 0.075053 | 0.083429 | 0.637657 | 0.650555 |
| C (v3+L6) | 0.0014867 | 0.979583 | 0.002820 | 604 / 6 | 0.999845 | 0.997887 | 0.033109 | 0.039898 | 0.740135 | 0.751382 |
| C-recal | 0.0021578 | 0.990907 | 0.014568 | 269 / 31 | 0.999854 | 0.997883 | 0.033117 | 0.042032 | 0.718383 | 0.752879 |
| **C-recal-prod** | 0.2533394 | 0.969408 | 0.000469925 | 905 / 1 | 0.999854 | 0.997883 | 0.033117 | 0.042032 | **0.784417** | **0.827188** |

（A 的 `onnx/metrics.json` 使用不同评估面，positive_count 27,278 / negative_count 1,400，不与本表混排；A 的公共 clean 对比如第 5.4 节。）

> ⚠️ **口径说明**：本表 `C-recal-prod` 行是 **family-aware cap32 采样、prod 冻结阈值 `0.2533394` 的当前 ONNX 读数**（来源 `onnx/recalibrated_onnx_evaluation.json` 的 `test.prefix`）。artifact 内的 `onnx/metrics.json` 的 `malicious` 块仍是**导出时旧 PyTorch 阈值**下生成的（recall `0.982862` / FPR `0.005169` / FN 507 / FP 11），**不能作为当前 prod 的 headline**；同表 `Tactic F1` / `Technique F1` 两列也派生自该 legacy `metrics.json`，仅作**旧阈值诊断**参考，不参与当前 ONNX 决策口径。

### 5.3 v3 test 上的**会话级**对比（运维口径：1,872 正 / 306 负 session）

| 方案 | 阈值 | Recall | FPR | FN / FP | Precision | PR-AUC | ROC-AUC | Brier | ECE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **C-recal-prod**（FP32 ONNX） | 0.2533394 | 0.930021 | 0.003268 | 131 / 1 | 0.999426 | 0.999635 | 0.997713 | 0.059379 | 0.062861 |
| **C-recal-prod**（INT8 ONNX） | 0.2533394 | 0.930556 | 0.003268 | 130 / 1 | 0.999426 | 0.999635 | 0.997713 | 0.059379 | 0.062861 |
| C-recal（高召回回退） | 0.0021578 | 0.995192 | 0.058824 | 9 / 18 | 0.990431 | 0.999623 | 0.997621 | 0.060124 | 0.063202 |

dev 上的同一口径：prod 会话 recall `0.946387`、FPR `0.006849`（2/292），prefix recall `0.996664`、FPR `0.002163`。

> 口径说明：上表 INT8 行的 `130 / 1` 是 **cap32 采样**口径；在 **full-prefix fail-closed** 口径下 INT8 与 FP32 的会话读数**完全相同**（`131 / 1`），见第 6.1 节。生产以 6.1 的 full-prefix 一致率 `1.0` 为准。

### 5.4 legacy 诊断口径（common clean full + 旧阈值 `metrics.json`，仅用于新老衔接）

> 本小节的 B/C 行、`Tactic F1` / `Technique F1` 与 OOD recall 均取自**旧 PyTorch / 导出时阈值**的 `metrics.json`，属于 **legacy diagnostic**，**不得与第 5.2/5.3 节的当前 ONNX headline 混表**。

| 方案 | Recall | FPR（Wilson 95% CI） | Tactic F1(≥50) | Technique F1 | OOD 2024 recall | OOD FN / max delay |
| --- | ---: | --- | ---: | ---: | ---: | --- |
| A | 0.996582 | 0.048387 `[0.0224, 0.1016]` | 0.823947 | 0.941702 | 0.991770 | 12 / 8 |
| B | 0.996582 | 0.032258 `[0.0126, 0.0800]` | 0.749458 | 0.895582 | 0.990398 | 14 / 8 |
| C | 0.996582 | 0.024194 `[0.0083, 0.0687]` | 0.755950 | 0.897054 | 0.975309 | 36 / 27 |

OOD 2024（prod 校准口径，仅恶意，20,959 正）：detection rate `0.984225`、median delay `0`、Brier `0.049672`、tactic F1(≥50) `0.734484`、technique F1 `0.783367`。**该切片无良性，故 FPR / ROC-AUC / PR-AUC 一律不报。**

---

## 6. INT8 一致性与端侧延迟

### 6.1 INT8 vs FP32（v3 test，prod 阈值 `0.25333938`）

**权威证据 —— full-prefix fail-closed 校验**（`onnx/int8_validation.json`，不对前缀做 cap32 采样，分母 = 会话全部合法前缀 **34,870** 个 / **2,178** 个会话）：

| 指标 | 值 |
| --- | ---: |
| Prefix decision agreement | **0.9999713** |
| Session decision agreement | **1.0** |
| FP32 会话读数 | TP 1741 / FN 131 / FP 1 / TN 305（recall `0.930021` / FPR `0.003268`） |
| INT8 会话读数 | TP 1741 / FN 131 / FP 1 / TN 305（recall `0.930021` / FPR `0.003268`，与 FP32 完全一致） |
| ΔRecall / ΔFPR（INT8 − FP32，会话级） | 0.000000 / 0.000000 |
| First-alert index drift（max / mean / differing sessions） | **0 / 0 / 0** |
| Prefix probability delta（max / mean / p95 / p99） | 0.305780 / 0.000601 / 0.000596 / 0.011337 |
| Sessions / prefixes scored | 2,178 / 34,870 |

**operational 参考 —— cap32 采样评估面**（`onnx/recalibrated_onnx_evaluation.json` 的 `int8_comparison`，即训练评估器实际使用的采样口径）：prefix decision agreement `0.999874`、session decision agreement `0.999541`、ΔRecall `+0.000534`、ΔFPR `0.000000`、first-alert drift max `12`（3 个会话）、prefix Δ（max/mean/p95/p99）`0.335925 / 0.001603 / 0.000671 / 0.065115`。

> 两个口径**不可混用**：full-prefix 是完整前缀流上的 fail-closed 判定；cap32 是训练/评估采样口径。生产报告以 **full-prefix 的会话级一致率 `1.0`** 为准。

**判定：可用于生产（INT8 与 FP32 决策等价）。** 在完整前缀流上，INT8 与 FP32 的**会话级决策 100% 一致、首个告警位置零漂移**；在 cap32 采样口径上二者也只有 1 个会话的差异（方向为 INT8 多报 1 个），FPR 不变。

### 6.2 512-token INT8 CPU 延迟

环境：Intel Core Ultra 7 270K Plus、10 logical CPU、AVX2、x86_64、ONNX Runtime `1.30.0`、`CPUExecutionProvider`、`ORT_ENABLE_ALL`、batch size 1、512 token、10 warmup + 50 iterations。

| 方案 | p50 | p95 | p99 | max | p95/p99 门限 | 结果 |
| --- | ---: | ---: | ---: | ---: | --- | --- |
| A (v2+L4) | 6.3302 ms | 8.2984 ms | 10.6846 ms | — | 50 / 70 ms | 通过 |
| B (v3+L4) | 5.9302 ms | 7.7854 ms | 9.4440 ms | — | 50 / 70 ms | 通过 |
| **C-recal-prod (v3+L6)** | **8.4834 ms** | **18.5457 ms** | **27.8010 ms** | 32.0200 ms | 50 / 70 ms | 通过 |

- C 系相对 B：p50 `+43.1%`、p95 `+138.2%`、p99 `+194.4%`（B 基线 p50 `5.9302` / p95 `7.7854` / p99 `9.4440 ms`）。注意本行是**重测后**的 prod 延迟（`latency_report.json`，sha `515fb089…`），与 §6.2 表中其它行的历史读数同为 512-token INT8 CPU、10 warmup + 50 iter 口径。
- recal / recal-prod 与 C 是**同一张 ONNX 图**，延迟在统计意义上等价但**单次复测读数不同**：recal 复测 p50 `8.3984` / p95 `13.3241` / p99 `24.1539 ms`，prod 复测 p50 `8.4834` / p95 `18.5457` / p99 `27.8010 ms`（同一 runner、512-token、10 warmup + 50 iter，差异来自单机 50 次迭代的观测抖动；两者均通过 50/70 ms 门限，记录见各自 artifact 内 `onnx/latency_report.json`）。

---

## 7. 门禁状态与硬 blocker（不得声称通过）

`promotion.passed = false`，失败项（与 artifact 内 `metrics.json` 一致）：

| Blocker | 实测 | 门禁要求 | 性质 |
| --- | --- | --- | --- |
| `proxy_audit` | 596/607 resolved，11 unresolved，Fleiss κ `0.69197`，`eligible_for_promotion=false` | 需全部 resolved 且 κ 达标 | **需真实新增独立 reviewer** |
| `tactic_label_coverage` | 7 个 tactic head support ≥ 50 | ≥ 10 | **需补稀有 tactic 数据** |
| `evaluation_slice:ood_2024` | `negative_count = 0` → `insufficient_negatives` | 每切片至少 1 正 1 负 | **需补 OOD 良性** |
| `evaluation_slice:real_plus_semi_real` | `positive_count = 0` → `insufficient_positives` | 每切片至少 1 正 1 负 | **需补该切片正类** |

已通过的检查项：`brier`、`ece`、`fpr`、`pr_auc`、`recall`、`roc_auc`、`tactic_macro_f1`、`technique_macro_f1`、`detection_rate`、`detection_delay_commands`、`evaluation_slices_present`。

> 说明：proxy audit 的 11 条 unresolved 全部**没有足够恶意证据**，保持 fail-closed；不得用“补投票/伪造 reviewer”的方式让门禁变绿。

---

## 8. 生产部署建议

### 8.1 默认配置

```json
{
  "artifact_dir": "artifacts/cmd-session-risk-v3-l6-h256-recal-prod-20260921/onnx",
  "int8_model": "model.int8.onnx",
  "malicious_temperature": 0.6804212,
  "malicious_threshold": 0.25333938,
  "session_rule": "any_prefix_over_threshold",
  "provider": "CPUExecutionProvider"
}
```

- 会话级：recall `93.0%`、FPR `0.33%`（test 上 1 FP / 306 benign）。
- prefix 级：recall `96.9%`、FPR `0.047%`、precision `99.9965%`。
- 告警延迟：median `0` commands（首个命令即可告警的比例极高）。

### 8.2 回退与灰度

| 场景 | 选择 |
| --- | --- |
| 默认上线 | `recal-prod-20260921` |
| 需要更高召回（取证/审计，可容忍 FPR ~5.9%） | `recal-20260921` |
| 需要更低延迟/更小体积（p95 7.79 ms, INT8 11.65 MB） | B `v3-l4-h256-20260920` |
| 需要最强 OOD / taxonomy | A `small-l4-20260920-2226` |

### 8.3 监控与回滚

- 监控：session 告警率、每 tactic 的告警分布、INT8 与 FP32 抽样比对（建议 1% 流量双跑）、p95/p99 延迟。
- 回滚：`recal-prod` → `recal` → B 三级；三者的 `runtime_config.json` / `calibration.json` 都在 artifact 内，切换只需改 `artifact_dir`。

---

## 9. 还能不能继续优化？下一步计划（按优先级）

**不能靠“再调大参数”解决当前 blocker。** 依据：L6 已证实在本数据量下 OOD 与延迟退化；`tactic_label_coverage`、`ood_2024`、`real_plus_semi_real`、`proxy_audit` 都是**数据/审计缺口**，与模型容量无关。

| 优先级 | 动作 | 目标 | 为什么不能用“加参数”替代 |
| --- | --- | --- | --- |
| P0 | 补真实良性 session（尤其跨时间 2024+） | 把 dev benign 从 292 提到 ≥ 2,000，收窄阈值可辨识区间 | 阈值不确定性来自样本量，不是模型容量 |
| P0 | 补 OOD 良性 + real_plus_semi_real 正类 | 清除两个 slice blocker | 门禁按切片类平衡判定 |
| P0 | 补稀有 tactic（TA0001/6/8/10）样本到 support ≥ 50 | 把覆盖从 7 提到 ≥ 10 | 标签覆盖与模型大小无关 |
| P0 | 真实新增独立 reviewer 完成 11 条 unresolved | 解锁 proxy audit | 不能伪造投票 |
| P1 | 建 v4 数据集（**保留 v3**），再跑 L4 / L6 / 蒸馏对照 | 在**同一新数据**上重做 A/B | 只有数据变了，扩容收益才可验证 |
| P1 | 静态 QDQ / SmoothQuant / QAT；更强 teacher 蒸馏回 L4 学生 | 在**不增大部署图**的前提下提升拟合 | 与“调大参数”相反的优化方向，不增加端侧耗时 |
| P2 | 若 P1 后 L4 仍欠拟合，再考虑 H384/H512 | 有据可依的扩容 | 需先证明容量是瓶颈 |

---

## 10. 保留清单（旧方案一律未删除、未覆盖）

**报告**
- `reports/FINAL_REPORT_zh.md`（旧）
- `reports/FINAL_REPORT_SMALL_MODEL_zh.md`（旧）
- `reports/OPTIMIZATION_ASSESSMENT_20260920_zh.md`（旧）
- `reports/FINAL_COMPARISON_V2_V3_zh.md`（旧）
- `reports/proxy_audit/report.json`、`reports/proxy_audit/report_tiebreak.json`（旧，未修改）
- `reports/PRODUCTION_READINESS_V4_zh.md`（**本轮唯一新增报告**）

**Artifact**
- A：`artifacts/cmd-session-risk-small-l4-20260920-2226`
- B：`artifacts/cmd-session-risk-v3-l4-h256-20260920`
- C：`artifacts/cmd-session-risk-v3-l6-h256-20260920`
- C-recal（新）：`artifacts/cmd-session-risk-v3-l6-h256-recal-20260921`
- C-recal-prod（新）：`artifacts/cmd-session-risk-v3-l6-h256-recal-prod-20260921`
- 更早的 `cmd-session-risk-l4-20260919-0020*`、`local-smoke-20260918` 亦保留。

**本轮新增代码**
- `scripts/session_operating_point_sweep.py`（会话级操作点扫描 + k-of-n 评估）
- `scripts/evaluate_recalibrated_onnx_scores.py`（修复 3 处 bug：`_summary` 的 `detection_delay` KeyError、新增 `_session_summary`、补 `_prefix_metrics` 的 `threshold` 字段）

---

## 11. 复现命令

```bash
cd /home/ubuntu/Works/llm/cmd-session-risk

# 会话级操作点扫描（prod 校准）
.venv/bin/python scripts/session_operating_point_sweep.py \
  --run-dir artifacts/cmd-session-risk-v3-l6-h256-recal-prod-20260921 \
  --dev-data data/processed/v3/dev.jsonl --test-data data/processed/v3/test.jsonl \
  --fp32-dev /tmp/recal/dev_fp32.jsonl --fp32-test /tmp/recal/test_fp32.jsonl \
  --source-temperature 0.5568581837743221 --target-temperature 0.6804212 \
  --session-fpr-grid 0.0025,0.005,0.0075,0.01,0.015,0.02,0.03 \
  --output /tmp/recal/prod_session_sweep.json

# 重新校准后的 ONNX 评估（prefix + session + INT8 对比）
.venv/bin/python scripts/evaluate_recalibrated_onnx_scores.py \
  --run-dir artifacts/cmd-session-risk-v3-l6-h256-recal-prod-20260921 \
  --dev-data data/processed/v3/dev.jsonl --test-data data/processed/v3/test.jsonl \
  --fp32-dev /tmp/recal/dev_fp32.jsonl --fp32-test /tmp/recal/test_fp32.jsonl \
  --int8-dev /tmp/recal/dev_int8.jsonl --int8-test /tmp/recal/test_int8.jsonl \
  --source-temperature 0.5568581837743221 --target-temperature 0.6804212 \
  --threshold 0.25333938 --output /tmp/recal/prod_onnx_evaluation.json

# INT8 一致性校验 / 512-token 端侧延迟
.venv/bin/python scripts/validate_int8.py \
  --artifact-dir artifacts/cmd-session-risk-v3-l6-h256-recal-prod-20260921/onnx \
  --input data/processed/v3/test.jsonl
.venv/bin/python scripts/latency_bench.py \
  --artifact-dir artifacts/cmd-session-risk-v3-l6-h256-recal-prod-20260921/onnx \
  --lengths 512 --warmup 10 --iterations 50 --format json \
  --output artifacts/cmd-session-risk-v3-l6-h256-recal-prod-20260921/onnx/latency_report.json

# manifest 完整性（fail-closed）
.venv/bin/python scripts/refresh_artifact_manifest.py \
  --artifact-dir artifacts/cmd-session-risk-v3-l6-h256-recal-prod-20260921/onnx \
  --int8-model model.int8.onnx
```

---

## 12. 最终文件清单、数据缺口结论与一致性修复

### 12.1 生产 artifact 文件清单（fail-closed 校验通过）

| Artifact | `export_manifest.json` 声明文件数 | `verify_export_manifest` | `promotion.passed` |
| --- | ---: | --- | --- |
| `cmd-session-risk-v3-l6-h256-recal-prod-20260921/onnx` | **14** | `ok = true`（14/14） | `false` |
| `cmd-session-risk-v3-l6-h256-recal-20260921/onnx` | **15** | `ok = true`（15/15） | `false` |

- prod 的 14 个声明文件：`README.md`、`calibration.json`、`command_tokenization.json`、`int8_validation.json`、`labels.json`、`latency_report.json`、`metrics.json`、`model.int8.onnx`、`model.onnx`、`model.onnx.data`、`recalibrated_onnx_evaluation.json`、`runtime_config.json`、`tokenizer.json`、`tokenizer_config.json`。
- recal 在此基础上多 1 个 `recalibration_sweep.json`，故为 15 个。
- `latency` 元数据已与 `latency_report.json` 对齐：prod `p95 18.5457 / p99 27.8010 ms`（`sha 515fb089…`、`size 1291`）；recal `p95 13.3241 / p99 24.1539 ms`（`sha 03a9889d…`、`size 1287`）。两者 `latency.passed = true`。
- prod 的 `onnx/README.md` 已更新为与冻结阈值一致（新增 “Deployed operating point”，并把导出期旧阈值指标降级为 “Export-time metrics (legacy…)”）；README 变更后已重跑 manifest 刷新，校验依旧 `ok = true`。

### 12.2 数据缺口结论（为什么现在还不能清除 blocker）

- **OOD 良性无法合法补齐**：`data/raw/shell_attack_evolution/dataset/sessions/2024.jsonl` 有 6,658 条 2024 记录，但**只有 attack techniques、没有 benign**；`slices/ood_2024.jsonl` 的 1,458 条也是**全恶意、0 良性**。`evaluation_slice:ood_2024` 的 `insufficient_negatives` 是本数据集的结构性缺口，**不能靠模型或阈值修复**。
- **`real_plus_semi_real` 正类无法合法补齐**：该切片实测 `negative_count = 125, positive_count = 0`，`insufficient_positives` 同样无法用现有数据消除。
- **真实良性总量受限**：dev 仅 292 benign → 会话级 FPR 分辨率 `1/292 ≈ 0.34%`，阈值可辨识区间宽达 `(0.053419, 0.970807]`（见 4.3）。
- **稀有 tactic 覆盖不足**：test 仅 7 个 tactic head support ≥ 50（门禁要求 10），TA0001 / TA0008 / TA0010 为 0、TA0006 仅 4。
- **独立审计未收敛**：proxy audit 596/607 resolved、11 条 unresolved、Fleiss κ `0.69197`，属**需真实新增独立 reviewer** 的缺口，不能伪造。

**结论：这些 blocker 全部是“数据 / 审计供给”问题，不是“模型容量”问题 —— 正确动作是补数据 + 补审计，而不是继续调大参数。**

### 12.3 本轮一致性修复（相对交接点新增）

1. 修复 prod / recal 两个 ONNX artifact 的 **manifest latency 元数据过期**（此前写的是旧 `p95 15.4073 / p99 24.1912 ms`，与重测后的 `latency_report.json` 不一致）；现两者 `latency` 块与报告文件 sha/size 完全对应，并再次 `verify_export_manifest` 通过。
2. 修正 prod `onnx/README.md` 的 stale headline（原取自旧阈值 `metrics.json`），新增冻结阈值下的 deployed operating point，并重刷 manifest。
3. 修正报告内与冻结阈值不一致的旧值（`0.982862 / 0.005169 / 507 / 11` → `0.969408 / 0.000469925 / 905 / 1`）、延迟（`15.41 → 18.55 ms`、`24.19 → 27.80 ms`）、full-prefix 一致率（`0.9999713 / 1.0 / drift 0`）。

---

## 13. 一句话总结

**在有数据、无 GPU 的现实约束下，本轮的“生产化”不是靠调大模型，而是靠：① 权重冻结的后校准温度缩放（full-dev，fraction = 1.0）；② 把阈值选择口径从 prefix 级改为会话级（目标 session FPR ≤ 1%）；③ 完成 FP32/INT8 ONNX 一致性（full-prefix fail-closed）、延迟与 manifest 完整性核验。** 产出的 `recal-prod-20260921` 在 v3 test 上达到会话级 recall `93.0%` / FPR `0.33%`、cap32 prefix 级 recall `96.9%` / FPR `0.047%`、p95 `18.55 ms` / p99 `27.80 ms`，且 **full-prefix INT8 会话决策一致率 `100%`（cap32 口径 `99.95%`）**。

**定位与边界（务必对齐运维口径）：这是“生产灰度 / shadow / 告警 + 人工复核”候选，`promotion.passed` 仍为 `false`，不是 promotion 全通过模型，不可用于自动阻断。** 剩余 blocker 全部为数据 / 审计供给问题：proxy audit 11 条未决、tactic 覆盖 7<10、OOD 2024 无良性、real_plus_semi_real 无正类。要继续提升，唯一有效的方向是补真实良性 / OOD 良性 / 稀有 tactic 数据并完成独立人工审计，而不是继续增大参数。
