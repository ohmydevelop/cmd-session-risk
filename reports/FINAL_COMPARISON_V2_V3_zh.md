# Command Session Risk A/B/C 最终对比报告（v2 / v3）

- 日期：2026-09-21
- 范围：旧方案保留，新增 v3 数据方案与 L6-H256 小规模扩容方案，在同一训练流程与端侧部署约束下做 A/B/C 对比。
- 结论先行：**生产部署推荐 B（v3 + L4-H256）；A（v2 + L4-H256）作为 OOD/taxonomy 回退；C（v3 + L6-H256）不晋级，保留为扩容对照。**
- 重要限制：本轮按用户要求未创建 token budget，因此 **teacher/LLM proxy audit 未执行，不能报告为通过**。A/B/C 的模型、数据、ONNX、延迟和质量对比已完成，但 `eligible_for_promotion` 仍不能据此写成生产 promotion 通过。

---

## 1. 先回答：当前训练数据有多少、如何布局

### 1.1 数据文件格式

主数据是 JSONL，每行一个 session。核心字段如下：

| 字段 | 含义 |
| --- | --- |
| `session_id` | session 唯一标识 |
| `family_id` | 去重与切分使用的 family 标识 |
| `commands` | 命令序列，元素为 `{cmd, dt}` |
| `source` | `kind`（real/local/LLM variant 等）、来源名、许可证、父 session |
| `label` | `malicious`、`first_evidence_index`、`tactics`、`techniques`、质量与 teacher votes |
| `split` | `train` / `dev` / `test` |
| `meta` | 原始事件、skeleton、augmentation 等来源元数据 |
| `schema_version` | 数据 schema 版本 |

prefix 样本由每个 session 的命令前缀生成。本报告中的 `full prefixes` 是 session 全部合法前缀数；`cap32 prefixes` 是训练/导出实际使用的最多 32 个 prefix 数。

### 1.2 v2 / v3 主数据规模

| 数据集 | Split | Sessions | Families | Malicious / Benign | Full prefixes | cap32 prefixes | 单例 family |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| v2 | train | 4,000 | 3,184 | 2,182 / 1,818 | 68,917 | 56,696 | 2,448 |
| v2 | dev | 500 | 347 | 275 / 225 | 10,389 | 7,456 | 257 |
| v2 | test | 1,958 | 1,749 | 1,751 / 207 | 31,467 | 28,678 | 1,645 |
| v3 | train | 5,759 | 4,416 | 3,117 / 2,642 | 101,591 | 82,433 | 3,273 |
| v3 | dev | 721 | 510 | 429 / 292 | 14,242 | 10,541 | 379 |
| v3 | test | 2,178 | 1,916 | 1,872 / 306 | 34,870 | 31,711 | 1,757 |

v3 相对 v2 train：

- session：`+1,759`，约 `+44.0%`；
- family：`+1,232`，约 `+38.7%`；
- full prefix：约 `+47.4%`；
- 实际 cap32 prefix：约 `+45.4%`。

### 1.3 v3 train 的组成

| 来源 kind | Sessions |
| --- | ---: |
| real malicious | 2,258 |
| real benign | 2,229 |
| LLM variant | 859 |
| hard negative | 296 |
| script benign | 117 |
| **合计** | **5,759** |

v3 主数据流水线为：

`原始池 21,194 sessions -> exact dedup 12,029 -> family-aware dedup 9,758 -> 配额/过滤后 8,658`。

其中 `8,658 = train 5,759 + dev 721 + test 2,178`；另有 OOD 2024 `1,458` 和 synthetic stress `1,440` 作为评估/压力切片。

### 1.4 公共评估集与压力集

| 数据集 | Sessions | Families | Malicious / Benign | Full prefixes | cap32 prefixes | 说明 |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| common clean | 3,927 | 3,774 | 3,803 / 124 | 96,236 | 75,394 | 二分类主公共集；benign 仅 124，FPR 必须给 Wilson CI |
| OOD 2024 | 1,458 | 1,281 | 1,458 / 0 | 21,052 | 20,959 | 仅恶意，只报告 recall / detection rate / delay |
| synthetic stress v3 | 1,440 | 1,381 | 1,080 / 360 | 31,782 | 26,035 | 压力切片，与主数据有重合，不作为独立泛化证明 |

### 1.5 切分隔离

- v2 内部以及 v3 内部的 `train/dev/test`：session overlap = `0`，family overlap = `0`。
- common clean 与 v2/v3 的 train、dev：session overlap = `0`，family overlap = `0`。
- common clean 与 v2 test 有 44 个 session、43 个 family 重合；与 v3 test 有 202 个 session、195 个 family 重合。因此 common clean 不能表述为相对主 test 的完全独立新 holdout。
- OOD 2024 与 v2/v3 train、dev 无 session/family 重合；其 1,458 个 session 同时出现在主 test 集合中，是时间切片评估，不应重复当作第二个完全独立测试集。
- synthetic stress 与 v3 train 有 1,155 个 session 重合，因此只用于压力分析，不用于宣称未见分布泛化。

### 1.6 数据量是否太少

**对当前二分类任务：现有数据足够支撑 L4-H256，继续盲目加参数的边际收益很低。**

依据：

- v3 有 5,759 个训练 session、4,416 个 family、82,433 个实际 cap32 prefix 样本；主集指标已经接近饱和。
- A/B 在 clean full 的 recall 都是 `0.99658`；C 增加到 12.76M 参数后，clean recall 仍完全相同。
- C 的 train/eval loss 更低，但 OOD recall 反而从 B 的 `0.99040` 降到 `0.97531`，说明更低训练损失没有转化为更好的跨时间泛化。

**对多标签、稀有 tactic/technique、跨来源/跨时间 OOD 和生产泛化：数据仍然偏少。**

- common clean 只有 124 个 benign，FPR 的 Wilson 95% CI 很宽；例如 B 的点估计为 `3.23%`，CI 为 `[1.26%, 8.00%]`。
- OOD 2024 没有 benign，FPR、ROC-AUC、PR-AUC 在该切片上未定义，不能计算后当作通过证据。
- tactic 标签覆盖不足：common clean 中只有 8 个 tactic head 的 support `>= 50`，OOD 中只有 7 个；多个 tactic 为 0 或极少样本。
- 主要缺口不是“总 session 再复制一批”，而是缺少真实攻击链、真实良性会话、跨时间/跨来源样本、稀有 tactic/technique 样本和可靠 hard negatives。

因此下一步优先补数据与审计覆盖，而不是继续把模型从 4 层推到 6 层或更大。

---

## 2. A/B/C 实验定义

| 方案 | 数据 | 模型初始化 | Artifact |
| --- | --- | --- | --- |
| A | v2 | `google/bert_uncased_L-4_H-256_A-4` | `artifacts/cmd-session-risk-small-l4-20260920-2226` |
| B | v3 | `google/bert_uncased_L-4_H-256_A-4` | `artifacts/cmd-session-risk-v3-l4-h256-20260920` |
| C | v3 | `google/bert_uncased_L-6_H-256_A-4` | `artifacts/cmd-session-risk-v3-l6-h256-20260920` |

共同训练设置：

- short sequence：512 token；
- command token cap：128；
- prefix cap：32；
- seed：42；
- epoch：3；
- `skip_long_phase=true`；
- 所有方案均重新训练并导出，没有复用或覆盖旧 artifact。

---

## 3. 参数量与训练成本

| 方案 | Checkpoint 参数 | ONNX initializer 元素 | 参数相对 A/B | 训练时间 | Train loss | Eval loss |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| A | 11,180,598 | 11,115,822 | 1.000x | 575.3 s | 0.3764 | 0.1396 |
| B | 11,180,598 | 11,115,822 | 1.000x | 983.4 s | 0.3004 | 0.1244 |
| C | 12,760,118 | 12,695,342 | 1.141x | 1,232.0 s | 0.2640 | 0.1124 |

结论：

- C 参数比 A/B 增加 `1,579,520`，约 `+14.13%`。
- B 因为使用更多 v3 训练 prefix，训练时间相对 A 增加约 `70.9%`。
- **C 相对 B 训练时间增加约 25.3%，不是零退化。**
- C 的 loss 最低，但 loss 不是部署质量指标；C 的 OOD、INT8 一致性和延迟均出现退化。

---

## 4. ONNX 导出、大小与数值一致性

### 4.1 ONNX 产物

| 方案 | Opset | FP32 graph | FP32 external data | FP32 合计 | INT8 ONNX |
| --- | ---: | ---: | ---: | ---: | ---: |
| A | 18 | 44,964,039 B | 44,505,088 B | 89,469,127 B | 11,652,743 B |
| B | 18 | 44,960,694 B | 44,505,088 B | 89,465,782 B | 11,649,977 B |
| C | 18 | 51,492,948 B | 50,796,544 B | 102,289,492 B | 13,459,220 B |

C 的 INT8 比 B 大 `1,809,243 B`，约 `+15.53%`。B 的 INT8 比 A 小 `2,766 B`，可视为同等大小。

### 4.2 PyTorch / FP32 / INT8 数值误差

| 方案 | FP32 max abs error | FP32 atol | INT8 max abs error | INT8 atol |
| --- | ---: | ---: | ---: | ---: |
| A | 2.1458e-6 | 1e-4 | 0.0186222 | 0.05 |
| B | 2.3842e-6 | 1e-4 | 0.0162776 | 0.05 |
| C | 2.3842e-6 | 1e-4 | 0.0257711 | 0.05 |

三者均低于对应 parity 阈值。INT8 数值误差 C 最大，但仍处于导出门限内。

---

## 5. 公共集质量对比

以下 `clean full` 指标来自各自 artifact 的部署 INT8 模型。`Tactic F1` 使用 support `>= 50` 的 head；`Technique F1` 使用 12 个已支持 head。

### 5.1 common clean full：二分类与概率质量

| 方案 | Recall | FPR（Wilson 95% CI） | FP / FN / TN | PR-AUC | ROC-AUC | Brier | ECE |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: |
| A | 0.996582 | 0.048387 `[0.022362, 0.101552]` | 6 / 13 / 118 | 0.99999840 | 0.99978372 | 0.0012593 | 0.0012752 |
| B | 0.996582 | 0.032258 `[0.012615, 0.080012]` | 4 / 13 / 120 | 0.99999888 | 0.99988374 | 0.0013858 | 0.0018953 |
| C | 0.996582 | 0.024194 `[0.008262, 0.068720]` | 3 / 13 / 121 | 0.99999881 | 0.99987632 | 0.0011359 | 0.0018600 |

观察：

- A/B/C recall 完全相同。
- B 相对 A 将 clean FPR 从 `4.84%` 降到 `3.23%`，FP 从 6 降到 4。
- C 的 clean FPR 最低，但 OOD 退化和端侧成本更大，不能只按 clean FPR 选择。

### 5.2 common clean full：taxonomy

| 方案 | Tactic F1（support >= 50） | Technique F1 |
| --- | ---: | ---: |
| A | 0.823947 | 0.941702 |
| B | 0.749458 | 0.895582 |
| C | 0.755950 | 0.897054 |

B/C 的二分类 clean recall 与 A 相同，但 taxonomy F1 低于 A。A 在 taxonomy 上仍是最强回退方案。

### 5.3 common clean cap32

| 方案 | Recall / FPR | PR-AUC | ROC-AUC | Brier | ECE | Tactic F1（support >= 50） | Technique F1 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| A | 0.996582 / 0.048387 | 0.99999784 | 0.99972110 | 0.0015942 | 0.0016144 | 0.832679 | 0.958068 |
| B | 0.996582 / 0.032258 | 0.99999837 | 0.99985212 | 0.0017445 | 0.0022176 | 0.762192 | 0.913949 |
| C | 0.996582 / 0.024194 | 0.99999815 | 0.99983232 | 0.0014374 | 0.0021411 | 0.766760 | 0.913545 |

cap32 与 full 的结论一致：A 的 taxonomy 更强；B 和 C 的 clean FPR 更低；C 未形成足以抵消 OOD 与端侧退化的质量优势。

### 5.4 OOD 2024

OOD 2024 只有恶意 session，因此这里只报告 recall、detection rate、Wilson CI、FN 与 delay。**不报告 FPR、ROC-AUC、PR-AUC**，因为缺少负类。

| 方案 | Recall / detection rate | Wilson 95% CI | FN | Max detection delay |
| --- | ---: | --- | ---: | ---: |
| A | 0.991770 | `[0.985669, 0.995286]` | 12 | 8 |
| B | 0.990398 | `[0.983947, 0.994272]` | 14 | 8 |
| C | 0.975309 | `[0.966007, 0.982112]` | 36 | 27 |

| 方案 | Tactic F1（support >= 50） | Technique F1 |
| --- | ---: | ---: |
| A | 0.637755 | 0.699376 |
| B | 0.576735 | 0.585791 |
| C | 0.738530 | 0.686207 |

OOD cap32 与 full 的 recall、FN、delay 基本相同。C cap32 为 recall `0.975309`、FN `36`、max delay `27`、tactic F1 `0.739873`、technique F1 `0.687449`。A/B/C 的 median 和 p90 delay 都是 0。

关键结论：

- B 的 OOD recall 仅比 A 低 `0.1372` 个百分点，FN 从 12 变为 14，仍保持 `99.04%`。
- C 的 OOD recall 比 B 低 `1.5089` 个百分点，FN 从 14 增至 36，最大 delay 从 8 增至 27。
- C 的 OOD tactic F1 较高，但 detection recall 和 delay 退化更接近生产风险；不能因此推荐 C。

---

## 6. INT8 一致性与端侧延迟

### 6.1 INT8 一致性

| 方案 | 测试集 | Prefix agreement | Session agreement | ΔRecall（INT8-FP32） | ΔFPR | First-alert drift max / mean / sessions | Prefix prob delta max / mean |
| --- | --- | ---: | ---: | ---: | ---: | --- | --- |
| A | v2/test | 0.99879239 | 0.99744637 | +0.0022844 | +0.0048309 | 22 / 0.021677 / 10 | 0.983334 / 0.000475 |
| B | v3/test | 0.99790651 | 0.99908173 | +0.0010684 | 0 | 21 / 0.015475 / 8 | 0.366719 / 0.005321 |
| C | v3/test | 0.99761973 | 0.99540863 | +0.0016026 | +0.0032680 | 132 / 0.096529 / 18 | 0.316130 / 0.000511 |

说明：

- 绝对 delta 只比较同一模型同一测试集的 FP32/INT8，不用于横向比较不同测试集上的模型质量。
- C 的 session agreement 低于 B，首个告警位置最大漂移 `132` 个 prefix，明显高于 A/B。
- A 的最大单 prefix 概率漂移较大，但 prefix/session decision agreement 仍高，且 first-alert drift 较小。

### 6.2 512-token INT8 CPU 延迟

环境：Intel Core Ultra 7 270K Plus、10 logical CPU、AVX2、x86_64、ONNX Runtime `1.30.0`、`CPUExecutionProvider`、`ORT_ENABLE_ALL`；512 token、batch size 1。

| 方案 | p50 | p95 | p99 | p95 门限 | p99 门限 | 结果 |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| A | 6.3302 ms | 8.2984 ms | 10.6846 ms | 50 ms | 70 ms | 通过 |
| B | 5.9302 ms | 7.7854 ms | 9.4440 ms | 50 ms | 70 ms | 通过 |
| C | 8.9337 ms | 15.4073 ms | 24.1912 ms | 50 ms | 70 ms | 通过门限但有退化 |

C 相对 B：

- p50 增加约 `50.7%`；
- p95 增加约 `97.9%`；
- p99 增加约 `156.2%`。

因此 C 虽然仍低于 50/70 ms 门限，但不满足“时间不损耗”的要求。

---

## 7. 最终对比与推荐

### 7.1 方案排序

1. **B：v3 + L4-H256，生产推荐。**
   - clean recall 与 A 相同；
   - clean FPR 更低：`3.23%` 对 `4.84%`；
   - 512-token 延迟最快，INT8 大小与 A 基本一致；
   - OOD recall 仍有 `99.04%`，相对 A 只多 2 个 FN。
   - 代价：OOD taxonomy 和 clean taxonomy F1 低于 A，这是推荐 B 时需要明确的残余风险。

2. **A：v2 + L4-H256，OOD/taxonomy 回退。**
   - OOD recall 最强：`99.18%`；
   - OOD tactic/technique F1 和 clean taxonomy F1 最好；
   - 512-token 延迟略慢于 B，clean FPR 高于 B。
   - 当跨时间 recall 或 taxonomy 是不可妥协指标时，选择 A，而不是 C。

3. **C：v3 + L6-H256，不晋级。**
   - 参数增加约 `14.13%`；
   - 训练时间相对 B 增加约 `25.3%`；
   - INT8 大小相对 B 增加约 `15.53%`；
   - OOD recall 下降 `1.51` 个百分点，FN 增至 36，max delay 增至 27；
   - INT8 first-alert drift 增大；
   - p50/p95/p99 明显变差。
   - 保留 C 作为“小规模扩容是否有收益”的对照证据，但不替换 B。

### 7.2 是否还能继续优化

可以，但“再调大参数”不是当前最优优化方向。更准确的结论是：

- 当前 L4-H256 的二分类容量不是主要瓶颈；
- L6-H256 已经证明“参数稍大”会带来训练、INT8、延迟和 OOD 风险，且没有带来 clean recall 增益；
- 继续扩大到 H512/L8 或引入更复杂架构，在当前数据和门禁条件下缺乏收益保证，不应作为下一轮默认动作；
- 更高优先级是数据、校准、切片定义和审计覆盖。

后续建议顺序：

1. **P0：补真实数据分布。**
   - 真实攻击链、真实良性会话、跨时间/跨来源样本；
   - 稀有 tactic/technique 样本，至少让目标 head 在独立评估集中有统计有效 support；
   - 重新建立不重叠的 temporal/source holdout。
2. **P0：修复评估和 promotion。**
   - 为 OOD slice 增加 benign，或明确该 slice 只做 recall/delay；
   - 解决 ECE 与 tactic coverage 门禁；
   - 按用户约束补齐 proxy audit 时，另开 token budget，并保留本轮未执行的事实。
3. **P1：仅在新增独立数据后做模型优化 A/B。**
   - 候选包括后校准、static QDQ / SmoothQuant / QAT、更强 teacher 蒸馏回 L4 学生；
   - 每种候选都必须保持本报告 A/B/C 作为 control，单独记录训练时间、INT8 大小、parity、OOD 和安全门禁；
   - 若要求训练与推理时间都不增，最合理的是保持 L4 部署图，只做数据、校准和蒸馏研究，而不是继续增大部署模型。

---

## 8. 验证状态、合规性与保留项

- FP32 ONNX、INT8 ONNX、parity、INT8 一致性、512-token latency 均已生成并复核。
- 各方案 latency 技术阈值均通过；C 的 p50/p95/p99 相对 B 明显退化。
- 各方案 manifest 的自然 `promotion` 状态仍为 `false`，失败项包括 evaluation slices / proxy audit 等门禁；**不能把本轮结果写成 promotion gate 全通过。**
- 按用户要求未创建 token budget，因此 **teacher/LLM proxy audit 未执行**；`proxy_audit` 不能报告为通过。
- 旧方案、旧 artifact、旧报告及历史对照全部保留，没有删除或覆盖：
  - `reports/FINAL_REPORT_zh.md`
  - `reports/FINAL_REPORT_SMALL_MODEL_zh.md`
  - `reports/OPTIMIZATION_ASSESSMENT_20260920_zh.md`
  - A artifact：`artifacts/cmd-session-risk-small-l4-20260920-2226`
  - B artifact：`artifacts/cmd-session-risk-v3-l4-h256-20260920`
  - C artifact：`artifacts/cmd-session-risk-v3-l6-h256-20260920`
- 本报告是本轮唯一新增的最终对比报告。

## 9. 最终一句话

**B（v3 + L4-H256）是当前最佳部署选择；A（v2 + L4-H256）是 OOD/taxonomy 回退；C（v3 + L6-H256）虽然稍大且 loss 更低，但在 OOD、INT8 一致性和延迟上没有满足“不损耗性能以及时间”，不晋级。数据层面，二分类已够用，多标签、稀有标签和跨时间/跨来源泛化仍明显不足，下一步应优先补数据而不是继续盲目增大参数。**
