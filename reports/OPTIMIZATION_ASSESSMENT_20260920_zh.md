# Command Session Risk 最新优化评估（截至 2026-09-20）

- 评估对象：`cmd-session-risk-small-l4-20260920-2226`
- 当前基座：`google/bert_uncased_L-4_H-256_A-4`
- 当前参数量：`11,180,571`
- 当前 ONNX INT8：`11,652,743 B`（约 `11.65 MB`）
- 当前独立 FP32 验证：Recall `0.989212`、FPR `0.010507`、PR-AUC `0.999806`
- 当前同机 CPU INT8 512-token：p95 `8.30 ms`、p99 `10.69 ms`
- 延迟门禁：p95 `<= 50 ms`、p99 `<= 70 ms`
- 当前 promotion：未通过；失败项为 ECE、tactic coverage、proxy audit、部分 evaluation slice

## 0. 结论

**可以继续优化，但没有一种已公开技术能够严格保证“参数变大、质量不降、训练时间不变、推理时间不变”。**

当前模型的主要问题已经不是参数量：

1. held-out 的 Recall、FPR、PR-AUC 已接近饱和；继续扩容对主指标的可提升空间很小。
2. promotion 失败集中在 ECE `0.0606 > 0.05`、tactic coverage `7/10`、proxy audit `0/500`，以及 `real_plus_semi_real`、`ood_2024` 等切片定义/覆盖问题。
3. 当前 CPU 延迟余量很大，所以“在门禁内做一次更宽的模型”可行，但这不等于原始 p95 不增加。
4. 若严格要求部署延迟完全不变，最合理路线不是直接放大模型，而是：
   - 先做后校准、数据覆盖和审计修复；
   - 若必须增加容量，用更大 teacher 蒸馏回当前 H256 学生；
   - 或在研究阶段验证 early-exit / elastic subnet，不作为本轮生产默认方案。

## 1. 当前基线

| 项目 | 当前值 |
| --- | ---: |
| Backbone | L4 / H256 / FFN 1024 / A4 |
| 参数 | 11.18 M |
| INT8 ONNX | 11.65 MB |
| 训练 | 3 epochs，约 575.3 s |
| FP32 独立 Recall | 0.989212 |
| FP32 独立 FPR | 0.010507 |
| FP32 独立 PR-AUC | 0.999806 |
| INT8 512-token p50 | 4.68 ms（专项探针） |
| INT8 512-token p95 | 8.30 ms（交付基准） |
| INT8 512-token p99 | 10.69 ms（交付基准） |
| INT8 与 FP32 prefix 决策一致率 | 0.998792 |

当前延迟报告还使用了 ONNX Runtime `1.30.0`、`CPUExecutionProvider` 和 `ORT_ENABLE_ALL`。在该机器上，线程数从默认值调到 4/8 会明显影响尾延迟；8 线程专项探针约为 p50 `5.38 ms`、p95 `7.68 ms`、p99 `9.46 ms`，说明部署线程配置本身已有约 10% 级别的优化空间。

## 2. 扩容候选的实测边界

下面的数字来自同一台 Intel CPU、动态 batch/sequence ONNX、512 token、batch size 1、FP32 导出后动态 INT8、200 次迭代的随机权重结构探针。它们只用于容量/延迟排序，**不能替代真实训练后的交付验证**。

| 候选 | 参数（相对当前） | INT8 大小 | p50 | p95 | p99 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 当前 L4-H256-I1024-A4 | 11.18 M（1.00x） | 11.72 MB | 8.40 ms | 12.25 ms | 15.77 ms |
| L4-H320-I1024-A5 | 14.32 M（1.28x） | 14.83 MB | 10.53 ms | 13.85 ms | 24.49 ms |
| L4-H320-I1280-A5 | 14.98 M（1.34x） | 15.50 MB | 10.74 ms | 15.60 ms | 26.30 ms |
| L4-H384-I1024-A6 | 17.60 M（1.57x） | 18.07 MB | 12.91 ms | 19.05 ms | 25.40 ms |
| L4-H384-I1536-A6 | 19.18 M（1.72x） | 19.67 MB | 13.69 ms | 23.13 ms | 31.48 ms |
| L4-H512-I1024-A8 | 24.59 M（2.20x） | 24.94 MB | 16.93 ms | 23.71 ms | 29.08 ms |
| L4-H512-I2048-A8（对应官方 H512 配置） | 28.78 M（2.57x） | 29.19 MB | 18.54 ms | 26.15 ms | 30.36 ms |
| L6-H256-I1024-A4 | 12.76 M（1.14x） | 13.56 MB | 13.02 ms | 21.19 ms | 28.64 ms |
| L8-H256-I1024-A4 | 14.34 M（1.28x） | 15.40 MB | 16.54 ms | 21.80 ms | 27.51 ms |

上述探针的 p95 仍低于 50 ms，但有两个重要限制：

- 随机权重探针的绝对时间和量化行为可能与真实模型不同；候选必须重新导出、重新跑 parity 和真实 latency。
- 延迟“有余量”不等于训练时间“不增加”。深度增加尤其会造成每层调度/算子开销，L6-H256 仅增加约 14% 参数，却在该探针中把 p95 推向约 1.7x。

### 候选选择

1. **最稳妥的官方预训练扩容**：`google/bert_uncased_L-4_H-512_A-8`
   - 2.57x 参数，官方预训练，初始化风险最低。
   - 探针 p99 仍远低于 70 ms，但训练时间预计约为当前的 2.5x 到 3x，不能承诺训练时间不变。
   - 更适合做一次性质量上限实验，而不是“略微增大”。

2. **最接近“略微增大”的官方深度候选**：`google/bert_uncased_L-6_H-256_A-4`
   - 1.14x 参数，官方预训练。
   - 深度导致延迟和训练步时增长高于参数比例，且质量收益没有保证；不建议仅因参数变大而选择。

3. **容量/延迟折中候选**：自定义 `L4-H320-I1024-A5`
   - 1.28x 参数，探针延迟仍在门禁内。
   - 没有对应的官方 BERT 预训练权重，需要从 H256 权重扩展或从头中小规模预训练；引入初始化、收敛和泛化风险。
   - 只有在前两种真正跑完 A/B 后仍有余量，才值得尝试。

## 3. 2026-09-20 时点的最新可落地技术

### A. 立即可落地，不需要训练

| 技术 | 状态 | 对本项目的判断 |
| --- | --- | --- |
| ONNX Runtime `1.30.0` | 2026-09-10 发布，项目已使用 | 保留；包含 CPU fused LinearAttention、AVX2 LayerNorm/RMSNorm 等优化 |
| ONNX `1.23.0` | 2026-09-18 发布，本地当前为 1.22.0 | 可做一次导出/兼容性回归，不应直接替换已验证 artifact |
| OpenVINO `2026.4.0` | 2026-09-16 发布，当前未安装 | Intel CPU 上值得单独 benchmark；必须同时验证数值、INT8 parity 和 p99 |
| ORT CPU 线程固定为 4 或 8 | 已完成探针 | 8 线程在当前机器上尾延迟最优；部署端需要固定，否则默认线程数会引入离散波动 |
| 后校准（temperature/isotonic/vector scaling） | 可用，无需改模型 | 对 ECE 最直接的零推理成本修复；必须在 family-disjoint 的 dev 上拟合，不能使用 test |
| 数据/词表/审计修复 | 最高收益 | tactic coverage、proxy audit 和切片问题不是扩容问题；增加参数不会创造缺失标签 |

### B. 可让“更大模型”更接近同等时间的技术

| 技术 | 收益 | 风险 |
| --- | --- | --- |
| `torch.compile` + fused AdamW | 对训练可能有 10% 到 30% 的加速，幅度取决于 GPU 和小模型开销 | 首次编译耗时，BERT 小模型可能被编译/调度开销吃掉收益；必须 A/B |
| SDPA / FlashAttention 类高效 attention | 512 token 下可降低 attention 开销 | 当前瓶颈不一定在 attention；需验证导出后的算子与量化兼容性 |
| BF16/FP16 | 项目已自动启用 | 当前 bf16 已使用，不能重复计收益 |
| INT8 静态 QDQ、SmoothQuant、TorchAO | 论文报告可在小 NLP CPU 模型上获得显著吞吐提升 | 当前动态 INT8 已通过 parity；静态量化需要新 calibration 流程，可能改变尾延迟和校准 |
| OpenVINO CPU EP / OpenVINO 原生编译 | 可能用更成熟的 CPU kernel 换取延迟余量 | 需要新 runtime 依赖、版本锁定和全量 parity，不应在未验证时作为生产替代 |

`Efficient INT8 Inference of Small NLP Models on Server CPUs with PyTorch Native Stack`（arXiv `2608.18182`，2026-08-18）报告 SmoothQuant + TorchAO + TorchInductor/oneDNN 可获得最高约 5.8x 吞吐且基本无精度损失。该结论针对其评测栈，不能直接外推到当前 ONNX Runtime 动态 INT8；最多作为下一轮 CPU 后端的实验依据。

### C. 研究型/弹性计算技术

| 方向 | 代表工作 | 为什么暂不建议直接生产 |
| --- | --- | --- |
| 弹性宽度/深度/embedding | m3BERT（arXiv `2605.19568`）、Tevatron-Elastic（arXiv `2608.08809`） | 一个 checkpoint 提供多档子网络，适合检索/多预算部署；需要重训和自定义导出，当前多标签分类收益未证明 |
| 输入自适应分辨率/宽度 | ProgResViT（arXiv `2609.03216`） | 面向视觉 Transformer，和固定 512-token 命令编码路径不匹配 |
| 运行时可弹性转换 | Elastoformer（arXiv `2609.10018`） | 论文主要是视觉 Transformer/CNN，且动态分支会给 ONNX CPU/INT8 带来较大风险 |
| Layer dropout / early exit | Don't Drop Dropout（arXiv `2609.05275`）、NAG（arXiv `2606.16112`） | 可能让更大最大深度获得接近小模型的平均计算；但动态跳过、校准和 ONNX 控制流复杂，本轮不做生产主路径 |
| MoD / depth-KV attention | MoDA（arXiv `2603.15619`） | 面向 decoder LLM，当前是 encoder 多标签分类，不适合直接迁移 |
| MoE / 路由 | MoRE（arXiv `2609.18176`）、TriRoute（arXiv `2607.06601`） | ONNX CPU、量化、动态路由和数据规模都不匹配；`Data Scarcity and Model Sparsity`（arXiv `2609.11917`）还指出重复数据下 MoE 更容易过拟合 |
| 量化感知蒸馏 | EdgeRazor（arXiv `2605.04062`）、边缘 VLM QAD（arXiv `2609.16689`） | 更适合边缘 VLM；当前 INT8 parity 和延迟都已经通过，不是优先瓶颈 |
| 权重投影扩容 | Wiring Beats Blending（arXiv `2608.02829`） | 说明直接 blending 会破坏结构，least-squares/初始化补偿更可靠，但仍需继续预训练 |
| 渐进式容量增长 | Successive Capacity Growth（arXiv `2608.27367`） | 目前证据主要在视觉 JEPA，不能作为 BERT 生产扩参依据 |

### D. 如果严格要“训练时间也不增加”

严格意义下只有三种可能：

1. 用明显更快的 GPU 或更高效的 kernel 抵消扩容成本；这是硬件/系统收益，不是模型免费扩容。
2. 用更大 teacher 蒸馏到当前 H256 学生：训练时间增加，但部署架构、参数量和推理延迟不变。
3. 用 elastic/early-exit 让平均激活计算接近当前模型：最大参数可增大，但输入相关延迟、ONNX 导出和校准复杂度都会增加。

对当前任务，方案 2 比 3 更稳妥：先在离线侧用官方 H512 或自定义 H320 作 teacher，只把收益蒸馏到已经验证过的 H256 部署图。这样能保持部署速度，但需要额外训练预算，且不能用 test 集调参。

## 4. 建议的实际优化顺序

### 第一步：先做零重训修复（优先级最高）

- 在独立 dev 上重做 ECE 后校准，目标 `ECE <= 0.05`，保持 FP32/INT8 排序与阈值行为不变。
- 补齐 tactic coverage 到至少 10 个有效 head，并明确哪些标签是真实缺失而不是模型失败。
- 完成 500 个 resolved proxy audit；该项不能靠扩大模型解决。
- 修复 `real_plus_semi_real`、`ood_2024` 的正负样本和切片定义。
- 在固定 4/8 CPU 线程下重跑 latency，不需要重新训练。

### 第二步：只做一个受控扩容 A/B

如果质量和训练预算允许，建议只选一个：

- **质量上限优先**：`google/bert_uncased_L-4_H-512_A-8`，使用官方预训练权重，新 model ID、新 remote root、新 output dir。
- **略微扩容优先**：`google/bert_uncased_L-6_H-256_A-4`，但必须接受深度带来的延迟增长。

两者都保持：

```text
--max-epochs 3
--skip-long-phase
--short-max-length 512
--command-max-tokens 128
--max-prefixes 32
```

命令骨架（GPU 可用后执行，不能复用旧 artifact 目录）：

```bash
scripts/train_remote.sh \
  --host "$LIGHTNING_SSH_HOST" \
  --remote-root "/teamspace/studios/this_studio/cmd-session-risk-l4-h512-20260920-XXXX" \
  --data-dir data/processed/v2 \
  --output-dir artifacts/cmd-session-risk-small-l4-h512-20260920-XXXX \
  --detach \
  -- --model-name google/bert_uncased_L-4_H-512_A-8 \
       --max-epochs 3 --skip-long-phase \
       --short-max-length 512 --command-max-tokens 128 \
       --max-prefixes 32
```

### 第三步：扩容验收门禁

候选只有同时满足以下条件，才可替换当前 artifact：

- FP32 ONNX parity `<= 1e-4`；INT8 parity `<= 0.05`。
- 512-token CPU p95 `<= 50 ms`、p99 `<= 70 ms`。
- 独立 held-out Recall 不低于当前 `0.989212`，FPR 不高于 `0.010507`，PR-AUC 不低于 `0.999806`。
- ECE 目标 `< 0.05`，且不高于当前独立 FP32 验证的 `0.054809`。
- tactic/technique macro-F1、detection rate 和 detection delay 不退化。
- `real_only`、`real_plus_semi_real`、`synthetic_stress`、`ood_2024` 切片均有有效正负样本并通过。
- proxy audit resolved cases `>= 500`。
- 训练墙钟时间单独记录；若不能接受训练变慢，需要额外 GPU/编译收益说明，不能口头假设。

## 5. 对当前请求的直接判断

- **能否优化**：能，但最高优先级是 calibration、coverage、audit 和 slice，而不是增加参数。
- **能否稍微调大参数**：技术上可以，`L4-H320` 是最接近“略微增大”的候选；若必须使用官方预训练，则 `L4-H512` 最稳，但参数量增加约 2.57x。
- **能否完全不影响性能和时间**：没有可验证的通用保证。当前推理延迟余量允许更大模型继续留在门禁内，但原始 p95/p99 和训练时间都会增加。
- **当前最推荐的生产前实验**：先完成零重训校准/审计修复；若要扩容，只选官方 H512 或官方 L6-H256 做一次受控 A/B，不复用旧 artifact，不绕过 promotion gate。

## 6. 当前执行状态

本轮已现场核验 Lightning SSH endpoint：连接成功，远端 `nvidia-smi` 可见 1 张 NVIDIA L4。此前“Studio 未运行/无法训练”的阻塞已解除。

本推荐方案刻意将扩容放在容量证据门禁之后，因此本轮没有启动 L6-H256 训练，也没有修改或覆盖现有交付 artifact。第一阶段的校准、切片和审计修复可立即执行；只有通过容量不足判定后，才启动第二阶段的唯一一次 L6-H256 受控实验。

## 7. 指定推荐执行方案

### 7.1 方案名称

**Stabilize-Then-Scale：先稳定 H256，再做一次 L6-H256 单点扩容。**

该方案的核心判断是：当前模型的主指标已经接近饱和，直接扩大到 H512 会显著增加训练时间、过拟合风险和部署差异，收益却无法保证。先用零重训手段解决 ECE、覆盖和审计问题，再用官方预训练的 L6-H256 做小幅容量实验，是收益、风险和训练预算最平衡的路径。

### 7.2 第一阶段：稳定当前 H256（现在执行，不训练新模型）

目标：保留当前部署图、参数量和推理延迟，只修复 promotion 失败项。

1. **CPU 后校准**
   - 使用 family-disjoint 的 dev/calibration 数据重新拟合 malicious 后校准。
   - 优先顺序：现有 temperature scaling → vector scaling → isotonic 或分桶校准。
   - 目标：ECE `< 0.05`，且不高于当前独立 FP32 验证的 `0.054809`。
   - 硬约束：阈值重新按 dev 目标 FPR 生成；Recall、FPR、PR-AUC、detection rate 不得低于当前独立验证。
   - 推理图、参数量和 ONNX 输入输出保持不变。

2. **数据与门禁修复**
   - 补齐 tactic coverage：至少 10 个有效 tactic head，而不是只依赖当前实测的 7 个有足够支持的 head。
   - 完成 `>= 500` 个 resolved proxy audit case。
   - 修复 `real_plus_semi_real`、`ood_2024`、`synthetic_stress` 的正负样本定义和覆盖不足。
   - 对缺失标签建立“数据缺失”而不是“模型失败”的明确审计记录。

3. **部署配置冻结**
   - ONNX Runtime 固定为 `1.30.0`。
   - 固定 CPU 线程数：以 4 线程和 8 线程做一次最终对比，选择 p99 更稳定的设置；当前证据倾向 8 线程。
   - ONNX `1.23.0` 和 OpenVINO `2026.4.0` 只做旁路 benchmark，不替换已验证 artifact。
   - 保存 FP32/INT8 parity、p50/p95/p99、CPU 型号和线程配置。

### 7.3 第二阶段：唯一允许的扩容实验

只有在第一阶段完成后，且出现以下任一证据时，才启动本阶段：

- 后校准仍无法把 ECE 降到 `0.05` 以下；或
- tactic/technique macro-F1 明显低于目标，且分析确认是模型欠拟合而不是标签缺失；或
- 需要提高异常/长依赖样本的表征能力，且 H256 的独立验证已证明容量不足。

此时只训练一个候选：

```text
google/bert_uncased_L-6_H-256_A-4
```

选择它而不是 H512 的原因：

- 参数只从 `11.18 M` 增至约 `12.76 M`，增加约 `14%`，符合“稍微调大”。
- 是 Google 官方预训练 BERT，初始化风险显著低于自定义 H320。
- 结构变更只增加 2 层，不需要修改模型类、tokenizer、ONNX 输入输出或多标签 head。
- 结构探针 INT8 p95 约 `16.9–21.2 ms`、p99 约 `28.6–29.9 ms`，仍低于 p95 `50 ms`、p99 `70 ms` 的门禁。
- 相比 H512，训练和过拟合风险更低；如果它不能带来质量收益，就没有必要尝试更大的 H512。

固定训练配置：

```text
--model-name google/bert_uncased_L-6_H-256_A-4
--max-epochs 3
--skip-long-phase
--short-max-length 512
--command-max-tokens 128
--max-prefixes 32
--learning-rate-short 2e-5
--seed 42
```

必须使用新的 model ID 和新的 remote root，例如：

```text
model: cmd-session-risk-small-l6-h256-20260920-XXXX
remote-root: /teamspace/studios/this_studio/cmd-session-risk-small-l6-h256-20260920-XXXX
output: artifacts/cmd-session-risk-small-l6-h256-20260920-XXXX
```

训练命令骨架：

```bash
scripts/train_remote.sh \
  --host "$LIGHTNING_SSH_HOST" \
  --remote-root "/teamspace/studios/this_studio/cmd-session-risk-small-l6-h256-20260920-XXXX" \
  --data-dir data/processed/v2 \
  --output-dir artifacts/cmd-session-risk-small-l6-h256-20260920-XXXX \
  --detach \
  -- --model-name google/bert_uncased_L-6_H-256_A-4 \
       --max-epochs 3 --skip-long-phase \
       --short-max-length 512 --command-max-tokens 128 \
       --max-prefixes 32
```

训练时间预算：基线为 `575.3 s`；L6-H256 应按 `1.3x–1.5x` 预留，即约 `750–865 s`。如果 `torch.compile` 或 fused AdamW 实测有效，可以作为候选运行的可选开关，但不能在未 A/B 的情况下计入收益。严格而言，这一阶段不能满足“训练时间完全不变”。

### 7.4 第三阶段：L6 不满足条件时的回退

如果 L6-H256 出现以下任一情况：

- 质量没有超过 H256；
- ECE 或 tactic macro-F1 反而退化；
- INT8 parity 超限；
- p95 或 p99 接近门禁；
- 训练时间超出预算；

则拒绝 L6，不尝试 H512，直接回到 H256。

如果质量提升明显、但要求部署延迟严格与 H256 相同，则唯一推荐的后续路线是：

1. 用 L6-H256 作为 teacher；
2. 将 logits/soft labels 蒸馏回当前 H256 学生；
3. 重新校准；
4. 只部署 H256 学生，保持当前参数量、ONNX 结构和推理延迟。

这条路线训练时间会增加，但部署时间不增加。它不是“免费扩容”；如果训练时间也被设为完全不能增加的硬约束，那么唯一正确选择就是第一阶段，不进行任何参数扩容。

### 7.5 替换当前模型的硬门禁

L6-H256 只有同时满足以下条件才允许替换：

- FP32 parity `<= 1e-4`、INT8 parity `<= 0.05`；
- 512-token CPU p95 `<= 50 ms`、p99 `<= 70 ms`，且 p99 不得比当前交付基准显著恶化；
- Recall `>= 0.989212`；
- FPR `<= 0.010507`；
- PR-AUC `>= 0.999806`；
- ROC-AUC、detection rate 和 detection delay 不退化；
- ECE `< 0.05`；
- tactic macro-F1、technique macro-F1 不退化且至少一项有实质提升；
- 所有 evaluation slice 均有有效正负样本并通过；
- proxy audit resolved cases `>= 500`；
- 重新生成 manifest、SHA256 和 promotion report。

### 7.6 最终指定结论

**现在不直接训练 H512，也不直接扩容当前生产图。**

先执行 H256 的零推理成本校准、数据覆盖和审计修复；只有确认是模型容量不足时，才启动一次 `google/bert_uncased_L-6_H-256_A-4` 的三 epoch A/B。L6 未通过硬门禁就保持 H256；若必须严格保持 H256 的部署延迟，则把 L6 仅作为 teacher，最终蒸馏回 H256。
