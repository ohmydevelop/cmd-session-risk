# cmd-session-risk 生产可用性报告 V5（GPU L8 + FP32/INT8 全链路证据）

- 日期：2026-09-22（America/New_York）
- 工作目录：`/home/ubuntu/Works/llm/cmd-session-risk`
- 范围：在**保留 v2/v3/v4 全部方案、artifact、报告**的前提下，把百万级 v5（L8-H256-A4，Lightning L4 GPU）推进到生产灰度候选：FP32/INT8 ONNX、dev-only/temporal-holdout 验证、会话级操作点、full-prefix INT8 一致性、256-token 延迟、manifest 完整性，并给出新老完整对比。
- 过程声明：**未创建 token budget**；**不删除/覆盖任何旧方案**；**不伪造 audit/reviewer/promotion**。v4 checkpoint-soup 是新增实验，其 dev 筛选因无不再劣化的候选而主动放弃 test，属于正确 fail-closed。
- 门禁声明：**所有 `promotion.passed` 仍为 `false`**。硬 blocker 见第 7 节。

---

## 0. 结论先行

1. **生产灰度默认候选（新）**：`artifacts/cmd-session-risk-v5-l8-h256-million-gpu-20260922-recal-prod-20260922`
   - 8 层 BERT、`14,339,611` 参数（比 v4 L6 的 `12,760,118` 多 `1.12×`）。
   - 百万级同一 v4 训练集（1,047,750 个 cap32 prefix）上训练；后半段在 **Lightning NVIDIA L4 (23 GB)** 完成，本地 CPU 用 checkpoint-6000 保留保底。
   - v4 test（temporal holdout，共 7,260 会话 / 148,861 cap32 prefix）prod 操作点：
     - prefix：recall `0.992102`、FPR `0.015812`、precision `0.995849`
     - PR-AUC `0.999698`、**ROC-AUC `0.998883`**（v4 `0.993475`）
     - session：recall `0.989317`、FPR `0.074014`（FN 51 / FP 184，固定 FPR-budget 拟合；会话级 FPR 目标档见表 5.2）
   - INT8 full-prefix：session 一致率 `0.997245`、prefix 一致率 `0.999775`；INT8−FP32 Δrecall `+0.000419`、ΔFPR `-0.001609`。
   - 256-token 延迟（机器空闲同条件）：p95 `6.231 ms`、p99 `9.061 ms`；正式导出件记录 p95 `15.576 ms`、p99 `16.692 ms`，均低于 50/70 ms 门限。
   - `promotion.passed=false`（预期，数据/审计 blocker 与架构大小无关）。
2. **回退生产默认**：`artifacts/cmd-session-risk-v4-l6-h256-million-20260921-recal-prod-20260921`
   - prefix：recall `0.991754`、FPR `0.016817`、precision `0.995585`、ROC-AUC `0.993475`
   - session：recall `0.993507`、FPR `0.067578`（FN 31 / FP 168）
   - INT8 session 一致率 `0.996694`、prefix 一致率 `0.999799`
   - 256-token 空闲 p95 `5.671 ms`
3. **高召回回退**：`artifacts/cmd-session-risk-v3-l6-h256-recal-prod-20260921`（v3 测试集口径）会话 recall `0.930021` / FPR `0.003268`；512-token INT8 p95 `18.546 ms`。
4. **soup 实验无胜者**：`soup-last3` / `soup-last4` 在 dev 上 prefix FPR 分别 `+0.000258` / `+0.000645`，session FPR 分别 `+0.004839` / `+0.005645`，**没有候选不劣于 final**，因此未读 test、不提升。旧 semilla 全保留。
5. **本轮零训练成本 + 低推理成本的真正收益**：
   - Lightning L4 完成 L8 续训，避免数小时 CPU；**部署端只多 1 层、约 +10% 空闲延迟**
   - full-prefix INT8 校验的 **token 饱和复用**：把 44M 次 batch-1 前向降到实际推理 74,042 次 + 179,406 次数学复用，仍全量计数 253,448 前缀
   - dev-only checkpoint 筛选、Stage A/B fail-closed 标记、256 深度契约的显式修复
6. **为什么不是「全通过生产模型」**：`tactic_label_coverage`(7<10)、`ood_2024` 无良性、`real_plus_semi_real` 无正类、`proxy_audit` 无独立人工投票。这些是数据/审计供给问题，增大模型无法修复；因此最终选择 v5 作为灰度/shadow/告警+人工复核候选，**不做自动阻断**。

---

## 1. 数据规模、布局与来源归因（权威口径）

| 集合 | 会话 | 恶/良 | cap32 prefixes | commands |
| --- | ---: | ---: | ---: | ---: |
| train | 46,398 | 27,988 / 18,410 | **1,047,750** | 1,920,875 |
| dev | 5,800 | 3,496 / 2,304 | 129,123 | 224,866 |
| test | 7,260 | 4,774 / 2,486 | 148,861 | 253,448 |

- **零泄漏**：train∩dev∩test 的 `family_id` 不相交；`train∩test session_id=0`。
- **full-prefix ≠ cap32**：训练/评估使用 family-aware cap32（最多 32 个前缀/会话）；`test.jsonl` 的 full prefix = 每条命令一个前缀，实测 **253,448**（不是 21,990,949——后者的三角形求和是“命令出现次数”，不是前缀数）。
- prefix 来源归因：`real_malicious 612,542`、`llm_variant 206,855`、`real_history_pool 131,677`、`script_benign 51,359`、`hard_negative 31,483`、`real_benign 13,834`。
- 真实来源小计 **809,412（77.3%）**；确定性增强 **238,338（22.7%）**；**无 LLM 生成**；`llm_variant` 的 `source.name` 是 `deterministic_attack_variant`。
- 保留缺口：`microsoft_iot` 占 prefix 57%（单源偏斜）；Microsoft「redistribution unconfirmed」覆盖 76.6% prefix；`real_history_pool` 11,595 session 「unverified」。

---

## 2. 数据量是不是太少？（结论不变）

- 二分类正/负样本量足够，v4/v5 在真实 temporal holdout 上都接近拟合上限。
- 不足的是分布构成：真实良性（尤其 2024+）少、OOD 2024 无良性、`real_plus_semi_real` 无正类、稀有 tactic TA0001/6/8/10 不够 support。
- **结论：下一步是补数据 + 补审计，而不是继续加层数。**

---

## 3. 保留的旧方案清单（未删除、未覆盖）

- `artifacts/cmd-session-risk-v3-l6-h256-recal-prod-20260921`（v3 生产基准，权重冻结）
- `artifacts/cmd-session-risk-v3-l6-h256-recal-20260921`、`artifacts/cmd-session-risk-v3-l6-h256-20260920`
- `artifacts/cmd-session-risk-v3-l4-h256-20260920`
- `artifacts/cmd-session-risk-v4-l6-h256-million-20260921` 及 `-recal-prod-20260921`
- `artifacts/cmd-session-risk-v4-l6-h256-million-20260921-soup-last3` / `-soup-last4`（soup provenance + 权重哈希）
- 旧报告：`FINAL_REPORT_zh.md`、`FINAL_REPORT_SMALL_MODEL_zh.md`、`OPTIMIZATION_ASSESSMENT_20260920_zh.md`、`FINAL_COMPARISON_V2_V3_zh.md`、`PRODUCTION_READINESS_V4_zh.md`（均保留）
- CPU 版 v5 检查点：`checkpoint-1500/3000/4500/6000` 保存在 `artifacts/cmd-session-risk-v5-l8-h256-million-20260921/`
- 脚本历史备份：`work/backups/*`

---

## 4. 本轮优化方案（各方案独立保留以便对比）

| 方案 | 说明 | 成本 | 结论 |
| --- | --- | --- | --- |
| A. L8-H256-A4（v5） | 比 L6 多 1 层；同数据、同超参 | +1.12× 参数、+1.33× FLOPs、训练耗时（GPU） | ROC +0.0054、prefix recall +0.00035；**新默认** |
| B. Lightning L4 续训 | checkpoint-4500 传远端 GPU 续训 | 上传/下载网络时间 | 免 CPU 约 8h 等待 |
| C. 512→256 深度契约 | 探针超过 `max_length` 快速失败、按 256 测量 | 0 | 消除死循环，口径与部署一致 |
| D. fast INT8 校验（饱和复用） | tail_budget=0 时超长前缀编码不变，计数量不变 | 0 | 旧实现保留并做 smoke 等价性 |
| E. dev-only screen 修 tokenizer-source | checkpoint 目录缺少 tokenizer 时回退到 `short_phase` | 0 | soup 成功跑通 |
| F. fail-closed Stage B | 子步骤失败即 `STAGE_B_EVIDENCE_FAILED`，不发成功标记 | 0 | 阻止了两次假成功 |
| G. checkpoint soup（soup-last3/last4） | 尾 checkpoints 权重平均 | 0（权重生成便宜） | dev 上无胜者，不提升 |
| H. manifest latency 元数据原子修复 | `update_export_manifest_latency` + 完整性刷新 | 0 | 证据一致性修复 |

---

## 5. 最终对比

### 5.1 v4 test（cap32 校准口径，prod 冻结阈值；microsoft_iot 等真实来源的重采样口径）

| 方案 | prefix recall | prefix FPR | precision | PR-AUC | ROC-AUC | Brier | ECE | tactic F1(≥50) | tech F1(≥50) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| v4 recal-prod | 0.991754 | 0.016817 | 0.995585 | 0.998633 | 0.993475 | 0.015869 | 0.016227 | 0.711633 | 0.727353 |
| **v5 recal-prod** | **0.992102** | 0.015812 | 0.995849 | **0.999698** | **0.998883** | 0.016165 | 0.016253 | 0.707828 | 0.726789 |

### 5.2 会话级（v4 全量 dev/test，对应各自固定 FPR-budget 校准温度）

| 方案 | threshold | session recall | session FPR | FN | FP | TP | TN |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| v4 recal-prod | 6.8181867e-05 | 0.993507 | 0.067578 | 31 | 168 | 4743 | 2318 |
| v5 recal-prod | 5.2595996e-05 | 0.989317 | 0.074014 | 51 | 184 | 4723 | 2302 |

部署视角的 session FPR 目标档（每次 fit 在 dev；只报 test）：v4 有 1% 目标档对应 test recall 0.9914/FPR 0.0129；v5 有 test recall 0.978634/FPR 0.008447 的 1% 目标档（不同温度区间，参考 `session_operating_point_sweep.json`，不可与上面冻结阈值行混排）。

### 5.3 模型阶梯与延迟（ONNX 图）

| variant | params | vs L6 | FLOPs vs L6 | 256-token 空闲 p95 | 256-token 空闲 p99 |
| --- | ---: | ---: | ---: | ---: | ---: |
| L4-H256-A4 | 11,180,571 | 0.88× | 0.67× | — | — |
| L6-H256-A4（v4） | 12,760,118 | 1.00× | 1.00× | 5.671 ms | 7.973 ms |
| **L8-H256-A4（v5）** | 14,339,611 | 1.12× | 1.33× | **6.231 ms** | **9.061 ms** |
| L12-H256-A4 | 17,498,651 | 1.37× | 2.00× | — | — |
| L6-H384-A6 | 22,728,219 | 1.78× | 2.25× | — | — |
| L6-H512-A8 | 35,088,411 | 2.75× | 4.34× | — | — |

### 5.4 INT8↔FP32（full-prefix，253,448 前缀 / 7,260 会话）

| 指标 | v4 | **v5** |
| --- | ---: | ---: |
| prefix decision agreement | 0.999799 | 0.999775 |
| session decision agreement | 0.996694 | 0.997245 |
| first-alert drift max / differing | 26 / 34 | 24 / 24 |
| Δrecall (INT8−FP32) | 0.000000 | +0.000419 |
| ΔFPR (INT8−FP32) | +0.008045 | -0.001609 |

验证方法均为 `full_prefix_saturation_reuse`；旧逐前缀实现保留 `scripts/validate_int8.py`，快速实现 `scripts/validate_int8_fast.py` 已与之做采样等价性对照。

### 5.5 soup（dev 筛分，budget 0.01，test 未读）

| 候选 | prefix ΔFPR | session ΔFPR | dominates |
| --- | ---: | ---: | ---: |
| soup-last3 | +0.000258 | +0.004839 | false |
| soup-last4 | +0.000645 | +0.005645 | false |

**无胜者：保留 v4 final 不变。**

---

## 6. 产物体积与关键哈希

- v5 FP32：`model.onnx` 57,966,487 B；外部数据 `model.onnx.data` 57,153,536 B
- v5 INT8：`model.int8.onnx` 15,220,327 B
- v5 权重 `model.safetensors` SHA256 `f5e1d07163b7c1a112e2cd898a1423111010492990155da1972571b3151ba3f1`
- v5 ONNX 导出：FP32 parity max err `4.77e-06`；INT8 parity max err `0.014585`（容差 0.05）
- v4 保留：INT8 SHA256 `154ec3fccd5f28fd49533dd069049c5ab4284f777930c6b938d581ffbcc72b75`；FP32 SHA256 `a7d8801fade2239a04eeb264231ad09cb9ae99b70f692b80dc66a514cb017168`

---

## 7. 门禁与 blocker（fail-closed，未清除）

1. `evaluation_slice:ood_2024`：结构性 `insufficient_negatives`（切片 1,458 mal / 0 benign），brier 亦失败。
2. `evaluation_slice:real_plus_semi_real`：切片映射口径缺陷导致 positive=0（`slice_breakdown.py` 可量化纠正）。
3. `tactic_label_coverage`：target 10，实测 7 个 head support ≥ 50（缺 TA0001/6/8/10/40/42/43 中至少 3 个）。
4. `proxy_audit`：无 500+ 独立人工 resolved cases；不得伪造。
5. 因此 `promotion.passed=false`、`promotion.eligible_for_promotion=false`。

---

## 8. 生产部署建议

- **默认（灰度/shadow/告警+人工复核）**：v5 recal-prod。
- **更严格 FPR 回退**：v4 recal-prod。
- **v3 兼容回退**：v3 recal-prod（512-token 契约）。
- 部署端点原则不变：`SessionScorer.from_onnx` 默认 `verify_manifest=True`，任何字节不符即拒绝加载。
- 监控：告警率、per-tactic 分布、INT8/FP32 1% 双跑抽样、p95/p99。**不得用于自动阻断**，直到第 7 节全部清除。

---

## 9. 下一步

1. P0 补真实良性 / OOD 良性 / 稀有 tactic / 独立 reviewer，逐个清除第 7 节 blocker。
2. P1 对 v5 可选静态 QDQ/SmoothQuant（减少 INT8 FPR 偏移）、更大 512-deep 契约复训一次并重新评估。
3. P2 若数据补足后 L8 仍不如 L6，回退 v4（新老 artifact 都保留，切换只需改 `artifact_dir`）。

---

## 10. 复现命令摘要

```bash
# v5 GPU 续训（远端）
bash scripts/run_v5_l8_million_chain.sh   # MODEL_NAME 自动指向 data/raw/pretrained/bert_uncased_L-8_H-256_A-4
# v5 Stage A（导出→打分→sweep）
RUN=artifacts/cmd-session-risk-v5-l8-h256-million-gpu-20260922 \
WORK=work/recal_v5_gpu MODEL_VERSION=cmd-session-risk-v5-l8-h256-million-gpu-20260922 \
LATENCY_LENGTHS=256 INT8_VALIDATOR=scripts/validate_int8_fast.py STAGE=a bash scripts/run_v4_posttrain.sh
# v5 Stage B（生产 artifact + 证据 + manifest）
RUN=artifact_dir WORK=... CELL=work/recal_v5_gpu/sweep/fpr_0.01 \
LATENCY_LENGTHS=256 INT8_VALIDATOR=scripts/validate_int8_fast.py bash scripts/run_v4_stage_b_evidence.sh
# soup 实验
nice -n 10 bash scripts/run_v4_soup_experiment.sh
```

---

## 11. 文件清单

- v5 run：`artifacts/cmd-session-risk-v5-l8-h256-million-gpu-20260922/`
- v5 prod：`artifacts/cmd-session-risk-v5-l8-h256-million-gpu-20260922-recal-prod-20260922/`
- 工作目录：`work/recal_v5_gpu/`
- v4 prod：`artifacts/cmd-session-risk-v4-l6-h256-million-20260921-recal-prod-20260921/`
- 分阶段日志：`work/recal_v4/posttrain_{a,b}.log`（含 fail-closed 历史）
- soup：`work/recal_soup/{soup.log,screen.json,selection.json}`
- 快速 INT8 校验：`scripts/validate_int8_fast.py`
- 报告：本文件；旧报告见第 3 节

---

## 12. 一句话总结

**在保留 v3/v4/soup 的前提下，把 v5 L8 用 Lightning L4 训完并完成 ONNX FP32/INT8、temporal-holdout 校准、full-prefix INT8 一致性（session 一致率 0.997245）与空闲 256-token 延迟（p95 6.231 ms）；v5 成为 prefix ROC-AUC 0.998883 的生产灰度候选，但 `promotion.passed=false`，仍受数据/审计 blocker 约束，只用于告警+人工复核，不可自动阻断。**


---

## 附录 A：v5 session1p 部署操作点（上线前 P0 优化证据）

- 新 artifact：`artifacts/cmd-session-risk-v5-l8-h256-million-gpu-20260922-recal-prod-session1p-20260922`
- 操作点：`malicious_temperature=0.7969617216679871`，`malicious_threshold=0.0001179935351014905`（dev 会话 FPR ≤1%）
- test（held-out）：session recall `0.978634`、session FPR `0.008447`（FN 102 / FP 21）；prefix recall `0.987729`、prefix FPR `0.003046`
- INT8 full-prefix：prefix 一致率 `0.999680`、session 一致率 `0.996694`；INT8 Δrecall `+0.004818`、ΔFPR `+0.000402`，first-alert 最大漂移 `26`（59 个会话）
- 256-token 线程矩阵（p95 ms）：t2 `17.20`、**t4 `9.28`**、t6 `7.10`、t8 `7.33`；默认推荐 `intra_op_threads=4`
- 并发压测（480 会话 × N worker，batch=16，t2/worker）：N=4 → `384.8 s`，N=8 → `791.7 s`；无崩溃/OOM，8 路 16 线程过订导致吞吐近线性下降
- 结论：以 INT8 + t4 + session 复用上线 gray；保持 `promotion.passed=false`，仅告警+人工复核
