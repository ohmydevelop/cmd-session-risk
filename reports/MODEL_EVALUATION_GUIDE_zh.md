# 如何判断 cmd-session-risk 模型好不好（benchmark / SOTA 视角）

## 1. 一句话结论

- 本项目目标域（Unix 命令前缀实时威胁检测）**没有公开 leaderboard**；不能直接拿 SQuAD/ImageNet 型 SOTA 头衔对比。
- 项目中已内置一套比“单一准确率”严格得多的门禁评估：ROC/PR-AUC、阈值无关排序、Temporal holdout（2024 时间外推）、Brier/ECE 校准、检测延迟、tactic/technique F1、INT8 一致性、许可/审计。
- 目前模型在私有构造测试集上表现强，但由于测试集私有 + 数据许可/审计 blocker 未清，**不宣称 SOTA**；对外商业声明应使用下一节口径。

## 2. 现有实测 benchmark（可复算，凭证在 artifact）

### 2.1 阈值无关排序指标（测试集 148,861 cap32 prefix / 7,260 会话）

| 模型 | ROC-AUC | PR-AUC | Brier | ECE |
|---|---:|---:|---:|---:|
| v3 recal-prod | 0.997883 | 0.999854 | 0.033117 | 0.042045 |
| v4 L6 百万级 | 0.993475 | 0.998633 | 0.015869 | 0.016227 |
| **v5 L8 百万级** | **0.998883** | **0.999698** | **0.016165** | **0.016253** |

### 2.2 会话级部署口径（配置阈值，first-prefix latch）

| 操作点 | 阈值 | Recall | FPR | FN / FP | Precision |
|---|---:|---:|---:|---:|---:|
| v5 session-FPR 1% | 0.0001179935 | 0.978634 | 0.008447 | 102 / 21 | 0.995525 |
| v5 prefix-FPR 1%（冻结） | 5.2595996e-05 | 0.989317 | 0.074014 | 51 / 184 | 0.962503 |
| v4 prefix-FPR 1% | 6.8181867e-05 | 0.993507 | 0.067578 | 31 / 168 | 0.965791 |

解释：部署应优先用 session-FPR 1% 行；信噪比最好。

### 2.3 时间外推（真实 2024 temporal holdout，无良性 → 只报检出率）

| 版本 | 2024 detection rate | tactic F1(>=50) | technique F1 |
|---|---:|---:|---:|
| v5 L8 百万级 | 0.960905 | 0.691371 | 0.722274 |

说明：该切片只有恶意样本，无法算 FPR/ROC；这是待补数据缺口，不是模型能力上限。

### 2.4 实时/SLO 指标

| 指标 | 实测 |
|---|---:|
| 256-token CPU p95/p99（v5 空闲） | 6.231 ms / 9.061 ms |
| INT8−FP32 session 一致性 | 0.997245 |
| 检测延迟（median alert command） | 0 条命令 |
| 单二进制 RSS | 36–61 MiB |

## 3. 我们的门禁目标（训练评估器内置）

- recall >= 0.90，FPR <= 0.02
- PR-AUC >= 0.90，ROC-AUC >= 0.95
- Brier <= 0.10，ECE <= 0.05
- tactic/technique macro-F1 >= 0.65
- tactic label coverage >= 10（当前 7，未过）
- 检测 delay <= 3 commands
- 全部 evaluation slice 通过（ood_2024 / real_plus_semi_real 未过）
- 500+ 独立 proxy audit（未过）
- 许可可重分发审计（未确认）

`promotion.passed=false` 是真实结果，包含这些缺口。

## 4. 和“SOTA”怎么比？

不允许直接比较原因：
1. 没有公开 benchmark 同意 evaluator；
2. 我们的训练集以公开抓取命令为主，匿名测试，与论文的 cut 不同；
3. 论文常用 accuracy/F1（类别倾斜敏感），我们使用 ROC/PR/Brier/延迟。

可打出的公开主张：**支持百分按 threshold 满足全部硬门禁**（除了明确列出的数据/审计 blocker）。

## 5. 补 benchmark 的执行方案（商业验收建议）

### 5.1 内部基线 suite（第一优先）
- char n-gram TF-IDF + Logistic Regression
- char n-gram + LightGBM/XGBoost
- BERT-base L12 蒸馏重训
- v5 导入、ModernBERT small
统一数据切分/度量；输出 metrics 对比表。这是最容易落地的 SOTA 证明。

### 5.2 公开数据集 zero-shot / 微调
- Shell command dataset(s) 可选：KYPO（已纳入 test 良性）、HDF 与公开 Linux APT 历史语料
- 统一 metric：ROC-AUC、PR-AUC、@FPR1 recall、Brier/ECE
- 发布结果与 code/evaluator，供第三方复核。

### 5.3 独立红队 + `proxy audit`
- 引入独立 reviewer/红队完成 500+ resolved audit cases
- 输出 Fleiss kappa、误报-漏报权衡，关闭审计 blocker

### 5.4 EDR/SIEM 联合评测
- 与真实日志管线做告警量和 SOC 误报疲劳指标对比

## 6. 如何向商业方表达

- 不在“精确度”上编 SOTA；说“本系统在自建 0.8M 真实前缀测试上：
  ROC 0.9989 / 会话 FPR1 档 recall 0.9786 FP 21；
  2024 时间外推检出率 96.1%；p95 6.2ms；单二进制 36–61MiB；INT8 一致性 99.72%。”
- 同时披露未通过门禁：ood/real_plus 切片缺失、tactic 7/14、proxy audit 空缺、Microsoft 重分发未确认。

这比任何单一 benchmark 分数更可靠、更诚实。


## 7. 外部参照与零样本基线（2026-09 当前技术）

- **GLiNER / GLiClass**（urchade/knowledgator）：通用化轻量编码器路线——
  GLiNer 做零样本 NER，GLiClass 把同架构用于单前向序列分类。
  我们与之同类：14.3M 参数、单前向、CPU 可跑。
  实测 `knowledgator/gliclass-instruct-base-v1.0` 零样本在我们的
  灾难命令评测切片上 pooled ROC-AUC 仅 **0.5521**、confusion 明显，
  说明“开箱零样本”在该窄域不可用；自训小模型是更诚实路线。
- **needle3（Needle 3 On-Device AI Engine）**：9–29MB、2-bit(CQ2)、
  C 原生、离线工具路由——验证了“极简端侧 + C 运行时”的可行性，
  也即本项目单二进制（57.6MB FP32 / INT8 ONNX 路径）的同类方向；
  未来可探索 INT8/更低比特 C 推理进一步缩小体积。
- 结论：外部不存在能直接打分的 SOTA leaderboard；我们以
  “同 slices 外部基线 + 内部多模型对比 + 门禁”组合作为标准答案。