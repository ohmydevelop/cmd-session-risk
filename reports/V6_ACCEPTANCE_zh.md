# v6 验收清单（2026-09-23 夜间训练 · 晨检用）

> 状态：训练已结束（train loss 0.0335 / eval loss 0.0598，1 epoch，16,428 steps）。
> 当前进度：Stage A（FP32/INT8 全量打分）→ Stage B（session-FPR1% 重校准）→ Stage C（新门禁）→ 烘焙单二进制 v1.4.0。

## 一、要验收的三条新门禁（fail-closed）

| 门禁 | 目标 | v5 基线（修复后模型但未训练过混淆） |
|---|---:|---:|
| critical_single_command_recall | >= 0.995 | 0.2475 FAIL |
| obfuscation_set_recall | >= 0.95 | 0.2294 FAIL |
| benign_twin_fpr | <= 0.02 | 0.1225 FAIL |
| pooled ROC-AUC (灾难正样本 vs 良性双生) | 越高越好 | 0.5840 |
| 外部基线 GLiClass instruct-base zero-shot | 参照 | 0.5521 AUC / recall@FPR2% 0.25%-2.6% |

## 二、不变的既有门禁（沿用 v5 全套，不看总分只看门禁）

ROC/PR-AUC、Brier/ECE、session-FPR1%、2024 temporal、tactic/technique F1、
INT8-FP32 一致性、检测延迟、proxy audit、许可重分发。
注意：**promotion.passed 仍会是 false**——ood_2024 无良性、real_plus_semi_real
缺正类、tactic 覆盖 7/14、proxy audit 空缺这些 blocker 与本次训练无关，
照旧如实披露。

## 三、晨检动作（用户侧一条命令）

```bash
gh release download v1.4.0 --repo ohmydevelop/cmd-session-risk \
  --pattern cmdrisk --output ~/.local/bin/cmdrisk --clobber
chmod +x ~/.local/bin/cmdrisk
sha256sum ~/.local/bin/cmdrisk   # 期望与 release 报告一致

# 三个必测样例（单二进制内嵌 v6 FP32 + 意图层）
printf 'rm -rf /\n'                      > /tmp/a.txt   # 期望 decision=1
printf 'sh -c "rm -rf /tmp/../etc"\n'   > /tmp/b.txt   # 期望 decision=1
printf 'rm -rf /tmp/a\n'                > /tmp/c.txt   # 期望 decision=0
cmdrisk /tmp/a.txt 0.0001179935351014905
cmdrisk /tmp/b.txt 0.0001179935351014905
cmdrisk /tmp/c.txt 0.0001179935351014905
```

## 四、交付物（按优先级，所有旧版本保留）

1. v6 模型 artifact（metrics.json / onnx FP32+INT8 / calibration / v6_gates.json / 混淆 eval 切片）
2. v1.4.0 单二进制（内嵌 v6 FP32 权重 + 更新后的 TH/MAL_TEMP + 意图层；SHA256 另附）
3. 本报告 + 数字对比（v5 vs v6 vs GLiClass）
4. 所有新脚本已入仓：build_v6_obfuscation_data / evaluate_v6_gates /
   v6_gate_threshold_sweep / run_v6_obf_chain / bake_v6_release

## 五、明早我要向你确认的诚实数字（训练完成后填）

- v6 在 session-FPR1% 阈值下三条门禁的实际值
- 若 obfuscation_recall 未达 0.95：sweep 出来的“三条门禁全过”的最小阈值及其代价（session FPR 上升多少）
- v6 主测试集 (ROC/PR/Brier/ECE) 与 v5 的对比（预期轻微下降，换来灾难类兜底）
- INT8 一致性 + C 二进制 3 套回归判定一致
