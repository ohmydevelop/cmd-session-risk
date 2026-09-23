# cmd-session-risk

Research implementation for Unix command-session malicious probability and
MITRE ATT&CK stage/technique prediction.

The first release scores command prefixes, emits alert-only decisions, and
exports a CPU INT8 ONNX artifact. It does not block commands.

See `configs/` for prompts, `scripts/` for data/teacher preparation, `kaggle/`
for the fallback GPU entry points, and `src/cmdrisk/` for the package.

## Install（单一二进制）——生产使用

v1.3.0 的 `cmdrisk` 是**单一可执行文件**（v1.1.1 修温度缩放；v1.1.2 子串守卫；v1.3.0 升级为 shell 语义意图层）：模型（FP32）和 BERT 词表已在编译期嵌入，无需安装 Python、无需额外模型目录。

```bash
# 1) 从 GitHub Release 下载到 ~/.local/bin
mkdir -p ~/.local/bin
gh release download v1.3.0 \
  --repo ohmydevelop/cmd-session-risk \
  --pattern cmdrisk \
  --output ~/.local/bin/cmdrisk
chmod +x ~/.local/bin/cmdrisk
export PATH="$HOME/.local/bin:$PATH"

# 2) 校验完整性（expected 86303342a86c292ceb3111db96051211e6a979a613006a85c99561c848c50b9d）
sha256sum ~/.local/bin/cmdrisk

# 3) 运行：每行一个命令前缀；命令之间用 0x1f 分隔；每条命令为 <cmd><TAB><dt>
cmdrisk ci_assets/smoke_text.txt 0.0001179935351014905
```

测试样例输出格式：

```text
1	9.77597713	0.999995351	1	0
2	9.6828537	0.999994755	1	0
```

字段：`行号 <TAB> mal_logit <TAB> mal_prob <TAB> decision(1=告警) <TAB> rule_hit(1=静态高危模式命中)`。
`decision = (mal_prob >= threshold) || rule_hit`；`rule_hit=1` 时 stderr 会打印 `intent-layer force-alert`。
意图层（`tools/cmdrisk_intent.h`，v1.3.0）是一个确定性的 mini shell 语义分析器：
拆分语句/管道、解 `sh -c` 包装、求值 `$(echo/printf)`、解码 `\xNN`/八进制/ANSI-C 引号/
`${IFS}`、解码 base64 管道载荷、词法路径规范化（`/tmp/../etc`→`/etc`），然后对
“递归删除系统路径 / 裸设备写 / mkfs / sysrq / useradd -u 0 / kill -9 1 / chmod -R 000,777 /
curl|bash / /dev/tcp 反向 shell / fork bomb”等意图谓词强制告警（只告警不阻断），
把模型在单命令分布外场景对 `rm -rf /` 级别命令的漏报降为零，详见
`reports/INTENT_LAYER_v1.3.0_zh.md`。

用 `curl` 没有 `gh` 时（需要 GitHub token）：

```bash
ASSET_ID=<release 里 cmdrisk 的 asset id>
curl -L \
  -H "Authorization: Bearer $GITHUB_TOKEN" \
  -H "Accept: application/octet-stream" \
  "https://api.github.com/repos/ohmydevelop/cmd-session-risk/releases/assets/$ASSET_ID" \
  -o cmdrisk
chmod +x cmdrisk
```

## 检查

```bash
# 使用规定阈值（session FPR ≤1%）
cmdrisk --help  # 见 usage 提示
```

指标：单一进程 RSS 实测约 `29–37 MiB`；在 2026-09-23 的 258 个真实前缀 test sample 上，决策（`mal_logit / malicious_temperature` 后过阈值）与 Python 校准管道一致率：vs FP32 `258/258`、vs INT8（生产默认）`258/258`（其中 246 个告警前缀）。
`ci_assets/regression_borderline.txt`（13 行近阈值回归样本，期望全部 decision=0）锁定 v1.1.0 的 temperature bug 不复现；`make -f Makefile.release smoke` 同时断言 8 条告警样本全为 decision=1。修复说明见 `reports/C_TEMP_FIX_v1.1.1_zh.md`。
