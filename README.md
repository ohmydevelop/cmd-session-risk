# cmd-session-risk

Research implementation for Unix command-session malicious probability and
MITRE ATT&CK stage/technique prediction.

The first release scores command prefixes, emits alert-only decisions, and
exports a CPU INT8 ONNX artifact. It does not block commands.

See `configs/` for prompts, `scripts/` for data/teacher preparation, `kaggle/`
for the fallback GPU entry points, and `src/cmdrisk/` for the package.

## Install（单一二进制）——生产使用

v1.1.0 的 `cmdrisk` 是**单一可执行文件**：模型（FP32）和 BERT 词表已在编译期嵌入，无需安装 Python、无需额外模型目录。

```bash
# 1) 从 GitHub Release 下载到 ~/.local/bin
mkdir -p ~/.local/bin
gh release download v1.1.0 \
  --repo ohmydevelop/cmd-session-risk \
  --pattern cmdrisk \
  --output ~/.local/bin/cmdrisk
chmod +x ~/.local/bin/cmdrisk
export PATH="$HOME/.local/bin:$PATH"

# 2) 校验完整性（expected e4598a03a43e6fbc250ef779cc8d1af4b0efbb58e32cc536f4b7dd8e59abd27a）
sha256sum ~/.local/bin/cmdrisk

# 3) 运行：每行一个命令前缀；命令之间用 0x1f 分隔；每条命令为 <cmd><TAB><dt>
cmdrisk ci_assets/smoke_text.txt 0.0001179935351014905
```

测试样例输出格式：

```text
1	9.77597618	0.999943256	1
2	9.68285465	0.999937654	1
```

字段：`行号 <TAB> mal_logit <TAB> mal_prob <TAB> decision(1=告警)`。

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

指标：单一进程 RSS 实测约 `36–61 MiB`；在 2026-09-23 生成的 test sample 上，与 Python 管道的 258 个真实前缀决策一致率 `258/258`。
