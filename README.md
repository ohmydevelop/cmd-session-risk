# cmd-session-risk

Research implementation for Unix command-session malicious probability and
MITRE ATT&CK stage/technique prediction.

The first release scores command prefixes, emits alert-only decisions, and
exports a CPU INT8 ONNX artifact. It does not block commands.

See `configs/` for prompts, `scripts/` for data/teacher preparation, `kaggle/`
for the fallback GPU entry points, and `src/cmdrisk/` for the package.

## v1.4.0（v6 模型，2026-09-23 夜间训练）

- 模型：v5 L8-H256 续训 1 epoch，+3600 混淆增强会话；灾难单命令门禁：critical 1.0 / obf 0.9863 / benign-twin FPR 2.0%（含意图层）
- 操作点：TH=1.1e-4，T=0.9085（详见 `reports/V1.4.0_RELEASE_zh.md`）

## v1.5.1（2026-09-24，阈值重校准 + 正/反测试集门禁）

- 操作点：**TH=2.1e-4、T=0.9085 不变**、权重不变、意图层不变。旧 TH=1.1e-4 在当前
  冷冻测试集上复测 benign-twin FPR=7.5%（60/800），与历史报告不符，如实重校准；
  `make -f Makefile.release smoke` / `serve-smoke` 与新门禁脚本全过。
- 正/反测试集（`data/processed/v6_obf_evals/`，`split:test`，与训练不重叠）：
  - 正（灾难单命令 400 条）：召回 **400/400 = 1.0000**（门禁 >=0.995）
  - 正（混淆变体 1600 条）：召回 **1597/1600 = 0.99813**（门禁 >=0.95）
  - 反（良性孪生 800 条）：误报 **15/800 = 0.01875**（门禁 <=0.02）
  - 一键复测：`scripts/evaluate_serve_gates.py --base-url http://127.0.0.1:8080`
  - 全过程与数字、披露：`reports/V1.5.1_RECALIBRATION_zh.md`
- 新增 `serve` HTTP daemon（v1.5.0 交付），见下方专节；旧版 v1.4.0/v1.4.2 及历史
  release 全部保留。

## Install（单一二进制）——生产使用

v1.5.1 的 `cmdrisk` 是**单一可执行文件**：TH/T 与模型架构常量运行时从内嵌 `calibration.json` / `model_config.json` 读取；输出新增**意图类别 + MITRE 标签**（tail：意图层类别名 / tactic argmax / technique argmax）（v1.1.1 修温度缩放；v1.1.2 子串守卫；v1.3.0 升级为 shell 语义意图层）：模型（FP32）和 BERT 词表已在编译期嵌入，无需安装 Python、无需额外模型目录。

```bash
# 1) 从 GitHub Release 下载到 ~/.local/bin
mkdir -p ~/.local/bin
gh release download v1.5.1 \
  --repo ohmydevelop/cmd-session-risk \
  --pattern cmdrisk \
  --output ~/.local/bin/cmdrisk
chmod +x ~/.local/bin/cmdrisk
export PATH="$HOME/.local/bin:$PATH"

# 2) 校验完整性（expected aa4dd873a89fbcfa7e51a72561b3c9f0b5b14a0b9910a811b9a1b81dc2f94c26）
sha256sum ~/.local/bin/cmdrisk

# 3) 运行：每行一个命令前缀；命令之间用 0x1f 分隔；每条命令为 <cmd><TAB><dt>
cmdrisk ci_assets/smoke_text.txt 0.00011
```

测试样例输出格式：

```text
1	9.77597713	0.999995351	1	0
2	9.6828537	0.999994755	1	0
```

字段：`行号 <TAB> mal_logit <TAB> mal_prob <TAB> decision(1=告警) <TAB> rule_hit(1=意图层命中) <TAB> intent_classes <TAB> top_tactic <TAB> top_technique`。
- `intent_classes`：意图层命中的类别（`destructive_delete/device_write/device_wipe/kernel_panic/uid0_backdoor/mass_kill/blanket_perms/download_exec/reverse_shell/fork_bomb`，可 `+` 连接）
- `top_tactic / top_technique`：模型多标签头的 argmax（仅 decision=1 时显示；TA0003 等 14 类 / T1059 等 12 类）
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

## Daemon / HTTP 服务（v1.5.0 新增）

v1.5.0 起，同一个单一二进制内置一个**持续运行的 HTTP 打分守护进程**，沿用与
批量 CLI 完全相同的 C 推理路径（概率**位级一致**，不作二次校准）：

```bash
# 前台运行（默认只监听回环地址，无鉴权）
cmdrisk serve --bind 127.0.0.1:8080

# 健康检查
curl -sS http://127.0.0.1:8080/healthz
# {"ok":true,"model":"v6","version":"v1.5.1","bind":"127.0.0.1:8080"}

# 打分一次完整命令会话（逐前缀返回恶意概率 + 意图类 + MITRE 标签）
curl -sS -X POST http://127.0.0.1:8080/v1/score \
  -H 'Content-Type: application/json' \
  --data '{"session_id":"s1","commands":["id","rm -rf /"]}'
```

`/v1/score` 请求：`session_id`（可选，缺省自动 `sess-<pid>-<seq>`）+
`commands`（字符串，或 `{"cmd":"...","dt":0.5}` 对象数组）。响应为：

```json
{
  "session_id": "s1",
  "model_version": "v6",
  "schema_version": "1.0",
  "peak_probability": 0.00191341515,
  "alerted": true,
  "answers": [
    {"command_index":0, "malicious_probability":0.000136836054, "risk_score":0.000136836054,
     "attack_stage":"Command and Control", "intent_classes":"-", "tactic":"TA0011",
     "technique":"T1070", "alert":true},
    {"command_index":1, "malicious_probability":0.00191341515, "risk_score":0.00191341515,
     "attack_stage":"Impact", "intent_classes":"destructive_delete",
     "tactic":"TA0040", "technique":"T1036", "alert":true}
  ]
}
（示例值为 `["id","rm -rf /"]` 的实际输出；数值随会话内容变化，以服务器返回为准。）
```

- `alert = (malicious_probability >= threshold) || 意图层命中`；阈值/温度从内嵌
  `calibration.json` 读取，与批量 CLI 同源。
- 错误统一 `{"error":{"code":"...","message":"..."}}`：400 `invalid_json`/`invalid_request`、
  404 `not_found`、405 `method_not_allowed`、408 `request_timeout`、413 `payload_too_large`、
  500 `inference_error`。请求体上限 `--max-body`（默认 1 MiB），单会话命令上限 1024。
- **默认无鉴权、仅监听 `127.0.0.1`**；`--bind` 到非回环地址会打印警告，生产跨机使用
  请自建反向代理/TLS/认证。定位是**告警 + 人工复核，不自动阻断**。
- 单线程串行处理，实测 3 命令会话约 88 ms/请求、常驻 `VmHWM` ≈ 31 MiB。

### systemd 部署

```bash
sudo install -m 0755 build/cmdrisk /usr/local/bin/cmdrisk
sudo cp deploy/cmdrisk.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now cmdrisk
curl -sS http://127.0.0.1:8080/healthz
```

### 集成测试

```bash
make -f Makefile.release all smoke serve-smoke
```

`serve-smoke` 自动覆盖：healthz 契约、恶意/良性打分、逐前缀 `answers`、400/404/405/413
错误码、与批量 CLI 的概率位级一致性，以及常驻内存 < 60 MiB 上限。
