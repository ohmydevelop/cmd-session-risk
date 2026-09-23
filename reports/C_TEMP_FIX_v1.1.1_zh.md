# v1.1.1 修复报告：C 推理温度缩放 + 分词器 + 长前缀越界

日期：2026-09-23
影响版本：v1.0.0 C runtime（`onnx_int8_c` / `cmd_pipeline_c`）、v1.1.0 单一二进制
修复版本：v1.1.1

## 1. 触发案例

用户运行：

```bash
cmdrisk /tmp/b 0.0001179935351014905   # /tmp/b 内容：rm -rf /tmp/a
# v1.1.0 输出：1  -8.03881073  0.000322588399  1   ← 告警
```

Python 校准管道（同一 FP32 模型）输出：

```text
malicious_probability = 4.2674e-05 (< 1.18e-04)  → decision = 0（不告警）
```

`rm -rf /tmp/a` 是普通临时目录清理，属**误报**；v1.1.0 单二进制与标定操作点判定相反。

## 2. 根因：C 端漏掉 temperature 缩放

- 校准文件 `onnx/calibration.json`：`malicious_temperature = 0.7969617216679871`，
  `malicious_threshold = 0.0001179935351014905`。
- Python 运行时：`prob = sigmoid(mal_logit / malicious_temperature)`，再用阈值判定。
- v1.1.0 的 C 代码：`prob = sigmoid(mal_logit)`，**未除 temperature**，然后用同一个阈值。
- 由于 temperature ≈ 0.797，logit 在 -8 附近时 C 概率被高估约 **7–8 倍**，
  把大量近阈值良性会话推过告警线。

量化影响：v5_runtime_sample 的 258 个真实前缀参考集上，
旧 C 判定 258 个告警，而标定操作点只有 **246** 个告警；**12/258 决策与生产参考相反**（全部为多报）。
这些样本已固化为回归集 `ci_assets/regression_borderline.txt`。

## 3. 次生修复 1：C 自带 tokenizer 多子词退化 [UNK]

v1.1.0 的 subword 回退逻辑只尝试“完整词（`tmp`）”或“`##`+剩余全串”，
无法切出多段子词，例如 `tmp` → Python WordPiece 是 `t` + `##mp`，旧 C 会整词退化为 `[UNK]`。

修复：改为 WordPiece 标准的“贪心最长优先匹配”，首段不带 `##`、后续段带 `##`：

```c
if(off==0) probe = substring(buf, 0, l);   /* 首段无 ## */
else       probe = "##" + substring(buf, off, l);
```

修复后 C 与 Python 对同一文本产出完全一致的 token 序列（`rm -rf /tmp/a` 对照验证通过）。

## 4. 次生修复 2：长前缀写越界（栈溢出）

ASAN 确认 `encode_prefix` 中 `ids0[n++]=SID_CMD_START` 与 `SID_EOS` 缺少上界检查；
当前修复的分词器会产出更多 token，长会话前缀（文本路径第 125 行为例）会写穿 `ids0[4096]`。

修复：这两处写入同样加 `if(n<4096)` 保护（序列下游已有 256-token 截断逻辑）。
修复后 258 行全量文本前缀：`rc=0`，130.0s，RSS 37.4 MiB，无崩溃。

## 5. 修复后验证（全部可复算）

| 检查 | 结果 |
|---|---:|
| C vs Python-FP32 决策一致（258 真实前缀） | 258/258 |
| C vs Python-INT8 决策一致（生产默认，258 前缀） | 258/258 |
| 与标定操作点一致的告警数 | 246/258 |
| C vs FP32 最大 \|Δprobability\| | 3.95e-05 |
| 近阈值回归集（13 行，期望全 0） | 13/13 decision=0 |
| 真实恶意 smoke（8 行，期望全 1） | 8/8 decision=1 |
| 用户案例 `rm -rf /tmp/a` | prob 4.2674e-05 → decision=0（与参考一致） |
| 单一进程 RSS（全量 258 行） | 37.4 MiB |

## 6. 交付变更

- 新单二进制 SHA256（v1.1.1）：
  `c67be5f68ad697b5adc9dd1d635179492157408d2df6259dc2a3a2812d0d745a`
- `tools/cmdrisk_calibration.h`：集中存放 `MAL_TEMP`，注明校准文件出处，供四个 C 工具共用。
- `Makefile.release` 的 `smoke` 目标升级为“行数 + decision 值”双重断言，
  并加入 13 行近阈值回归集，CI（`.github/workflows/ci-build.yml`）自动执行。
- v1.0.0 C-runtime 的 `onnx_int8_c` / `cmd_pipeline_c` 同样 bug，源码已同步修复；
  旧 tarball 由 v1.1.1 取代。

## 7. 不改变的东西

模型权重、训练数据、校准阈值、FP32/INT8 图全部未动；本次只修正 C 侧推理后处理与分词，
因此不产生新的模型版本，也不影响此前所有模型对比报告的有效性。

## 8. 遗留注意（诚实披露）

- 单条命令、无时间戳输入仍是模型分布外用法；`rm -rf /tmp/a` 这类样本依赖规则白名单
  与人工复核兜底，模型本身不能自动“放行”危险语义。
- INT8 与 FP32 在阈值附近仍存在量化抖动（前缀概率最大差可达 1.5e-4），
  生产部署建议固定 FP32 单二进制或明确 INT8 操作点再上线。
