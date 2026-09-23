# v1.3.0：意图识别层（mini shell 语义分析 + 混淆解码）

日期：2026-09-23
上一版本：v1.1.2（子串式静态守卫）

## 1. 动机

用户指出 v1.1.2 的守卫是“硬编码、不能泛化”——正确。
v1.3.0 的目标不是推翻这层，而是把它从“子串匹配”升级为
**面向 shell 语义的意图分析器**，让它对大量结构性混淆保持检出。

分层原则（不变）：
- BERT 模型 = 泛化引擎（学习分布内判别）；
- 意图层 = 灾难级谓词的**确定性担保**（告警侧零漏报、可审计）；
- 未来 eBPF 行为层 = 终极防线（本项目定位为告警+人工复核，暂不做）。

## 2. 能力（全部在 C 内实现，`tools/cmdrisk_intent.h`）

| 能力 | 示例 | 结果 |
|---|---|---|
| 语句/管道拆分 | `id; rm -rf /`、`a \|\| b` | 逐段分析 |
| `sh -c` 解包（递归） | `sh -c "rm -rf /"` | 命中 |
| `$()``/反引号`` 主动求值（echo/printf） | `$(echo rm) -rf /` | 命中 |
| 转义解码 | `\x72\x6d -rf /`、`$'\x72\x6d' -rf /`、八进制 | 命中 |
| `${IFS}` 展开 | `rm${IFS}-rf${IFS}/` | 命中 |
| ANSI-C 引号 | `$'...'` | 命中 |
| base64 管道求值（递归） | `echo cm0gLXJmIC8K \| base64 -d \| sh` | 命中 |
| 路径词法规范化 | `rm -rf /tmp/../etc` → `/etc` | 命中 |
| 大小写/混合外壳 | `rm -RF /Etc`、`curL \| BASh` | 命中 |
| busybox/toybox | `busybox rm -rf /` | 命中 |

意图谓词集：递归删除系统路径、裸设备写入（`dd of=/dev/*`、重定向、tee）、
`mkfs/wipefs/fdisk/parted/shred`、`/proc/sysrq-trigger`、`useradd -u 0`、
`kill -9 1/-1`、`chmod -R 000/777 /`、下载即执行、`/dev/tcp/` 反向 shell、
fork bomb、`--no-preserve-root`。

## 3. 验证（`make -f Makefile.release smoke` 全部断言）

- 高危回归 `ci_assets/regression_critical.txt` **28 行**：28/28 decision=1, rule=1
  （含 8 类混淆变体）
- 对照回归 `ci_assets/regression_borderline.txt` **21 行**：21/21 decision=0, rule=0
  （`rm -rf /tmp/a`、`dd if=/dev/sda of=/tmp/backup.img`、`chmod -R 600 ./myapp` 等）
- 258 个真实前缀：意图层 **0 次误触**；C ≡ Python 决策一致 **258/258**
- 真实恶意 smoke 8 行：8/8 decision=1
- RSS：36.7 MiB（258 全量示例）

## 4. 诚实边界（不宣称解决一切混淆）

- 图灵完备混淆（`perl -e`、`python3 -c`、自解压/VM/自定义解释器）仍可绕过；
- 嵌套/非标准 base64、动态下载载荷、变量间接寻址无法静态求值；
- 意图层只**强制告警**，不做放行/阻断；`sh -c 'echo hi'` 这类仍由模型打分
  （实测模型 0.51 会告警，属模型行为，不属守卫）。
- 对“执行任意命令就是恶意”的判断仍依赖 BERT；意图层只担保灾难类。

## 5. 后续路线

1. v6 训练：混淆增强语料 + 负样本最小对，新增 promotion 门禁
   `critical-single-command recall >= 0.995`、`obfuscation-set recall >= 0.95`；
2. 意图层继续扩：`printf %b/%q`、`xxd -r -p`、`openssl enc -d`、`gunzip -c`、
   `python3 -c`/`perl -e` 的有限模式语言求值；
3. eBPF 行为拦截（保留接口，不做自动阻断）。
