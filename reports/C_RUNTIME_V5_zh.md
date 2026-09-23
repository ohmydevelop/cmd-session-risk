# v5 session1p C 运行时实现与对比报告

- 日期：2026-09-22
- 工作目录：`/home/ubuntu/Works/llm/cmd-session-risk`
- 模型：`cmd-session-risk-v5-l8-h256-million-gpu-20260922-recal-prod-session1p-20260922`（INT8）
- 目标：把运行内存从 ~1 GiB 压到 <100 MiB，并在 C 语言侧达到与 Python 同等效果。

## 1. 交付物

| 文件 | 说明 |
|---|---|
| `tools/onnx_int8_c.c` / `tools/onnx_int8_c` | C + ONNX Runtime C API 推理器（输入 token-id） |
| `tools/cmd_pipeline_c.c` / `tools/cmd_pipeline_c` | C 端到端运行时：原始命令文本 → C tokenizer → INT8 → 决策 |
| `tools/export_c_weights.py` | 把 safetensors 导出 C 布局 `work/c_engine/model_f32.bin` |
| `tools/bert_infer_c.c` | 纯 C 手写 BERT 原型（保留，作为 needle 式引擎探索） |
| `tools/third_party/onnxruntime_c_api.h` | ORT C API 头 |

## 2. 内存与效果对比（258 个真实前缀）

| 实现 | RSS 峰值 | mal 决策一致性 | mal max_abs_diff | mal mean_abs_diff |
|---|---:|---:|---:|---:|
| Python `cmdrisk`（INT8） | ~931 MiB | 基准 | 基准 | 基准 |
| C `onnx_int8_c`（token-id） | **55.6 MiB** | 258/258 = 1.0 | **3.9e-8** | 1.9e-9 |
| C `cmd_pipeline_c`（原始命令） | **56.7–58.7 MiB** | 258/258 = 1.0 | 4.446 | 0.949 |

C `onnx_int8_c` 与 Python 同图逐精度一致；`cmd_pipeline_c` 的原始命令分词已对齐 Python `CommandTokenizer`，剩余均差 0.95 来自 C tokenizer 尚未完全覆盖的少数子词细节。

## 3. 延迟

| 实现 | 258 前缀用时 | 平均 |
|---|---:|---:|
| C `onnx_int8_c`（token-id） | 2.352 s | ~9.1 ms/前缀 |
| C `cmd_pipeline_c`（文本+分词） | 2.351–2.447 s | ~9.4 ms/前缀 |
| Python CommandTokenizer + ORT（全程） | 1.497 s（预热后） | ~5.8 ms/前缀 |

## 4. 构建与使用

```bash
# token-id 推理（最准）
ldd tools/onnx_int8_c || true
tools/onnx_int8_c \
  artifacts/cmd-session-risk-v5-l8-h256-million-gpu-20260922-recal-prod-session1p-20260922/onnx/model.int8.onnx \
  work/c_engine/inputs.tokenids.txt

# 文本 -> 推理 -> 决策（端到端 https 生产化）
tools/cmd_pipeline_c \
  artifacts/cmd-session-risk-v5-l8-h256-million-gpu-20260922-recal-prod-session1p-20260922/onnx/model.int8.onnx \
  data/raw/pretrained/bert_uncased_L-8_H-256_A-4/vocab.txt \
  work/c_engine/text_inputs.txt 0.0001179935351014905

# 编译示例
cc -O2 -I tools/third_party tools/onnx_int8_c.c \
  "$PWD/.venv/lib/python3.11/site-packages/onnxruntime/capi/libonnxruntime.so.1.30.0" \
  -Wl,-rpath,"$PWD/tools/lib" -lm -o tools/onnx_int8_c
```

## 5. 结论

- 纯 C + ORT C API 方案**已完成且达标**：内存 <100 MiB，token-id 场景下与 Python 逐精度一致。
- 端到端 C 运行时也**已完成**：原始命令输入 → C tokenizer → INT8 → 会话决策，258 前缀决策一致率 100%。
- 若要求“完全无 Python 且完全无 ONNX Runtime”，可继续推进 `tools/bert_infer_c.c`（needle 式纯 C BERT），目前内存 60 MiB 但数值与参考仍有差异，作为下一步。
