# DeepSeek API 调用基准数据（阶段5实测）

> 记录于 2026-07-10，模型：`deepseek-chat`，`base_url=https://api.deepseek.com`。
> 用于后续第 21 节实验对比表格的"平均延迟"列，以及成本估算参考。

## 连通性测试（scripts/test_deepseek_connection.py）

| 指标 | 数值 |
|---|---|
| 首次调用延迟 | 8606.2 ms（含首次 TLS 握手/连接建立开销） |
| 二次调用延迟 | 2258.7 ms |
| prompt_tokens | 10 |
| completion_tokens | 28 |
| total_tokens | 38 |

## 端到端 RAG 生成测试（真实 pipeline：BM25 召回 + Reranker 精排 + build_rag_prompt + DeepSeek 生成）

问题：《我的笔记本电池鼓包了，可以自己拆开吗？》

| 指标 | 数值 |
|---|---|
| 生成延迟 | 8.04 s |
| 输出格式 | 结论 / 处理步骤 / 注意事项 三段式，均命中 |

## 简化 Prompt 生成测试（不含检索资料，仅系统指令）

| 指标 | 数值 |
|---|---|
| 生成延迟 | 7.55 s |
| prompt_tokens | 34 |
| completion_tokens | 314 |
| total_tokens | 348 |

## 结论

- 单次生成延迟稳定在 7～9 秒区间，主要开销在 completion 阶段（输出内容较长，300+ tokens）。
- 简单问答（无检索上下文）的 prompt_tokens 很低（34），实际接入完整检索资料后 prompt_tokens 会显著上升，需要在阶段 11 正式评估时重新采样记录。
- 超时重试链路（`tenacity` 外层 3 次 + OpenAI SDK 内部重试）在网络异常时会显著放大总耗时（实测约 14s 才会耗尽重试后失败），`generate_safe` 降级机制确保该场景下用户仍能获得响应而不是报错。
