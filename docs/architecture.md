# 系统架构图

```mermaid
flowchart TD
    A[用户问题] --> B[安全风险识别<br/>risk_detector.py]
    B --> C[Hybrid Retrieval]
    C --> C1[BM25 关键词召回]
    C --> C2[Dense Embedding 向量召回]
    C1 --> D[候选合并 + 归一化融合]
    C2 --> D
    D --> E[Reranker 重排<br/>CrossEncoder]
    E --> F{低置信度判断<br/>rerank top1 < 0.1}
    F -->|是| G[低置信度拒答模板]
    F -->|否| H[build_rag_prompt]
    H --> I[DeepSeek API 生成]
    I --> J[三段式答案<br/>结论/处理步骤/注意事项]
    G --> K[工单判断]
    J --> K
    K --> L{need_human?}
    L -->|是| M[生成工单<br/>分类/优先级/部门]
    L -->|否| N[直接返回]
    M --> O[返回答案 + 引用来源 + 工单]
    N --> O
    O --> P[日志记录 SQLite]
    O --> Q[Streamlit 前端展示]
    Q --> R[用户点赞/点踩反馈]
    R --> P
```

## 模块与代码对应关系

| 架构环节 | 对应代码 |
|---|---|
| 安全风险识别 | `app/safety/risk_detector.py`、`app/safety/policy.py`、`app/safety/refusal.py` |
| BM25 召回 | `app/retrieval/bm25_retriever.py` |
| Dense 召回 | `app/retrieval/dense_retriever.py`（FAISS 持久化索引） |
| Hybrid 融合 | `app/retrieval/hybrid_retriever.py`（`bm25_weight` 可调） |
| Reranker 重排 | `app/retrieval/reranker.py`（CrossEncoder） |
| Prompt 构造 | `app/generation/prompt.py` |
| LLM 生成 | `app/generation/llm_client.py`（DeepSeek API + 降级兜底） |
| 工单判断与生成 | `app/ticket/classifier.py`、`app/ticket/ticket_builder.py` |
| 主流程串联 | `app/main_pipeline.py`（`CustomerSupportRAG.answer`） |
| FastAPI 后端 | `app/main.py`（`/ask`、`/feedback`、`/health`） |
| 日志与反馈存储 | `app/storage/db.py`（SQLite：`query_logs`、`feedback`） |
| Streamlit 前端 | `frontend/streamlit_app.py` |
| 离线评估 | `app/evaluation/`、`scripts/run_eval.py` |
