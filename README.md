# 中文 3C 售后 RAG 问答系统

基于模拟品牌的售后文档，使用 BM25、向量检索、CrossEncoder 和 DeepSeek 提供资料问答、来源展示、安全提示与工单建议。后端为 FastAPI，页面为 Streamlit，日志和反馈存储于 SQLite。

当前正在按 [plan.md](plan.md) 改进正确性、独立评估和性能，之后再开展领域重排序训练。知识库含 24 篇模拟文档、103 个片段；项目尚不代表真实企业生产部署或完整工单平台。

## 当前状态

- 回答区分正常生成、无依据拒答、安全响应和上游服务故障。
- 已识别的高风险请求使用确定性安全响应，不依赖 LLM 生成。
- 生成故障返回 HTTP 503 和结构化处理建议；普通业务拒答返回 HTTP 200。
- 响应、日志与工单建议使用一致的最终转人工状态；工单建议尚未接入实际派单。
- 索引记录知识库哈希与模型信息，知识库变化时重建，空候选与超范围 k 有边界处理。
- 日志记录请求 ID、上下文片段与分数、分阶段耗时、生成状态和可用的 token 用量。
- 检索评估已区分 Hit@k 与 Recall@k；生成故障不会获得质量分。

当前运行环境中，后端 Python SDK 访问 DeepSeek 曾受网络权限限制；通过 curl 的模型验证不能替代完整后端成功验证。详见 [正确性报告](docs/correctness_report.md)。原有评估数据用于历史开发和回归，不作为独立测试。历史文档中的实验成绩尚不能作为本轮最终结论。

## 本地启动

建议使用独立环境。已有临时 `.venv` 继承系统包，不能作为干净安装已验证的证据。`requirements.windows.observed.txt` 是当前机器上的依赖版本快照，包含 CUDA 构建；不是跨平台通用锁文件。

```powershell
python -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt
Copy-Item .env.example .env
# 在 .env 中填入自己的模型配置，勿提交密钥。
.venv\Scripts\python.exe scripts/build_chunks.py
.venv\Scripts\python.exe scripts/build_index.py
.venv\Scripts\python.exe -m uvicorn app.main:app --host 127.0.0.1 --port 8000
```

另开终端：

```powershell
.venv\Scripts\python.exe -m streamlit run frontend/streamlit_app.py --server.address 127.0.0.1 --server.port 8501
```

页面：<http://127.0.0.1:8501>；接口文档：<http://127.0.0.1:8000/docs>。

首次加载需下载模型；已有完整本地缓存时，可在启动前设置 `HF_HUB_OFFLINE=1` 和 `TRANSFORMERS_OFFLINE=1`，仅禁止 Hugging Face 联网，不影响 DeepSeek 配置。

```powershell
docker compose up --build
```

仓库提供 Docker 配置，当前改动后的容器验收仍待完成。

## 接口状态

| 接口／回答类型 | HTTP 状态 | 含义 |
|---|---|---|
| `GET /health` | 200 | 进程存活 |
| `GET /ready` | 200／503 | 知识库与模型已加载／未就绪，不主动付费调用 LLM |
| `POST /ask`，`generated` | 200 | 获得非空且未截断的模型回答，仍需评估事实与引用 |
| `POST /ask`，`low_confidence` | 200 | 当前依据不足，建议人工核实 |
| `POST /ask`，`safety_refusal` | 200 | 已识别高风险，提供确定性安全提示 |
| `POST /ask`，`service_unavailable` | 503 | 上游连接、限流、空输出或截断等生成故障 |
| `POST /feedback` | 200／404 | 写入反馈／查询 ID 不存在 |

故障响应同样保留 `log_id`、`request_id`、`answer_type` 和人工处理建议。安全规则基于关键词并包含有限的否定表达处理，其通过率不能代表所有表达下的安全性。

## 测试与实验

```powershell
.venv\Scripts\python.exe -m pip install pytest httpx
.venv\Scripts\python.exe -m pytest -q

# 只跑离线评估，保存逐题输出与配置；每次使用新目录。
.venv\Scripts\python.exe scripts/run_eval.py --output data/eval/runs/new-eval-name

# 显式增加真实生成，会调用配置的模型并产生相应用量。
.venv\Scripts\python.exe scripts/run_eval.py --output data/eval/runs/new-online-eval-name --with-generation

# 候选数量的开发实验，不调用 LLM。
.venv\Scripts\python.exe scripts/benchmark.py --output data/eval/runs/new-benchmark-name --repeats 3
```

默认数据属于历史开发集。新独立数据的分组、标注和最终测试流程见 [评估协议](docs/evaluation_protocol.md)。尚未审核的数据不得被标记为独立测试成果；不得将重复耗时测量次数当作不同问题数。

## 目录

| 路径 | 职责 |
|---|---|
| `app/ingestion/` | 文档读取、清理和切分 |
| `app/retrieval/` | BM25、FAISS、融合和重排序 |
| `app/generation/` | Prompt 与有明确故障语义的模型调用 |
| `app/safety/`、`app/ticket/` | 风险识别、安全响应和工单建议 |
| `app/storage/` | 非破坏性 SQLite 迁移、查询日志和反馈 |
| `app/evaluation/`、`scripts/` | 评估、数据检查、索引与基准实验 |
| `tests/` | 状态、HTTP、故障、迁移和评估口径回归 |
| `data/eval/runs/` | 版本化配置、逐题结果与汇总 |
| `docs/` | 基线、验证、评估协议及性能记录 |

简历只写已完成且有证据的功能和指标。领域模型尚未完成训练时，不写“已微调”；工单仅为建议时，不写“已实现完整工单流转”。
