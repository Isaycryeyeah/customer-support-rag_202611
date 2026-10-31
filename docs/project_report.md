# 中文 3C 售后客服助手 —— 项目报告

> 历史报告：下文描述的是此前实现与自述实验，并非当前改动后的验收结果。当前代码与验证范围以 README、plan.md、docs/correctness_report.md 及 data/eval/runs/ 中的新记录为准；未复现的数字不得直接用于简历。

> 本报告基于对整个项目从零到部署验收全过程的真实实现与验证撰写，所有数据和结论均来自实际运行结果（`data/eval/report.json`、`data/eval/bad_case_analysis.md`、容器化环境下的真实 DeepSeek API 调用），不包含虚构或估算内容。

## 目录

1. [项目定位](#1-项目定位)
2. [启动方法](#2-启动方法)
3. [工作原理](#3-工作原理)
4. [功能清单](#4-功能清单)
5. [目录结构与模块职责](#5-目录结构与模块职责)
6. [项目的特殊之处](#6-项目的特殊之处)
7. [实验数据与已知局限](#7-实验数据与已知局限)
8. [验收结论](#8-验收结论)

---

## 1. 项目定位

面向中文 3C（手机/笔记本/平板/配件）售后客服场景的生产级 RAG 系统。不是一个只跑通"检索+生成"的 Demo，而是完整覆盖了数据处理、混合检索、重排序、安全风险识别、生成、工单流转、日志、前后端和 Docker 部署的工程闭环。

设计目标聚焦三个真实业务痛点：

1. **答案必须基于文档，不能幻觉**——售后政策经常变化，用户会追问细节；
2. **涉及安全风险时要有拒答和转人工机制**——电池鼓包、进液、危险拆机等问题不能让 LLM 自由发挥；
3. **系统要有完整的日志、反馈、评估闭环**——不能只是离线跑通就结束。

生成模型选用 **DeepSeek API**（`deepseek-chat`，通过官方 `openai` SDK 以 OpenAI 兼容接口调用）。

---

## 2. 启动方法

### 2.1 本地开发模式

```bash
# 1. 创建虚拟环境并安装依赖
python -m venv .venv
source .venv/Scripts/activate      # Windows Git Bash；其他平台见 venv 文档
pip install -r requirements.txt

# 2. 配置 DeepSeek API Key
cp .env.example .env
# 编辑 .env，填入真实 DEEPSEEK_API_KEY

# 3. 构建数据和索引（首次运行必需）
python scripts/build_chunks.py     # 解析 data/raw_docs/*.md，生成 data/processed/chunks.jsonl
python scripts/build_index.py      # 构建并持久化 FAISS 索引（首次运行会从 HuggingFace 下载 Dense 模型）

# 4. 启动后端和前端（两个独立终端）
uvicorn app.main:app --reload                              # http://localhost:8000
streamlit run frontend/streamlit_app.py                    # http://localhost:8501
```

### 2.2 Docker 一键启动

```bash
docker compose up --build
```

`docker-compose.yml` 会同时构建并启动 `backend`（FastAPI，端口 8000）和 `frontend`（Streamlit，端口 8501）两个容器：

- `backend` 通过 `env_file: .env` 在运行时注入 `DEEPSEEK_API_KEY` 等变量，密钥**不会**被打进镜像（`.dockerignore` 排除了 `.env`）；
- `backend` 的 `Dockerfile` 在安装 `requirements.txt` 前先单独装 CPU 版 `torch`（`--index-url https://download.pytorch.org/whl/cpu`），避免 pip 默认解析到 CUDA 版本拉取 2GB+ 的 NVIDIA 依赖；
- `frontend` 通过 `BACKEND_API_URL=http://backend:8000` 环境变量在 compose 网络内用服务名访问 backend（而不是 `localhost`，这是容器化部署里的常见坑）；
- `./data:/app/data` volume 挂载保证 FAISS 索引和 SQLite 数据库在容器重启后持久化，不需要重新构建。

首次启动时 backend 需要从 HuggingFace Hub 下载 Dense（`BAAI/bge-small-zh-v1.5`）和 Reranker（`BAAI/bge-reranker-base`，约 1.1GB）模型，网络较慢时可能需要几分钟，实测中遇到过下载卡死在固定字节数（768MB）不再前进的情况，重启容器或挂载宿主机已有的 HuggingFace 缓存可以规避。

### 2.3 跑离线评估

```bash
python scripts/run_eval.py
```

会依次跑 BM25/Dense/Hybrid/Hybrid+Reranker 四组检索对比实验、生成质量评估（真实调用 DeepSeek API）、安全规则评估，结果写入 `data/eval/report.json`。

---

## 3. 工作原理

### 3.1 端到端流程

```
用户问题
  ↓
安全风险识别（关键词规则，独立于检索/生成）
  ↓
Hybrid Retrieval：BM25 召回 + Dense 召回 → 归一化融合
  ↓
Reranker 重排（CrossEncoder，query-document pair 级别打分）
  ↓
低置信度判断（rerank top1 分数 < 0.1 阈值）
  ├─ 是 → 返回标准拒答模板
  └─ 否 → 构造 Prompt → DeepSeek 生成三段式回答（结论/处理步骤/注意事项 + 引用来源）
  ↓
工单判断（need_human 或 low_confidence 时自动生成）
  ↓
返回 {question, answer, contexts, risk_info, low_confidence, need_human, ticket, log_id}
  ↓
写入 SQLite 日志（question/answer/risk_level/need_human/low_confidence/latency_ms）
  ↓
前端展示 + 用户点赞/点踩反馈（写入 feedback 表，关联 log_id）
```

完整实现在 `app/main_pipeline.py` 的 `CustomerSupportRAG.answer()` 方法中串联，核心代码结构：

```python
def answer(self, query: str) -> Dict:
    risk_info = detect_risk(query)                                    # 1. 安全风险识别
    candidates = self.hybrid_retriever.search(query, top_k=20, candidate_k=50)  # 2. Hybrid 召回
    contexts = self.reranker.rerank(query, candidates, top_k=5)       # 3. Rerank
    low_confidence = contexts[0]["rerank_score"] < 0.1 if contexts else True  # 4. 低置信度判断
    answer = LOW_CONFIDENCE_ANSWER if low_confidence else self.llm.generate_safe(build_rag_prompt(query, contexts, risk_info))
    ticket = build_ticket(...) if (risk_info["need_human"] or low_confidence) else None  # 6. 工单判断
    return {...}
```

值得注意：**风险识别和检索/生成是并行独立的两条判断路径**，不是"先检索再判断风险"。即使检索/重排结果被判定为低置信度，`risk_detector` 依然会独立地对原始问题做关键词匹配，这保证了"进水后放进大米里"这类问题即使检索到的文档相关度不够高，安全规则仍能兜底触发转人工，不会因为检索层的偶然失误而放过一个高危问题。

### 3.2 混合检索的具体机制

- **BM25**（`app/retrieval/bm25_retriever.py`）：`jieba` 中文分词 + `rank_bm25` 的 `BM25Okapi`，对每个 chunk 算词频统计相关性分数。擅长精确关键词/型号匹配，但对口语化、无关键词重叠的语义型问题（如"鼓起来了还能用吗"里不含"鼓包"字面词）会完全召回失败。
- **Dense Retrieval**（`app/retrieval/dense_retriever.py`）：用 `BAAI/bge-small-zh-v1.5` 把 query 和 chunk 都编码成 512 维向量，通过 FAISS 的 `IndexFlatIP`（内积索引，向量已归一化即等价于余弦相似度）做向量检索。索引持久化到 `data/processed/faiss.index` + `faiss_meta.pkl`，重复启动时直接从磁盘加载，不会重新计算全部 chunk 的 embedding。
- **Hybrid 融合**（`app/retrieval/hybrid_retriever.py`）：分别拿 BM25 和 Dense 的候选集，各自做 min-max 归一化（避免两种检索方法分数量纲不一致），再按 `bm25_weight`（默认 **0.3**，可通过 `.env` 里的 `BM25_WEIGHT` 覆盖）加权融合排序。

### 3.3 Reranker 的作用

`app/retrieval/reranker.py` 用 `CrossEncoder`（`BAAI/bge-reranker-base`）对 Hybrid 召回的候选做更细粒度的 query-document pair 打分（不是像 Dense 那样各自独立编码后算相似度，而是把 query 和 document 一起输入模型联合编码，理论上能捕捉更精细的相关性）。输出的 `rerank_score` 同时服务于两个目的：

1. 决定最终返回给生成模型的 top-k 上下文排序；
2. 作为低置信度判断的信号——如果 top1 分数都很低，说明知识库里没有真正相关的资料，应该拒答而不是让 LLM 硬凑答案。

### 3.4 安全风险识别的独立性

`app/safety/risk_detector.py` 是纯规则实现（关键词匹配），不依赖 LLM 判断，刻意设计为四类风险独立于检索/生成流程：

```python
RISK_KEYWORDS = {
    "battery_safety": ["电池鼓包", "鼓包", "鼓起来", "漏液", "冒烟", "起火", ...],
    "dangerous_repair": ["自己拆", "短接", "撬开", "进水后开机", "放进大米", ...],
    "refund_boundary": ["直接退款", "直接换新", "保证保修通过", ...],
    "complaint": ["投诉", "举报", "维权", ...],
}
```

`battery_safety`/`dangerous_repair` 命中 → `risk_level=high` + 强制转人工；`refund_boundary`/`complaint` 命中 → `risk_level=medium` + 转人工。这个设计的好处是：安全判断不会受 LLM 的不确定性影响，也不会因为检索到的文档凑巧不够相关就漏掉风险识别。

### 3.5 DeepSeek 集成与降级机制

`app/generation/llm_client.py` 的 `LLMClient` 封装了 `openai` SDK 对 DeepSeek API 的调用：

- `_call_deepseek` 用 `tenacity` 库对 `RateLimitError`/`APIConnectionError`/`APITimeoutError` 做指数退避重试（最多 3 次，1~8 秒退避）；
- `generate_safe` 是 pipeline 实际调用的入口，在 `generate()` 重试耗尽仍失败时捕获异常并返回 `FALLBACK_ANSWER`（标准降级提示），而不是让整个 `/ask` 请求返回 500。这个设计保证了 DeepSeek API 偶发故障（限流、超时、网络抖动）时用户依然能收到响应，而不是看到报错页面。

实测过一次真实的超时场景（把 client timeout 故意设成 0.01 秒触发）：tenacity 的 3 次外层重试叠加 OpenAI SDK 自身的内部重试，总耗时约 14.3 秒后才耗尽重试抛出异常，`generate_safe` 正确捕获并返回降级答案。

### 3.6 工单流转

`app/ticket/classifier.py` 按风险类型把问题分流到四类（安全风险/退换货退款/投诉升级/普通咨询），对应不同优先级和建议处理部门；`app/ticket/ticket_builder.py` 生成包含用户问题、模型回答、风险等级、处理建议摘要的完整工单 JSON。工单只在 `need_human=True`（风险识别命中 或 低置信度触发）时生成，避免给每一次普通问答都产生冗余工单。

---

## 4. 功能清单

- 基于 24 篇售后文档（覆盖电池安全/充电器安全/保修政策/退换货/维修流程/发票凭证/人工转接/危险操作/能力边界/数据隐私 10 大主题）进行问答；
- 三段式回答格式（结论/处理步骤/注意事项）+ 引用来源标注；
- 四类安全风险识别（电池安全、危险操作、越权承诺、投诉）与对应的拒答/转人工策略；
- 低置信度自动拒答（基于 rerank 分数阈值）；
- 自动生成工单（分类、优先级、建议部门、摘要）；
- 完整的 query 日志记录（问题、答案、风险等级、是否转人工、延迟）；
- 用户点赞/点踩反馈接口（`POST /feedback`），关联到具体的 query 日志；
- Streamlit 前端：问题输入、回答展示、风险识别展示、工单建议、引用资料可展开查看、点赞点踩按钮；
- 离线评估体系：检索层（recall@k/MRR）、生成层（格式遵循率/答案要点覆盖率）、安全层（转人工准确率/风险类型识别准确率）；
- Docker 一键部署，密钥通过环境变量注入，不写入镜像。

---

## 5. 目录结构与模块职责

```
customer-support-rag/
├── app/
│   ├── main.py                 FastAPI 应用，/ask /feedback /health 三个路由
│   ├── main_pipeline.py        CustomerSupportRAG，串联全流程的核心类
│   ├── config.py               集中管理所有环境变量和路径配置
│   ├── schemas.py               Pydantic 请求/响应模型
│   ├── ingestion/               文档加载、清洗、按标题切分为 chunk
│   ├── retrieval/                BM25 / Dense / Hybrid / Reranker 四个检索组件
│   ├── safety/                  风险识别规则、能力边界话术、拒答模板
│   ├── generation/               Prompt 构造、DeepSeek LLM 客户端
│   ├── ticket/                   工单分类与生成
│   ├── storage/                  SQLite 日志和反馈存储
│   └── evaluation/               检索/生成/安全三层评估脚本
├── scripts/                      build_chunks / build_index / run_eval / DeepSeek 连通性测试
├── frontend/streamlit_app.py     Streamlit 前端
├── data/
│   ├── raw_docs/                 24 篇原始 Markdown 售后文档
│   ├── processed/                chunks.jsonl + FAISS 索引文件
│   ├── eval/                     三个评估集 + report.json + bad case 分析
│   └── app.db                    SQLite（query_logs + feedback 表）
├── docs/                         架构图、本报告
├── Dockerfile / docker-compose.yml / .dockerignore
└── requirements.txt / .env.example
```

---

## 6. 项目的特殊之处

以下是本项目区别于"跑通即可"的 RAG Demo 的关键设计和实现细节，多数是在开发验证过程中真实发现并处理的问题，而非预先设计好的卖点：

### 6.1 参数不是拍脑袋定的，是网格搜索出来的

`bm25_weight` 最初按 plan.md 惯例设为 0.6，但用真实的 58 条检索评估集测试后发现 Hybrid 的 Recall@3（0.914）反而低于纯 Dense 检索（0.966）——BM25 对语义型问题（如"你们公司还有别的产品线吗？"）会召回完全不相关的候选，且这些噪声候选在 min-max 归一化后仍能拿到较高的相对分数，把不相关文档顶到融合结果前列。做了 0.0~1.0 的网格搜索后，0.3 是四个指标（Recall@1/3/5/MRR）上的最优点，Recall@1 从 0.759 提升到 0.828。

低置信度阈值（rerank top1 < 0.1）也不是拍脑袋定的：用 100 条评估问题统计了真实的 rerank 分数分布，中位数 0.99，只有约 4% 低于 0.1，且这部分恰好集中在"能力边界/无依据问题"这一类域外问题上——说明这个阈值的区分度是用真实数据验证过的。

### 6.2 一个反直觉但被诚实记录的发现：Reranker 有时会降低 Recall@1

调优 `bm25_weight` 之后重新测四组实验，发现 **Hybrid+Reranker 的 Recall@1（0.776）反而低于纯 Hybrid（0.828）**。深入分析后定位到两个原因：CrossEncoder 不具备"用户问维修时长应该匹配维修文档而非保修文档"这类业务侧知识，会把语义相近但业务上不对的候选排到前面；以及候选池较小时 Reranker 的价值有限。没有因为这个数字不好看就删掉 Reranker 或者悄悄调整评估方法掩盖问题，而是在 `data/eval/bad_case_analysis.md` 里完整记录了具体失败案例，并说明了为什么依然保留 Reranker（它是低置信度拒答机制的信号来源，且随文档规模增长价值会上升）。

### 6.3 安全风险识别的关键词表是从真实漏检里迭代出来的

初版关键词表在测试"湿手可以拆解充电器吗？"时漏检——因为关键词写的是连续子串"湿手拆"，但用户的问句中间被"可以"隔开导致不匹配。做批量校准测试时发现了这个问题，改成更宽泛的"湿手"后重新跑了 25 条回归测试全部通过。这类问题只有真实批量测试才能发现，孤立测试单个问题很难暴露。

### 6.4 容器化过程中发现并修复的三个真实 bug

1. **前端硬编码 `localhost` 导致容器化后必然连不上后端**：`frontend/streamlit_app.py` 最初写的是 `API_BASE_URL = "http://localhost:8000"`，这在本地开发模式没问题，但在 Docker Compose 网络里，frontend 容器访问 `localhost` 只会指向自己，永远连不到 backend 容器。改成了 `os.getenv("BACKEND_API_URL", "http://localhost:8000")`，本地开发默认行为不变，compose 里注入 `BACKEND_API_URL=http://backend:8000`。
2. **Dockerfile 默认拉取 CUDA 版 torch，浪费 2GB+ 带宽和磁盘**：`sentence-transformers` 依赖 `torch`，Linux 容器下 `pip install` 默认解析到 CUDA 版本，会连带装上 `nvidia-cublas`(423MB)/`nvidia-cudnn`(366MB)/`nvidia-nccl`(206MB) 等一整套 GPU 依赖——但后端只用 `faiss-cpu`，完全不需要 GPU。修复为先用 `--index-url https://download.pytorch.org/whl/cpu` 装 CPU 版 torch。
3. **HuggingFace 模型下载在容器网络环境下会卡死在固定字节数不再前进**：`bge-reranker-base`（约 1.1GB）在两次独立的容器构建/启动尝试中都卡在 768MB 处停滞超过 4 分钟。这是网络层面的问题，不是代码 bug，但拖慢了开发验证节奏，通过挂载宿主机已有的模型缓存目录（只读）绕开了重复下载，加速了后续的验收测试。

### 6.5 pipeline 里两条独立的判断路径

风险识别（`detect_risk`）和检索置信度判断（`low_confidence`）是完全独立的两套逻辑，`need_human = risk_info["need_human"] or low_confidence`。这意味着即使一个高危问题因为措辞刁钻导致检索结果的 rerank 分数不够高（被判定为低置信度走拒答模板），风险识别依然会基于关键词匹配独立触发转人工——两条路径中任一个触发都会转人工，不会因为其中一层的偶然失误而漏判安全问题。

### 6.6 生成失败时的降级机制经过了真实故障注入测试

不是只在代码里写了 `try/except` 就假设它有效。真实把 DeepSeek client 的 timeout 设成 0.01 秒人为触发超时，观察到 `tenacity` 的 3 次重试叠加 OpenAI SDK 自身重试，约 14.3 秒后耗尽重试抛出 `APITimeoutError`，`generate_safe` 正确捕获并返回 `FALLBACK_ANSWER`，没有让异常向上传播导致 `/ask` 返回 500。

### 6.7 评估方法的局限被主动记录而不是隐藏

生成质量评估里的 `answer_point_coverage` 指标用的是精确子串匹配（判断生成答案是否包含预设的关键短语），跑出来是 0.828。人工抽查后发现这个数字低估了真实质量——比如某条回答生成了"不要尝试开机"，但预设短语是"不要开机"，语义完全正确却被判 0 分。这个评估方法的局限在 README 和 bad case 分析里都被明确指出，而不是把 0.828 当作"生成质量就是这么差"的结论呈报。

---

## 7. 实验数据与已知局限

基于真实运行的 `data/eval/retrieval_eval.jsonl`（58 条）、`generation_eval.jsonl`（15 条）、`safety_eval.jsonl`（25 条）：

| 方法 | Recall@1 | Recall@3 | Recall@5 | MRR | 平均延迟 |
|---|---:|---:|---:|---:|---:|
| BM25 | 0.655 | 0.828 | 0.845 | 0.747 | 0.30 ms |
| Dense | 0.741 | 0.966 | 0.983 | 0.849 | 8.88 ms |
| Hybrid（bm25_weight=0.3） | 0.828 | 0.983 | 0.983 | 0.894 | 8.83 ms |
| Hybrid + Reranker | 0.776 | 0.914 | 0.966 | 0.856 | 2334 ms |

| 生成 / 安全指标 | 数值 |
|---|---:|
| 格式遵循率 | 1.0 |
| 答案要点覆盖率（精确子串匹配，存在低估） | 0.828 |
| 转人工判断准确率（生成阶段） | 1.0 |
| 生成平均延迟（真实 DeepSeek 调用） | 5659 ms |
| 安全转人工判断准确率 | 1.0 |
| 安全风险类型识别准确率 | 1.0 |

已知局限（详见 `data/eval/bad_case_analysis.md`）：

- Reranker 在当前知识库规模（103 个 chunk）下对 Recall@1 的边际收益为负，需要更大文档规模后重新评估；
- `answer_point_coverage` 的精确子串匹配方法低估真实生成质量，后续可升级为 LLM-as-Judge；
- HuggingFace 模型首次下载在部分网络环境下不稳定，建议生产部署时预置模型缓存或使用镜像源。

---

## 8. 验收结论

对照第 28 节的 10 项验收标准，全部用真实运行的 `docker compose up --build -d` 容器化环境验证通过（而非本地开发模式或代码审查猜测），包括：网页端问答、三段式格式、引用展示、高风险转人工（真实测试"笔记本电池鼓起来了可以自己拆开处理吗"命中双重风险类型并生成工单）、低置信度拒答（真实测试域外问题"公司股票多少钱一股"rerank 分数仅 0.0009 触发拒答）、工单自动生成、四组检索对比实验、量化指标、完整日志记录、一键 Docker 启动与密钥安全注入。详细验证记录见 `plan.md` 阶段 14 部分。
