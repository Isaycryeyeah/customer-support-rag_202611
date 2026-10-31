# 中文 3C 售后 RAG 问答系统

使用模拟品牌文档实现 BM25＋Dense／FAISS 检索、CrossEncoder 重排、DeepSeek 生成、安全提示、来源展示及工单建议。FastAPI 提供接口，Streamlit 提供页面，SQLite 留存请求和反馈。

[实施计划](plan.md)的正确性修复、独立评估、性能优化和领域训练可行性实验已完成。24 篇模拟文档、103 个切片；没有真实企业数据或完整工单流转。

## 已完成结果

- 隔离环境 **56 项测试通过**；Docker 内真实 SDK 问答、来源、人工建议和反馈验证通过。
- 构建 449 题／446 个证据组合组的 AI 生成与审核数据，按组隔离 train/dev/test；test_v1 44 题，test_v2 47 题。用户另实际评价了四个旧输出，未把 AI 标注冒充人工标注。
- test_v1 的预训练重排 MRR@10 为 0.9342，Hybrid 为 0.8504，排序分母 38 个有答案问题。
- RTX 3060 Laptop 上配对复测：FP16 将检索＋重排 P95 从 **172.5 ms 降至 65.4 ms（62.1%）**；dev 排序指标一致，独立 test_v1 排序亦持平。数字不含 LLM，也不适用于 CPU 容器。
- PyTorch/HF 实际训练最后两层与分类头，约 1477 万可训练参数；232 个训练组、875 对，随机负例／困难负例与 60 组学习曲线均完成。test_v2 未获得排序提升。
- 审计负例后完成一轮清洗实验（874 对），新增封存 test_v3 65 题；清洗模型仍与预训练模型持平，**保留预训练模型部署**。未为取得正结果继续调参。
- 四条人工反馈已修复：隐私投诉人工状态一致、明确域外无需售后人工、冒烟插座安全内容补全、来源按文档合并并保留切片定位。

## 运行

本机已运行验证容器：页面 <http://127.0.0.1:8503>，API <http://127.0.0.1:8003/docs>。

```powershell
$env:HF_CACHE_DIR='C:/Users/86136/.cache/huggingface/hub'
docker compose -p rag-validation -f docker-compose.validation.yml up -d --build
```

该方式要求已有完整模型缓存及有效 .env。普通 `docker compose up --build` 使用 8501/8000，可联网下载模型。

新环境本地启动：

```powershell
python -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements-dev.txt
Copy-Item .env.example .env
# 已有 .env 时不要覆盖；填入自己的模型凭证，勿提交。
.venv\Scripts\python.exe scripts/build_chunks.py
.venv\Scripts\python.exe scripts/build_index.py
.venv\Scripts\python.exe -m uvicorn app.main:app --host 127.0.0.1 --port 8000
# 另开终端
.venv\Scripts\python.exe -m streamlit run frontend/streamlit_app.py
```

Windows／Linux 分别有实测依赖锁文件；CUDA Torch 来源和平台说明见 [环境基线](docs/baseline.md)。模型 revision 已固定，首次下载需要网络。

## 业务状态与边界

| 返回类型 | HTTP | 行为 |
|---|---:|---|
| generated | 200 | 非空未截断生成，按文档列出来源，语义仍需质量评估 |
| low_confidence | 200 | 域内缺少依据时建议人工；明确域外只说明服务范围 |
| safety_refusal | 200 | 已识别高风险的确定性安全提示 |
| service_unavailable | 503 | 上游连接、限流、超时、空白或截断，保留日志 ID 与处理建议 |

`/health` 为存活、`/ready` 为就绪；`/feedback` 将反馈关联到日志。工单只是建议，没有实际派单。风险和领域规则有覆盖边界，测试通过不等于对任意输入都安全。引用编号校验不替代语义支持判断。

## 验证与实验

```powershell
.venv-clean\Scripts\python.exe -m pytest -q
.venv-clean\Scripts\python.exe -m pip check
# 新输出目录；真实调用 DeepSeek 并产生用量
.venv-clean\Scripts\python.exe scripts/validate_deployment.py --output data/eval/runs/new-deployment-check --capture-docker
.venv-clean\Scripts\python.exe scripts/validate_human_feedback.py --output data/eval/runs/new-feedback-replay
# 仅 dev 性能复测，不调用 LLM
.venv-clean\Scripts\python.exe scripts/confirm_performance.py --output data/eval/runs/new-performance-confirmation
```

旧 test 已查看；重放不能继续称为盲测，冻结源码哈希不同需检出原版本。每次使用新目录，不覆盖归档结果。

## 报告与代码入口

| 路径 | 内容 |
|---|---|
| [架构](docs/architecture.md)、[正确性](docs/correctness_report.md) | 状态、安全、来源、迁移与验收 |
| [评估协议](docs/evaluation_protocol.md)、[检索基线](docs/evaluation_baseline.md) | 数据分组、指标、冻结、人工与 AI 评分边界 |
| [性能报告](docs/performance_report.md) | 配对计时、线上耗时、token 成本与限制 |
| [训练报告](docs/reranker_training_report.md)、[模型说明](docs/model_card.md) | 数据、损失、学习曲线、消融、负结果与加载 |
| [人工反馈](docs/human_feedback_report.md) | 原始评分、重复来源原因与实际修复 |
| [简历描述](docs/resume.md) | 可用成果与面试证据 |
| app/、frontend/ | 服务、检索、生成、安全与页面 |
| scripts/、configs/ | 数据构造、训练、评估和冻结配置 |
| data/eval/runs/ | 配置、逐题结果、汇总、数据与模型哈希 |

数据、单次硬件会话和单种子训练均有限制，详见各报告。凭证、虚拟环境、运行数据库与模型权重不进入 Git。
