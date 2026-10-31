# 性能与成本验收

## 正式配对复测

证据：`data/eval/runs/performance-confirmation/report.json` 和 rows.jsonl。Windows、RTX 3060 Laptop 6GB、CUDA、CPU 4 线程、并发 1；89 个 dev 问题，每模型预热 3 题，重复 3 轮，每精度 267 次计时。相同问题共用 20 条候选，问题及每题精度顺序随机化，两模型同时驻留、CUDA 同步。测量时安装、训练和 HTTP 回放均已结束。

| 精度 | 检索＋重排 P50 | 检索＋重排 P95 | 重排 P95 | dev MRR@10 |
|---|---:|---:|---:|---:|
| FP32 | 106.0 ms | 172.5 ms | 166.0 ms | 0.9037 |
| FP16 | 46.2 ms | 65.4 ms | 58.8 ms | 0.9037 |

P95 降低 **62.1%**，dev Hit@1、Hit@5、Recall@5、MRR 一致。这是组件耗时，排除 LLM，不能写成整个问答 API 加速 62.1%。267 次是重复计时数，独立问题为 89 个。

先前顺序探索测得 67.1%，最终使用控制条件更好的配对复测 62.1%。历史 58 题候选数实验见 candidate-count-isolated；新 dev 比较 10/20/30，多证据召回与拒答约束下保留 20。未使用答案缓存；RRF、条件重排、输入改造是未实施的可选项。

## 启动与在线耗时

index-load-profile.json：新进程首次索引和元数据读取 1.78 ms，后续约 1.0–1.1 ms，OS 缓存未知，不是模型冷启动。dev-benchmark-float16/report.json 模型初始化约 42.3 s，不混入预热请求 P95。

冻结 M1 的 12 题在线抽样：选定方案总 P50 250.0 ms、P95 5105.4 ms，LLM P95 4788.0 ms，串行约 0.55 请求/s，服务失败 0/12。混合包含不调用生成的路径，不能推断负载容量。基线同批 P95 6700.6 ms，生成和缓存有波动，差异不能归因于 FP16。

## 成本

该 12 次分层抽样共输入 3961 token（缓存 2816）、输出 2457 token。按运行时快照 configs/pricing/deepseek-flash-2026-10-31.json 估算约 0.0016544 美元，等比例每千次同类请求约 0.1379 美元。公式为 (缓存输入×命中价＋未缓存输入×未命中价＋输出×输出价)/1e6。这是 token 估算，非账单或真实流量成本承诺。

来源：[DeepSeek 官方价格](https://api-docs.deepseek.com/quick_start/pricing/)，快照日期 2026-10-31，复现时重新核对。默认关闭 thinking，参见 [官方说明](https://api-docs.deepseek.com/guides/thinking_mode/)。

## 决策

采用 GPU FP16、CPU FP32 的 auto 策略，20 候选和 dev 校准阈值。冻结 test_v1 排序持平，达到本轮性能目标。GPU 温度、频率和系统负载影响结果；CPU Docker 演示不享有此 CUDA 收益。训练采用与否见 [训练报告](reranker_training_report.md)。
