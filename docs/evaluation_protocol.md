# 评估协议与版本边界

## 数据

知识库为 24 篇模拟中文售后文档、103 个切片。历史 58 条检索、15 条生成、25 条安全样本只作开发／回归。

`data/eval/v1/manifest.json` 记录 **449 个问题、446 个证据组合组**。AI 生成后经过全库证据复核、原文引用校验、多 gold 补充和近重复归组；它们不是人类逐条确认的真实客服请求。按组约 60/20/20 划分，再将 test 封为两阶段。

| 分区 | 问题数 | 用途 |
|---|---:|---|
| train | 269 | 训练和负例挖掘，其中 232 个有答案独立组用于排序训练 |
| dev | 89 | 参数、阈值、精度与 checkpoint 选择 |
| test_v1 | 44 | M1 冻结后测试，38 个有答案 |
| test_v2 | 47 | M2 冻结后测试，41 个有答案 |
| clean-study-v1/test_v3 | 65 | 一轮负例清洗补充实验，55 个有答案 |

同组改写不跨分区。划分种子、去重与原始字节 SHA256 留存于 manifest。分区间共享基础知识，因此评估已知库上的新问题组合，不能称为新文档或真实用户泛化。test_v3 排除原始目标证据组合及已审核 gold 子集，并过滤近似措辞，仍共享事实。`data/eval/v2/` 为未用于正式测试的早期草稿，不合并报数。

## 指标与选择

- Hit@k：至少命中一个 gold。Recall@k：命中 gold 数／全部 gold，逐问题宏平均。
- MRR@10：前十项首个相关项倒数；nDCG@5：多 gold 的归一化折扣增益。
- 无答案不混入排序分母，单独统计拒答 precision/recall、错误拒答和无依据应答。
- effective Recall@5 额外计入阈值／路由影响，不能混同纯排序 Recall。
- 安全识别与最终内容安全分开。`retrieval_experiment.py` 不执行最新领域预判和引用展示，其路由结果不是当前线上全链路指标。
- 生成故障保留在全量分母；有效生成子集另报，空白／截断不算成功或质量得分。

Hybrid BM25 权重固定 0.3，上游参数 50；未重新搜索融合权重或 RRF。dev 比较 10/20/30 候选、FP32/FP16，并校准每个模型阈值。冻结配置记录数据、风险源码和权重哈希，test 不参与本阶段选择。测试暴露后的修复属于新版本，旧结果和 gold 不重写、不冒充新盲测。

M1 风险代码、M2 风险补充版本及人工反馈后策略分别留痕。旧配置与新代码哈希不符应拒绝运行；复现历史版需检出相应源码，不移除保护。

## 人工与 AI 评分

事实支持、要点覆盖、引用支持各 0/1/2：0 为缺失或明显错误，1 为部分成立，2 为充分支持。危险指导、越权承诺和路由另标。词面分数只作辅助。

AI 审核留存模型、Prompt、证据和输出；同模型生成与审核存在相关错误。用户实际评价四个旧输出，见 [人工抽检](human_review.md)。这四题 AI/人工事实分一致 3/4，其余两项各 4/4；样本定向选取，不代表总体一致性。syn_t0350 的“不确定／均可”保存为 null。没有代填修改后人工分。

## 复现

```powershell
.venv-clean\Scripts\python.exe -m pytest -q
.venv-clean\Scripts\python.exe scripts/retrieval_experiment.py --input data/eval/v1/test_v2.jsonl --model BAAI/bge-reranker-base --dtype float16 --frozen configs/experiments/m2-final-pretrained.json --output data/eval/runs/new-test2-replay
```

每次使用新目录；重放已查看测试不再是盲测。训练见 [训练报告](reranker_training_report.md)。
