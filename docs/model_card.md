# 重排序模型说明

用途：模拟中文 3C 售后知识库的 query-document 相关性排序；不是生成模型，不直接判断电气安全或执行售后操作。

基础模型 BAAI/bge-reranker-base，revision `2cfc18c9415c912f9d8155881c133215df768a70`。使用 PyTorch BCEWithLogitsLoss 训练最后两层与分类头，seed=42。训练数据由模拟文档生成和 AI 审核，没有真实用户隐私数据；232 个有答案证据组，原始 875 对、清洗后 874 对。

## 本地适配权重

| 目录 | 实验 |
|---|---|
| models/reranker-random | 随机负例 |
| models/reranker-hard | 原始困难负例 |
| models/reranker-hard-60 | 60 组学习曲线 |
| models/reranker-hard-clean | 一轮负例清洗 |

adapter.pt 仅含可训练参数，加载时结合固定基础模型；adapter.json 保存参数清单和基础模型来源。加载使用 weights_only 并校验键名。权重不提交 Git，训练报告、配置、哈希保存于 data/eval/runs/training-*/，可按训练报告重新生成。没有向外上传或发布权重；基础模型使用约束以其原始模型说明为准。

## 效果与部署

test_v2 原始困难负例模型与预训练模型 MRR@10 同为 0.8943；test_v3 清洗模型与预训练同为 0.8909。均未达到预设采用门槛，默认 configs/deployment.json **保留预训练模型**。

临时加载训练模型进行演示：

```powershell
$env:RERANKER_MODEL_NAME='models/reranker-hard-clean'
$env:RERANKER_DTYPE='auto'
$env:LOW_CONFIDENCE_THRESHOLD='0.2'
```

阈值应以 configs/experiments/clean-study-clean.json 的实际冻结值为准（配置变量名称见 app/config.py）；不能将 sigmoid 分数当作校准后的正确概率。回退时移除上述覆盖并重启，恢复 deployment.json。

局限：单种子、AI 相关性标签、已知小知识库，多证据完整标注可能遗漏；不能宣称跨品牌、真实售后或新文档泛化。详细消融与未采用原因见 [训练报告](reranker_training_report.md)。
