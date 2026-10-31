# 实验索引与版本边界

所有目录记录各自运行时的配置和结果，不事后改写旧汇总。当前项目验收见 `final-acceptance.json`；旧 M1/M2 文件中的 human_review_complete=False 等字段是当时状态，四条实际人工评分单独追加于 human-review-status.json，不代表全量数据已经人工验证。

| 目录／文件 | 用途 |
|---|---|
| baseline/ | 修改前环境和文件快照 |
| candidate-count-* | 历史开发集候选数量探索，非独立测试 |
| dev-* | v1 dev 参数、阈值、精度与模型选择 |
| performance-confirmation/ | 正式三轮同题配对 FP32/FP16 计时，简历采用 62.1% |
| m1-test-baseline、m1-test-selected | 冻结 test_v1 检索对照 |
| m1-online-*、m1-summary.json | 冻结 M1 在线抽样和当时里程碑 |
| m2-dev-*、m2-test-*、m2-summary.json | 训练 checkpoint 对照与冻结 test_v2，模型未采用 |
| training-*/ | 权重元信息、训练曲线与配置，不含大权重 |
| negative-audit/ | train 负例 AI 诊断，非人工真值 |
| clean-study-test-*、clean-study-summary.json | 一轮清洗补充实验，最终 test_v3 为 clean-study-v1 的 65 题 |
| trained-inference-comparison/ | 开发集单轮训练模型推理效率诊断，非三轮性能复测 |
| training-category-analysis.json | test_v2 事后分类描述，不用于追加调参 |
| docker-validation-*/ | 不同部署版本的真实 SDK／HTTP 验收 |
| human-review-status.json | 用户实际评价四个旧输出，不确定项保留 null |
| human-feedback-regression/ | 修复后四题真实 HTTP 回归，不是盲测或新人工分 |
| final-regressions.xml | 最终 56 项回归测试 |
| final-artifact-verification.json | 数据哈希、分组隔离、同候选池校验 |
| frozen-input-guard.json | 修改测试输入会在加载模型前被拒绝 |
| final-runtime-integrity.json | 当前容器 35 个源码／配置文件与工作区一致 |
| final-secret-scan.json | Git 跟踪内容不含配置中的真实凭证 |

M1 的原始风险代码与当前不同，旧冻结配置的源码校验失败是预期保护；复现历史结果需检出相应源码。test_v1/v2/v3 已查看，重放均不再是盲测。模拟问题、AI 审核、小样本、单种子与共享基础事实的限制见 docs/evaluation_protocol.md 和 docs/reranker_training_report.md。
