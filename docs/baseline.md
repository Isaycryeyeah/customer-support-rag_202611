# 基线记录

日期：2026-10-31。原始文件哈希和环境信息保存在 `data/eval/runs/baseline/environment.json`，于本轮业务代码修改前捕获。

## 环境

- Windows，Python 3.12.0。
- NVIDIA GeForce RTX 3060 Laptop GPU，6GB 显存。
- 当前 PyTorch 2.6.0+cu124，Sentence Transformers 5.2.0，Transformers 5.9.0，FAISS CPU 1.13.2。
- 缓存 Dense revision：`7999e1d3359715c523056ef9478215996d62a620`。
- 缓存 Reranker revision：`2cfc18c9415c912f9d8155881c133215df768a70`。
- `requirements.windows.observed.txt` 保存项目直接依赖和当前平台活动传递依赖的版本快照；尚未完成全新环境安装验证。

临时 `.venv` 使用 `--system-site-packages`。全环境 `pip check` 报告 browser-use/rich、opencv-python/numpy、tensorflow-intel/protobuf 冲突，这些属于机器上的其他应用依赖。不得通过本项目的单元测试通过来宣称全环境依赖健康，也不直接修改全局依赖修复这些冲突。

## 数据与已有证据

- 24 篇模拟品牌文档，103 个片段。
- 历史检索 58 条、生成 15 条、安全 25 条，均属于开发／回归数据。
- 新基线保留文件哈希；当前目录尚无 Git 提交记录，未宣称已有源码版本发布或开源贡献。
- 本轮确定性回归测试、实际 HTTP 检查与性能开发实验分别保存，避免混用。
- 先前通过 curl 调用 DeepSeek 成功，后端 SDK 在执行环境中出现 WinError 10013；两者属于不同执行条件，不能合并宣称端到端成功。

## 索引一致性

索引旁增加 `faiss.manifest.json`，保存 chunks 文件哈希、模型标识、可获得的 revision、向量维度和片段数。旧索引缺少清单或清单失配时重建；加载后校验元数据、向量数量和维度。索引与 pickle 文件属于可信本机生成物，不提供外部上传 pickle 的入口。

## 复现入口

```powershell
python -m pytest -q
python scripts/run_eval.py --output data/eval/runs/unique-eval-name
python scripts/benchmark.py --output data/eval/runs/unique-benchmark-name --repeats 3
```

现有缓存完整时可设置 `HF_HUB_OFFLINE=1` 和 `TRANSFORMERS_OFFLINE=1`。在线生成另外使用 `--with-generation`，且需要后端运行环境具有 SDK 联网权限。

## 未完成事项

- 干净依赖环境的安装与完整复现。
- 本轮在线后端成功生成答案的验收；当前只能确认故障被正确表达。
- 新独立问题集标注及最终测试。
- 改动后的 Docker 环境验收。
- 领域重排序实际训练与消融。
