# 环境与复现基线

日期：2026-10-31。修改前快照：`data/eval/runs/baseline/environment.json`。本地源码基线提交 `16b3f66`，数据与训练实现冻结提交 `b91343a`；后续改动通过本地 Git 留存，未推送远程。

## 实际验证环境

| 环境 | 用途 | 验证 |
|---|---|---|
| Windows / Python 3.12 / RTX 3060 Laptop 6GB | 训练、CUDA 性能、回归 | .venv-clean 独立安装，未继承系统包，pip check 通过 |
| Linux Docker / Python 3.10 / CPU | FastAPI、Streamlit、真实 DeepSeek | 容器启动、就绪、真实问答与反馈通过 |

Windows 最终锁定在 `requirements.windows.lock.txt`，Linux 实测锁定在 `requirements.linux.lock.txt`。`requirements.windows.observed.txt` 是早期继承环境快照。直接依赖与开发依赖分别见 requirements.txt、requirements-dev.txt。

Torch 为 2.6.0+cu124。由于供应商大文件下载不稳定，本机从已有可信安装重建 wheel，并校验 **10,933 个原始 RECORD 文件哈希**；它是本地重打包产物，不是供应商原始 wheel。脚本为 `scripts/repack_installed_wheel.py`，私有 wheel 与来源记录在 .venv-clean/wheelhouse/，不提交。新机器应从官方安装源获取平台匹配的 Torch，不能将本机缓存当作公开依赖源。

Dense revision：`7999e1d3359715c523056ef9478215996d62a620`；Reranker revision：`2cfc18c9415c912f9d8155881c133215df768a70`。Docker 基础镜像 digest 固定。模型、凭证、数据库与虚拟环境不进入 Git 或镜像构建上下文。

## 启动

```powershell
python -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements-dev.txt
Copy-Item .env.example .env
# 填自己的模型配置；已有 .env 时不要覆盖。
.venv\Scripts\python.exe scripts/build_chunks.py
.venv\Scripts\python.exe scripts/build_index.py
.venv\Scripts\python.exe -m uvicorn app.main:app --host 127.0.0.1 --port 8000
# 另开终端
.venv\Scripts\python.exe -m streamlit run frontend/streamlit_app.py
```

首次需要下载模型。缓存完整时可设置 HF_HUB_OFFLINE=1、TRANSFORMERS_OFFLINE=1，不影响 DeepSeek 联网。

本机已验证的缓存容器启动方式：

```powershell
$env:HF_CACHE_DIR='C:/Users/86136/.cache/huggingface/hub'
docker compose -p rag-validation -f docker-compose.validation.yml up -d --build
```

页面 http://127.0.0.1:8503，后端 http://127.0.0.1:8003。普通 docker compose 使用 8501/8000，可在线下载模型。SDK 网络问题已解决；容器验收见 `data/eval/runs/docker-validation-final/report.json`，人工反馈回放见 human-feedback-regression/report.json。

索引侧车记录语料、模型、维度与数量。仓库 .gitattributes 保持字节一致，实验清单校验文件原始字节哈希。
