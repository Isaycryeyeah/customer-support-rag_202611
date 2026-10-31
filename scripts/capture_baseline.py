"""Capture a credential-free local baseline without importing the models."""
import argparse
import hashlib
import importlib.metadata as metadata
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def capture():
    files = sorted({*ROOT.glob('app/**/*.py'), *ROOT.glob('scripts/*.py'),
                    *ROOT.glob('data/raw_docs/*.md'), *ROOT.glob('data/eval/*.jsonl'),
                    ROOT / 'data/processed/chunks.jsonl', ROOT / 'requirements.txt'})
    packages = {}
    for name in ['fastapi', 'starlette', 'uvicorn', 'pydantic', 'rank-bm25', 'jieba',
                 'sentence-transformers', 'transformers', 'torch', 'faiss-cpu',
                 'numpy', 'pandas', 'streamlit', 'requests', 'python-dotenv',
                 'openai', 'httpx', 'tenacity', 'pytest']:
        try:
            packages[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            packages[name] = None
    snapshots = {}
    hub = Path.home() / '.cache/huggingface/hub'
    for name in ['BAAI/bge-small-zh-v1.5', 'BAAI/bge-reranker-base']:
        folder = hub / ('models--' + name.replace('/', '--')) / 'snapshots'
        snapshots[name] = sorted(p.name for p in folder.glob('*') if p.is_dir())
    try:
        gpu = subprocess.check_output(
            ['nvidia-smi', '--query-gpu=name,memory.total,driver_version', '--format=csv,noheader'],
            text=True, timeout=10).strip()
    except (OSError, subprocess.SubprocessError):
        gpu = None
    return {
        'captured_at': datetime.now(timezone.utc).isoformat(),
        'python': sys.version, 'platform': platform.platform(),
        'gpu': gpu, 'packages': packages, 'cached_model_revisions': snapshots,
        'files_sha256': {p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
                         for p in files if p.is_file()},
        'notes': ['Local environment inherits system packages; this is not a clean-install verification.',
                  'Historical evaluation numbers have not been reproduced.'],
    }


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', default='data/eval/runs/baseline/environment.json')
    args = parser.parse_args()
    output = ROOT / args.output
    if output.exists():
        raise SystemExit('Refusing to overwrite an existing baseline. Choose a new --output.')
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(capture(), ensure_ascii=False, indent=2), encoding='utf-8')
    print(output)
