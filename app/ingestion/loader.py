from datetime import date
from pathlib import Path
from typing import List, Dict


def load_markdown_docs(raw_dir: str) -> List[Dict]:
    """加载 Markdown 文档。"""
    docs = []
    raw_path = Path(raw_dir)

    for file_path in sorted(raw_path.glob("*.md")):
        text = file_path.read_text(encoding="utf-8")
        mtime = date.fromtimestamp(file_path.stat().st_mtime).isoformat()

        docs.append({
            "doc_id": file_path.stem,
            "title": file_path.stem.replace("_", " "),
            "source": "manual",
            "source_path": str(file_path),
            "updated_at": mtime,
            "content": text
        })

    return docs
