import re
from typing import List, Dict, Tuple, Optional


def split_text(
    text: str,
    chunk_size: int = 500,
    overlap: int = 80
) -> List[str]:
    """按照固定长度和 overlap 切分文本。"""
    chunks = []
    start = 0

    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunk = text[start:end]

        # 尽量在中文句号、问号、感叹号、换行处结束
        boundary = max(
            chunk.rfind("。"),
            chunk.rfind("？"),
            chunk.rfind("！"),
            chunk.rfind("\n")
        )

        if boundary > chunk_size * 0.5:
            end = start + boundary + 1
            chunk = text[start:end]

        chunks.append(chunk.strip())

        if end >= len(text):
            break

        start = max(0, end - overlap)

    return [c for c in chunks if c]


def extract_title(text: str) -> Optional[str]:
    """提取 Markdown 一级标题（# 标题）。"""
    match = re.search(r"^#\s+(.+)$", text, flags=re.MULTILINE)
    return match.group(1).strip() if match else None


def split_by_sections(text: str) -> List[Tuple[str, str]]:
    """按二级标题（## 标题）切分文档，返回 (section, section_text) 列表。

    一级标题之前或没有二级标题归属的内容归入 section="概述"。
    """
    lines = text.split("\n")
    sections: List[Tuple[str, List[str]]] = []
    current_section = "概述"
    current_lines: List[str] = []

    for line in lines:
        heading_match = re.match(r"^##\s+(.+)$", line)
        title_match = re.match(r"^#\s+(.+)$", line)

        if title_match:
            continue

        if heading_match:
            if current_lines:
                sections.append((current_section, current_lines))
            current_section = heading_match.group(1).strip()
            current_lines = []
        else:
            current_lines.append(line)

    if current_lines:
        sections.append((current_section, current_lines))

    return [
        (section, "\n".join(section_lines).strip())
        for section, section_lines in sections
        if "\n".join(section_lines).strip()
    ]


def build_chunks(
    docs: List[Dict],
    chunk_size: int = 500,
    overlap: int = 80
) -> List[Dict]:
    """将文档切分为带 metadata 的 chunks。"""
    all_chunks = []

    for doc in docs:
        title = extract_title(doc["content"]) or doc["title"]
        sections = split_by_sections(doc["content"])

        idx = 0
        for section, section_text in sections:
            for chunk in split_text(section_text, chunk_size=chunk_size, overlap=overlap):
                all_chunks.append({
                    "chunk_id": f"{doc['doc_id']}_{idx:04d}",
                    "doc_id": doc["doc_id"],
                    "title": title,
                    "section": section,
                    "source": doc.get("source", "manual"),
                    "source_path": doc["source_path"],
                    "updated_at": doc.get("updated_at"),
                    "content": chunk
                })
                idx += 1

    return all_chunks
