import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.ingestion.loader import load_markdown_docs
from app.ingestion.cleaner import clean_text
from app.ingestion.chunker import build_chunks, extract_title
from app.config import CHUNK_SIZE, CHUNK_OVERLAP, BASE_DIR


def main():
    docs = load_markdown_docs(str(BASE_DIR / "data" / "raw_docs"))

    for doc in docs:
        doc["content"] = clean_text(doc["content"])
        doc["title"] = extract_title(doc["content"]) or doc["title"]
        doc["source_path"] = str(Path(doc["source_path"]).relative_to(BASE_DIR)).replace("\\", "/")

    chunks = build_chunks(docs, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP)

    output_path = BASE_DIR / "data" / "processed" / "chunks.jsonl"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as f:
        for item in chunks:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"Saved {len(chunks)} chunks from {len(docs)} docs to {output_path}")

    metadata = {
        "doc_count": len(docs),
        "chunk_count": len(chunks),
        "chunk_size": CHUNK_SIZE,
        "chunk_overlap": CHUNK_OVERLAP,
        "docs": [
            {
                "doc_id": doc["doc_id"],
                "title": doc["title"],
                "source_path": doc["source_path"],
                "updated_at": doc["updated_at"]
            }
            for doc in docs
        ]
    }

    metadata_path = BASE_DIR / "data" / "processed" / "metadata.json"
    with metadata_path.open("w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    print(f"Saved metadata to {metadata_path}")


if __name__ == "__main__":
    main()
