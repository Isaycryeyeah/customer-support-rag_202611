import json
import jieba
from pathlib import Path
from typing import List, Dict
from rank_bm25 import BM25Okapi


class BM25Retriever:
    def __init__(self, chunks_path: str):
        self.chunks = self._load_chunks(chunks_path)
        self.tokenized_corpus = [
            self._tokenize(item["content"]) for item in self.chunks
        ]
        self.index = BM25Okapi(self.tokenized_corpus) if self.tokenized_corpus else None

    def _load_chunks(self, path: str) -> List[Dict]:
        with Path(path).open("r", encoding="utf-8") as f:
            return [json.loads(line) for line in f]

    def _tokenize(self, text: str) -> List[str]:
        return list(jieba.cut(text))

    def search(self, query: str, top_k: int = 10) -> List[Dict]:
        if self.index is None or top_k <= 0:
            return []
        query_tokens = self._tokenize(query)
        scores = self.index.get_scores(query_tokens)

        ranked_indices = sorted(
            range(len(scores)),
            key=lambda i: scores[i],
            reverse=True
        )[:top_k]

        results = []
        for idx in ranked_indices:
            item = dict(self.chunks[idx])
            item["bm25_score"] = float(scores[idx])
            results.append(item)

        return results
