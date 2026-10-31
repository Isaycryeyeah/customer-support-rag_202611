from typing import List, Dict


def min_max_normalize(items: List[Dict], score_key: str) -> Dict[str, float]:
    scores = [item.get(score_key, 0.0) for item in items]

    if not scores:
        return {}

    min_score = min(scores)
    max_score = max(scores)

    result = {}
    for item in items:
        chunk_id = item["chunk_id"]
        score = item.get(score_key, 0.0)

        if max_score == min_score:
            norm_score = 1.0
        else:
            norm_score = (score - min_score) / (max_score - min_score)

        result[chunk_id] = norm_score

    return result


class HybridRetriever:
    def __init__(self, bm25_retriever, dense_retriever, bm25_weight=0.3):
        self.bm25 = bm25_retriever
        self.dense = dense_retriever
        self.bm25_weight = bm25_weight
        self.dense_weight = 1.0 - bm25_weight

    def search(self, query: str, top_k: int = 10, candidate_k: int = 30) -> List[Dict]:
        if top_k <= 0 or candidate_k <= 0:
            return []
        bm25_results = self.bm25.search(query, top_k=candidate_k)
        dense_results = self.dense.search(query, top_k=candidate_k)

        bm25_scores = min_max_normalize(bm25_results, "bm25_score")
        dense_scores = min_max_normalize(dense_results, "dense_score")

        merged = {}

        for item in bm25_results + dense_results:
            chunk_id = item["chunk_id"]
            if chunk_id not in merged:
                merged[chunk_id] = dict(item)

            merged[chunk_id]["hybrid_score"] = (
                self.bm25_weight * bm25_scores.get(chunk_id, 0.0)
                + self.dense_weight * dense_scores.get(chunk_id, 0.0)
            )

        ranked = sorted(
            merged.values(),
            key=lambda x: x["hybrid_score"],
            reverse=True
        )

        return ranked[:top_k]
