"""Retrieval metrics: distinguish any-hit from fraction of relevant chunks."""
import json
import math
from pathlib import Path


def load_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text(encoding='utf-8').splitlines() if line.strip()]


def hit_at_k(retrieved_ids, gold_ids, k):
    return float(bool(set(retrieved_ids[:k]) & set(gold_ids)))


def recall_at_k(retrieved_ids, gold_ids, k):
    gold = set(gold_ids)
    return len(set(retrieved_ids[:k]) & gold) / len(gold) if gold else None


def reciprocal_rank(retrieved_ids, gold_ids):
    return next((1. / rank for rank, item in enumerate(retrieved_ids, 1) if item in set(gold_ids)), 0.0)


def ndcg_at_k(retrieved_ids, gold_ids, k):
    gold = set(gold_ids)
    if not gold:
        return None
    seen = set()
    dcg = 0.0
    for rank, item in enumerate(retrieved_ids[:k], 1):
        if item in gold and item not in seen:
            dcg += 1. / math.log2(rank + 1)
        seen.add(item)
    ideal = sum(1. / math.log2(i + 1) for i in range(1, min(k, len(gold)) + 1))
    return dcg / ideal if ideal else 0.0


def retrieval_metrics(retrieved_ids, gold_ids):
    if not gold_ids:
        return {}
    result = {f'hit@{k}': hit_at_k(retrieved_ids, gold_ids, k) for k in (1, 3, 5)}
    result.update({f'recall@{k}': recall_at_k(retrieved_ids, gold_ids, k) for k in (1, 3, 5)})
    result.update({'mrr@10': reciprocal_rank(retrieved_ids[:10], gold_ids),
                   'ndcg@5': ndcg_at_k(retrieved_ids, gold_ids, 5)})
    return result


def summarize_rows(rows):
    eligible = [r['metrics'] for r in rows if r['metrics']]
    summary = {key: sum(r[key] for r in eligible) / len(eligible) for key in eligible[0]} if eligible else {}
    return dict(summary, sample_count=len(rows), answerable_count=len(eligible),
                unanswerable_count=len(rows) - len(eligible))


def _evaluate(search, eval_path, details_path=None):
    rows = []
    for index, sample in enumerate(load_jsonl(eval_path)):
        results = search(sample['question'])
        gold = sample.get('gold_chunk_ids', []) if sample.get('answerable', True) else []
        ids = [r['chunk_id'] for r in results]
        rows.append({'query_id': sample.get('query_id', str(index)), 'question': sample['question'],
                     'retrieved_ids': ids, 'gold_chunk_ids': gold, 'metrics': retrieval_metrics(ids, gold)})
    if details_path:
        path = Path(details_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in rows), encoding='utf-8')
    return summarize_rows(rows)


def evaluate_retriever(retriever, eval_path, top_k=10, details_path=None):
    return _evaluate(lambda q: retriever.search(q, top_k=top_k), eval_path, details_path)


def evaluate_hybrid_retriever(hybrid_retriever, eval_path, top_k=10, candidate_k=30, details_path=None):
    return _evaluate(lambda q: hybrid_retriever.search(q, top_k=top_k, candidate_k=candidate_k), eval_path, details_path)


def evaluate_hybrid_with_reranker(hybrid_retriever, reranker, eval_path, top_k=10, candidate_k=30, details_path=None, rerank_k=None):
    return _evaluate(lambda q: reranker.rerank(q, hybrid_retriever.search(q, top_k=rerank_k or candidate_k,
                                                                      candidate_k=candidate_k), top_k=top_k),
                     eval_path, details_path)
