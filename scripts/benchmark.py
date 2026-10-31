"""Fair candidate-count experiment on a declared development dataset; no LLM calls."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import random
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.config import (BM25_WEIGHT, CHUNKS_PATH, DENSE_MODEL_NAME, FAISS_INDEX_PATH,
                        FAISS_META_PATH, RERANKER_MODEL_NAME)
from app.evaluation.eval_retrieval import load_jsonl, retrieval_metrics, summarize_rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', default='data/eval/retrieval_eval.jsonl')
    parser.add_argument('--dataset-role', choices=['legacy_dev', 'dev'], default='legacy_dev')
    parser.add_argument('--output', required=True)
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--candidate-k', type=int, default=50)
    parser.add_argument('--pools', type=int, nargs='+', default=[10, 20, 30])
    parser.add_argument('--seed', type=int, default=20261031)
    args = parser.parse_args()
    if args.repeats < 1 or not args.pools or min(args.pools) < 1 or args.candidate_k < max(args.pools):
        raise SystemExit('Positive repeats/pools required; candidate-k must cover all pools.')
    samples = load_jsonl(args.input)
    if not samples or any(s.get('split') in ('test', 'train') for s in samples):
        raise SystemExit('Use non-empty development data; this tuning entry point rejects train/test rows.')
    output = Path(args.output)
    if output.exists():
        raise SystemExit('Output exists; choose another run directory.')
    output.mkdir(parents=True)
    source_hashes = {p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
                     for folder in ('app', 'scripts') for p in (ROOT / folder).rglob('*.py')}

    import numpy as np
    import torch
    from app.retrieval.bm25_retriever import BM25Retriever
    from app.retrieval.dense_retriever import DenseRetriever
    from app.retrieval.hybrid_retriever import HybridRetriever
    from app.retrieval.reranker import Reranker

    cold = time.perf_counter()
    bm25 = BM25Retriever(CHUNKS_PATH)
    dense = DenseRetriever(CHUNKS_PATH, DENSE_MODEL_NAME, FAISS_INDEX_PATH, FAISS_META_PATH)
    hybrid = HybridRetriever(bm25, dense, BM25_WEIGHT)
    reranker = Reranker(RERANKER_MODEL_NAME)
    def sync():
        if torch.cuda.is_available():
            torch.cuda.synchronize()
    sync()
    cold_ms = (time.perf_counter() - cold) * 1000
    # Warm all candidate lengths before measurement; no query-result cache.
    for pool in args.pools:
        candidates = hybrid.search(samples[0]['question'], top_k=pool, candidate_k=args.candidate_k)
        reranker.rerank(samples[0]['question'], candidates, top_k=10)
    rows = []
    rng = random.Random(args.seed)
    for repeat in range(args.repeats):
        order = list(enumerate(samples))
        rng.shuffle(order)
        for index, sample in order:
            q, gold = sample['question'], sample.get('gold_chunk_ids', [])
            if not sample.get('answerable', True):
                gold = []
            sync()
            start = time.perf_counter()
            candidates = hybrid.search(q, top_k=max(args.pools), candidate_k=args.candidate_k)
            sync()
            retrieval_ms = (time.perf_counter() - start) * 1000
            pools = list(args.pools)
            rng.shuffle(pools)
            for pool in pools:
                subset = candidates[:pool]
                sync()
                start = time.perf_counter()
                contexts = reranker.rerank(q, subset, top_k=10)
                sync()
                rerank_ms = (time.perf_counter() - start) * 1000
                rows.append({
                    'query_id': sample.get('query_id', str(index)), 'question': q,
                    'repeat': repeat, 'pool': pool, 'gold_chunk_ids': gold,
                    'candidate_ids': [c['chunk_id'] for c in subset],
                    'retrieved_ids': [c['chunk_id'] for c in contexts],
                    'candidate_metrics': retrieval_metrics([c['chunk_id'] for c in subset], gold),
                    'metrics': retrieval_metrics([c['chunk_id'] for c in contexts], gold),
                    'candidate_recall': len({c['chunk_id'] for c in subset} & set(gold)) / len(set(gold)) if gold else None,
                    'top_score': contexts[0]['rerank_score'] if contexts else 0,
                    'retrieval_ms': retrieval_ms, 'rerank_ms': rerank_ms,
                    'pipeline_ms': retrieval_ms + rerank_ms,
                })
        print(f'repeat {repeat + 1}/{args.repeats} complete', flush=True)
    summary = {}
    for pool in args.pools:
        selected = [r for r in rows if r['pool'] == pool]
        unique = [r for r in selected if r['repeat'] == 0]
        metric = summarize_rows(unique)
        eligible = [r['candidate_recall'] for r in unique if r['candidate_recall'] is not None]
        metric['candidate_recall'] = float(np.mean(eligible)) if eligible else None
        metric['timing_samples'] = len(selected)
        for stage in ('retrieval_ms', 'rerank_ms', 'pipeline_ms'):
            metric[stage] = {'p50': float(np.percentile([r[stage] for r in selected], 50)),
                             'p95': float(np.percentile([r[stage] for r in selected], 95))}
        summary[str(pool)] = metric
    report = {
        'created_at': datetime.now(timezone.utc).isoformat(),
        'claim_scope': 'exploratory development experiment; not an independent test or end-to-end API latency',
        'config': vars(args), 'dataset_sha256': hashlib.sha256(Path(args.input).read_bytes()).hexdigest(),
        'knowledge_manifest': dense.manifest, 'cold_model_load_ms': cold_ms,
        'source_sha256': source_hashes,
        'device': str(reranker.model.device), 'torch_version': torch.__version__,
        'reranker_model': RERANKER_MODEL_NAME, 'bm25_weight': BM25_WEIGHT,
        'reranker_revision': getattr(reranker.model.model.config, '_commit_hash', None),
        'warmup_queries': 1, 'warmup_per_pool': 1, 'concurrency': 1,
        'cuda': torch.cuda.is_available(), 'cpu_threads': torch.get_num_threads(),
        'summary': summary,
    }
    (output / 'rows.jsonl').write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in rows), encoding='utf-8')
    (output / 'report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == '__main__':
    main()
