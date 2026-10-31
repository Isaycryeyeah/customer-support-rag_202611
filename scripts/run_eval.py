"""Versioned evaluation outputs. Offline by default; real generation is opt-in."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--retrieval-data', default='data/eval/retrieval_eval.jsonl')
    parser.add_argument('--generation-data', default='data/eval/generation_eval.jsonl')
    parser.add_argument('--safety-data', default='data/eval/safety_eval.jsonl')
    parser.add_argument('--output', required=True, help='New run directory, never overwrite previous results.')
    parser.add_argument('--dataset-role', choices=['legacy_dev', 'dev', 'test'], default='legacy_dev')
    parser.add_argument('--final-test', action='store_true', help='Confirm the test configuration was frozen before this run.')
    parser.add_argument('--with-generation', action='store_true', help='Call the configured LLM; may incur API charges.')
    args = parser.parse_args()
    if args.dataset_role == 'test' and not args.final_test:
        raise SystemExit('Final test requires --final-test after configuration freeze.')

    from app.config import (CHUNKS_PATH, DENSE_MODEL_NAME, FAISS_INDEX_PATH, FAISS_META_PATH,
                            RERANKER_MODEL_NAME, LLM_PROVIDER, BM25_WEIGHT, LOW_CONFIDENCE_THRESHOLD,
                            CANDIDATE_K, RERANK_K, CONTEXT_K)
    from app.evaluation.eval_retrieval import (load_jsonl, evaluate_retriever, evaluate_hybrid_retriever,
                                             evaluate_hybrid_with_reranker)
    from app.evaluation.eval_generation import evaluate_generation
    from app.evaluation.eval_safety import evaluate_safety
    from app.generation.llm_client import LLMClient
    from app.main_pipeline import CustomerSupportRAG
    from app.retrieval.bm25_retriever import BM25Retriever
    from app.retrieval.dense_retriever import DenseRetriever
    from app.retrieval.hybrid_retriever import HybridRetriever
    from app.retrieval.reranker import Reranker

    inputs = {key: str(ROOT / getattr(args, key)) for key in ('retrieval_data', 'generation_data', 'safety_data')}
    for name, path in inputs.items():
        if name == 'generation_data' and not args.with_generation:
            continue
        rows = load_jsonl(path)
        if not rows:
            raise SystemExit(f'Empty dataset: {path}')
        if any(row.get('split') is not None and row['split'] != args.dataset_role for row in rows):
            raise SystemExit(f'Unexpected split in {path}; do not mix dev/train/test.')
    if args.with_generation and LLM_PROVIDER == 'mock':
        raise SystemExit('Mock generation is a regression tool, not an evaluation of answer quality.')
    output = Path(args.output)
    if output.exists():
        raise SystemExit('Output exists; choose a new run directory.')
    output.mkdir(parents=True)
    bm25 = BM25Retriever(CHUNKS_PATH)
    dense = DenseRetriever(CHUNKS_PATH, DENSE_MODEL_NAME, FAISS_INDEX_PATH, FAISS_META_PATH)
    hybrid = HybridRetriever(bm25, dense, BM25_WEIGHT)
    reranker = Reranker(RERANKER_MODEL_NAME)
    report = {'created_at': datetime.now(timezone.utc).isoformat(), 'dataset_role': args.dataset_role,
              'configuration': {'dense': DENSE_MODEL_NAME, 'reranker': RERANKER_MODEL_NAME,
                                'bm25_weight': BM25_WEIGHT, 'threshold': LOW_CONFIDENCE_THRESHOLD,
                                'candidate_k': CANDIDATE_K, 'rerank_k': RERANK_K, 'context_k': CONTEXT_K},
              'knowledge_manifest': dense.manifest,
              'source_sha256': {p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
                                for folder in ('app', 'scripts') for p in (ROOT / folder).rglob('*.py')},
              'input_sha256': {name: hashlib.sha256(Path(path).read_bytes()).hexdigest()
                               for name, path in inputs.items() if name != 'generation_data' or args.with_generation},
              'generation_status': 'not_run', 'metrics': {}}
    tasks = {
        'bm25': lambda path: evaluate_retriever(bm25, inputs['retrieval_data'], details_path=path),
        'dense': lambda path: evaluate_retriever(dense, inputs['retrieval_data'], details_path=path),
        'hybrid': lambda path: evaluate_hybrid_retriever(hybrid, inputs['retrieval_data'], candidate_k=CANDIDATE_K, details_path=path),
        'hybrid_reranker': lambda path: evaluate_hybrid_with_reranker(hybrid, reranker, inputs['retrieval_data'],
                                                                   candidate_k=CANDIDATE_K, rerank_k=RERANK_K, details_path=path),
    }
    count = len(load_jsonl(inputs['retrieval_data']))
    for name, evaluate in tasks.items():
        start = time.perf_counter()
        metric = evaluate(output / f'{name}.jsonl')
        metric['loop_avg_latency_ms'] = (time.perf_counter() - start) * 1000 / count
        report['metrics'][name] = metric
        print(name, metric, flush=True)
    report['metrics']['safety_rules'] = evaluate_safety(inputs['safety_data'])
    if args.with_generation:
        rag = CustomerSupportRAG(hybrid, reranker, LLMClient(LLM_PROVIDER), threshold=LOW_CONFIDENCE_THRESHOLD,
                                 candidate_k=CANDIDATE_K, rerank_k=RERANK_K, context_k=CONTEXT_K)
        report['metrics']['generation'] = evaluate_generation(rag, inputs['generation_data'], output / 'generation.jsonl')
        report['generation_status'] = 'attempted; inspect generation_failure_rate and per-question outputs'
    (output / 'report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(output / 'report.json', flush=True)


if __name__ == '__main__':
    main()
