"""Paired inference-only dev benchmark; does not select or train another model."""
import argparse
import hashlib
import json
from pathlib import Path
import random
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    out = Path(args.output)
    if out.exists():
        raise SystemExit('Use a new benchmark directory')
    out.mkdir(parents=True)
    import numpy as np
    import torch
    from app.config import CHUNKS_PATH
    from app.retrieval.reranker import Reranker

    if not torch.cuda.is_available():
        raise SystemExit('CUDA is required for this FP16 comparison')
    torch.set_num_threads(4)
    chunks = {r['chunk_id']: r for r in map(json.loads, Path(CHUNKS_PATH).read_text(encoding='utf-8').splitlines())}
    cache = Path('data/eval/runs/dev-fp32-reviewed/rows.jsonl')
    samples = [json.loads(line) for line in cache.read_text(encoding='utf-8').splitlines()]
    assert all(r['split'] == 'dev' for r in samples)
    paths = {'pretrained': 'BAAI/bge-reranker-base', 'random': 'models/reranker-random',
             'hard': 'models/reranker-hard', 'clean': 'models/reranker-hard-clean'}
    models = {name: Reranker(path, dtype='float16') for name, path in paths.items()}
    def candidates(sample):
        return [{**chunks[c['chunk_id']], **c} for c in sample['candidates'][:20]]
    for sample in samples[:3]:
        for model in models.values():
            model.rerank(sample['question'], candidates(sample), top_k=10)
    rng = random.Random(20261031)
    rng.shuffle(samples)
    rows = []
    for sample in samples:
        names = list(models)
        rng.shuffle(names)
        for name in names:
            torch.cuda.synchronize()
            start = time.perf_counter()
            models[name].rerank(sample['question'], candidates(sample), top_k=10)
            torch.cuda.synchronize()
            rows.append({'query_id': sample['query_id'], 'model': name,
                         'rerank_ms': (time.perf_counter() - start) * 1000})
    report = {'scope': 'inference-only paired dev diagnostic; no model selection or training',
              'gpu': torch.cuda.get_device_name(), 'cpu_threads': 4, 'pool': 20,
              'dtype': 'float16', 'concurrency': 1, 'warmup_queries': 3, 'repeats': 1,
              'seed': 20261031, 'input_sha256': hashlib.sha256(cache.read_bytes()).hexdigest(),
              'model_paths': paths, 'models_resident_together': True,
              'summary': {name: {'count': sum(r['model'] == name for r in rows),
                                **{f'p{q}_ms': float(np.percentile([r['rerank_ms'] for r in rows if r['model'] == name], q))
                                   for q in (50, 95)}} for name in models},
              'limitations': ['One timing pass; hardware-session variation remains.',
                              'Excludes retrieval, model initialization and LLM generation.']}
    (out / 'rows.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in rows), encoding='utf-8')
    (out / 'report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report['summary'], indent=2))


if __name__ == '__main__':
    main()
