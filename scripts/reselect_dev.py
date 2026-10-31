"""Reuse unchanged pairwise dev scores after an audited metadata-only correction."""
import hashlib
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from app.evaluation.eval_retrieval import load_jsonl
from scripts.retrieval_experiment import select_dev

if __name__=='__main__':
    samples=load_jsonl('data/eval/v1/dev.jsonl')
    old=load_jsonl('data/eval/runs/dev-fp32/rows.jsonl')
    previous={r['query_id']:r for r in old}
    rows=[]
    for s in samples:
        row=previous[s['query_id']]
        assert s['question']==row['question'] and s['gold_chunk_ids']==row['gold_chunk_ids']
        assert s['split']=='dev'
        rows.append(dict(s,candidates=row['candidates'],retrieval_ms=row['retrieval_ms']))
    chosen,grid,baseline=select_dev(rows)
    target=Path('data/eval/runs/dev-fp32-reviewed');target.mkdir(exist_ok=True)
    (target/'rows.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows),encoding='utf-8')
    settings={'model':'BAAI/bge-reranker-base','dtype':'float32','pool':chosen['pool'],'threshold':chosen['threshold'],
              'dev_sha256':hashlib.sha256(Path('data/eval/v1/dev.jsonl').read_bytes()).hexdigest(),
              'selection':'dev non-inferiority constraints then balanced evidence/refusal score; smallest pool within .01',
              'selected_dev_metrics':chosen['metrics'],'baseline_dev_metrics':baseline,
              'provenance':'reused unchanged FP32 pair scores; only annotation protocol metadata and dev-derived safety routing updated'}
    config=Path('configs/experiments/dev-fp32-reviewed.json')
    if config.exists():raise SystemExit('Do not overwrite frozen configuration')
    config.parent.mkdir(parents=True,exist_ok=True)
    config.write_text(json.dumps(settings,ensure_ascii=False,indent=2),encoding='utf-8')
    (target/'report.json').write_text(json.dumps({'frozen':settings,'dev_grid':grid},ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(settings,ensure_ascii=True,indent=2))
