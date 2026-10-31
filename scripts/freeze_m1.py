"""Freeze a single M1 candidate from dev quality and sequential timing experiments."""
import hashlib
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from app.evaluation.eval_retrieval import load_jsonl
from scripts.retrieval_experiment import select_dev


if __name__=='__main__':
    outcomes={}
    for dtype,run in [('float32','dev-fp32-reviewed'),('float16','dev-fp16')]:
        rows=load_jsonl(f'data/eval/runs/{run}/rows.jsonl')
        chosen,grid,baseline=select_dev(rows)
        bench=json.loads(Path(f'data/eval/runs/dev-benchmark-{dtype}/report.json').read_text(encoding='utf-8'))
        chosen['p95_ms']=bench['summary'][str(chosen['pool'])]['pipeline_ms']['p95']
        outcomes[dtype]={'choice':chosen,'baseline':baseline}
    fp32=outcomes['float32'];fp16=outcomes['float16']
    s=fp16['choice']['metrics'];b=fp32['baseline']
    quality_ok=s['hit@5']>=b['hit@5']-.02 and (s['effective_recall@5'] or 0)>=(b['effective_recall@5'] or 0)-.02
    quality_ok &= (s['refusal']['recall'] or 0)>=(b['refusal']['recall'] or 0)
    faster=1-fp16['choice']['p95_ms']/fp32['choice']['p95_ms']
    dtype='float16' if quality_ok and faster>=.2 else 'float32'
    chosen=outcomes[dtype]['choice']
    common={'model':'BAAI/bge-reranker-base',
            'dev_sha256':hashlib.sha256(Path('data/eval/v1/dev.jsonl').read_bytes()).hexdigest(),
            'risk_detector_sha256':hashlib.sha256(Path('app/safety/risk_detector.py').read_bytes()).hexdigest(),
            'test_v1_sha256':hashlib.sha256(Path('data/eval/v1/test_v1.jsonl').read_bytes()).hexdigest(),
            'selection_scope':'DEV only; test file hash recorded without scoring',
            'acceptance_gate':'test Hit@5/effective Recall@5 drop <= .02, no-answer refusal recall non-inferior; otherwise retain baseline'}
    final=dict(common,dtype=dtype,pool=chosen['pool'],threshold=chosen['threshold'],
               dev_metrics=chosen['metrics'],dev_timing_speedup_fraction=faster,
               precision_quality_gate=bool(quality_ok))
    baseline=dict(common,dtype='float32',pool=20,threshold=.1)
    directory=Path('configs/experiments');directory.mkdir(parents=True,exist_ok=True)
    for name,data in [('m1-final.json',final),('m1-baseline.json',baseline)]:
        p=directory/name
        if p.exists():raise SystemExit('Do not overwrite frozen M1 configurations')
        p.write_text(json.dumps(data,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(final,ensure_ascii=True,indent=2))
