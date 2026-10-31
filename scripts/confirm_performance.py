"""Confirm FP16 speed with interleaved paired measurements after installation/training finish."""
import argparse
import hashlib
import json
from pathlib import Path
import random
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from app.evaluation.eval_retrieval import load_jsonl,retrieval_metrics,summarize_rows


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',required=True)
    args=p.parse_args()
    out=Path(args.output)
    if out.exists():raise SystemExit('Use a new confirmation directory')
    out.mkdir(parents=True)
    import numpy as np
    import torch
    from app.config import CHUNKS_PATH,DENSE_MODEL_NAME,FAISS_INDEX_PATH,FAISS_META_PATH
    from app.retrieval.bm25_retriever import BM25Retriever
    from app.retrieval.dense_retriever import DenseRetriever
    from app.retrieval.hybrid_retriever import HybridRetriever
    from app.retrieval.reranker import Reranker
    if not torch.cuda.is_available():raise SystemExit('This confirmation targets CUDA FP16 inference')
    torch.set_num_threads(4)
    samples=load_jsonl('data/eval/v1/dev.jsonl')
    hybrid=HybridRetriever(BM25Retriever(CHUNKS_PATH),DenseRetriever(CHUNKS_PATH,DENSE_MODEL_NAME,FAISS_INDEX_PATH,FAISS_META_PATH),.3)
    models={dtype:Reranker('BAAI/bge-reranker-base',dtype=dtype) for dtype in ('float32','float16')}
    for s in samples[:3]:
        candidates=hybrid.search(s['question'],top_k=20,candidate_k=50)
        for model in models.values():model.rerank(s['question'],candidates,top_k=10)
    rng=random.Random(20261031);rows=[]
    for repeat in range(3):
        order=list(samples);rng.shuffle(order)
        for s in order:
            torch.cuda.synchronize();start=time.perf_counter()
            candidates=hybrid.search(s['question'],top_k=20,candidate_k=50)
            torch.cuda.synchronize();retrieval_ms=(time.perf_counter()-start)*1000
            dtypes=list(models);rng.shuffle(dtypes)
            for dtype in dtypes:
                torch.cuda.synchronize();start=time.perf_counter()
                ranked=models[dtype].rerank(s['question'],candidates,top_k=10)
                torch.cuda.synchronize();rerank_ms=(time.perf_counter()-start)*1000
                ids=[c['chunk_id'] for c in ranked]
                rows.append({'query_id':s['query_id'],'repeat':repeat,'dtype':dtype,
                             'retrieval_ms':retrieval_ms,'rerank_ms':rerank_ms,'pipeline_ms':retrieval_ms+rerank_ms,
                             'metrics':retrieval_metrics(ids,s['gold_chunk_ids'] if s['answerable'] else []),
                             'retrieved_ids':ids})
        print(f'paired repeat {repeat+1}/3 complete',flush=True)
    summary={}
    for dtype in models:
        selected=[r for r in rows if r['dtype']==dtype]
        result=summarize_rows([r for r in selected if r['repeat']==0])
        result['timing_samples']=len(selected)
        for stage in ('retrieval_ms','rerank_ms','pipeline_ms'):
            result[stage]={f'p{q}':float(np.percentile([r[stage] for r in selected],q)) for q in (50,95)}
        summary[dtype]=result
    reduction=1-summary['float16']['pipeline_ms']['p95']/summary['float32']['pipeline_ms']['p95']
    report={'scope':'component latency on development queries; no LLM calls or whole-service throughput claim',
            'gpu':torch.cuda.get_device_name(),'cpu_threads':torch.get_num_threads(),'pool':20,'candidate_k':50,
            'warmup_queries_per_model':3,'repeats':3,'concurrency':1,'seed':20261031,
            'order':'randomized dtype order within each shuffled query, both models resident',
            'conditions':'installation, training and request replays finished; Docker services idle',
            'input_sha256':hashlib.sha256(Path('data/eval/v1/dev.jsonl').read_bytes()).hexdigest(),
            'source_sha256':{p.relative_to(ROOT).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in (ROOT/'app').rglob('*.py')},
            'summary':summary,'pipeline_p95_reduction_fraction':reduction,
            'quality_gate':summary['float16']['hit@5']>=summary['float32']['hit@5']-.02 and summary['float16']['recall@5']>=summary['float32']['recall@5']-.02}
    (out/'rows.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows),encoding='utf-8')
    (out/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(report['summary'],indent=2),flush=True)
    print('P95 reduction:',reduction,flush=True)


if __name__=='__main__':main()
