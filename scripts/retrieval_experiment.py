"""Collect pairwise scores once; choose settings on dev, evaluate frozen settings on test."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from app.evaluation.eval_retrieval import load_jsonl, retrieval_metrics, summarize_rows
from app.evaluation.eval_generation import refusal_metrics
from app.safety.risk_detector import detect_risk


def summarize(rows,pool,threshold,hybrid_only=False):
    retrieval=[];decisions=[];effective=[];safety=[]
    for r in rows:
        subset=r['candidates'][:pool]
        ranked=sorted(subset,key=lambda c:c['rerank_score'],reverse=True) if not hybrid_only else subset
        ids=[c['chunk_id'] for c in ranked]
        gold=r['gold_chunk_ids'] if r['answerable'] else []
        metric=retrieval_metrics(ids,gold)
        retrieval.append({'metrics':metric})
        expected_safety=bool(set(r.get('risk_types',[])) & {'battery_safety','dangerous_repair'})
        actual_safety=detect_risk(r['question'])['risk_level']=='high'
        if expected_safety:safety.append(actual_safety)
        refused=not ranked or (not hybrid_only and ranked[0]['rerank_score']<threshold)
        kind='safety_refusal' if actual_safety else 'low_confidence' if refused else 'generated'
        decisions.append({'answerable':r['answerable'],'answer_type':kind,
                          'service_failure':False,'expected_safety':expected_safety})
        if r['answerable'] and not expected_safety:
            effective.append(metric.get('recall@5',0) if kind=='generated' else 0)
    summary=summarize_rows(retrieval)
    summary['effective_recall@5']=sum(effective)/len(effective) if effective else None
    summary['refusal']=refusal_metrics(decisions)
    summary['recognized_high_risk_fraction']=sum(safety)/len(safety) if safety else None
    return summary


def collect(samples,model_name,dtype='float32'):
    from app.config import CHUNKS_PATH,DENSE_MODEL_NAME,FAISS_INDEX_PATH,FAISS_META_PATH,BM25_WEIGHT
    from app.retrieval.bm25_retriever import BM25Retriever
    from app.retrieval.dense_retriever import DenseRetriever
    from app.retrieval.hybrid_retriever import HybridRetriever
    from app.retrieval.reranker import Reranker
    bm25=BM25Retriever(CHUNKS_PATH)
    dense=DenseRetriever(CHUNKS_PATH,DENSE_MODEL_NAME,FAISS_INDEX_PATH,FAISS_META_PATH)
    hybrid=HybridRetriever(bm25,dense,BM25_WEIGHT)
    reranker=Reranker(model_name,dtype=dtype)
    rows=[]
    for i,s in enumerate(samples):
        start=time.perf_counter()
        candidates=hybrid.search(s['question'],top_k=30,candidate_k=50)
        retrieval_ms=(time.perf_counter()-start)*1000
        rerank_start=time.perf_counter()
        scored=reranker.rerank(s['question'],candidates,top_k=30)
        rerank_ms=(time.perf_counter()-rerank_start)*1000
        scores={c['chunk_id']:c['rerank_score'] for c in scored}
        row=dict(s)
        row['candidates']=[{'chunk_id':c['chunk_id'],'hybrid_score':c['hybrid_score'],'rerank_score':scores[c['chunk_id']]} for c in candidates]
        row['retrieval_ms']=retrieval_ms
        row['rerank_30_ms']=rerank_ms
        row['bm25_ids']=[c['chunk_id'] for c in bm25.search(s['question'],top_k=10)]
        row['dense_ids']=[c['chunk_id'] for c in dense.search(s['question'],top_k=10)]
        rows.append(row)
        if (i+1)%20==0:print(f'scored {i+1}/{len(samples)}',flush=True)
    return rows


def select_dev(rows,pools=(10,20,30)):
    baseline=summarize(rows,20,.1)
    choices=[]
    for pool in pools:
        for threshold in (.005,.01,.03,.05,.1,.2,.4,.6,.8,.9,.95,.99):
            s=summarize(rows,pool,threshold)
            eff=s['effective_recall@5'] or 0
            refusal=s['refusal']['recall'] or 0
            score=.5*eff+.5*refusal
            choices.append({'pool':pool,'threshold':threshold,'score':score,'metrics':s})
    # Predeclared choice: maximize balanced grounding/refusal, choose smaller pool within 0.01.
    # Non-inferiority constraints are declared on dev before any final test is opened.
    feasible=[c for c in choices
              if c['metrics']['hit@5']>=baseline['hit@5']-.02
              and (c['metrics']['effective_recall@5'] or 0)>=(baseline['effective_recall@5'] or 0)-.02
              and (c['metrics']['refusal']['recall'] or 0)>=(baseline['refusal']['recall'] or 0)
              and (c['metrics']['refusal']['false_refusal_rate'] or 0)<=(baseline['refusal']['false_refusal_rate'] or 0)+.02]
    best=max(c['score'] for c in feasible)
    acceptable=[c for c in feasible if c['score']>=best-.01]
    chosen=min(acceptable,key=lambda c:(c['pool'],-c['score'],c['threshold']))
    return chosen,choices,baseline


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input',required=True)
    p.add_argument('--model',default='BAAI/bge-reranker-base')
    p.add_argument('--output',required=True)
    p.add_argument('--dtype',choices=['float32','float16'],default='float32')
    p.add_argument('--fixed-pool',type=int,choices=[10,20,30],help='Control candidate count during training ablations.')
    p.add_argument('--freeze',help='Write dev-selected model/pool/threshold before final test.')
    p.add_argument('--frozen',help='Read fixed settings for final test; no tuning.')
    args=p.parse_args()
    samples=load_jsonl(args.input)
    if not samples:raise SystemExit('Empty input')
    split={s['split'] for s in samples}
    if args.freeze and split!={'dev'}:raise SystemExit('Only dev may choose settings')
    if split=={'test'} and not args.frozen:raise SystemExit('Test requires --frozen')
    if split not in ({'dev'},{'test'}):raise SystemExit('Use a single dev or test split')
    out=Path(args.output)
    if out.exists():raise SystemExit('Do not overwrite an experiment')
    out.mkdir(parents=True)
    settings=json.loads(Path(args.frozen).read_text(encoding='utf-8')) if args.frozen else None
    if settings and split == {'test'}:
        partitions = {s.get('test_partition') for s in samples}
        if len(partitions) != 1 or None in partitions:
            raise SystemExit('Frozen tests require one explicit test_partition.')
        expected_hash = settings.get(f'{next(iter(partitions))}_sha256')
        if not expected_hash:
            raise SystemExit('Frozen configuration has no hash for this test partition.')
        if hashlib.sha256(Path(args.input).read_bytes()).hexdigest() != expected_hash:
            raise SystemExit('Test data changed after configuration freeze.')
    if settings and settings.get('risk_detector_sha256'):
        if hashlib.sha256((ROOT/'app/safety/risk_detector.py').read_bytes()).hexdigest()!=settings['risk_detector_sha256']:
            raise SystemExit('Safety routing changed after configuration freeze.')
    if settings and settings.get('adapter_sha256'):
        if hashlib.sha256((Path(args.model)/'adapter.pt').read_bytes()).hexdigest()!=settings['adapter_sha256']:
            raise SystemExit('Adapter weights changed after configuration freeze.')
    if settings and settings['model']!=args.model:raise SystemExit('Model differs from frozen configuration')
    if settings and settings.get('dtype','float32')!=args.dtype:raise SystemExit('dtype differs from frozen settings')
    rows=collect(samples,args.model,args.dtype)
    report={'created_at':datetime.now(timezone.utc).isoformat(),'input_sha256':hashlib.sha256(Path(args.input).read_bytes()).hexdigest(),
            'model':args.model,'dtype':args.dtype,'split':list(split),'baseline':summarize(rows,20,.1),
            'hybrid_without_refusal':summarize(rows,30,0,True),
            'scope':'synthetic known-KB query evaluation; no LLM answer quality in these routing scores'}
    report['retrieval_methods']={
        name:summarize_rows([{'metrics':retrieval_metrics(r[key],r['gold_chunk_ids'] if r['answerable'] else [])} for r in rows])
        for name,key in [('bm25','bm25_ids'),('dense','dense_ids')]
    }
    report['retrieval_methods']['hybrid']=report['hybrid_without_refusal']
    report['retrieval_methods']['hybrid_reranker']=report['baseline']
    if args.freeze:
        chosen,choices,baseline=select_dev(rows,(args.fixed_pool,) if args.fixed_pool else (10,20,30))
        settings={'model':args.model,'dtype':args.dtype,'pool':chosen['pool'],'threshold':chosen['threshold'],
                  'dev_sha256':report['input_sha256'],'selection':'dev non-inferiority (Hit@5/effective Recall@5 -2pp, refusal recall >= baseline, false refusal +2pp); maximize balanced score; pool tie within .01',
                  'selected_dev_metrics':chosen['metrics'],'baseline_dev_metrics':baseline}
        path=Path(args.freeze);path.parent.mkdir(parents=True,exist_ok=True)
        if path.exists():raise SystemExit('Frozen configuration already exists')
        path.write_text(json.dumps(settings,ensure_ascii=False,indent=2),encoding='utf-8')
        report['dev_grid']=choices
    if settings:report['selected']=summarize(rows,settings['pool'],settings['threshold']);report['frozen']=settings
    (out/'rows.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows),encoding='utf-8')
    (out/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in report.items() if k not in ('dev_grid','frozen')},ensure_ascii=True,indent=2),flush=True)


if __name__=='__main__':main()
