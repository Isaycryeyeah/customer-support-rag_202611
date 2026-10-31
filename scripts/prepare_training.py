"""Mine random/hard negatives from TRAIN queries only, excluding all reviewed positives."""
import argparse
import hashlib
import json
from pathlib import Path
import random
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from app.evaluation.eval_retrieval import load_jsonl


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input',default='data/eval/v1/train.jsonl')
    p.add_argument('--output',default='data/training/v1')
    args=p.parse_args()
    rows=load_jsonl(args.input)
    if not rows or any(r['split']!='train' for r in rows):raise SystemExit('TRAIN only')
    out=Path(args.output)
    if out.exists():raise SystemExit('Use a new version directory')
    out.mkdir(parents=True)
    from app.config import CHUNKS_PATH,DENSE_MODEL_NAME,FAISS_INDEX_PATH,FAISS_META_PATH,BM25_WEIGHT
    from app.retrieval.bm25_retriever import BM25Retriever
    from app.retrieval.dense_retriever import DenseRetriever
    from app.retrieval.hybrid_retriever import HybridRetriever
    chunks=load_jsonl(CHUNKS_PATH);ids=[c['chunk_id'] for c in chunks]
    hybrid=HybridRetriever(BM25Retriever(CHUNKS_PATH),DenseRetriever(CHUNKS_PATH,DENSE_MODEL_NAME,FAISS_INDEX_PATH,FAISS_META_PATH),BM25_WEIGHT)
    rng=random.Random(42);seen=set();examples=[]
    for row in rows:
        if not row['answerable'] or row['group_id'] in seen:continue
        seen.add(row['group_id'])
        gold=set(row['gold_chunk_ids'])
        hard=[c['chunk_id'] for c in hybrid.search(row['question'],top_k=30,candidate_k=50) if c['chunk_id'] not in gold][:2]
        random_neg=rng.sample([i for i in ids if i not in gold],2)
        assert not gold.intersection(hard+random_neg)
        examples.append({'query_id':row['query_id'],'group_id':row['group_id'],'question':row['question'],
                         'split':'train',
                         'positives':sorted(gold),'hard_negatives':hard,'random_negatives':random_neg,
                         'source':'AI-reviewed synthetic TRAIN queries; full-KB positive annotations'})
    data=''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in examples)
    (out/'queries.jsonl').write_text(data,encoding='utf-8')
    manifest={'input_sha256':hashlib.sha256(Path(args.input).read_bytes()).hexdigest(),
              'queries_sha256':hashlib.sha256((out/'queries.jsonl').read_bytes()).hexdigest(),'seed':42,
              'independent_training_groups':len(examples),
              'pairs_per_negative_strategy':sum(len(r['positives'])+2 for r in examples),
              'all_reviewed_positives_excluded_from_negatives':True,
              'limitations':['AI annotations may miss relevant positives; hard negatives are not human-verified.',
                            'Fewer than 500 independent groups: feasibility and learning-curve experiment, not a large-data claim.']}
    (out/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(manifest,indent=2),flush=True)


if __name__=='__main__':main()
