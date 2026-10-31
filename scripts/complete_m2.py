"""Record honest training outcomes, clustered paired intervals, and the frozen adoption decision."""
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from app.evaluation.eval_retrieval import load_jsonl,retrieval_metrics


def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))


if __name__=='__main__':
    target=Path('data/eval/runs/m2-summary.json')
    if target.exists():raise SystemExit('Do not overwrite recorded M2 results')
    names=['pretrained','random','hard','hard-60']
    results={n:read(f'data/eval/runs/m2-test-{n}/report.json')['selected'] for n in names}
    decision=read('configs/experiments/m2-final-decision.json')
    chosen=decision['dev_selected_name'];base=results['pretrained'];candidate=results[chosen]
    gain=candidate['mrr@10']-base['mrr@10']
    adopted=(chosen!='pretrained' and gain>=.005 and
             candidate['effective_recall@5']>=base['effective_recall@5']-.02 and
             candidate['refusal']['recall']>=base['refusal']['recall'])
    import numpy as np
    by_name={n:{r['query_id']:r for r in load_jsonl(f'data/eval/runs/m2-test-{n}/rows.jsonl')} for n in names}
    deltas=defaultdict(list)
    for qid,row in by_name['pretrained'].items():
        if not row['answerable']:continue
        def rr(n):
            r=by_name[n][qid]
            ordered=sorted(r['candidates'][:20],key=lambda c:c['rerank_score'],reverse=True)
            return retrieval_metrics([c['chunk_id'] for c in ordered],r['gold_chunk_ids'])['mrr@10']
        deltas[row['group_id']].append(rr(chosen)-rr('pretrained'))
    groups=sorted(deltas);rng=np.random.default_rng(20261031)
    bootstrap=[float(np.mean([v for g in rng.choice(groups,size=len(groups),replace=True) for v in deltas[g]])) for _ in range(2000)]
    ci=[float(np.percentile(bootstrap,p)) for p in (2.5,97.5)]
    training={}
    for name in names[1:]:
        source=Path('models/reranker-'+name)
        report=read(source/'training_report.json')
        assert report['training_groups']>0 and report['trainable_parameters']>0
        assert hashlib.sha256((source/'adapter.pt').read_bytes()).hexdigest()==report['adapter_sha256']
        out=Path('data/eval/runs/training-'+name);out.mkdir(parents=True,exist_ok=True)
        for filename in ('training_report.json','adapter.json','config.json'):
            (out/filename).write_bytes((source/filename).read_bytes())
        training[name]=report
    if adopted:
        cfg=read(f'configs/experiments/m2-final-{chosen}.json')
        deployment={'reranker_model':cfg['model'],'reranker_dtype':'auto','rerank_k':cfg['pool'],
                    'threshold':cfg['threshold'],'training_model_adopted':True,
                    'selection_record':'data/eval/runs/m2-summary.json'}
        Path('configs/deployment.json').write_text(json.dumps(deployment,indent=2),encoding='utf-8')
    summary={'milestone':'M2','feasibility_training_complete':True,'human_review_complete':False,
             'dev_selected_name':chosen,'trained_model_adopted':bool(adopted),
             'test_results':results,'training':training,'mrr_gain':gain,
             'paired_group_bootstrap_95_interval':ci,'bootstrap_iterations':2000,'bootstrap_seed':20261031,
             'bootstrap_scope':'Conditional on this synthetic test; zero observed pair differences do not establish population equivalence.',
             'decision_rule':decision['test_acceptance_rule'],
             'decision':'adopt dev-selected model' if adopted else 'retain the M1 pretrained model; held-out gain did not meet the declared gate',
             'limitations':['232 full-data training evidence groups, not 500 independent human-verified questions.',
                            'Partial final-layer/head training, single seed, simulated and AI-reviewed data.',
                            'Keyword risk recognition remains incomplete on held-out labels.']}
    target.write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({'M2':True,'dev_selected':chosen,'adopted':bool(adopted),'test_mrr_gain':gain,'paired_interval':ci},indent=2))
