"""Finalize the bounded data-quality study; do not chase a positive held-out result."""
import hashlib
import json
from pathlib import Path


def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))


if __name__=='__main__':
    target=Path('data/eval/runs/clean-study-summary.json')
    if target.exists():raise SystemExit('Study already recorded')
    decision=read('configs/experiments/clean-study-decision.json')
    assert hashlib.sha256(Path('data/eval/clean-study-v1/test_v3.jsonl').read_bytes()).hexdigest()==decision['test_sha256']
    metrics={n:read(f'data/eval/runs/clean-study-test-{n}/report.json')['selected'] for n in ('pretrained','raw','clean')}
    b=metrics['pretrained'];c=metrics['clean'];gain=c['mrr@10']-b['mrr@10']
    adopted=(decision['dev_selected_name']=='clean' and gain>=.005 and
             c['effective_recall@5']>=b['effective_recall@5']-.02 and c['refusal']['recall']>=b['refusal']['recall'])
    if adopted:
        cfg=read('configs/experiments/clean-study-clean.json')
        deployment={'reranker_model':cfg['model'],'reranker_dtype':'auto','rerank_k':cfg['pool'],'threshold':cfg['threshold'],
                    'training_model_adopted':True,'selection_record':str(target).replace('\\','/')}
        Path('configs/deployment.json').write_text(json.dumps(deployment,indent=2),encoding='utf-8')
    source=Path('models/reranker-hard-clean');out=Path('data/eval/runs/training-hard-clean');out.mkdir(parents=True,exist_ok=True)
    for name in ('training_report.json','adapter.json','config.json'):(out/name).write_bytes((source/name).read_bytes())
    report={'bounded_study_complete':True,'trained_model_adopted':bool(adopted),'decision':decision,'metrics':metrics,
            'mrr_gain_clean_vs_pretrained':gain,'training':read(source/'training_report.json'),
            'negative_audit':read('data/eval/runs/negative-audit/report.json'),
            'clean_data':read('data/training/v2/manifest.json'),'test_manifest':read('data/eval/clean-study-v1/manifest.json'),
            'limitations':['AI negative review is not human ground truth.','Cleaning uses verified random fallback as well as retained hard negatives.',
                          'Single seed and synthetic data; no statistical significance claim.','Old test_v1/test_v2 results and data were not rewritten.']}
    target.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({'complete':True,'adopted':bool(adopted),'mrr_gain':gain},indent=2))
