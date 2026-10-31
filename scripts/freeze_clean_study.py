"""Predeclare the single clean-negative candidate and v3 adoption gate."""
import hashlib
import json
from pathlib import Path


def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))


if __name__=='__main__':
    baseline=read('configs/experiments/m2-final-pretrained.json')
    raw=read('configs/experiments/m2-final-hard.json')
    clean=read('configs/experiments/m2-hard-clean.json')
    b=baseline['selected_dev_metrics'];c=clean['selected_dev_metrics']
    score=lambda m:.5*(m['effective_recall@5'] or 0)+.5*(m['refusal']['recall'] or 0)
    eligible=c['mrr@10']>=b['mrr@10']+.005 and score(c)>=score(b)-.005
    chosen='clean' if eligible else 'pretrained'
    test_hash=hashlib.sha256(Path('data/eval/clean-study-v1/test_v3.jsonl').read_bytes()).hexdigest()
    for name,cfg in [('pretrained',baseline),('raw',raw),('clean',clean)]:
        cfg['test_v3_sha256']=test_hash
        cfg['risk_detector_sha256']=hashlib.sha256(Path('app/safety/risk_detector.py').read_bytes()).hexdigest()
        weights=Path(cfg['model'])/'adapter.pt'
        if weights.exists():cfg['adapter_sha256']=hashlib.sha256(weights.read_bytes()).hexdigest()
        cfg['clean_study_dev_selected_name']=chosen
        p=Path(f'configs/experiments/clean-study-{name}.json')
        if p.exists():raise SystemExit('Do not overwrite frozen clean-study settings')
        p.write_text(json.dumps(cfg,ensure_ascii=False,indent=2),encoding='utf-8')
    decision={'dev_selected_name':chosen,'test_sha256':test_hash,
              'dev_mrr_pretrained':b['mrr@10'],'dev_mrr_clean':c['mrr@10'],
              'rule':'Clean model only: dev MRR gain >=.005 with balanced grounding/refusal within .005; adopt only with test_v3 MRR gain >=.005, effective Recall@5 drop <=.02 and refusal recall non-inferior.',
              'no_further_training_iterations':True}
    Path('configs/experiments/clean-study-decision.json').write_text(json.dumps(decision,indent=2),encoding='utf-8')
    print(json.dumps(decision,indent=2))
