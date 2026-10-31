"""Select the training candidate on dev before opening test_v2; keep the curve control diagnostic."""
import hashlib
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from app.evaluation.eval_retrieval import load_jsonl
from scripts.retrieval_experiment import select_dev


def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))


if __name__=='__main__':
    original=read('configs/experiments/m1-final.json')
    configs={'pretrained':dict(original)}
    for name in ('random','hard','hard-60'):
        configs[name]=read(f'configs/experiments/m2-{name}.json')
    # Recalibrate cached pair scores after explicitly recorded post-M1 routing fixes.
    for name,cfg in configs.items():
        run='dev-fp16' if name=='pretrained' else 'm2-dev-'+name
        choice,_,_=select_dev(load_jsonl(f'data/eval/runs/{run}/rows.jsonl'),(20,))
        cfg.update(pool=choice['pool'],threshold=choice['threshold'],selected_dev_metrics=choice['metrics'])
    original_metrics=configs['pretrained']['selected_dev_metrics']
    def metrics(name):return original_metrics if name=='pretrained' else configs[name]['selected_dev_metrics']
    def grounding_score(m):return .5*(m['effective_recall@5'] or 0)+.5*(m['refusal']['recall'] or 0)
    candidates=['pretrained']
    for name in ('random','hard'):
        m=metrics(name)
        if grounding_score(m)>=grounding_score(original_metrics)-.005 and m['mrr@10']>=original_metrics['mrr@10']+.005:
            candidates.append(name)
    chosen=max(candidates,key=lambda n:(metrics(n)['mrr@10'],grounding_score(metrics(n))))
    for name,cfg in configs.items():
        weights=Path(cfg['model'])/'adapter.pt'
        if weights.is_file():cfg['adapter_sha256']=hashlib.sha256(weights.read_bytes()).hexdigest()
        cfg['risk_detector_sha256']=hashlib.sha256(Path('app/safety/risk_detector.py').read_bytes()).hexdigest()
        cfg['test_v2_sha256']=hashlib.sha256(Path('data/eval/v1/test_v2.jsonl').read_bytes()).hexdigest()
        cfg['dev_selected_candidate_name']=chosen
        cfg['selection_scope']='DEV only; small-data model is a learning-curve control, not an adoption candidate'
        target=Path(f'configs/experiments/m2-final-{name}.json')
        if target.exists():raise SystemExit('Do not overwrite frozen M2 configurations')
        target.write_text(json.dumps(cfg,ensure_ascii=False,indent=2),encoding='utf-8')
    decision={'dev_selected_name':chosen,'candidate_names':candidates,
              'selection_rule':'Compared to pretrained: balanced grounding/refusal score within .005 and dev MRR gain >= .005; choose highest dev MRR.',
              'test_acceptance_rule':'Adopt only the dev-selected model if test_v2 MRR gain >= .005, effective Recall@5 drop <= .02 and no-answer refusal recall non-inferior; otherwise keep M1 model.',
              'dev_metrics':{n:metrics(n) for n in configs}}
    Path('configs/experiments/m2-final-decision.json').write_text(json.dumps(decision,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({'dev_selected':chosen,'dev_mrr':{n:metrics(n)['mrr@10'] for n in configs}},indent=2))
