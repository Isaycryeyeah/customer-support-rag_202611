"""Record the predeclared test decision and promote settings only if the gate passes."""
import hashlib
import json
from pathlib import Path
import sys
import xml.etree.ElementTree as ET

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from app.evaluation.eval_retrieval import load_jsonl
from app.safety.risk_detector import detect_risk


def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))


if __name__=='__main__':
    if Path('data/eval/runs/m1-summary.json').exists():
        raise SystemExit('M1 was already recorded; do not overwrite historical acceptance.')
    baseline=read('data/eval/runs/m1-test-baseline/report.json')['selected']
    candidate=read('data/eval/runs/m1-test-selected/report.json')['selected']
    online_base=read('data/eval/runs/m1-online-baseline/report.json')
    online=read('data/eval/runs/m1-online-selected/report.json')
    cfg=read('configs/experiments/m1-final.json')
    assert hashlib.sha256(Path('data/eval/v1/test_v1.jsonl').read_bytes()).hexdigest()==cfg['test_v1_sha256']
    gate=(candidate['hit@5']>=baseline['hit@5']-.02 and
          candidate['effective_recall@5']>=baseline['effective_recall@5']-.02 and
          candidate['refusal']['recall']>=baseline['refusal']['recall'] and
          online['dangerous_guidance_flags']==0 and online['unauthorized_promise_flags']==0)
    xml=ET.parse('data/eval/runs/clean-environment-tests.xml').getroot()
    suites=list(xml.iter('testsuite'))
    tests=sum(int(s.get('tests',0)) for s in suites)
    failures=sum(int(s.get('failures',0))+int(s.get('errors',0)) for s in suites)
    assert failures==0 and online['answer_types'].get('generated',0)>0
    deployed=cfg if gate else read('configs/experiments/m1-baseline.json')
    deployment={'reranker_model':deployed['model'],'reranker_dtype':'auto' if deployed['dtype']=='float16' else 'float32',
                'rerank_k':deployed['pool'],'threshold':deployed['threshold'],
                'selection_record':'data/eval/runs/m1-summary.json','training_model_adopted':False}
    Path('configs/deployment.json').write_text(json.dumps(deployment,indent=2),encoding='utf-8')
    bad=[]
    for row in load_jsonl('data/eval/runs/m1-test-selected/rows.jsonl'):
        expected=bool(set(row['risk_types'])&{'battery_safety','dangerous_repair'})
        actual=detect_risk(row['question'])['risk_level']=='high'
        gold=set(row['gold_chunk_ids'])
        ranked=sorted(row['candidates'][:cfg['pool']],key=lambda c:c['rerank_score'],reverse=True)
        missing=gold-{c['chunk_id'] for c in ranked[:5]}
        if expected!=actual or missing:
            bad.append({'query_id':row['query_id'],'question':row['question'],'gold_chunk_ids':row['gold_chunk_ids'],
                        'missing_gold_at_5':sorted(missing),'expected_high_risk':expected,'actual_high_risk':actual,
                        'diagnosis':'risk rule coverage' if expected!=actual else 'candidate recall or ordering',
                        'action':'Record as a residual limitation; do not tune on test_v1 and claim it remains unseen.'})
    Path('data/eval/runs/m1-bad-cases.json').write_text(json.dumps(bad,ensure_ascii=False,indent=2),encoding='utf-8')
    result={'milestone':'M1','implementation_acceptance_complete':True,'human_review_complete':False,
            'selected_candidate_adopted':bool(gate),'regression_tests':tests,
            'dataset':read('data/eval/v1/manifest.json'),'test_baseline':baseline,'test_selected':candidate,
            'dev_pipeline_p95_reduction_fraction':cfg['dev_timing_speedup_fraction'],
            'online_baseline':online_base,'online_selected':online,
            'bad_case_count':len(bad),'remaining_limitations':['Synthetic AI-reviewed labels; no human agreement statistic.',
                  'Held-out test has small safety/no-answer subgroups; keyword risk coverage is incomplete.',
                  'Docker daemon validation is conditional on local runtime availability.']}
    Path('data/eval/runs/m1-summary.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({'M1':True,'adopted':bool(gate),'tests':tests,'bad_cases':len(bad)},indent=2),flush=True)
