"""Replay the four human-reviewed cases as regressions, not as a new blind benchmark."""
import argparse
import json
from pathlib import Path
import requests


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--base-url',default='http://127.0.0.1:8003')
    p.add_argument('--output',required=True)
    args=p.parse_args()
    out=Path(args.output)
    if out.exists():raise SystemExit('Use a new regression run directory')
    out.mkdir(parents=True)
    original={r['query_id']:r for r in [json.loads(l) for l in Path('data/eval/runs/m1-online-selected/rows.jsonl').read_text(encoding='utf-8').splitlines()]}
    rows=[]
    for qid in ('syn_t0108','syn_t0350','ood_035','syn_t0255'):
        old=original[qid]['result']
        response=requests.post(args.base_url+'/ask',json={'question':original[qid]['sample']['question']},timeout=60)
        response.raise_for_status();r=response.json()
        if qid=='syn_t0108':assert r['need_human'] and r['ticket']['priority']=='高'
        if qid=='ood_035':assert not r['need_human'] and r['ticket'] is None and r['domain_scope']=='out_of_scope'
        if qid=='syn_t0350':assert r['answer_type']=='safety_refusal'
        if qid=='syn_t0255':
            assert r['need_human']
            assert all(t in r['answer'] for t in ('安全操作','切断电源','不要反复插拔','兼容性风险'))
        documents=[s.get('source_path') or s.get('doc_id') or s['title'] for s in r['sources']]
        assert len(documents)==len(set(documents))
        if r['answer_type']=='generated':assert r['citation_status']['status']=='valid'
        rows.append({'query_id':qid,'old_need_human':old['need_human'],'new_need_human':r['need_human'],
                     'context_count':len(r['contexts']),'unique_document_count':len(r['sources']),'result':r})
        print(qid, 'passed', 'passages:',len(r['contexts']),'documents:',len(r['sources']),flush=True)
    (out/'report.json').write_text(json.dumps({'scope':'post-human-feedback regression; not unseen testing or a new human score',
         'cases':rows,'manual_scores_inferred':False},ensure_ascii=False,indent=2),encoding='utf-8')
