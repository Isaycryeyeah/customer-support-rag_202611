"""Clean audited TRAIN negatives and construct a new support-disjoint held-out study."""
from concurrent.futures import ThreadPoolExecutor,as_completed
from difflib import SequenceMatcher
import hashlib
import itertools
import json
from pathlib import Path
import random
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from app.evaluation.eval_retrieval import load_jsonl
from scripts.prepare_eval import normalize
from scripts.build_synthetic_dataset import SYSTEM,call_json,make_tasks
from scripts.audit_annotations import PROMPT
from app.config import DEEPSEEK_API_KEY,DEEPSEEK_BASE_URL


if __name__=='__main__':
    out=Path('data/training/v2');out.mkdir(parents=True,exist_ok=True)
    audit={r['query_id']:r for r in load_jsonl('data/eval/runs/negative-audit/judgments.jsonl')}
    clean=[];dropped=[]
    for q in load_jsonl('data/training/v1/queries.jsonl'):
        judgments=audit[q['query_id']]['negatives']
        suspect={n['chunk_id'] for n in judgments if n['judgment']!='negative_ok'}
        good={s:[n['chunk_id'] for n in judgments if n['strategy']==s and n['judgment']=='negative_ok' and n['chunk_id'] not in suspect] for s in ('hard','random')}
        negatives=list(dict.fromkeys(good['hard']+good['random']))[:2]
        if not negatives:dropped.append(q['query_id']);continue
        r=dict(q,hard_negatives=negatives,random_negatives=good['random'],negative_review='strict AI negative_ok only; random fallback when hard candidates are suspect')
        clean.append(r)
    path=out/'queries.jsonl'
    path.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in clean),encoding='utf-8')
    (out/'manifest.json').write_text(json.dumps({'training_groups':len(clean),'pairs':sum(len(r['positives'])+len(r['hard_negatives']) for r in clean),
        'dropped_queries':dropped,'source':'v1 train queries and diagnostic negative review, no test mining',
        'queries_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'human_verified':False},indent=2),encoding='utf-8')
    cfg=json.loads(Path('configs/training/hard.json').read_text(encoding='utf-8'))
    cfg.update(training_queries=str(path).replace('\\','/'),output='models/reranker-hard-clean')
    Path('configs/training/hard-clean.json').write_text(json.dumps(cfg,indent=2),encoding='utf-8')
    holdout=Path('data/eval/clean-study-v1');holdout.mkdir(parents=True,exist_ok=True)
    cache=holdout/'api-cache';cache.mkdir(exist_ok=True)
    previous=[]
    for name in ('train','dev','test_v1','test_v2'):previous+=load_jsonl(f'data/eval/v1/{name}.jsonl')
    old_support={tuple(sorted(r['gold_chunk_ids'])) for r in previous if r['answerable']}
    original_tasks=json.loads(Path('data/eval/synthetic-source-v1/tasks.json').read_text(encoding='utf-8'))
    old_pairs={tuple(sorted(c['chunk_id'] for c in t['chunks'])) for t in original_tasks if len(t['chunks'])==2}
    old_pairs.update(pair for gold in old_support for pair in itertools.combinations(gold,2))
    kb_rows=load_jsonl('data/processed/chunks.jsonl');kb={c['chunk_id']:c for c in kb_rows}
    tasks=[t for t in make_tasks(kb_rows,10000,20261102) if len(t['chunks'])==2 and tuple(sorted(c['chunk_id'] for c in t['chunks'])) not in old_pairs][:60]
    for t in tasks:t['task_id']='new_'+hashlib.sha256('|'.join(sorted(c['chunk_id'] for c in t['chunks'])).encode()).hexdigest()[:10]
    (holdout/'tasks.json').write_text(json.dumps(tasks,ensure_ascii=False,indent=2),encoding='utf-8')
    kb_text=json.dumps([{k:c[k] for k in ('chunk_id','title','section','content')} for c in kb_rows],ensure_ascii=False)
    from openai import OpenAI
    client=OpenAI(api_key=DEEPSEEK_API_KEY,base_url=DEEPSEEK_BASE_URL,max_retries=0)
    def work(i,batch):
        binding=hashlib.sha256(json.dumps(batch,sort_keys=True).encode()).hexdigest()[:12]
        generated=call_json(client,SYSTEM,{'tasks':batch},cache/f'{binding}-generate.json')['data']['items']
        items=[]
        by_id={t['task_id']:t for t in batch}
        for g in generated:
            t=by_id[g['task_id']]
            items.append(dict(g,query_id=g['task_id'],gold_chunk_ids=[c['chunk_id'] for c in t['chunks']]))
        return call_json(client,PROMPT+kb_text,{'items':items},cache/f'{binding}-audit.json')['data']['items'],generated
    audited={};drafts={}
    with ThreadPoolExecutor(max_workers=4) as pool:
        futures=[pool.submit(work,i,tasks[start:start+6]) for i,start in enumerate(range(0,len(tasks),6))]
        for f in as_completed(futures):
            labels,generated=f.result()
            for a in labels:audited[a['query_id']]=a
            for g in generated:drafts[g['task_id']]=g
            print('new held-out tasks reviewed:',len(audited),flush=True)
    accepted=[];seen=set();rejected=[]
    task_map={t['task_id']:t for t in tasks}
    for qid in sorted(audited):
        a=audited[qid]
        q=drafts[qid]['question'];gold=tuple(sorted(a.get('gold_chunk_ids',[])));evidence=a.get('evidence',[])
        valid=a.get('accepted') is True and a.get('answerable') is True and bool(gold)
        valid &= gold not in old_support and gold not in seen
        valid &= {e.get('chunk_id') for e in evidence}==set(gold)
        valid &= all(e.get('quote') and e['quote'] in kb.get(e.get('chunk_id'),{}).get('content','') for e in evidence)
        valid &= all(SequenceMatcher(None,normalize(q),normalize(r['question'])).ratio()<.8 for r in previous+accepted)
        if not valid:rejected.append(qid);continue
        seen.add(gold)
        accepted.append({'query_id':qid,'group_id':'v3_'+hashlib.sha256('|'.join(gold).encode()).hexdigest()[:12],
            'question':q,'split':'test','test_partition':'test_v3','category':task_map[qid]['category'],
            'answerable':True,'gold_chunk_ids':list(gold),'expected_points':a['expected_points'],
            'risk_types':a['risk_types'],'need_human':bool(a['risk_types']),
            'evidence':evidence,'source':'new AI-generated/full-KB-reviewed evidence combinations; not human labels'})
    outside=[('ceramics','陶瓷釉色和烧制温度是什么关系？'),('birds','候鸟为什么能在迁徙时辨别方向？'),
             ('latin','中古拉丁文与古典拉丁文有哪些语法差异？'),('fabric','亚麻和棉布的纤维结构有什么区别？'),
             ('ocean','深海热液喷口附近的生态系统如何获取能量？'),('archaeology','考古学如何根据陶片判断遗址年代？'),
             ('geology','玄武岩和花岗岩的形成环境有什么区别？'),('optics','偏振镜如何消除水面的反光？'),
             ('ecology','湿地恢复通常如何评估生物多样性？'),('translation','请把这首西班牙语短诗翻译成中文。')]
    for topic,q in outside:
        accepted.append({'query_id':'v3_ood_'+topic,'group_id':'v3_ood_'+topic,'question':q,'split':'test',
            'test_partition':'test_v3','category':'unanswerable','answerable':False,'gold_chunk_ids':[],
            'expected_points':['超出售后服务范围'],'risk_types':[],'need_human':False,'evidence':[],
            'source':'assistant-authored outside-domain intent; handoff policy from human feedback'})
    accepted.sort(key=lambda r:r['query_id'])
    destination=holdout/'test_v3.jsonl'
    destination.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in accepted),encoding='utf-8')
    manifest={'query_count':len(accepted),'answerable_count':sum(r['answerable'] for r in accepted),
        'test_sha256':hashlib.sha256(destination.read_bytes()).hexdigest(),'rejected_candidates':rejected,
        'no_reused_support_sets':True,'no_reused_original_target_pairs':True,'max_allowed_wording_similarity_to_previous':.8,
        'frozen_before_clean_training':True,'human_labels_verified':False,
        'scope':'additional synthetic held-out data-quality study; known KB facts still shared'}
    (holdout/'manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    print(json.dumps(manifest,indent=2),flush=True)
