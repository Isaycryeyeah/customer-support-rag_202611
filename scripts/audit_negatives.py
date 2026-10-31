"""Diagnostic review of training negatives; never retroactively alter recorded training/tests."""
from concurrent.futures import ThreadPoolExecutor,as_completed
from collections import Counter
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from app.config import DEEPSEEK_API_KEY,DEEPSEEK_BASE_URL
from app.evaluation.eval_retrieval import load_jsonl
from scripts.build_synthetic_dataset import call_json

PROMPT='''你在审查模拟检索训练的负例是否错误。只依据问题、已标注正例及候选片段判断，不输出任何维修操作步骤。
输出json {"items":[{"query_id":"...","negatives":[{"strategy":"hard/random","chunk_id":"...","judgment":"negative_ok/positive_like/ambiguous","reason":"简短解释"}]}]}。
如果候选对问题中至少一项必要结论提供直接支持，应为positive_like，即使不能单独回答整个复合问题；仅主题相关不算正例；无法确定则ambiguous。每个候选恰好评一次。
这是同一AI的诊断性复核，不是人工真值。不能因为上游称它为负例就默认正确。'''


if __name__=='__main__':
    out=Path('data/eval/runs/negative-audit');out.mkdir(parents=True,exist_ok=True)
    cache=out/'api-cache';cache.mkdir(exist_ok=True)
    queries=load_jsonl('data/training/v1/queries.jsonl')
    kb={c['chunk_id']:c for c in load_jsonl('data/processed/chunks.jsonl')}
    from openai import OpenAI
    client=OpenAI(api_key=DEEPSEEK_API_KEY,base_url=DEEPSEEK_BASE_URL,max_retries=0)
    def work(i,batch):
        cases=[]
        for q in batch:
            cases.append({'query_id':q['query_id'],'question':q['question'],'positives':[kb[c] for c in q['positives']],
                          'negatives':[dict(kb[c],strategy=strategy) for strategy in ('hard','random') for c in q[strategy+'_negatives']]})
        return call_json(client,PROMPT,{'items':cases},cache/f'{i:03d}.json')['data']['items']
    judgments=[]
    with ThreadPoolExecutor(max_workers=4) as pool:
        futures=[pool.submit(work,i,queries[start:start+6]) for i,start in enumerate(range(0,len(queries),6))]
        for f in as_completed(futures):
            judgments.extend(f.result());print('negative audit queries:',len(judgments),flush=True)
    lookup={q['query_id']:q for q in queries}
    expected={(q['query_id'],s,c) for q in queries for s in ('hard','random') for c in q[s+'_negatives']}
    observed=[]
    for q in judgments:
        for n in q['negatives']:
            observed.append((q['query_id'],n['strategy'],n['chunk_id']))
            assert n['judgment'] in ('negative_ok','positive_like','ambiguous')
    assert set(observed)==expected and len(observed)==len(expected)
    counts={s:dict(Counter(n['judgment'] for q in judgments for n in q['negatives'] if n['strategy']==s)) for s in ('hard','random')}
    report={'scope':'post-experiment diagnostic AI review; no training pairs or test outcomes were rewritten',
            'query_count':len(queries),'negative_count':len(observed),'counts':counts,
            'human_verified':False,'limitations':['Candidate labels may be ambiguous; same-model review is not ground truth.',
                 'Suspect negatives motivate a future cleaned dataset and untouched test, not retrospective improvement claims.']}
    (out/'judgments.jsonl').write_text(''.join(json.dumps(j,ensure_ascii=False)+'\n' for j in judgments),encoding='utf-8')
    (out/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(report,indent=2),flush=True)
