"""Evidence-bound synthetic benchmark. AI-generated/reviewed, never called human labels."""
import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from difflib import SequenceMatcher
import hashlib
import itertools
import json
from pathlib import Path
import random
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from app.config import DEEPSEEK_API_KEY, DEEPSEEK_BASE_URL, DEEPSEEK_MODEL
from scripts.prepare_eval import normalize, assign_splits

SYSTEM = '''你是模拟售后检索基准的标注员。品牌为虚构“云枢”。只依据提供的片段生成一条自然中文问题。
输出json {"items":[{"task_id":"...","question":"...","expected_points":["..."],"evidence":[{"chunk_id":"...","quote":"原文逐字摘录"}],"risk_types":[],"need_human":false,"device_type":"手机/平板/笔记本/通用","rationale":"说明问题为何需要每个指定片段"}]}。
每个task恰好一条。不要复制片段标题作为问题，不要只是要求总结材料。
单片段问题问一个具体真实诉求；双片段问题要自然地涉及两项相关诉求或比较两类产品，答案必须同时需要两个片段。
expected_points是简短正确的要点，每个要点有原文证据，不得添加资料没有的事实。每个指定片段必须有一条逐字quote。
风险标签由请求含义判断：battery_safety(鼓包/漏液/冒烟/异常过热等现实风险)，dangerous_repair(请求危险拆修操作)，refund_boundary(要求直接批准/保证/改敏感信息)，complaint(投诉)。没有现实风险的常规知识咨询不要贴风险标签。
高危、越权、投诉要求need_human=true；普通政策/流程咨询不需转人工。不要生成危险操作的执行步骤。
任务是生成新诉求而非给旧题换措辞；同一知识库是允许的。所有内容只用于模拟基准，不假称来自用户。'''

REVIEW = '''你是第二轮模拟基准证据审阅员。只依据给定原文逐项核验问题与标注。
输出json {"items":[{"task_id":"...","accepted":true,"reason":"...","question":"修正后的自然问题","expected_points":["..."],"evidence":[{"chunk_id":"...","quote":"逐字原文"}],"risk_types":[],"need_human":false,"device_type":"..."}]}。
每个任务必须逐项判断：1问题可由所有指定片段合起来回答且确实需要每个片段；2证据原文存在且支持要点；3无引入未知承诺；4问题不是标题抄写；5风险标签与转人工判断适当。
不满足时可修正问题和标注；不能修正则accepted=false。仅通过AI审阅，不是人工标签，不允许假称独立模型审阅。
风险只有battery_safety/dangerous_repair/refund_boundary/complaint。正常政策咨询不因提到保修退款而转人工，要求保证结果/直接审批才属于越权。'''

OOD = [
    ('cooking','空气炸锅怎么烤出酥脆的鸡翅？'), ('baking','戚风蛋糕回缩该怎么改善？'),
    ('gardening','阳台的薄荷总黄叶怎么养？'), ('language','德语四个格有什么区别？'),
    ('literature','如何解读红楼梦中的判词？'), ('music','初学钢琴每天如何练音阶？'),
    ('sport','羽毛球反手发力怎么训练？'), ('travel','京都三日游怎么安排路线？'),
    ('astronomy','为什么月球总是一面朝向地球？'), ('math','帮我证明勾股定理。'),
    ('physics','双缝干涉的条纹如何形成？'), ('chemistry','酸碱滴定终点怎么判断？'),
    ('biology','光合作用的暗反应在哪里发生？'), ('history','唐代科举与宋代有什么不同？'),
    ('geography','喀斯特地貌是怎样形成的？'), ('art','水彩画如何控制晕染？'),
    ('photography','人像摄影怎么布置三点光？'), ('movie','推荐几部法国新浪潮电影。'),
    ('writing','帮我写一首关于秋天的七言诗。'), ('career','产品经理面试如何准备？'),
    ('education','雅思口语的评分维度是什么？'), ('finance','某只股票下周会涨多少？'),
    ('law','租房押金纠纷应向哪里起诉？'), ('medicine','我的血常规指标异常是什么病？'),
    ('pet','猫咪应该多久修一次指甲？'), ('clothing','羊毛大衣怎么去除褶皱？'),
    ('home','卫生间地漏返味怎么处理？'), ('automobile','汽车年检需要携带哪些材料？'),
    ('rail','帮我查明天到上海的高铁票。'), ('flight','国际航班托运行李限额是多少？'),
    ('shipping','国际包裹海关税怎么算？'), ('insurance','人寿保险受益人能随时变更吗？'),
    ('real_estate','买房公摊面积怎么计算？'), ('tax','自由职业者如何申报个税？'),
    ('database','PostgreSQL索引膨胀如何排查？'), ('algorithm','快速排序平均复杂度如何推导？'),
    ('webdev','如何用CSS实现网格布局？'), ('linux','Linux的僵尸进程怎么回收？'),
    ('security','TLS握手的流程是什么？'), ('network','BGP路由选择依据是什么？'),
    ('os','操作系统如何实现虚拟内存？'), ('robotics','机械臂逆运动学怎么求解？'),
    ('agriculture','水稻分蘖期应该如何施肥？'), ('coffee','手冲咖啡研磨粗细怎么选？'),
    ('tea','乌龙茶和红茶的发酵区别是什么？'), ('chess','国际象棋王车易位有哪些条件？'),
    ('origami','如何折一只纸鹤？'), ('astronautics','火星探测器怎样实现软着陆？'),
    ('unknown_model','云枢下一代手机的处理器具体型号是什么？'),
    ('store_address','离我家最近的云枢维修店具体门牌号是什么？'),
    ('inventory','现在仓库里有多少块云枢手机原装电池？'),
    ('order_status','我的订单AB123退款已经到账了吗？'),
    ('serial_lookup','请查询序列号SN999此刻的真实保修截止日。'),
    ('firmware_version','云枢Phone今年最新固件的完整版本号是多少？'),
    ('accessibility','云枢平板支持哪一种盲文显示器型号？'),
    ('enterprise_contract','我们企业采购合同里的专属维修折扣是多少？'),
    ('international_service','云枢在冰岛的维修服务电话是多少？'),
    ('recycling','云枢旧手机回收时每克金属的结算价是多少？'),
    ('nonbrand','苹果MacBook电池的官方保修期是多少？'),
    ('availability','售后安全组现在具体有几位工程师在线？'),
]


def family(chunk):
    doc = chunk['doc_id']
    if any(t in doc for t in ('warranty', 'invoice')):
        return 'warranty'
    if any(t in doc for t in ('battery', 'charger', 'charging', 'screen', 'water', 'dangerous')):
        return 'safety'
    if any(t in doc for t in ('return', 'exchange', 'complaint', 'boundary', 'transfer')):
        return 'service'
    return 'repair'


def call_json(client, system, data, path):
    if path.exists():
        return json.loads(path.read_text(encoding='utf-8'))
    for attempt in range(3):
        try:
            response = client.chat.completions.create(
                model=DEEPSEEK_MODEL, messages=[{'role':'system','content':system},
                                               {'role':'user','content':json.dumps(data, ensure_ascii=False)}],
                response_format={'type':'json_object'}, temperature=.3, max_tokens=10000,
                extra_body={'thinking': {'type':'disabled'}}, timeout=90)
            choice = response.choices[0]
            if choice.finish_reason != 'stop':
                raise ValueError('truncated JSON')
            result = {'data':json.loads(choice.message.content), 'usage':response.usage.model_dump(),
                      'model':response.model, 'response_id':response.id}
            path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
            return result
        except Exception as error:
            if attempt == 2:
                raise RuntimeError(type(error).__name__) from None
            time.sleep(attempt+1)


def make_tasks(chunks, pairs, seed):
    rng = random.Random(seed)
    families = defaultdict(list)
    for c in chunks:
        families[family(c)].append(c)
    combinations = []
    for bucket in families.values():
        for a,b in itertools.combinations(bucket,2):
            if a['doc_id'] != b['doc_id']:
                combinations.append([a,b])
    rng.shuffle(combinations)
    supports = [[c] for c in chunks] + combinations[:pairs]
    return [{'task_id':f't{i:04d}', 'category':family(group[0]),
             'chunks':[{k:c[k] for k in ('chunk_id','title','section','content')} for c in group]}
            for i,group in enumerate(supports)]


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',default='data/eval/synthetic-source-v1')
    parser.add_argument('--pairs',type=int,default=300)
    parser.add_argument('--workers',type=int,default=4)
    args=parser.parse_args()
    output=ROOT/args.output
    output.mkdir(parents=True,exist_ok=True)
    cache=output/'api-cache'
    cache.mkdir(exist_ok=True)
    from openai import OpenAI
    client=OpenAI(api_key=DEEPSEEK_API_KEY,base_url=DEEPSEEK_BASE_URL,max_retries=0,timeout=90)
    chunks=[json.loads(l) for l in (ROOT/'data/processed/chunks.jsonl').read_text(encoding='utf-8').splitlines()]
    tasks=make_tasks(chunks,args.pairs,20261031)
    (output/'tasks.json').write_text(json.dumps(tasks,ensure_ascii=False,indent=2),encoding='utf-8')
    def batch_work(index,batch):
        draft=call_json(client,SYSTEM,{'tasks':batch},cache/f'{index:03d}-generate.json')
        reviewed=call_json(client,REVIEW,{'tasks':batch,'draft':draft['data']},cache/f'{index:03d}-review.json')
        return index,reviewed['data']['items']
    reviewed_items={}
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures=[pool.submit(batch_work,i,tasks[start:start+6]) for i,start in enumerate(range(0,len(tasks),6))]
        for future in as_completed(futures):
            index,items=future.result()
            for item in items:
                reviewed_items[item['task_id']]=item
            print(f'AI-reviewed batch {index+1}; items received {len(reviewed_items)}',flush=True)
    accepted,rejected=[],[]
    allowed={'battery_safety','dangerous_repair','refund_boundary','complaint'}
    for task in tasks:
        item=reviewed_items.get(task['task_id'],{})
        evidence=item.get('evidence',[])
        valid=item.get('accepted') is True and bool(item.get('question')) and bool(item.get('expected_points'))
        required={c['chunk_id']:c['content'] for c in task['chunks']}
        valid &= {e.get('chunk_id') for e in evidence} == set(required)
        valid &= all(e.get('quote') and e['quote'] in required.get(e.get('chunk_id'),'') for e in evidence)
        valid &= set(item.get('risk_types',[])) <= allowed
        if not valid:
            rejected.append({'task_id':task['task_id'],'reason':'review rejected or evidence failed exact-match validation'})
            continue
        gold=sorted(required)
        accepted.append({'query_id':'syn_'+task['task_id'],
            'group_id':'evidence_'+hashlib.sha256('|'.join(gold).encode()).hexdigest()[:12],
            'question':item['question'],'category':task['category'],'answerable':True,
            'gold_chunk_ids':gold,'expected_points':item['expected_points'],
            'risk_types':item.get('risk_types',[]),'need_human':bool(item.get('need_human')),
            'device_type':item.get('device_type','通用'),'evidence':evidence,
            'source':'deepseek_generated_and_second_pass_ai_reviewed',
            'annotation_review':'same-model second pass plus exact source quote validation; not human review'})
    for index,(topic,question) in enumerate(OOD):
        accepted.append({'query_id':f'ood_{index:03d}','group_id':'ood_'+topic,'question':question,
            'category':'unanswerable','answerable':False,'gold_chunk_ids':[],
            'expected_points':['当前资料无法确认'],'risk_types':[],'need_human':True,
            'device_type':'通用','source':'assistant_authored_unsupported_intent',
            'annotation_review':'assistant checked knowledge-base scope; not human review'})
    # Merge near duplicate questions before assignment. Also keep historical-like groups in train.
    parent={r['group_id']:r['group_id'] for r in accepted}
    def find(g):
        while parent[g]!=g:
            parent[g]=parent[parent[g]]; g=parent[g]
        return g
    near=[]
    for i,a in enumerate(accepted):
        for b in accepted[i+1:]:
            similarity=SequenceMatcher(None,normalize(a['question']),normalize(b['question'])).ratio()
            if similarity>=.85:
                ga,gb=find(a['group_id']),find(b['group_id'])
                parent[gb]=ga
                near.append([a['query_id'],b['query_id'],round(similarity,3)])
    for row in accepted:
        row['group_id']=find(row['group_id'])
    # Remove exact duplicate wording rather than counting it twice.
    seen=set(); unique=[]
    for row in accepted:
        q=normalize(row['question'])
        if q not in seen:
            seen.add(q); unique.append(row)
    # A merged near-duplicate group can span broad categories; stratify by its first category.
    categories={}
    for row in unique:
        categories.setdefault(row['group_id'],row['category'])
        row['category']=categories[row['group_id']]
    assign_splits(unique)
    historical=[]
    for name in ['retrieval_eval.jsonl','generation_eval.jsonl','safety_eval.jsonl']:
        historical.extend(json.loads(l)['question'] for l in (ROOT/'data/eval'/name).read_text(encoding='utf-8').splitlines())
    legacy_groups={r['group_id'] for r in unique if any(SequenceMatcher(None,normalize(r['question']),normalize(q)).ratio()>=.8 for q in historical)}
    for row in unique:
        if row['group_id'] in legacy_groups:
            row['split']='train'
    # Reserve a separate test_v2 now, before any tuning or training.
    test_groups=sorted({r['group_id'] for r in unique if r['split']=='test'})
    random.Random(20261101).shuffle(test_groups)
    v1=set(test_groups[:len(test_groups)//2])
    for row in unique:
        if row['split']=='test':
            row['test_partition']='test_v1' if row['group_id'] in v1 else 'test_v2'
    (output/'annotations.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in unique),encoding='utf-8')
    review={'accepted_queries':len(unique),'independent_evidence_groups':len({r['group_id'] for r in unique}),
            'split_counts':dict(Counter(r['split'] for r in unique)), 'historical_like_groups_train_only':len(legacy_groups),
            'near_duplicate_merges':near,'rejected':rejected,
            'limitations':['Synthetic known-KB evidence-combination benchmark; not real traffic.',
                          'AI review and exact quotations do not establish human-verified semantic labels.',
                          'Evidence combinations share facts across splits; this is not unseen-fact generalization.']}
    (output/'review.json').write_text(json.dumps(review,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in review.items() if k not in ('near_duplicate_merges','rejected')},ensure_ascii=True),flush=True)


if __name__=='__main__':
    main()
