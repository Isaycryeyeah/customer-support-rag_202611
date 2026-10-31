"""Partial-layer PyTorch BCE training with dev MRR checkpointing and small adapters."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import random
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from app.evaluation.eval_retrieval import load_jsonl, retrieval_metrics, summarize_rows


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',required=True)
    args=parser.parse_args()
    cfg=json.loads(Path(args.config).read_text(encoding='utf-8'))
    out=Path(cfg['output'])
    if out.exists():raise SystemExit('Use a new model run directory')
    out.mkdir(parents=True)
    train=load_jsonl(cfg['training_queries']);dev=load_jsonl(cfg['dev_cache'])
    if not train or any(r.get('split')!='train' for r in train):raise SystemExit('Training pairs must declare TRAIN provenance')
    if not dev or any(r['split']!='dev' for r in dev):raise SystemExit('Checkpoint selection requires DEV only')
    dev_groups={r['group_id'] for r in dev}
    if dev_groups & {r['group_id'] for r in train}:raise SystemExit('Training/dev groups overlap')
    import numpy as np
    import torch
    from torch.utils.data import DataLoader
    from sentence_transformers import CrossEncoder
    random.seed(cfg['seed']);np.random.seed(cfg['seed']);torch.manual_seed(cfg['seed'])
    if torch.cuda.is_available():torch.cuda.manual_seed_all(cfg['seed'])
    torch.set_num_threads(4)
    device='cuda' if torch.cuda.is_available() else 'cpu'
    model=CrossEncoder(cfg['base_model'],revision=cfg.get('base_revision'),device=device)
    network=model.model;tokenizer=model.tokenizer
    count_layers=network.config.num_hidden_layers
    suffixes=tuple(f'encoder.layer.{i}.' for i in range(count_layers-cfg['train_last_layers'],count_layers))
    trainable=[]
    for name,param in network.named_parameters():
        param.requires_grad=any(s in name for s in suffixes) or name.startswith('classifier.')
        if param.requires_grad:trainable.append(name)
    if not trainable:raise SystemExit('No matching trainable layers')
    train=sorted(train,key=lambda r:r['group_id'])
    random.Random(cfg['seed']).shuffle(train)
    if cfg.get('limit_groups'):train=train[:cfg['limit_groups']]
    kb={c['chunk_id']:c for c in load_jsonl('data/processed/chunks.jsonl')}
    pairs=[]
    for row in train:
        for cid in row['positives']:pairs.append((row['question'],kb[cid]['content'],1.))
        for cid in row[cfg['negative_strategy']+'_negatives']:pairs.append((row['question'],kb[cid]['content'],0.))
    def collate(batch):
        inputs=tokenizer([r[0] for r in batch],[r[1] for r in batch],padding=True,truncation=True,
                         max_length=cfg['max_length'],return_tensors='pt')
        return {k:v.to(device) for k,v in inputs.items()},torch.tensor([r[2] for r in batch],device=device)
    loader=DataLoader(pairs,batch_size=cfg['batch_size'],shuffle=True,collate_fn=collate,num_workers=0)
    optimizer=torch.optim.AdamW([p for p in network.parameters() if p.requires_grad],lr=cfg['learning_rate'],weight_decay=.01)
    criterion=torch.nn.BCEWithLogitsLoss()
    def score_dev():
        network.eval();rows=[]
        with torch.no_grad():
            for row in dev:
                if not row['answerable']:continue
                subset=row['candidates'][:20]
                encoded=tokenizer([row['question']]*len(subset),[kb[c['chunk_id']]['content'] for c in subset],
                                  padding=True,truncation=True,max_length=cfg['max_length'],return_tensors='pt')
                scores=network(**{k:v.to(device) for k,v in encoded.items()}).logits.flatten().sigmoid().cpu().tolist()
                ranked=sorted(zip(subset,scores),key=lambda x:x[1],reverse=True)
                rows.append({'metrics':retrieval_metrics([c['chunk_id'] for c,s in ranked],row['gold_chunk_ids'])})
        return summarize_rows(rows)
    start=time.perf_counter();history=[];best=-1.;bad_epochs=0
    optimizer.zero_grad(set_to_none=True)
    for epoch in range(cfg['epochs']):
        network.train();loss_sum=0.
        for step,(inputs,labels) in enumerate(loader):
            logits=network(**inputs).logits.flatten()
            loss=criterion(logits,labels)
            (loss/cfg['gradient_accumulation']).backward()
            loss_sum+=loss.item()*len(labels)
            if (step+1)%cfg['gradient_accumulation']==0 or step+1==len(loader):
                torch.nn.utils.clip_grad_norm_([p for p in network.parameters() if p.requires_grad],1.)
                optimizer.step();optimizer.zero_grad(set_to_none=True)
        metric=score_dev()
        record={'epoch':epoch+1,'loss':loss_sum/len(pairs),'dev':metric,'elapsed_seconds':time.perf_counter()-start}
        history.append(record)
        print(json.dumps(record),flush=True)
        if metric['mrr@10']>best:
            best=metric['mrr@10'];bad_epochs=0
            torch.save({n:p.detach().cpu() for n,p in network.named_parameters() if p.requires_grad},out/'adapter.pt')
            best_epoch=epoch+1
        else:bad_epochs+=1
        if bad_epochs>=cfg['patience']:break
    adapter={'base_model':cfg['base_model'],'base_revision':cfg.get('base_revision'),
             'trainable_names':trainable,'max_length':cfg['max_length'],'best_epoch':best_epoch}
    (out/'adapter.json').write_text(json.dumps(adapter,indent=2),encoding='utf-8')
    report={'configuration':cfg,'device':device,'torch_version':torch.__version__,
            'training_groups':len(train),'training_pairs':len(pairs),
            'trainable_parameters':sum(p.numel() for p in network.parameters() if p.requires_grad),
            'total_parameters':sum(p.numel() for p in network.parameters()),
            'loss_function':'BCEWithLogitsLoss on query-document relevance',
            'optimizer':'AdamW','best_epoch':best_epoch,'best_dev_mrr@10':best,
            'history':history,'elapsed_seconds':time.perf_counter()-start,
            'peak_allocated_gpu_bytes':torch.cuda.max_memory_allocated() if device=='cuda' else None,
            'input_sha256':{name:hashlib.sha256(Path(cfg[name]).read_bytes()).hexdigest() for name in ('training_queries','dev_cache')},
            'adapter_sha256':hashlib.sha256((out/'adapter.pt').read_bytes()).hexdigest(),
            'limitations':['Single seed feasibility experiment on AI-generated/reviewed synthetic data.',
                          'Only final transformer layers and classification head trained; not full-model or LoRA fine-tuning.']}
    (out/'training_report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    (out/'config.json').write_text(json.dumps(cfg,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in report.items() if k in ('training_groups','training_pairs','elapsed_seconds','best_dev_mrr@10','trainable_parameters')},indent=2),flush=True)


if __name__=='__main__':main()
