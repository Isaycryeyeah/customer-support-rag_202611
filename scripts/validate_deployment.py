"""Real HTTP smoke test and credential-free Docker provenance capture."""
import argparse
import json
from pathlib import Path
import subprocess
import time
import requests


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--base-url',default='http://127.0.0.1:8003')
    p.add_argument('--frontend-url',default='http://127.0.0.1:8503')
    p.add_argument('--output',required=True)
    p.add_argument('--capture-docker',action='store_true')
    args=p.parse_args()
    out=Path(args.output)
    if out.exists():raise SystemExit('Choose a new validation output directory')
    out.mkdir(parents=True)
    ready=requests.get(args.base_url+'/ready',timeout=10);ready.raise_for_status()
    ui=requests.get(args.frontend_url+'/_stcore/health',timeout=10);ui.raise_for_status()
    checks=[]
    for question,kind in [('手机保修期是多久？','generated'),('今天天气怎么样？','low_confidence'),
                          ('电池鼓包了还能充电吗？','safety_refusal'),
                          ('担心上次维修人员偷看了我的照片，能查有没有数据泄露吗？','generated')]:
        start=time.perf_counter()
        response=requests.post(args.base_url+'/ask',json={'question':question},timeout=60)
        response.raise_for_status();r=response.json()
        assert r['answer_type']==kind
        if r['ticket']:assert r['ticket']['need_human']==r['need_human']
        if '偷看' in question:assert r['need_human'] and r['ticket']['priority']=='高'
        if '天气' in question:assert not r['need_human'] and r['ticket'] is None
        checks.append({'question':question,'http_status':response.status_code,'elapsed_seconds':time.perf_counter()-start,'result':r})
        print(kind,'verified',flush=True)
    feedback=requests.post(args.base_url+'/feedback',json={'log_id':checks[0]['result']['log_id'],'rating':1},timeout=10)
    feedback.raise_for_status()
    report={'ready':ready.json(),'frontend_health':ui.text,'checks':checks,'feedback':feedback.json()}
    if args.capture_docker:
        def cmd(arguments):return subprocess.check_output(arguments,text=True,encoding='utf-8').strip()
        container='rag-validation-backend-1'
        report['docker_image_id']=cmd(['docker','image','inspect','customer-support-rag:validation','--format','{{.Id}}'])
        report['dependency_check']=cmd(['docker','exec',container,'python','-m','pip','check'])
        report['env_file_in_image']=cmd(['docker','exec',container,'python','-c',"from pathlib import Path; print(Path('/app/.env').exists())"])
        assert report['env_file_in_image']=='False'
        report['runtime']=json.loads(cmd(['docker','exec',container,'python','-c',
            "import json,sys,torch,transformers; print(json.dumps({'python':sys.version,'torch':torch.__version__,'cuda':torch.cuda.is_available(),'transformers':transformers.__version__}))"]))
        lock=cmd(['docker','exec',container,'python','-m','pip','freeze','--all'])+'\n'
        (out/'requirements.linux.observed.txt').write_text(lock,encoding='utf-8')
    (out/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print('deployment verified:',out,flush=True)


if __name__=='__main__':main()
