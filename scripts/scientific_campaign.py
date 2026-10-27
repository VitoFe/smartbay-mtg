"""Isolated scientific campaign transport; credentials come from the environment or an
uncommitted .env file (src/settings.py), never from a tracked file."""
import argparse
import json
import shlex
import sys
import hashlib
import tarfile
import io
import re
from datetime import datetime,timezone
from pathlib import Path
import paramiko

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from src.settings import ssh_settings

def connect():
    cfg=ssh_settings()
    client=paramiko.SSHClient()
    client.load_system_host_keys()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(cfg['host'],port=cfg['port'],username=cfg['user'],password=cfg['password'],
                   key_filename=cfg['key_filename'],timeout=20,banner_timeout=25,auth_timeout=25)
    client.get_transport().set_keepalive(15)
    return client,cfg['remote_dir']

def execute(client,command):
    _,stdout,stderr=client.exec_command(command,timeout=60)
    out=stdout.read().decode('utf-8',errors='replace')
    err=stderr.read().decode('utf-8',errors='replace')
    code=stdout.channel.recv_exit_status()
    if err: out+='\n'+err
    return code,out

def preflight(client,base):
    probe=r'''
import os, sys, json, subprocess, importlib.metadata, pathlib, hashlib
root=pathlib.Path.cwd()
out={'python':sys.version,'root':str(root),'cpu_count':os.cpu_count(),'packages':{},'inputs':{}}
for p in ['numpy','pandas','scipy','scikit-learn','lightgbm','optuna','torch','paramiko','xarray']:
    try: out['packages'][p]=importlib.metadata.version(p)
    except importlib.metadata.PackageNotFoundError: out['packages'][p]=None
for p in ['data/processed/ctd_all_casts.csv.gz','data/raw/TBaiaP_01m-CTD(calibrato).txt','data/raw/TBaia_01m-Ossigeno(calibrato).txt','data/raw/TMitiliE_05m-Ossigeno(calibrato).txt']:
    f=root/p
    out['inputs'][p]={'exists':f.exists(),'bytes':f.stat().st_size if f.exists() else None,'sha256':hashlib.sha256(f.read_bytes()).hexdigest() if f.is_file() else None}
for name,args in [('gpu',['nvidia-smi','--query-gpu=name,memory.total,memory.used,utilization.gpu','--format=csv,noheader']),('memory',['free','-h']),('load',['uptime']),('scheduler',['sh','-c','command -v sbatch || true']),('active_python',['ps','-u',str(os.getuid()),'-o','pid,etime,comm'])]:
    try: out[name]=subprocess.run(args,text=True,capture_output=True,timeout=10).stdout
    except (OSError,subprocess.TimeoutExpired) as e: out[name]=type(e).__name__
print(json.dumps(out,indent=2))
'''
    code,out=execute(client,f'cd {shlex.quote(base)} && .venv/bin/python -c {shlex.quote(probe)}')
    dest=ROOT/'reports/revalidation/trinity_preflight.json'
    dest.parent.mkdir(parents=True,exist_ok=True)
    if code==0: dest.write_text(out,encoding='utf-8')
    print(out)
    return code

def launch(client,base,stage,quick=False):
    run_id=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'_'+stage+('_smoke' if quick else '')
    files=list((ROOT/'src').rglob('*.py'))+list((ROOT/'tests').glob('*.py'))
    files=[p for p in files if 'deploy' not in p.parts and '__pycache__' not in p.parts]
    files += [ROOT/'config/scientific_campaign.json',ROOT/'config/recovery_campaign.json',ROOT/'data/processed/ctd_all_casts.csv.gz']
    for p in (ROOT/'data/raw').glob('*.txt'):
        if p.name.startswith(('TBaia','TMitili','TPortovenere')) or p.name in ('T2-Meteo.txt','Meteo_Period_Processed_Data.txt'):
            files.append(p)
    manifest={'run_id':run_id,'prepared_utc':datetime.now(timezone.utc).isoformat(),'stage':stage,
              'files_sha256':{p.relative_to(ROOT).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in files},
              'scope':'Frozen isolated snapshot. Historical output reports and credentials are excluded.'}
    local=ROOT/'reports/revalidation'/run_id; local.mkdir(parents=True)
    (local/'snapshot_manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    bundle_dir=ROOT/'scratch/trinity_bundles'; bundle_dir.mkdir(exist_ok=True)
    archive=bundle_dir/(run_id+'.tar.gz')
    with tarfile.open(archive,'w:gz',compresslevel=4) as tar:
        for p in files: tar.add(p,arcname=p.relative_to(ROOT).as_posix())
        data=json.dumps(manifest,indent=2).encode()
        info=tarfile.TarInfo('snapshot_manifest.json'); info.size=len(data)
        tar.addfile(info,io.BytesIO(data))
    remote=base.rstrip('/')+'/revalidation/'+run_id
    command=f'mkdir -p {shlex.quote(remote)}'
    code,out=execute(client,command)
    if code: raise RuntimeError(out)
    sftp=client.open_sftp()
    try: sftp.put(str(archive),remote+'/snapshot.tar.gz')
    finally: sftp.close()
    extract="import tarfile,pathlib; p=pathlib.Path.cwd(); t=tarfile.open('snapshot.tar.gz'); t.extractall(p,filter='data'); (p/'data/external').mkdir(parents=True,exist_ok=True)"
    code,out=execute(client,f'cd {shlex.quote(remote)} && {shlex.quote(base+"/.venv/bin/python")} -c {shlex.quote(extract)}')
    if code: raise RuntimeError(out)
    record={'run_id':run_id,'remote_dir':remote,'stage':stage,'quick':quick}
    (local/'transport.json').write_text(json.dumps(record,indent=2),encoding='utf-8')
    launch_args=['nice','-n','10',base+'/.venv/bin/python','-u','-m','src.experiments.campaign_bootstrap','--stage',stage]
    if quick: launch_args.append('--quick')
    detach=("import os,subprocess; env=os.environ.copy(); "
            "env.update(OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',MKL_NUM_THREADS='4',CUDA_VISIBLE_DEVICES='1'); "
            f"log=open('run.log','ab'); p=subprocess.Popen({launch_args!r},stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,"
            "env=env,start_new_session=True,close_fds=True); print(p.pid)")
    code,out=execute(client,f'cd {shlex.quote(remote)} && {shlex.quote(base+"/.venv/bin/python")} -c {shlex.quote(detach)}')
    if code: raise RuntimeError(out)
    record['pid']=int(out.strip())
    (local/'transport.json').write_text(json.dumps(record,indent=2),encoding='utf-8')
    print(json.dumps(record,indent=2))
    return 0

def status(client,base,run_id,fetch=False):
    if not re.fullmatch(r'[0-9]{8}T[0-9]{6}Z_[a-z_]+',run_id): raise ValueError('Invalid campaign id.')
    remote=base.rstrip('/')+'/revalidation/'+run_id
    code,out=execute(client,f'cat {shlex.quote(remote+"/results/job_status.json")} 2>/dev/null; tail -n 12 {shlex.quote(remote+"/run.log")}')
    print(out)
    if fetch:
        target=ROOT/'reports/revalidation'/run_id/'results'; target.mkdir(parents=True,exist_ok=True)
        sftp=client.open_sftp()
        try:
            for item in sftp.listdir_attr(remote+'/results'):
                if '/' in item.filename or item.filename in ('.','..'): raise ValueError('Unexpected result filename')
                sftp.get(remote+'/results/'+item.filename,str(target/item.filename))
            sftp.get(remote+'/run.log',str(target.parent/'run.log'))
        finally: sftp.close()
        status_data=json.loads((target/'job_status.json').read_text())
        for name,expected in status_data.get('result_sha256',{}).items():
            if hashlib.sha256((target/name).read_bytes()).hexdigest()!=expected: raise ValueError(f'Result checksum mismatch: {name}')
        print('Fetched and verified:',target)
    return code

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('action',choices=['preflight','launch','status','fetch'])
    parser.add_argument('--stage',choices=['temporal','vertical','spatial','assimilation','sensitivity','missing','profiles'])
    parser.add_argument('--run-id')
    parser.add_argument('--quick',action='store_true')
    args=parser.parse_args()
    client,base=connect()
    try:
        if args.action=='preflight': return preflight(client,base)
        if args.action=='launch':
            if not args.stage: raise ValueError('--stage is required')
            return launch(client,base,args.stage,args.quick)
        if not args.run_id: raise ValueError('--run-id is required')
        return status(client,base,args.run_id,args.action=='fetch')
    finally: client.close()

if __name__=='__main__': sys.exit(main())
