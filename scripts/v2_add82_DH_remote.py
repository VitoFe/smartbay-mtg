"""Addendum 82, parts D and H, on the compute host: snapshot, a bounded job queue, status and fetch.

The snapshot holds exactly the files a traced quick run of src/experiments/v2_held_site_nested.py
opened or imported (plus the declaration), hash-checked on arrival. Jobs run under a small queue on
the host (`QUEUE` below, uploaded beside the snapshot) that keeps at most `--slots` jobs alive,
each with two threads (V2_THREADS, OMP, MKL and OpenBLAS limits, LightGBM n_jobs through V2_THREADS)
and no GPU, and that waits before a launch while the one-minute load is above 60. Credentials are
read by scientific_campaign.connect() and never printed.

Usage (SMARTBAY_SSH_HOST and SMARTBAY_SSH_PORT in the environment):
  python -m scripts.v2_add82_DH_remote pack --files <traced list.json>
  python -m scripts.v2_add82_DH_remote launch --jobs <jobs.json> --slots 8 --tag <name>
  python -m scripts.v2_add82_DH_remote status --tag <name>
  python -m scripts.v2_add82_DH_remote fetch --part D
"""
import argparse
import hashlib
import io
import json
import shlex
import sys
import tarfile
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
from scientific_campaign import connect, execute  # noqa: E402

DECLARED = 'config/bay_reserved_protocol_v2_addendum82.json'
DIRS = {'D': 'review_20260919/v2_add82_D', 'H': 'review_20260919/v2_add82_H'}
LOCAL = ROOT / 'reports/bay_twin/runs/v2_addendum82'

QUEUE = r'''
import json, os, subprocess, sys, time
jobs = json.load(open(sys.argv[1])); slots = int(sys.argv[2]); status_path = sys.argv[3]
py = sys.argv[4]
env = os.environ.copy()
env.update(OMP_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2', MKL_NUM_THREADS='2', V2_THREADS='2',
           CUDA_VISIBLE_DEVICES='', PYTHONDONTWRITEBYTECODE='1')
state = {j['name']: {'state': 'pending'} for j in jobs}
running = {}
def write():
    tmp = status_path + '.tmp'
    json.dump({'updated': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), 'slots': slots,
               'load': os.getloadavg(), 'jobs': state}, open(tmp, 'w'), indent=1)
    os.replace(tmp, status_path)
queue = list(jobs)
while queue or running:
    for name, (p, t0) in list(running.items()):
        code = p.poll()
        if code is not None:
            state[name].update(state='done' if code == 0 else 'failed', exit=code, seconds=round(time.time() - t0, 1))
            del running[name]
    while queue and len(running) < slots and os.getloadavg()[0] <= 60:
        j = queue.pop(0)
        os.makedirs(os.path.join(j['cwd'], 'logs'), exist_ok=True)
        log = open(os.path.join(j['cwd'], 'logs', j['name'] + '.log'), 'ab')
        p = subprocess.Popen([py, '-u', '-m'] + j['args'], cwd=j['cwd'], env=env, stdin=subprocess.DEVNULL,
                             stdout=log, stderr=subprocess.STDOUT)
        running[j['name']] = (p, time.time())
        state[j['name']].update(state='running', pid=p.pid, started=time.strftime('%H:%M:%S', time.gmtime()))
    write()
    time.sleep(5)
write()
'''


def session():
    for i in range(6):
        try:
            return connect()
        except Exception as e:
            print(f'[connect attempt {i + 1} failed: {type(e).__name__}]', flush=True)
            time.sleep(4 + 3 * i)
    raise SystemExit('unreachable')


def run(client, cmd):
    code, out = execute(client, cmd)
    if code:
        raise RuntimeError(out[-3000:])
    return out


def pack(files_json):
    files = json.loads(Path(files_json).read_text()) + [DECLARED]
    files = sorted(dict.fromkeys(files))
    manifest = {'created_utc': datetime.now(timezone.utc).isoformat(), 'declared_in': DECLARED,
                'files_sha256': {f: hashlib.sha256((ROOT / f).read_bytes()).hexdigest() for f in files}}
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode='w:gz') as tar:
        for f in files:
            tar.add(ROOT / f, arcname=f)
        content = json.dumps(manifest, indent=1).encode()
        info = tarfile.TarInfo('snapshot_manifest.json')
        info.size = len(content)
        tar.addfile(info, io.BytesIO(content))
        q = QUEUE.encode()
        info = tarfile.TarInfo('queue82.py')
        info.size = len(q)
        tar.addfile(info, io.BytesIO(q))
    local = LOCAL / 'remote'
    local.mkdir(parents=True, exist_ok=True)
    archive = local / 'snapshot.tar.gz'
    archive.write_bytes(buf.getvalue())
    (local / 'snapshot_manifest.json').write_bytes(json.dumps(manifest, indent=1).encode())
    c, base = session()
    try:
        probe = ("import tarfile,pathlib,hashlib,json,importlib.metadata,platform,sys\n"
                 "p=pathlib.Path.cwd()\n"
                 "t=tarfile.open('snapshot.tar.gz'); t.extractall(p,filter='data'); t.close()\n"
                 "m=json.loads((p/'snapshot_manifest.json').read_text())\n"
                 "bad=[n for n,w in m['files_sha256'].items() if hashlib.sha256((p/n).read_bytes()).hexdigest()!=w]\n"
                 "assert not bad, bad\n"
                 "rt={'host':platform.node(),'python':sys.version.split()[0],'packages':{n:importlib.metadata.version(n) for n in ['numpy','pandas','optuna','lightgbm','scikit-learn','scipy']}}\n"
                 "(p/'runtime.json').write_text(json.dumps(rt,indent=1)); print(json.dumps(rt))\n")
        for part, d in DIRS.items():
            remote = f'{base}/{d}'
            run(c, f'mkdir -p {shlex.quote(remote)}/logs')
            sftp = c.open_sftp()
            try:
                sftp.put(str(archive), remote + '/snapshot.tar.gz')
            finally:
                sftp.close()
            out = run(c, f'cd {shlex.quote(remote)} && {shlex.quote(base + "/.venv/bin/python")} -c {shlex.quote(probe)}')
            print(part, len(files), 'files verified;', out.strip().replace(base, '<base>'))
    finally:
        c.close()


def launch(jobs_json, slots, tag, after=None):
    jobs = json.loads(Path(jobs_json).read_text())
    c, base = session()
    try:
        for j in jobs:
            j['cwd'] = f"{base}/{DIRS[j.pop('part')]}"
        remote_jobs = f'{base}/{DIRS["D"]}/jobs_{tag}.json'
        sftp = c.open_sftp()
        try:
            with sftp.open(remote_jobs, 'w') as f:
                f.write(json.dumps(jobs, indent=1))
        finally:
            sftp.close()
        d = f'{base}/{DIRS["D"]}'
        py = base + '/.venv/bin/python'
        inner = (f'cd {shlex.quote(d)} && exec {shlex.quote(py)} -u queue82.py {shlex.quote(remote_jobs)} {slots} '
                 f'{shlex.quote(d + "/queue_" + tag + ".json")} {shlex.quote(py)} >> logs/queue_{tag}.log 2>&1')
        if after:
            # start only once the earlier queue's driver has exited, so the two never share the slots;
            # the bracket keeps pgrep from matching this command line itself
            inner = f"while pgrep -f '[q]ueue82.py .*jobs_{after}.json' > /dev/null; do sleep 30; done; " + inner
        run(c, f"setsid -f bash -c {shlex.quote(inner)} < /dev/null > /dev/null 2>&1")
        time.sleep(8)
        print(run(c, 'uptime').strip())
        print(run(c, "ps -eo nlwp,pcpu,args | grep '[v]2_held_site_nested\\|[v]2_held_station' | cut -c1-150 | head -20")
              .replace(base, '<base>'))
    finally:
        c.close()
    print('launched', len(jobs), 'jobs as', tag)


def status(tag):
    c, base = session()
    try:
        out = run(c, f'cat {base}/{DIRS["D"]}/queue_{tag}.json; uptime')
    finally:
        c.close()
    text, up = out.rsplit('}', 1)
    s = json.loads(text + '}')
    counts = {}
    for name, v in s['jobs'].items():
        counts[v['state']] = counts.get(v['state'], 0) + 1
    print(s['updated'], 'load', [round(x, 1) for x in s['load']], counts, up.strip())
    for name, v in s['jobs'].items():
        if v['state'] in ('failed', 'running'):
            print('  ', v['state'], name, v.get('exit', ''), v.get('started', ''))


def fetch(part, include_smoke=False):
    c, base = session()
    try:
        d = f'{base}/{DIRS[part]}'
        bundle = ("import pathlib,tarfile,hashlib,json\n"
                  "p=pathlib.Path.cwd()\n"
                  f"src=p/'reports/bay_twin/runs/v2_addendum82/{part}'\n"
                  "paths=[f for f in src.rglob('*') if f.is_file()" + ("" if include_smoke else " and 'smoke' not in f.relative_to(src).parts") + "]\n"
                  "paths+=[f for f in (p/'logs').glob('*.log')]+[p/'runtime.json',p/'snapshot_manifest.json']\n"
                  "paths+=[f for f in p.glob('queue_*.json')]+[f for f in p.glob('jobs_*.json')]\n"
                  "paths+=[f for f in (p/'reports/bay_twin/runs/v2_held_station').rglob('*') if f.is_file()] if (p/'reports/bay_twin/runs/v2_held_station').exists() else []\n"
                  "paths=[f for f in paths if not f.name.endswith(('-journal','.tmp'))]\n"
                  "a=tarfile.open('fetch.tar.gz','w:gz',compresslevel=1); n=0\n"
                  "for f in paths:\n"
                  "    try:\n"
                  "        a.add(f,arcname=f.relative_to(p).as_posix()); n+=1\n"
                  "    except FileNotFoundError:\n"
                  "        pass\n"
                  "a.close(); f=p/'fetch.tar.gz'; paths=range(n)\n"
                  "print(json.dumps({'sha256':hashlib.sha256(f.read_bytes()).hexdigest(),'files':len(paths)}))\n")
        expected = json.loads(run(c, f'cd {shlex.quote(d)} && {base}/.venv/bin/python -c {shlex.quote(bundle)}').strip()
                              .splitlines()[-1])
        local = LOCAL / 'remote' / part
        local.mkdir(parents=True, exist_ok=True)
        archive = local / 'fetch.tar.gz'
        sftp = c.open_sftp()
        try:
            sftp.get(d + '/fetch.tar.gz', str(archive))
        finally:
            sftp.close()
    finally:
        c.close()
    if hashlib.sha256(archive.read_bytes()).hexdigest() != expected['sha256']:
        raise ValueError('download checksum mismatch')
    with tarfile.open(archive) as a:
        members = a.getmembers()
        results = [m for m in members if m.name.startswith(f'reports/bay_twin/runs/v2_addendum82/{part}/')]
        a.extractall(ROOT, members=results, filter='data')
        a.extractall(local, members=[m for m in members if m not in results], filter='data')
    print('fetched', expected['files'], 'files, sha256 verified')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('action', choices=['pack', 'launch', 'status', 'fetch'])
    p.add_argument('--files')
    p.add_argument('--jobs')
    p.add_argument('--slots', type=int, default=8)
    p.add_argument('--tag')
    p.add_argument('--part', choices=['D', 'H'])
    p.add_argument('--smoke', action='store_true')
    p.add_argument('--after')
    a = p.parse_args()
    if a.action == 'pack':
        pack(a.files)
    elif a.action == 'launch':
        launch(a.jobs, a.slots, a.tag, a.after)
    elif a.action == 'status':
        status(a.tag)
    else:
        fetch(a.part, a.smoke)


if __name__ == '__main__':
    main()
