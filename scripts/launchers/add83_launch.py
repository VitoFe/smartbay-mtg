"""Addendum 83 on the compute host: hash-verified snapshot, an adaptive job queue, status, fetch.

usage: python scripts/launchers/add83_launch.py pack [--smoke]        -> snapshot and job scripts, locally
       python scripts/launchers/add83_launch.py upload <run_id>       -> uploads, extracts, verifies, runtime.json
       python scripts/launchers/add83_launch.py start  <run_id> [--max N] [--load L]
       python scripts/launchers/add83_launch.py width  <run_id> N     -> changes the queue's width while it runs
       python scripts/launchers/add83_launch.py status <run_id>
       python scripts/launchers/add83_launch.py fetch  <run_id>
The shared studies of config/bay_reserved_protocol_v2_addendum83.json run here (CPU only, two threads per job);
the per-fold studies run locally. Connection settings are read by scientific_campaign.connect() and never
printed. Packs are kept under reports/bay_twin/runs/v2_addendum83/launch/<run_id>/ (the tarball is git-ignored).
"""
import hashlib
import io
import json
import shlex
import sys
import tarfile
import time
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / 'scripts'))
sys.path.insert(0, str(REPO))
from scientific_campaign import connect, execute  # noqa: E402

PY = '<base>/.venv/bin/python'
DECLARED = 'config/bay_reserved_protocol_v2_addendum83.json'
RUNS = REPO / 'reports/bay_twin/runs'
TARGET = RUNS / 'v2_addendum83'
PACKS = TARGET / 'launch'
#: the input list part B of addendum 82 traced for the same tree designs (its data, geometry and audits)
B_MANIFEST = RUNS / 'v2_addendum82/B/run_20261012T081855Z_add82B/snapshot_manifest.json'
MODULE = 'src.experiments.v2_lead_family'
LEADS = (1, 3, 6, 12, 24, 36, 48, 72)
CELLS = [(k, None) for k in range(6)] + [(0, 7), (0, 123)]
ARMS = ('river_tree', 'weather_river_tree', 'sealevel_tree', 'extboundary_tree', 'bathy_tree')
T, O = 'temperature', 'oxygen'
SWEPT_BEFORE = ({(a, v, h) for a in ('tree', 'weather_tree') for v, h in [(T, 6), (O, 6), (T, 24), (O, 24), (O, 48)]}
                | {('river_tree', v, h) for v, h in [(T, 6), (O, 6), (O, 24)]}
                | {('weather_river_tree', O, 6), ('weather_river_tree', O, 24)}
                | {('sealevel_tree', v, h) for v, h in [(O, 6), (T, 6), (O, 48)]}
                | {('extboundary_tree', O, 6), ('extboundary_tree', T, 6)}
                | {('bathy_tree', O, 24), ('bathy_tree', T, 24)})
#: arms whose offset-0 seed-42 study reproduces a published reference-grid value (gate L-G1)
REFERENCE = ({('base', v, h) for v, h in [(T, 1), (T, 3), (T, 12), (T, 36), (T, 48), (O, 1), (O, 3), (O, 12), (O, 36)]}
             | {('river_tree', T, 24), ('river_tree', T, 48), ('river_tree', O, 48),
                ('weather_river_tree', T, 6), ('weather_river_tree', T, 24), ('weather_river_tree', T, 48),
                ('weather_river_tree', O, 48), ('sealevel_tree', T, 24), ('sealevel_tree', T, 48),
                ('sealevel_tree', O, 24), ('extboundary_tree', T, 24), ('extboundary_tree', T, 48),
                ('extboundary_tree', O, 24), ('extboundary_tree', O, 48), ('bathy_tree', T, 6), ('bathy_tree', O, 6)})


def remote_dir(run):
    return f'<base>/review_20260919/v2_add83/{run}'


def session(tries=6):
    for i in range(tries):
        try:
            return connect()
        except Exception as e:
            print(f'[connect attempt {i + 1} failed: {type(e).__name__}]', flush=True)
            time.sleep(4 + 3 * i)
    sys.exit('unreachable')


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def files():
    listed = json.loads(B_MANIFEST.read_text(encoding='utf-8'))['files_sha256']
    out = [REPO / p for p in listed if not p.startswith('src/') and 'addendum82' not in p]
    out += [p for p in sorted((REPO / 'src').rglob('*.py')) if '__pycache__' not in p.parts]
    out += [REPO / DECLARED, REPO / 'config/tuning_protocol_v2.json', REPO / 'scripts/launchers/add83_queue.py']
    out = list(dict.fromkeys(out))
    missing = [p for p in out if not p.is_file()]
    if missing:
        # a file of part B's traced list that is gone from the checkout and that no module names is left out
        code = '\n'.join(p.read_text(encoding='utf-8', errors='replace') for p in (REPO / 'src').rglob('*.py'))
        named = [p for p in missing if p.name in code]
        if named:
            raise SystemExit(f'missing: {named[:5]}')
        print('left out (gone from the checkout, named by no module):', [p.relative_to(REPO).as_posix() for p in missing])
        out = [p for p in out if p.is_file()]
    return out


def sfx(s):
    return '' if s is None else f'_seed{s}'


def sarg(s):
    return [] if s is None else ['--seed', str(s)]


def jobs(smoke):
    if smoke:
        q = ['--quick']
        return [('smoke_base_temperature_h1_offset1', [MODULE, 'base', '--variable', T, '--horizon', '1', '--offset', '1'] + q),
                ('smoke_river_tree_temperature_h1_offset2', [MODULE, 'arm', '--arm', 'river_tree', '--variable', T,
                                                             '--horizon', '1', '--offset', '2'] + q),
                ('smoke_base_oxygen_h72_offset1', [MODULE, 'base', '--variable', O, '--horizon', '72', '--offset', '1'] + q),
                ('smoke_sealevel_tree_oxygen_h72_offset2', [MODULE, 'arm', '--arm', 'sealevel_tree', '--variable', O,
                                                            '--horizon', '72', '--offset', '2'] + q)]
    first, rest = [], []
    for v in (T, O):
        for h in LEADS:
            if ('tree', v, h) not in SWEPT_BEFORE:
                for o, s in CELLS:
                    job = (f'base_{v}_h{h}_offset{o}{sfx(s)}',
                           [MODULE, 'base', '--variable', v, '--horizon', str(h), '--offset', str(o)] + sarg(s))
                    (first if (o, s) == (0, None) and ('base', v, h) in REFERENCE else rest).append(job)
            for arm in ARMS:
                if (arm, v, h) in SWEPT_BEFORE:
                    continue
                for o, s in CELLS:
                    job = (f'{arm}_{v}_h{h}_offset{o}{sfx(s)}',
                           [MODULE, 'arm', '--arm', arm, '--variable', v, '--horizon', str(h), '--offset', str(o)] + sarg(s))
                    (first if (o, s) == (0, None) and (arm, v, h) in REFERENCE else rest).append(job)
    return first + rest


def job_script(name, args):
    cmd = ' '.join(shlex.quote(x) for x in args)
    return ('#!/bin/bash\n'
            'cd "$(dirname "$0")/.."\n'
            'unset V2_ISPRA_BINNING V2_ADD82_OUT V2_ADD83_OUT\n'
            "export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 V2_THREADS=2 "
            "CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1\n"
            f'{PY} -u -m {cmd} > logs/{name}.log 2>&1\n'
            f'echo $? > status/{name}.exit\n')


def pack(smoke):
    fs = files()
    js = jobs(smoke)
    run = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ') + '_add83' + ('_smoke' if smoke else '')
    local = PACKS / run
    local.mkdir(parents=True)
    manifest = {'run_id': run, 'declared_in': DECLARED, 'declaration_sha256': sha(REPO / DECLARED),
                'smoke_only': smoke, 'created_utc': datetime.now(timezone.utc).isoformat(),
                'launcher': 'scripts/launchers/add83_launch.py', 'launcher_sha256': sha(Path(__file__)),
                'queue_runner_sha256': sha(REPO / 'scripts/launchers/add83_queue.py'),
                'files_sha256': {p.relative_to(REPO).as_posix(): sha(p) for p in fs},
                'jobs': [{'name': n, 'args': a} for n, a in js]}
    (local / 'snapshot_manifest.json').write_bytes(json.dumps(manifest, indent=2).encode('utf-8'))
    with tarfile.open(local / 'snapshot.tar.gz', 'w:gz') as tar:
        for p in fs:
            tar.add(p, arcname=p.relative_to(REPO).as_posix())
        extra = {'snapshot_manifest.json': json.dumps(manifest).encode(),
                 'queue.txt': ('\n'.join(n for n, _ in js) + '\n').encode(),
                 'add83_queue.py': (REPO / 'scripts/launchers/add83_queue.py').read_bytes()}
        for n, a in js:
            extra[f'jobs/{n}.sh'] = job_script(n, a).encode()
        for name, content in extra.items():
            info = tarfile.TarInfo(name)
            info.size = len(content)
            info.mode = 0o755 if name.endswith('.sh') else 0o644
            tar.addfile(info, io.BytesIO(content))
    print(run, len(fs), 'files', len(js), 'jobs', round((local / 'snapshot.tar.gz').stat().st_size / 1e6, 1), 'MB')


def sub(base, s):
    return s.replace('<base>', base)


def upload(run):
    local = PACKS / run
    c, base = session()
    try:
        remote = sub(base, remote_dir(run))
        code, out = execute(c, f'mkdir -p {shlex.quote(remote)}/logs {shlex.quote(remote)}/status')
        assert code == 0, out
        sftp = c.open_sftp()
        try:
            sftp.put(str(local / 'snapshot.tar.gz'), remote + '/snapshot.tar.gz')
        finally:
            sftp.close()
        probe = ("import tarfile,pathlib,hashlib,json,importlib.metadata,platform,socket,os\n"
                 "p=pathlib.Path.cwd()\n"
                 "t=tarfile.open('snapshot.tar.gz'); t.extractall(p,filter='data'); t.close()\n"
                 "m=json.loads((p/'snapshot_manifest.json').read_text())\n"
                 "bad=[n for n,w in m['files_sha256'].items() if hashlib.sha256((p/n).read_bytes()).hexdigest()!=w]\n"
                 "assert not bad, bad[:5]\n"
                 "r={'host':socket.gethostname(),'python':platform.python_version(),'cores':os.cpu_count(),"
                 "'packages':{n:importlib.metadata.version(n) for n in ['numpy','pandas','optuna','lightgbm','scikit-learn']}}\n"
                 "(p/'runtime.json').write_text(json.dumps(r,indent=2)); print(json.dumps(r)); print('verified',len(m['files_sha256']))\n")
        code, out = execute(c, f'cd {shlex.quote(remote)} && {sub(base, PY)} -c {shlex.quote(probe)}')
        print(out.replace(base, '<base>'))
        assert code == 0
    finally:
        c.close()


def start(run, n, load):
    c, base = session()
    try:
        remote = sub(base, remote_dir(run))
        code, out = execute(c, f"cd {shlex.quote(remote)} && sed -i 's#<base>#{base}#g' jobs/*.sh && "
                               f"setsid -f bash -c 'cd {shlex.quote(remote)} && {sub(base, PY)} add83_queue.py --max {int(n)} "
                               f"--load {float(load)} --gap 3 > queue.log 2>&1' < /dev/null > /dev/null 2>&1; sleep 8; "
                               f"ps -eo pid,nlwp,pcpu,args | grep '[v]2_lead_family' | wc -l; uptime")
        print(out.replace(base, '<base>'))
    finally:
        c.close()


def width(run, n):
    c, base = session()
    try:
        remote = sub(base, remote_dir(run))
        code, out = execute(c, f'cd {shlex.quote(remote)} && echo {int(n)} > width.txt && cat width.txt')
        print(out)
    finally:
        c.close()


def status(run):
    c, base = session()
    try:
        remote = sub(base, remote_dir(run))
        code, out = execute(c, f"cd {shlex.quote(remote)} && echo jobs $(wc -l < queue.txt) done $(ls status | wc -l) "
                               f"failed $(grep -L '^0$' status/*.exit 2>/dev/null | wc -l) "
                               f"running $(ps -eo args | grep -c '[v]2_lead_family'); "
                               f"cat queue.done 2>/dev/null; grep -L '^0$' status/*.exit 2>/dev/null | head -5; "
                               f"tail -n 2 load.log 2>/dev/null; uptime")
        print(out.replace(base, '<base>'))
    finally:
        c.close()


def fetch(run):
    c, base = session()
    try:
        remote = sub(base, remote_dir(run))
        bundle = ("import pathlib,tarfile,hashlib,json\n"
                  "p=pathlib.Path.cwd()\n"
                  "o=p/'reports/bay_twin/runs/v2_addendum83'\n"
                  "paths=[f for f in o.rglob('*') if f.is_file()] if o.exists() else []\n"
                  "paths+=[p/'runtime.json',p/'snapshot_manifest.json',p/'queue.txt',p/'load.log']"
                  "+[f for f in p.glob('queue.*') if f.is_file()]"
                  "+list((p/'logs').glob('*.log'))+list((p/'status').glob('*.exit'))\n"
                  "with tarfile.open('results_download.tar.gz','w:gz',compresslevel=1) as a:\n"
                  " for f in paths: a.add(f,arcname=f.relative_to(p).as_posix())\n"
                  "f=p/'results_download.tar.gz'; print(json.dumps({'sha256':hashlib.sha256(f.read_bytes()).hexdigest(),'files':len(paths)}))\n")
        code, out = execute(c, f'cd {shlex.quote(remote)} && {sub(base, PY)} -c {shlex.quote(bundle)}')
        assert code == 0, out
        expected = json.loads(out.strip().splitlines()[-1])
        local = PACKS / run
        local.mkdir(parents=True, exist_ok=True)
        dl = local / 'results_download.tar.gz'
        sftp = c.open_sftp()
        try:
            sftp.get(remote + '/results_download.tar.gz', str(dl))
        finally:
            sftp.close()
        assert sha(dl) == expected['sha256'], 'checksum'
        record = TARGET / f'run_{run}'
        with tarfile.open(dl) as a:
            for m in a.getmembers():
                name = m.name
                dest = REPO / name if name.startswith('reports/bay_twin/runs/v2_addendum83/') else record / name
                data = a.extractfile(m).read()
                if dest.exists():
                    if hashlib.sha256(dest.read_bytes()).hexdigest() != hashlib.sha256(data).hexdigest():
                        raise SystemExit(f'refusing to overwrite a different file: {dest}')
                    continue
                dest.parent.mkdir(parents=True, exist_ok=True)
                dest.write_bytes(data)
        (record / 'download_manifest.json').write_bytes(json.dumps(expected, indent=2).encode('utf-8'))
        print('fetched and verified', expected['files'], 'files')
    finally:
        c.close()


if __name__ == '__main__':
    op = sys.argv[1]
    if op == 'pack':
        pack('--smoke' in sys.argv)
    elif op == 'upload':
        upload(sys.argv[2])
    elif op == 'start':
        n = int(sys.argv[sys.argv.index('--max') + 1]) if '--max' in sys.argv else 48
        load = float(sys.argv[sys.argv.index('--load') + 1]) if '--load' in sys.argv else 100.0
        start(sys.argv[2], n, load)
    elif op == 'width':
        width(sys.argv[2], sys.argv[3])
    elif op == 'status':
        status(sys.argv[2])
    elif op == 'fetch':
        fetch(sys.argv[2])
    else:
        raise SystemExit(op)
