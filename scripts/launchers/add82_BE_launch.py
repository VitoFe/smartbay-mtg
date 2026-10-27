"""Addendum 82 parts B and E on the compute host: hash-verified snapshot, a bounded job queue, status, fetch.

usage: python launch.py pack   B|E [--smoke]        -> builds the snapshot and job scripts locally
       python launch.py upload B|E <run_id>         -> uploads, extracts, verifies hashes, writes runtime.json
       python launch.py start  B|E <run_id> [--max N]
       python launch.py status B|E <run_id>
       python launch.py fetch  B|E <run_id>
Credentials are read by scientific_campaign.connect() and never printed.
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

REPO = Path(r'C:\Users\vferri\git\smartbay-do')
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO / 'scripts'))
sys.path.insert(0, str(REPO))
from scientific_campaign import connect, execute  # noqa: E402

PY = '<base>/.venv/bin/python'


def remote_dir(part, run):
    return f"<base>/review_20260919/v2_add82_{'B' if part == 'B' else 'E'}/{run}"


def session(tries=6):
    for i in range(tries):
        try:
            return connect()
        except Exception as e:
            print(f'[connect attempt {i + 1} failed: {type(e).__name__}]', flush=True)
            time.sleep(4 + 3 * i)
    sys.exit('unreachable')


def files_for(part):
    import v2_campaign as vc
    files = vc.base_files() + [REPO / p for p in vc.FORECAST_FILES]
    files += [p for p in sorted((REPO / 'src').rglob('*.py')) if '__pycache__' not in p.parts]
    files += [REPO / 'config/bay_reserved_protocol_v2_addendum82.json', REPO / 'config/tuning_protocol_v2.json']
    for t in sorted(HERE.glob('trace_*.json')):
        files += [REPO / p for p in json.loads(t.read_text())]
    files += sorted((REPO / 'data/external/Rivers').glob('arpa_magra_ameglia_portata_*.csv'))
    files += [REPO / 'data/external/Ispra/ISPRA_La_Spezia_2026_06_10.csv']
    if part == 'E':
        runs = REPO / 'reports/bay_twin/runs'
        for f in ('v2_forecast_temperature_learned_source_nested', 'v2_forecast_temperature_conservative_exchange_nested',
                  'v2_forecast_oxygen_learned_source_nested', 'v2_forecast_oxygen_conservative_exchange_nested',
                  'v2_forecast_oxygen_saturation_exchange_nested', 'v2_forecast_oxygen_saturation_source_nested'):
            folder = runs / '20260919T144822Z_v2_forecast/results' / f
            files += [folder / 'selected_predictions.csv.gz', folder / f"{f.removeprefix('v2_forecast_')}_selection.json"]
        for f in ('v2_horizon_temperature_learned_source_h6', 'v2_horizon_temperature_conservative_exchange_h6',
                  'v2_horizon_oxygen_learned_source_h6', 'v2_horizon_oxygen_conservative_exchange_h6',
                  'v2_horizon_oxygen_saturation_exchange_h6', 'v2_horizon_oxygen_learned_source_h24'):
            folder = runs / '20260920T083322Z_v2_horizon/results' / f
            files += [folder / 'selected_predictions.csv.gz', folder / f"{f.removeprefix('v2_horizon_')}_selection.json"]
    files = list(dict.fromkeys(files))
    missing = [p for p in files if not p.is_file()]
    if missing:
        raise SystemExit(f'missing: {missing[:5]}')
    return files


def jobs_for(part, smoke):
    q = ['--quick'] if smoke else []
    jobs = []
    if part == 'B':
        cells = [(k, None) for k in range(6)] + [(0, 7), (0, 123)]
        sfx = lambda s: '' if s is None else f'_seed{s}'
        sarg = lambda s: [] if s is None else ['--seed', str(s)]
        arms = [('oxygen', 6, 'weather_river_tree'), ('oxygen', 6, 'sealevel_tree'), ('oxygen', 48, 'sealevel_tree'),
                ('oxygen', 6, 'extboundary_tree'), ('oxygen', 24, 'bathy_tree'), ('temperature', 6, 'sealevel_tree'),
                ('temperature', 6, 'extboundary_tree'), ('temperature', 24, 'bathy_tree')]
        if smoke:
            jobs.append(('base_oxygen_h48_offset1', ['src.experiments.v2_phase_family', 'base', '--variable', 'oxygen',
                                                     '--horizon', '48', '--offset', '1'] + q))
            for v, h, a in arms[:2] + arms[4:5] + arms[6:7]:
                jobs.append((f'{a}_{v}_h{h}_offset2', ['src.experiments.v2_phase_family', 'arm', '--arm', a,
                                                       '--variable', v, '--horizon', str(h), '--offset', '2'] + q))
            return jobs
        for o, s in cells:
            jobs.append((f'base_oxygen_h48_offset{o}{sfx(s)}', ['src.experiments.v2_phase_family', 'base', '--variable',
                                                               'oxygen', '--horizon', '48', '--offset', str(o)] + sarg(s)))
        for v, h, a in arms:
            for o, s in cells:
                jobs.append((f'{a}_{v}_h{h}_offset{o}{sfx(s)}',
                             ['src.experiments.v2_phase_family', 'arm', '--arm', a, '--variable', v, '--horizon', str(h),
                              '--offset', str(o)] + sarg(s)))
        return jobs
    # part E
    mod = 'src.experiments.v2_phase_graph'
    refits = [('temperature', 'learned_source', 6), ('temperature', 'conservative_exchange', 6),
              ('oxygen', 'learned_source', 6), ('oxygen', 'conservative_exchange', 6),
              ('oxygen', 'saturation_exchange', 6), ('oxygen', 'saturation_source', 6), ('oxygen', 'learned_source', 24)]
    if smoke:
        jobs.append(('smoke_temperature_learned_source_h6_offset1',
                     [mod, '--variable', 'temperature', '--arm', 'learned_source', '--horizon', '6', '--offset', '1'] + q))
        jobs.append(('smoke_oxygen_saturation_source_h6_offset0_seed7',
                     [mod, '--variable', 'oxygen', '--arm', 'saturation_source', '--horizon', '6', '--offset', '0',
                      '--seed', '7'] + q))
        return jobs
    if part == 'Egates':
        pass
    return jobs


def e_gate_jobs():
    mod = 'src.experiments.v2_phase_graph'
    refits = [('temperature', 'learned_source', 6), ('temperature', 'conservative_exchange', 6),
              ('oxygen', 'learned_source', 6), ('oxygen', 'conservative_exchange', 6),
              ('oxygen', 'saturation_exchange', 6), ('oxygen', 'saturation_source', 6), ('oxygen', 'learned_source', 24)]
    return [(f'refit_{v}_{a}_h{h}', [mod, '--variable', v, '--arm', a, '--horizon', str(h), '--refit-published'])
            for v, a, h in refits]


def e_g3_job():
    db = '<base>/bay_twin/20260919T144822Z_v2_forecast/results/v2_forecast_temperature_learned_source_nested/' \
         'temperature_learned_source_nested.sqlite3'
    return [('reproduce_temperature_learned_source_h6',
             ['src.experiments.v2_phase_graph', '--variable', 'temperature', '--arm', 'learned_source', '--horizon', '6',
              '--reproduce', '--published-db', db])]


def e_study_jobs():
    mod = 'src.experiments.v2_phase_graph'
    out = []
    phases = [(k, None) for k in range(1, 6)]
    all8 = phases + [(0, 7), (0, 123)]
    plan = [('temperature', 'learned_source', 6, phases), ('temperature', 'conservative_exchange', 6, phases),
            ('oxygen', 'conservative_exchange', 6, phases), ('oxygen', 'learned_source', 24, phases),
            ('oxygen', 'saturation_exchange', 6, all8), ('oxygen', 'saturation_source', 6, all8),
            ('oxygen', 'learned_source', 6, phases)]
    for v, a, h, cells in plan:
        for o, s in cells:
            name = f'{v}_{a}_h{h}_offset{o}' + ('' if s is None else f'_seed{s}')
            out.append((name, [mod, '--variable', v, '--arm', a, '--horizon', str(h), '--offset', str(o)]
                        + ([] if s is None else ['--seed', str(s)])))
    return out


def job_script(part, name, args):
    gpu = '' if part == 'B' else '1'
    cmd = ' '.join(shlex.quote(x) for x in args)
    return ('#!/bin/bash\n'
            'cd "$(dirname "$0")/.."\n'
            'unset V2_ISPRA_BINNING\n'
            f"export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 V2_THREADS=2 "
            f"CUDA_VISIBLE_DEVICES='{gpu}' PYTHONDONTWRITEBYTECODE=1\n"
            f'{PY}' + f' -u -m {cmd} > logs/{name}.log 2>&1\n'
            f'echo $? > status/{name}.exit\n')


def pack(part, smoke, which=None):
    files = files_for('E' if part.startswith('E') else part)
    run = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ') + f'_add82{part}' + ('_smoke' if smoke else '')
    local = HERE / 'runs' / run
    local.mkdir(parents=True)
    manifest = {'run_id': run, 'declared_in': 'config/bay_reserved_protocol_v2_addendum82.json', 'part': part,
                'smoke_only': smoke, 'created_utc': datetime.now(timezone.utc).isoformat(),
                'files_sha256': {p.relative_to(REPO).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for p in files}}
    if part == 'B' or smoke:
        jobs = jobs_for('E' if part.startswith('E') else part, smoke)
    elif part == 'Egates':
        jobs = e_gate_jobs()
    elif part == 'E':
        jobs = e_g3_job() + e_study_jobs()
    else:
        raise SystemExit(part)
    manifest['jobs'] = [{'name': n, 'args': a} for n, a in jobs]
    (local / 'snapshot_manifest.json').write_text(json.dumps(manifest, indent=2))
    with tarfile.open(local / 'snapshot.tar.gz', 'w:gz') as tar:
        for p in files:
            tar.add(p, arcname=p.relative_to(REPO).as_posix())
        extra = {'snapshot_manifest.json': json.dumps(manifest).encode(),
                 'queue.txt': ('\n'.join(n for n, _ in jobs) + '\n').encode()}
        for n, a in jobs:
            extra[f'jobs/{n}.sh'] = job_script('B' if part == 'B' else 'E', n, a).encode()
        for name, content in extra.items():
            info = tarfile.TarInfo(name)
            info.size = len(content)
            info.mode = 0o755 if name.endswith('.sh') else 0o644
            tar.addfile(info, io.BytesIO(content))
    print(run, len(files), 'files', len(jobs), 'jobs', round((local / 'snapshot.tar.gz').stat().st_size / 1e6, 1), 'MB')


def sub(base, s):
    return s.replace('<base>', base)


def upload(part, run):
    local = HERE / 'runs' / run
    c, base = session()
    try:
        remote = sub(base, remote_dir(part, run))
        code, out = execute(c, f'mkdir -p {shlex.quote(remote)}/logs {shlex.quote(remote)}/status')
        assert code == 0, out
        sftp = c.open_sftp()
        try:
            sftp.put(str(local / 'snapshot.tar.gz'), remote + '/snapshot.tar.gz')
        finally:
            sftp.close()
        probe = ("import tarfile,pathlib,hashlib,json,importlib.metadata,subprocess,platform,socket\n"
                 "p=pathlib.Path.cwd()\n"
                 "t=tarfile.open('snapshot.tar.gz'); t.extractall(p,filter='data'); t.close()\n"
                 "m=json.loads((p/'snapshot_manifest.json').read_text())\n"
                 "bad=[n for n,w in m['files_sha256'].items() if hashlib.sha256((p/n).read_bytes()).hexdigest()!=w]\n"
                 "assert not bad, bad[:5]\n"
                 "r={'host':socket.gethostname(),'python':platform.python_version(),'packages':{n:importlib.metadata.version(n) for n in ['numpy','pandas','torch','optuna','lightgbm','scikit-learn']},"
                 "'gpu':subprocess.check_output(['nvidia-smi','--query-gpu=index,name,memory.used,memory.total','--format=csv,noheader'],text=True)}\n"
                 "(p/'runtime.json').write_text(json.dumps(r,indent=2)); print(json.dumps(r)); print('verified',len(m['files_sha256']))\n")
        code, out = execute(c, f'cd {shlex.quote(remote)} && {sub(base, PY)} -c {shlex.quote(probe)}')
        print(out.replace(base, '<base>'))
        assert code == 0
    finally:
        c.close()


def start(part, run, n):
    c, base = session()
    try:
        remote = sub(base, remote_dir(part, run))
        # the job scripts carry the '<base>' placeholder for the interpreter: expand it in place
        code, out = execute(c, f"cd {shlex.quote(remote)} && sed -i 's#<base>#{base}#g' jobs/*.sh && "
                               f"setsid -f bash -c 'xargs -a queue.txt -P {int(n)} -I{{}} bash jobs/{{}}.sh "
                               f"> queue.log 2>&1; date -u > queue.done' < /dev/null > /dev/null 2>&1; sleep 5; "
                               f"ps -eo pid,nlwp,pcpu,args | grep '[v]2_phase_' | wc -l; uptime")
        print(out.replace(base, '<base>'))
    finally:
        c.close()


def status(part, run):
    c, base = session()
    try:
        remote = sub(base, remote_dir(part, run))
        code, out = execute(c, f"cd {shlex.quote(remote)} && echo jobs $(wc -l < queue.txt) done $(ls status | wc -l) "
                               f"failed $(grep -L '^0$' status/*.exit 2>/dev/null | wc -l); "
                               f"ls queue.done 2>/dev/null; grep -L '^0$' status/*.exit 2>/dev/null | head; "
                               f"ps -eo nlwp,pcpu,args | grep '[v]2_phase_' | awk '{{print $1, $2, $NF}}' | head -20; uptime; "
                               f"nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader")
        print(out.replace(base, '<base>'))
    finally:
        c.close()


def fetch(part, run):
    c, base = session()
    try:
        remote = sub(base, remote_dir(part, run))
        sub_out = 'B' if part == 'B' else 'E'
        bundle = ("import pathlib,tarfile,hashlib,json\n"
                  "p=pathlib.Path.cwd()\n"
                  f"o=p/'reports/bay_twin/runs/v2_addendum82/{sub_out}'\n"
                  "paths=[f for f in o.rglob('*') if f.is_file()] if o.exists() else []\n"
                  "paths+=[p/'runtime.json',p/'snapshot_manifest.json',p/'queue.txt']+[f for f in p.glob('queue.*') if f.is_file()]"
                  "+list((p/'logs').glob('*.log'))+list((p/'status').glob('*.exit'))\n"
                  "with tarfile.open('results_download.tar.gz','w:gz',compresslevel=1) as a:\n"
                  " for f in paths: a.add(f,arcname=f.relative_to(p).as_posix())\n"
                  "f=p/'results_download.tar.gz'; print(json.dumps({'sha256':hashlib.sha256(f.read_bytes()).hexdigest(),'files':len(paths)}))\n")
        code, out = execute(c, f'cd {shlex.quote(remote)} && {sub(base, PY)} -c {shlex.quote(bundle)}')
        assert code == 0, out
        expected = json.loads(out.strip().splitlines()[-1])
        local = HERE / 'runs' / run
        local.mkdir(parents=True, exist_ok=True)
        dl = local / 'results_download.tar.gz'
        sftp = c.open_sftp()
        try:
            sftp.get(remote + '/results_download.tar.gz', str(dl))
        finally:
            sftp.close()
        assert hashlib.sha256(dl.read_bytes()).hexdigest() == expected['sha256'], 'checksum'
        target = REPO / 'reports/bay_twin/runs/v2_addendum82' / sub_out
        record = target / f'run_{run}'
        with tarfile.open(dl) as a:
            for m in a.getmembers():
                name = m.name
                if name.startswith(f'reports/bay_twin/runs/v2_addendum82/{sub_out}/'):
                    dest = REPO / name
                else:
                    dest = record / name
                if dest.exists():
                    if hashlib.sha256(dest.read_bytes()).hexdigest() != hashlib.sha256(a.extractfile(m).read()).hexdigest():
                        raise SystemExit(f'refusing to overwrite a different file: {dest}')
                    continue
                dest.parent.mkdir(parents=True, exist_ok=True)
                dest.write_bytes(a.extractfile(m).read())
        (record / 'download_manifest.json').parent.mkdir(parents=True, exist_ok=True)
        (record / 'download_manifest.json').write_text(json.dumps(expected, indent=2))
        print('fetched and verified', expected['files'], 'files into', target)
    finally:
        c.close()


if __name__ == '__main__':
    op, part = sys.argv[1], sys.argv[2]
    if op == 'pack':
        pack(part, '--smoke' in sys.argv)
    elif op == 'upload':
        upload(part, sys.argv[3])
    elif op == 'start':
        n = int(sys.argv[sys.argv.index('--max') + 1]) if '--max' in sys.argv else 8
        start(part, sys.argv[3], n)
    elif op == 'status':
        status(part, sys.argv[3])
    elif op == 'fetch':
        fetch(part, sys.argv[3])
