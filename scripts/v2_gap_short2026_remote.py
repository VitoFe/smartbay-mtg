"""Addendum 82, part C: the compute-host side of the 1-3 h Gap-2026 runs.

  selections   (local) the published selections of the addendum-34 and addendum-35 tuned arms on
               Gap-2026, read from each published run's report.json, written to
               runs/v2_addendum82/C/published_selections.json
  pack         (local) a hash-manifested snapshot: the files the published gap-recovery campaign
               shipped (scripts/v2_campaign.plan('gaprecovery')), this part's module, the
               declaration and the selections
  upload       put the snapshot in <base>/review_20260919/v2_add82_C/<run>, unpack, verify every
               hash, record the runtime
  crosscheck   read each published study database that survives on the host (a copy, never the
               original) and compare its best trial with the injected selection
  launch       three detached workers: one on GPU 1 (every Chronos arm), two on CPU at two
               threads (the Gaussian processes; the anchored trees and ARX bridges)
  status       worker records and the load
  fetch        results (csv.gz, json) and logs, hash-verified, into runs/v2_addendum82/C/remote/<run>

Credentials are read by scientific_campaign.connect and never printed.
Usage: python -m scripts.v2_gap_short2026_remote <action> [--run RUN] [--quick]
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
RUNS = ROOT / 'reports/bay_twin/runs'
PART = RUNS / 'v2_addendum82' / 'C'
LOCAL = PART / 'remote'
SELECTIONS = PART / 'published_selections.json'
MODULE = 'src/experiments/v2_gap_short2026.py'
DECLARATION = 'config/bay_reserved_protocol_v2_addendum82.json'
WORK = 'review_20260919/v2_add82_C'
RESULTS = 'v2_gap_recovery_reserved2026'

#: published runs whose selections are injected (declaration, part C, arms)
A34 = {'oxygen': '20260922T215413Z_v2_gaprecovery', 'temperature_arx': '20260922T215413Z_v2_gaprecovery',
       'temperature_anchored': '20260923T101736Z_v2_gaprecovery'}
A35_RUNS = ('20260922T224158Z_v2_gaprecovery', '20260922T225405Z_v2_gaprecovery',
            '20260922T225420Z_v2_gaprecovery')
ANCHORED = ('anchored_gbm', 'anchored_gbm_exo', 'anchored_gbm_rel', 'anchored_gbm_rel_exo')
ARX = ('arx_bridge', 'arx_bridge_exo')
RESIDUAL = ('bolt_resid_bidir', 'chronos2_cov_resid_bidir', 'bolt_resid_lin_bidir',
            'chronos2_cov_resid_lin_bidir')
ZERO_SHOT = ('chronos_bidir', 'chronos_forward', 'chronos2_bidir', 'chronos2_cov_bidir',
             'chronos2_exo_bidir', 'chronos2_all_bidir')
GP = ('gpr', 'gpr_k8')
VARIABLES = ('temperature', 'oxygen')
KINDS = ('gate', 'short')


def published_source(variable, arm):
    if arm in ARX:
        return A34['temperature_arx'] if variable == 'temperature' else A34['oxygen']
    if arm in ANCHORED:
        return A34['temperature_anchored'] if variable == 'temperature' else A34['oxygen']
    done = [r for r in A35_RUNS if (RUNS / r / RESULTS / f'{variable}_{arm}' / 'report.json').exists()]
    if len(done) != 1:
        raise SystemExit(f'{variable} {arm}: completed in {len(done)} published runs, expected one')
    return done[0]


def selections():
    studies = {}
    for variable in VARIABLES:
        for arm in ANCHORED + ARX + RESIDUAL:
            run = published_source(variable, arm)
            report = json.loads((RUNS / run / RESULTS / f'{variable}_{arm}' / 'report.json').read_text())
            for entry in report['selections']:
                s = entry['selection']
                if entry['fold'] != 'frozen' or not s.get('study'):
                    raise SystemExit(f'{run} {variable} {arm}: no frozen selection recorded')
                if s.get('smoke_only') or s['completed_trials'] < 100:
                    raise SystemExit(f'{run} {variable} {arm}: selection is not a full study')
                studies[s['study']] = {'variable': variable, 'arm': arm, 'source_run': run,
                                       'protocol': s['protocol'], 'params': s['params'],
                                       'completed_trials': s['completed_trials'],
                                       'failed_trials': s['failed_trials'],
                                       'best_trial': s['best_trial'], 'best_value': s['best_value']}
    PART.mkdir(parents=True, exist_ok=True)
    record = {'declared_in': DECLARATION, 'part': 'C',
              'note': 'published selections of the tuned Gap-2026 arms, injected by '
                      'src/experiments/v2_gap_short2026.py in place of run_study',
              'studies': dict(sorted(studies.items()))}
    SELECTIONS.write_bytes((json.dumps(record, indent=2) + '\n').encode('utf-8'))
    print(f'{len(studies)} selections ->', SELECTIONS.relative_to(ROOT))


def snapshot_files():
    import v2_campaign
    files, _ = v2_campaign.plan('gaprecovery')
    files = list(dict.fromkeys(files + [ROOT / MODULE, ROOT / DECLARATION, SELECTIONS]))
    missing = [p for p in files if not p.exists()]
    if missing:
        raise SystemExit(f'missing: {missing[:5]}')
    return files


def pack(quick):
    run = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ') + '_v2_add82C' + ('_smoke' if quick else '')
    local = LOCAL / run
    local.mkdir(parents=True)
    files = snapshot_files()
    manifest = {'run_id': run, 'declared_in': DECLARATION, 'part': 'C', 'smoke_only': quick,
                'created_utc': datetime.now(timezone.utc).isoformat(),
                'files_sha256': {p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
                                 for p in files}}
    (local / 'snapshot_manifest.json').write_bytes((json.dumps(manifest, indent=2) + '\n').encode())
    with tarfile.open(local / 'snapshot.tar.gz', 'w:gz') as tar:
        for p in files:
            tar.add(p, arcname=p.relative_to(ROOT).as_posix())
        content = json.dumps(manifest).encode()
        info = tarfile.TarInfo('snapshot_manifest.json')
        info.size = len(content)
        tar.addfile(info, io.BytesIO(content))
    print(run, len(files), 'files', round((local / 'snapshot.tar.gz').stat().st_size / 1e6, 1), 'MB')


def session(tries=6):
    from scientific_campaign import connect
    for i in range(tries):
        try:
            return connect()
        except Exception as exc:  # the tunnel drops sessions now and then
            print(f'[connect attempt {i + 1} failed: {type(exc).__name__}]', flush=True)
            time.sleep(4 + 3 * i)
    raise SystemExit('compute host unreachable')


def run_cmd(client, command):
    from scientific_campaign import execute
    return execute(client, command)


def remote_dir(base, run):
    return f'{base}/{WORK}/' + ('smoke/' if run.endswith('_smoke') else '') + run


def upload(run):
    client, base = session()
    try:
        remote = remote_dir(base, run)
        code, out = run_cmd(client, 'mkdir -p ' + shlex.quote(remote))
        if code:
            raise SystemExit(out)
        sftp = client.open_sftp()
        try:
            sftp.put(str(LOCAL / run / 'snapshot.tar.gz'), remote + '/snapshot.tar.gz')
        finally:
            sftp.close()
        probe = """import tarfile,pathlib,hashlib,json,importlib.metadata,subprocess,socket,sys
p=pathlib.Path.cwd()
with tarfile.open('snapshot.tar.gz') as t: t.extractall(p,filter='data')
m=json.loads((p/'snapshot_manifest.json').read_text())
for name,want in m['files_sha256'].items():
 assert hashlib.sha256((p/name).read_bytes()).hexdigest()==want,name
pk={}
for n in ['numpy','pandas','scipy','scikit-learn','lightgbm','optuna','torch','chronos-forecasting','transformers']:
 try: pk[n]=importlib.metadata.version(n)
 except Exception: pk[n]=None
r={'host':socket.gethostname(),'python':sys.version.split()[0],'packages':pk,'gpu':subprocess.check_output(['nvidia-smi','--query-gpu=index,name,memory.used','--format=csv,noheader'],text=True)}
(p/'runtime.json').write_text(json.dumps(r,indent=2)); print(json.dumps(r))
"""
        py = base + '/.venv/bin/python'
        code, out = run_cmd(client, f'cd {shlex.quote(remote)} && {shlex.quote(py)} -c {shlex.quote(probe)}')
        print(out.replace(base, '<base>').strip())
        if code:
            raise SystemExit('unpack or hash check failed')
    finally:
        client.close()


def crosscheck(run):
    """Best trial of every surviving published study database against the injected selection."""
    client, base = session()
    try:
        remote = remote_dir(base, run)
        probe = r"""import json,pathlib,shutil,tempfile,optuna
optuna.logging.set_verbosity(optuna.logging.ERROR)
sel=json.loads(pathlib.Path('reports/bay_twin/runs/v2_addendum82/C/published_selections.json').read_text())['studies']
base=pathlib.Path('BASE')/'bay_twin'
out={}
for name,s in sel.items():
    db=base/s['source_run']/'reports/bay_twin/runs/v2_gap_recovery_reserved2026'/f"{s['variable']}_{s['arm']}"/f'{name}.sqlite3'
    if not db.exists():
        out[name]={'database':'absent'}; continue
    with tempfile.TemporaryDirectory() as d:
        copy=pathlib.Path(d)/'s.sqlite3'; shutil.copy2(db,copy)
        st=optuna.load_study(study_name=name,storage='sqlite:///'+str(copy))
        done=[t for t in st.trials if t.state==optuna.trial.TrialState.COMPLETE]
        same=st.best_params==s['params'] and st.best_trial.number==s['best_trial']
        out[name]={'database':'present','completed':len(done),'best_trial':st.best_trial.number,'agrees':bool(same)}
pathlib.Path('selection_crosscheck.json').write_text(json.dumps(out,indent=2))
print(json.dumps({'studies':len(out),'present':sum(v['database']=='present' for v in out.values()),'agree':sum(v.get('agrees',False) for v in out.values())}))
""".replace('BASE', base)
        py = base + '/.venv/bin/python'
        code, out = run_cmd(client, f'cd {shlex.quote(remote)} && {shlex.quote(py)} -c {shlex.quote(probe)}')
        print(out.replace(base, '<base>').strip())
    finally:
        client.close()


def worker_jobs():
    """(worker, gpu, jobs): gate jobs first, so the reproduction can be read before the rest ends."""
    gpu = [f'{k}:{v}:{a}' for k in KINDS for v in VARIABLES for a in ZERO_SHOT + RESIDUAL]
    gp = [f'{k}:{v}:{a}' for k in KINDS for v in VARIABLES for a in GP]
    trees = [f'{k}:{v}:{a}' for k in KINDS for v in VARIABLES for a in ANCHORED + ARX]
    return [('gpu', '1', gpu), ('gp', '', gp), ('trees', '', trees)]


def launch(run, quick, only=None):
    client, base = session()
    try:
        remote = remote_dir(base, run)
        py = base + '/.venv/bin/python'
        code, out = run_cmd(client, 'nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader')
        print(out.strip())
        record = []
        for name, gpu, jobs in worker_jobs():
            if only and name not in only:
                continue
            env = (f'OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 V2_THREADS=2 '
                   f'CUDA_VISIBLE_DEVICES={gpu} HF_HOME={base}/.hf HF_HUB_OFFLINE=1 PYTHONHASHSEED=0')
            cmd = (f'cd {shlex.quote(remote)} && mkdir -p logs results && env {env} {shlex.quote(py)} -u -m '
                   f'src.experiments.v2_gap_short2026 --jobs {",".join(jobs)} --worker {name} '
                   f'--selections reports/bay_twin/runs/v2_addendum82/C/published_selections.json '
                   f'--out results' + (' --quick' if quick else '') + f' > logs/{name}.log 2>&1')
            code, out = run_cmd(client, f"setsid -f bash -c {shlex.quote(cmd)} < /dev/null > /dev/null 2>&1")
            if code:
                raise SystemExit(out)
            record.append({'worker': name, 'gpu': gpu, 'threads': 2, 'jobs': jobs})
        time.sleep(5)
        code, out = run_cmd(client, "uptime; ps -eo nlwp,pcpu,args | grep '[v]2_gap_short2026' | cut -c1-160")
        print(out.replace(base, '<base>'))
        path = LOCAL / run / 'transport.json'
        earlier = json.loads(path.read_text())['workers'] if path.exists() else []
        record = earlier + record
        path.write_bytes((json.dumps(
            {'run_id': run, 'remote_dir': f'<base>/{WORK}/' + ('smoke/' if quick else '') + run,
             'workers': record, 'smoke_only': quick,
             'launched_utc': datetime.now(timezone.utc).isoformat()}, indent=2) + '\n').encode())
    finally:
        client.close()


def status(run):
    client, base = session()
    try:
        remote = remote_dir(base, run)
        probe = """import json,pathlib
for f in sorted(pathlib.Path('results').glob('worker_*.json')):
    w=json.loads(f.read_text()); j=w['jobs']
    print(w['worker'], len(j), 'done', sum(x['state']=='COMPLETED' for x in j), 'failed', [(x['kind'],x['variable'],x['arm'],x.get('error','')[:120]) for x in j if x['state']!='COMPLETED'], 'last', (j[-1]['kind'],j[-1]['variable'],j[-1]['arm'],j[-1].get('seconds')) if j else None)
"""
        py = base + '/.venv/bin/python'
        code, out = run_cmd(client, f'cd {shlex.quote(remote)} && {shlex.quote(py)} -c {shlex.quote(probe)}; '
                                    f"uptime; ps -eo nlwp,pcpu,etime,args | grep '[v]2_gap_short2026' | cut -c1-120; "
                                    # never the log tails: the producer prints its scores there, and
                                    # no 1-3 h score is read before the reproduction gate passes
                                    f"for f in logs/*.log; do echo == $f $(grep -c -E 'Traceback|Error' $f) error lines; done")
        print(out.replace(base, '<base>'))
    finally:
        client.close()


def fetch(run):
    client, base = session()
    try:
        remote = remote_dir(base, run)
        bundle = """import pathlib,tarfile,hashlib,json
p=pathlib.Path.cwd()
paths=[p/'runtime.json',p/'snapshot_manifest.json']+[f for f in (p/'results').rglob('*') if f.is_file() and f.name.endswith(('.json','.csv.gz'))]+list((p/'logs').glob('*.log'))+([p/'selection_crosscheck.json'] if (p/'selection_crosscheck.json').exists() else [])
with tarfile.open('results_download.tar.gz','w:gz',compresslevel=1) as a:
 for f in paths: a.add(f,arcname=f.relative_to(p).as_posix())
f=p/'results_download.tar.gz'; print(json.dumps({'sha256':hashlib.sha256(f.read_bytes()).hexdigest(),'bytes':f.stat().st_size,'files':len(paths)}))
"""
        py = base + '/.venv/bin/python'
        code, out = run_cmd(client, f'cd {shlex.quote(remote)} && {shlex.quote(py)} -c {shlex.quote(bundle)}')
        if code:
            raise SystemExit(out)
        expected = json.loads(out.strip().splitlines()[-1])
        local = LOCAL / run
        download = local / 'results_download.tar.gz'
        sftp = client.open_sftp()
        try:
            sftp.get(remote + '/results_download.tar.gz', str(download))
        finally:
            sftp.close()
        if hashlib.sha256(download.read_bytes()).hexdigest() != expected['sha256']:
            raise SystemExit('download checksum mismatch')
        with tarfile.open(download) as a:
            a.extractall(local, filter='data')
        (local / 'download_manifest.json').write_bytes((json.dumps(expected, indent=2) + '\n').encode())
        print('fetched and hash-verified', expected['files'], 'files')
    finally:
        client.close()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('action', choices=['selections', 'pack', 'upload', 'crosscheck', 'launch', 'status',
                                      'fetch'])
    p.add_argument('--run')
    p.add_argument('--quick', action='store_true')
    p.add_argument('--workers', help='launch only these workers (gpu, gp, trees)')
    a = p.parse_args()
    if a.action == 'selections':
        return selections()
    if a.action == 'pack':
        return pack(a.quick)
    if not a.run:
        raise SystemExit('--run is required')
    {'upload': upload, 'crosscheck': crosscheck, 'status': status, 'fetch': fetch}.get(
        a.action, lambda r: launch(r, a.run.endswith('_smoke'),
                                   a.workers.split(',') if a.workers else None))(a.run)


if __name__ == '__main__':
    main()
