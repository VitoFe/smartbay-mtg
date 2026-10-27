"""Addenda 49, 51 and 52 on the compute host: snapshot, upload, queue launch, status, fetch.

One run directory <base>/review_20260919/t1/<run id> holds a hash-verified snapshot of what the jobs
read (the experiment modules, the raw data behind v2_horizon.build, the published recurrent and
phase-sweep records, and under a52/ the published Chronos scripts with their reference columns),
plus jobs.json and the queue runner. The published graph and tree runs are read in place under
<base>/bay_twin. Connection settings are read by scientific_campaign.connect and never printed.

  python -m scripts.v2_t1_remote upload [--jobs a,b]   build the snapshot and unpack it on the host
                                                       (--jobs keeps only the named jobs, in that order;
                                                       --graph-designs published,perfold narrows the
                                                       graph jobs: addendum 49 amendment 1)
  python -m scripts.v2_t1_remote launch [--after <id>] start the queue; --after makes it wait until
                                                       that run's queue has no GPU job pending
  python -m scripts.v2_t1_remote status                queue state, GPU, load, log tails
  python -m scripts.v2_t1_remote fetch                 results into reports/bay_twin/runs/
  python -m scripts.v2_t1_remote stop                  ask the queue to start nothing new
Add --run <id> to address an earlier run; the last one is recorded in runs/v2_t1_remote.json.
Declared in config/bay_reserved_protocol_v2_addendum49.json, _addendum51.json, _addendum52.json.
"""
import hashlib
import io
import json
import shlex
import sys
import tarfile
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
from scientific_campaign import connect, execute  # noqa: E402

RUNS = ROOT / 'reports/bay_twin/runs'
RECORD = RUNS / 'v2_t1_remote.json'
CODE = ROOT / 'scripts/experiments'
TRINITY = ROOT / 'reports/bay_twin/runs/battery'
PHASEFIX = RUNS / '20260923T103430Z_v2_phasefix/reports/bay_twin/runs/v2_phase_shift'
PY = '{base}/.venv/bin/python'

MODULES = ['src/experiments/v2_horizon.py', 'src/experiments/v2_perfold.py', 'src/experiments/v2_perfold_neural.py',
           'src/experiments/v2_sequence_arms.py', 'src/experiments/v2_sequence_sweep.py',
           'src/experiments/v2_phase_shift.py', 'src/data/thermodynamics.py', 'src/models/bay_query_field.py',
           'config/bay_reserved_protocol_v2_addendum49.json', 'config/bay_reserved_protocol_v2_addendum51.json',
           'config/bay_reserved_protocol_v2_addendum52.json',
           'config/bay_reserved_protocol_v2_addendum49_amendment1.json']
A52 = {'code/exp_chronos.py': CODE / 'exp_chronos.py', 'code/exp_chronos_valtest.py': CODE / 'exp_chronos_valtest.py',
       'v2_chronos.py': CODE / 'v2_chronos.py', 'v2_chronos2.py': CODE / 'v2_chronos2.py',
       'v2_chronos_c2.py': CODE / 'v2_chronos_c2.py', 'v2_chronos_lora.py': CODE / 'v2_chronos_lora.py',
       'v2_chronos_timeval.py': CODE / 'v2_chronos_timeval.py', 'v2_chronos_refit.py': CODE / 'v2_chronos_refit.py',
       'v2_aou_units_chronos.py': CODE / 'v2_aou_units_chronos.py',
       # addendum 77: weight interpolation between zero-shot and each addendum-42 selection
       'v2_chronos_wiseft.py': CODE / 'v2_chronos_wiseft.py',
       'selections.json': RUNS / 'v2_chronos_refit/selections.json',
       'ref/chronos_predictions.csv.gz': TRINITY / 'results/chronos/chronos_predictions.csv.gz',
       'ref/v2_chronos_c2_predictions.csv.gz': TRINITY / 'v2_chronos/v2_chronos_c2_predictions.csv.gz',
       'ref/v2_chronos_c2_report.json': TRINITY / 'v2_chronos/v2_chronos_c2_report.json',
       'ref/v2_chronos2_predictions.csv.gz': TRINITY / 'v2_chronos/v2_chronos2_predictions.csv.gz',
       'ref/v2_chronos_report_oxygen.json': TRINITY / 'v2_chronos/v2_chronos_report_oxygen.json',
       'ref/chronos_oxygen_aou_curve.csv': TRINITY / 'v2_chronos/chronos_oxygen_aou_curve.csv',
       'ref/v2_chronos_lora_report.json': RUNS / 'v2_chronos_lora/v2_chronos_lora_report.json',
       'ref/lora_oxygen_aou_curve.csv': RUNS / 'v2_chronos_lora/lora_oxygen_aou_curve.csv',
       'ref/lora_oxygen_aou_2026_columns.npz': RUNS / 'v2_chronos_lora/lora_oxygen_aou_2026_columns.npz',
       'ref/refit_timeval_full_oxygen_aou.json': RUNS / 'v2_chronos_refit/refit_timeval_full_oxygen_aou.json',
       'ref/refit_timeval_lora_oxygen_aou.json': RUNS / 'v2_chronos_refit/refit_timeval_lora_oxygen_aou.json'}


def snapshot_files():
    """(archive name, local path) for everything the jobs read from the snapshot."""
    import v2_campaign
    files = v2_campaign.base_files() + [ROOT / p for p in v2_campaign.FORECAST_FILES] + [ROOT / p for p in MODULES]
    files += sorted((RUNS / 'v2_sequence_arms').glob('*_rnn_fold*_selection.json'))
    files += sorted((RUNS / 'v2_sequence_arms').glob('sequence_arms_*.csv.gz'))
    files += sorted(PHASEFIX.glob('phase_shift_offset*_ratio*_summary.json'))
    out = [(p.relative_to(ROOT).as_posix(), p) for p in dict.fromkeys(files)]
    out += [(f'a52/{k}', v) for k, v in A52.items()]
    out += [('v2_t1_queue.py', CODE / 'v2_t1_queue.py')]
    return out


def jobs(base):
    py = PY.format(base=base)
    gpu = lambda name, *args, cwd='.': {'name': name, 'cmd': [py, '-u', *args], 'gpu': True, 'threads': 2, 'cwd': cwd}
    cpu = lambda name, *args: {'name': name, 'cmd': [py, '-u', *args], 'gpu': False, 'threads': 2, 'cwd': '.'}
    pn = ['-m', 'src.experiments.v2_perfold_neural']
    ss = ['-m', 'src.experiments.v2_sequence_sweep']
    graph = lambda v, a, h: gpu(f'a49_graph_{v}_{a}_h{h}', *pn, 'graph', '--variable', v, '--arm', a, '--horizon', str(h))
    from src.experiments.v2_perfold_neural import GRAPH_SIX, GRAPH_OTHER
    out = [gpu('a52_zeroshot', 'v2_aou_units_chronos.py', 'zeroshot', cwd='a52')]
    for v in ('oxygen', 'temperature'):
        out.append(gpu(f'a49_gru_{v}', *ss, 'gru', '--variable', v, '--offset', '0', '--sampler-seed', '42',
                       '--design', 'published', 'random_split', 'calendar'))
    for v in ('oxygen', 'temperature'):
        for o in range(6):
            out.append(gpu(f'a51_panel_{v}_offset{o}', *ss, 'panel', '--variable', v, '--offset', str(o)))
    for arm in ('ft_posthoc', 'lora_posthoc', 'tv_lora'):
        out.append(gpu(f'a52_adapted_{arm}', 'v2_aou_units_chronos.py', 'adapted', '--arm', arm, cwd='a52'))
    out += [graph(v, a, 6) for v in ('oxygen', 'temperature') for a in GRAPH_SIX[v]]
    for arm in ('tv_full', 'ft_random', 'lora_random'):
        out.append(gpu(f'a52_adapted_{arm}', 'v2_aou_units_chronos.py', 'adapted', '--arm', arm, cwd='a52'))
    for v in ('oxygen', 'temperature'):
        cells = [(o, 42) for o in range(1, 6)] + [(0, 7), (0, 123)]
        for o, s in cells:
            out.append(gpu(f'a51_gru_{v}_offset{o}_seed{s}', *ss, 'gru', '--variable', v, '--offset', str(o),
                           '--sampler-seed', str(s), '--design', 'random_split', 'calendar'))
    for h in (24, 1, 3, 12, 36, 48):
        out += [graph(v, a, h) for v in ('oxygen', 'temperature') for a in GRAPH_OTHER[v]]
    for h in (6, 24, 1, 3, 12, 36, 48):
        for v in ('oxygen', 'temperature'):
            out.append(cpu(f'a49_tree_{v}_h{h}', *pn, 'tree', '--variable', v, '--horizon', str(h)))
    for job in ('timeval_lora_oxygen_aou', 'timeval_full_oxygen_aou', 'timeval_lora_temperature_raw',
                'timeval_full_temperature_raw'):
        out.append(gpu(f'a77_wiseft_{job}', 'v2_chronos_wiseft.py', '--job', job, cwd='a52'))
    return out


def upload(args=()):
    client, base = connect()
    keep = args[args.index('--jobs') + 1].split(',') if '--jobs' in args else None
    graph_designs = args[args.index('--graph-designs') + 1].split(',') if '--graph-designs' in args else None
    try:
        run_id = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ') + '_t1'
        remote = f'{base}/review_20260919/t1/{run_id}'
        files = snapshot_files()
        manifest = {'run_id': run_id, 'created_utc': datetime.now(timezone.utc).isoformat(),
                    'declared_in': ['config/bay_reserved_protocol_v2_addendum49.json',
                                    'config/bay_reserved_protocol_v2_addendum51.json',
                                    'config/bay_reserved_protocol_v2_addendum52.json'],
                    'files_sha256': {}}
        buf = io.BytesIO()
        with tarfile.open(fileobj=buf, mode='w:gz', compresslevel=6) as tar:
            for name, path in files:
                data = path.read_bytes()
                if name.endswith('.py') or name.endswith('.sh'):
                    data = data.replace(b'\r\n', b'\n')
                manifest['files_sha256'][name] = hashlib.sha256(data).hexdigest()
                info = tarfile.TarInfo(name); info.size = len(data); info.mode = 0o644
                tar.addfile(info, io.BytesIO(data))
            by_name = {j['name']: j for j in jobs(base)}
            if keep is not None and not set(keep) <= set(by_name):
                raise SystemExit(f'unknown job names in --jobs: {sorted(set(keep) - set(by_name))}')
            job_list = list(by_name.values()) if keep is None else [by_name[n] for n in keep]   # in the order given
            if graph_designs:
                job_list = [dict(j, cmd=j['cmd'] + ['--design', *graph_designs]) if j['name'].startswith('a49_graph_')
                            else j for j in job_list]
            for name, obj in (('jobs.json', job_list), ('snapshot_manifest.json', manifest)):
                data = json.dumps(obj, indent=1).encode()
                info = tarfile.TarInfo(name); info.size = len(data)
                tar.addfile(info, io.BytesIO(data))
        print(f'{len(files)} files, {len(buf.getvalue()) / 1e6:.1f} MB compressed')
        code, out = execute(client, f'mkdir -p {remote}/logs')
        if code:
            raise RuntimeError(out)
        sftp = client.open_sftp()
        try:
            with sftp.open(f'{remote}/snapshot.tar.gz', 'wb') as f:
                f.set_pipelined(True)
                f.write(buf.getvalue())
        finally:
            sftp.close()
        probe = ("import tarfile,pathlib,hashlib,json\np=pathlib.Path.cwd()\n"
                 "with tarfile.open('snapshot.tar.gz') as t: t.extractall(p,filter='data')\n"
                 "m=json.loads((p/'snapshot_manifest.json').read_text())\n"
                 "bad=[n for n,h in m['files_sha256'].items() if hashlib.sha256((p/n).read_bytes()).hexdigest()!=h]\n"
                 "print(len(m['files_sha256']),'files verified' if not bad else ('MISMATCH',bad[:5]))\n")
        code, out = execute(client, f'cd {remote} && {PY.format(base=base)} -c {shlex.quote(probe)}')
        print('unpack:', code, out.strip())
        code, out = execute(client, f'cd {base}/review_20260919 && sha256sum code/exp_chronos.py code/exp_chronos_valtest.py '
                                    f'v2/v2_chronos.py v2/v2_chronos2.py v2/v2_chronos_c2.py v2_lora/v2_chronos_lora.py '
                                    f'v2_timeval/v2_chronos_timeval.py v2_refit/v2_chronos_refit.py')
        published = {line.split()[1]: line.split()[0] for line in out.strip().splitlines() if len(line.split()) == 2}
        mine = {k: manifest['files_sha256'][f'a52/{k2}'] for k, k2 in [
            ('code/exp_chronos.py', 'code/exp_chronos.py'), ('code/exp_chronos_valtest.py', 'code/exp_chronos_valtest.py'),
            ('v2/v2_chronos.py', 'v2_chronos.py'), ('v2/v2_chronos2.py', 'v2_chronos2.py'), ('v2/v2_chronos_c2.py', 'v2_chronos_c2.py'),
            ('v2_lora/v2_chronos_lora.py', 'v2_chronos_lora.py'), ('v2_timeval/v2_chronos_timeval.py', 'v2_chronos_timeval.py'),
            ('v2_refit/v2_chronos_refit.py', 'v2_chronos_refit.py')]}
        same = {k: published.get(k) == v for k, v in mine.items()}
        print('published Chronos scripts identical to the snapshot copies:', same)
        record = {'run_id': run_id, 'remote_dir_relative': f'review_20260919/t1/{run_id}', 'files': len(files),
                  'manifest_sha256': hashlib.sha256(json.dumps(manifest, indent=1).encode()).hexdigest(),
                  'published_chronos_scripts_identical': same, 'jobs': [j['name'] for j in job_list]}
        history = json.loads(RECORD.read_text()).get('history', []) if RECORD.exists() else []
        if RECORD.exists():
            old = json.loads(RECORD.read_text())
            history.append({k: v for k, v in old.items() if k != 'history'})
        record['history'] = history
        RECORD.write_text(json.dumps(record, indent=2))
    finally:
        client.close()


def run_dir(base, args):
    rid = args[args.index('--run') + 1] if '--run' in args else json.loads(RECORD.read_text())['run_id']
    return f'{base}/review_20260919/t1/{rid}', rid


def launch(args):
    client, base = connect()
    try:
        remote, rid = run_dir(base, args)
        running = execute(client, 'ps -eo pid,args | grep "[v]2_t1_queue.py" || true')[1].strip()
        # --after or --parallel: a second queue beside a running one; the queues count the lane's GPU
        # processes on the host, so together they never exceed T1_MAX_GPU.
        if running and '--after' not in args and '--parallel' not in args:
            raise SystemExit(f'a T1 queue is already running:\n{running}')
        print(execute(client, 'uptime; nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader')[1])
        after = ''
        if '--after' in args:
            after = f'T1_AFTER={base}/review_20260919/t1/{args[args.index("--after") + 1]}/logs/queue_status.json '
        cmd = (f'cd {remote} && {after}T1_BASE={base} setsid nohup {PY.format(base=base)} -u v2_t1_queue.py '
               f'> logs/queue.log 2>&1 < /dev/null & disown; sleep 3; cat {remote}/logs/queue_status.json | head -5')
        print(execute(client, cmd)[1])
    finally:
        client.close()


def status(args):
    client, base = connect()
    try:
        remote, rid = run_dir(base, args)
        probe = ("import json;s=json.load(open('logs/queue_status.json'));"
                 "print('queue',s['queue_pid'],'updated',s['updated'],'load',[round(x,1) for x in s['load']],'gpu1',s['gpu1_used_mb']);"
                 "[print(f\"{k:48s} {v['state']:8s} {v.get('pid','')} {v.get('start','')} {v.get('end','')} {v.get('rc','')}\") "
                 "for k,v in s['jobs'].items() if v['state']!='pending'];"
                 "print('pending',sum(v['state']=='pending' for v in s['jobs'].values()))")
        print(execute(client, f'cd {remote} && {PY.format(base=base)} -c {shlex.quote(probe)}; uptime; '
                              'nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader; '
                              'ps -eo pid,nlwp,pcpu,etime,args | grep -E "[v]2_(perfold_neural|sequence_sweep|aou_units)" | cut -c1-160')[1])
        if '--tail' in args:
            print(execute(client, f'cd {remote}/logs && for f in $(ls -t *.log | head -4); do echo "== $f"; tail -n 6 $f; done')[1])
    finally:
        client.close()


def stop(args):
    client, base = connect()
    try:
        remote, _ = run_dir(base, args)
        print(execute(client, f'touch {remote}/logs/STOP && echo requested')[1])
    finally:
        client.close()


def fetch(args):
    client, base = connect()
    try:
        remote, rid = run_dir(base, args)
        bundle = ("import pathlib,tarfile,hashlib,json\np=pathlib.Path.cwd()\n"
                  "paths=[f for d in ('reports/bay_twin/runs/v2_perfold_neural','reports/bay_twin/runs/v2_sequence_sweep',"
                  "'a52/results','logs') for f in (p/d).rglob('*') if f.is_file() and not f.name.endswith('.npy')]\n"
                  "with tarfile.open('results_download.tar.gz','w:gz',compresslevel=1) as a:\n"
                  " for f in paths: a.add(f,arcname=f.relative_to(p).as_posix())\n"
                  "f=p/'results_download.tar.gz'\n"
                  "print(json.dumps({'sha256':hashlib.sha256(f.read_bytes()).hexdigest(),'bytes':f.stat().st_size,'files':len(paths)}))\n")
        code, out = execute(client, f'cd {remote} && {PY.format(base=base)} -c {shlex.quote(bundle)}')
        if code:
            raise RuntimeError(out)
        expected = json.loads(out.strip().splitlines()[-1])
        local = RUNS / 'v2_t1_download'
        local.mkdir(parents=True, exist_ok=True)
        target = local / 'results_download.tar.gz'
        sftp = client.open_sftp()
        try:
            sftp.get(f'{remote}/results_download.tar.gz', str(target), max_concurrent_prefetch_requests=64)
        finally:
            sftp.close()
        if hashlib.sha256(target.read_bytes()).hexdigest() != expected['sha256']:
            raise ValueError('download checksum mismatch')
        with tarfile.open(target) as a:
            members = a.getmembers()
            for m in members:
                name = m.name
                if name.startswith('reports/bay_twin/runs/'):
                    dest = ROOT / name
                elif name.startswith('a52/results/'):
                    dest = RUNS / 'v2_aou_units_chronos' / name.removeprefix('a52/results/')
                else:
                    dest = RUNS / 'v2_t1_logs' / rid / name.removeprefix('logs/')
                dest.parent.mkdir(parents=True, exist_ok=True)
                dest.write_bytes(a.extractfile(m).read())
        target.unlink()
        local.rmdir()
        print('fetched and hash-verified', expected['files'], 'files from', rid)
    finally:
        client.close()


if __name__ == '__main__':
    action = sys.argv[1]
    {'upload': lambda: upload(sys.argv), 'launch': lambda: launch(sys.argv), 'status': lambda: status(sys.argv),
     'fetch': lambda: fetch(sys.argv), 'stop': lambda: stop(sys.argv)}[action]()
