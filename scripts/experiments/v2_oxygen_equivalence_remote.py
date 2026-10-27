"""Addendum 60 part B on the compute host: pack, upload, launch, status, fetch.

  python scripts/experiments/v2_oxygen_equivalence_remote.py pack
  python scripts/experiments/v2_oxygen_equivalence_remote.py upload
  python scripts/experiments/v2_oxygen_equivalence_remote.py launch [b2 b3 b5 b6 b5c]
  python scripts/experiments/v2_oxygen_equivalence_remote.py refresh   (re-uploads the producer alone)
  python scripts/experiments/v2_oxygen_equivalence_remote.py status
  python scripts/experiments/v2_oxygen_equivalence_remote.py fetch

The pack holds the code (src/ and scripts/v2_oxygen_equivalence.py), the probe and meteorological
exports the forecasting grid and the Gap-Dev cohort are built from, the sensor and exchange
geometry, part A's hourly factor table, and the published rows and selections that arms B2, B3, B5
and B6 refit and gate on. Never credentials: the connection reads them itself and prints none.

Each arm runs detached with its own thread count, set explicitly on every estimator through
V2_THREADS: B2 on four CPU threads, as the published XGBoost run (its row subsampling depends on the
thread count); B3 and B6 on GPU 1 with two threads, B5 on two CPU threads, as their published runs.
Results are written under reports/bay_twin/runs/v2_oxygen_mgl/equivalence/<arm>/ on the host and
fetched to the same place here; the run record is reports/bay_twin/runs/v2_oxygen_mgl/equivalence/remote_run.json.
"""
import json
import sys
import tarfile
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'scripts'))
RUNS = ROOT / 'reports/bay_twin/runs'
OUT = RUNS / 'v2_oxygen_mgl/equivalence'
RECORD = OUT / 'remote_run.json'
PACK = ROOT / 'companion/.cache/oxygen_equivalence_pack.tar.gz'   # holds raw data: git-ignored
REMOTE_ARMS = ('B2', 'B3', 'B5', 'B6')
#: arm -> (threads, GPU index or ''); B5C is B5 with unit-consistent kernel constants (b5-consistent),
#: which reads B5's rows and so runs on a snapshot where B5 has already run
JOBS = {'B2': (4, ''), 'B3': (2, '1'), 'B5': (2, ''), 'B6': (2, '1'), 'B5C': (2, '')}
COMMANDS = {'B5C': 'b5-consistent'}
#: folders fetched back, under reports/bay_twin/runs/v2_oxygen_mgl/equivalence/
FETCHED = REMOTE_ARMS + ('B5_unit_consistent',)
PUBLISHED = [
    # B2: the published XGBoost weather tree, its selection and its stored fold models (gate G1x-a)
    *[f'20260919T183751Z_v2_xgb/results/v2_forecast_oxygen_xgb_weather_tree_nested/{f}' for f in (
        'oxygen_xgb_weather_tree_nested_selection.json', 'report.json', 'selected_predictions.csv.gz',
        'selected_fold0_model.json', 'selected_fold1_model.json', 'contract.json')],
    # B3: the published learned-source graph arm
    *[f'20260919T144822Z_v2_forecast/results/v2_forecast_oxygen_learned_source_nested/{f}' for f in (
        'oxygen_learned_source_nested_selection.json', 'report.json', 'selected_predictions.csv.gz',
        'contract.json')],
    # B5, B6: the published fills, and the table values they are checked against
    '20261001T110649Z_v2_gaprecovery/v2_gap_recovery/oxygen_gpr/predictions.csv.gz',
    '20261001T110649Z_v2_gaprecovery/v2_gap_recovery/oxygen_gpr/report.json',
    '20260922T171859Z_v2_gaprecovery/v2_gap_recovery/oxygen_chronos_bidir/predictions.csv.gz',
    '20260922T171859Z_v2_gaprecovery/v2_gap_recovery/oxygen_chronos_bidir/report.json',
    'v2_gap_variants.json',
    # part A
    'v2_oxygen_units.json', 'v2_oxygen_units/hourly_factor.csv.gz']


def files():
    import csv
    out = [ROOT / 'scripts/v2_oxygen_equivalence.py', ROOT / 'config/tuning_protocol_v2.json',
           ROOT / 'config/bay_reserved_protocol_v2_addendum60.json',
           ROOT / 'config/bay_reserved_protocol_v2_addendum60_amendment1.json',
           ROOT / 'reports/bay_twin/sensor_geometry.csv', ROOT / 'reports/bay_twin/exchange_geometry.json',
           ROOT / 'data/raw/T2-Meteo.txt']
    with (ROOT / 'reports/bay_twin/sensor_geometry.csv').open() as stream:
        out += [ROOT / 'data/raw' / f'{row["sensor"]}.txt' for row in csv.DictReader(stream)]
    out += [RUNS / p for p in PUBLISHED]
    out += [p for p in (ROOT / 'src').rglob('*.py') if '__pycache__' not in p.parts]
    return list(dict.fromkeys(out))


def pack():
    PACK.parent.mkdir(parents=True, exist_ok=True)
    chosen = files()
    missing = [str(f) for f in chosen if not f.exists()]
    if missing:
        raise SystemExit(f'missing: {missing[:5]}')
    with tarfile.open(PACK, 'w:gz') as tar:
        for f in chosen:
            if f.name.startswith('.env') or 'deploy.toml' in f.name:
                raise SystemExit(f'refusing to pack {f}')
            tar.add(f, arcname=f.relative_to(ROOT).as_posix())
    print(f'packed {len(chosen)} files into {PACK} ({PACK.stat().st_size / 1e6:.1f} MB)')


def _record():
    return json.loads(RECORD.read_text()) if RECORD.exists() else {}


def _save(record):
    RECORD.parent.mkdir(parents=True, exist_ok=True)
    RECORD.write_bytes((json.dumps(record, indent=2) + '\n').encode('utf-8'))


def main(action, arms):
    if action == 'pack':
        return pack()
    from scientific_campaign import connect, execute
    client, base = connect()
    record = _record()

    def sh(cmd):
        code, out = execute(client, cmd)
        print(out.strip())
        return code, out
    try:
        if action == 'upload':
            stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
            run = f'{base}/bay_twin/{stamp}_v2_oxygen_equivalence'
            sh(f'mkdir -p {run}/logs')
            sftp = client.open_sftp()
            try:
                sftp.put(str(PACK), f'{run}/pack.tar.gz')
            finally:
                sftp.close()
            sh(f'cd {run} && tar -xzf pack.tar.gz && rm pack.tar.gz && du -sh . && ls')
            record = {'run_dir': run, 'uploaded_utc': stamp, 'files': len(files()), 'jobs': {}}
            _save(record)
        elif action == 'refresh':
            # the producer alone, replaced in the uploaded snapshot before a relaunch
            src = ROOT / 'scripts/v2_oxygen_equivalence.py'
            sftp = client.open_sftp()
            try:
                sftp.put(str(src), f'{record["run_dir"]}/scripts/v2_oxygen_equivalence.py')
            finally:
                sftp.close()
            import hashlib
            record.setdefault('producer_refreshed', []).append(
                {'utc': datetime.now(timezone.utc).isoformat(timespec='seconds'),
                 'sha256': hashlib.sha256(src.read_bytes()).hexdigest()})
            _save(record)
            print('producer replaced on the host')
        elif action == 'launch':
            run, py = record['run_dir'], f'{base}/.venv/bin/python'
            for arm in arms or REMOTE_ARMS:
                arm = arm.upper()
                threads, gpu = JOBS[arm]
                env = (f'OMP_NUM_THREADS={threads} MKL_NUM_THREADS={threads} OPENBLAS_NUM_THREADS={threads} '
                       f'V2_THREADS={threads} CUDA_VISIBLE_DEVICES={gpu} PYTHONHASHSEED=0 PYTHONPATH={run}')
                command = COMMANDS.get(arm, arm.lower())
                cmd = f'env {env} {py} -u -m scripts.v2_oxygen_equivalence {command} > logs/{arm}.log 2>&1'
                if arm == 'B5':   # then the kernel-bound diagnostic, which reads B5's rows
                    cmd += (f'; env {env} {py} -u -m scripts.v2_oxygen_equivalence b5-kernels '
                            f'> logs/B5_kernels.log 2>&1')
                sh(f'cd {run} && setsid -f nohup sh -c "{cmd}" < /dev/null > /dev/null 2>&1')
                record['jobs'][arm] = {'threads': threads, 'gpu': gpu or None,
                                       'launched_utc': datetime.now(timezone.utc).isoformat(timespec='seconds')}
                print(f'launched {arm} ({threads} threads, GPU {gpu or "none"})')
            _save(record)
            sh('sleep 20; ps -eo pid,etime,nlwp,pcpu,args | grep "[v]2_oxygen_equivalence"; uptime; '
               'nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader')
        elif action == 'status':
            run = record['run_dir']
            sh(f'cd {run} && for f in logs/*.log; do echo "== $f"; grep -v Warning $f | tail -n 4; done; '
               'ps -eo pid,etime,nlwp,pcpu,args | grep "[v]2_oxygen_equivalence"; uptime; '
               'nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader')
        elif action == 'fetch':
            run = record['run_dir']
            sftp = client.open_sftp()
            got = 0
            try:
                def walk(remote, local):
                    nonlocal got
                    from stat import S_ISDIR
                    for entry in sftp.listdir_attr(remote):
                        r, l = f'{remote}/{entry.filename}', local / entry.filename
                        if S_ISDIR(entry.st_mode):
                            walk(r, l)
                        else:
                            l.parent.mkdir(parents=True, exist_ok=True)
                            sftp.get(r, str(l))
                            got += 1
                for arm in FETCHED:
                    remote = f'{run}/reports/bay_twin/runs/v2_oxygen_mgl/equivalence/{arm}'
                    try:
                        sftp.stat(remote)
                    except FileNotFoundError:
                        print(f'{arm}: nothing yet')
                        continue
                    walk(remote, OUT / arm)
                logs = OUT / 'remote_logs'
                logs.mkdir(parents=True, exist_ok=True)
                for f in sftp.listdir(f'{run}/logs'):
                    sftp.get(f'{run}/logs/{f}', str(logs / f))
                    got += 1
            finally:
                sftp.close()
            record['fetched_utc'] = datetime.now(timezone.utc).isoformat(timespec='seconds')
            _save(record)
            print(f'fetched {got} files')
        else:
            raise SystemExit(f'unknown action {action}')
    finally:
        client.close()


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2:])
