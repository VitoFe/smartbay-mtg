"""Addendum 54 on the compute host: pack, upload, run, poll, fetch.

The pack holds the thesis's code, the probe geometry, the raw exports the gap-recovery functions read,
the weather station and the Magra, the companion's adapters and this addendum's inputs; never
credentials (the connection reads them itself and never prints them). The run reads the companion
precompute already on the host (its panels and zero-shot forecasts) and writes results/.

  python scripts/v2_o2_temperature_remote.py pack | upload | smoke | run | status | fetch
"""
import os
import sys
import tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
PRECOMPUTE = 'review_20260919/companion_precompute/cache'
#: Addendum 59 reran the gaps stage with the corrected Gaussian process: V2_O2T_RERUN=1 gives it its
#: own directory on the host and locally, packs the rerun's inputs (stack weights refitted) in place
#: of the published ones, and runs the gaps stage only. The published run is never touched.
RERUN = os.environ.get('V2_O2T_RERUN') == '1'
RUN = 'review_20260919/o2_temperature' + ('_gp_by_pattern' if RERUN else '')
#: holds raw data: kept out of the repository (V2_O2T_PACK moves it elsewhere)
PACK = Path(os.environ.get('V2_O2T_PACK', ROOT / 'companion/.cache/o2_temperature_pack.tar.gz'))
RESULTS = ROOT / 'reports/bay_twin/runs/v2_o2_temperature' / (
    'results_gp_by_pattern' if RERUN else 'results')
INPUTS = ROOT / 'reports/bay_twin/runs/v2_o2_temperature' / (
    'inputs_gp_by_pattern' if RERUN else 'inputs')
STAGES = '--stages gaps' if RERUN else ''


def pack():
    import pandas as pd
    sensors = pd.read_csv(ROOT / 'reports/bay_twin/sensor_geometry.csv').sensor.tolist()
    files = [ROOT / 'reports/bay_twin/sensor_geometry.csv', ROOT / 'data/raw/T2-Meteo.txt',
             ROOT / 'config/tuning_protocol_v2.json', ROOT / 'config/bay_reserved_protocol_v2.json',
             ROOT / 'scripts/experiments/v2_o2_temperature.py']
    files += [ROOT / 'data/raw' / f'{s}.txt' for s in sensors]
    files += sorted((ROOT / 'data/external/Rivers').glob('arpa_magra_ameglia_portata_*.csv'))
    files += sorted(INPUTS.glob('*'))
    for folder in (ROOT / 'src', ROOT / 'scripts', ROOT / 'companion/smartbay_companion'):
        files += [p for p in folder.rglob('*.py') if '__pycache__' not in p.parts]
    with tarfile.open(PACK, 'w:gz') as tar:
        for f in dict.fromkeys(files):
            if f.name.startswith('.env') or 'deploy.toml' in f.name:
                raise SystemExit(f'refusing to pack {f}')
            name = str(f.relative_to(ROOT)).replace('\\', '/')
            # the experiment reads its inputs from the published path; a rerun's go there in its pack
            name = name.replace('v2_o2_temperature/inputs_gp_by_pattern/', 'v2_o2_temperature/inputs/')
            tar.add(f, arcname=name)
    print(f'packed {len(set(files))} files into {PACK} ({PACK.stat().st_size / 1e6:.1f} MB)')


def main(action):
    if action == 'pack':
        return pack()
    from scientific_campaign import connect, execute
    client, base = connect()
    run, py = f'{base}/{RUN}', f'{base}/.venv/bin/python'

    def sh(cmd):
        code, out = execute(client, cmd)
        print(out.strip())
        return code
    try:
        if action == 'upload':
            sh(f'mkdir -p {run}/logs')
            sftp = client.open_sftp()
            try:
                sftp.put(str(PACK), f'{run}/pack.tar.gz')
            finally:
                sftp.close()
            sh(f'cd {run} && tar -xzf pack.tar.gz && ls && du -sh data')
        elif action in ('smoke', 'run'):
            flag = '--quick' if action == 'smoke' else ''
            sh(f'cd {run} && setsid nohup env CUDA_VISIBLE_DEVICES=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 '
               f'OPENBLAS_NUM_THREADS=1 V2_THREADS=3 V2_WORKERS=8 HF_HOME={base}/.hf PYTHONPATH={run} '
               f'SMARTBAY_FULL_CACHE={base}/{PRECOMPUTE} {py} -u scripts/experiments/v2_o2_temperature.py '
               f'{STAGES} {flag} > logs/{action}.log 2>&1 < /dev/null & disown')
        elif action == 'status':
            sh(f'cd {run} && for f in logs/*.log; do echo == $f; grep -v "Loading weights\\|HF Hub\\|gpr: fitted" $f | tail -n 12; done; '
               'ps -eo pid,etime,nlwp,pcpu,args | grep "[v]2_o2_temperature.py" | head -3; uptime; '
               'nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv')
        elif action == 'fetch':
            RESULTS.mkdir(parents=True, exist_ok=True)
            sftp = client.open_sftp()
            try:
                for item in sftp.listdir_attr(f'{run}/results'):
                    sftp.get(f'{run}/results/{item.filename}', str(RESULTS / item.filename))
                    print('got', item.filename, item.st_size)
                sftp.get(f'{run}/logs/run.log', str(RESULTS / 'run.log'))
            finally:
                sftp.close()
    finally:
        client.close()


if __name__ == '__main__':
    main(sys.argv[1])
