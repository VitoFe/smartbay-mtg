"""Addendum 57 on the compute host: pack, upload, smoke, run, status, fetch.

  python scripts/v2_fouling_jump_remote.py pack | upload | smoke | run [WORKERS] | status | fetch

The pack holds the thesis's code, the four probe exports (the two hand-cleaned first-generation oxygen
probes and the wiped second-generation sondes at the same sites), the station logbook, the Santa Teresa
weather file, the Magra series and the CTD casts; never credentials (the connection reads them itself
and never prints them). The run writes reports/bay_twin/runs/v2_fouling_jump*.
"""
import sys
import tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
RUN = 'review_20260919/fouling_jump'
PACK = ROOT / 'companion/.cache/fouling_jump_pack.tar.gz'   # holds raw data: git-ignored
PROBES = ('TBaia_01m-Ossigeno(calibrato)', 'TMitiliE_05m-Ossigeno(calibrato)',
          'TBaiaP_01m-CTD(calibrato)', 'TMitiliE_05m-CTD(calibrato)')
RESULTS = ROOT / 'reports/bay_twin/runs'


def pack():
    files = [ROOT / 'config/tuning_protocol_v2.json', ROOT / 'config/bay_reserved_protocol_v2.json',
             ROOT / 'config/bay_reserved_protocol_v2_addendum57.json',
             ROOT / 'config/bay_reserved_protocol_v2_addendum57_amendment1.json', ROOT / 'data/logbook.txt',
             ROOT / 'data/raw/T2-Meteo.txt', ROOT / 'reports/bay_twin/sensor_geometry.csv',
             ROOT / 'data/processed/ctd_all_casts.csv.gz']
    files += [ROOT / 'data/raw' / f'{p}.txt' for p in PROBES]
    files += sorted((ROOT / 'data/external/Rivers').glob('arpa_magra_ameglia_portata_*.csv'))
    files += [p for p in (ROOT / 'src').rglob('*.py') if '__pycache__' not in p.parts]
    with tarfile.open(PACK, 'w:gz') as tar:
        for f in dict.fromkeys(files):
            if f.name.startswith('.env') or 'deploy.toml' in f.name:
                raise SystemExit(f'refusing to pack {f}')
            tar.add(f, arcname=str(f.relative_to(ROOT)).replace('\\', '/'))
    print(f'packed {len(set(files))} files into {PACK} ({PACK.stat().st_size / 1e6:.1f} MB)')


def main(action, workers=None):
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
            n = workers or 20
            # One LightGBM thread per worker, set explicitly (OMP_NUM_THREADS does not bound LightGBM);
            # setsid -f forks, so the SSH channel is released at once.
            cmd = (f'env OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 V2_THREADS=1 '
                   f'PYTHONPATH={run} {py} -u -m src.experiments.v2_fouling_jump {flag} --workers {n} '
                   f'> logs/{action}.log 2>&1')
            sh(f'cd {run} && setsid -f nohup sh -c "{cmd}" < /dev/null > /dev/null 2>&1')
            print(f'launched {action} with {n} workers')
        elif action == 'status':
            sh(f'cd {run} && for f in logs/*.log; do echo "== $f"; grep -v "^  \\[" $f | tail -n 4; done; '
               'ps -eo pid,etime,nlwp,pcpu,args | grep "[v]2_fouling_jump" | wc -l; uptime')
        elif action == 'fetch':
            sftp = client.open_sftp()
            try:
                remote = f'{run}/reports/bay_twin/runs'
                for name in ('v2_fouling_jump.json', 'v2_fouling_jump_smoke.json'):
                    try:
                        sftp.get(f'{remote}/{name}', str(RESULTS / name))
                        print('got', name)
                    except FileNotFoundError:
                        pass
                folder = RESULTS / 'v2_fouling_jump'
                (folder / 'studies').mkdir(parents=True, exist_ok=True)

                def listing(path):
                    try:
                        return sftp.listdir(path)
                    except FileNotFoundError:
                        return []
                for f in listing(f'{remote}/v2_fouling_jump'):
                    if f.endswith('.csv.gz'):
                        sftp.get(f'{remote}/v2_fouling_jump/{f}', str(folder / f))
                for f in listing(f'{remote}/v2_fouling_jump/studies'):
                    if f.endswith('_selection.json'):
                        sftp.get(f'{remote}/v2_fouling_jump/studies/{f}', str(folder / 'studies' / f))
                for f in listing(f'{run}/logs'):
                    sftp.get(f'{run}/logs/{f}', str(folder / f))
            finally:
                sftp.close()
    finally:
        client.close()


if __name__ == '__main__':
    main(sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else None)
