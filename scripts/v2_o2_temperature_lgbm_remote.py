"""Addendum 55 on the compute host: inputs, pack, upload, smoke, run, status, fetch.

  python scripts/v2_o2_temperature_lgbm_remote.py inputs | pack | upload | smoke | run | status | fetch

`inputs` copies the published selections the gates refit (the four anchored_gbm_rel oxygen runs and the
seven oxygen horizon trees) into reports/bay_twin/runs/v2_o2_temperature_lgbm/inputs/ with a manifest.
The pack holds the thesis's code, the probe geometry, the raw exports, these inputs and addendum 54's
inputs and temperature fills; never credentials. The run reads the companion precompute already on the
host (its temperature forecasts) and writes results_lgbm/.

`run` launches, detached: the temperature fill of the episodes addendum 54 did not cover, followed by
the four rel_T_fill jobs; the four gap gates and the four rel_T_obs jobs; and for each of the seven
leads the tree gate, tree_T_obs and tree_T_fc. The gap jobs use two LightGBM threads, as the published
runs did; the horizon trees are deterministic and use three.
"""
import hashlib
import json
import os
import shutil
import sys
import tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
RUNS = ROOT / 'reports/bay_twin/runs'
PRECOMPUTE = 'review_20260919/companion_precompute/cache'
#: Addendum 59: the filled temperature reads the Gaussian process, so with V2_O2T_RERUN=1 the tfill
#: task and the four rel_T_fill jobs are rerun in their own directories, on addendum 54's rerun
#: (its inputs and fills, `_gp_by_pattern`), packed where the experiment reads the published ones.
#: The gates and the observed-temperature and forecast arms read no fill and are not rerun.
RERUN = os.environ.get('V2_O2T_RERUN') == '1'
RUN = 'review_20260919/o2_temperature_lgbm' + ('_gp_by_pattern' if RERUN else '')
#: holds raw data: kept out of the repository (V2_O2T_PACK moves it elsewhere)
PACK = Path(os.environ.get('V2_O2T_PACK', ROOT / 'companion/.cache/o2_temperature_lgbm_pack.tar.gz'))
HERE = RUNS / 'v2_o2_temperature_lgbm'
INPUTS = HERE / 'inputs'
RESULTS = HERE / ('results_gp_by_pattern' if RERUN else 'results')
A54_INPUTS = RUNS / 'v2_o2_temperature' / ('inputs_gp_by_pattern' if RERUN else 'inputs')
A54_RESULTS = RUNS / 'v2_o2_temperature' / ('results_gp_by_pattern' if RERUN else 'results')
#: the published anchored_gbm_rel oxygen runs and the environment each was launched with
GAP = {('strict', 'development'): ('20260922T215308Z_v2_gaprecovery/v2_gap_recovery', {}),
       ('strict', 'reserved'): ('20260922T215413Z_v2_gaprecovery/v2_gap_recovery_reserved2026',
                                {'V2_GAP_PERIOD': 'reserved', 'V2_GAP_LENGTHS': '6,12,18,24,48,72'}),
       ('relaxed', 'development'): ('20260922T215328Z_v2_gaprecovery/v2_gap_recovery_long',
                                   {'V2_GAP_DESIGN': 'relaxed', 'V2_GAP_LENGTHS': '168,336,504,720,1440'}),
       ('relaxed', 'reserved'): ('20260922T215431Z_v2_gaprecovery/v2_gap_recovery_long_reserved2026',
                                 {'V2_GAP_DESIGN': 'relaxed', 'V2_GAP_PERIOD': 'reserved',
                                  'V2_GAP_LENGTHS': '168,336,504,720,1440'})}
#: the published oxygen horizon trees, by lead
TREE = {1: '20260920T133747Z', 3: '20260920T133747Z', 6: '20260920T083322Z', 12: '20260920T133747Z',
        24: '20260920T083322Z', 36: '20260920T133747Z', 48: '20260920T083322Z'}


def gap_dir(design, period):
    return RUNS / GAP[(design, period)][0] / 'oxygen_anchored_gbm_rel'


def tree_dir(lead):
    return RUNS / f'{TREE[lead]}_v2_horizon/results/v2_horizon_oxygen_tree_h{lead}'


def inputs():
    INPUTS.mkdir(parents=True, exist_ok=True)
    for (design, period) in GAP:
        report = json.loads((gap_dir(design, period) / 'report.json').read_text(encoding='utf-8'))
        assert (report['design'], report['period'], report['arm']) == (design, period, 'anchored_gbm_rel')
        keep = {'source': str(gap_dir(design, period).relative_to(ROOT)).replace('\\', '/'),
                'rows': report['rows'], 'episodes': report['episodes'], 'feature_names': report['feature_names'],
                'selections': report['selections']}
        (INPUTS / f'published_gap_{design}_{period}.json').write_bytes((json.dumps(keep, indent=1) + '\n').encode())
    for lead in TREE:
        src = tree_dir(lead) / f'oxygen_tree_h{lead}_selection.json'
        shutil.copyfile(src, INPUTS / f'published_tree_h{lead}.json')
    files = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(INPUTS.glob('*')) if p.name != 'manifest.json'}
    fills = {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
             for p in sorted((RUNS / 'v2_o2_temperature/results').glob('gaps_*.csv.gz'))}
    manifest = {'addendum': 55, 'files': files, 'addendum54_fills': fills}
    (INPUTS / 'manifest.json').write_bytes((json.dumps(manifest, indent=1) + '\n').encode())
    print(f'wrote {len(files)} inputs to {INPUTS.relative_to(ROOT)}')


def pack():
    import pandas as pd
    sensors = pd.read_csv(ROOT / 'reports/bay_twin/sensor_geometry.csv').sensor.tolist()
    files = [ROOT / 'reports/bay_twin/sensor_geometry.csv', ROOT / 'reports/bay_twin/exchange_geometry.json',
             ROOT / 'data/raw/T2-Meteo.txt', ROOT / 'config/tuning_protocol_v2.json',
             ROOT / 'config/bay_reserved_protocol_v2.json']
    files += [ROOT / 'data/raw' / f'{s}.txt' for s in sensors]
    files += sorted((ROOT / 'data/external/Rivers').glob('arpa_magra_ameglia_portata_*.csv'))
    files += sorted(INPUTS.glob('*'))
    files += sorted(A54_INPUTS.glob('*'))
    files += sorted(A54_RESULTS.glob('gaps_*.csv.gz'))
    for folder in (ROOT / 'src', ROOT / 'scripts', ROOT / 'companion/smartbay_companion'):
        files += [p for p in folder.rglob('*.py') if '__pycache__' not in p.parts]
    with tarfile.open(PACK, 'w:gz') as tar:
        for f in dict.fromkeys(files):
            if f.name.startswith('.env') or 'deploy.toml' in f.name:
                raise SystemExit(f'refusing to pack {f}')
            name = str(f.relative_to(ROOT)).replace('\\', '/')
            # a rerun's addendum-54 inputs and fills go where the experiment reads the published ones
            name = (name.replace('v2_o2_temperature/inputs_gp_by_pattern/', 'v2_o2_temperature/inputs/')
                    .replace('v2_o2_temperature/results_gp_by_pattern/', 'v2_o2_temperature/results/'))
            tar.add(f, arcname=name)
    print(f'packed {len(set(files))} files into {PACK} ({PACK.stat().st_size / 1e6:.1f} MB)')


def jobs(quick):
    """(name, environment, arguments, threads) for every job, in launch order."""
    out = []
    for (design, period), (_, env) in GAP.items():
        for arm in ('gate', 'rel_T_obs', 'rel_T_fill'):
            out.append((f'gap_{design}_{period}_{arm}', env,
                        f'gap --design {design} --period {period} --arm {arm}', 2))
    for lead in TREE:
        for arm in ('gate', 'tree_T_obs', 'tree_T_fc'):
            out.append((f'horizon_h{lead}_{arm}', {}, f'horizon --lead {lead} --arm {arm}', 3))
    if quick:
        out = [j for j in out if j[0] in ('gap_strict_development_gate', 'gap_strict_reserved_rel_T_obs',
                                          'gap_relaxed_development_rel_T_obs', 'gap_relaxed_reserved_rel_T_fill',
                                          'horizon_h24_gate',
                                          'horizon_h24_tree_T_fc')]
    return out


def main(action):
    if action == 'inputs':
        return inputs()
    if action == 'pack':
        return pack()
    from scientific_campaign import connect, execute
    client, base = connect()
    run, py = f'{base}/{RUN}', f'{base}/.venv/bin/python'

    def sh(cmd):
        code, out = execute(client, cmd)
        print(out.strip())
        return code

    def command(name, env, args, threads, flag):
        env = ' '.join(f'{k}={v}' for k, v in env.items())
        return (f'env CUDA_VISIBLE_DEVICES=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 '
                f'V2_THREADS={threads} {env} HF_HOME={base}/.hf PYTHONPATH={run} '
                f'SMARTBAY_FULL_CACHE={base}/{PRECOMPUTE} {py} -u scripts/experiments/v2_o2_temperature_lgbm.py '
                f'{args} {flag} > logs/{name}.log 2>&1')
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
            quick = action == 'smoke'
            flag = '--quick' if quick else ''
            todo = jobs(quick)
            fill = [] if quick else [j for j in todo if j[0].endswith('rel_T_fill')]
            rest = [] if RERUN else [j for j in todo if j not in fill]
            if RERUN:
                todo = fill
            _, listing = execute(client, f'ls {run}/logs')
            started = {f[:-4] for f in listing.split() if f.endswith('.log')}   # a relaunch skips these
            for name, env, args, threads in rest:
                if name in started:
                    continue
                # setsid -f forks, so the SSH channel is released at once
                sh(f'cd {run} && setsid -f nohup sh -c "{command(name, env, args, threads, flag)}" '
                   f'< /dev/null > /dev/null 2>&1')
            if not quick and 'tfill' not in started:
                chain = ' ; '.join(['(' + ' & '.join(command(*j, flag) for j in fill) + ' & wait)'])
                tfill = (f'env CUDA_VISIBLE_DEVICES=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 '
                         f'V2_THREADS=3 HF_HOME={base}/.hf PYTHONPATH={run} {py} -u '
                         f'scripts/experiments/v2_o2_temperature_lgbm.py tfill --workers 12 > logs/tfill.log 2>&1')
                sh(f"cd {run} && setsid -f nohup sh -c '{tfill} && {chain}' < /dev/null > /dev/null 2>&1")
            print(f'launched {len(todo)} jobs' + ('' if quick else ' (rel_T_fill after tfill)'))
        elif action == 'status':
            sh(f'cd {run} && for f in logs/*.log; do echo "== $f: $(grep -v "Loading weights\\|HF Hub\\|gpr: fitted" $f | tail -n 1)"; done; '
               'echo; ls results_lgbm 2>/dev/null | wc -l; ls results_lgbm/*/report.json 2>/dev/null | wc -l; '
               'ps -eo pid,etime,nlwp,pcpu,args | grep "[v]2_o2_temperature_lgbm.py" | wc -l; uptime; '
               'nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv')
        elif action == 'fetch':
            RESULTS.mkdir(parents=True, exist_ok=True)
            sftp = client.open_sftp()
            try:
                for item in sftp.listdir_attr(f'{run}/results_lgbm'):
                    name = item.filename
                    remote = f'{run}/results_lgbm/{name}'
                    if name.endswith(('.csv.gz', '.json')):
                        sftp.get(remote, str(RESULTS / name))
                        continue
                    (RESULTS / name).mkdir(exist_ok=True)
                    for f in sftp.listdir(remote):
                        if f.endswith(('.json', '.csv.gz')):
                            sftp.get(f'{remote}/{f}', str(RESULTS / name / f))
                    print('got', name)
                (RESULTS / 'logs').mkdir(exist_ok=True)
                for f in sftp.listdir(f'{run}/logs'):
                    sftp.get(f'{run}/logs/{f}', str(RESULTS / 'logs' / f))
            finally:
                sftp.close()
    finally:
        client.close()


if __name__ == '__main__':
    main(sys.argv[1])
