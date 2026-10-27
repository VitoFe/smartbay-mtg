"""Addendum 59, part B: the ISPRA station's hourly binning, its gate, and the per-fold reruns.

  python scripts/v2_ispra_rebin_remote.py gate
  python scripts/v2_ispra_rebin_remote.py pack | upload | smoke | run | status | fetch

`gate` runs locally and writes runs/v2_ispra_rebin/gate.json:
  * the station's timestamp convention (repeats, steps across every daylight-saving change in the
    record, and where the radiation-weighted mean hour falls against the sun's transit);
  * how far past issuance each hourly binning reaches, on the development issuance grid;
  * the declared gate: with the hourly means labelled at their start, as the addendum-7 runs were
    made, the published configuration of each variable refitted reproduces the published predictions
    to 1e-9 (sea level at 6 h on each variable), and the same check on all twelve published cells;
  * the same published configurations refitted with the means labelled at the end of their hour,
    the binning's own effect before any reselection.

The twelve published studies are rerun as published, through `scripts/v2_campaign.py --campaign ispra`.
The other actions then run the per-fold studies of the station family (src/experiments/v2_perfold_rest.py,
all three designs) on the compute host against those reruns. The pack holds the thesis's code, the inputs
of the forecasting builders, the station record and the rerun's selections and predictions; never
credentials (the connection reads them itself and never prints them). `fetch` writes the twelve cells
into runs/v2_perfold_rest/forecast/<design>/ and the logs into runs/v2_ispra_rebin/perfold_logs/.
"""
import hashlib
import json
import os
import sys
import tarfile
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'scripts'))
RUNS = ROOT / 'reports/bay_twin/runs'
OUT = RUNS / 'v2_ispra_rebin'
PUBLISHED_START = '20260920T190718Z_v2_ispra'    # made with the means labelled at their start
REMOTE = 'review_20260919/ispra_rebin_perfold'
PACK = ROOT / 'scratch/ispra_rebin_pack.tar.gz'   # holds raw data: git-ignored
KEY = ['fold', 'sensor', 'time']
CELLS = [(v, a, h) for v in ('temperature', 'oxygen') for a in ('sealevel_tree', 'extboundary_tree')
         for h in (6, 24, 48)]
DECLARED_GATE = [('temperature', 'sealevel_tree', 6), ('oxygen', 'sealevel_tree', 6)]
LONGITUDE_E = 9.83   # the Gulf of La Spezia, for the sun's transit only


def _dump(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes((json.dumps(obj, indent=2, allow_nan=False) + '\n').encode())


def _macro(d, col='prediction'):
    e = (d.truth - d[col]).abs()
    return float(d.assign(e=e).groupby(['fold', 'sensor']).e.mean().groupby('fold').mean().mean())


# ---------------------------------------------------------------- the station's timestamps

def raw_record():
    from src.data.bay_ispra import SOURCE
    frame = pd.read_csv(ROOT / SOURCE, sep=';', decimal=',', encoding='utf-8', low_memory=False)
    stamp = pd.to_datetime(frame.DATA.astype(str) + ' ' + frame.ORA.astype(str), errors='coerce', utc=True)
    return frame.assign(t=stamp)


def transit_hour_utc(day_of_year):
    """The sun's transit at LONGITUDE_E in UTC hours (equation of time after Spencer, 1971)."""
    g = 2 * np.pi * (np.asarray(day_of_year) - 1) / 365
    eot = 229.18 * (0.000075 + 0.001868 * np.cos(g) - 0.032077 * np.sin(g) - 0.014615 * np.cos(2 * g)
                    - 0.040849 * np.sin(2 * g))
    return 12 - LONGITUDE_E / 15 - eot / 60


def timestamps(frame):
    t = frame.t
    steps = t.diff().dropna().dt.total_seconds().div(60).round().astype(int).value_counts()
    dst = []
    for year in range(t.min().year, t.max().year + 1):
        for month in (3, 10):
            day = pd.Timestamp(year=year, month=month, day=31)
            while day.dayofweek != 6:
                day -= pd.Timedelta(days=1)
            lo, hi = pd.Timestamp(day, tz='UTC'), pd.Timestamp(day, tz='UTC') + pd.Timedelta(hours=4)
            window = t[(t >= lo) & (t <= hi)]
            if len(window) == 0:
                continue
            gaps = window.diff().dropna().dt.total_seconds().div(60).round().astype(int)
            dst.append({'change': str(day.date()), 'stamps_00_to_04_utc': int(len(window)),
                        'steps_minutes': sorted(set(gaps.tolist())), 'repeats': int(window.duplicated().sum())})
    rad = pd.to_numeric(frame['RADIAZIONE_SOLARE_MEDIA_(30_MIN)'], errors='coerce')
    r = pd.DataFrame({'t': t, 'r': rad}).dropna()
    r = r[r.r > 0]
    hour = r.t.dt.hour + r.t.dt.minute / 60
    by_month = {}
    for m, g in r.groupby(r.t.dt.month):
        h = hour[g.index]
        centroid = float((h * g.r).sum() / g.r.sum())
        transit = float(np.average(transit_hour_utc(g.t.dt.dayofyear.to_numpy()), weights=g.r.to_numpy()))
        by_month[int(m)] = {'readings': int(len(g)), 'radiation_weighted_mean_hour_utc': centroid,
                            'sun_transit_hour_utc': transit, 'difference_hours': centroid - transit}
    return {'rows': int(len(t)), 'unparsed': int(t.isna().sum()), 'repeated_stamps': int(t.duplicated().sum()),
            'monotonic': bool(t.is_monotonic_increasing),
            'steps_minutes': {str(k): int(v) for k, v in steps.sort_index().items()},
            'daylight_saving_changes': dst,
            'radiation_by_month': by_month,
            'radiation_note': 'RADIAZIONE_SOLARE_MEDIA_(30_MIN) is a 30 min mean, so its stamp may sit up to '
                              '30 min after the interval it covers; stamps in local time with daylight saving '
                              'would shift the mean hour by one hour between March and April and between '
                              'October and November, and stamps in UTC+1 would put it about an hour later '
                              'against the transit than UTC does'}


def reach(frame, times):
    """Minutes from issuance q to the readings in the hourly mean reindexed at q, per binning."""
    from src.data.bay_ispra import BINNINGS
    frame = frame.dropna(subset=['t']).set_index('t').sort_index()
    frame = frame[frame.index < pd.Timestamp('2026-01-01', tz='UTC')]
    out = {}
    for name, rule in BINNINGS.items():
        out[name] = {}
        for col, key in (('LIVELLO_IDROMETRICO', 'level'), ('TEMPERATURA_ACQUA', 'water')):
            v = pd.to_numeric(frame[col], errors='coerce').dropna()
            # the bin label each reading falls under
            label = v.index.floor('h') + (pd.Timedelta(hours=1) if rule['label'] == 'right' else pd.Timedelta(0))
            offset = pd.Series((v.index - label).total_seconds() / 60, index=v.index)
            at = pd.DataFrame({'label': label, 'offset': offset.to_numpy()})
            at = at[at.label.isin(times)]
            per = at.groupby('label').offset.agg(['min', 'max'])
            out[name][key] = {'issuances_with_a_reading': int(len(per)), 'issuances': int(len(times)),
                              'latest_reading_minutes_after_issuance_max': float(per['max'].max()),
                              'earliest_reading_minutes_after_issuance_min': float(per['min'].min()),
                              'issuances_reading_at_or_after_issuance': int((per['max'] >= 0).sum())}
    return out


# ---------------------------------------------------------------- the declared gate

def refit(variable, arm, horizon, run):
    from src.experiments.v2_horizon import build, folds, climatology, tree_fit_score
    from src.experiments.v2_ispra import design_for
    data = build(variable, horizon)
    outer = folds(data, horizon)
    folder = RUNS / run / 'results' / f'v2_ispra_{variable}_{arm}_h{horizon}'
    sel = json.loads((folder / f'{variable}_{arm}_h{horizon}_selection.json').read_text())
    params = {**sel['params'], 'subsample_freq': 1}
    d = pd.concat([tree_fit_score(data, design_for(data, f, arm), f, params, climatology(data, f))[0]
                   for f in outer], ignore_index=True)
    published = pd.read_csv(folder / 'selected_predictions.csv.gz')
    d['time'] = d.time.astype(str)
    published['time'] = published.time.astype(str)
    j = published.merge(d, on=KEY, suffixes=('_pub', ''), validate='one_to_one')
    same = len(j) == len(published) == len(d)
    other = {c: float(np.abs(j[c] - j[c + '_pub']).max()) for c in ('truth', 'persistence')}
    return {'rows_identical': bool(same), 'endpoints': int(len(d)),
            'max_abs_prediction_diff': float(np.abs(j.prediction - j.prediction_pub).max()),
            'other_columns_max_abs_diff': other,
            'mae_published': _macro(published), 'mae_refit': _macro(d),
            'fold_mae_refit': {str(k): _macro(g) for k, g in d.groupby('fold')},
            'persistence_mae': _macro(published, 'persistence')}


def gate():
    from src.experiments.v2_horizon import build
    frame = raw_record()
    times = build('temperature', 6)['times']
    report = {'declared_in': 'config/bay_reserved_protocol_v2_addendum59.json', 'part': 'B',
              'published_run': PUBLISHED_START, 'timestamps': timestamps(frame),
              'reach_past_issuance': reach(frame, times), 'cells': {}}
    previous = os.environ.get('V2_ISPRA_BINNING')
    try:
        for binning in ('start', 'end'):
            os.environ['V2_ISPRA_BINNING'] = binning
            for v, a, h in CELLS:
                r = refit(v, a, h, PUBLISHED_START)
                report['cells'].setdefault(f'{v}|{a}|{h}', {})[binning] = r
                print(f'{binning:5s} {v:11s} {a:16s} h{h:<2d} max|d| {r["max_abs_prediction_diff"]:.3e}  '
                      f'MAE {r["mae_published"]:.6f} -> {r["mae_refit"]:.6f}', flush=True)
    finally:
        if previous is None:
            os.environ.pop('V2_ISPRA_BINNING', None)
        else:
            os.environ['V2_ISPRA_BINNING'] = previous

    def passes(cell):
        r = report['cells'][cell]['start']
        return bool(r['rows_identical'] and r['max_abs_prediction_diff'] <= 1e-9
                    and max(r['other_columns_max_abs_diff'].values()) <= 1e-9)
    report['gate_declared'] = {'rule': 'with the published binning, one refit per variable at its published '
                                       'configuration reproduces the published predictions to 1e-9',
                               'cells': [f'{v}|{a}|{h}' for v, a, h in DECLARED_GATE],
                               'pass': all(passes(f'{v}|{a}|{h}') for v, a, h in DECLARED_GATE)}
    report['gate_all_cells'] = {'pass': all(passes(c) for c in report['cells']),
                                'max_abs_prediction_diff': max(c['start']['max_abs_prediction_diff']
                                                               for c in report['cells'].values())}
    report['fixed_configuration_effect'] = {
        c: {'mae_start': r['start']['mae_refit'], 'mae_end': r['end']['mae_refit'],
            'end_minus_start': r['end']['mae_refit'] - r['start']['mae_refit'],
            'max_abs_prediction_change': r['end']['max_abs_prediction_diff']}
        for c, r in report['cells'].items()}
    _dump(OUT / 'gate.json', report)
    print('declared gate', report['gate_declared']['pass'], '| all cells', report['gate_all_cells'])
    return report


# ---------------------------------------------------------------- the per-fold studies on the host

def rerun_id():
    from src.experiments.v2_perfold_rest import PUBLISHED_RUN
    run = PUBLISHED_RUN['ispra']
    if run == PUBLISHED_START:
        raise SystemExit('src/experiments/v2_perfold_rest.py still names the start-binned run')
    return run


def pack():
    from v2_campaign import plan
    run = rerun_id()
    files, _ = plan('ispra')
    files += [ROOT / 'config/bay_reserved_protocol_v2_addendum48.json',
              ROOT / 'config/bay_reserved_protocol_v2_addendum59.json']
    files += [p for p in (ROOT / 'src').rglob('*.py') if '__pycache__' not in p.parts]
    for v, a, h in CELLS:
        folder = RUNS / run / 'results' / f'v2_ispra_{v}_{a}_h{h}'
        contract = json.loads((folder / 'contract.json').read_text())
        if contract.get('hourly_binning') != 'end' or contract.get('smoke_only'):
            raise SystemExit(f'{folder} is not a full run with the end binning')
        files += [folder / n for n in ('contract.json', 'report.json', 'selected_predictions.csv.gz',
                                       f'{v}_{a}_h{h}_selection.json')]
    PACK.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(PACK, 'w:gz') as tar:
        for f in dict.fromkeys(files):
            if f.name.startswith('.env') or 'deploy.toml' in f.name:
                raise SystemExit(f'refusing to pack {f}')
            tar.add(f, arcname=f.relative_to(ROOT).as_posix())
    print(f'packed {len(set(files))} files for {run} into {PACK} ({PACK.stat().st_size / 1e6:.1f} MB)')


def remote(action):
    from scientific_campaign import connect, execute
    client, base = connect()
    run, py = f'{base}/{REMOTE}', f'{base}/.venv/bin/python'

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
            sh(f'cd {run} && tar -xzf pack.tar.gz && ls && sha256sum pack.tar.gz')
            print('local', hashlib.sha256(PACK.read_bytes()).hexdigest())
        elif action in ('smoke', 'run'):
            jobs = ([('smoke', 'temperature', 'sealevel_tree', 6, '--design published')] if action == 'smoke' else
                    [(f'{v}_{a}_h{h}', v, a, h, '') for v, a, h in CELLS])
            for name, v, a, h, extra in jobs:
                # Two threads per job, set explicitly: v2_forecast.make_tree passes n_jobs=V2_THREADS to
                # LightGBM, which ignores OMP_NUM_THREADS. setsid -f releases the SSH channel at once.
                cmd = (f'env OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 V2_THREADS=2 '
                       f'CUDA_VISIBLE_DEVICES= PYTHONPATH={run} {py} -u -m src.experiments.v2_perfold_rest forecast '
                       f'--family ispra --variable {v} --arm {a} --horizon {h} {extra} > logs/{name}.log 2>&1')
                sh(f'cd {run} && setsid -f nohup sh -c "{cmd}" < /dev/null > /dev/null 2>&1')
            print(f'launched {len(jobs)} {action} job(s)')
        elif action == 'status':
            sh(f'cd {run} && for f in logs/*.log; do echo "== $f"; grep -v "^  \\[" $f | tail -n 3; '
               'grep "^  \\[" $f | tail -n 1; done; '
               'ps -eo pid,etime,nlwp,pcpu,args | grep "[v]2_perfold_rest forecast --family ispra" | wc -l; uptime')
        elif action == 'fetch':
            sftp = client.open_sftp()
            try:
                got = 0
                for design in ('published', 'shared', 'perfold'):
                    for v, a, h in CELLS:
                        name = f'ispra_{v}_{a}_h{h}'
                        src = f'{run}/reports/bay_twin/runs/v2_perfold_rest/forecast/{design}/{name}'
                        dst = RUNS / 'v2_perfold_rest/forecast' / design / name
                        files = sftp.listdir(src)
                        if 'report.json' not in files:
                            raise SystemExit(f'{design}/{name} has no report yet')
                        dst.mkdir(parents=True, exist_ok=True)
                        for f in files:
                            sftp.get(f'{src}/{f}', str(dst / f))
                            got += 1
                logs = OUT / 'perfold_logs'
                logs.mkdir(parents=True, exist_ok=True)
                for f in sftp.listdir(f'{run}/logs'):
                    sftp.get(f'{run}/logs/{f}', str(logs / f))
                print('fetched', got, 'files')
            finally:
                sftp.close()
    finally:
        client.close()


def main(action):
    if action == 'gate':
        return gate()
    if action == 'pack':
        return pack()
    return remote(action)


if __name__ == '__main__':
    main(sys.argv[1])
