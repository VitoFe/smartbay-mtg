"""Addendum 73 part A: the older cast archive, without moored context, against the model on Field-57.

The development folds' comparators use the casts of July 2024 onwards. This scores two parameter-free
estimators built from every quality-controlled cast of the archive since June 2021 at the stations
other than the held one, dated before the fold's scoring half-year, on exactly the Field-57 rows of the
per-fold released recipe:

  archive_harmonic  an annual harmonic per 1 m depth bin (v2_field_reserved2026.seasonal, the
                    construction of the 2026 comparator)
  archive_table     the month by 1 m depth-bin mean, with the depth-bin mean and then the overall median
                    where a cell or bin is empty (v2_field_naive.oracle_frame with a temporal cut)

and sets them against the model and the oracle month-by-depth table (other stations' 2024-2025 casts,
no temporal cut). Oxygen is scored in mg/l, each archive prediction converted with its row's own factor
(the C1 rerun's truth over the record's truth), as the truth was.

Usage: python -m src.experiments.v2_field_archive
"""
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.v2_field_contrasts import contrast, nested
from src.experiments.v2_field_reserved2026 import TARGETS, profiles_window, seasonal

ROOT = Path(__file__).resolve().parents[2]
RUNS = ROOT / 'reports/bay_twin/runs'
OUT = RUNS / 'v2_field_archive'
DECLARATION = 'config/bay_reserved_protocol_v2_addendum73.json'
MODEL_TS = RUNS / 'v2_perfold/field/perfold/A1_both_direct/selected_predictions.csv.gz'
MODEL_O = RUNS / 'v2_oxygen_mgl/C1/v2_perfold/field/perfold/A1_both_direct/selected_predictions.csv.gz'
ORACLE = RUNS / 'v2_field_naive/perfold/pred_oracle.csv.gz'
NAIVE_SUMMARY = RUNS / 'v2_field_naive/perfold/field_naive_summary.json'
#: the first day of each fold's scoring half-year; the archive estimators read only casts before it
CUTS = {0: pd.Timestamp('2025-01-01', tz='UTC'), 1: pd.Timestamp('2025-07-01', tz='UTC')}
ARCHIVE_START = pd.Timestamp('2021-06-01', tz='UTC')
MOORED_START = pd.Timestamp('2024-07-24', tz='UTC')
KEY = ['fold', 'station', 'cast_id', 'depth_m']
CONTRASTS = (('model', 'archive_harmonic'), ('model', 'archive_table'), ('archive_harmonic', 'oracle'),
             ('archive_table', 'oracle'))


def table(train_rows, query):
    """Month by 1 m depth-bin mean of the training casts, as v2_field_naive.oracle_frame."""
    month = train_rows.t_utc.dt.month.to_numpy()
    depth = np.floor(train_rows.depth_m.to_numpy()).astype(int)
    train = pd.DataFrame({'month': month, 'depth': depth}).join(train_rows[list(TARGETS)].reset_index(drop=True))
    cell = train.groupby(['month', 'depth'])[list(TARGETS)].mean()
    by_depth = train.groupby('depth')[list(TARGETS)].mean()
    qm, qd = query.t_utc.dt.month.to_numpy(), np.floor(query.depth_m.to_numpy()).astype(int)
    pred = cell.reindex(pd.MultiIndex.from_arrays([qm, qd])).to_numpy()
    pred = np.where(np.isfinite(pred), pred, by_depth.reindex(qd).to_numpy())
    return np.where(np.isfinite(pred), pred, train[list(TARGETS)].median().to_numpy()[None, :])


def frames():
    m = pd.read_csv(MODEL_TS).sort_values(KEY).reset_index(drop=True)
    o = pd.read_csv(MODEL_O).sort_values(KEY).reset_index(drop=True)
    r = pd.read_csv(ORACLE).sort_values(KEY).reset_index(drop=True)
    for name, f in (('C1', o), ('oracle', r)):
        if not f[KEY].equals(m[KEY]):
            raise SystemExit(f'{name} does not score the rows of the model')
    m['t_utc'] = pd.to_datetime(m.t_utc, utc=True)
    factor = (o.do_umol_l / m.do_umol_l).to_numpy(float)          # mg/l per record unit, row by row
    return m, o, r, factor


def report():
    m, o, r, factor = frames()
    archive = profiles_window(ARCHIVE_START, max(CUTS.values()))
    preds = {'archive_harmonic': np.full((len(m), 3), np.nan), 'archive_table': np.full((len(m), 3), np.nan)}
    used = {}
    for (fold, station), q in m.groupby(['fold', 'station']):
        train = archive[(archive.station != station) & (archive.t_utc < CUTS[fold])]
        rows = q.index.to_numpy()
        preds['archive_harmonic'][rows] = seasonal(train, q)
        preds['archive_table'][rows] = table(train, q)
        used[f'{fold}|{station}'] = {'casts': int(train.cast_id.nunique()),
                                     'casts_before_moorings': int(train[train.t_utc < MOORED_START].cast_id.nunique()),
                                     'stations': sorted(train.station.unique().tolist())}
    for p in preds.values():
        p[:, 2] *= factor
    truth = np.column_stack([m.temp_c, m.salinity_psu, o.do_umol_l]).astype(float)
    arms = {'model': np.column_stack([m.pred_temp_c, m.pred_salinity_psu, o.pred_do_umol_l]).astype(float),
            'oracle': np.column_stack([r.pred_temp_c, r.pred_salinity_psu, r.pred_do_umol_l * factor]).astype(float),
            **preds}
    keys = m[KEY].assign(day=m.t_utc.dt.strftime('%Y-%m-%d').to_numpy())
    mae = {a: {t: round(nested(keys, np.abs(truth[:, j] - p[:, j])), 6) for j, t in enumerate(TARGETS)}
           for a, p in arms.items()}
    published = json.loads(NAIVE_SUMMARY.read_text())['nested_mae']['model']
    gate = {t: [mae['model'][t], published[t]] for t in ('temp_c', 'salinity_psu')}
    passed = all(abs(a - b) <= 1e-4 for a, b in gate.values())
    if not passed:
        raise SystemExit(f'gate GA failed: {gate}')
    contrasts = {f'{x} - {y}': {t: contrast(keys, truth, arms, x, y, t, 'day', f'archive|{x}|{y}') for t in TARGETS}
                 for x, y in CONTRASTS}
    rep = {'declared_in': DECLARATION, 'part': 'A', 'rows': int(len(m)), 'casts': int(m.cast_id.nunique()),
           'dates': int(keys.day.nunique()), 'cuts': {str(k): str(v.date()) for k, v in CUTS.items()},
           'archive_start': str(ARCHIVE_START.date()), 'oxygen_unit': 'mg/l', 'gate_GA': {'values': gate, 'passed': passed},
           'archive_casts_per_checkpoint': used, 'nested_mae': mae, 'contrasts': contrasts}
    floor = 0.029
    pa1 = all(contrasts[f'model - {a}']['temp_c']['point'] < 0 and contrasts[f'model - {a}']['temp_c']['excludes_zero']
              for a in ('archive_harmonic', 'archive_table'))
    pa2 = not any(contrasts[f'model - {a}']['do_umol_l']['excludes_zero'] for a in ('archive_harmonic', 'archive_table'))
    pa3 = all(-contrasts[f'model - {a}']['temp_c']['point'] >= floor for a in ('archive_harmonic', 'archive_table'))
    rep['predictions'] = {'PA1': pa1, 'PA2': pa2, 'PA3': pa3}
    OUT.mkdir(parents=True, exist_ok=True)
    out = m[KEY + ['t_utc']].copy()
    for a in preds:
        for j, t in enumerate(TARGETS):
            out[f'{a}|{t}'] = preds[a][:, j]
    path = OUT / 'archive_predictions.csv.gz'
    out.to_csv(path, index=False, lineterminator='\n', compression={'method': 'gzip', 'mtime': 0})
    rep['prediction_sha256'] = hashlib.sha256(path.read_bytes()).hexdigest()
    (OUT / 'report.json').write_bytes(json.dumps(rep, indent=1).encode('utf-8'))
    for a, v in mae.items():
        print(f'{a:17s}', '  '.join(f'{t} {x:.4f}' for t, x in v.items()))
    for k, v in contrasts.items():
        print(k, {t: (c['point'], c['ci'], '*' if c['excludes_zero'] else '') for t, c in v.items()})
    print('predictions', rep['predictions'])


if __name__ == '__main__':
    report()
