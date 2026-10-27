"""Addendum 82, parts D and H: the local reproduction gates, run before anything new is read.

  D-G1  the new held-site module, at each published (fold, site) selection of the nested
        temperature run (the frozen repository's selection records), reproduces held_station_nested
        on all 3,962 rows to 1e-12; at the shared configuration it reproduces the all_sensor and
        held_station columns of both published frames to 1e-12
  D-G3  the inner-window rule reproduces v2_horizon.inner_fold in every cell where another site is
        eligible, and moves only oxygen fold 1 with Baia Santa Teresa held, to the declared window
  D-G5  the per-fold all-site weather tree shares every key and the truth with the held-site frames
        and reproduces the per-fold MAEs of runs/v2_perfold.json
  D-G6  build_shifted(v, 6, 0) equals v2_horizon.build (addendum 50's gate, read)
  H-G1  the published held-probe seabed rows are reproduced (v2_held_sensor's fit at the shared
        configuration, in memory, so nothing is written over its files), and the new module's seabed
        design equals the published one bit for bit in every fold
  H-G2  the new module's plain and weather designs equal v2_perfold's in every outer and inner fold
  H-G3  the per-fold seabed, weather and plain all-site predictions share every key and the truth
        with the held-site frames and reproduce the per-fold MAEs of v2_perfold(_rest).json

Writes reports/bay_twin/runs/v2_addendum82/gates_local.json. Usage (V2_THREADS=2):
  python -m scripts.v2_add82_DH_gates
"""
import os
os.environ.setdefault('V2_THREADS', '2')

import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.experiments import v2_held_site_nested as M
from src.experiments.v2_closed_loop import selected
from src.experiments.v2_horizon import ROOT, folds, inner_fold

RUNS = ROOT / 'reports/bay_twin/runs'
FROZEN = Path(r'C:\Users\vferri\git\inrim2026\SmartBayDO\reports\bay_twin\runs\v2_held_station')
OUT = RUNS / 'v2_addendum82/gates_local.json'
KEY = ['fold', 'sensor', 'time']
TOL = 1e-12
DECLARED_WINDOW = {'start': '2025-08-01 06:00:00+00:00', 'end': '2025-09-15 06:00:00+00:00',
                   'scored': 10, 'training': 456}


def bits(a, b):
    """Bit-for-bit identity (NaN in the same places compares equal)."""
    a, b = np.ascontiguousarray(a), np.ascontiguousarray(b)
    return bool(a.shape == b.shape and a.dtype == b.dtype and a.tobytes() == b.tobytes())


def macro(frame, col):
    return float(frame.assign(e=(frame.truth - frame[col]).abs()).groupby(['fold', 'sensor']).e.mean()
                 .groupby('fold').mean().mean())


def frames():
    t = pd.read_csv(RUNS / 'v2_held_station/held_station_temperature_h6_nested.csv.gz')
    o = pd.read_csv(RUNS / 'v2_held_station/held_station_oxygen_h6.csv.gz')
    return {'temperature': t, 'oxygen': o}


def max_diff(a, b, col_a, col_b):
    m = a.merge(b, on=KEY, how='outer', validate='one_to_one', indicator=True)
    unmatched = int((m._merge != 'both').sum())
    return float((m[col_a] - m[col_b]).abs().max()), unmatched, int(len(m))


def g3():
    w = M.windows()
    cells = {}
    ok = True
    for key, v in w.items():
        variable, fold, site = key.split('|')
        moved = variable == 'oxygen' and fold == '1' and site == 'Baia_S.Teresa'
        if moved:
            a = v['audit'] or {}
            mitili = 'TMitiliE_05m-CTD(calibrato)'
            good = (not v['published_window_selectable'] and v['moved_hours'] is not None and v['moved_hours'] > 0
                    and a.get('inner_selection_start') == DECLARED_WINDOW['start']
                    and a.get('inner_selection_end') == DECLARED_WINDOW['end']
                    and a.get('other_scored_issuances', {}).get(mitili) == DECLARED_WINDOW['scored']
                    and a.get('other_training_issuances', {}).get(mitili) == DECLARED_WINDOW['training'])
        else:
            good = v['published_window_selectable'] and v['identical_to_published_inner_fold']
        cells[key] = {**v, 'pass': bool(good)}
        ok &= bool(good)
    counts = {'temperature_cells': sum(k.startswith('temperature') for k in w),
              'oxygen_cells': sum(k.startswith('oxygen') for k in w)}
    return {'pass': bool(ok and counts == {'temperature_cells': 14, 'oxygen_cells': 5}), **counts, 'cells': cells}


def g1(published):
    out = {'nested_temperature': {}, 'shared': {}}
    nested = []
    for f in sorted(FROZEN.glob('held_station_temperature_h6_nested_*_selection.json')):
        rec = json.loads(f.read_text())
        params = {int(k): v['params'] for k, v in rec['selections'].items() if 'params' in v}
        nested.append(M.refit('temperature', 'weather_tree', rec['site'], params, column='refit'))
        out['nested_temperature'][rec['site']] = {k: rec['selections'][str(k)]['study'] for k in params}
    nested = pd.concat(nested, ignore_index=True)
    d, unmatched, n = max_diff(published['temperature'], nested[KEY + ['refit']], 'held_station_nested', 'refit')
    out['nested_temperature_check'] = {'rows': n, 'unmatched': unmatched, 'max_abs_diff': d,
                                       'pass': bool(unmatched == 0 and n == 3962 and d <= TOL)}
    for variable, frame in published.items():
        params = selected(variable, 'weather_tree', 6)
        data = M.build_data(variable, 0)
        sites, distance = M.geometry(data['names'])
        met = M.weather(data['times']).to_numpy()
        held_rows, all_rows = [], []
        for fold in folds(data, 6):
            full = M.fit_predict(data, M.design(data, fold, 'weather_tree', None, None, met), fold, params, exclude=None)
            mask = data['valid'][fold['va']] & fold['eligible'][None, :]
            for site in pd.unique(sites):
                held = np.flatnonzero((sites == site) & fold['eligible'])
                if not len(held):
                    continue
                pred = M.fit_predict(data, M.design(data, fold, 'weather_tree', held, distance, met), fold, params,
                                     exclude=held)
                held_rows += M.held_rows(data, fold, site, held, pred, 'refit_held')
            for j, s in np.argwhere(mask):
                i = fold['va'][j]
                all_rows.append({'fold': fold['number'], 'sensor': data['names'][s], 'time': str(data['times'][i]),
                                 'refit_all': float(full[j, s] * data['scale'] + data['center'])})
        held_df, all_df = pd.DataFrame(held_rows), pd.DataFrame(all_rows)
        dh, uh, nh = max_diff(frame, held_df[KEY + ['refit_held']], 'held_station', 'refit_held')
        da, ua, na = max_diff(frame, all_df, 'all_sensor', 'refit_all')
        out['shared'][variable] = {'rows': nh, 'held_station_max_abs_diff': dh, 'all_sensor_max_abs_diff': da,
                                   'unmatched': uh + ua,
                                   'pass': bool(uh + ua == 0 and dh <= TOL and da <= TOL)}
    out['pass'] = bool(out['nested_temperature_check']['pass'] and all(v['pass'] for v in out['shared'].values()))
    return out


def references(published):
    perfold = json.loads((RUNS / 'v2_perfold.json').read_text())['forecast']['cells']
    rest = json.loads((RUNS / 'v2_perfold_rest.json').read_text())['forecast']['cells']
    out = {}
    for variable, frame in published.items():
        for arm, folder, report in [
                ('weather_tree', RUNS / f'v2_perfold/forecast/perfold/{variable}_weather_tree_h6', perfold),
                ('tree', RUNS / f'v2_perfold/forecast/perfold/{variable}_tree_h6', perfold),
                ('bathy_tree', RUNS / f'v2_perfold_rest/forecast/perfold/bathy_{variable}_bathy_tree_h6', rest)]:
            d = pd.read_csv(folder / 'selected_predictions.csv.gz')
            m = frame[KEY + ['truth']].merge(d[KEY + ['truth', 'prediction']], on=KEY, how='outer',
                                             validate='one_to_one', indicator=True, suffixes=('', '_ref'))
            unmatched = int((m._merge != 'both').sum())
            truth = float((m.truth - m.truth_ref).abs().max())
            want = report[f'{variable}|6']['arms'][arm]['perfold']['mae']
            got = macro(d, 'prediction')
            out[f'{variable}|{arm}'] = {'source': folder.relative_to(ROOT).as_posix(), 'rows': int(len(m)),
                                        'unmatched': unmatched, 'truth_max_abs_diff': truth,
                                        'mae': got, 'published_mae': want, 'abs_diff': abs(got - want),
                                        'pass': bool(unmatched == 0 and truth == 0 and abs(got - want) <= 1e-12)}
    return out


def designs():
    """H-G1's design identity and H-G2: the new module's designs against the published producers'."""
    from src.experiments.v2_bathy import design_for
    from src.experiments.v2_perfold import forecast_setup
    out = {}
    for variable in ('temperature', 'oxygen'):
        for arm in ('tree', 'weather_tree'):
            data, outer, inner, _, cache = forecast_setup(variable, arm, 6)
            met = M.weather(data['times']).to_numpy()
            same = all(bits(M.design(data, f, arm, None, None, met), cache[id(f)]) for f in outer + inner)
            out[f'{variable}|{arm}|v2_perfold'] = bool(same)
        data = M.build_data(variable, 0)
        outer = folds(data, 6)
        inner = [inner_fold(data, f, 6) for f in outer]
        met = M.weather(data['times']).to_numpy()
        out[f'{variable}|bathy_tree|v2_bathy'] = bool(all(
            bits(M.design(data, f, 'bathy_tree', None, None, met), design_for(data, f)) for f in outer + inner))
    return {'pass': bool(all(out.values())), 'cells': out}


def h_g1():
    """v2_held_sensor --bathy, refitted in memory at its shared configuration."""
    from src.experiments.bay_dynamics_tree import features
    from src.experiments.v2_bathy import bathymetric_block
    from src.experiments.v2_held_sensor import fit_predict
    out = {}
    for variable in ('temperature', 'oxygen'):
        published = pd.read_csv(RUNS / f'v2_held_sensor/held_sensor_{variable}_h6_bathy.csv.gz')
        data = M.build_data(variable, 0)
        met = M.weather(data['times']).to_numpy()
        params = selected(variable, 'weather_tree', 6)
        rows = []
        for fold in folds(data, 6):
            base, _ = features(data, {'fallback': fold['fallback']}, 'forecast')
            blocks = [base, np.broadcast_to(met[:, None], (len(data['times']), len(data['names']), met.shape[1])),
                      bathymetric_block(data)]
            design = np.concatenate(blocks, axis=-1)
            same = bits(design, M.design(data, fold, 'bathy_tree', None, None, met))
            full = fit_predict(data, design, fold, params, exclude=None)
            mask = data['valid'][fold['va']] & fold['eligible'][None, :]
            for s in np.flatnonzero(fold['eligible']):
                held = fit_predict(data, design, fold, params, exclude=s)
                if held is None:
                    continue
                for j in np.flatnonzero(mask[:, s]):
                    i = fold['va'][j]
                    rows.append({'fold': fold['number'], 'sensor': data['names'][s], 'time': str(data['times'][i]),
                                 'all_r': float(full[j, s] * data['scale'] + data['center']),
                                 'held_r': float(held[j, s] * data['scale'] + data['center']),
                                 'design_identical': bool(same)})
        r = pd.DataFrame(rows)
        da, ua, n = max_diff(published, r, 'all_sensor', 'all_r')
        dh, _, _ = max_diff(published, r, 'held_sensor', 'held_r')
        out[variable] = {'rows': n, 'unmatched': ua, 'all_sensor_max_abs_diff': da, 'held_sensor_max_abs_diff': dh,
                         'design_identical': bool(r.design_identical.all()),
                         'pass': bool(ua == 0 and da <= TOL and dh <= TOL and r.design_identical.all())}
    return {'pass': bool(all(v['pass'] for v in out.values())), 'cells': out}


def main():
    published = frames()
    report = {'declared_in': M.DECLARED, 'runtime': M.runtime(), 'tolerance': TOL}
    report['D-G3_inner_window'] = g3()
    print('D-G3', report['D-G3_inner_window']['pass'], flush=True)
    report['D-G1_refit'] = g1(published)
    print('D-G1', report['D-G1_refit']['pass'], json.dumps(report['D-G1_refit']['nested_temperature_check']),
          json.dumps(report['D-G1_refit']['shared']), flush=True)
    refs = references(published)
    report['D-G5_pairing'] = {k: v for k, v in refs.items() if k.endswith('weather_tree')}
    report['D-G5_pairing']['pass'] = bool(all(v['pass'] for v in report['D-G5_pairing'].values()))
    report['H-G3_references'] = {'cells': refs, 'pass': bool(all(v['pass'] for v in refs.values()))}
    print('D-G5', report['D-G5_pairing']['pass'], 'H-G3', report['H-G3_references']['pass'], flush=True)
    xgb = json.loads((RUNS / 'v2_xgb_sweep/gate_G1.json').read_text())
    report['D-G6_build_shifted_offset0'] = {'source': 'reports/bay_twin/runs/v2_xgb_sweep/gate_G1.json',
                                            'pass': bool(all(xgb[v]['pass'] for v in ('temperature', 'oxygen')))}
    report['H-G2_designs'] = designs()
    print('H-G2', report['H-G2_designs']['pass'], flush=True)
    report['H-G1_seabed_block'] = h_g1()
    print('H-G1', report['H-G1_seabed_block']['pass'], json.dumps(report['H-G1_seabed_block']['cells']), flush=True)
    keys = [k for k in report if k[:2] in ('D-', 'H-')]
    report['all_pass'] = bool(all(report[k]['pass'] for k in keys))
    M.save(OUT, report)
    print('wrote', OUT, 'all pass:', report['all_pass'])


if __name__ == '__main__':
    main()
