"""Addendum 82, part D: forecasting at an unseen site, per fold, with intervals, swept.

Reads the held-site predictions of src/experiments/v2_held_site_nested.py (fetched under
reports/bay_twin/runs/v2_addendum82/D/), the published held-site frames, and the per-fold all-site
references, and scores the declared contrasts:

  D1  temperature transfer penalty: the held-site weather tree minus the all-site weather tree
      selected per fold (primary), and minus the shared all-site selection (published, beside)
  D2  the same for oxygen, with the nested oxygen arm; the declared sensitivity for the moved cell
  D3  held site minus the hour-of-day increment built from the other sites
  D4  held site minus persistence
  D5  held site minus the hour-of-day increment from the probe's own record (secondary)
  D6  per-site penalties and the penalty without each site (points)
  D7  the nested oxygen arm minus the shared-configuration one (artifact only)

Intervals. Temperature: whole sites (8) drawn with relabelling, 2,000 draws, and the delete-one-site
jackknife-t beside it; 'robust over sites' when both exclude zero, and the hull of the two is the
interval a verdict reads. Joint ISO weeks within fold, probes fixed, beside them. Oxygen: three sites
are fewer than five, so no interval over sites (n.e.); joint ISO weeks within fold with the sondes
fixed. The scheme functions here generalise those of scripts/v2_transfer_intervals.py to a linear
combination of arms' macro errors, drawing in the same order, and gate D-G4 checks that they
reproduce every interval of runs/v2_transfer_intervals.json under the published labels before any
new contrast is read. New contrasts draw under labels prefixed 'a82|'.

The sweep: the penalty (D1, D2) at offsets 1-5 on seed 42 and seeds 7 and 123 on offset 0, each
replicate with its own held-site studies and its own per-fold all-site weather trees; D1 and D2 get
sweep_verdict over the eight replicates, D3 and D4 are stored at every replicate.

Writes reports/bay_twin/runs/v2_addendum82_D.json (six decimals). Oxygen stays in umol/kg, the
sondes' stored column, with mg/l copies (x c, src/units.py) where a number would be printed.
Usage: python -m scripts.v2_transfer_perfold [--gate-only]
"""
import argparse
import json
import zlib
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm
from scipy.stats import t as student

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / 'reports/bay_twin/runs'
A82 = RUNS / 'v2_addendum82'
OUT = RUNS / 'v2_addendum82_D.json'
DECLARED = 'config/bay_reserved_protocol_v2_addendum82.json'
DRAWS = 2000
MIN_SITES = 5
KEY = ['fold', 'sensor', 'time']
Z = float(norm.ppf(0.975))
Q = 0.05
REPLICATES = [(o, 42) for o in range(1, 6)] + [(0, 7), (0, 123)]
FLOOR_T = 0.20


def c_mgl():
    from src.units import factor
    return float(factor())


def r6(x):
    return None if x is None else round(float(x), 6)


# ------------------------------------------------------------------ the schemes, for any combination

def macro(frame, col):
    f = frame.dropna(subset=[col])
    return float(f.assign(e=(f.truth - f[col]).abs()).groupby(['fold', 'sensor']).e.mean()
                 .groupby('fold').mean().mean())


def value(frame, combo):
    return sum(w * macro(frame, c) for c, w in combo)


def weeks(frame):
    return pd.to_datetime(frame.time, utc=True).dt.isocalendar().week.astype(int).to_numpy()


def joint_weeks(frame, combo, label):
    """ISO weeks drawn jointly within fold, cells fixed (v2_transfer_intervals.joint_weeks, any combo)."""
    rng = np.random.default_rng([20260923, zlib.crc32(label.encode())])
    cols = [c for c, _ in combo]
    f = frame.assign(week=weeks(frame))
    for k, c in enumerate(cols):
        e = (frame.truth - frame[c]).abs()
        f[f'e{k}'] = e.fillna(0.0).to_numpy()
        f[f'n{k}'] = e.notna().astype(float).to_numpy()
    names = [x for k in range(len(cols)) for x in (f'e{k}', f'n{k}')]
    tables = []
    for _, fold in f.groupby('fold', sort=True):
        t = fold.groupby(['sensor', 'week'])[names].sum()
        weeks_ = sorted(t.index.get_level_values('week').unique())
        sensors = sorted(t.index.get_level_values('sensor').unique())
        grid = {c: t[c].unstack('week', fill_value=0.0).reindex(index=sensors, columns=weeks_, fill_value=0.0)
                .to_numpy() for c in names}
        for k in range(len(cols)):
            grid[f'has{k}'] = grid[f'n{k}'].sum(axis=1) > 0
        tables.append(grid)
    out = np.empty(DRAWS)
    redrawn = 0
    for i in range(DRAWS):
        per_arm = [[] for _ in cols]
        for grid in tables:
            w = grid['n0'].shape[1]
            for _ in range(10_000):
                m = np.bincount(rng.integers(0, w, w), minlength=w).astype(float)
                ns = [grid[f'n{k}'] @ m for k in range(len(cols))]
                if all((ns[k][grid[f'has{k}']] > 0).all() for k in range(len(cols))):
                    break
                redrawn += 1
            else:
                raise RuntimeError('no joint week draw keeps every cell')
            for k in range(len(cols)):
                h = grid[f'has{k}']
                if h.any():
                    per_arm[k].append(np.mean((grid[f'e{k}'] @ m)[h] / ns[k][h]))
        out[i] = sum(w_ * np.mean(v) for (_, w_), v in zip(combo, per_arm))
    return out, redrawn


def within_cell_weeks(frame, combo, label):
    """ISO weeks within each (fold, sensor) cell on their own (the earlier scheme, gate D-G4 only)."""
    rng = np.random.default_rng([20260923, zlib.crc32(label.encode())])
    cols = [c for c, _ in combo]
    f = frame.assign(week=weeks(frame))
    cells = []
    for (fold, _), cell in f.groupby(['fold', 'sensor']):
        per_week = []
        for _, g in cell.groupby('week'):
            row = []
            for c in cols:
                e = (g.truth - g[c]).abs()
                row += [e.sum(), e.notna().sum()]
            per_week.append(row)
        cells.append((fold, np.array(per_week, dtype=float)))
    folds_in = sorted({c[0] for c in cells})
    out = np.empty(DRAWS)
    for i in range(DRAWS):
        acc = [{k: [] for k in folds_in} for _ in cols]
        for fold, table in cells:
            s = table[rng.integers(0, len(table), len(table))].sum(axis=0)
            for k in range(len(cols)):
                if s[2 * k + 1] > 0:
                    acc[k][fold].append(s[2 * k] / s[2 * k + 1])
        out[i] = sum(w * np.mean([np.mean(v) for v in acc[k].values() if v]) for k, (_, w) in enumerate(combo))
    return out


def by_site(frame, combo, label):
    """Whole sites drawn with relabelling (v2_transfer_intervals.by_site), from per-cell errors."""
    names = sorted(frame.site.unique())
    if len(names) < MIN_SITES:
        return None
    rng = np.random.default_rng([20260924, zlib.crc32(label.encode())])
    folds_ = sorted(frame.fold.unique())
    per_arm = []
    for c, _ in combo:
        f = frame.dropna(subset=[c])
        cell = f.assign(e=(f.truth - f[c]).abs()).groupby(['site', 'fold', 'sensor']).e.mean().reset_index()
        S = np.zeros((len(names), len(folds_)))
        N = np.zeros((len(names), len(folds_)))
        for _, r in cell.iterrows():
            S[names.index(r.site), folds_.index(r.fold)] += r.e
            N[names.index(r.site), folds_.index(r.fold)] += 1
        per_arm.append((S, N))
    out = np.empty(DRAWS)
    for i in range(DRAWS):
        k = np.bincount(rng.integers(0, len(names), len(names)), minlength=len(names)).astype(float)
        total = 0.0
        for (S, N), (_, w) in zip(per_arm, combo):
            s, n = k @ S, k @ N
            total += w * np.mean(s[n > 0] / n[n > 0])
        out[i] = total
    return out


def jackknife(frame, combo, point):
    """Delete-one-site jackknife-t centred on the point; None below MIN_SITES."""
    names = sorted(frame.site.unique())
    k = len(names)
    if k < MIN_SITES:
        return None
    theta = np.array([value(frame[frame.site != s], combo) for s in names])
    se = float(np.sqrt((k - 1) / k * ((theta - theta.mean()) ** 2).sum()))
    half = float(student.ppf(0.975, k - 1)) * se
    return {'ci': [point - half, point + half], 'leave_one_out': dict(zip(names, theta.tolist()))}


def interval(draws):
    lo, hi = np.percentile(draws, [2.5, 97.5])
    return [float(lo), float(hi)]


def excludes(ci):
    return ci is not None and (ci[0] > 0 or ci[1] < 0)


def p_side(point, lo, hi):
    """Addendum 79's p from the estimate and the interval's end towards zero (scripts/v2_fdr.p_side)."""
    if point == 0:
        return 1.0
    s = (hi - point) / Z if point < 0 else (point - lo) / Z
    if s <= 0:
        return None
    return float(2 * norm.sf(abs(point) / s))


def step_up(p, dependence_robust=False):
    """Benjamini-Hochberg (or -Yekutieli) at Q, in input order (scripts/v2_fdr.step_up)."""
    p = np.asarray(p, dtype=float)
    m = len(p)
    if m == 0:
        return []
    c = np.sum(1.0 / np.arange(1, m + 1)) if dependence_robust else 1.0
    order = np.argsort(p, kind='stable')
    below = np.nonzero(p[order] <= Q * np.arange(1, m + 1) / (m * c))[0]
    reject = np.zeros(m, dtype=bool)
    if len(below):
        reject[order[:below[-1] + 1]] = True
    return reject.tolist()


def contrast(frame, combo, label, variable, floor):
    """Every declared scheme for one contrast, six decimals, with the floor flags and p-values."""
    cols = [c for c, _ in combo]
    paired = frame.dropna(subset=cols)
    point = value(paired, combo)
    jw, redrawn = joint_weeks(paired, combo, label)
    weeks_ci = interval(jw)
    out = {'combo': [[c, w] for c, w in combo], 'label': label, 'paired_rows': int(len(paired)),
           'sites': int(paired.site.nunique()), 'point': r6(point),
           'weeks': {'scheme': 'joint ISO weeks within fold, ' + ('probes' if variable == 'temperature' else 'sondes')
                               + ' fixed', 'ci': [r6(x) for x in weeks_ci], 'excludes_zero': excludes(weeks_ci),
                     'redrawn': int(redrawn), 'p': r6(p_side(point, *weeks_ci))}}
    boot = by_site(paired, combo, label)
    jack = jackknife(paired, combo, point)
    if boot is None:
        out['sites'] = {'ci': None, 'status': 'n.e.', 'reason': f'{paired.site.nunique()} sites, fewer than {MIN_SITES}'}
        out['jackknife'] = {'ci': None, 'status': 'n.e.'}
        out['robust_over_sites'] = None
        decision = weeks_ci
        out['decision'] = {'scheme': 'weeks, sondes fixed (site level n.e.)', 'ci': [r6(x) for x in decision],
                           'excludes_zero': excludes(decision), 'p': out['weeks']['p']}
    else:
        sites_ci = interval(boot)
        out['sites'] = {'ci': [r6(x) for x in sites_ci], 'excludes_zero': excludes(sites_ci),
                        'p': r6(p_side(point, *sites_ci))}
        out['jackknife'] = {'ci': [r6(x) for x in jack['ci']], 'excludes_zero': excludes(jack['ci']),
                            'p': r6(p_side(point, *jack['ci'])),
                            'leave_one_out': {k: r6(v) for k, v in jack['leave_one_out'].items()}}
        out['robust_over_sites'] = bool(excludes(sites_ci) and excludes(jack['ci']))
        decision = [min(sites_ci[0], jack['ci'][0]), max(sites_ci[1], jack['ci'][1])]
        ps = [out['sites']['p'], out['jackknife']['p']]
        out['decision'] = {'scheme': 'hull of the site bootstrap and the site jackknife', 'ci': [r6(x) for x in decision],
                           'excludes_zero': excludes(decision),
                           'p': None if None in ps else max(ps)}
    out['status'] = ('excludes zero' if out['decision']['excludes_zero'] else 'not distinguishable')
    out['floor'] = floor
    out['point_beyond_floor'] = bool(abs(point) > floor)
    out['interval_beyond_floor'] = bool(decision[0] > floor or decision[1] < -floor)
    if variable == 'oxygen':
        c = c_mgl()
        out['mgl'] = {'point': r6(point * c), 'weeks_ci': [r6(x * c) for x in weeks_ci],
                      'decision_ci': [r6(x * c) for x in decision], 'floor': r6(floor * c)}
    return out


# ------------------------------------------------------------------ frames

def published_frames():
    t = pd.read_csv(RUNS / 'v2_held_station/held_station_temperature_h6_nested.csv.gz')
    o = pd.read_csv(RUNS / 'v2_held_station/held_station_oxygen_h6.csv.gz')
    return {'temperature': t, 'oxygen': o}


def hour_only(data, site_map):
    """v2_transfer_intervals.hour_only_climatology on a given (possibly shifted) grid."""
    from src.experiments.v2_horizon import climatology, folds
    t = data['times']
    increment = data['truth'] - data['x']
    site_of_probe = np.array([site_map[n] for n in data['names']])
    rows = []
    for fold in folds(data, 6):
        _, hour_table = climatology(data, fold)
        mask = data['valid'][fold['va']] & fold['eligible'][None, :]
        tr = fold['tr']
        hours = t.hour.to_numpy()[tr]
        others = {}
        for site in pd.unique(site_of_probe[fold['eligible']]):
            keep = fold['eligible'] & (site_of_probe != site)
            per_probe = []
            for s in np.flatnonzero(keep):
                ok = data['valid'][tr, s]
                per_probe.append(pd.Series(increment[tr, s][ok]).groupby(hours[ok]).mean())
            others[site] = pd.concat(per_probe, axis=1).mean(axis=1)
        for j, s in np.argwhere(mask):
            i = fold['va'][j]
            step = others[site_of_probe[s]].get(t[i].hour, np.nan)
            rows.append({'fold': fold['number'], 'sensor': data['names'][s], 'time': str(t[i]),
                         'clim_hour': float(hour_table[i, s] * data['scale'] + data['center']),
                         'clim_hour_other_sites': float((data['x'][i, s] + step) * data['scale'] + data['center'])})
    return pd.DataFrame(rows)


def site_map():
    meta = pd.read_csv(ROOT / 'reports/bay_twin/sensor_geometry.csv').set_index('sensor')
    return meta.site.to_dict()


def combine_cells(folder, expected_keys=None, column='prediction'):
    """Join a cell folder's per-site predictions; refuse a duplicate row or a missing expected row."""
    parts = [pd.read_csv(p) for p in sorted(folder.glob('*.csv.gz'))]
    frame = pd.concat([p for p in parts if len(p)], ignore_index=True)
    if frame.duplicated(KEY).any():
        raise SystemExit(f'{folder}: duplicated rows')
    if expected_keys is not None:
        got = set(map(tuple, frame[KEY].astype(str).values))
        want = set(map(tuple, expected_keys[KEY].astype(str).values))
        if got != want:
            raise SystemExit(f'{folder}: {len(want - got)} expected rows missing, {len(got - want)} unexpected')
    return frame.rename(columns={column: folder.name})


def records(folder):
    out = {}
    for p in sorted(folder.glob('*_record.json')):
        r = json.loads(p.read_text())
        out[r['site']] = {k: ({kk: vv for kk, vv in v.items() if kk != 'selection'} |
                              ({'study': v['selection']['study'], 'completed_trials': v['selection']['completed_trials'],
                                'failed_trials': v['selection']['failed_trials'],
                                'best_value': v['selection']['best_value'], 'params': v['selection']['params']}
                               if 'selection' in v else {}))
                          for k, v in r['folds'].items()}
    return out


# ------------------------------------------------------------------ D-G4

def gate_g4():
    """The schemes reproduce every contrast of runs/v2_transfer_intervals.json under its labels."""
    import scripts.v2_transfer_intervals as P
    published = json.loads((RUNS / 'v2_transfer_intervals.json').read_text())['studies']
    sites = P.site_of()
    checks, ok = {}, True
    for variable in ('temperature', 'oxygen'):
        nested = RUNS / f'v2_held_station/held_station_{variable}_h6_nested.csv.gz'
        station = pd.read_csv(nested if nested.exists() else RUNS / f'v2_held_station/held_station_{variable}_h6.csv.gz')
        station = station.merge(P.hour_only_climatology(variable, sites, {}), on=KEY, how='left', validate='one_to_one')
        held = pd.read_csv(RUNS / f'v2_held_sensor/held_sensor_{variable}_h6.csv.gz')
        bathy = pd.read_csv(RUNS / f'v2_held_sensor/held_sensor_{variable}_h6_bathy.csv.gz')
        merged = held.merge(bathy[KEY + ['all_sensor', 'held_sensor']], on=KEY, suffixes=('', '_bathy'),
                            validate='one_to_one')
        merged['site'] = merged.sensor.map(sites)
        for study, frame, prefix in ((f'held_station_{variable}', station, 'station'),
                                     (f'held_sensor_{variable}', merged, 'sensor')):
            for key, want in published[study]['contrasts'].items():
                x, y = key.split(' - ')
                combo = [(x, 1.0), (y, -1.0)]
                label = f'{prefix}|{variable}|{x}|{y}'
                point = value(frame, combo)
                jw, redrawn = joint_weeks(frame, combo, label)
                wc = interval(within_cell_weeks(frame, combo, label))
                boot = by_site(frame, combo, label)
                jack = jackknife(frame, combo, point)
                got = {'point': round(point, 4), 'weeks': [round(v, 4) for v in interval(jw)], 'redrawn': redrawn,
                       'within_cell': [round(v, 4) for v in wc],
                       'sites': None if boot is None else [round(v, 4) for v in interval(boot)],
                       'jackknife': None if jack is None else [round(v, 4) for v in jack['ci']]}
                exp = {'point': want['week_blocks']['point'], 'weeks': want['week_blocks']['ci'],
                       'redrawn': want['week_blocks']['joint_redrawn'],
                       'within_cell': want['week_blocks']['within_cell']['ci'],
                       'sites': want['sites']['ci'], 'jackknife': want['sites_jackknife']['ci']}
                same = got == exp
                ok &= same
                checks[f'{study}|{key}'] = {'pass': bool(same), **({} if same else {'got': got, 'published': exp})}
    # the generalised increment from the other sites, on the reference grid, against the published one
    from src.experiments.v2_horizon import build
    for variable in ('temperature', 'oxygen'):
        a = P.hour_only_climatology(variable, sites, {})
        b = hour_only(build(variable, 6), sites)
        m = a.merge(b, on=KEY, suffixes=('', '_new'), validate='one_to_one')
        same = bool(len(m) == len(a) == len(b) and (m.clim_hour == m.clim_hour_new).all()
                    and np.array_equal(m.clim_hour_other_sites.to_numpy(), m.clim_hour_other_sites_new.to_numpy(),
                                       equal_nan=True))
        ok &= same
        checks[f'hour_only|{variable}'] = {'pass': same, 'rows': int(len(m))}
    return {'pass': bool(ok), 'compared_at': 'four decimals, as published', 'cells': checks}


# ------------------------------------------------------------------ part D

def reference_frames(published):
    from src.experiments.v2_horizon import build
    sites = site_map()
    perfold = {v: pd.read_csv(RUNS / f'v2_perfold/forecast/perfold/{v}_weather_tree_h6/selected_predictions.csv.gz')
               for v in ('temperature', 'oxygen')}
    t = published['temperature'].rename(columns={'held_station_nested': 'HW', 'all_sensor': 'AW_shared',
                                                   'held_station': 'HW_shared_config'})
    t = t.merge(perfold['temperature'][KEY + ['prediction']].rename(columns={'prediction': 'AW_perfold'}),
                on=KEY, validate='one_to_one')
    t = t.merge(hour_only(build('temperature', 6), sites), on=KEY, validate='one_to_one')
    o = published['oxygen'].rename(columns={'all_sensor': 'AW_shared', 'held_station': 'HW_O_shared'})
    hw = combine_cells(A82 / 'D/cells/oxygen_weather_tree_h6_o0_s42', o)
    o = o.merge(hw[KEY + ['oxygen_weather_tree_h6_o0_s42']].rename(columns={'oxygen_weather_tree_h6_o0_s42': 'HW'}),
                on=KEY, validate='one_to_one')
    o = o.merge(perfold['oxygen'][KEY + ['prediction']].rename(columns={'prediction': 'AW_perfold'}),
                on=KEY, validate='one_to_one')
    o = o.merge(hour_only(build('oxygen', 6), sites), on=KEY, validate='one_to_one')
    return t, o


def sensitivity(o, rec):
    """The declared sensitivity for oxygen fold 1 / Baia Santa Teresa: fold 0's Baia-held configuration.
    Also checks that a local refit at each cell's own selection reproduces the host's predictions."""
    from src.experiments import v2_held_site_nested as M
    check = {}
    for site, folds_ in rec.items():
        params = {int(k): v['params'] for k, v in folds_.items() if 'params' in v}
        if not params:
            continue
        local = M.refit('oxygen', 'weather_tree', site, params, column='local')
        m = o[KEY + ['HW']].merge(local[KEY + ['local']], on=KEY, validate='one_to_one')
        check[site] = {'rows': int(len(m)), 'max_abs_diff_host_vs_local': float((m.HW - m.local).abs().max())}
    borrowed = rec['Baia_S.Teresa']['0']['params']
    sens = M.refit('oxygen', 'weather_tree', 'Baia_S.Teresa', {1: borrowed}, column='HW_sens_cell')
    o = o.merge(sens[KEY + ['HW_sens_cell']], on=KEY, how='left', validate='one_to_one')
    o['HW_sensitivity'] = o.HW_sens_cell.fillna(o.HW)
    return o.drop(columns='HW_sens_cell'), {'local_refit_reproduces_host': check,
                                            'pass': bool(all(v['max_abs_diff_host_vs_local'] <= 1e-9
                                                             for v in check.values())),
                                            'sensitivity_rows_replaced': int(len(sens))}


def per_site(frame, x, y):
    out = {'penalty_per_site': {}, 'penalty_without_site': {}, 'per_cell': {}}
    for site, g in frame.groupby('site'):
        out['penalty_per_site'][site] = r6(macro(g, x) - macro(g, y))
        rest = frame[frame.site != site]
        out['penalty_without_site'][site] = r6(macro(rest, x) - macro(rest, y))
    for (fold, site), g in frame.groupby(['fold', 'site']):
        out['per_cell'][f'{fold}|{site}'] = {'endpoints': int(len(g)), 'penalty': r6(macro(g, x) - macro(g, y))}
    return out


def mae_table(frame, cols):
    return {c: r6(macro(frame, c)) for c in cols}


def sweep_frame(variable, o, s):
    from src.experiments import v2_held_site_nested as M
    tag = f'{variable}_weather_tree_h6_o{o}_s{s}'
    allsite = pd.read_csv(A82 / 'D/allsite' / tag / 'predictions.csv.gz')
    hw = combine_cells(A82 / 'D/cells' / tag, allsite)
    f = hw[['fold', 'site', 'sensor', 'time', 'truth', 'persistence', tag]].rename(columns={tag: 'HW'})
    m = f.merge(allsite[KEY + ['truth', 'prediction', 'persistence']].rename(
        columns={'prediction': 'AW_perfold', 'truth': 'truth_a', 'persistence': 'persistence_a'}), on=KEY,
        validate='one_to_one')
    if not (np.array_equal(m.truth, m.truth_a) and np.array_equal(m.persistence, m.persistence_a)):
        raise SystemExit(f'{tag}: truth or persistence differs between the held-site and all-site frames')
    m = m.drop(columns=['truth_a', 'persistence_a'])
    m = m.merge(hour_only(M.build_data(variable, o), site_map()), on=KEY, validate='one_to_one')
    return m, records(A82 / 'D/cells' / tag), json.loads((A82 / 'D/allsite' / tag / 'record.json').read_text())


def main(gate_only=False):
    from src.experiments.sweep_verdict import label as vlabel, verdict
    c = c_mgl()
    floors = {'temperature': FLOOR_T, 'oxygen': 0.059 / c}
    report = {'declared_in': DECLARED, 'part': 'D', 'draws': DRAWS, 'min_sites': MIN_SITES,
              'units': {'temperature': 'degC', 'oxygen': 'umol/kg as stored, with mg/l copies by c = %.6f' % c},
              'floors': {'temperature_degC': FLOOR_T, 'oxygen_mgl': 0.059, 'oxygen_umol_per_kg': r6(floors['oxygen']),
                         'rule': 'every transfer contrast is read against the 0.20 degC co-located margin; oxygen 0.059 mg/l'},
              'gates': {}}
    report['gate_notes'] = ['A first local gate run compared design arrays with numpy.array_equal, which counts '
                           'NaN as unequal (the weather channels have missing hours) and so reported H-G1 and '
                           'H-G2 as failed while every prediction agreed to 6e-14; the gates were rerun with a '
                           'bit-for-bit comparison and passed. The rerun overwrote the record of the first run.']
    local = json.loads((A82 / 'gates_local.json').read_text())
    report['gates'].update({k: {'pass': local[k]['pass'], 'source': 'reports/bay_twin/runs/v2_addendum82/gates_local.json'}
                            for k in ('D-G1_refit', 'D-G3_inner_window', 'D-G5_pairing', 'D-G6_build_shifted_offset0')})
    report['gates']['D-G2_study_reproduction'] = json.loads((A82 / 'D/gate_DG2.json').read_text())
    report['gates']['D-G4_scorer'] = gate_g4()
    print('D-G4', report['gates']['D-G4_scorer']['pass'], flush=True)
    if not all(g['pass'] for g in report['gates'].values()):
        OUT.write_bytes(json.dumps(report, indent=1).encode())
        raise SystemExit('a gate failed; part D is not read')
    if gate_only:
        OUT.write_bytes(json.dumps(report, indent=1).encode())
        return

    published = published_frames()
    t, o = reference_frames(published)
    rec_o = records(A82 / 'D/cells/oxygen_weather_tree_h6_o0_s42')
    o, sens = sensitivity(o, rec_o)
    report['gates']['oxygen_local_refit'] = sens
    report['reference_grid'] = {
        'temperature': {'endpoints': int(len(t)), 'sites': int(t.site.nunique()), 'sensors': int(t.sensor.nunique()),
                        'mae': mae_table(t, ['HW', 'AW_perfold', 'AW_shared', 'HW_shared_config', 'persistence',
                                             'clim_hour', 'clim_hour_other_sites'])},
        'oxygen': {'endpoints': int(len(o)), 'sites': int(o.site.nunique()), 'sensors': int(o.sensor.nunique()),
                   'selections': rec_o,
                   'mae': mae_table(o, ['HW', 'HW_sensitivity', 'HW_O_shared', 'AW_perfold', 'AW_shared',
                                        'persistence', 'clim_hour', 'clim_hour_other_sites'])}}
    spec = {
        'temperature': [('D1', 'HW', 'AW_perfold', True), ('D1_shared', 'HW', 'AW_shared', False),
                        ('D3', 'HW', 'clim_hour_other_sites', False), ('D4', 'HW', 'persistence', False),
                        ('D5', 'HW', 'clim_hour', False)],
        'oxygen': [('D2', 'HW', 'AW_perfold', True), ('D2_shared', 'HW', 'AW_shared', True),
                   ('D2_sensitivity', 'HW_sensitivity', 'AW_perfold', True),
                   ('D3', 'HW', 'clim_hour_other_sites', True), ('D4', 'HW', 'persistence', True),
                   ('D5', 'HW', 'clim_hour', True), ('D7', 'HW', 'HW_O_shared', True)]}
    published_ivals = json.loads((RUNS / 'v2_transfer_intervals.json').read_text())['studies']['held_station_temperature']
    published_key = {'D1_shared': 'held_station_nested - all_sensor', 'D3': 'held_station_nested - clim_hour_other_sites',
                     'D4': 'held_station_nested - persistence', 'D5': 'held_station_nested - clim_hour'}
    family = []
    for variable, frame in (('temperature', t), ('oxygen', o)):
        block = report['reference_grid'][variable]
        block['contrasts'] = {}
        for name, x, y, new in spec[variable]:
            if not new:
                # the same arms on the same rows as a published contrast: its published intervals stand
                block['contrasts'][name] = {'published_contrast': published_key[name], 'arms': [x, y],
                                            'point': r6(value(frame, [(x, 1.0), (y, -1.0)])),
                                            'published': published_ivals['contrasts'][published_key[name]],
                                            'in_family': False}
                continue
            r = contrast(frame, [(x, 1.0), (y, -1.0)], f'a82|D|ref|{variable}|{x}|{y}', variable, floors[variable])
            block['contrasts'][name] = r
            family += family_members(f'ref|{variable}|{name}', r, variable)
            print(variable, name, r['point'], r['weeks']['ci'], r.get('sites', {}).get('ci'),
                  r.get('jackknife', {}).get('ci'), r['status'], flush=True)
        block['D6'] = per_site(frame, 'HW', 'AW_perfold')
        if variable == 'oxygen':
            block['D6_sensitivity'] = per_site(frame, 'HW_sensitivity', 'AW_perfold')
            block['D6_shared'] = per_site(frame, 'HW', 'AW_shared')
        else:
            block['D6_shared'] = per_site(frame, 'HW', 'AW_shared')

    # the sweep
    sweep = {'replicates': {}, 'verdicts': {}}
    cells = {('temperature', 'D1'): [], ('oxygen', 'D2'): []}
    for variable, frame, name in (('temperature', t, 'D1'), ('oxygen', o, 'D2')):
        r = report['reference_grid'][variable]['contrasts'][name]
        cells[(variable, name)].append(((0, 42), (r['point'], *r['decision']['ci'])))
    for o_, s_ in REPLICATES:
        for variable in ('temperature', 'oxygen'):
            frame, rec, arec = sweep_frame(variable, o_, s_)
            key = f'{variable}|o{o_}|s{s_}'
            entry = {'endpoints': int(len(frame)), 'sites': int(frame.site.nunique()),
                     'held_site_cells': {site: {k: {kk: v[kk] for kk in ('study', 'completed_trials', 'failed_trials',
                                                                           'inner') if kk in v} | (
                         {'not_selectable': v['not_selectable']} if 'not_selectable' in v else {})
                         for k, v in folds_.items()} for site, folds_ in rec.items()},
                     'allsite_studies': {k: {'study': v['selection']['study'],
                                             'completed_trials': v['selection']['completed_trials']}
                                         for k, v in arec['folds'].items()},
                     'mae': mae_table(frame, ['HW', 'AW_perfold', 'persistence', 'clim_hour_other_sites']),
                     'contrasts': {}}
            name = 'D1' if variable == 'temperature' else 'D2'
            for cname, x, y in ((name, 'HW', 'AW_perfold'), ('D3', 'HW', 'clim_hour_other_sites'),
                                ('D4', 'HW', 'persistence')):
                r = contrast(frame, [(x, 1.0), (y, -1.0)], f'a82|D|sweep|o{o_}|s{s_}|{variable}|{x}|{y}', variable,
                             floors[variable])
                entry['contrasts'][cname] = r
                family += family_members(f'sweep|o{o_}|s{s_}|{variable}|{cname}', r, variable)
            sweep['replicates'][key] = entry
            r = entry['contrasts'][name]
            cells[(variable, name)].append(((o_, s_), (r['point'], *r['decision']['ci'])))
            print('sweep', key, name, r['point'], r['decision']['ci'], flush=True)
    for (variable, name), cs in cells.items():
        d = dict(cs)
        seeds = [d[(0, 42)], d[(0, 7)], d[(0, 123)]]
        phases = [d[(o_, 42)] for o_ in range(6)]
        code = verdict(seeds, phases)
        pts = [p for p, _, _ in phases]
        sweep['verdicts'][f'{variable}|{name}'] = {
            'code': code, 'label': vlabel(code), 'interval_read': ('hull of the site bootstrap and jackknife'
                                                                   if variable == 'temperature' else
                                                                   'joint weeks, sondes fixed (site level n.e.)'),
            'seeds': [[r6(x) for x in c_] for c_ in seeds], 'phases': [[r6(x) for x in c_] for c_ in phases],
            'phase_mean': r6(np.mean(pts)), 'phase_range': r6(max(pts) - min(pts)),
            'cells_excluding_zero': int(sum(excludes(c_[1:]) for _, c_ in cs))}
        print('verdict', variable, name, code, flush=True)
    report['sweep'] = sweep

    # false-discovery control over the part's own intervals
    bh = step_up([m['p'] for m in family])
    by = step_up([m['p'] for m in family], dependence_robust=True)
    for m, a, b in zip(family, bh, by):
        m['bh'] = bool(a and m['excludes_zero'])
        m['by'] = bool(b and m['excludes_zero'])
    report['multiplicity'] = {
        'family': 'part D: every interval on a difference computed by this part (reference-grid contrasts new to '
                  'this addendum and every sweep cell); excluded are contrasts whose arms and rows are published '
                  '(their published intervals stand) and intervals of a level',
        'p_value': 'addendum 79: from the estimate and the interval end towards zero; for the hull of the site '
                   'bootstrap and the jackknife the larger of the two',
        'tests': len(family), 'exclude_zero': int(sum(m['excludes_zero'] for m in family)),
        'bh_retained': int(sum(m['bh'] for m in family)), 'by_retained': int(sum(m['by'] for m in family)),
        'members': family}
    report['compute'] = compute_record()
    OUT.write_bytes(json.dumps(report, indent=1, allow_nan=False).encode())
    print('wrote', OUT, report['multiplicity']['tests'], report['multiplicity']['exclude_zero'],
          report['multiplicity']['bh_retained'], report['multiplicity']['by_retained'])


def compute_record():
    """The host queues' outcome: jobs, exit states, job-hours, and the host's runtime."""
    out = {}
    remote = A82 / 'remote/D'
    for p in sorted(remote.glob('queue_*.json')):
        s = json.loads(p.read_text())
        jobs = s['jobs'].values()
        out[p.stem] = {'slots': s['slots'], 'jobs': len(s['jobs']),
                       'done': sum(j['state'] == 'done' for j in jobs),
                       'failed': sum(j['state'] == 'failed' for j in jobs),
                       'job_hours': round(sum(j.get('seconds', 0) for j in jobs) / 3600, 3),
                       'last_load': s['load']}
    rt = remote / 'runtime.json'
    return {'queues': out, 'host_runtime': json.loads(rt.read_text()) if rt.exists() else None,
            'threads': 'V2_THREADS=2, OMP/MKL/OpenBLAS=2, LightGBM n_jobs=2, CUDA_VISIBLE_DEVICES empty',
            'max_concurrent_jobs': 8}


def family_members(name, r, variable):
    out = [{'name': f'{name}|weeks', 'point': r['point'], 'ci': r['weeks']['ci'], 'p': r['weeks']['p'],
            'excludes_zero': r['weeks']['excludes_zero']}]
    if variable == 'temperature':
        out.append({'name': f'{name}|sites+jackknife', 'point': r['point'], 'ci': r['decision']['ci'],
                    'p': r['decision']['p'], 'excludes_zero': r['decision']['excludes_zero']})
    return out


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--gate-only', action='store_true')
    main(p.parse_args().gate_only)
