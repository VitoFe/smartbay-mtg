"""Addendum 47: what selecting per outer fold moves, with the thesis's own interval conventions.

Reads the three designs written by `src/experiments/v2_perfold.py` (published refit, shared re-run,
per-fold) and reports, for the eight forecasting cells and the five Field-57 arms:

  * every gate the declaration names (G1-G5), before any number is used;
  * each arm's error under each design and its movement, perfold - shared (the design effect in one
    environment), perfold - published (how far the thesis's number moves) and shared - published
    (the sampler and library);
  * the headline contrasts under each design and their movement;
  * the declared decision rules D2-D5 and predictions P1-P7, evaluated mechanically.

Forecast intervals follow scripts/v2_horizon_table.py exactly: (fold, ISO week) blocks, 600 draws,
generator [20260920, crc32(stream)]. Field intervals follow scripts/v2_field_contrasts.py: cast
dates, 2,000 draws, a date drawn twice counting twice, generator [20260923, crc32(label)]; a
published contrast keeps that script's label, which ends in '|day' (its column name), and a new
quantity is labelled 'a47|...|date'. Both are computed here from per-block (per-cast) sums, which
gives the same draws as the pandas implementations; gates G3 and G4b check that on the published
rows. Field-57 'published' is read from the published files themselves (amendment 1).

Usage: python -m scripts.v2_perfold_contrasts
"""
import json
import zlib
from pathlib import Path

import numpy as np
import pandas as pd

from src.experiments.sweep_verdict import verdict as sweep_verdict

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / 'reports/bay_twin/runs'
WORK = RUNS / 'v2_perfold'
OUT = RUNS / 'v2_perfold.json'
HORIZON_RUN = RUNS / '20260920T083322Z_v2_horizon/results'
DESIGNS = ('published', 'shared', 'perfold')
ARMS = ('tree', 'weather_tree')
CELLS = [(v, h) for v in ('temperature', 'oxygen') for h in (6, 24)]
KEY = ['fold', 'sensor', 'time']
FLOOR = {'temperature': 0.029, 'oxygen': 1.8}
DECLARED_SEED_RANGE = {('temperature', 6): 0.0031, ('temperature', 24): 0.0165,
                       ('oxygen', 6): 0.246, ('oxygen', 24): 0.360}
FIELD_ARMS = ('A1_both_direct', 'A2_none_direct', 'A3_design_only_direct', 'A4_base_only_direct',
              'A7_ctdprior_direct')
TARGETS = ('temp_c', 'salinity_psu', 'do_umol_l')
# (label as scripts/v2_field_contrasts.py names it, arm, reference)
FIELD_CONTRASTS = [('boundary product, both uses, vs none', 'A1_both_direct', 'A2_none_direct'),
                   ('background offset only vs none', 'A4_base_only_direct', 'A2_none_direct'),
                   ('regressors only vs none', 'A3_design_only_direct', 'A2_none_direct'),
                   ('background offset only vs both', 'A4_base_only_direct', 'A1_both_direct'),
                   ('CTD prior, harmonic, vs none', 'A7_ctdprior_direct', 'A2_none_direct'),
                   ('CTD prior, harmonic, vs Copernicus', 'A7_ctdprior_direct', 'A1_both_direct')]


def star(ci):
    return bool(ci[0] > 0 or ci[1] < 0)


# ---------------------------------------------------------------- forecast: week-block bootstrap

class WeekBlocks:
    """The v2_horizon_table bootstrap, from per-(block, cell) sums of absolute error."""

    def __init__(self, frame):
        week = pd.to_datetime(frame.time, utc=True).dt.isocalendar().week.astype(int).to_numpy()
        pairs = list(zip(frame.fold.to_numpy().tolist(), week.tolist()))
        order = list(dict.fromkeys(pairs))          # first appearance, as drop_duplicates keeps it
        position = {p: i for i, p in enumerate(order)}
        self.block = np.array([position[p] for p in pairs])
        cells = list(dict.fromkeys(zip(frame.fold.tolist(), frame.sensor.tolist())))
        index = {c: i for i, c in enumerate(cells)}
        self.cell = np.array([index[c] for c in zip(frame.fold.tolist(), frame.sensor.tolist())])
        self.cell_fold = np.array([c[0] for c in cells])
        self.folds = np.unique(self.cell_fold)
        self.nb, self.nc = len(order), len(cells)
        self.N = np.zeros((self.nb, self.nc))
        np.add.at(self.N, (self.block, self.cell), 1.0)
        self.frame = frame

    def sums(self, column):
        err = (self.frame.truth - self.frame[column]).abs().to_numpy()
        S = np.zeros((self.nb, self.nc))
        np.add.at(S, (self.block, self.cell), err)
        return S

    def macro(self, m, S, fold=None):
        tot, cnt = m @ S, m @ self.N
        present = cnt > 0
        if fold is not None:
            present &= self.cell_fold == fold
        means = [np.mean(tot[present & (self.cell_fold == f)] / cnt[present & (self.cell_fold == f)])
                 for f in self.folds if (present & (self.cell_fold == f)).any()]
        return float(np.mean(means))

    def value(self, weights, fold=None):
        ones = np.ones(self.nb)
        return float(sum(w * self.macro(ones, self.sums(c), fold) for c, w in weights.items()))

    def boot(self, weights, stream, draws=600):
        rng = np.random.default_rng([20260920, zlib.crc32(stream.encode())])
        sums = {c: self.sums(c) for c in weights}
        point = self.value(weights)
        out = np.empty(draws)
        for i in range(draws):
            m = np.bincount(rng.integers(0, self.nb, self.nb), minlength=self.nb).astype(float)
            out[i] = sum(w * self.macro(m, sums[c]) for c, w in weights.items())
        lo, hi = np.percentile(out, [2.5, 97.5])
        return [point, float(lo), float(hi)]


def load_cell(variable, horizon):
    """All six prediction columns of one (variable, horizon) on the published row order."""
    base = None
    for arm in ARMS:
        for design in DESIGNS:
            d = pd.read_csv(WORK / 'forecast' / design / f'{variable}_{arm}_h{horizon}' / 'selected_predictions.csv.gz')
            col = f'{design}:{arm}'
            if base is None:
                base = d[KEY + ['truth', 'persistence']].copy()
            j = base[KEY].merge(d[KEY + ['truth', 'prediction']], on=KEY, how='left', validate='one_to_one')
            assert len(j) == len(base) == len(d) and np.allclose(j.truth, base.truth, rtol=0, atol=1e-12), (variable, horizon, col)
            base[col] = j.prediction.to_numpy()
    return base


#: Addendum 69: Field-57 arms whose cast prior is rebuilt per inner split. Their designs are read from
#: the inner-prior runs: 'published' is v2_field_ctdprior --inner-prior (results/<arm>_innerprior/), and
#: shared and per fold are WORK/field_innerprior/. That study ran locally, so the shared re-run is the
#: same study and stands as its refit (gate G4a compares the two). Empty by default.
INNER_PRIOR = set()
#: the contrasts gate G4b reproduces; None is RUNS/v2_field_contrasts.json (addendum 69 passes the
#: same contrasts on its own sources, at the four decimals the gate compares)
FIELD_CONTRASTS_PATH = None


def inner_prior_published(name):
    """The report fields the gates read, for the inner-prior 'published' run of `name`."""
    run = ROOT / 'results' / f'v2_field_held_site_{name}_innerprior'
    r = json.loads((run / 'report.json').read_text())
    shared = WORK / 'field_innerprior' / 'shared' / name
    pub, ref = pd.read_csv(run / 'selected_predictions.csv.gz'), pd.read_csv(shared / 'selected_predictions.csv.gz')
    cols = [c for c in pub.columns if c.startswith('pred_')]
    same = (list(pub.columns) == list(ref.columns) and len(pub) == len(ref)
            and (pub[['cast_id', 'depth_m', 'fold']].to_numpy() == ref[['cast_id', 'depth_m', 'fold']].to_numpy()).all())
    a, b = pub[cols].to_numpy(float), ref[cols].to_numpy(float)
    diff = float(np.nanmax(np.abs(a - b)))
    rel = float(np.nanmax(np.abs(a - b) / np.maximum(np.abs(a), 1e-12)))
    sr = json.loads((shared / 'report.json').read_text())
    mae_diff = max(abs(r['nested_mae'][t] - sr['nested_mae'][t]) for t in r['nested_mae'])
    alpha = r['selection']['params']['alpha']
    return {'alpha_by_fold': {'0': alpha, '1': alpha}, 'selection': r['selection'], 'nested_mae': r['nested_mae'],
            'nested_mae_mean_over_draws': r['nested_mae'],
            'gate_G4_rows_identical': bool(same), 'gate_G4_max_abs_prediction_diff': diff,
            'gate_G4_max_relative_prediction_diff': rel, 'gate_G4_nested_mae_abs_diff': mae_diff,
            'gate_G4_pass': bool(same and diff <= 1e-9 and mae_diff <= 1e-9),
            'refit_is': 'the shared study re-run (v2_perfold field --inner-prior), the same local study'}


def reports(kind, design, name):
    if kind == 'field' and name in INNER_PRIOR:
        if design == 'published':
            return inner_prior_published(name)
        return json.loads((WORK / 'field_innerprior' / design / name / 'report.json').read_text())
    return json.loads((WORK / kind / design / name / 'report.json').read_text())


def forecast():
    summary = json.loads((RUNS / 'v2_horizon_summary.json').read_text())
    matched = json.loads((RUNS / 'v2_phase_matched.json').read_text())
    out = {'gates': {}, 'cells': {}}
    g1, g2, g5 = [], [], []
    for v, h in CELLS:
        for arm in ARMS:
            name = f'{v}_{arm}_h{h}'
            p = reports('forecast', 'published', name)
            g1.append({'cell': name, 'pass': p['gate_G1_pass'], 'max_abs_prediction_diff': p['gate_G1_max_abs_prediction_diff']})
            g2.append({'cell': name, 'pass': p['gate_G2_pass'], 'abs_diff': p['gate_G2_abs_diff']})
            s = reports('forecast', 'shared', name)
            g5.append({'study': s['selection']['study'], 'completed': s['selection']['completed_trials']})
            for k, sel in reports('forecast', 'perfold', name)['selection'].items():
                g5.append({'study': sel['study'], 'completed': sel['completed_trials']})
    # G3: the bootstrap here reproduces the published contrasts from the published rows.
    g3 = []
    for v, h in CELLS:
        pub = {a: pd.read_csv(HORIZON_RUN / f'v2_horizon_{v}_{a}_h{h}/selected_predictions.csv.gz') for a in ARMS}
        j = pub['tree'][KEY + ['truth']].assign(t=pub['tree'].prediction.to_numpy()).merge(
            pub['weather_tree'][KEY + ['prediction']].rename(columns={'prediction': 'w'}), on=KEY, validate='one_to_one')
        got = WeekBlocks(j).boot({'w': 1.0, 't': -1.0}, f'{v}|{h}|weather_tree-tree')
        want = summary[v][str(h)]['contrasts']['weather_tree - tree']['value']
        g3.append({'cell': f'{v}|{h}', 'published': want, 'here': got,
                   'max_abs_diff': float(np.max(np.abs(np.array(got) - np.array(want)))),
                   'pass': bool(np.max(np.abs(np.array(got) - np.array(want))) <= 1e-12)})
    out['gates'] = {'G1_refit_reproduces_published_rows': {'pass': all(x['pass'] for x in g1), 'cells': g1},
                    'G2_shared_objective_reproduces_published_best_value': {'pass': all(x['pass'] for x in g2), 'cells': g2},
                    'G3_bootstrap_reproduces_published_contrasts': {'pass': all(x['pass'] for x in g3), 'cells': g3},
                    'G5_trials': {'pass': all(x['completed'] >= 100 for x in g5), 'studies': g5}}
    if not all(out['gates'][g]['pass'] for g in out['gates']):
        print(json.dumps(out['gates'], indent=1))
        raise SystemExit('a forecasting gate failed; nothing is reported')

    for v, h in CELLS:
        frame = load_cell(v, h)
        wb = WeekBlocks(frame)
        cell = {'endpoints': int(len(frame)), 'arms': {}, 'contrast': {}, 'movement': {}}
        for arm in ARMS:
            a = {}
            for d in DESIGNS:
                col = f'{d}:{arm}'
                a[d] = {'mae': wb.value({col: 1.0}), 'fold0': wb.value({col: 1.0}, 0), 'fold1': wb.value({col: 1.0}, 1),
                        'fold0_ratio_to_persistence': wb.value({col: 1.0}, 0) / wb.value({'persistence': 1.0}, 0)}
            if 'selection' in reports('forecast', 'shared', f'{v}_{arm}_h{h}'):
                a['params'] = {'published': reports('forecast', 'published', f'{v}_{arm}_h{h}')['params'],
                               'shared': reports('forecast', 'shared', f'{v}_{arm}_h{h}')['selection']['params'],
                               'perfold': {k: s['params'] for k, s in
                                           reports('forecast', 'perfold', f'{v}_{arm}_h{h}')['selection'].items()}}
            for x, y in [('perfold', 'shared'), ('perfold', 'published'), ('shared', 'published')]:
                a[f'{x} - {y}'] = wb.boot({f'{x}:{arm}': 1.0, f'{y}:{arm}': -1.0}, f'a47|{v}|{h}|{arm}|{x}-{y}')
            cell['arms'][arm] = a
        cell['persistence_mae'] = wb.value({'persistence': 1.0})
        for d in DESIGNS:
            stream = f'{v}|{h}|weather_tree-tree' if d == 'published' else f'a47|{v}|{h}|{d}|weather_tree-tree'
            c = wb.boot({f'{d}:weather_tree': 1.0, f'{d}:tree': -1.0}, stream)
            if d == 'published':   # the refit rows, in their own order, give the published contrast too
                want = summary[v][str(h)]['contrasts']['weather_tree - tree']['value']
                assert np.max(np.abs(np.array(c) - np.array(want))) <= 1e-12, (v, h, c, want)
            cell['contrast'][d] = {'value': c, 'excludes_zero': star(c[1:]),
                                   'fold0': wb.value({f'{d}:weather_tree': 1.0, f'{d}:tree': -1.0}, 0),
                                   'fold1': wb.value({f'{d}:weather_tree': 1.0, f'{d}:tree': -1.0}, 1)}
        for x, y in [('perfold', 'published'), ('perfold', 'shared'), ('shared', 'published')]:
            cell['movement'][f'{x} - {y}'] = wb.boot(
                {f'{x}:weather_tree': 1.0, f'{x}:tree': -1.0, f'{y}:weather_tree': -1.0, f'{y}:tree': 1.0},
                f'a47|{v}|{h}|contrast|{x}-{y}')
        # D2 and D3 against the published seed and phase replicates (all under the shared design).
        m = matched[f'h{h}']
        seeds = {'42': m['phases']['0'][v]['contrast'], **{s: m['seeds'][s][v]['contrast'] for s in ('7', '123')}}
        phases = {k: m['phases'][k][v]['contrast'] for k in m['phases']}
        seed_range = max(x[0] for x in seeds.values()) - min(x[0] for x in seeds.values())
        assert abs(seed_range - DECLARED_SEED_RANGE[(v, h)]) < 6e-4, (v, h, seed_range)
        before = verdict(seeds, phases)
        after = verdict({**seeds, 'perfold': cell['contrast']['perfold']['value']}, phases)
        design_effect = abs(cell['contrast']['perfold']['value'][0] - cell['contrast']['shared']['value'][0])
        cell['verdict'] = {'replicates_seed': seeds, 'phase_points': {k: x[0] for k, x in phases.items()},
                           'phase_mean': float(np.mean([x[0] for x in phases.values()])),
                           'without_perfold': before, 'with_perfold': after, 'D2_triggered': before != after}
        cell['size'] = {'seed_range': seed_range, 'design_effect': design_effect,
                        'D3_triggered': bool(design_effect > seed_range)}
        out['cells'][f'{v}|{h}'] = cell
        pf, pb, sh = (cell['contrast'][d]['value'] for d in ('perfold', 'published', 'shared'))
        print(f'{v:12s} h{h:<3d} weather-tree  published {pb[0]:+.4f} [{pb[1]:+.4f}, {pb[2]:+.4f}]'
              f'  shared {sh[0]:+.4f} [{sh[1]:+.4f}, {sh[2]:+.4f}]  perfold {pf[0]:+.4f} [{pf[1]:+.4f}, {pf[2]:+.4f}]'
              f'  verdict {before} -> {after}  design {design_effect:.4f} vs seeds {seed_range:.4f}')
        for arm in ARMS:
            a = cell['arms'][arm]
            print(f'    {arm:13s} ' + '  '.join(f'{d} {a[d]["mae"]:.4f}' for d in DESIGNS)
                  + '  pf-pub {:+.4f} [{:+.4f}, {:+.4f}]'.format(*a['perfold - published'])
                  + '  pf-sh {:+.4f} [{:+.4f}, {:+.4f}]'.format(*a['perfold - shared']))
    return out


def verdict(seeds, phases):
    """Chapter 4's rule (src.experiments.sweep_verdict): `seeds` maps '42', '7', '123' (and, as one more
    replicate, 'perfold') to (point, lo, hi); `phases` maps the offsets '0' to '5', of which '0' is the
    cell of seed 42."""
    return sweep_verdict(list(seeds.values()), [phases[k] for k in sorted(phases, key=int)])


# ---------------------------------------------------------------- Field-57: cast-date bootstrap

class CastDates:
    """The v2_field_contrasts date bootstrap, from per-cast mean errors."""

    def __init__(self, keys):
        self.keys = keys
        casts = keys.cast_id.to_numpy()
        self.cast_codes, self.cast = np.unique(casts, return_inverse=True)
        first = pd.DataFrame({'c': self.cast, 'fold': keys.fold, 'station': keys.station, 'day': keys.day}).drop_duplicates('c').sort_values('c')
        assert len(first) == len(self.cast_codes)
        cells = list(dict.fromkeys(zip(first.fold, first.station)))
        self.cast_cell = np.array([cells.index(c) for c in zip(first.fold, first.station)])
        self.cell_fold = np.array([c[0] for c in cells])
        self.days = sorted(keys.day.unique())             # groupby('day').indices is ordered by day
        self.cast_day = np.array([self.days.index(d) for d in first.day])
        self.nd = len(self.days)

    def cast_means(self, err):
        s = np.bincount(self.cast, weights=err, minlength=len(self.cast_codes))
        n = np.bincount(self.cast, minlength=len(self.cast_codes))
        return s / n

    def nested(self, w, e):
        """w: weight of each cast (its date's multiplicity); e: each cast's mean error."""
        tot = np.bincount(self.cast_cell, weights=w * e, minlength=self.cast_cell.max() + 1)
        cnt = np.bincount(self.cast_cell, weights=w, minlength=self.cast_cell.max() + 1)
        present = cnt > 0
        folds = [np.mean(tot[present & (self.cell_fold == f)] / cnt[present & (self.cell_fold == f)])
                 for f in np.unique(self.cell_fold) if (present & (self.cell_fold == f)).any()]
        return float(np.mean(folds))

    def boot(self, errs, weights, label, draws=2000):
        rng = np.random.default_rng([20260923, zlib.crc32(label.encode())])
        e = {k: self.cast_means(errs[k]) for k in weights}
        ones = np.ones(len(self.cast_codes))
        point = sum(w * self.nested(ones, e[k]) for k, w in weights.items())
        out = np.empty(draws)
        for i in range(draws):
            m = np.bincount(rng.integers(0, self.nd, self.nd), minlength=self.nd).astype(float)
            wc = m[self.cast_day]
            out[i] = sum(w * self.nested(wc, e[k]) for k, w in weights.items())
        lo, hi = np.percentile(out, [2.5, 97.5])
        return [float(point), float(lo), float(hi)]


def field_source(design, arm):
    """Amendment 1: 'published' is the published file itself; the refit is kept as 'refit' for G4.
    An arm in INNER_PRIOR is read from its inner-prior runs (addendum 69)."""
    if arm in INNER_PRIOR:
        if design == 'published':
            return ROOT / 'results' / f'v2_field_held_site_{arm}_innerprior' / 'selected_predictions.csv.gz'
        return WORK / 'field_innerprior' / ('shared' if design == 'refit' else design) / arm / 'selected_predictions.csv.gz'
    if design == 'published':
        run = (ROOT / 'results') if arm == 'A7_ctdprior_direct' else (RUNS / '20260919T145045Z_v2_field/results')
        return run / f'v2_field_held_site_{arm}/selected_predictions.csv.gz'
    folder = 'published' if design == 'refit' else design
    return WORK / 'field' / folder / arm / 'selected_predictions.csv.gz'


def load_field():
    keys, truth, preds = None, None, {}
    for design in DESIGNS + ('refit',):
        for arm in FIELD_ARMS:
            f = pd.read_csv(field_source(design, arm))
            f = f.sort_values(['fold', 'station', 'cast_id', 'depth_m']).reset_index(drop=True)
            k = f[['fold', 'station', 'cast_id', 'depth_m']]
            if keys is None:
                keys, truth = k, f[list(TARGETS)].to_numpy(float)
                day = pd.to_datetime(f.t_utc, utc=True).dt.strftime('%Y-%m-%d').to_numpy()
            else:
                assert k.equals(keys) and np.allclose(f[list(TARGETS)].to_numpy(float), truth, equal_nan=True), (design, arm)
            preds[f'{design}:{arm}'] = f[[f'pred_{t}' for t in TARGETS]].to_numpy(float)
    # Descriptive, not declared: the physical background alone (Table bay_field_comparators).
    f = pd.read_csv(RUNS / 'v2_field_naive/pred_background.csv.gz')
    f = f.sort_values(['fold', 'station', 'cast_id', 'depth_m']).reset_index(drop=True)
    assert f[['fold', 'station', 'cast_id', 'depth_m']].equals(keys)
    assert np.allclose(f[list(TARGETS)].to_numpy(float), truth, equal_nan=True)
    preds['background'] = f[[f'pred_{t}' for t in TARGETS]].to_numpy(float)
    return keys.assign(day=day), truth, preds


def field():
    published = json.loads((FIELD_CONTRASTS_PATH or RUNS / 'v2_field_contrasts.json').read_text())
    keys, truth, preds = load_field()
    cd = CastDates(keys)
    errs = {t: {c: np.abs(truth[:, j] - p[:, j]) for c, p in preds.items()} for j, t in enumerate(TARGETS)}
    assert all(np.isfinite(e).all() for t in errs for e in errs[t].values())
    out = {'gates': {}, 'arms': {}, 'contrasts': [], 'dates': cd.nd, 'casts': len(cd.cast_codes)}
    g4 = []   # G4 as declared (1e-9) and G4a as amended (relative 1e-6)
    for arm in FIELD_ARMS:
        r = reports('field', 'published', arm)
        a, b = preds[f'published:{arm}'], preds[f'refit:{arm}']
        rel = float(np.max(np.abs(a - b) / np.maximum(np.abs(a), 1e-12)))
        g4.append({'arm': arm, 'declared_1e-9_pass': r['gate_G4_pass'],
                   'max_abs_prediction_diff': r['gate_G4_max_abs_prediction_diff'],
                   'max_relative_prediction_diff': rel, 'nested_mae_abs_diff': r['gate_G4_nested_mae_abs_diff'],
                   'pass': bool(r['gate_G4_rows_identical'] and rel <= 1e-6 and r['gate_G4_nested_mae_abs_diff'] <= 1e-6)})
    g4b, g4c = [], []
    for label, arm, ref in FIELD_CONTRASTS:
        for t in TARGETS:
            got = cd.boot(errs[t], {f'published:{arm}': 1.0, f'published:{ref}': -1.0}, f'{label}|{t}|day')
            want = next(c['date'] for c in published['contrasts'] if c['contrast'] == label and c['target'] == t)
            ok = round(got[0], 4) == want['point'] and [round(got[1], 4), round(got[2], 4)] == want['ci']
            g4b.append({'contrast': label, 'target': t, 'published': [want['point'], *want['ci']],
                        'here': [round(x, 4) for x in got], 'pass': bool(ok)})
            refit = cd.boot(errs[t], {f'refit:{arm}': 1.0, f'refit:{ref}': -1.0}, f'{label}|{t}|day')
            diff = float(np.max(np.abs(np.array(refit) - np.array(got))))
            g4c.append({'contrast': label, 'target': t, 'max_abs_diff': diff, 'pass': bool(diff <= 1e-6)})
    g5 = []
    for arm in FIELD_ARMS:
        g5.append({'study': reports('field', 'shared', arm)['selection']['study'],
                   'completed': reports('field', 'shared', arm)['selection']['completed_trials']})
        for k, sel in reports('field', 'perfold', arm)['selection'].items():
            g5.append({'study': sel['study'], 'completed': sel['completed_trials']})
    out['gates'] = {'G4a_refit_reproduces_published_rows': {'pass': all(x['pass'] for x in g4), 'arms': g4,
                                                            'declared_1e-9_pass': all(x['declared_1e-9_pass'] for x in g4),
                                                            'amended_by': 'runs/v2_perfold/amendment1.json'},
                    'G4b_bootstrap_reproduces_published_contrasts': {'pass': all(x['pass'] for x in g4b), 'contrasts': g4b},
                    'G4c_refit_rows_reproduce_published_contrasts': {'pass': all(x['pass'] for x in g4c), 'contrasts': g4c},
                    'G5_trials': {'pass': all(x['completed'] >= 100 for x in g5), 'studies': g5}}
    if not all(out['gates'][g]['pass'] for g in out['gates']):
        print(json.dumps(out['gates'], indent=1))
        raise SystemExit('a Field-57 gate failed; nothing is reported')
    ones = np.ones(len(cd.cast_codes))
    for arm in FIELD_ARMS:
        a = {'alpha': {d: reports('field', d, arm)['alpha_by_fold'] for d in DESIGNS},
             'nested_mae': {d: {t: cd.nested(ones, cd.cast_means(errs[t][f'{d}:{arm}'])) for t in TARGETS}
                            for d in DESIGNS}}
        for t in TARGETS:
            for x, y in [('perfold', 'published'), ('perfold', 'shared'), ('shared', 'published')]:
                a.setdefault(t, {})[f'{x} - {y}'] = cd.boot(errs[t], {f'{x}:{arm}': 1.0, f'{y}:{arm}': -1.0},
                                                             f'a47|field|{arm}|{x}-{y}|{t}|date')
        out['arms'][arm] = a
        print(f'{arm:22s} alpha ' + '  '.join(f'{d} {a["alpha"][d]}' for d in DESIGNS))
        print(' ' * 23 + '  '.join(f'{t} ' + ' '.join(f'{a["nested_mae"][d][t]:.4f}' for d in DESIGNS) for t in TARGETS))
    for label, arm, ref in FIELD_CONTRASTS:
        for t in TARGETS:
            e = {'contrast': label, 'arm': arm, 'reference': ref, 'target': t}
            for d in DESIGNS:
                lab = f'{label}|{t}|day' if d == 'published' else f'a47|{d}|{label}|{t}|date'
                c = cd.boot(errs[t], {f'{d}:{arm}': 1.0, f'{d}:{ref}': -1.0}, lab)
                e[d] = {'value': c, 'excludes_zero': star(c[1:])}
            e['movement perfold - published'] = cd.boot(
                errs[t], {f'perfold:{arm}': 1.0, f'perfold:{ref}': -1.0, f'published:{arm}': -1.0, f'published:{ref}': 1.0},
                f'a47|movement|{label}|{t}|date')
            e['D4_triggered'] = bool(e['perfold']['excludes_zero'] != e['published']['excludes_zero']
                                     or np.sign(e['perfold']['value'][0]) != np.sign(e['published']['value'][0]))
            out['contrasts'].append(e)
            print(f'{label:40s} {t:12s} pub {e["published"]["value"][0]:+.4f} {"*" if e["published"]["excludes_zero"] else " "}'
                  f'  sh {e["shared"]["value"][0]:+.4f} {"*" if e["shared"]["excludes_zero"] else " "}'
                  f'  pf {e["perfold"]["value"][0]:+.4f} [{e["perfold"]["value"][1]:+.4f}, {e["perfold"]["value"][2]:+.4f}]'
                  f'{"*" if e["perfold"]["excludes_zero"] else " "}  move {e["movement perfold - published"][0]:+.4f}'
                  + ('  D4' if e['D4_triggered'] else ''))
    # Descriptive, NOT declared by addendum 47: the released arm against its own background alone,
    # the 'model vs background alone' contrast of runs/v2_field_contrasts.json, under both designs.
    published_rows = {c['target']: c['date'] for c in published['contrasts'] if c['contrast'] == 'model vs background alone'}
    out['descriptive_not_declared'] = {'note': 'A1_both_direct against the physical background alone; not a declared '
                                               'contrast of addendum 47, reported because the per-fold A1 moves it',
                                       'model_minus_background': {}}
    for t in TARGETS:
        pub = cd.boot(errs[t], {'published:A1_both_direct': 1.0, 'background': -1.0}, f'model vs background alone|{t}|day')
        assert round(pub[0], 4) == published_rows[t]['point'] and [round(pub[1], 4), round(pub[2], 4)] == published_rows[t]['ci'], (t, pub)
        pf = cd.boot(errs[t], {'perfold:A1_both_direct': 1.0, 'background': -1.0}, f'a47|posthoc|model vs background alone|{t}|date')
        out['descriptive_not_declared']['model_minus_background'][t] = {'published': pub, 'perfold': pf}
        print(f'model - background {t:12s} published {pub[0]:+.4f} [{pub[1]:+.4f}, {pub[2]:+.4f}]'
              f'  perfold {pf[0]:+.4f} [{pf[1]:+.4f}, {pf[2]:+.4f}]')
    return out


def predictions(fc, fd):
    c = fc['cells']
    t6 = c['temperature|6']['contrast']['perfold']
    p4 = [c[f'{v}|{h}']['arms'][a]['perfold']['fold0_ratio_to_persistence']
          - c[f'{v}|{h}']['arms'][a]['shared']['fold0_ratio_to_persistence'] for v, h in CELLS for a in ARMS]
    p5 = {f'{v}|{h}|{a}': c[f'{v}|{h}']['arms'][a]['perfold - published'][0] for v, h in CELLS for a in ARMS}
    temp = [e for e in fd['contrasts'] if e['target'] == 'temp_c' and e['reference'] == 'A2_none_direct'
            and e['arm'] in ('A1_both_direct', 'A4_base_only_direct', 'A7_ctdprior_direct')]
    return {
        'P1': {'held': bool(t6['value'][0] < 0 and t6['excludes_zero']), 'perfold': t6['value']},
        'P2': {'held': bool(c['oxygen|24']['contrast']['perfold']['value'][0] < 0),
               'perfold': c['oxygen|24']['contrast']['perfold']['value']},
        'P3': {'held': not any(c[k]['size']['D3_triggered'] for k in c),
               'design_effect_vs_seed_range': {k: [c[k]['size']['design_effect'], c[k]['size']['seed_range']] for k in c}},
        'P4': {'held': bool(np.mean(p4) > 0), 'mean_fold0_ratio_change': float(np.mean(p4)), 'by_cell': p4,
               'cells_positive': int(sum(x > 0 for x in p4))},
        'P5': {'held': all(abs(x) < FLOOR[k.split('|')[0]] for k, x in p5.items()), 'perfold_minus_published': p5},
        'P6': {'held': all(e['perfold']['value'][0] < 0 and e['perfold']['excludes_zero']
                           and abs(e['movement perfold - published'][0]) < 0.01 for e in temp),
               'contrasts': {e['arm']: {'perfold': e['perfold']['value'],
                                        'movement': e['movement perfold - published'][0]} for e in temp}},
        'P7': {'held': not any(e['D4_triggered'] for e in fd['contrasts'])},
    }


def main():
    fc = forecast()
    fd = field()
    preds = predictions(fc, fd)
    rules = {'D2_triggered': [k for k, x in fc['cells'].items() if x['verdict']['D2_triggered']],
             'D3_triggered': [k for k, x in fc['cells'].items() if x['size']['D3_triggered']],
             'D4_triggered': [f"{e['contrast']}|{e['target']}" for e in fd['contrasts'] if e['D4_triggered']]}
    rules['D5_table'] = bool(rules['D2_triggered'] or rules['D3_triggered'] or rules['D4_triggered'])
    overlap = json.loads((WORK / 'overlap.json').read_text())
    report = {'declared_in': 'config/bay_reserved_protocol_v2_addendum47.json',
              'overlap': {'forecast': {k: {kk: overlap['forecast'][k][kk] for kk in
                                           ('endpoints', 'endpoints_total', 'fold0_scored_that_were_fold1_inner_validation',
                                            'fold0_scored_that_were_fold1_inner_training', 'share_of_fold0', 'share_of_all')}
                                       for k in overlap['forecast']},
                          'field57': {k: overlap['field57'][k] for k in ('fold0', 'fold1')}},
              'forecast': fc, 'field57': fd, 'decision_rules': rules, 'registered_predictions': preds}
    OUT.write_text(json.dumps(report, indent=2, allow_nan=False), encoding='utf-8')
    print(json.dumps(rules))
    print({k: v['held'] for k, v in preds.items()})
    print('wrote', OUT)


if __name__ == '__main__':
    main()
