"""Addendum 48: what selecting per outer fold moves in the remaining arms, with each producer's own intervals.

Reads the three designs written by `src/experiments/v2_perfold_rest.py` (published refit, shared re-run,
per fold), the plain and weather trees of addendum 47 at 6 and 24 h (`runs/v2_perfold/`), and the
published rows, and reports:

  * every gate the declaration and its amendment name (G1, G1x, G2, G3, G4, G5), before any number is used;
  * each arm's error under each design and its movement (perfold - shared, perfold - published,
    shared - published), on identical rows;
  * every primary contrast the thesis quotes, under each design, its movement perfold - published, and
    the decision rules D2 (verdict changed), D3 (attributed to the design or to the environment) and D4
    (the two swept river contrasts re-read with the per-fold value as one more replicate);
  * the stream sensitivity of amendment 1 (XGBoost and LightGBM under ten subsampling streams);
  * registered predictions P1-P12.

A published contrast is recomputed from the published rows with its producer's generator and stream
(gate G3); every new quantity uses [20260920, crc32('a48|...')]. Field-57 follows
scripts/v2_field_contrasts.py (cast dates, 2,000 draws, a date drawn twice counting twice).

Usage: python -m scripts.v2_perfold_rest_contrasts
"""
import json
import zlib
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.v2_perfold_contrasts import CastDates, verdict

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / 'reports/bay_twin/runs'
WORK = RUNS / 'v2_perfold_rest'
A47 = RUNS / 'v2_perfold'
OUT = RUNS / 'v2_perfold_rest.json'
DESIGNS = ('published', 'shared', 'perfold')
KEY = ['fold', 'sensor', 'time']
FLOOR = {'temperature': 0.029, 'oxygen': 1.8}
TARGETS = ('temp_c', 'salinity_psu', 'do_umol_l')
HORIZON_RUNS = {h: ('20260920T083322Z_v2_horizon' if h in (6, 24, 48) else '20260920T133747Z_v2_horizon')
                for h in (1, 3, 6, 12, 24, 36, 48)}
FAMILIES = {
    'horizon': {'arms': ('tree', 'weather_tree'), 'horizons': (1, 3, 12, 36, 48)},
    'river': {'arms': ('river_tree', 'weather_river_tree'), 'horizons': (6, 24, 48)},
    'riverlag': {'arms': ('ameglia_lag', 'bocca_lag', 'bocca_gated'), 'horizons': (6, 24)},
    'ispra': {'arms': ('sealevel_tree', 'extboundary_tree'), 'horizons': (6, 24, 48)},
    'bathy': {'arms': ('bathy_tree',), 'horizons': (6, 24)},
    'xgb': {'arms': ('xgb_tree', 'xgb_weather_tree'), 'horizons': (6,)},
}


def star(ci):
    return bool(ci[0] > 0 or ci[1] < 0)


class Blocks:
    """The (fold, ISO week) block bootstrap of the forecast producers, from per-(block, cell) sums.

    Blocks are numbered in their order of first appearance in the frame, as drop_duplicates keeps
    them, so a frame in the producer's row order reproduces its draws. `rows` restricts to a subgroup.
    """

    def __init__(self, frame, rows=None):
        f = frame if rows is None else frame[rows]
        self.frame = f.reset_index(drop=True)
        week = pd.to_datetime(self.frame.time, utc=True).dt.isocalendar().week.astype(int).to_numpy()
        pairs = list(zip(self.frame.fold.to_numpy().tolist(), week.tolist()))
        order = list(dict.fromkeys(pairs))
        position = {p: i for i, p in enumerate(order)}
        self.block = np.array([position[p] for p in pairs])
        cells = list(dict.fromkeys(zip(self.frame.fold.tolist(), self.frame.sensor.tolist())))
        index = {c: i for i, c in enumerate(cells)}
        self.cell = np.array([index[c] for c in zip(self.frame.fold.tolist(), self.frame.sensor.tolist())])
        self.cell_fold = np.array([c[0] for c in cells])
        self.folds = np.unique(self.cell_fold)
        self.nb, self.nc = len(order), len(cells)
        self.N = np.zeros((self.nb, self.nc))
        np.add.at(self.N, (self.block, self.cell), 1.0)
        self._sums = {}

    def sums(self, column):
        if column not in self._sums:
            err = (self.frame.truth - self.frame[column]).abs().to_numpy()
            assert np.isfinite(err).all(), column
            S = np.zeros((self.nb, self.nc))
            np.add.at(S, (self.block, self.cell), err)
            self._sums[column] = S
        return self._sums[column]

    def macro(self, m, S, fold=None):
        tot, cnt = m @ S, m @ self.N
        present = cnt > 0
        return float(np.mean([np.mean(tot[present & (self.cell_fold == f)] / cnt[present & (self.cell_fold == f)])
                              for f in self.folds if (present & (self.cell_fold == f)).any()
                              and (fold is None or f == fold)]))

    def value(self, weights, fold=None):
        ones = np.ones(self.nb)
        return float(sum(w * self.macro(ones, self.sums(c), fold) for c, w in weights.items()))

    def boot(self, weights, seed, stream, draws=600):
        rng = np.random.default_rng([seed, zlib.crc32(stream.encode())])
        sums = {c: self.sums(c) for c in weights}
        point = self.value(weights)
        out = np.empty(draws)
        for i in range(draws):
            m = np.bincount(rng.integers(0, self.nb, self.nb), minlength=self.nb).astype(float)
            out[i] = sum(w * self.macro(m, sums[c]) for c, w in weights.items())
        lo, hi = np.percentile(out, [2.5, 97.5])
        return [point, float(lo), float(hi)]


# ---------------------------------------------------------------- loading the rows

def source(family, design, variable, arm, h):
    """Prediction file of one arm under one design. 'published' is the published file itself."""
    if family in ('tree6', 'tree24'):          # the plain and weather trees of addendum 47
        if design == 'published':
            return RUNS / HORIZON_RUNS[h] / 'results' / f'v2_horizon_{variable}_{arm}_h{h}/selected_predictions.csv.gz'
        return A47 / 'forecast' / design / f'{variable}_{arm}_h{h}' / 'selected_predictions.csv.gz'
    if design == 'published':
        if family == 'horizon':
            return RUNS / HORIZON_RUNS[h] / 'results' / f'v2_horizon_{variable}_{arm}_h{h}/selected_predictions.csv.gz'
        if family == 'xgb':
            return RUNS / f'20260919T183751Z_v2_xgb/results/v2_forecast_{variable}_{arm}_nested/selected_predictions.csv.gz'
        run = {'river': '20260920T183451Z_v2_river', 'riverlag': '20260922T065137Z_v2_riverlag',
               'ispra': '20261001T105500Z_v2_ispra', 'bathy': '20260920T194734Z_v2_bathy'}[family]
        return RUNS / run / 'results' / f'v2_{family}_{variable}_{arm}_h{h}/selected_predictions.csv.gz'
    return WORK / 'forecast' / design / f'{family}_{variable}_{arm}_h{h}' / 'selected_predictions.csv.gz'


def arms_at(variable, h):
    """Every (family, arm) scored at this horizon, the trees first."""
    tree_family = 'horizon' if h in FAMILIES['horizon']['horizons'] else f'tree{h}'
    out = [(tree_family, 'tree'), (tree_family, 'weather_tree')]
    for fam, spec in FAMILIES.items():
        if fam != 'horizon' and h in spec['horizons']:
            out += [(fam, a) for a in spec['arms']]
    return out


def frame_at(variable, h):
    """All arms and designs at one (variable, horizon), on the published plain tree's row order."""
    base = None
    order = {}
    for fam, arm in arms_at(variable, h):
        for d in DESIGNS:
            f = pd.read_csv(source(fam, d, variable, arm, h))
            if 'seed' in f:
                raise SystemExit('a tree file carries seeds')
            if base is None:
                base = f[KEY + ['truth', 'persistence']].copy()
                base['time'] = base.time.astype(str)
            f['time'] = f.time.astype(str)
            if d == 'published':
                order[arm] = bool(len(f) == len(base) and (f[KEY].to_numpy() == base[KEY].to_numpy()).all())
            j = base[KEY].merge(f[KEY + ['truth', 'prediction']], on=KEY, how='left', validate='one_to_one')
            assert len(j) == len(base) == len(f) and np.allclose(j.truth, base.truth, rtol=0, atol=1e-9), (variable, h, arm, d)
            base[f'{d}:{arm}'] = j.prediction.to_numpy()
    return base, order


#: Addendum 69: Field-57 arms whose cast prior is rebuilt per inner split, read from their inner-prior
#: runs (scripts/v2_perfold_contrasts.INNER_PRIOR has the same rule). Empty by default.
INNER_PRIOR = set()
#: the contrasts gate G4b reproduces; None is RUNS/v2_field_contrasts.json
FIELD_CONTRASTS_PATH = None


def reports(kind, design, name):
    if kind == 'field' and name in INNER_PRIOR:
        if design == 'published':
            from scripts import v2_perfold_contrasts as P47
            saved = P47.WORK
            P47.WORK = WORK
            try:
                return P47.inner_prior_published(name)
            finally:
                P47.WORK = saved
        return json.loads((WORK / 'field_innerprior' / design / name / 'report.json').read_text())
    return json.loads((WORK / kind / design / name / 'report.json').read_text())


# ---------------------------------------------------------------- forecast contrasts

def contrast_specs(variable, h):
    """(label, arm, reference, producer seed, producer stream, published value, rounding, subgroup)."""
    horizon = json.loads((RUNS / 'v2_horizon_summary.json').read_text())
    river = json.loads((RUNS / 'v2_river_summary.json').read_text())
    ispra = json.loads((RUNS / 'v2_ispra_summary.json').read_text())
    plain = json.loads((RUNS / 'v2_augmented_vs_plain.json').read_text())
    lag = json.loads((RUNS / 'v2_riverlag_summary.json').read_text())
    bathy = json.loads((RUNS / 'v2_bathy_pooled.json').read_text())
    fc = json.loads((RUNS / '20260919T144822Z_v2_forecast/v2_forecast_summary.json').read_text())
    v, H = variable, str(h)
    out = []
    if h in FAMILIES['horizon']['horizons']:
        out.append(('weather_tree - tree', 'weather_tree', 'tree', 20260920, f'{v}|{h}|weather_tree-tree',
                    horizon[v][H]['contrasts']['weather_tree - tree']['value'], None, None))
    if h in FAMILIES['river']['horizons']:
        rc = river[v][H]['contrasts']
        for new, ref in (('weather_river_tree', 'weather_tree'), ('river_tree', 'tree')):
            out.append((f'{new} - {ref}', new, ref, 20260920, f'{v}|{h}|{new}|all', rc[f'{new} - {ref}']['all'][:3], None, None))
            out.append((f'{new} - {ref} | high flow', new, ref, 20260920, f'{v}|{h}|{new}|high',
                        rc[f'{new} - {ref}']['high_flow'][:3], None, 'high'))
        out.append(('weather_river_tree - tree', 'weather_river_tree', 'tree', 20260920,
                    f'{v}|{h}|weather_river_tree|vsplain', plain[v][H]['vs_plain']['weather_river_tree']['minus_plain'],
                    None, None))
    if h in FAMILIES['ispra']['horizons']:
        for arm in FAMILIES['ispra']['arms']:
            out.append((f'{arm} - weather_tree', arm, 'weather_tree', 20260920, f'{v}|{h}|{arm}|all',
                        ispra[v][H]['contrasts'][f'{arm} - weather_tree']['all'][:3], None, None))
            out.append((f'{arm} - tree', arm, 'tree', 20260920, f'{v}|{h}|{arm}|vsplain',
                        plain[v][H]['vs_plain'][arm]['minus_plain'], None, None))
    if h in FAMILIES['riverlag']['horizons']:
        rows = {r['arm']: r for r in lag['rows'] if r['variable'] == v and r['horizon'] == h}
        for arm in FAMILIES['riverlag']['arms']:
            for label, ref in (('vs_plain_tree', 'tree'), ('vs_weather_tree', 'weather_tree'),
                               ('vs_published_river_arm', 'weather_river_tree')):
                r = rows[arm][label]
                out.append((f'{arm} - {ref}', arm, ref, 20260922, f'{v}{h}{arm}{label}',
                            [r['delta'], r['ci_low'], r['ci_high']], 4, None))
    if h in FAMILIES['bathy']['horizons']:
        for ref in ('tree', 'weather_tree'):
            out.append((f'bathy_tree - {ref}', 'bathy_tree', ref, 20260921, f'{v}|{h}|{ref}',
                        bathy['results'][v][H]['contrasts'][ref]['bathy_minus_reference'], None, None))
    if h in FAMILIES['xgb']['horizons']:
        for arm in FAMILIES['xgb']['arms']:
            out.append((f'{arm} - tree', arm, 'tree', 20260919, f'{v}|{arm}', fc[v][arm]['minus_tree'], None, None))
        out.append(('xgb_weather_tree - weather_tree', 'xgb_weather_tree', 'weather_tree', None, None, None, None, None))
    return out


def high_flow_rows(frame):
    from src.data.bay_river import high_flow
    times = pd.to_datetime(frame.time, utc=True)
    flags, _ = high_flow(pd.DatetimeIndex(times.unique()))
    lookup = dict(zip(pd.DatetimeIndex(times.unique()), flags))
    return times.map(lookup).fillna(False).to_numpy().astype(bool)


def forecast():
    out = {'gates': {}, 'cells': {}, 'contrasts': []}
    g1, g2, g5, g3 = [], [], [], []
    for fam, spec in FAMILIES.items():
        for v in ('temperature', 'oxygen'):
            for arm in spec['arms']:
                for h in spec['horizons']:
                    name = f'{fam}_{v}_{arm}_h{h}'
                    p = reports('forecast', 'published', name)
                    entry = {'cell': name, 'pass': p['gate_G1_pass'], 'max_abs_prediction_diff': p['gate_G1_max_abs_prediction_diff']}
                    if fam == 'xgb':
                        entry.update(G1x_a=p['gate_G1x_a_stored_models_max_abs_diff'], G1x_b=p['gate_G1x_b_repeat_max_abs_diff'],
                                     note='G1 and G2 reported, not gated (amendment 1)')
                    g1.append(entry)
                    g2.append({'cell': name, 'pass': p['gate_G2_pass'], 'abs_diff': p['gate_G2_abs_diff']})
                    g5.append({'study': reports('forecast', 'shared', name)['selection']['study'],
                               'completed': reports('forecast', 'shared', name)['selection']['completed_trials']})
                    for sel in reports('forecast', 'perfold', name)['selection'].values():
                        g5.append({'study': sel['study'], 'completed': sel['completed_trials']})
    frames = {}
    for v in ('temperature', 'oxygen'):
        for h in (1, 3, 6, 12, 24, 36, 48):
            frame, order = frame_at(v, h)
            frames[(v, h)] = frame
            hf = high_flow_rows(frame) if h in FAMILIES['river']['horizons'] else None
            blocks = {None: Blocks(frame), 'high': Blocks(frame, hf) if hf is not None else None}
            cell = {'endpoints': int(len(frame)), 'published_files_in_one_row_order': order, 'arms': {},
                    'persistence_mae': blocks[None].value({'persistence': 1.0})}
            if hf is not None:
                cell['high_flow_endpoints'] = int(hf.sum())
            for fam, arm in arms_at(v, h):
                a = {d: {'mae': blocks[None].value({f'{d}:{arm}': 1.0}),
                         'fold0': blocks[None].value({f'{d}:{arm}': 1.0}, 0),
                         'fold1': blocks[None].value({f'{d}:{arm}': 1.0}, 1)} for d in DESIGNS}
                for d in DESIGNS:
                    a[d]['fold0_ratio_to_persistence'] = a[d]['fold0'] / blocks[None].value({'persistence': 1.0}, 0)
                if not fam.startswith('tree'):
                    for x, y in (('perfold', 'shared'), ('perfold', 'published'), ('shared', 'published')):
                        a[f'{x} - {y}'] = blocks[None].boot({f'{x}:{arm}': 1.0, f'{y}:{arm}': -1.0}, 20260920,
                                                            f'a48|{v}|{h}|{arm}|{x}-{y}')
                    rep = {d: reports('forecast', d, f'{fam}_{v}_{arm}_h{h}') for d in ('shared', 'perfold')}
                    a['params'] = {'shared': rep['shared']['selection']['params'],
                                   'perfold': {k: s['params'] for k, s in rep['perfold']['selection'].items()}}
                    if fam == 'riverlag':
                        a['selected_lag_hours'] = {d: reports('forecast', d, f'{fam}_{v}_{arm}_h{h}')['selected_lag_hours']
                                                   for d in DESIGNS}
                a['family'] = fam
                cell['arms'][arm] = a
            out['cells'][f'{v}|{h}'] = cell
            for label, arm, ref, seed, stream, published, rnd, sub in contrast_specs(v, h):
                b = blocks[sub]
                e = {'variable': v, 'horizon': h, 'contrast': label, 'arm': arm, 'reference': ref,
                     'subgroup': sub or 'all', 'endpoints': int(len(b.frame))}
                for d in DESIGNS:
                    w = {f'{d}:{arm}': 1.0, f'{d}:{ref}': -1.0}
                    if d == 'published' and seed is not None:
                        c = b.boot(w, seed, stream)
                        got = [round(x, rnd) for x in c] if rnd else c
                        diff = float(np.max(np.abs(np.array(got) - np.array(published))))
                        g3.append({'contrast': f'{v}|{h}|{label}', 'published': published, 'here': got,
                                   'max_abs_diff': diff, 'pass': bool(diff <= (1e-12 if rnd is None else 0))})
                    else:
                        c = b.boot(w, 20260920, f'a48|{v}|{h}|{d}|{label}')
                    e[d] = {'value': c, 'excludes_zero': star(c[1:])}
                e['movement perfold - published'] = b.boot(
                    {f'perfold:{arm}': 1.0, f'perfold:{ref}': -1.0, f'published:{arm}': -1.0, f'published:{ref}': 1.0},
                    20260920, f'a48|{v}|{h}|movement|{label}')
                pub, sh, pf = e['published'], e['shared'], e['perfold']
                changed = bool(pf['excludes_zero'] != pub['excludes_zero'] or np.sign(pf['value'][0]) != np.sign(pub['value'][0]))
                shared_changed = bool(sh['excludes_zero'] != pub['excludes_zero'] or np.sign(sh['value'][0]) != np.sign(pub['value'][0]))
                e['D2_verdict_changed'] = changed
                e['D3_attribution'] = (None if not changed else ('environment and sampler' if shared_changed else 'design'))
                out['contrasts'].append(e)
                print(f'{v:11s} h{h:<3d} {label:42s} {e["subgroup"]:5s} pub {pub["value"][0]:+.4f}{"*" if pub["excludes_zero"] else " "}'
                      f'  sh {sh["value"][0]:+.4f}{"*" if sh["excludes_zero"] else " "}'
                      f'  pf {pf["value"][0]:+.4f} [{pf["value"][1]:+.4f}, {pf["value"][2]:+.4f}]{"*" if pf["excludes_zero"] else " "}'
                      + (f'  D2 ({e["D3_attribution"]})' if changed else ''), flush=True)
    # The graph arm as published, descriptive (it is not re-selected here).
    conservative = pd.read_csv(RUNS / '20260920T083322Z_v2_horizon/results/v2_horizon_oxygen_conservative_exchange_h6/selected_predictions.csv.gz')
    conservative = conservative.groupby(KEY, as_index=False).prediction.mean()
    conservative['time'] = conservative.time.astype(str)
    fo = frames[('oxygen', 6)]
    j = fo.merge(conservative.rename(columns={'prediction': 'conservative'}), on=KEY, how='left', validate='one_to_one')
    assert j.conservative.notna().all()
    b = Blocks(j)
    out['xgb_vs_conservative_graph_oxygen_6h'] = {
        'note': 'descriptive: the graph arm as published (shared design, three-network mean); not re-selected here',
        **{d: b.boot({f'{d}:xgb_tree': 1.0, 'conservative': -1.0}, 20260920, f'a48|oxygen|6|{d}|xgb_tree-conservative')
           for d in DESIGNS}}
    out['gates'] = {'G1_refit_reproduces_published_rows': {'pass': all(x['pass'] for x in g1), 'cells': g1},
                    'G2_shared_objective_reproduces_published_best_value': {'pass': all(x['pass'] for x in g2), 'cells': g2},
                    'G3_bootstrap_reproduces_published_contrasts': {'pass': all(x['pass'] for x in g3), 'contrasts': g3},
                    'G5_trials': {'pass': all(x['completed'] >= 100 for x in g5), 'studies': len(g5)}}
    return out, frames


def swept(out):
    """D4: the two swept river contrasts, oxygen 24 h, with the per-fold value as one more replicate."""
    res = {}
    for name, path, key in (('river_tree - tree', 'v2_phase_river.json', 'river_tree - tree'),
                            ('weather_river_tree - weather_tree', 'v2_phase_weather_river.json',
                             'weather_river_tree - weather_tree')):
        r = json.loads((RUNS / path).read_text())
        phases = {k: p['contrasts'][key] for k, p in r['phases'].items()}
        seeds = {'42': phases['0'], **{s: r['seeds'][s]['contrasts'][key] for s in ('7', '123')}}
        pf = next(e['perfold']['value'] for e in out['contrasts'] if e['variable'] == 'oxygen' and e['horizon'] == 24
                  and e['contrast'] == name and e['subgroup'] == 'all')
        before, after = verdict(seeds, phases), verdict({**seeds, 'perfold': pf}, phases)
        res[name] = {'published_verdict': r['decision'], 'rule_without_perfold': before, 'with_perfold': after,
                     'perfold': pf, 'D4_triggered': before != after}
    return res


# ---------------------------------------------------------------- stream sensitivity (amendment 1)

def streams():
    res = {}
    for v in ('temperature', 'oxygen'):
        f = pd.read_csv(WORK / 'streams' / f'{v}_h6.csv.gz')
        rep = json.loads((WORK / 'streams' / f'{v}_h6_report.json').read_text())
        b = Blocks(f)
        per = {}
        for s in rep['seeds']:
            per[str(s)] = {'xgb_tree': b.value({f'xgb_tree_s{s}': 1.0}), 'tree': b.value({f'tree_s{s}': 1.0}),
                           'xgb_tree - tree': b.boot({f'xgb_tree_s{s}': 1.0, f'tree_s{s}': -1.0}, 20260920,
                                                     f'a48|stream|{v}|{s}')}
        diffs = [x['xgb_tree - tree'][0] for x in per.values()]
        res[v] = {'checks': rep['checks'], 'per_stream': per,
                  'xgb_mae_range': [min(x['xgb_tree'] for x in per.values()), max(x['xgb_tree'] for x in per.values())],
                  'tree_mae_range': [min(x['tree'] for x in per.values()), max(x['tree'] for x in per.values())],
                  'contrast_range': [min(diffs), max(diffs)], 'contrast_spread': max(diffs) - min(diffs),
                  'streams_excluding_zero': int(sum(star(x['xgb_tree - tree'][1:]) for x in per.values()))}
        print(f'streams {v}: xgb - tree over 10 streams {min(diffs):+.4f} to {max(diffs):+.4f}, '
              f'{res[v]["streams_excluding_zero"]} excluding zero', flush=True)
    return res


# ---------------------------------------------------------------- the apparent-utilisation tree

AOU_PAIRS = [('aou_tree', 'weather_tree'), ('aou_tree', 'plain_tree'), ('aou_tree', 'persistence'),
             ('thermodynamic', 'persistence'), ('thermodynamic', 'weather_tree'), ('thermodynamic', 'climatology')]


def aou():
    published = json.loads((RUNS / 'v2_aou_forecast/aou_forecast_h6_summary.json').read_text())
    frames = {d: pd.read_csv(WORK / 'aou' / d / 'aou_forecast_h6.csv.gz') for d in DESIGNS}
    frames['published'] = pd.read_csv(RUNS / 'v2_aou_forecast/aou_forecast_h6.csv.gz')
    for d in DESIGNS:
        assert (frames[d][KEY].astype(str).to_numpy() == frames['published'][KEY].astype(str).to_numpy()).all(), d
    base = frames['published'][KEY + ['truth', 'persistence', 'climatology']].copy()
    for d in DESIGNS:
        for c in ('aou_tree', 'weather_tree', 'plain_tree', 'thermodynamic'):
            base[f'{d}:{c}'] = frames[d][c].to_numpy()
        base[f'{d}:persistence'] = base.persistence
        base[f'{d}:climatology'] = base.climatology
    b = Blocks(base)
    rep = {d: json.loads((WORK / 'aou' / d / 'report.json').read_text()) for d in DESIGNS}
    out = {'gates': {'G1': rep['published']['gate_G1_pass'], 'G2': rep['published']['gate_G2_pass'],
                     'G5': [rep['shared']['selection']['completed_trials']]
                     + [s['completed_trials'] for s in rep['perfold']['selection'].values()]},
           'mae': {d: rep[d]['mae'] for d in DESIGNS}, 'contrasts': []}
    g3 = []
    for x, y in AOU_PAIRS:
        e = {'contrast': f'{x} - {y}'}
        for d in DESIGNS:
            if d == 'published':
                c = b.boot({f'{d}:{x}': 1.0, f'{d}:{y}': -1.0}, 20260921, f'{x}|{y}')
                diff = float(np.max(np.abs(np.array(c) - np.array(published['contrasts'][f'{x} - {y}']))))
                g3.append({'contrast': f'{x} - {y}', 'max_abs_diff': diff, 'pass': bool(diff <= 1e-12)})
            else:
                c = b.boot({f'{d}:{x}': 1.0, f'{d}:{y}': -1.0}, 20260920, f'a48|aou|{d}|{x}-{y}')
            e[d] = {'value': c, 'excludes_zero': star(c[1:])}
        e['movement perfold - published'] = b.boot({f'perfold:{x}': 1.0, f'perfold:{y}': -1.0, f'published:{x}': -1.0,
                                                    f'published:{y}': 1.0}, 20260920, f'a48|aou|movement|{x}-{y}')
        pub, sh, pf = e['published'], e['shared'], e['perfold']
        changed = bool(pf['excludes_zero'] != pub['excludes_zero'] or np.sign(pf['value'][0]) != np.sign(pub['value'][0]))
        shared_changed = bool(sh['excludes_zero'] != pub['excludes_zero'] or np.sign(sh['value'][0]) != np.sign(pub['value'][0]))
        e['D2_verdict_changed'] = changed
        e['D3_attribution'] = None if not changed else ('environment and sampler' if shared_changed else 'design')
        out['contrasts'].append(e)
        print(f'aou {x} - {y}: pub {pub["value"][0]:+.3f}{"*" if pub["excludes_zero"] else " "} sh {sh["value"][0]:+.3f}'
              f'{"*" if sh["excludes_zero"] else " "} pf {pf["value"][0]:+.3f} [{pf["value"][1]:+.3f}, {pf["value"][2]:+.3f}]'
              f'{"*" if pf["excludes_zero"] else " "}' + ('  D2' if changed else ''), flush=True)
    out['gates']['G3'] = {'pass': all(x['pass'] for x in g3), 'contrasts': g3}
    return out


# ---------------------------------------------------------------- Field-57

FIELD_CONTRASTS = [
    ('fitted intercept vs released', 'A5_intercept_direct', 'A1_both_direct', TARGETS),
    ('CTD prior, month x depth, vs Copernicus', 'A6_ctdprior_direct', 'A1_both_direct', TARGETS),
    ('CTD prior, month x depth, vs none', 'A6_ctdprior_direct', 'A2_none_direct', TARGETS),
    ('fitted depth basis vs Copernicus', 'P1_placebo_direct', 'A1_both_direct', ('temp_c',)),
    ('fitted depth basis vs none', 'P1_placebo_direct', 'A2_none_direct', ('temp_c',)),
    ('random depth basis vs none', 'P2_placebo_direct', 'A2_none_direct', ('temp_c',)),
    ('random depth basis vs Copernicus', 'P2_placebo_direct', 'A1_both_direct', ('temp_c',)),
    ('apparent utilisation vs direct, released family', 'A1_both_aou', 'A1_both_direct', ('do_umol_l',)),
]
NEW_FIELD = ('A5_intercept_direct', 'A6_ctdprior_direct', 'P1_placebo_direct', 'P2_placebo_direct', 'A1_both_aou')


def field_file(design, arm):
    if arm in INNER_PRIOR:
        if design == 'published':
            return ROOT / 'results' / f'v2_field_held_site_{arm}_innerprior' / 'selected_predictions.csv.gz'
        return WORK / 'field_innerprior' / ('shared' if design == 'refit' else design) / arm / 'selected_predictions.csv.gz'
    if arm in ('A1_both_direct', 'A2_none_direct'):
        if design == 'published':
            return RUNS / f'20260919T145045Z_v2_field/results/v2_field_held_site_{arm}/selected_predictions.csv.gz'
        return A47 / 'field' / design / arm / 'selected_predictions.csv.gz'
    if design == 'published':
        if arm == 'A1_both_aou':
            return RUNS / '20260919T145045Z_v2_field/results/v2_field_held_site_A1_both_aou/selected_predictions.csv.gz'
        return ROOT / 'results' / f'v2_field_held_site_{arm}' / 'selected_predictions.csv.gz'
    folder = 'published' if design == 'refit' else design
    return WORK / 'field' / folder / arm / 'selected_predictions.csv.gz'


def field():
    published = json.loads((FIELD_CONTRASTS_PATH or RUNS / 'v2_field_contrasts.json').read_text())
    keys = truth = None
    preds = {}
    for design in DESIGNS + ('refit',):
        for arm in ('A1_both_direct', 'A2_none_direct') + NEW_FIELD:
            if design == 'refit' and arm in ('A1_both_direct', 'A2_none_direct'):
                continue
            f = pd.read_csv(field_file(design, arm)).sort_values(['fold', 'station', 'cast_id', 'depth_m']).reset_index(drop=True)
            k = f[['fold', 'station', 'cast_id', 'depth_m']]
            if keys is None:
                keys, truth = k, f[list(TARGETS)].to_numpy(float)
                day = pd.to_datetime(f.t_utc, utc=True).dt.strftime('%Y-%m-%d').to_numpy()
            else:
                assert k.equals(keys) and np.allclose(f[list(TARGETS)].to_numpy(float), truth, equal_nan=True), (design, arm)
            preds[f'{design}:{arm}'] = f[[f'pred_{t}' for t in TARGETS]].to_numpy(float)
    keys = keys.assign(day=day)
    cd = CastDates(keys)
    errs = {t: {c: np.abs(truth[:, j] - p[:, j]) for c, p in preds.items()} for j, t in enumerate(TARGETS)}
    out = {'gates': {}, 'arms': {}, 'contrasts': [], 'dates': cd.nd, 'casts': len(cd.cast_codes)}
    g4, g4b, g4c, g5 = [], [], [], []
    for arm in NEW_FIELD:
        r = reports('field', 'published', arm)
        g4.append({'arm': arm, 'max_relative_prediction_diff': r['gate_G4_max_relative_prediction_diff'],
                   'nested_mae_abs_diff': r['gate_G4_nested_mae_abs_diff'], 'pass': r['gate_G4_pass']})
        g5.append(reports('field', 'shared', arm)['selection']['completed_trials'])
        g5 += [s['completed_trials'] for s in reports('field', 'perfold', arm)['selection'].values()]
    for label, arm, ref, targets in FIELD_CONTRASTS:
        for t in targets:
            got = cd.boot(errs[t], {f'published:{arm}': 1.0, f'published:{ref}': -1.0}, f'{label}|{t}|day')
            want = next(c['date'] for c in published['contrasts'] if c['contrast'] == label and c['target'] == t)
            ok = round(got[0], 4) == want['point'] and [round(got[1], 4), round(got[2], 4)] == want['ci']
            g4b.append({'contrast': label, 'target': t, 'published': [want['point'], *want['ci']],
                        'here': [round(x, 4) for x in got], 'pass': bool(ok)})
            if ref != 'A1_both_direct' or True:
                refit = cd.boot(errs[t], {f'refit:{arm}': 1.0, f'published:{ref}': -1.0}, f'{label}|{t}|day')
                diff = float(np.max(np.abs(np.array(refit) - np.array(got))))
                g4c.append({'contrast': label, 'target': t, 'max_abs_diff': diff, 'pass': bool(diff <= 1e-6)})
    out['gates'] = {'G4a_refit_reproduces_published_rows': {'pass': all(x['pass'] for x in g4), 'arms': g4},
                    'G4b_bootstrap_reproduces_published_contrasts': {'pass': all(x['pass'] for x in g4b), 'contrasts': g4b},
                    'G4c_refit_rows_reproduce_published_contrasts': {'pass': all(x['pass'] for x in g4c), 'contrasts': g4c},
                    'G5_trials': {'pass': all(x >= 100 for x in g5), 'studies': len(g5)}}
    ones = np.ones(len(cd.cast_codes))
    for arm in NEW_FIELD:
        a = {'alpha': {d: reports('field', d, arm)['alpha_by_fold'] for d in DESIGNS},
             'nested_mae': {d: {t: cd.nested(ones, cd.cast_means(errs[t][f'{d}:{arm}'])) for t in TARGETS} for d in DESIGNS},
             'nested_mae_mean_over_draws': {d: reports('field', d if d != 'published' else 'published', arm)
                                            ['nested_mae_mean_over_draws'] for d in DESIGNS}}
        for t in TARGETS:
            for x, y in (('perfold', 'published'), ('perfold', 'shared'), ('shared', 'published')):
                a.setdefault(t, {})[f'{x} - {y}'] = cd.boot(errs[t], {f'{x}:{arm}': 1.0, f'{y}:{arm}': -1.0},
                                                             f'a48|field|{arm}|{x}-{y}|{t}|date')
        out['arms'][arm] = a
    for label, arm, ref, targets in FIELD_CONTRASTS:
        for t in targets:
            e = {'contrast': label, 'arm': arm, 'reference': ref, 'target': t}
            for d in DESIGNS:
                lab = f'{label}|{t}|day' if d == 'published' else f'a48|{d}|{label}|{t}|date'
                c = cd.boot(errs[t], {f'{d}:{arm}': 1.0, f'{d}:{ref}': -1.0}, lab)
                e[d] = {'value': c, 'excludes_zero': star(c[1:])}
            e['movement perfold - published'] = cd.boot(
                errs[t], {f'perfold:{arm}': 1.0, f'perfold:{ref}': -1.0, f'published:{arm}': -1.0, f'published:{ref}': 1.0},
                f'a48|movement|{label}|{t}|date')
            pub, sh, pf = e['published'], e['shared'], e['perfold']
            changed = bool(pf['excludes_zero'] != pub['excludes_zero'] or np.sign(pf['value'][0]) != np.sign(pub['value'][0]))
            shared_changed = bool(sh['excludes_zero'] != pub['excludes_zero'] or np.sign(sh['value'][0]) != np.sign(pub['value'][0]))
            e['D2_verdict_changed'] = changed
            e['D3_attribution'] = None if not changed else ('environment and sampler' if shared_changed else 'design')
            out['contrasts'].append(e)
            print(f'{label:48s} {t:12s} pub {pub["value"][0]:+.4f}{"*" if pub["excludes_zero"] else " "}'
                  f'  sh {sh["value"][0]:+.4f}{"*" if sh["excludes_zero"] else " "}'
                  f'  pf {pf["value"][0]:+.4f} [{pf["value"][1]:+.4f}, {pf["value"][2]:+.4f}]{"*" if pf["excludes_zero"] else " "}'
                  + ('  D2' if changed else ''), flush=True)
    return out


# ---------------------------------------------------------------- predictions

def predictions(fc, fa, fd):
    def find(v, h, label, sub='all'):
        return next(e for e in fc['contrasts'] if e['variable'] == v and e['horizon'] == h and e['contrast'] == label
                    and e['subgroup'] == sub)
    def fcell(target, label):
        return next(e for e in fd['contrasts'] if e['target'] == target and e['contrast'] == label)
    p1 = {h: find('temperature', h, 'weather_tree - tree')['perfold'] for h in (1, 3, 12)}
    p2 = find('temperature', 36, 'weather_tree - tree')['perfold']
    p5 = {a: find('oxygen', 6, f'{a} - weather_tree')['perfold'] for a in ('weather_river_tree', 'sealevel_tree', 'extboundary_tree')}
    p6_arms = [('river_tree', 'tree'), ('sealevel_tree', 'tree'), ('extboundary_tree', 'tree'), ('bathy_tree', 'tree'),
               ('ameglia_lag', 'tree'), ('bocca_lag', 'tree'), ('bocca_gated', 'tree'), ('weather_river_tree', 'tree')]
    p6 = {a: find('oxygen', 6, f'{a} - {r}')['perfold'] for a, r in p6_arms}
    xt, xo = find('temperature', 6, 'xgb_tree - tree')['perfold'], find('oxygen', 6, 'xgb_tree - tree')['perfold']
    moves = {f'{k}|{a}': c['arms'][a]['perfold']['mae'] - c['arms'][a]['published']['mae']
             for k, c in fc['cells'].items() for a in c['arms'] if not c['arms'][a]['family'].startswith('tree')}
    ratios = [c['arms'][a]['perfold']['fold0_ratio_to_persistence'] - c['arms'][a]['shared']['fold0_ratio_to_persistence']
              for c in fc['cells'].values() for a in c['arms'] if not c['arms'][a]['family'].startswith('tree')]
    return {
        'P1': {'held': all(x['value'][0] < 0 and x['excludes_zero'] for x in p1.values()), 'perfold': p1},
        'P2': {'held': not p2['excludes_zero'], 'perfold': p2},
        'P3': {'held': bool(find('oxygen', 24, 'river_tree - tree')['perfold']['value'][0] < 0
                            and not fc['swept']['river_tree - tree']['D4_triggered']),
               'swept': fc['swept']['river_tree - tree']},
        'P4': {'held': fc['swept']['weather_river_tree - weather_tree']['with_perfold'] == 'not estimable',
               'swept': fc['swept']['weather_river_tree - weather_tree']},
        'P5': {'held': sum(not x['excludes_zero'] for x in p5.values()) >= 2, 'perfold': p5},
        'P6': {'held': not any(x['excludes_zero'] and x['value'][0] < 0 for x in p6.values()), 'perfold': p6},
        'P7': {'held': bool(abs(xt['value'][0]) <= 0.002 and not xt['excludes_zero'] and not xo['excludes_zero']),
               'temperature': xt, 'oxygen': xo},
        'P8': {'held': bool(abs(fcell('temp_c', 'fitted intercept vs released')['perfold']['value'][0]) <= 0.01
                            and not fcell('temp_c', 'fitted intercept vs released')['perfold']['excludes_zero']
                            and fcell('temp_c', 'CTD prior, month x depth, vs none')['perfold']['value'][0] < 0
                            and fcell('temp_c', 'CTD prior, month x depth, vs none')['perfold']['excludes_zero']
                            and not fcell('do_umol_l', 'CTD prior, month x depth, vs none')['perfold']['excludes_zero'])},
        'P9': {'held': bool(not fcell('temp_c', 'fitted depth basis vs none')['perfold']['excludes_zero']
                            and fcell('temp_c', 'fitted depth basis vs Copernicus')['perfold']['value'][0] > 0
                            and fcell('temp_c', 'fitted depth basis vs Copernicus')['perfold']['excludes_zero'])},
        'P10': {'held': bool(abs(fcell('do_umol_l', 'apparent utilisation vs direct, released family')['perfold']['value'][0]) <= 0.05
                             and not fcell('do_umol_l', 'apparent utilisation vs direct, released family')['perfold']['excludes_zero']),
                'perfold': fcell('do_umol_l', 'apparent utilisation vs direct, released family')['perfold']},
        'P11': {'held': all(abs(x) < FLOOR[k.split('|')[0]] for k, x in moves.items())
                and all(abs(fa['mae']['perfold'][a] - fa['mae']['published'][a]) < 1.8 for a in ('aou_tree',)),
                'perfold_minus_published': moves,
                'largest': max(moves.items(), key=lambda kv: abs(kv[1]) / FLOOR[kv[0].split('|')[0]])},
        'P12': {'held': bool(np.mean(ratios) > 0), 'mean_fold0_ratio_change': float(np.mean(ratios)),
                'cells_positive': int(sum(x > 0 for x in ratios)), 'cells': len(ratios)},
    }


def main():
    fc, _ = forecast()
    fc['swept'] = swept(fc)
    fc['streams'] = streams()
    fa = aou()
    fd = field()
    gates = {**{f'forecast {k}': v['pass'] for k, v in fc['gates'].items()},
             **{f'aou {k}': (v if isinstance(v, bool) else v['pass'] if isinstance(v, dict) else all(x >= 100 for x in v))
                for k, v in fa['gates'].items()},
             **{f'field {k}': v['pass'] for k, v in fd['gates'].items()}}
    if not all(gates.values()):
        print(json.dumps(gates, indent=1))
    preds = predictions(fc, fa, fd)
    changed = [f"{e['variable']}|{e['horizon']}|{e['contrast']}|{e['subgroup']}" for e in fc['contrasts'] if e['D2_verdict_changed']]
    changed += [f"aou|{e['contrast']}" for e in fa['contrasts'] if e['D2_verdict_changed']]
    changed += [f"field|{e['contrast']}|{e['target']}" for e in fd['contrasts'] if e['D2_verdict_changed']]
    report = {'declared_in': 'config/bay_reserved_protocol_v2_addendum48.json',
              'amended_by': 'config/bay_reserved_protocol_v2_addendum48_amendment1.json',
              'gates': gates, 'forecast': fc, 'aou': fa, 'field57': fd,
              'decision_rules': {'D2_changed': changed,
                                 'D3_design': [c for c, e in zip(changed, [e for e in fc['contrasts'] if e['D2_verdict_changed']]
                                                                 + [e for e in fa['contrasts'] if e['D2_verdict_changed']]
                                                                 + [e for e in fd['contrasts'] if e['D2_verdict_changed']])
                                               if e['D3_attribution'] == 'design'],
                                 'D4': fc['swept']},
              'registered_predictions': preds}
    OUT.write_bytes(json.dumps(report, indent=2, allow_nan=False).encode('utf-8'))
    print(json.dumps(gates))
    print('changed verdicts:', len(changed), changed)
    print({k: v['held'] for k, v in preds.items()})
    print('wrote', OUT)


if __name__ == '__main__':
    main()
