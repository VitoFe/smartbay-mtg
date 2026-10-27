"""Addendum 81, task E: the gap-recovery contexts every pretrained arm reads, built locally.

One input file per (variable, cohort, direction, gap length) in the common format of
scripts/v2_fm_zoo_io.py, written to reports/bay_twin/runs/v2_fm_zoo/inputs/ as
E_{variable}_{cohort}_{fwd|bwd}_L{length}.npz, with manifest_gaps.json beside them.

Rows. The episodes are those of the published bidirectional Chronos arms (Chronos-Bolt
`chronos_bidir` and Chronos-2 `chronos2_cov_bidir`, runs listed in v2_gap_variants.COHORTS, read
through v2_gap_climatology.read_arm), one context per (episode, direction), in the order of the
published rows. The two published arms score identical rows; that is checked.

Contexts. As v2_gap_recovery.chronos_fill builds them: forward, the probe's record over the up to
512 h before the gap; backward, the up to 512 h after it, reversed in time; each interpolated in
both directions at its own length and any NaN left set to zero (none is). Raw units, no scaling.
The horizon of a context is its gap length L. Step k (0-based) of a forward context forecasts gap
position k + 1 (time gap_start + k h); a backward context's step k forecasts position L - k. The
published bidirectional fill reverses the backward trace and blends w * fwd + (1 - w) * bwd with
w = (L - k') / (L + 1), k' = position - 1, on the published rows (episode, position).

Covariates (cov mode). As chronos_fill's `neighbour_covariates`: the four nearest other probes of
the same variable, nearest first, named nb0..nb3 by rank, over the context window (past) and through
the gap (future), both reversed for the backward window, raw, NaN kept. The published arm dropped a
neighbour from an item when it had no finite value over its context window or none through the
gap; `cov_used` lists those it kept, and a dropped neighbour is still stored here, its values as
recorded. The published order comes from numpy's default argsort, which breaks ties between
co-located probes differently from v2_gap_recovery.nearest_others' stable sort (the probe itself is
excluded by index in both); the published fills are reproduced only by the published order, so
that order is used and every difference is listed in the manifest. The 2026 oxygen panel has three
sondes, so its files carry two neighbours.

Gates (all must pass, or nothing is written):
  E1 rows        both published arms score the same rows; each row's time is gap_start + position - 1
  E2 adapt       Gap-2026 contexts equal runs/v2_chronos_adapt/inputs/gap_{variable}.npz
  E3 holes       Gap-Dev and Gap-2026 forward contexts (and their raw windows) equal
                 runs/v2_context_holes/inputs/B_{cohort}_{variable}.npz
  E4 function    v2_gap_recovery.chronos_fill itself, run with a recording stand-in for the model,
                 passes exactly these targets and covariates, in both directions, for every cohort
  E5 order       Chronos-2 run on CPU through chronos_fill reproduces the published
                 chronos2_cov_bidir fills with the published neighbour order on the probes whose
                 order differs from nearest_others (skipped, and recorded, without local weights)
  E6 round trip  every file read back with load_inputs gives what was written

Usage: python -m scripts.v2_fm_zoo_inputs_gaps [--no-model-gate]
"""
import argparse
import contextlib
import json
import os
import sys
import types
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.v2_fm_zoo_io import digest, load_inputs, save_inputs
from scripts.v2_gap_climatology import read_arm
from scripts.v2_gap_variants import COHORTS, PERIOD as COHORT_PERIOD
from src.experiments import v2_gap_recovery as gap

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / 'reports/bay_twin/runs'
OUT = RUNS / 'v2_fm_zoo' / 'inputs'
MANIFEST = OUT / 'manifest_gaps.json'
ADAPT = RUNS / 'v2_chronos_adapt' / 'inputs'
HOLES = RUNS / 'v2_context_holes' / 'inputs'
WEIGHTS = ROOT / 'companion' / '.cache' / 'hf'
VARIABLES = ('temperature', 'oxygen')
DIRECTIONS = ('fwd', 'bwd')
UNITS = {'temperature': 'degC', 'oxygen': 'umol/L'}
CTX = gap.CHRONOS_CTX
TOLERANCE = 1e-6
#: G1 of the declaration: a model run reproduces the published column to 1e-4
MODEL_TOLERANCE = 1e-4
MODEL_EPISODES_PER_PROBE = 2


def published_order(distance, i):
    """chronos_fill's neighbour_covariates order, line for line (default argsort, i excluded)."""
    return [int(j) for j in np.argsort(distance[i]) if j != i][:gap.NEIGHBOURS]


def same(a, b):
    """Largest absolute difference, inf if the shapes or the NaN positions differ."""
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    if a.shape != b.shape:
        return float('inf')
    na, nb = np.isnan(a), np.isnan(b)
    if (na != nb).any():
        return float('inf')
    return float(np.max(np.abs(a[~na] - b[~na]))) if (~na).any() else 0.0


@contextlib.contextmanager
def gap_period(period):
    """chronos_fill reads the record through the module's PERIOD; set it, and always restore it."""
    saved = gap.PERIOD
    gap.PERIOD = period
    try:
        yield
    finally:
        gap.PERIOD = saved


class _Recorder:
    """Stands in for the Chronos pipelines: records what chronos_fill passes, returns zeros."""
    calls = None

    def __init__(self, repo):
        self.repo = repo
        self.quantiles = [0.1, 0.5, 0.9]

    @classmethod
    def from_pretrained(cls, repo, **_):
        return cls(repo)

    def predict_quantiles(self, inputs, prediction_length, quantile_levels):
        import torch
        _Recorder.calls.append((prediction_length, [x.numpy().copy() for x in inputs]))
        return torch.zeros(len(inputs), prediction_length, len(quantile_levels)), None

    def predict(self, items, prediction_length):
        import torch
        _Recorder.calls.append((prediction_length, list(items)))
        return [torch.zeros(1, len(self.quantiles), prediction_length) for _ in items]


@contextlib.contextmanager
def recording_chronos(calls):
    saved = sys.modules.get('chronos')
    fake = types.ModuleType('chronos')
    fake.BaseChronosPipeline = _Recorder
    _Recorder.calls = calls
    sys.modules['chronos'] = fake
    try:
        yield
    finally:
        if saved is None:
            sys.modules.pop('chronos', None)
        else:
            sys.modules['chronos'] = saved


class _StableNumpy(types.ModuleType):
    """numpy with a stable argsort, to run chronos_fill with nearest_others' tie-break (gate E5)."""

    def __getattr__(self, name):
        return getattr(np, name)

    @staticmethod
    def argsort(a, *args, **kwargs):
        kwargs['kind'] = 'stable'
        return np.argsort(a, *args, **kwargs)


@contextlib.contextmanager
def stable_argsort():
    saved = gap.np
    gap.np = _StableNumpy('numpy_stable_argsort')
    try:
        yield
    finally:
        gap.np = saved


def panel(period, variable, cache):
    """The panel chronos_fill builds (v2_gap_recovery.py, chronos_fill's first lines)."""
    if period not in cache:
        cache[period] = gap.observations(period)
    meta, frames = cache[period]
    names = [n for n, f in frames.items() if variable in f and f[variable].notna().any()]
    index = pd.date_range(min(frames[n].index.min() for n in names),
                          max(frames[n].index.max() for n in names), freq='h', tz='UTC')
    data = pd.DataFrame({n: frames[n][variable].reindex(index) for n in names}, index=index)
    xy = meta.loc[names, ['x_m', 'y_m']].to_numpy()
    distance = np.linalg.norm(xy[:, None] - xy[None, :], axis=-1)
    return names, index, data, distance


def published_rows(cohort, variable):
    """The rows both published bidirectional Chronos arms score, checked identical (gate E1)."""
    run, results, run2, _ = COHORTS[cohort]
    bolt = read_arm(run, results, variable, 'chronos_bidir')
    c2 = read_arm(run2, results, variable, 'chronos2_cov_bidir')
    keys = ['episode', 'position']
    identical = bool(bolt[keys].equals(c2[keys]))
    sources = [RUNS / run / results / f'{variable}_chronos_bidir' / 'predictions.csv.gz',
               RUNS / run2 / results / f'{variable}_chronos2_cov_bidir' / 'predictions.csv.gz']
    rows = bolt[['episode', 'sensor', 'gap_length', 'position', 'fold', 'time']].copy()
    rows['chronos_bidir'] = bolt['chronos_bidir'].to_numpy()
    rows['chronos2_cov_bidir'] = c2['chronos2_cov_bidir'].to_numpy()
    return rows, identical, sources


def build(cohort, variable, cache):
    """Every (episode, direction) context of one cohort and variable, grouped by (length, direction)."""
    period = COHORT_PERIOD[cohort]
    names, index, data, distance = panel(period, variable, cache)
    position_of = {t: i for i, t in enumerate(index)}
    column = {n: i for i, n in enumerate(names)}
    rows, identical, sources = published_rows(cohort, variable)

    # Gate E1: each row sits at gap_start + position - 1, one fold per episode.
    starts = pd.to_datetime(rows.episode.str.split('|').str[1], utc=True)
    times = pd.to_datetime(rows.time, utc=True)
    time_ok = bool((times == starts + pd.to_timedelta(rows.position - 1, unit='h')).all())
    within = bool(((rows.position >= 1) & (rows.position <= rows.gap_length)).all())
    one_fold = bool((rows.groupby('episode').fold.nunique() == 1).all())

    order = {i: published_order(distance, i) for i in range(len(names))}
    stable = {i: gap.nearest_others(distance, i, gap.NEIGHBOURS) for i in range(len(names))}
    n_cov = len(order[0])
    cov_names = tuple(f'nb{c}' for c in range(n_cov))
    scored = rows.groupby('episode', sort=False).agg(fold=('fold', 'first'), n=('position', 'size'))

    items = {}
    for key in rows.episode.unique():
        sensor, start, length = key.rsplit('|', 2)
        g0, length = position_of[pd.Timestamp(start)], int(length)
        i = column[sensor]
        near = order[i]
        if i in near or len(near) != n_cov:
            raise SystemExit(f'{cohort} {variable} {key}: neighbour set {near} for probe {i}')
        series = data[sensor].to_numpy(float)
        for direction in DIRECTIONS:
            if direction == 'fwd':
                window = series[max(0, g0 - CTX):g0]
                edge = index[g0 - 1]
            else:
                window = series[g0 + length:g0 + length + CTX][::-1]
                edge = index[g0 + length]
            filled = pd.Series(window).interpolate(limit_direction='both').to_numpy()
            target = np.nan_to_num(filled).astype(np.float32)
            pcov = np.full((n_cov, len(target)), np.nan, np.float32)
            fcov = np.full((n_cov, length), np.nan, np.float32)
            used = []
            for c, j in enumerate(near):
                s = data[names[j]].to_numpy(float)
                if direction == 'fwd':
                    p, f = s[max(0, g0 - CTX):g0], s[g0:g0 + length]
                else:
                    p, f = s[g0 + length:g0 + length + CTX][::-1], s[g0:g0 + length][::-1]
                if len(p) != len(target):
                    raise SystemExit(f'{key} {direction}: covariate window {len(p)} != {len(target)}')
                pcov[c], fcov[c] = p, f
                if np.isfinite(p).any() and np.isfinite(f).any():
                    used.append(f'nb{c}')
            items.setdefault((length, direction), []).append({
                'episode': key, 'sensor': sensor, 'gap_length': length,
                'fold': int(scored.fold[key]), 'gap_start': index[g0], 'direction': direction,
                'edge_time': edge, 'ctx_missing': int((~np.isfinite(window)).sum()),
                'zero_filled': int((~np.isfinite(filled)).sum()), 'n_scored': int(scored.n[key]),
                **{f'nb{c}_sensor': names[j] for c, j in enumerate(near)},
                'cov_used': ','.join(used), 'used': tuple(used),
                'raw': window.astype(np.float32), 'target': target, 'pcov': pcov, 'fcov': fcov})

    differs = {}
    for i, n in enumerate(names):
        if order[i] != stable[i]:
            differs[n] = {'published': [names[j] for j in order[i]],
                          'nearest_others': [names[j] for j in stable[i]],
                          'same_set': set(order[i]) == set(stable[i])}
    episodes_by_sensor = rows.drop_duplicates('episode').sensor.value_counts()
    affected = {n: {**d, 'episodes_in_cohort': int(episodes_by_sensor.get(n, 0))}
                for n, d in differs.items() if episodes_by_sensor.get(n, 0)}
    e1 = {'arms_score_identical_rows': identical, 'row_time_is_gap_start_plus_position': time_ok,
          'positions_within_gap': within, 'one_fold_per_episode': one_fold,
          'rows': int(len(rows)), 'episodes': int(rows.episode.nunique())}
    e1['passed'] = identical and time_ok and within and one_fold
    return {'period': period, 'items': items, 'rows': rows, 'cov_names': cov_names, 'e1': e1,
            'sources': sources, 'names': names, 'order_differs': affected,
            'probes': len(names)}


def gate_adapt(cohort, variable, built):
    """E2: the Gap-2026 contexts of addendum 41, both directions."""
    z = np.load(ADAPT / f'gap_{variable}.npz')
    theirs = {k: r for r, k in enumerate(z['keys'])}
    mine = {(it['episode'], it['direction']): it for group in built['items'].values() for it in group}
    episodes = {e for e, _ in mine}
    worst, checked = 0.0, 0
    for key, r in theirs.items():
        for direction, ctx, n in (('fwd', z['fwd'], z['fwd_len']), ('bwd', z['bwd'], z['bwd_len'])):
            it = mine.get((key, direction))
            if it is None:
                return {'passed': False, 'reason': f'{key} missing here'}
            worst = max(worst, same(ctx[r, CTX - n[r]:], it['target']))
            checked += 1
    passed = set(theirs) == episodes and worst <= TOLERANCE
    return {'passed': bool(passed), 'contexts': checked, 'max_abs_difference': worst,
            'same_episodes': set(theirs) == episodes}


def gate_holes(cohort, variable, built):
    """E3: the forward contexts of addendum 53 (gaps of 1 to 72 h), interpolated and raw."""
    z = np.load(HOLES / f'B_{cohort}_{variable}.npz')
    mine = {it['episode']: it for (_, d), group in built['items'].items() if d == 'fwd'
            for it in group}
    worst_linear, worst_raw = 0.0, 0.0
    for r, key in enumerate(z['key_episode']):
        it = mine.get(key)
        if it is None:
            return {'passed': False, 'reason': f'{key} missing here'}
        n = int(z['length'][r])
        worst_linear = max(worst_linear, same(z['linear'][r, CTX - n:], it['target']))
        worst_raw = max(worst_raw, same(z['raw'][r, CTX - n:], it['raw']))
    expected = {k for k, it in mine.items() if it['gap_length'] <= 72}
    same_set = set(z['key_episode']) == expected
    passed = same_set and max(worst_linear, worst_raw) <= TOLERANCE
    return {'passed': bool(passed), 'contexts': int(len(z['key_episode'])),
            'max_abs_difference_interpolated': worst_linear, 'max_abs_difference_raw': worst_raw,
            'same_episodes': same_set}


def gate_function(cohort, variable, built):
    """E4: chronos_fill itself, the model replaced by a recorder, in the published call order."""
    frame = built['rows'][['episode', 'position', 'gap_length', 'sensor']]
    lengths = sorted({length for length, _ in built['items']})
    out = {}
    for arm, kwargs in (('chronos_bidir', {}),
                        ('chronos2_cov_bidir', {'repo': gap.CHRONOS2_REPO, 'covariates': True})):
        calls = []
        with recording_chronos(calls), gap_period(built['period']), warnings.catch_warnings():
            warnings.simplefilter('ignore')
            gap.chronos_fill(variable, frame, True, **kwargs)
        flat = [(h, x) for h, xs in calls for x in xs]
        worst, pos, names_ok = 0.0, 0, True
        for length in lengths:
            for direction in DIRECTIONS:
                mine = built['items'][(length, direction)]
                if arm == 'chronos_bidir':
                    sequence = range(len(mine))
                else:
                    # Chronos-2 calls are grouped by the set of neighbours an item carries
                    groups = {}
                    for n, it in enumerate(mine):
                        groups.setdefault(tuple(sorted(it['used'])), []).append(n)
                    sequence = [n for members in groups.values() for n in members]
                for n in sequence:
                    if pos >= len(flat):
                        return {'passed': False, 'reason': f'{arm}: fewer calls than contexts'}
                    h, x = flat[pos]
                    pos += 1
                    it = mine[n]
                    if h != length:
                        return {'passed': False, 'reason': f'{arm}: horizon {h} for length {length}'}
                    if arm == 'chronos_bidir':
                        worst = max(worst, same(x, it['target']))
                        continue
                    worst = max(worst, same(x['target'], it['target']))
                    past = x.get('past_covariates', {})
                    future = x.get('future_covariates', {})
                    if list(past) != list(it['used']) or list(future) != list(it['used']):
                        names_ok = False
                    for name in it['used']:
                        c = int(name[2:])
                        worst = max(worst, same(past.get(name), it['pcov'][c]),
                                    same(future.get(name), it['fcov'][c]))
        complete = pos == len(flat)
        out[arm] = {'passed': bool(complete and names_ok and worst <= TOLERANCE),
                    'contexts': pos, 'max_abs_difference': worst,
                    'covariate_names_match': names_ok, 'all_calls_matched': complete}
    out['passed'] = all(v['passed'] for v in out.values())
    return out


def gate_model(data, enabled):
    """E5: on the probes whose published neighbour order differs from nearest_others', Chronos-2
    (CPU, local weights) through chronos_fill reproduces the published fills with that order."""
    if not enabled:
        return {'state': 'skipped', 'reason': 'disabled on the command line'}
    if not (WEIGHTS / 'hub' / 'models--amazon--chronos-2').exists():
        return {'state': 'skipped', 'reason': 'no local Chronos-2 weights'}
    os.environ.setdefault('HF_HOME', str(WEIGHTS))
    os.environ.setdefault('HF_HUB_OFFLINE', '1')
    cells, passed = {}, True
    for (cohort, variable), built in data.items():
        probes = list(built['order_differs'])
        if not probes:
            continue
        rows = built['rows']
        pick = []
        for sensor in probes:
            mine = rows[rows.sensor == sensor]
            shortest = mine[mine.gap_length == mine.gap_length.min()].episode.unique()
            step = max(1, len(shortest) // MODEL_EPISODES_PER_PROBE)
            pick += list(shortest[::step][:MODEL_EPISODES_PER_PROBE])
        sub = rows[rows.episode.isin(pick)].reset_index(drop=True)
        frame = sub[['episode', 'position', 'gap_length', 'sensor']]
        kwargs = {'repo': gap.CHRONOS2_REPO, 'covariates': True}
        with gap_period(built['period']), warnings.catch_warnings():
            warnings.simplefilter('ignore')
            ours = np.asarray(gap.chronos_fill(variable, frame, True, **kwargs), float)
            with stable_argsort():
                alt = np.asarray(gap.chronos_fill(variable, frame, True, **kwargs), float)
        published = sub.chronos2_cov_bidir.to_numpy(float)
        cell = {}
        for sensor in probes:
            m = (sub.sensor == sensor).to_numpy()
            cell[sensor] = {
                'episodes': int(sub[m].episode.nunique()),
                'gap_length': int(sub[m].gap_length.min()),
                'same_set': built['order_differs'][sensor]['same_set'],
                'published_order_max_abs_difference': float(np.max(np.abs(ours[m] - published[m]))),
                'nearest_others_max_abs_difference': float(np.max(np.abs(alt[m] - published[m])))}
            passed &= cell[sensor]['published_order_max_abs_difference'] <= MODEL_TOLERANCE
        cells[f'{cohort} {variable}'] = cell
    return {'state': 'run', 'device': 'cpu', 'tolerance': MODEL_TOLERANCE, 'passed': bool(passed),
            'cells': cells}


def relative(paths):
    """Sources as repository-relative POSIX paths (read from the repository root)."""
    return [Path(p).resolve().relative_to(ROOT.resolve()).as_posix() if Path(p).is_absolute()
            else Path(p).as_posix() for p in paths]


def write(cohort, variable, built):
    """One file per (direction, gap length); returns the manifest entries."""
    entries = {}
    common = relative([ROOT / 'reports/bay_twin/sensor_geometry.csv',
                       ROOT / 'src/experiments/v2_gap_recovery.py', Path(__file__)]
                      + built['sources']) + [f'data/raw/{n}.txt' for n in built['names']]
    key_columns = (['episode', 'sensor', 'gap_length', 'fold', 'gap_start', 'direction', 'edge_time',
                    'ctx_missing', 'n_scored']
                   + [f'nb{c}_sensor' for c in range(len(built['cov_names']))] + ['cov_used'])
    for (length, direction), group in sorted(built['items'].items()):
        name = f'E_{variable}_{cohort}_{direction}_L{length}.npz'
        keys = pd.DataFrame({c: [it[c] for it in group] for c in key_columns})
        note = ('Gap recovery (addendum 81, task E). One context per published episode, in the '
                'order of the published rows; horizon = gap length. '
                + ('Forward: the record before the gap, oldest first; step k forecasts position '
                   'k + 1. ' if direction == 'fwd' else
                   'Backward: the record after the gap, reversed in time (latest first); step k '
                   'forecasts position L - k; reverse the trace before blending. ')
                + 'Context interpolated in both directions as the published arms read it. '
                'Covariates nb0.. are the nearest other probes (published order), raw with NaN; '
                'cov_used lists those the published Chronos-2 arm kept (a neighbour with no finite '
                'value over the context or through the gap was dropped). Blend: '
                'w * fwd + (1 - w) * bwd, w = (L - k) / (L + 1), k = position - 1.')
        meta = save_inputs(
            OUT / name, task='E_gaps', variable=variable, cohort=cohort, keys=keys,
            contexts=[it['target'] for it in group], horizon=length, units=UNITS[variable],
            past_cov=[it['pcov'] for it in group],
            future_cov=np.stack([it['fcov'] for it in group]),
            cov_names=built['cov_names'], reverse=[direction == 'bwd'] * len(group),
            standardisation='none: raw units, as the published gap arms read them',
            sources=common, note=note)
        back = load_inputs(OUT / name)
        roundtrip = (len(back['contexts']) == len(group)
                     and all(same(a, it['target']) == 0.0 for a, it in zip(back['contexts'], group))
                     and all(same(a, it['pcov']) == 0.0 for a, it in zip(back['past_cov'], group))
                     and same(back['future_cov'], np.stack([it['fcov'] for it in group])) == 0.0
                     and list(back['keys'].episode) == [it['episode'] for it in group]
                     and bool((back['reverse'] == (direction == 'bwd')).all()))
        entries[name] = {
            'cohort': cohort, 'variable': variable, 'direction': direction, 'gap_length': length,
            'n': meta['n'], 'horizon': meta['horizon'], 'cov_names': meta['cov_names'],
            'context_hours_min': int(min(len(it['target']) for it in group)),
            'contexts_with_bridged_hours': int(sum(it['ctx_missing'] > 0 for it in group)),
            'contexts_zero_filled': int(sum(it['zero_filled'] > 0 for it in group)),
            'neighbours_dropped': int(sum(len(built['cov_names']) - len(it['used']) for it in group)),
            'sha256': digest(OUT / name), 'roundtrip_passed': bool(roundtrip)}
    return entries


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--no-model-gate', action='store_true',
                        help='skip gate E5 (Chronos-2 on CPU on the probes with tied neighbours)')
    args = parser.parse_args()
    os.chdir(ROOT)          # sources are recorded, and digested, relative to the repository root
    cache, data, gates = {}, {}, {}
    for cohort in COHORTS:
        for variable in VARIABLES:
            built = build(cohort, variable, cache)
            data[(cohort, variable)] = built
            cell = {'E1_rows': built['e1']}
            if cohort == 'Gap-2026':
                cell['E2_adapt'] = gate_adapt(cohort, variable, built)
            if cohort in ('Gap-Dev', 'Gap-2026'):
                cell['E3_holes'] = gate_holes(cohort, variable, built)
            cell['E4_function'] = gate_function(cohort, variable, built)
            gates[f'{cohort} {variable}'] = cell
            print(cohort, variable, json.dumps({k: v.get('passed') for k, v in cell.items()}),
                  flush=True)
    gates['E5_model'] = gate_model(data, not args.no_model_gate)
    print('E5', json.dumps(gates['E5_model']), flush=True)
    failed = [f'{c} {g}' for c, cell in gates.items() if c != 'E5_model'
              for g, v in cell.items() if not v.get('passed')]
    if gates['E5_model'].get('state') == 'run' and not gates['E5_model']['passed']:
        failed.append('E5_model')
    if failed:
        raise SystemExit(f'gates failed, nothing written: {failed}')

    OUT.mkdir(parents=True, exist_ok=True)
    files = {}
    for (cohort, variable), built in data.items():
        files.update(write(cohort, variable, built))
    bad = [n for n, e in files.items() if not e['roundtrip_passed']]
    gates['E6_roundtrip'] = {'passed': not bad, 'files': len(files), 'failed': bad}
    if bad:
        raise SystemExit(f'round trip failed: {bad}')

    manifest = {
        'declared_in': 'config/bay_reserved_protocol_v2_addendum81.json',
        'task': 'E_gaps',
        'builder': 'scripts/v2_fm_zoo_inputs_gaps.py',
        'format': 'scripts/v2_fm_zoo_io.py (save_inputs)',
        'layout': 'one file per (variable, cohort, direction, gap length): E_{variable}_{cohort}_'
                  '{fwd|bwd}_L{length}.npz; horizon = gap length',
        'keys': {'episode': 'join key of the published rows (sensor|gap start|length)',
                 'gap_start': 'first hour of the gap; a row at position p is at gap_start + p - 1 h',
                 'direction': 'fwd: step k forecasts position k + 1; bwd: step k forecasts '
                              'position L - k',
                 'edge_time': 'the context hour next to the gap (fwd: last before; bwd: first after)',
                 'ctx_missing': 'hours of the context window bridged by interpolation',
                 'n_scored': 'published rows of the episode',
                 'nbC_sensor': 'the neighbour named nbC (published order)',
                 'cov_used': 'neighbours the published Chronos-2 arm passed (others dropped)'},
        'cohorts': {c: {'period': COHORT_PERIOD[c], 'chronos_bolt_run': COHORTS[c][0],
                        'chronos2_run': COHORTS[c][2], 'results': COHORTS[c][1],
                        **{v: {'episodes': data[(c, v)]['e1']['episodes'],
                               'rows': data[(c, v)]['e1']['rows'],
                               'probes_in_panel': data[(c, v)]['probes'],
                               'cov_names': list(data[(c, v)]['cov_names'])} for v in VARIABLES}}
                    for c in COHORTS},
        'neighbour_order': {
            'rule': 'four nearest other probes, nearest first, by numpy default argsort with the '
                    'probe excluded by index (chronos_fill.neighbour_covariates); ties between '
                    'co-located probes break differently from nearest_others (stable sort)',
            'differs_from_nearest_others': {f'{c} {v}': d['order_differs']
                                            for (c, v), d in data.items() if d['order_differs']}},
        'covariates': {
            'past': 'raw neighbour readings over the context window, NaN kept, reversed for bwd',
            'future': 'raw neighbour readings through the gap, NaN kept, reversed for bwd',
            'published_missing_rule': 'NaN passed to Chronos-2 as NaN; a neighbour with no finite '
                                      'value over the context window or none through the gap '
                                      'dropped from that item (cov_used)'},
        'files': files,
        'totals': {'files': len(files), 'contexts': int(sum(e['n'] for e in files.values()))},
        'gates': gates,
    }
    MANIFEST.write_bytes((json.dumps(manifest, indent=2) + '\n').encode('utf-8'))
    print(len(files), 'files,', manifest['totals']['contexts'], 'contexts; wrote', MANIFEST)


if __name__ == '__main__':
    main()
