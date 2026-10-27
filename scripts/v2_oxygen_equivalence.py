"""Addendum 60 part B: is a published oxygen result times c the same as refitting it in mg/l?

For each of the seven declared arms this refits the published configuration through the arm's own
producer code, never a copy of it:

  umol    as published, oxygen per kilogram; the gate: the refit reproduces the published values
  mgl     the declared refit (amendment 1): every oxygen value in mg/l, the sondes hour by hour by
          0.0319988 x rho/1000 (runs/v2_oxygen_units/hourly_factor.csv.gz; c where an hour has no
          density), per-kilogram cast bins by 0.0319988 x rho/1000 with the bin's TEOS-10 density,
          per-litre cast bins (mg/l sources and the one percent-saturation cast, which was formed
          from the per-litre solubility) by 0.0319988
  mgl_c   diagnostic only: every oxygen value times the one constant c, so a deviation that
          survives it comes from the model, not from the per-reading factor
  mgl_constants_as_coded   B3 only, a sensitivity: as mgl, but the forecasting grid's z-score
          constants left at the 230 and 50 written in v2_horizon.build
  umol_ulp0-4    B1 and B2, diagnostic: the published unit with one float32 ulp added to a random
          9.4 % of the grid's own-trend feature (the smallest change a refit can see)
  umol_nudge0-4  B7, diagnostic: the published unit with one float64 ulp added to a random 9.4 % of
          the sondes' hourly oxygen readings
  b5-kernels     B5, diagnostic: the Gaussian process's fitted hyperparameters per unit and how
          often each sits on a bound of the published kernel
  b5-consistent  B5 again with the kernel's amplitudes and noise level (initial values and bounds),
          and in a second variant also alpha, times c^2; under mgl_c and mgl (not declared)

and compares the mg/l values with c x published under the declared criterion: an MAE passes when
|refit - c x published| <= 0.5 % of c x published, a contrast when |refit - c x published| <= 0.5 %
of c x the reference arm's published MAE.

  B1  plain and weather tree, oxygen 6 h, Dev-2025, published shared selection     (local)
  B2  XGBoost weather tree, oxygen 6 h, Dev-2025, against the LightGBM weather tree (cluster, CPU)
  B3  learned-source graph arm, seeds 42, 7 and 123, against the plain tree        (cluster, GPU)
  B4  Field-57 arm A2 (no boundary product), oxygen nested MAE                     (local)
  B5  per-pattern Gaussian-process filler, Gap-Dev 12-72 h, against linear         (cluster, CPU)
  B6  bidirectional zero-shot Chronos-Bolt, Gap-Dev 12-72 h, against linear        (cluster, GPU)
  B7  anchored continuation, Bridge-Dev 6/24/48 h, against the hour-of-day increment (local)

How the unit enters. No producer is edited. For a mg/l refit the loaders the producers call
(`load_observations`, `load_profiles`) are wrapped to return converted values. Two data constants
carry the oxygen unit and are converted by c: the z-score centre and scale of the forecasting grid
(230 and 50 umol/kg in v2_horizon.build, re-expressed after the build) and the Field-57 target
scale (20 umol/kg; the ridge is linear in its target, so this changes nothing but the units). The
bridge's solubility-change feature is formed in mg/l. Every selected hyperparameter, seed, fold,
mask, row and model setting is the published one; the Gaussian process keeps normalize_y=False and
its kernel bounds unrescaled, because the question is whether the published configuration gives the
converted result.

Usage (repository root, V2_THREADS=2 unless stated):
  python -m scripts.v2_oxygen_equivalence            the local arms B1, B4, B7, then the report
  python -m scripts.v2_oxygen_equivalence b1 ... b7  one arm (B2 with V2_THREADS=4, as published)
  python -m scripts.v2_oxygen_equivalence b5-kernels after b5, on the same host
  python -m scripts.v2_oxygen_equivalence report     runs/v2_oxygen_mgl/equivalence.json
The cluster arms are packed, launched and fetched by scripts/experiments/v2_oxygen_equivalence_remote.py.
Declared in config/bay_reserved_protocol_v2_addendum60.json and its amendment 1.
"""
import os
os.environ.setdefault('V2_THREADS', '2')

import argparse
import hashlib
import json
import platform
import sys
import time
import zlib
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
RUNS = ROOT / 'reports/bay_twin/runs'
OUT = RUNS / 'v2_oxygen_mgl/equivalence'
REPORT = RUNS / 'v2_oxygen_mgl/equivalence.json'
UNITS_JSON = RUNS / 'v2_oxygen_units.json'
HOURLY = RUNS / 'v2_oxygen_units/hourly_factor.csv.gz'
DECLARED = ['config/bay_reserved_protocol_v2_addendum60.json',
            'config/bay_reserved_protocol_v2_addendum60_amendment1.json']
TOLERANCE = 0.005          # the declared criterion, as a share of c x published (or of the reference)
PB2_BOUND = 0.002          # PB2: the largest relative MAE deviation below 0.2 %
ARMS = ('B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7')
LOCAL = ('B1', 'B4', 'B7')
GZIP = {'method': 'gzip', 'mtime': 0}
BAND = (12, 72)
KEY = ['fold', 'sensor', 'time']

_UNITS = json.loads(UNITS_JSON.read_text())
C = float(_UNITS['c_mg_per_l_per_umol_kg'])
PER_LITRE = float(_UNITS['mg_per_l_per_umol_l'])

#: the published inputs each arm reads (relative to RUNS)
GP_PUBLISHED = '20261001T110649Z_v2_gaprecovery/v2_gap_recovery/oxygen_gpr/predictions.csv.gz'
CHRONOS_PUBLISHED = '20260922T171859Z_v2_gaprecovery/v2_gap_recovery/oxygen_chronos_bidir/predictions.csv.gz'
GAP_VARIANTS = 'v2_gap_variants.json'


# ------------------------------------------------------------------------------------ conversion

_MODE = {'unit': 'umol'}
_FACTOR = {}
_AUDIT = {}


def hourly_factor():
    if 'table' not in _FACTOR:
        f = pd.read_csv(HOURLY)
        f['time'] = pd.to_datetime(f.time, utc=True)
        _FACTOR['table'] = f.pivot(index='time', columns='sensor', values='factor')
    return _FACTOR['table']


def convert_frames(frames):
    """Each sonde's oxygen in mg/l: its own hour's factor, c where the hour has no density."""
    table = hourly_factor()
    out, audit = {}, {}
    for name, frame in frames.items():
        if 'oxygen' not in frame or not frame['oxygen'].notna().any():
            out[name] = frame
            continue
        frame = frame.copy()
        observed = frame['oxygen'].notna()
        if _MODE['unit'] == 'mgl_c':
            factor = pd.Series(C, index=frame.index)
            own = pd.Series(False, index=frame.index)
        else:
            factor = (table[name].reindex(frame.index) if name in table.columns
                      else pd.Series(np.nan, index=frame.index))
            own = factor.notna()
            factor = factor.where(own, C)
        frame['oxygen'] = frame['oxygen'] * factor
        audit[name] = {'oxygen_hours': int(observed.sum()),
                       'hours_own_factor': int((observed & own).sum()),
                       'hours_constant_c': int((observed & ~own).sum())}
        out[name] = frame
    _AUDIT['sondes'] = audit
    return out


def convert_casts(d):
    """Cast bins in mg/l: per-kilogram bins by 0.0319988 x rho/1000 (TEOS-10, depth as dbar),
    per-litre bins (mg/l sources, and the percent cast formed from the per-litre solubility) by
    0.0319988; under mgl_c every bin by c."""
    from src.data.thermodynamics import seawater_density_kg_per_m3
    d = d.copy()
    source = d['do_unit_source'].astype(str)
    unknown = sorted(set(source) - {'umol/l', 'mg/l', '%'})
    if unknown:
        raise SystemExit(f'cast oxygen source unit not declared: {unknown}')
    if _MODE['unit'] == 'mgl_c':
        factor = np.full(len(d), C)
    else:
        rho = seawater_density_kg_per_m3(d.temp_c.to_numpy(float), d.salinity_psu.to_numpy(float),
                                         d.depth_m.to_numpy(float))
        factor = np.where((source == 'umol/l').to_numpy(), PER_LITRE * np.asarray(rho) / 1000, PER_LITRE)
    for col in ('do_umol_l', 'sigma_bin_umol_l', 'sigma_obs_umol_l'):
        if col in d:
            d[col] = d[col] * factor
    if 'o2_sol_umol_l' in d:        # the per-litre Garcia-Gordon solubility
        d['o2_sol_umol_l'] = d['o2_sol_umol_l'] * (C if _MODE['unit'] == 'mgl_c' else PER_LITRE)
        if 'aou_umol_l' in d:
            d['aou_umol_l'] = d['o2_sol_umol_l'] - d['do_umol_l']
    casts = d.groupby('cast_id').do_unit_source.first().value_counts().to_dict()
    _AUDIT['casts'] = {'casts_by_source_unit': {str(k): int(v) for k, v in casts.items()},
                       'factor_min': float(np.min(factor)), 'factor_max': float(np.max(factor))}
    return d


#: share of the grid's float32 own-trend entries nudged by one unit in the last place in the
#: `umol_ulp<k>` diagnostic: what a post-hoc float32 rescaling of that column changed (9.4 %)
ULP_SHARE = 0.094
ULP_DRAWS = 5
UNIT_MODES = (('umol', 'mgl', 'mgl_c', 'mgl_constants_as_coded') + tuple(f'umol_ulp{k}' for k in range(ULP_DRAWS))
              + tuple(f'umol_nudge{k}' for k in range(ULP_DRAWS)))
DIAGNOSTIC_PREFIXES = ('umol_ulp', 'umol_nudge')


@contextmanager
def _reading_nudge(k):
    """Diagnostic: the published unit, with one float64 ulp added to a random ULP_SHARE of every
    sonde's hourly oxygen readings (stream k), the same readings at every load."""
    import src.data.bay_profile_context as bpc
    original = bpc.load_observations

    def load_observations():
        meta, frames = original()
        out = {}
        for j, (name, frame) in enumerate(frames.items()):
            if 'oxygen' in frame.columns and frame['oxygen'].notna().any():
                rng = np.random.default_rng([20261002, k, j])
                values = frame['oxygen'].to_numpy(float).copy()
                pick = (rng.random(len(values)) < ULP_SHARE) & np.isfinite(values)
                values[pick] = np.nextafter(values[pick], np.inf)
                frame = frame.assign(oxygen=values)
            out[name] = frame
        return meta, out
    saved = [(m, getattr(m, 'load_observations')) for m in list(sys.modules.values())
             if getattr(m, 'load_observations', None) is original]
    for m, _ in saved:
        m.load_observations = load_observations
    try:
        yield
    finally:
        for m, value in saved:
            m.load_observations = value


@contextmanager
def _ulp_nudge(k):
    """Diagnostic: the published unit, with one float32 ulp added to a random ULP_SHARE of the grid's
    own-trend feature (stream k). How far a refit moves under the smallest possible input change."""
    import src.experiments.v2_horizon as hz
    original_build = hz.build

    def build(variable, horizon):
        data = original_build(variable, horizon)
        if variable == 'oxygen':
            rng = np.random.default_rng([20261002, k])
            forcing = data['forcing'].copy()
            trend = forcing[..., 0]
            pick = rng.random(trend.shape) < ULP_SHARE
            trend[pick] = np.nextafter(trend[pick], np.float32(np.inf))
            data['forcing'] = forcing
        return data
    hz.build = build
    try:
        yield
    finally:
        hz.build = original_build


@contextmanager
def oxygen_unit(unit):
    """Run the producers' own code with oxygen in `unit` (see the module docstring)."""
    if unit not in UNIT_MODES:
        raise ValueError(unit)
    _MODE['unit'] = unit
    _AUDIT.clear()
    if unit == 'umol':
        yield
        return
    if unit.startswith('umol_ulp'):
        with _ulp_nudge(int(unit.removeprefix('umol_ulp'))):
            yield
        _MODE['unit'] = 'umol'
        return
    if unit.startswith('umol_nudge'):
        with _reading_nudge(int(unit.removeprefix('umol_nudge'))):
            yield
        _MODE['unit'] = 'umol'
        return
    import src.data.bay_profile_context as bpc
    import src.experiments.bay_profile_field as bpf
    from src.data.thermodynamics import o2sol_garcia_gordon_umol_per_l
    original_obs, original_prof = bpc.load_observations, bpc.load_profiles

    def load_observations():
        meta, frames = original_obs()
        return meta, convert_frames(frames)

    def load_profiles():
        d, excluded = original_prof()
        return convert_casts(d), excluded

    saved = []

    def patch(module, name, value):
        saved.append((module, name, getattr(module, name)))
        setattr(module, name, value)
    # every loaded module holding the original loaders by name (the producers import them by name)
    for module in list(sys.modules.values()):
        if getattr(module, 'load_observations', None) is original_obs:
            patch(module, 'load_observations', load_observations)
        if getattr(module, 'load_profiles', None) is original_prof:
            patch(module, 'load_profiles', load_profiles)
    hz = sys.modules.get('src.experiments.v2_horizon')
    convert_constants = unit in ('mgl', 'mgl_c')
    factor = C
    if hz is not None and convert_constants:
        # v2_horizon.build with its oxygen centre and scale converted, 230 c and 50 c mg/l. They are
        # local constants of the build, so it is run on the mg/l values divided by c with the coded
        # 230 and 50, which gives (r/c - 230)/50 = (r - 230 c)/(50 c) in float64 before any cast, and
        # its centre and scale are then set to 230 c and 50 c, the values rows_from converts back with.
        original_build = hz.build

        def over_c():
            meta, frames = load_observations()
            return meta, {n: (f.assign(oxygen=f['oxygen'] / factor) if 'oxygen' in f.columns else f)
                          for n, f in frames.items()}

        def build(variable, horizon):
            if variable != 'oxygen':
                return original_build(variable, horizon)
            hz.load_observations = over_c
            try:
                data = original_build(variable, horizon)
            finally:
                hz.load_observations = load_observations
            data['center'], data['scale'] = data['center'] * factor, data['scale'] * factor
            return data
        patch(hz, 'build', build)
    bridge = sys.modules.get('src.experiments.v2_bridge')
    if bridge is not None:
        patch(bridge, 'o2sol', lambda t, s: o2sol_garcia_gordon_umol_per_l(t, s) * PER_LITRE)
    scale = bpf.SCALE
    old = float(scale[2])
    if convert_constants:
        scale[2] = old * factor          # in place: v2_field.SCALE is the same array
    try:
        yield
    finally:
        scale[2] = old
        for module, name, value in reversed(saved):
            setattr(module, name, value)
        _MODE['unit'] = 'umol'


# ------------------------------------------------------------------------------------ helpers

def _json_write(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes((json.dumps(obj, indent=2, allow_nan=False) + '\n').encode('utf-8'))


def _csv(frame, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, compression=GZIP, lineterminator='\n')
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _rel(path):
    return Path(path).resolve().relative_to(ROOT).as_posix()


def _host():
    out = {'host': platform.node(), 'platform': platform.platform(), 'python': platform.python_version(),
           'V2_THREADS': os.environ.get('V2_THREADS'), 'OMP_NUM_THREADS': os.environ.get('OMP_NUM_THREADS'),
           'CUDA_VISIBLE_DEVICES': os.environ.get('CUDA_VISIBLE_DEVICES')}
    for pkg in ('numpy', 'pandas', 'scikit-learn', 'lightgbm', 'xgboost', 'torch', 'chronos-forecasting'):
        try:
            from importlib.metadata import version
            out[pkg] = version(pkg)
        except Exception:                                   # noqa: BLE001 - absent package recorded
            out[pkg] = None
    return out


def forecast_macro(d, col='prediction'):
    e = (d.truth - d[col]).abs()
    return float(d.assign(e=e).groupby(['fold', 'sensor']).e.mean().groupby('fold').mean().mean())


def episode_mae(frame, col):
    d = frame.dropna(subset=[col, 'truth'])
    return float(d.assign(e=(d.truth - d[col]).abs()).groupby('episode').e.mean().mean())


def episode_contrast(frame, a, b):
    """v2_gap_intervals.paired's point estimate, unrounded: per-episode means on rows where both exist."""
    d = frame.dropna(subset=[a, b, 'truth'])
    per = (d.assign(ea=(d.truth - d[a]).abs(), eb=(d.truth - d[b]).abs())
           .groupby('episode')[['ea', 'eb']].mean())
    return float(per.ea.mean() - per.eb.mean())


def mae_q(name, published, values, source, role='arm'):
    """`role` is 'arm' for the arm's own MAE and 'reference' for the comparand a contrast is set against."""
    return {'name': name, 'kind': 'mae', 'role': role, 'published': published, 'published_source': source,
            **values}


def contrast_q(name, published, values, reference, reference_published, source):
    return {'name': name, 'kind': 'contrast', 'published': published, 'published_source': source,
            'reference': reference, 'reference_published_mae': reference_published, **values}


def units_for(arm, units):
    """The published unit first (the gate), then the declared mg/l refit, then the diagnostics: every
    value times c; for the graph arm the z-score constants left as coded; for the two forecasting tree
    arms, whose grid carries a float32 feature, five one-ulp nudges of it in the published unit."""
    if units:
        return list(units)
    extra = {'B3': ['mgl_constants_as_coded'],
             'B1': [f'umol_ulp{k}' for k in range(ULP_DRAWS)],
             'B2': [f'umol_ulp{k}' for k in range(ULP_DRAWS)],
             'B7': [f'umol_nudge{k}' for k in range(ULP_DRAWS)]}
    return ['umol', 'mgl', 'mgl_c'] + extra.get(arm, [])


def finish(arm, report, started):
    report.update(arm=arm, declared_in=DECLARED, c_mg_per_l_per_umol_kg=C, mg_per_l_per_umol_l=PER_LITRE,
                  seconds=round(time.time() - started, 1), compute=_host(),
                  producer_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    _json_write(OUT / arm / 'report.json', report)
    print(f'{arm}: wrote {_rel(OUT / arm / "report.json")}', flush=True)


def gate_failed(arm, report, started):
    report['stopped'] = 'the refit in the published unit did not reproduce the published values'
    finish(arm, report, started)
    print(f'{arm}: GATE FAILED, stopped', flush=True)


# ------------------------------------------------------------------------------------ B1, B2

def arm_b1(units=None):
    """Plain and weather tree, oxygen 6 h, through addendum 47's refit of the published selection."""
    from src.experiments import v2_perfold as a47
    import src.experiments.v2_horizon  # noqa: F401 - loaded before any patch, so build is wrapped
    started = time.time()
    arms = ('tree', 'weather_tree')
    published, pub_rows, sources = {}, {}, {}
    for arm in arms:
        folder = a47.HORIZON_RUN / f'v2_horizon_oxygen_{arm}_h6'
        published[arm] = json.loads((folder / 'report.json').read_text())['mean_forecast_mae']
        pub_rows[arm] = pd.read_csv(folder / 'selected_predictions.csv.gz')
        sources[arm] = _rel(folder / 'report.json')
    report = {'what': 'B1: plain and weather LightGBM residual trees, oxygen, 6 h, Dev-2025, the published '
                      'shared selection (v2_perfold.forecast_setup / score_folds)',
              'cohort': 'Dev-2025', 'units': {}, 'gate': {}}
    values = {arm: {} for arm in arms}
    for unit in units_for('B1', units):
        entry = {}
        with oxygen_unit(unit):
            for arm in arms:
                data, outer, inner, clim, cache = a47.forecast_setup('oxygen', arm, 6)
                params, sel, source = a47.published_forecast_selection('oxygen', arm, 6)
                d = a47.score_folds(data, outer, {f['number']: params for f in outer}, clim, cache)
                sha = _csv(d, OUT / 'B1' / unit / f'oxygen_{arm}_h6' / 'selected_predictions.csv.gz')
                entry[arm] = {'mae': forecast_macro(d), 'persistence_mae': forecast_macro(d, 'persistence'),
                              'endpoints': int(len(d)), 'params': params, 'selection_source': source,
                              'center': float(data['center']), 'scale': float(data['scale']),
                              'prediction_sha256': sha}
                values[arm][unit] = entry[arm]['mae']
                if unit == 'umol':
                    pub = pub_rows[arm]
                    j = pub.merge(d, on=KEY, suffixes=('_pub', ''), validate='one_to_one')
                    same = len(j) == len(pub) == len(d)
                    diff = float(np.abs(j.prediction - j.prediction_pub).max())
                    other = {c: float(np.abs(j[c] - j[c + '_pub']).max()) for c in ('truth', 'persistence')}
                    report['gate'][arm] = {
                        'rule': 'rows identical, every prediction and truth within 1e-9 of the published rows',
                        'rows_identical': bool(same), 'max_abs_prediction_diff': diff,
                        'other_columns_max_abs_diff': other,
                        'mae_abs_diff': abs(entry[arm]['mae'] - published[arm]),
                        'pass': bool(same and diff <= 1e-9 and max(other.values()) <= 1e-9)}
            entry['conversion'] = dict(_AUDIT)
        report['units'][unit] = entry
        if unit == 'umol' and not all(g['pass'] for g in report['gate'].values()):
            return gate_failed('B1', report, started)
    report['gate']['pass'] = all(g['pass'] for g in report['gate'].values())
    contrast = {u: values['weather_tree'][u] - values['tree'][u] for u in values['tree']}
    report['quantities'] = [
        mae_q('plain tree MAE', published['tree'], values['tree'], sources['tree']),
        mae_q('weather tree MAE', published['weather_tree'], values['weather_tree'], sources['weather_tree']),
        contrast_q('weather tree - plain tree', published['weather_tree'] - published['tree'], contrast,
                   'plain tree MAE', published['tree'], 'difference of the two published MAEs')]
    finish('B1', report, started)


def arm_b2(units=None):
    """XGBoost weather tree, oxygen 6 h, through addendum 48's refit, at the published thread count."""
    from src.experiments import v2_perfold_rest as a48
    import src.experiments.v2_horizon as hz
    from src.experiments.v2_forecast import THREADS
    started = time.time()
    source, stem = a48.published_folder('xgb', 'oxygen', 'xgb_weather_tree', 6)
    sel = json.loads((source / f'{stem}_selection.json').read_text())
    params, _ = a48.to_params('xgb', sel['params'])
    published = json.loads((source / 'report.json').read_text())['mean_forecast_mae']
    pub_rows = pd.read_csv(source / 'selected_predictions.csv.gz')
    report = {'what': 'B2: XGBoost residual tree with weather forcing, oxygen, 6 h, Dev-2025, the published '
                      'configuration (v2_perfold_rest design_builder / v2_horizon.tree_fit_score)',
              'cohort': 'Dev-2025', 'threads': THREADS, 'params': params,
              'published_threads': 4, 'units': {}, 'gate': {}}
    values = {}
    for unit in units_for('B2', units):
        with oxygen_unit(unit):
            data = hz.build('oxygen', 6)
            outer = hz.folds(data, 6)
            clim = {id(f): hz.climatology(data, f) for f in outer}
            design = a48.design_builder('xgb', 'xgb_weather_tree', data)
            X = {id(f): design(f) for f in outer}
            d = pd.concat([hz.tree_fit_score(data, X[id(f)], f, params, clim[id(f)])[0] for f in outer],
                          ignore_index=True)
            sha = _csv(d, OUT / 'B2' / unit / 'oxygen_xgb_weather_tree_h6' / 'selected_predictions.csv.gz')
            values[unit] = forecast_macro(d)
            report['units'][unit] = {'mae': values[unit], 'endpoints': int(len(d)), 'prediction_sha256': sha,
                                     'center': float(data['center']), 'scale': float(data['scale']),
                                     'conversion': dict(_AUDIT)}
            if unit == 'umol':
                j = pub_rows.merge(d, on=KEY, suffixes=('_pub', ''), validate='one_to_one')
                same = len(j) == len(pub_rows) == len(d)
                diff = float(np.abs(j.prediction - j.prediction_pub).max())
                other = {c: float(np.abs(j[c] - j[c + '_pub']).max()) for c in ('truth', 'persistence')}
                stored = a48.xgb_stored_gate(data, outer, lambda f: X[id(f)], source)
                again = pd.concat([hz.tree_fit_score(data, X[id(f)], f, params, clim[id(f)])[0] for f in outer],
                                  ignore_index=True)
                repeat = float(np.abs(again.prediction.to_numpy() - d.prediction.to_numpy()).max())
                report['gate'] = {
                    'rule': 'the refit at the published thread count reproduces the published rows within 1e-9 '
                            '(XGBoost row subsampling depends on threads and platform, addendum 48 amendment 1)',
                    'rows_identical': bool(same), 'max_abs_prediction_diff': diff,
                    'other_columns_max_abs_diff': other, 'mae_abs_diff': abs(values[unit] - published),
                    'stored_models_max_abs_diff': stored, 'repeat_refit_max_abs_diff': repeat,
                    'pass': bool(same and diff <= 1e-9 and max(other.values()) <= 1e-9)}
                if not report['gate']['pass']:
                    return gate_failed('B2', report, started)
    report['quantities'] = [mae_q('XGBoost weather tree MAE', published, values, _rel(source / 'report.json'))]
    finish('B2', report, started)


# ------------------------------------------------------------------------------------ B3

def arm_b3(units=None):
    """The learned-source graph arm, oxygen 6 h, seeds 42, 7, 123, through addendum 49's refit."""
    from src.experiments import v2_perfold_neural as a49
    from src.experiments.v2_forecast import SEEDS
    from src.experiments.v2_horizon import neural_forecast
    import src.experiments.v2_horizon  # noqa: F401
    started = time.time()
    pub_folder, record = a49.published_paths('oxygen', 'learned_source', 6)
    sel = json.loads(record.read_text())
    params = sel['params']
    pub = a49.published_rows(pub_folder)
    # the published rows carry no clim_hour column, so the mean of the networks is formed here
    pub_ens = pub.groupby(KEY, as_index=False).agg(truth=('truth', 'first'), prediction=('prediction', 'mean'))
    published = a49._macro(pub_ens)
    pub_report = json.loads((pub_folder / 'report.json').read_text())
    pub_seed = {int(s): m['forecast_prediction_mae'] for s, m in pub_report['per_seed_metrics'].items()}
    report = {'what': 'B3: learned-source graph dynamics (no explicit exchange), oxygen, 6 h, Dev-2025, the '
                      'published configuration, networks at seeds 42, 7 and 123 and their mean '
                      '(v2_perfold_neural.graph_setup / v2_horizon.neural_forecast)',
              'cohort': 'Dev-2025', 'params': params, 'seeds': list(SEEDS),
              'published_mae_definition': 'macro MAE of the mean of the three networks (Table 5.x prints 6.5607)',
              'units': {}, 'gate': {}}
    values, seed_values = {}, {s: {} for s in SEEDS}
    for unit in units_for('B3', units):
        t0 = time.time()
        with oxygen_unit(unit):
            data, outer, inner, clim, exchange, relaxation, device = a49.graph_setup('oxygen', 'learned_source', 6)
            d = pd.concat([neural_forecast(data, f, params, exchange, relaxation, s, device, clim[id(f)])
                           .assign(seed=s) for s in SEEDS for f in outer], ignore_index=True)
            conversion = dict(_AUDIT)
        sha = _csv(d, OUT / 'B3' / unit / 'oxygen_learned_source_h6' / 'selected_predictions.csv.gz')
        ens = a49.seed_mean(d)
        values[unit] = a49._macro(ens)
        per_seed = {int(s): a49._macro(g.drop(columns='seed')) for s, g in d.groupby('seed')}
        for s in SEEDS:
            seed_values[s][unit] = per_seed[int(s)]
        report['units'][unit] = {'mae_seed_mean_prediction': values[unit], 'mae_per_seed': per_seed,
                                 'endpoints': int(len(ens)), 'device': device, 'prediction_sha256': sha,
                                 'center': float(data['center']), 'scale': float(data['scale']),
                                 'seconds': round(time.time() - t0, 1), 'conversion': conversion}
        if unit == 'umol':
            j = pub.merge(d, on=KEY + ['seed'], suffixes=('_pub', ''), how='outer', indicator=True)
            same = bool((j['_merge'] == 'both').all()) and len(pub) == len(d)
            both = j[j['_merge'] == 'both']
            diff = float(np.abs(both.prediction - both.prediction_pub).max())
            other = {c: float(np.abs(both[c] - both[c + '_pub']).max()) for c in ('truth', 'persistence')}
            report['gate'] = {'rule': 'rows identical, predictions within 1e-6 and truth within 1e-9 of the '
                                      'published rows (addendum 49, gate G1)',
                              'rows_identical': same, 'max_abs_prediction_diff': diff,
                              'other_columns_max_abs_diff': other, 'mae_abs_diff': abs(values[unit] - published),
                              'per_seed_mae_abs_diff': {int(s): abs(per_seed[int(s)] - pub_seed[int(s)]) for s in SEEDS},
                              'pass': bool(same and diff <= 1e-6 and max(other.values()) <= 1e-9)}
            if not report['gate']['pass']:
                return gate_failed('B3', report, started)
    source = _rel(pub_folder / 'selected_predictions.csv.gz')
    report['quantities'] = [mae_q('graph arm MAE (three-network mean)', published, values, source)] + [
        mae_q(f'graph arm MAE, seed {s}', pub_seed[int(s)], seed_values[s], _rel(pub_folder / 'report.json'))
        for s in SEEDS]
    finish('B3', report, started)


# ------------------------------------------------------------------------------------ B4

def arm_b4(units=None):
    """Field-57 arm A2 (sonde context only, no boundary product), oxygen, through addendum 47's refit."""
    from src.experiments import v2_perfold as a47
    from src.experiments.bay_profile_field import prediction_frame
    from src.experiments.v2_field import fit_predict, nested_mae
    started = time.time()
    arm = 'A2_none_direct'
    sel, pub_report, pub_folder = a47.published_field(arm)
    alpha = sel['params']['alpha']
    pub = pd.read_csv(pub_folder / 'selected_predictions.csv.gz')
    published = pub_report['nested_mae']['do_umol_l']
    report = {'what': 'B4: Field-57 held-station ridge without a boundary product (A2_none_direct), oxygen '
                      'nested MAE (casts, stations, folds), the published penalty (v2_perfold.field_setup / '
                      'v2_field.fit_predict)', 'cohort': 'Field-57', 'alpha': alpha, 'units': {}, 'gate': {}}
    values = {}
    for unit in units_for('B4', units):
        with oxygen_unit(unit):
            bundles, contracts, mats, common = a47.field_setup([arm])
            rows = []
            for b, (base, y, feats) in zip(bundles, mats[arm]):
                pred, _, _ = fit_predict(feats, base, y, b['tr'], b['va'], np.asarray(b['weights'], float),
                                         alpha, False)
                rows.append(prediction_frame(b, pred))
            from src.experiments.bay_profile_field import SCALE
            target_scale = float(SCALE[2])
            conversion = dict(_AUDIT)
        d = pd.concat(rows, ignore_index=True)
        sha = _csv(d, OUT / 'B4' / unit / arm / 'selected_predictions.csv.gz')
        m = nested_mae(d)
        values[unit] = m['do_umol_l']
        scored = d.groupby('cast_id').size()
        report['units'][unit] = {'nested_mae': m, 'rows': int(len(d)), 'casts': int(d.cast_id.nunique()),
                                 'oxygen_target_scale': target_scale, 'prediction_sha256': sha,
                                 'conversion': conversion,
                                 'note': 'column do_umol_l holds the unit of this run'}
        if unit == 'umol':
            cols = [c for c in pub.columns if c.startswith('pred_')]
            same = list(pub.columns) == list(d.columns) and len(pub) == len(d) and \
                (pub[['cast_id', 'depth_m', 'fold']].to_numpy() == d[['cast_id', 'depth_m', 'fold']].to_numpy()).all()
            a, b_ = pub[cols].to_numpy(float), d[cols].to_numpy(float)
            rel = float(np.nanmax(np.abs(a - b_) / np.maximum(np.abs(a), 1e-12)))
            mae_diff = max(abs(m[t] - pub_report['nested_mae'][t]) for t in m)
            report['gate'] = {'rule': 'rows identical, predictions within 1e-6 relative and nested MAE within 1e-6 '
                                      'of the published (addendum 48, gate G4)',
                              'rows_identical': bool(same), 'max_relative_prediction_diff': rel,
                              'nested_mae_abs_diff': mae_diff, 'pass': bool(same and rel <= 1e-6 and mae_diff <= 1e-6)}
            if not report['gate']['pass']:
                return gate_failed('B4', report, started)
            # which casts are scored, by the unit their oxygen came in
            from src.data.bay_profile_context import load_profiles
            profiles, _ = load_profiles()
            unit_of = profiles.groupby('cast_id').do_unit_source.first()
            report['scored_casts_by_source_unit'] = {str(k): int(v) for k, v in
                                                     unit_of.reindex(scored.index).value_counts().items()}
            report['scored_per_litre_casts'] = sorted(c for c in scored.index if unit_of.get(c) != 'umol/l')
        if unit == 'mgl' and report.get('scored_per_litre_casts'):
            # Attribution, not part of the declared test: the same mg/l rows with the per-litre casts'
            # truth converted as if they were per kilogram (0.0319988 x rho/1000, TEOS-10).
            from src.data.thermodynamics import seawater_density_kg_per_m3
            alt = d.copy()
            pick = alt.cast_id.isin(report['scored_per_litre_casts'])
            rho = np.asarray(seawater_density_kg_per_m3(alt.loc[pick, 'temp_c'].to_numpy(float),
                                                        alt.loc[pick, 'salinity_psu'].to_numpy(float),
                                                        alt.loc[pick, 'depth_m'].to_numpy(float)))
            alt.loc[pick, 'do_umol_l'] = alt.loc[pick, 'do_umol_l'] * rho / 1000
            got = nested_mae(alt)['do_umol_l']
            report['attribution_per_litre_casts'] = {
                'casts': report['scored_per_litre_casts'], 'rows': int(pick.sum()),
                'mgl_nested_mae_with_them_as_per_kg': got,
                'relative_deviation_from_c_x_published': got / (C * published) - 1,
                'note': 'the percent cast was formed from the per-litre solubility, so the declared mg/l '
                        'conversion is per litre; this line shows what the deviation is without that cast\'s '
                        'unit difference'}
    report['quantities'] = [mae_q('A2 oxygen nested MAE', published, values, _rel(pub_folder / 'report.json'))]
    finish('B4', report, started)


# ------------------------------------------------------------------------------------ B5, B6

def gap_band():
    """The Gap-Dev oxygen episodes exactly as v2_gap_recovery.build draws them, cut to 12-72 h.
    Every length is built, because the episode draw consumes one random stream across lengths."""
    import src.experiments.v2_gap_recovery as gr
    frame = gr.build('oxygen', np.random.default_rng(gr.SEED))
    band = frame[(frame.gap_length >= BAND[0]) & (frame.gap_length <= BAND[1])].copy()
    return band


def _gap_arm(name, arm, column, fill, published_path, artifact_keys, units):
    import src.experiments.v2_gap_recovery as gr
    started = time.time()
    if (gr.PERIOD, gr.DESIGN, gr.SEED) != ('development', 'strict', 20260922):
        raise SystemExit('the Gap-Dev cohort needs the default V2_GAP_* environment')
    pub_all = pd.read_csv(RUNS / published_path)
    pub = pub_all[(pub_all.gap_length >= BAND[0]) & (pub_all.gap_length <= BAND[1])].copy()
    published = {'arm': episode_mae(pub, column), 'linear': episode_mae(pub, 'linear'),
                 'contrast': episode_contrast(pub, column, 'linear')}
    variants = json.loads((RUNS / GAP_VARIANTS).read_text())['results']['Gap-Dev']['oxygen']['12-72']
    stored = {'arm': variants[column]['mae'], 'linear': variants['linear']['mae'],
              'contrast': variants[column]['minus_linear']['delta']}
    report = {'what': arm, 'cohort': 'Gap-Dev', 'band_hours': list(BAND),
              'published_rows': int(len(pub)), 'published_episodes': int(pub.episode.nunique()),
              'published_full_precision': published, 'published_artifact_4dp': stored,
              'published_matches_artifact': bool(all(abs(round(published[k], 4) - stored[k]) < 1e-9 for k in stored)),
              'units': {}, 'gate': {}, **artifact_keys}
    values = {'arm': {}, 'linear': {}, 'contrast': {}}
    for unit in units_for(name, units):
        t0 = time.time()
        with oxygen_unit(unit):
            band = gap_band()
            band[column] = fill(band)
            conversion = dict(_AUDIT)
        keep = [c for c in pub_all.columns if c in band.columns]
        sha = _csv(band[keep], OUT / name / unit / f'oxygen_{column}' / 'predictions.csv.gz')
        values['arm'][unit] = episode_mae(band, column)
        values['linear'][unit] = episode_mae(band, 'linear')
        values['contrast'][unit] = episode_contrast(band, column, 'linear')
        report['units'][unit] = {'mae': values['arm'][unit], 'linear_mae': values['linear'][unit],
                                 'minus_linear': values['contrast'][unit], 'rows': int(len(band)),
                                 'episodes': int(band.episode.nunique()),
                                 'rows_without_prediction': int(band[column].isna().sum()),
                                 'prediction_sha256': sha, 'seconds': round(time.time() - t0, 1),
                                 'conversion': conversion}
        if unit == 'umol':
            j = pub.merge(band[['episode', 'position', 'truth', 'linear', column]], on=['episode', 'position'],
                          how='outer', suffixes=('_pub', ''), indicator=True)
            same = bool((j['_merge'] == 'both').all()) and len(pub) == len(band)
            both = j[j['_merge'] == 'both']
            nan_same = bool((both[column].isna() == both[column + '_pub'].isna()).all())
            diff = float(np.nanmax(np.abs(both[column] - both[column + '_pub'])))
            other = {c: float(np.nanmax(np.abs(both[c] - both[c + '_pub']))) for c in ('truth', 'linear')}
            tol = report['gate_tolerance']
            report['gate'] = {'rule': report['gate_rule'], 'rows_identical': same,
                              'missing_predictions_identical': nan_same, 'max_abs_prediction_diff': diff,
                              'other_columns_max_abs_diff': other,
                              'mae_abs_diff': abs(values['arm'][unit] - published['arm']),
                              'pass': bool(same and nan_same and diff <= tol and max(other.values()) <= 1e-9)}
            if not report['gate']['pass']:
                return gate_failed(name, report, started)
    source = _rel(RUNS / published_path)
    label = report['label']
    report['quantities'] = [
        mae_q(f'{label} MAE', published['arm'], values['arm'], source),
        mae_q('linear interpolation MAE', published['linear'], values['linear'], source, 'reference'),
        contrast_q(f'{label} - linear', published['contrast'], values['contrast'], 'linear interpolation MAE',
                   published['linear'], source)]
    finish(name, report, started)


def arm_b5(units=None):
    import src.experiments.v2_gap_recovery as gr
    if not gr.GP_BY_PATTERN or gr.GP_NEIGHBOUR_ORDER != 'nearest_others':
        raise SystemExit('B5 is the per-pattern process with nearest_others (addendum 59)')
    _gap_arm('B5', 'B5: the Gaussian-process filler with its linear mean per pattern of reporting neighbours '
                   '(addendum 59), oxygen, Gap-Dev, 12-72 h, against linear interpolation '
                   '(v2_gap_recovery.gpr_fill)',
             'gpr', lambda band: gr.gpr_fill(band, 'oxygen'), GP_PUBLISHED,
             {'label': 'Gaussian process', 'gate_tolerance': 1e-9,
              'gate_rule': 'rows identical, the same rows without a prediction, every prediction within 1e-9 '
                           '(addendum 59, part A gate)',
              'kernel': 'published: ConstantKernel(1.0, (1e-3, 1e3)) * RBF(2.0, (1e-2, 1e3)) + '
                        'ConstantKernel(0.5, (1e-3, 1e3)) * ExpSineSquared(2.0, 1.0, periodicity fixed) + '
                        'WhiteKernel(0.1, (1e-3, 1e1)), normalize_y=False, alpha=1e-6; bounds NOT rescaled '
                        'for mg/l'}, units)


def arm_b5_kernels():
    """Diagnostic for B5, not part of the declared test: the temporal process's fitted hyperparameters
    in each unit, and how often each sits on a bound of the published kernel. Each fit is the
    published `_gp_temporal` (same kernel, settings and data), recorded as it is made; the
    predictions are checked against the arm's own rows written by `b5`."""
    import src.experiments.v2_gap_recovery as gr
    from sklearn.gaussian_process import GaussianProcessRegressor
    started = time.time()
    names = {'k1__k1__k1__constant_value': 'smooth amplitude', 'k1__k1__k2__length_scale': 'smooth length (days)',
             'k1__k2__k1__constant_value': 'daily amplitude', 'k1__k2__k2__length_scale': 'daily length',
             'k2__noise_level': 'white noise'}
    out = {'what': 'B5 diagnostic: fitted hyperparameters of the temporal Gaussian process per fit, by unit, '
                   'and the share of fits at a bound of the published kernel', 'units': {}}
    original = gr._gp_temporal
    for unit in ('umol', 'mgl'):
        fits = []

        def recording(t_fit, target, t_pred):
            gp = GaussianProcessRegressor(kernel=gr._gp_kernel(), normalize_y=False,
                                          n_restarts_optimizer=0, alpha=1e-6)
            gp.fit(t_fit, target)
            row = {'n': int(len(target)), 'target_variance': float(np.var(target))}
            free = [hp for hp in gp.kernel_.hyperparameters if not hp.fixed]   # theta holds only these
            for hp, theta in zip(free, gp.kernel_.theta):
                lo, hi = np.log(np.asarray(hp.bounds, float).ravel())
                row[hp.name] = float(np.exp(theta))
                row[hp.name + '__at'] = 'lower' if theta <= lo + 1e-6 else 'upper' if theta >= hi - 1e-6 else ''
            fits.append(row)
            return gp.predict(t_pred)
        gr._gp_temporal = recording
        try:
            with oxygen_unit(unit):
                band = gap_band()
                pred = gr.gpr_fill(band, 'oxygen')
        finally:
            gr._gp_temporal = original
        rows = pd.read_csv(OUT / 'B5' / unit / 'oxygen_gpr' / 'predictions.csv.gz')
        same = band[['episode', 'position']].assign(local=pred).merge(rows[['episode', 'position', 'gpr']],
                                                                       on=['episode', 'position'])
        if (same.local.isna() != same.gpr.isna()).any():
            raise SystemExit(f'b5-kernels {unit}: the recorded fits do not cover the rows the arm predicted')
        f = pd.DataFrame(fits)
        summary = {'fits': int(len(f)), 'target_variance_median': float(f.target_variance.median()),
                   'max_abs_diff_from_arm_rows': float(np.nanmax(np.abs(same.local - same.gpr))),
                   'hyperparameters': {}}
        for key, label in names.items():
            if key in f:
                summary['hyperparameters'][label] = {
                    'median': float(f[key].median()), 'p10': float(f[key].quantile(.1)),
                    'p90': float(f[key].quantile(.9)),
                    'share_at_lower_bound': float((f[key + '__at'] == 'lower').mean()),
                    'share_at_upper_bound': float((f[key + '__at'] == 'upper').mean())}
        out['units'][unit] = summary
    out.update(bounds={names[hp.name]: [float(b) for b in np.asarray(hp.bounds, float).ravel()]
                       for hp in gr._gp_kernel().hyperparameters if hp.name in names and not hp.fixed},
               seconds=round(time.time() - started, 1), compute=_host(),
               producer_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    _json_write(OUT / 'B5' / 'kernel_bounds.json', out)
    for unit, s in out['units'].items():
        print(unit, s['fits'], 'fits; max diff from arm rows', s['max_abs_diff_from_arm_rows'])
        for label, h in s['hyperparameters'].items():
            print(f"   {label:<22} median {h['median']:.4g}  at lower {h['share_at_lower_bound']:.1%}  "
                  f"at upper {h['share_at_upper_bound']:.1%}")


#: B5 with unit-consistent kernel constants (coordinator's follow-up of 2026-10-02, not declared):
#: (variant, unit, factor on the kernel's variance-type values and bounds, factor on the GP's alpha)
B5_CONSISTENT = (
    ('published_kernel', 'umol', 1.0, 1.0, 'default'),
    ('kernel_c2_alpha_as_published', 'mgl_c', C ** 2, 1.0, 'default'),
    ('kernel_c2_alpha_as_published', 'mgl', C ** 2, 1.0, 'default'),
    ('kernel_and_alpha_c2', 'mgl_c', C ** 2, C ** 2, 'default'),
    ('kernel_and_alpha_c2', 'mgl', C ** 2, C ** 2, 'default'),
    # does the mgl_c gap close when L-BFGS-B is run to convergence? Its ftol is relative to the
    # objective, which shifts by n log c with the unit while the gradient does not.
    ('published_kernel_tight_optimiser', 'umol', 1.0, 1.0, 'tight'),
    ('kernel_and_alpha_c2_tight_optimiser', 'mgl_c', C ** 2, C ** 2, 'tight'))


def tight_optimiser(obj_func, initial_theta, bounds):
    """sklearn's L-BFGS-B with ftol 1e-15 and gtol 1e-12 in place of its defaults (2.2e-9 and 1e-5)."""
    import scipy.optimize
    r = scipy.optimize.minimize(obj_func, initial_theta, method='L-BFGS-B', jac=True, bounds=bounds,
                                options={'ftol': 1e-15, 'gtol': 1e-12, 'maxiter': 15000})
    return r.x, r.fun


def scaled_kernel(kernel, factor):
    """The published kernel with every amplitude (ConstantKernel) and noise level (WhiteKernel), initial
    value and bounds, multiplied by `factor`: they are variances in the oxygen unit squared. Length
    scales and the fixed periodicity are in time and stay."""
    params = kernel.get_params()
    updates = {}
    for key, value in params.items():
        if key.endswith(('constant_value', 'noise_level')):
            updates[key] = value * factor
            bounds = params[key + '_bounds']
            updates[key + '_bounds'] = bounds if isinstance(bounds, str) else tuple(b * factor for b in bounds)
    return kernel.set_params(**updates)


def arm_b5_consistent():
    """B5 refitted with the Gaussian process's variance-type constants converted with the data: the
    published kernel's amplitudes and noise level (initial values and bounds) times c^2, and, in a
    second variant, its alpha (1e-6, a noise variance added to the diagonal) times c^2 as well. Runs
    the published `gpr_fill` / `_gp_temporal` unchanged: `_gp_kernel` is replaced by the scaled
    kernel, and the GaussianProcessRegressor that `_gp_temporal` imports is a subclass whose fit
    scales alpha and records the fitted hyperparameters and whether each sits on a bound. The
    published-kernel run in umol checks that this hook changes nothing (it must reproduce B5's rows)."""
    import sklearn.gaussian_process as skgp
    import src.experiments.v2_gap_recovery as gr
    started = time.time()
    if (gr.PERIOD, gr.DESIGN, gr.SEED) != ('development', 'strict', 20260922) or not gr.GP_BY_PATTERN:
        raise SystemExit('B5 needs the default V2_GAP_* environment and the per-pattern process')
    base = json.loads((OUT / 'B5' / 'report.json').read_text())
    published = base['published_full_precision']
    names = {'k1__k1__k1__constant_value': 'smooth amplitude', 'k1__k1__k2__length_scale': 'smooth length (days)',
             'k1__k2__k1__constant_value': 'daily amplitude', 'k1__k2__k2__length_scale': 'daily length',
             'k2__noise_level': 'white noise'}
    original_kernel, original_gpr = gr._gp_kernel, skgp.GaussianProcessRegressor
    out = {'what': 'B5 with the kernel\'s variance-type constants converted by c^2 (requested by the coordinator '
                   'on 2026-10-02 after the declared check; not part of the declared test)',
           'published': published, 'c': C, 'variants': {}}
    tight_umol = None
    for variant, unit, kfactor, afactor, optimiser in B5_CONSISTENT:
        t0 = time.time()
        fits = []

        class Recording(original_gpr):
            def fit(self, X, y):
                self.alpha = self.alpha * afactor
                if optimiser == 'tight':
                    self.optimizer = tight_optimiser
                super().fit(X, y)
                free = [hp for hp in self.kernel_.hyperparameters if not hp.fixed]
                row = {'alpha': float(self.alpha)}
                for hp, theta in zip(free, self.kernel_.theta):
                    lo, hi = np.log(np.asarray(hp.bounds, float).ravel())
                    row[hp.name] = float(np.exp(theta))
                    row[hp.name + '__at'] = ('lower' if theta <= lo + 1e-6 else
                                             'upper' if theta >= hi - 1e-6 else '')
                fits.append(row)
                return self
        gr._gp_kernel = (lambda: scaled_kernel(original_kernel(), kfactor)) if kfactor != 1.0 else original_kernel
        skgp.GaussianProcessRegressor = Recording
        try:
            with oxygen_unit(unit):
                band = gap_band()
                band['gpr'] = gr.gpr_fill(band, 'oxygen')
        finally:
            gr._gp_kernel, skgp.GaussianProcessRegressor = original_kernel, original_gpr
        kernel_used = scaled_kernel(original_kernel(), kfactor) if kfactor != 1.0 else original_kernel()
        f = pd.DataFrame(fits)
        at = f[[k + '__at' for k in names if k + '__at' in f]]
        entry = {'unit': unit, 'kernel_factor': kfactor, 'alpha_factor': afactor, 'optimiser': optimiser,
                 'kernel': repr(kernel_used),
                 'alpha': float(f.alpha.iloc[0]), 'rows': int(len(band)), 'episodes': int(band.episode.nunique()),
                 'rows_without_prediction': int(band.gpr.isna().sum()),
                 'mae': episode_mae(band, 'gpr'), 'linear_mae': episode_mae(band, 'linear'),
                 'minus_linear': episode_contrast(band, 'gpr', 'linear'),
                 'fits': int(len(f)), 'fits_with_any_bound': int((at != '').any(axis=1).sum()),
                 'fits_at_a_lower_bound': int((at == 'lower').any(axis=1).sum()),
                 'fits_at_an_upper_bound': int((at == 'upper').any(axis=1).sum()),
                 'hyperparameters': {label: {'median': float(f[k].median()),
                                             'share_at_lower_bound': float((f[k + '__at'] == 'lower').mean()),
                                             'share_at_upper_bound': float((f[k + '__at'] == 'upper').mean())}
                                     for k, label in names.items() if k in f}}
        entry['share_of_fits_with_any_bound'] = entry['fits_with_any_bound'] / entry['fits']
        if variant == 'published_kernel_tight_optimiser':
            tight_umol = band[['episode', 'position', 'gpr']].copy()
            entry['prediction_sha256'] = _csv(
                band[['episode', 'sensor', 'gap_length', 'position', 'fold', 'time', 'truth', 'linear', 'gpr']],
                OUT / 'B5_unit_consistent' / variant / unit / 'oxygen_gpr' / 'predictions.csv.gz')
        elif unit == 'umol':
            rows = pd.read_csv(OUT / 'B5' / 'umol' / 'oxygen_gpr' / 'predictions.csv.gz')
            same = band[['episode', 'position', 'gpr']].merge(rows[['episode', 'position', 'gpr']],
                                                              on=['episode', 'position'], suffixes=('', '_arm'))
            entry['hook_check'] = {'rule': 'with the published kernel the hooked run reproduces B5\'s umol rows '
                                           'within 1e-9', 'rows': int(len(same)),
                                   'max_abs_diff': float(np.nanmax(np.abs(same.gpr - same.gpr_arm))),
                                   'missing_identical': bool((same.gpr.isna() == same.gpr_arm.isna()).all())}
            entry['hook_check']['pass'] = bool(len(same) == len(rows) and entry['hook_check']['missing_identical']
                                               and entry['hook_check']['max_abs_diff'] <= 1e-9)
            if not entry['hook_check']['pass']:
                raise SystemExit(f'b5-consistent: the hook changes the published process {entry["hook_check"]}')
        else:
            sha = _csv(band[['episode', 'sensor', 'gap_length', 'position', 'fold', 'time', 'truth', 'linear', 'gpr']],
                       OUT / 'B5_unit_consistent' / variant / unit / 'oxygen_gpr' / 'predictions.csv.gz')
            entry['prediction_sha256'] = sha
            cm, cl = C * published['arm'], C * published['linear']
            entry['c_x_published'] = {'mae': cm, 'linear_mae': cl, 'minus_linear': C * published['contrast']}
            entry['mae_relative_deviation'] = entry['mae'] / cm - 1
            entry['linear_mae_relative_deviation'] = entry['linear_mae'] / cl - 1
            entry['minus_linear_deviation_over_c_x_linear'] = (entry['minus_linear'] - C * published['contrast']) / cl
            entry['mae_within_criterion'] = bool(abs(entry['mae_relative_deviation']) <= TOLERANCE)
            entry['minus_linear_within_criterion'] = bool(abs(entry['minus_linear_deviation_over_c_x_linear'])
                                                          <= TOLERANCE)
            if optimiser == 'tight' and tight_umol is not None:
                ref = out['variants']['published_kernel_tight_optimiser']['umol']
                both = band[['episode', 'position', 'gpr']].merge(tight_umol, on=['episode', 'position'],
                                                                  suffixes=('', '_umol'))
                entry['against_c_x_tight_umol'] = {
                    'mae_relative_deviation': entry['mae'] / (C * ref['mae']) - 1,
                    'minus_linear_deviation_over_c_x_linear': (entry['minus_linear'] - C * ref['minus_linear'])
                    / (C * ref['linear_mae']),
                    'max_abs_row_diff_over_c': float(np.nanmax(np.abs(both.gpr / C - both.gpr_umol))),
                    'median_abs_row_diff_over_c': float(np.nanmedian(np.abs(both.gpr / C - both.gpr_umol)))}
        entry['seconds'] = round(time.time() - t0, 1)
        out['variants'].setdefault(variant, {})[unit] = entry
        tail = ''
        if 'mae_relative_deviation' in entry:
            tail = (f"  dev {entry['mae_relative_deviation']:+.4%} / "
                    f"{entry['minus_linear_deviation_over_c_x_linear']:+.4%}")
        if 'hook_check' in entry:
            tail = f"  hook check {entry['hook_check']['max_abs_diff']:.1e}"
        if 'against_c_x_tight_umol' in entry:
            g = entry['against_c_x_tight_umol']
            tail += (f"  vs c x tight umol {g['mae_relative_deviation']:+.2e} / "
                     f"{g['minus_linear_deviation_over_c_x_linear']:+.2e}, rows {g['max_abs_row_diff_over_c']:.1e}")
        print(f"{variant:<36} {unit:<6} MAE {entry['mae']:.6f}  GP-linear {entry['minus_linear']:+.6f}  "
              f"fits on a bound {entry['fits_with_any_bound']}/{entry['fits']}{tail}", flush=True)
    out.update(seconds=round(time.time() - started, 1), compute=_host(),
               producer_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    _json_write(OUT / 'B5_unit_consistent' / 'report.json', out)
    print('wrote', _rel(OUT / 'B5_unit_consistent' / 'report.json'))


def arm_b6(units=None):
    import src.experiments.v2_gap_recovery as gr
    _gap_arm('B6', 'B6: zero-shot Chronos-Bolt (amazon/chronos-bolt-base) run forward and backward across the '
                   'gap and blended by distance, oxygen, Gap-Dev, 12-72 h, against linear interpolation '
                   '(v2_gap_recovery.chronos_fill)',
             'chronos_bidir', lambda band: gr.chronos_fill('oxygen', band, bidirectional=True), CHRONOS_PUBLISHED,
             {'label': 'bidirectional Chronos-Bolt', 'gate_tolerance': 1e-4,
              'gate_rule': 'rows identical, the same rows without a prediction, every prediction within 1e-4 '
                           'umol/kg (float32 network on a GPU; the artifact prints four decimals)'}, units)


# ------------------------------------------------------------------------------------ B7

def arm_b7(units=None):
    """The anchored continuation on Bridge-Dev, refitted with each fold's published selection."""
    from src.experiments import v2_bridge as vb
    import src.experiments.v2_horizon as hz
    from src.experiments.v2_closed_loop import episodes, STEP
    started = time.time()
    leads = (6, 24, 48)
    studies = json.loads((vb.OUT / 'bridge_oxygen_studies.json').read_text())
    pub_all = pd.read_csv(vb.OUT / 'bridge_oxygen.csv.gz', parse_dates=['onset'])
    pub = pub_all[pub_all.cohort == 'Bridge-Dev'].copy()
    rep = json.loads((vb.OUT / 'report.json').read_text())['oxygen']['Bridge-Dev']['by_lead']
    eps_pub = pd.read_csv(vb.CLOSED_LOOP / 'closed_loop_oxygen.csv.gz', usecols=['sensor', 'time'],
                          parse_dates=['time']).drop_duplicates()
    eps_pub = set(zip(eps_pub.sensor, eps_pub.time))

    def lead_values(frame, lead):
        sub = frame[frame.lead == lead]
        both = sub.dropna(subset=['anchored_direct', 'clim_hour'])
        return {'anchored_direct': vb.macro(sub, 'anchored_direct'), 'clim_hour': vb.macro(sub, 'clim_hour'),
                'contrast': vb.macro(both, 'anchored_direct') - vb.macro(both, 'clim_hour')}
    published = {L: lead_values(pub, L) for L in leads}
    stored = {L: {'anchored_direct': rep[str(L)]['mae']['anchored_direct'], 'clim_hour': rep[str(L)]['mae']['clim_hour'],
                  'contrast': rep[str(L)]['contrasts']['anchored_direct - clim_hour']['point']} for L in leads}
    report = {'what': 'B7: the continuation anchored to the last real reading (a direct LightGBM model of the '
                      'change since onset), oxygen, Bridge-Dev, each fold with its published selection, against '
                      'the hour-of-day climatological increment (v2_bridge Context / fit / score_rows)',
              'cohort': 'Bridge-Dev', 'leads': list(leads), 'params': {k: studies[k]['params'] for k in ('fold0', 'fold1')},
              'published_full_precision': {str(L): v for L, v in published.items()},
              'published_report_6dp': {str(L): v for L, v in stored.items()},
              'published_matches_report': bool(all(abs(round(published[L][k], 6) - stored[L][k]) < 2e-6
                                                   for L in leads for k in stored[L])),
              'units': {}, 'gate': {}}
    values = {L: {k: {} for k in ('anchored_direct', 'clim_hour', 'contrast')} for L in leads}
    for unit in units_for('B7', units):
        t0 = time.time()
        with oxygen_unit(unit):
            ctx = vb.Context('oxygen', 'dev')
            frames, counts = [], {}
            for fold in hz.folds(ctx.data, STEP):
                label = f'fold{fold["number"]}'
                eps = [(i0, s) for i0, s in episodes(ctx.data, fold, vb.MAX_LEAD)
                       if (ctx.names[s], ctx.data['times'][i0]) in eps_pub]
                lookup = vb.climatology_lookup(ctx.data, fold)
                first_outer = ctx.data['times'][fold['va'].min()]
                rows = vb.subsample(vb.training_rows(ctx, fold['tr'], fold['eligible'], first_outer),
                                    zlib.crc32(f'{label}|outer'.encode()))
                counts[label] = {'outer_training_rows': len(rows), 'episodes': len(eps)}
                if len(rows) != studies[label]['outer_training_rows']:
                    raise SystemExit(f'B7 {unit} {label}: {len(rows)} outer training rows, published '
                                     f'{studies[label]["outer_training_rows"]}')
                model = vb.fit(ctx, rows, lookup, studies[label]['params'])
                frames.append(vb.score_rows(ctx, fold, eps, model, lookup).assign(cohort='Bridge-Dev'))
            conversion = dict(_AUDIT)
            center, scale = float(ctx.center), float(ctx.scale)
        frame = pd.concat(frames, ignore_index=True)
        sha = _csv(frame, OUT / 'B7' / unit / 'bridge_oxygen.csv.gz')
        got = {L: lead_values(frame, L) for L in leads}
        for L in leads:
            for k in values[L]:
                values[L][k][unit] = got[L][k]
        report['units'][unit] = {'by_lead': {str(L): got[L] for L in leads}, 'rows': int(len(frame)),
                                 'fold_counts': counts, 'center': center, 'scale': scale,
                                 'prediction_sha256': sha, 'seconds': round(time.time() - t0, 1),
                                 'conversion': conversion}
        if unit == 'umol':
            mine = frame.assign(onset=pd.to_datetime(frame.onset, utc=True).dt.as_unit('ns'))
            theirs = pub.assign(onset=pd.to_datetime(pub.onset, utc=True).dt.as_unit('ns'))
            j = theirs.merge(mine, on=['fold', 'sensor', 'onset', 'lead'], how='outer', suffixes=('_pub', ''),
                             indicator=True)
            same = bool((j['_merge'] == 'both').all()) and len(pub) == len(frame)
            both = j[j['_merge'] == 'both']
            cols = ('truth', 'persistence', 'clim_month_hour', 'clim_hour', 'neighbour_change', 'anchored_direct')
            diffs = {c: float(np.nanmax(np.abs(both[c] - both[c + '_pub']))) for c in cols}
            report['gate'] = {'rule': 'rows identical, every column within 1e-9 of the published Bridge-Dev rows',
                              'rows_identical': same, 'max_abs_diff': diffs,
                              'pass': bool(same and max(diffs.values()) <= 1e-9)}
            if not report['gate']['pass']:
                return gate_failed('B7', report, started)
    source = _rel(vb.OUT / 'bridge_oxygen.csv.gz')
    report['quantities'] = []
    for L in leads:
        report['quantities'] += [
            mae_q(f'anchored continuation MAE, {L} h', published[L]['anchored_direct'], values[L]['anchored_direct'], source),
            mae_q(f'hour-of-day increment MAE, {L} h', published[L]['clim_hour'], values[L]['clim_hour'], source,
                  'reference'),
            contrast_q(f'anchored continuation - hour-of-day increment, {L} h', published[L]['contrast'],
                       values[L]['contrast'], f'hour-of-day increment MAE, {L} h', published[L]['clim_hour'], source)]
    finish('B7', report, started)


# ------------------------------------------------------------------------------------ the report

def check(q):
    """The declared criterion on one quantity, for the mg/l refit and the diagnostics."""
    out = dict(q)
    cp = C * q['published']
    out['c_x_published'] = cp
    if q['kind'] == 'mae':
        bound = TOLERANCE * cp
    else:
        out['c_x_reference_published_mae'] = C * q['reference_published_mae']
        bound = TOLERANCE * C * q['reference_published_mae']
    out['allowed_abs_deviation'] = bound
    out['reproduction_abs_diff'] = abs(q['umol'] - q['published']) if q.get('umol') is not None else None
    for unit in ('mgl', 'mgl_c', 'mgl_constants_as_coded'):
        if q.get(unit) is None:
            continue
        dev = q[unit] - cp
        out[f'{unit}_abs_deviation'] = dev
        out[f'{unit}_relative_deviation'] = dev / cp if cp else None
        if q['kind'] == 'contrast':
            out[f'{unit}_deviation_over_c_x_reference'] = dev / (C * q['reference_published_mae'])
        out[f'{unit}_pass'] = bool(abs(dev) <= bound)
    out['pass'] = out.get('mgl_pass', False)
    # the one-ulp diagnostic, in the published unit: the refit's own movement under the smallest change
    ulps = [u for u in q if u.startswith(DIAGNOSTIC_PREFIXES) and q[u] is not None]
    if ulps:
        scale = q['published'] if q['kind'] == 'mae' else q['reference_published_mae']
        moves = [(q[u] - q['published']) / scale for u in ulps]
        out['ulp_relative_moves'] = moves
        out['ulp_max_abs_relative_move'] = max(abs(m) for m in moves)
        out['ulp_draws_beyond_criterion'] = int(sum(abs(m) > TOLERANCE for m in moves))
    return out


def scaled(q, unit):
    """A checked quantity's deviation on the criterion's own scale: relative to c x published for an
    MAE, over c x the reference's published MAE for a contrast."""
    key = f'{unit}_relative_deviation' if q['kind'] == 'mae' else f'{unit}_deviation_over_c_x_reference'
    return q.get(key)


def cross_contrast(name, x, y, reference):
    """A contrast between quantities of two arm reports (pairwise on identical endpoints)."""
    values = {u: x[u] - y[u] for u in UNIT_MODES if x.get(u) is not None and y.get(u) is not None}
    return contrast_q(name, x['published'] - y['published'], values, reference['name'],
                      reference['published'], 'difference of the published MAEs on identical endpoints')


def report():
    arms = {}
    for arm in ARMS:
        path = OUT / arm / 'report.json'
        arms[arm] = json.loads(path.read_text()) if path.exists() else None
    out = {'declared_in': DECLARED, 'question': json.loads((ROOT / DECLARED[0]).read_text())['part_B_equivalence']['question'],
           'c_mg_per_l_per_umol_kg': C, 'mg_per_l_per_umol_l': PER_LITRE, 'units_artifact': 'runs/v2_oxygen_units.json',
           'criterion': {'mae': f'|refit - c x published| <= {TOLERANCE:.1%} of c x published',
                         'contrast': f"|refit - c x published| <= {TOLERANCE:.1%} of c x the reference arm's published MAE"},
           'conversion': {
               'sondes': 'each hour by its own 0.0319988 x rho/1000 (runs/v2_oxygen_units/hourly_factor.csv.gz), '
                         'c where the hour has no density',
               'casts': 'per-kilogram bins (do_unit_source umol/l) by 0.0319988 x rho/1000 with the bin\'s TEOS-10 '
                        'density at depth_m dbar; per-litre bins (mg/l sources, and the percent cast, which was '
                        'formed from the per-litre Garcia-Gordon solubility) by 0.0319988',
               'unit_bearing_constants': 'converted by c: the z-score centre and scale of the forecasting grid '
                                         '(230 and 50 in v2_horizon.build) and the Field-57 oxygen target scale (20); '
                                         'the bridge solubility feature formed in mg/l',
               'left_as_published': 'every selected hyperparameter, seed, fold, mask, row and model setting, '
                                    'including the Gaussian process kernel bounds and normalize_y=False'},
           'units_run': {'umol': 'as published: the reproduction gate',
                         'mgl': 'the declared refit (amendment 1, part B data)',
                         'mgl_c': 'diagnostic, not declared: every oxygen value times the one constant c',
                         'mgl_constants_as_coded': 'B3 sensitivity, not declared: as mgl but with the z-score '
                                                   'constants left at 230 and 50'},
           'arms': {}}
    quantities = {}
    for arm, rep in arms.items():
        if rep is None:
            out['arms'][arm] = {'status': 'not run', 'pass': False}
            continue
        entry = {'what': rep['what'], 'cohort': rep.get('cohort'), 'gate': rep.get('gate'),
                 'compute': rep.get('compute'), 'seconds': rep.get('seconds'),
                 'arm_report': _rel(OUT / arm / 'report.json')}
        if 'stopped' in rep:
            entry.update({'status': 'stopped at the reproduction gate', 'reason': rep['stopped'], 'pass': False})
            out['arms'][arm] = entry
            continue
        qs = list(rep['quantities'])
        quantities[arm] = {q['name']: q for q in qs}
        entry['quantities'] = qs
        out['arms'][arm] = entry
    # contrasts across arm reports
    if 'B2' in quantities and 'B1' in quantities:
        x, y = quantities['B2']['XGBoost weather tree MAE'], quantities['B1']['weather tree MAE']
        out['arms']['B2']['quantities'].append(
            cross_contrast('XGBoost weather tree - LightGBM weather tree', x, y, y))
    if 'B3' in quantities and 'B1' in quantities:
        y = quantities['B1']['plain tree MAE']
        out['arms']['B3']['quantities'].append(
            cross_contrast('graph arm (three-network mean) - plain tree', quantities['B3']['graph arm MAE (three-network mean)'], y, y))
    worst, worst_all = None, None
    for arm, entry in out['arms'].items():
        if 'quantities' not in entry:
            continue
        entry['quantities'] = [check(q) for q in entry['quantities']]
        gate = entry.get('gate') or {}
        entry['pass'] = bool(gate.get('pass', True) and all(q['pass'] for q in entry['quantities']))
        entry['status'] = 'evaluated'
        entry['failing_quantities'] = [q['name'] for q in entry['quantities'] if not q['pass']]
        # what the diagnostics say about the source of a deviation (none of this is the declared test)
        diag = {}
        for unit in ('mgl', 'mgl_c', 'mgl_constants_as_coded'):
            got = [abs(scaled(q, unit)) for q in entry['quantities'] if scaled(q, unit) is not None]
            if got:
                diag[f'{unit}_largest_scaled_deviation'] = max(got)
        moves = [q['ulp_max_abs_relative_move'] for q in entry['quantities'] if 'ulp_max_abs_relative_move' in q]
        if moves:
            diag['one_ulp_refits_largest_scaled_move'] = max(moves)
            diag['one_ulp_refits_beyond_criterion'] = {q['name']: q['ulp_draws_beyond_criterion']
                                                       for q in entry['quantities'] if 'ulp_draws_beyond_criterion' in q}
        rep = arms[arm]
        for key in ('attribution_per_litre_casts', 'scored_casts_by_source_unit', 'scored_per_litre_casts',
                    'kernel', 'published_matches_artifact', 'published_matches_report'):
            if key in rep:
                diag[key] = rep[key]
        kernels = OUT / arm / 'kernel_bounds.json'
        if kernels.exists():
            k = json.loads(kernels.read_text())
            diag['kernel_fits'] = {'source': _rel(kernels), 'host': k['compute']['host'], 'bounds': k['bounds'],
                                   'units': k['units']}
        entry['diagnostics'] = diag
        consistent = OUT / 'B5_unit_consistent' / 'report.json'
        if arm == 'B5' and consistent.exists():
            k = json.loads(consistent.read_text())
            entry['B5_unit_consistent_bounds'] = {
                'source': _rel(consistent), 'what': k['what'], 'host': k['compute']['host'],
                'producer_sha256': k['producer_sha256'],
                'note': 'Not part of the declared test and not counted in PB1 or PB2: the published configuration '
                        'is the unscaled kernel. kernel_c2_alpha_as_published scales the ConstantKernel amplitudes '
                        'and the WhiteKernel noise level (initial values and bounds) by c^2; kernel_and_alpha_c2 '
                        'also scales the regressor\'s alpha (1e-6), a noise variance added to the diagonal. The '
                        '*_tight_optimiser pair reruns the published kernel in umol and kernel_and_alpha_c2 under '
                        'mgl_c with L-BFGS-B at ftol 1e-15 and gtol 1e-12, to separate the optimiser\'s stopping '
                        'rule (relative to the objective, which shifts by n log c with the unit) from the model.',
                'variants': k['variants']}
        for q in entry['quantities']:
            if q['kind'] == 'mae' and q.get('mgl_relative_deviation') is not None:
                cand = (abs(q['mgl_relative_deviation']), arm, q['name'])
                worst_all = max(worst_all or cand, cand)
                if q.get('role') == 'arm':
                    worst = max(worst or cand, cand)
    failing = [a for a, e in out['arms'].items() if not e.get('pass')]
    out['predictions'] = {
        'PB1': {'statement': 'All seven arms pass.', 'held': not failing, 'arms_failing': failing},
        'PB2': {'statement': 'The largest relative MAE deviation over the seven arms is below 0.2 %, the order of '
                             'the spread of the hourly ratio.',
                'largest_relative_mae_deviation': worst[0] if worst else None,
                'at': f'{worst[1]}: {worst[2]}' if worst else None,
                'largest_including_reference_arms': worst_all[0] if worst_all else None,
                'at_including_reference_arms': f'{worst_all[1]}: {worst_all[2]}' if worst_all else None,
                'held': bool(worst is not None and worst[0] < PB2_BOUND and len(quantities) == len(ARMS))}}
    _json_write(REPORT, out)
    print(f'wrote {_rel(REPORT)}')
    for arm, entry in out['arms'].items():
        print(f"{arm}: {entry.get('status')}  pass={entry.get('pass')}")
        for q in entry.get('quantities', []):
            rd = q.get('mgl_relative_deviation' if q['kind'] == 'mae' else 'mgl_deviation_over_c_x_reference')
            moves = q.get('ulp_max_abs_relative_move')
            print(f"   {q['name']:<58} pub {q['published']:+10.5f}  c*pub {q['c_x_published']:+9.6f}  "
                  f"mg/l {q.get('mgl', float('nan')):+9.6f}  dev {rd if rd is None else f'{rd:+.3%}'}  "
                  f"{'PASS' if q['pass'] else 'FAIL'}"
                  + ('' if moves is None else f'   (one-ulp refits move up to {moves:.3%})'))
    print('PB1', out['predictions']['PB1']['held'], ' PB2', out['predictions']['PB2']['held'],
          out['predictions']['PB2']['largest_relative_mae_deviation'], out['predictions']['PB2']['at'])


RUNNERS = {'b1': arm_b1, 'b2': arm_b2, 'b3': arm_b3, 'b4': arm_b4, 'b5': arm_b5, 'b6': arm_b6, 'b7': arm_b7}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('command', nargs='?', choices=list(RUNNERS) + ['b5-kernels', 'b5-consistent', 'report',
                                                                    'local'], default='local')
    ap.add_argument('--units', nargs='+', choices=list(UNIT_MODES))
    a = ap.parse_args()
    if a.command == 'report':
        return report()
    if a.command == 'b5-kernels':
        return arm_b5_kernels()
    if a.command == 'b5-consistent':
        return arm_b5_consistent()
    if a.command == 'local':
        for arm in LOCAL:
            RUNNERS[arm.lower()](a.units)
        return report()
    RUNNERS[a.command](a.units)


if __name__ == '__main__':
    main()
