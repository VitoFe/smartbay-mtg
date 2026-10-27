"""Review R4: a spatial background that knows how deep each remote probe hangs.

The released background (src/data/bay_profile_context.py: context_features) is a weighted mean of the
latest valid readings of the remote moored probes, w_j = exp(-age_j / 6 h) / (distance_j + 100 m)^2,
and it is treated as the value at 3 m: the vertical prior then shifts it to the query depth z
(src/experiments/v2_field.py: matrices). The probes do not hang at 3 m. Their own depth channel puts
them at 0.8 to 6.6 m (scripts/v2_probe_depth.py), and no probe depth enters the background. This
module builds the background a vertical prior implies when it is applied on both sides,

    b(z) = sum_j w_j (y_j - D(z_j)) / sum_j w_j + D(z),

where D is the arm's own vertical offset relative to the value at 3.0 m (D(3 m) = 0 by construction:
the cast deltas are taken against each cast's value interpolated at exactly 3.0 m), evaluated per 1 m
bin as the query side evaluates it, and z_j is the neighbour's depth at the hour its reading carries.
A neighbour in the query's own bin therefore enters unchanged. Neighbours, ages and weights are those
of context_features (same 250 m query mask, same held-station mask, same six-hour search); the design
matrix, and so every feature the ridge reads, is unchanged, so an arm differs from its reference
through the background alone. Oxygen and salinity come only from the EXO2 sondes, as in the released
background, each corrected with its own variable's prior at the sonde's own depth.

Field-57 (held station, one study per (arm, outer fold) as addendum 47, src/experiments/v2_perfold.py):
    A2    no prior                                    A2_none_direct, control re-run here
    A7    cast prior, every probe taken at 3 m        A7_ctdprior_direct, control re-run here
    A7d   cast prior, each probe at its measured depth (QC'd hourly, scripts/v2_probe_depth.py)
    A7n   cast prior, each probe at its nominal depth (sensor_geometry.csv): separates "depth-aware"
          from "measured depth"
A7d and A7n use A7's design (Copernicus boundary regressors with the cast-prior delta in place of the
Copernicus delta) and A7's prior, fitted per checkpoint on archive casts at that checkpoint's training
stations up to its last training time (v2_field_ctdprior.harmonic_prior).

Field-2026 (the 38 casts of January-June 2026; a replication on a reused cohort, mirroring
src/experiments/v2_field_reserved2026.py): A2_none and A7o (controls) and A7od / A7on (measured /
nominal), one 100-trial study each on the development folds, frozen, then read with the held station
excluded from its own context and prior, exactly as there. The controls must reproduce the published
development studies and the published 2026 numbers before a depth arm is reported.

Contrasts are paired on the same rows. Field-57 resamples cast dates (scripts/v2_perfold_contrasts.py:
CastDates, 2,000 draws, a date drawn twice counting twice); Field-2026 resamples cast dates with the
delete-one-date jackknife beside it (as v2_field_reserved2026, six decimals here). Each report also
gives the gain by depth band of the scored bins (0-2, 2-4, 4-6, over 6 m) for A7 - A2 and A7d - A2.

Declared as addendum 68 (config/bay_reserved_protocol_v2_addendum68.json) before the full run.
Usage (repository root, PYTHONPATH=., V2_THREADS=2):
  python -m scripts.v2_probe_depth                                  # hourly depth, once
  python -m src.experiments.v2_field_depth check [--with-2026]      # gates, no depth arm scored
  python -m src.experiments.v2_field_depth field57 [--arm A2 A7 A7d A7n] [--quick]
  python -m src.experiments.v2_field_depth field57-report [--quick]
  python -m src.experiments.v2_field_depth field2026 [--quick]
--quick runs the smallest protocol (fold 0, held station PBoaT, two trials; Field-2026 reads the
second half of 2025, never 2026) into reports/bay_twin/runs/v2_field_depth/smoke/.
"""
import os
os.environ.setdefault('V2_THREADS', '2')

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

from scripts import v2_probe_depth as probe_depth
from scripts.v2_perfold_contrasts import CastDates
from src.data.bay_profile_context import (TARGETS, allowed_sources, context_features, load_observations,
                                          load_profiles)
from src.experiments.bay_field_export import EXTRAS
from src.experiments.bay_profile_field import prediction_frame, prepare
from src.experiments.v2_field import fit_predict, inner_score, matrices, nested_mae
from src.experiments.v2_field_ctdprior import BIN_M, REFERENCE_DEPTH_M, archive, harmonic_prior
from src.experiments.v2_field_reserved2026 import (END, GAP, PUBLISHED_A2, START, observations_through,
                                                   profiles_window, seasonal)
from src.tuning.protocol_v2 import PROTOCOL, run_study

ROOT = Path(__file__).resolve().parents[2]
RUNS = ROOT / 'reports/bay_twin/runs'
OUT = RUNS / 'v2_field_depth'
PERFOLD = RUNS / 'v2_perfold/field/perfold'
PERFOLD_JSON = RUNS / 'v2_perfold.json'
RESERVED_REPORT = RUNS / 'v2_field_reserved2026/report.json'
BOUNDARY = ROOT / 'reports/bay_twin/profile_boundary_context.csv.gz'
CONFIG = ROOT / 'config/bay_profile_field.json'

VARIABLES = ('temperature', 'salinity', 'oxygen')    # the background's columns, in TARGETS order
FIELD57 = {'A2': 'A2_none_direct', 'A7': 'A7_ctdprior_direct',
           'A7d': 'A7d_ctdprior_measured_direct', 'A7n': 'A7n_ctdprior_nominal_direct'}
FIELD2026 = ('A2_none', 'A7o', 'A7od', 'A7on')
MODE = {'A7d': 'measured', 'A7n': 'nominal', 'A7od': 'measured', 'A7on': 'nominal'}
BANDS = ((0.0, 2.0, '0-2m'), (2.0, 4.0, '2-4m'), (4.0, 6.0, '4-6m'), (6.0, np.inf, 'over_6m'))
DRAWS = 2000
SMOKE_DRAWS = 200                                     # --quick only: exercises the code, quotes nothing
RNG_SEED = 20260923                                   # v2_field_reserved2026.RNG_SEED
DECLARATION = 'config/bay_reserved_protocol_v2_addendum68.json'   # declared 2026-10-05T16:31Z

FIELD57_CONTRASTS = [('depth-aware measured vs cast prior', 'A7d', 'A7'),
                     ('depth-aware measured vs none', 'A7d', 'A2'),
                     ('depth-aware nominal vs cast prior', 'A7n', 'A7'),
                     ('depth-aware nominal vs none', 'A7n', 'A2'),
                     ('measured vs nominal depth', 'A7d', 'A7n'),
                     ('cast prior vs none', 'A7', 'A2')]
FIELD2026_CONTRASTS = [('A7od', 'A7o'), ('A7od', 'A2_none'), ('A7on', 'A7o'), ('A7on', 'A2_none'),
                       ('A7od', 'A7on'), ('A7o', 'A2_none'), ('A7od', 'seasonal'),
                       ('background_A7od', 'background_A7o'), ('background_A7on', 'background_A7o'),
                       ('background_A7od', 'A7od')]
BAND_CONTRASTS = [('A7', 'A2'), ('A7d', 'A2'), ('A7n', 'A2'), ('A7d', 'A7')]


# ---------------------------------------------------------------------------------------- helpers

def _clean(obj):
    if isinstance(obj, dict):
        return {str(k): _clean(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_clean(v) for v in obj]
    if isinstance(obj, (np.floating, float)):
        return None if not np.isfinite(obj) else float(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    return obj


def _json(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(json.dumps(_clean(obj), indent=2, allow_nan=False, default=str).encode('utf-8'))


def _csv(frame, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, lineterminator='\n')
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _utc(stamp):
    stamp = pd.Timestamp(stamp)
    return stamp.tz_localize('UTC') if stamp.tzinfo is None else stamp


# ------------------------------------------------------------------------- the probes' own depth

class ProbeDepth:
    """z_j for a reading of probe j labelled L: 'measured' (QC'd hourly depth, then the fallback rule
    of scripts/v2_probe_depth.py) or 'nominal' (sensor_geometry.csv nominal_depth_m)."""

    def __init__(self, mode, meta):
        if mode not in ('measured', 'nominal'):
            raise ValueError(mode)
        self.mode = mode
        self.nominal = meta.nominal_depth_m.astype(float).to_dict()
        self.digest = None
        if mode == 'measured':
            if not (probe_depth.HOURLY.exists() and probe_depth.SUMMARY.exists()):
                probe_depth.build()
            summary = json.loads(probe_depth.SUMMARY.read_text(encoding='utf-8'))
            self.digest = hashlib.sha256(probe_depth.HOURLY.read_bytes()).hexdigest()
            if self.digest != summary['files'][probe_depth.HOURLY.name]:
                raise SystemExit('hourly_depth.csv.gz does not match its summary; rebuild with scripts.v2_probe_depth')
            h = pd.read_csv(probe_depth.HOURLY)
            h = h[np.isfinite(h.depth_m)]
            h['time'] = pd.to_datetime(h.time, utc=True)
            self.hourly = {s: pd.Series(g.depth_m.to_numpy(), index=pd.DatetimeIndex(g.time).as_unit('ns')).sort_index()
                           for s, g in h.groupby('sensor')}

    def __call__(self, sensors, labels):
        sensors = np.asarray(sensors, dtype=object)
        labels = pd.DatetimeIndex(labels).as_unit('ns')
        z = np.full(len(sensors), np.nan)
        source = np.full(len(sensors), 'nominal', dtype=object)
        if self.mode == 'nominal':
            return np.array([self.nominal[s] for s in sensors], float), source
        for s in np.unique(sensors):
            m = np.flatnonzero(sensors == s)
            series = self.hourly.get(s)
            own = series.reindex(labels[m]).to_numpy(float) if series is not None else np.full(len(m), np.nan)
            z[m] = own
            source[m[np.isfinite(own)]] = 'hour'
            missing = m[~np.isfinite(own)]
            if len(missing):
                # causal fallback: the probe's last QC'd depth before the hour, else its nominal depth
                last = (series.index.searchsorted(labels[missing], side='left') - 1 if series is not None
                        else np.full(len(missing), -1))
                for i, k in zip(missing, last):
                    if k >= 0:
                        z[i], source[i] = float(series.iloc[k]), 'last'
                    else:
                        z[i], source[i] = self.nominal[s], 'nominal'
        return z, source


# --------------------------------------------------------- the neighbours context_features weighs

def neighbour_records(data, meta, frames):
    """Every (cast, variable, probe) reading context_features weighs with no station denied: its value,
    age, distance and the label of the hour it was read at (no delay, no outage, as the field arms)."""
    if list(frames) != list(meta.index):
        raise ValueError('frames must follow sensor_geometry order, as load_observations builds them')
    source_xy = meta[['x_m', 'y_m']].to_numpy()
    names = list(frames)
    rows = []
    for cast, g in data.groupby('cast_id', sort=False):
        q = g.iloc[0]
        qxy = np.array([q.UTM_X, q.UTM_Y])
        allow = allowed_sources(source_xy, qxy)
        distance = np.linalg.norm(source_xy - qxy, axis=1)
        stamp = q.t_utc
        for j, name in enumerate(names):
            if not allow[j]:
                continue
            frame = frames[name]
            window = frame.loc[(frame.index <= stamp) & (frame.index >= stamp - pd.Timedelta(hours=6))]
            for col in frame.columns:
                s = window[col].dropna()
                if len(s) and np.isfinite(s.iloc[-1]):
                    rows.append((cast, col, j, name, float(s.iloc[-1]),
                                 (stamp - s.index[-1]).total_seconds() / 3600, float(distance[j]), s.index[-1]))
    out = pd.DataFrame(rows, columns=['cast_id', 'variable', 'j', 'sensor', 'value', 'age', 'distance', 'label'])
    out['label'] = pd.DatetimeIndex(out.label).as_unit('ns')
    return out


def deny(records, data, meta, station):
    """context_features' held-station mask: drop probes within 250 m of the station's first cast."""
    if station is None:
        return records
    row = data[data.station == station].iloc[0]
    far = np.linalg.norm(meta[['x_m', 'y_m']].to_numpy() - np.array([row.UTM_X, row.UTM_Y]), axis=1) > 250
    return records[far[records.j.to_numpy()]].reset_index(drop=True)


def weighted(records):
    r = records.copy()
    w = np.exp(-r.age.to_numpy() / 6) / (r.distance.to_numpy() + 100) ** 2
    r['w'] = w / pd.Series(w).groupby([r.cast_id.to_numpy(), r.variable.to_numpy()]).transform('sum').to_numpy()
    return r


def per_cast(r, values):
    """sum_j w_j values_j per (cast, variable)."""
    return pd.Series(r.w.to_numpy() * values).groupby([r.cast_id.to_numpy(), r.variable.to_numpy()]).sum()


def rows_of(casts, by_cast, fill=0.0):
    """(rows, 3) array from a (cast, variable) series; `fill` where the cast has no reading."""
    out = np.full((len(casts), 3), fill, dtype=float)
    present = set(by_cast.index.get_level_values(1)) if len(by_cast) else set()
    for k, v in enumerate(VARIABLES):
        if v in present:
            out[:, k] = pd.Series(casts).map(by_cast.xs(v, level=1)).fillna(fill).to_numpy()
    return out


def neighbour_offset(r, train, depth, cast_doy):
    """sum_j w_j D(z_j) per (cast, variable): D the harmonic prior fitted on `train`, per 1 m bin, on the
    day of year of the cast; z_j the neighbour's depth at its reading's hour."""
    z, source = depth(r.sensor.to_numpy(), r.label)
    bins = np.floor(z / BIN_M).astype(int)
    doy = cast_doy.reindex(r.cast_id.to_numpy()).to_numpy()
    delta, fell = harmonic_prior(train, bins, doy)
    col = r.variable.map({v: k for k, v in enumerate(VARIABLES)}).to_numpy()
    d = delta[np.arange(len(r)), col]
    shift = per_cast(r, d)
    zbar = per_cast(r, z)
    diag = {'scope': 'every cast passed in (training and scored rows of the checkpoint or read)',
            'neighbour_readings': int(len(r)),
            'depth_source': {k: int(v) for k, v in pd.Series(source).value_counts().items()},
            'prior_fallback_bin_mean': int((fell == 1).sum()), 'prior_fallback_zero': int((fell == 2).sum()),
            'neighbour_depth_m': {'min': float(np.min(z)) if len(z) else None,
                                  'median': float(np.median(z)) if len(z) else None,
                                  'max': float(np.max(z)) if len(z) else None},
            'weighted_mean_depth_by_variable_m': {v: float(zbar.xs(v, level=1).mean())
                                                  for v in VARIABLES if len(zbar) and v in zbar.index.get_level_values(1)}}
    return shift, diag


def shift_summary(shift_rows, scored):
    """The background change (minus the neighbour offset) on the scored rows, per variable."""
    out = {}
    for k, (v, t) in enumerate(zip(VARIABLES, TARGETS)):
        x = -shift_rows[scored, k]
        out[t] = {'mean': float(x.mean()), 'mean_abs': float(np.abs(x).mean()), 'min': float(x.min()),
                  'max': float(x.max()), 'rows_changed': int((x != 0).sum()), 'rows': int(len(x))}
    return out


def context_gate(r, data, base, exact=False):
    """The records reproduce the background the bundle (or read) actually uses: float32 as prepare stores
    it (relative 1e-6), or the float64 context_features column (1e-9) when `exact`."""
    ctx = rows_of(data.cast_id.to_numpy(), per_cast(r, r.value.to_numpy()), fill=np.nan)
    ok = np.isfinite(ctx)
    ref = np.asarray(base, float)
    if exact:
        diff = float(np.max(np.abs(ctx[ok] - ref[ok]))) if ok.any() else 0.0
        same_missing = bool((np.isfinite(ref) == ok).all())
        return {'max_abs_diff': diff, 'missing_pattern_identical': same_missing,
                'pass': bool(diff <= 1e-9 and same_missing)}
    rel = float(np.max(np.abs(ctx[ok] - ref[ok]) / np.maximum(np.abs(ref[ok]), 1e-12))) if ok.any() else 0.0
    return {'max_relative_diff_vs_float32_base': rel, 'pass': bool(rel <= 1e-6)}


# ------------------------------------------------------------------------------------- Field-57

def field57_setup(keys, quick):
    cfg = json.loads(CONFIG.read_text())
    bundles, contracts, excluded = prepare('held_site', cfg, quick)
    data, _ = load_profiles()
    meta, frames = load_observations()
    context = data[['cast_id', 'depth_m']].merge(pd.read_csv(BOUNDARY), on=['cast_id', 'depth_m'],
                                                 validate='one_to_one')
    extras = context[EXTRAS].to_numpy(dtype=float)
    times = pd.to_datetime(data.t_utc, utc=True)
    query_bin = np.floor(data.depth_m.to_numpy() / BIN_M).astype(int)
    query_doy = times.dt.dayofyear.to_numpy()
    cast_doy = pd.Series(query_doy, index=data.cast_id.to_numpy()).groupby(level=0).first()
    full = archive()
    records = neighbour_records(data, meta, frames)
    depths = {m: ProbeDepth(m, meta) for m in {MODE[k] for k in keys if k in MODE}}
    mats = {k: [] for k in keys}
    backgrounds = {k: [] for k in keys}
    diagnostics = {k: [] for k in keys if k in MODE}
    gates = []
    for b, c in zip(bundles, contracts):
        cutoff = _utc(c['last_training'])
        train = full[full.station.isin(set(c['training_stations'])) & (full.t_utc <= cutoff)]
        delta, _ = harmonic_prior(train, query_bin, query_doy)
        e = extras.copy()
        e[:, 3:6] = delta
        r = weighted(deny(records, data, meta, c['held_station']))
        gates.append({'checkpoint': b['identity'], **context_gate(r, data, b['base'])})
        for k in keys:
            if k == 'A2':
                m = matrices(b, extras, False, False, False)
            elif k == 'A7':
                m = matrices(b, e, True, True, False)
            else:
                shift, diag = neighbour_offset(r, train, depths[MODE[k]], cast_doy)
                rows = rows_of(data.cast_id.to_numpy(), shift)
                bb = dict(b)
                bb['base'] = b['base'].astype(float) - rows
                m = matrices(bb, e, True, True, False)
                diagnostics[k].append({'checkpoint': b['identity'], 'fold': b['fold'],
                                       'held_station': c['held_station'], **diag,
                                       'background_change_on_scored_rows': shift_summary(rows, b['va'])})
            mats[k].append(m)
            backgrounds[k].append(m[0][b['va']])
    common = {'stations': data.station.to_numpy(), 'casts': data.cast_id.to_numpy(),
              'truth': data[TARGETS].to_numpy(float)}
    return {'bundles': bundles, 'contracts': contracts, 'excluded': excluded, 'data': data, 'meta': meta,
            'frames': frames, 'records': records, 'mats': mats, 'backgrounds': backgrounds,
            'diagnostics': diagnostics, 'context_gate': gates, 'common': common, 'depths': depths}


def _score(s):
    c = s['common']
    return lambda b, m, alpha: inner_score(b, m, alpha, False, c['stations'], c['casts'], c['truth'])


def _predict(s, k, alpha_by_fold):
    rows = []
    for b, (base, y, feats), bg in zip(s['bundles'], s['mats'][k], s['backgrounds'][k]):
        pred, _, _ = fit_predict(feats, base, y, b['tr'], b['va'], np.asarray(b['weights'], float),
                                 alpha_by_fold[b['fold']], False)
        f = prediction_frame(b, pred)
        for j, t in enumerate(TARGETS):
            f['background_' + t] = bg[:, j]
        rows.append(f)
    return pd.concat(rows, ignore_index=True)


def _compare_rows(mine, published):
    cols = [c for c in published.columns if c.startswith('pred_')]
    same = (len(mine) == len(published) and
            (mine[['cast_id', 'depth_m', 'fold']].to_numpy() == published[['cast_id', 'depth_m', 'fold']].to_numpy()).all())
    if not same:
        return {'rows_identical': False, 'pass': False}
    a, p = mine[cols].to_numpy(float), published[cols].to_numpy(float)
    diff = float(np.nanmax(np.abs(a - p)))
    rel = float(np.nanmax(np.abs(a - p) / np.maximum(np.abs(p), 1e-12)))
    # addendum 47 amendment 1: relative 1e-6 (float32 ridge design) as the passing rule
    return {'rows_identical': True, 'max_abs_prediction_diff': diff, 'max_relative_prediction_diff': rel,
            'pass': bool(rel <= 1e-6)}


def published_perfold(name):
    folder = PERFOLD / name
    return json.loads((folder / 'report.json').read_text()), pd.read_csv(folder / 'selected_predictions.csv.gz')


def run_field57(keys, quick=False):
    root = (OUT / 'smoke' if quick else OUT) / 'field57'
    t_setup = time.time()
    s = field57_setup(keys, quick)
    if not all(g['pass'] for g in s['context_gate']):
        raise SystemExit(f'the neighbour records do not reproduce the released background: {s["context_gate"]}')
    print(f'set up {len(s["bundles"])} checkpoints in {time.time() - t_setup:.0f} s; context gate passed', flush=True)
    score = _score(s)
    folds = sorted({b['fold'] for b in s['bundles']})
    for k in keys:
        name = FIELD57[k]
        folder = root / k
        t0 = time.time()
        alpha, sels = {}, {}
        for f in folds:
            pairs = [(b, m) for b, m in zip(s['bundles'], s['mats'][k]) if b['fold'] == f]

            def objective(trial, _pairs=pairs):
                a = trial.suggest_float('alpha', .1, 1e8, log=True)
                return float(np.mean([score(b, m, a) for b, m in _pairs]))
            sel = run_study(f'held_site_{name}_perfold_fold{f}', objective, folder, quick)
            sels[str(f)] = sel
            alpha[f] = sel['params']['alpha']
        d = _predict(s, k, alpha)
        digest = _csv(d, folder / 'selected_predictions.csv.gz')
        report = {'review': 'R4', 'declared_in': DECLARATION, 'smoke_only': quick, 'arm': name, 'key': k,
                  'design': 'perfold (one study per outer fold, as src/experiments/v2_perfold.py)',
                  'protocol': PROTOCOL['version'],
                  'background': {'A2': 'released context, no prior',
                                 'A7': 'released context taken at 3 m, plus the cast prior at the query depth',
                                 'A7d': 'context with each neighbour moved to 3 m by the cast prior at its measured depth, plus the prior at the query depth',
                                 'A7n': 'as A7d at each neighbour\'s nominal depth'}[k],
                  'reference_depth_m': REFERENCE_DEPTH_M, 'depth_bin_m': BIN_M,
                  'checkpoints': [{'fold': c['fold'], 'held_station': c['held_station']} for c in s['contracts']],
                  'selection': sels, 'alpha_by_fold': {str(f): v for f, v in alpha.items()},
                  'objective': "mean over that calendar fold's own checkpoints of the inner station-fold score",
                  'nested_mae': nested_mae(d), 'rows': int(len(d)), 'casts': int(d.cast_id.nunique()),
                  'context_gate': s['context_gate'], 'prediction_sha256': digest,
                  'seconds': round(time.time() - t0, 1)}
        if k in MODE:
            report['depth'] = {'mode': MODE[k], 'hourly_depth_sha256': s['depths'][MODE[k]].digest,
                               'by_checkpoint': s['diagnostics'][k]}
        if k in ('A2', 'A7') and not quick:
            pub_report, pub_rows = published_perfold(name)
            report['control_reproduces_published_perfold'] = {
                'alpha_published': pub_report['alpha_by_fold'], 'alpha_here': report['alpha_by_fold'],
                'alpha_identical': all(np.isclose(pub_report['alpha_by_fold'][str(f)], alpha[f], rtol=1e-9, atol=0)
                                       for f in folds),
                **_compare_rows(d, pub_rows)}
        _json(folder / 'report.json', report)
        print(f'{k:4s} {name}: alpha {report["alpha_by_fold"]}  T {report["nested_mae"]["temp_c"]:.4f}'
              f'  S {report["nested_mae"]["salinity_psu"]:.4f}  O {report["nested_mae"]["do_umol_l"]:.3f}'
              f'  ({report["seconds"]} s)', flush=True)


def _load_field57(root, keys, published):
    """Every arm's rows on one order; `published` adds the addendum-47 per-fold A2 and A7 rows."""
    sources = {k: root / k / 'selected_predictions.csv.gz' for k in keys}
    if published:
        sources.update({'pub:A2': PERFOLD / FIELD57['A2'] / 'selected_predictions.csv.gz',
                        'pub:A7': PERFOLD / FIELD57['A7'] / 'selected_predictions.csv.gz'})
    keys_frame, truth, preds, bgs = None, None, {}, {}
    for k, path in sources.items():
        f = pd.read_csv(path).sort_values(['fold', 'station', 'cast_id', 'depth_m']).reset_index(drop=True)
        kk = f[['fold', 'station', 'cast_id', 'depth_m']]
        if keys_frame is None:
            keys_frame, truth = kk, f[list(TARGETS)].to_numpy(float)
            day = pd.to_datetime(f.t_utc, utc=True).dt.strftime('%Y-%m-%d').to_numpy()
        elif not kk.equals(keys_frame) or not np.allclose(f[list(TARGETS)].to_numpy(float), truth, equal_nan=True):
            raise SystemExit(f'{k} does not score the rows of the others')
        preds[k] = f[[f'pred_{t}' for t in TARGETS]].to_numpy(float)
        if f'background_{TARGETS[0]}' in f:
            bgs[k] = f[[f'background_{t}' for t in TARGETS]].to_numpy(float)
    return keys_frame.assign(day=day), truth, preds, bgs


def _cd_contrast(keys, errs, arm, ref, label, folds=True):
    cd = CastDates(keys)
    value = cd.boot({'a': errs[arm], 'b': errs[ref]}, {'a': 1.0, 'b': -1.0}, label, DRAWS)
    out = {'value': [round(x, 6) for x in value], 'excludes_zero': bool(value[1] > 0 or value[2] < 0),
           'dates': int(cd.nd), 'casts': int(len(cd.cast_codes)), 'rows': int(len(keys))}
    if folds:
        out['by_fold'] = {}
        for f in sorted(keys.fold.unique()):
            m = (keys.fold == f).to_numpy()
            sub = CastDates(keys[m].reset_index(drop=True))
            ones = np.ones(len(sub.cast_codes))
            out['by_fold'][str(f)] = round(sub.nested(ones, sub.cast_means(errs[arm][m]))
                                           - sub.nested(ones, sub.cast_means(errs[ref][m])), 6)
    return out


def field57_report(quick=False):
    root = (OUT / 'smoke' if quick else OUT) / 'field57'
    keys, truth, preds, bgs = _load_field57(root, list(FIELD57), not quick)
    errs = {t: {k: np.abs(truth[:, j] - p[:, j]) for k, p in preds.items()} for j, t in enumerate(TARGETS)}
    berrs = {t: {k: np.abs(truth[:, j] - p[:, j]) for k, p in bgs.items()} for j, t in enumerate(TARGETS)}
    reports = {k: json.loads((root / k / 'report.json').read_text()) for k in FIELD57}
    gates = {}
    trials = [{'study': sel['study'], 'completed': sel['completed_trials']}
              for r in reports.values() for sel in r['selection'].values()]
    gates['trials'] = {'pass': quick or all(x['completed'] >= PROTOCOL['min_completed_trials'] for x in trials),
                       'studies': trials}
    gates['context'] = {'pass': all(g['pass'] for r in reports.values() for g in r['context_gate'])}
    if not quick:
        # The bootstrap here, on the published per-fold rows, gives the published per-fold contrast.
        published = json.loads(PERFOLD_JSON.read_text())['field57']['contrasts']
        label = 'CTD prior, harmonic, vs none'
        rows = []
        for t in TARGETS:
            got = CastDates(keys).boot({'a': errs[t]['pub:A7'], 'b': errs[t]['pub:A2']}, {'a': 1.0, 'b': -1.0},
                                       f'a47|perfold|{label}|{t}|date', DRAWS)
            want = next(c['perfold']['value'] for c in published if c['contrast'] == label and c['target'] == t)
            rows.append({'target': t, 'published': want, 'here': got,
                         'pass': bool(np.max(np.abs(np.array(got) - np.array(want))) <= 1e-9)})
        gates['bootstrap_reproduces_published_perfold'] = {'pass': all(x['pass'] for x in rows), 'targets': rows}
        controls = {k: reports[k].get('control_reproduces_published_perfold', {}) for k in ('A2', 'A7')}
        gates['controls_reproduce_published_perfold'] = {
            'pass': all(c.get('pass') and c.get('alpha_identical') for c in controls.values()), 'arms': controls,
            'blocking': False,
            'note': 'informational: the contrasts below use the controls re-run in this environment; when this '
                    'fails (another Optuna version), the depth arms are also set against the published rows'}
    blocking = [g for g, v in gates.items() if not v['pass'] and v.get('blocking', True)]
    if blocking:
        _json(root / 'report.json', {'gates': gates, 'stopped': blocking})
        raise SystemExit(f'gate(s) {blocking} failed; nothing is reported')
    out = {'review': 'R4', 'declared_in': DECLARATION, 'smoke_only': quick, 'cohort': 'Field-57',
           'rows': int(len(keys)), 'casts': int(keys.cast_id.nunique()), 'dates': int(keys.day.nunique()),
           'metric': 'nested macro MAE: depth bins within cast, casts within (fold, station), stations within fold, folds',
           'interval': f'cast-date bootstrap, {DRAWS} draws, a date drawn twice counting twice (CastDates)',
           'gates': gates, 'arms': {}, 'contrasts': [], 'background_only': [], 'depth_bands': {}}
    ones_cd = CastDates(keys)
    ones = np.ones(len(ones_cd.cast_codes))
    for k in FIELD57:
        out['arms'][k] = {'arm': FIELD57[k], 'alpha_by_fold': reports[k]['alpha_by_fold'],
                          'nested_mae': {t: round(ones_cd.nested(ones, ones_cd.cast_means(errs[t][k])), 6) for t in TARGETS},
                          'background_nested_mae': {t: round(ones_cd.nested(ones, ones_cd.cast_means(berrs[t][k])), 6)
                                                    for t in TARGETS}}
    refs = [('', '')] if quick or gates['controls_reproduce_published_perfold']['pass'] else [('', ''), ('pub:', 'published per-fold ')]
    for label, arm, ref in FIELD57_CONTRASTS:
        for t in TARGETS:
            for prefix, note in refs:
                if prefix and ref not in ('A2', 'A7'):
                    continue
                # A7 - A2 keeps addendum 47's stream, so it reproduces runs/v2_perfold.json when the controls do.
                stream = (f'a47|perfold|CTD prior, harmonic, vs none|{t}|date' if (arm, ref) == ('A7', 'A2')
                          else f'r4|field57|{prefix}{arm}-{ref}|{t}|date')
                e = {'contrast': note + label, 'arm': arm, 'reference': prefix + ref, 'target': t,
                     **_cd_contrast(keys, errs[t], arm, prefix + ref, stream)}
                out['contrasts'].append(e)
                print(f'{e["contrast"]:44s} {t:12s} {e["value"][0]:+.4f} [{e["value"][1]:+.4f}, {e["value"][2]:+.4f}]'
                      f'{"*" if e["excludes_zero"] else " "}  folds {e["by_fold"]}', flush=True)
    # Descriptive: the backgrounds alone (no ridge), which isolates the background from the refit.
    for arm, ref in (('A7d', 'A7'), ('A7n', 'A7'), ('A7d', 'A7n'), ('A7', 'A2')):
        for t in TARGETS:
            out['background_only'].append({'arm': arm, 'reference': ref, 'target': t,
                                           **_cd_contrast(keys, berrs[t], arm, ref, f'r4|field57|background|{arm}-{ref}|{t}|date')})
    depth = keys.depth_m.to_numpy()
    for lo, hi, band in BANDS:
        m = (depth >= lo) & (depth < hi)
        if not m.any():
            continue
        sub = keys[m].reset_index(drop=True)
        block = {'rows': int(m.sum()), 'casts': int(sub.cast_id.nunique()), 'dates': int(sub.day.nunique()), 'contrasts': {}}
        for arm, ref in BAND_CONTRASTS:
            block['contrasts'][f'{arm} - {ref}'] = {
                t: _cd_contrast(sub, {a: errs[t][a][m] for a in (arm, ref)}, arm, ref,
                                f'r4|field57|band|{band}|{arm}-{ref}|{t}|date', folds=True) for t in TARGETS}
        out['depth_bands'][band] = block
        c = block['contrasts']
        print(f'band {band:8s} rows {block["rows"]:4d} casts {block["casts"]:3d}  T: A7-A2 {c["A7 - A2"]["temp_c"]["value"][0]:+.4f}'
              f'  A7d-A2 {c["A7d - A2"]["temp_c"]["value"][0]:+.4f}  A7d-A7 {c["A7d - A7"]["temp_c"]["value"][0]:+.4f}', flush=True)
    _json(root / 'report.json', out)
    print('wrote', (root / 'report.json').relative_to(ROOT).as_posix())


# ------------------------------------------------------------------------------------ Field-2026

def dev2026_matrices(arm, s, full, query_bin, query_doy, cast_doy, depths):
    mats, diags = [], []
    for b, c in zip(s['bundles'], s['contracts']):
        extras = np.zeros((len(s['data']), 8))
        bb = b
        if arm != 'A2_none':
            cutoff = _utc(c['last_training'])
            train = full[full.station.isin(set(c['training_stations'])) & (full.t_utc <= cutoff)]
            extras[:, 3:6] = harmonic_prior(train, query_bin, query_doy)[0]
            if arm in MODE:
                r = weighted(deny(s['records'], s['data'], s['meta'], c['held_station']))
                shift, diag = neighbour_offset(r, train, depths[MODE[arm]], cast_doy)
                rows = rows_of(s['data'].cast_id.to_numpy(), shift)
                bb = dict(b)
                bb['base'] = b['base'].astype(float) - rows
                diags.append({'checkpoint': b['identity'], **diag,
                              'background_change_on_scored_rows': shift_summary(rows, b['va'])})
        mats.append(matrices(bb, extras, arm != 'A2_none', False, False))
    return mats, diags


def development2026(out, quick):
    cfg = json.loads(CONFIG.read_text())
    bundles, contracts, _ = prepare('held_site', cfg, quick)
    data, _ = load_profiles()
    meta, frames = load_observations()
    full = archive()
    query_bin = np.floor(data.depth_m.to_numpy() / BIN_M).astype(int)
    query_doy = data.t_utc.dt.dayofyear.to_numpy()
    cast_doy = pd.Series(query_doy, index=data.cast_id.to_numpy()).groupby(level=0).first()
    s = {'bundles': bundles, 'contracts': contracts, 'data': data, 'meta': meta,
         'records': neighbour_records(data, meta, frames)}
    depths = {m: ProbeDepth(m, meta) for m in ('measured', 'nominal')}
    stations, casts, truth = data.station.to_numpy(), data.cast_id.to_numpy(), data[TARGETS].to_numpy(float)
    result, preds = {}, {}
    for arm in FIELD2026:
        t0 = time.time()
        mats, diags = dev2026_matrices(arm, s, full, query_bin, query_doy, cast_doy, depths)

        def objective(trial, mats=mats):
            alpha = trial.suggest_float('alpha', .1, 1e8, log=True)
            return float(np.mean([inner_score(b, m, alpha, False, stations, casts, truth)
                                  for b, m in zip(bundles, mats)]))
        sel = run_study(f'held_site_{arm}_dev', objective, out / 'studies', quick)
        rows = []
        for b, (base, y, feats) in zip(bundles, mats):
            pred, _, _ = fit_predict(feats, base, y, b['tr'], b['va'], np.asarray(b['weights'], float),
                                     sel['params']['alpha'], False)
            rows.append(prediction_frame(b, pred))
        d = pd.concat(rows, ignore_index=True)
        preds[arm] = d
        result[arm] = {'selection': sel, 'nested_mae': nested_mae(d), 'casts': int(d.cast_id.nunique()),
                       'seconds': round(time.time() - t0, 1)}
        if diags:
            result[arm]['depth'] = {'mode': MODE[arm], 'by_checkpoint': diags}
        print(f'development {arm}: alpha {sel["params"]["alpha"]:.6g}  T {result[arm]["nested_mae"]["temp_c"]:.4f}'
              f'  ({result[arm]["seconds"]} s)', flush=True)
    published = json.loads(RESERVED_REPORT.read_text())['development']
    a2, a7 = result['A2_none'], result['A7o']
    result['gate_controls_reproduce_published'] = {
        'A2_none': bool(np.isclose(a2['selection']['params']['alpha'], PUBLISHED_A2['alpha'], rtol=1e-6)
                        and np.isclose(a2['nested_mae']['temp_c'], PUBLISHED_A2['temp_c'], rtol=1e-6)),
        'A7o': bool(np.isclose(a7['selection']['params']['alpha'], published['A7o']['selection']['params']['alpha'], rtol=1e-6)
                    and all(np.isclose(a7['nested_mae'][t], published['A7o']['nested_mae'][t], rtol=1e-6) for t in TARGETS))}
    return result, preds


def reserved_depth(alphas, arms, held_out=True, start=START, end=END):
    """v2_field_reserved2026.reserved with the depth-aware arms beside A2_none and A7o: the same rows,
    context, training set, weights, prior and seasonal climatology."""
    pre, _ = load_profiles()
    pre = pre[pre.t_utc < start]
    new = profiles_window(start, end)
    data = pd.concat([pre, new], ignore_index=True)
    meta, frames = observations_through(end)
    full = archive()
    y = data[TARGETS].to_numpy(float)
    query_bin = np.floor(data.depth_m.to_numpy() / BIN_M).astype(int)
    query_doy = data.t_utc.dt.dayofyear.to_numpy()
    cast_doy = pd.Series(query_doy, index=data.cast_id.to_numpy()).groupby(level=0).first()
    records = neighbour_records(data, meta, frames)
    depths = {m: ProbeDepth(m, meta) for m in {MODE[a] for a in arms if a in MODE}}
    held = sorted(new.station.unique()) if held_out else [None]
    frames_out, diags, gates = [], {a: [] for a in arms if a in MODE}, []
    for s in held:
        X = context_features(data, meta, frames, s)
        test_station = new.station.unique() if s is None else [s]
        tr = np.flatnonzero((data.t_utc < start - GAP).to_numpy() & (data.station != s).to_numpy())
        te = np.flatnonzero((data.t_utc >= start).to_numpy() & data.station.isin(test_station).to_numpy())
        fallback = np.median(y[tr], axis=0)
        context = X[['temperature_context', 'salinity_context', 'oxygen_context']].to_numpy().copy()
        base = context.copy()
        for j in range(3):
            base[~np.isfinite(base[:, j]), j] = fallback[j]
        imputer = SimpleImputer(strategy='median', add_indicator=True, keep_empty_features=True).fit(X.iloc[tr])
        raw = imputer.transform(X)
        sx = StandardScaler().fit(raw[tr]).transform(raw)
        training = data.iloc[tr]
        w = 1 / (training.groupby('cast_id').cast_id.transform('size').to_numpy()
                 * training.groupby('station').cast_id.transform('nunique').to_numpy())
        w = w / w.mean()
        last = data.t_utc.iloc[tr].max()
        prior_rows = full[(full.station != s) & (full.t_utc <= last)] if s is not None else full[full.t_utc <= last]
        delta = harmonic_prior(prior_rows, query_bin, query_doy)[0]
        r = weighted(deny(records, data, meta, s))
        gates.append({'held_station': s, **context_gate(r, data, context, exact=True)})
        clim_rows = full[(full.station != s) & (full.t_utc < start)] if s is not None else full[full.t_utc < start]
        frame = data.iloc[te][['cast_id', 'station', 't_utc', 'depth_m'] + TARGETS].reset_index(drop=True)
        frame['fold'] = 0
        for arm in arms:
            if arm == 'A2_none':
                bs = base + 0.0
            elif arm == 'A7o':
                bs = base + delta
            else:
                shift, diag = neighbour_offset(r, prior_rows, depths[MODE[arm]], cast_doy)
                rows = rows_of(data.cast_id.to_numpy(), shift)
                bs = (base - rows) + delta
                diags[arm].append({'held_station': s, **diag, 'background_change_on_scored_rows': shift_summary(rows, te)})
            bs[:, 0] = np.clip(bs[:, 0], -2, 40)
            bs[:, 1] = np.clip(bs[:, 1], 0, 42)
            pred, _, _ = fit_predict(sx, bs, y, tr, te, w, alphas[arm], False)
            for j, t in enumerate(TARGETS):
                frame[f'{arm}|{t}'] = pred[:, j]
                frame[f'background_{arm}|{t}'] = bs[te, j]
        clim = seasonal(clim_rows, data.iloc[te])
        for j, t in enumerate(TARGETS):
            frame[f'seasonal|{t}'] = clim[:, j]
        frames_out.append(frame)
        print('held' if held_out else 'seen', s, 'train casts', training.cast_id.nunique(),
              'test casts', frame.cast_id.nunique(), flush=True)
    return pd.concat(frames_out, ignore_index=True), diags, gates


def cast_errors(d, arm, target):
    e = (d[target] - d[f'{arm}|{target}']).abs()
    return d.assign(e=e).groupby(['fold', 'station', 'cast_id']).agg(
        e=('e', 'mean'), day=('t_utc', lambda x: x.iloc[0].floor('D'))).reset_index()


def macro(c):
    return float(c.groupby(['fold', 'station']).e.mean().groupby('fold').mean().mean())


def contrast(ca, cb, draws=DRAWS, decimals=6):
    """v2_field_reserved2026.contrast with the same draws (one generator seeded RNG_SEED per contrast,
    `choice` over the dates in order of appearance, each drawn date's casts relabelled), computed from
    date multiplicities instead of concatenated frames: a cast drawn m times weighs m in its (fold,
    station) mean, which is what the relabelled concatenation averages. Stored at six decimals, as every
    stored report; `decimals=4` gives the published rounding, which the gates compare."""
    j = ca.merge(cb[['fold', 'station', 'cast_id', 'e']], on=['fold', 'station', 'cast_id'], suffixes=('_a', '_b'))
    point = macro(j.rename(columns={'e_a': 'e'})) - macro(j.rename(columns={'e_b': 'e'}))
    days = j.day.unique()
    n = len(days)
    day_index = pd.Index(days).get_indexer(j.day)
    cell, cells = pd.factorize(j.fold.astype(str) + '|' + j.station.astype(str))
    fold = j.fold.to_numpy()
    cell_fold = np.array([fold[cell == c][0] for c in range(len(cells))])
    folds = np.unique(cell_fold)
    ea, eb = j.e_a.to_numpy(float), j.e_b.to_numpy(float)

    def wmacro(e, w):
        tot = np.bincount(cell, weights=w * e, minlength=len(cells))
        cnt = np.bincount(cell, weights=w, minlength=len(cells))
        present = cnt > 0
        return float(np.mean([np.mean(tot[present & (cell_fold == f)] / cnt[present & (cell_fold == f)])
                              for f in folds if (present & (cell_fold == f)).any()]))
    rng = np.random.default_rng(RNG_SEED)
    boot = np.empty(draws)
    for i in range(draws):
        pick = rng.choice(n, n, replace=True)          # the stream of rng.choice(days, n, replace=True)
        w = np.bincount(pick, minlength=n).astype(float)[day_index]
        boot[i] = wmacro(ea, w) - wmacro(eb, w)
    lo, hi = np.percentile(boot, [2.5, 97.5])
    jack = []
    for k in range(n):
        w = (day_index != k).astype(float)
        jack.append(wmacro(ea, w) - wmacro(eb, w))
    se = np.sqrt((n - 1) / n * np.sum((np.array(jack) - np.mean(jack)) ** 2))
    t = stats.t.ppf(.975, n - 1)
    r = lambda x: round(float(x), decimals)
    return {'point': r(point), 'ci': [r(lo), r(hi)], 'excludes_zero': bool(lo > 0 or hi < 0),
            'jackknife_t': [r(point - t * se), r(point + t * se)],
            'jackknife_excludes_zero': bool(point - t * se > 0 or point + t * se < 0), 'dates': int(n),
            'casts': int(j.cast_id.nunique())}


def read_block(d, arms, quick):
    draws = SMOKE_DRAWS if quick else DRAWS
    cols = list(arms) + [f'background_{a}' for a in arms] + ['seasonal']
    block = {'casts': int(d.cast_id.nunique()), 'dates': int(d.t_utc.dt.floor('D').nunique()),
             'stations': d.groupby('station').cast_id.nunique().to_dict(),
             'nested_mae': {a: {t: round(macro(cast_errors(d, a, t)), 6) for t in TARGETS} for a in cols},
             'contrasts': {}, 'depth_bands': {}}
    for a, b in FIELD2026_CONTRASTS:
        block['contrasts'][f'{a} - {b}'] = {t: contrast(cast_errors(d, a, t), cast_errors(d, b, t), draws) for t in TARGETS}
    for a in ('A7od', 'A7on'):
        block[f'per_station_{a}_minus_A7o'] = {
            t: {s: round(float(g.e_a.mean() - g.e_b.mean()), 6) for s, g in
                cast_errors(d, a, t).merge(cast_errors(d, 'A7o', t), on=['fold', 'station', 'cast_id', 'day'],
                                           suffixes=('_a', '_b')).groupby('station')} for t in TARGETS}
    for lo, hi, band in BANDS:
        sub = d[(d.depth_m >= lo) & (d.depth_m < hi)]
        if sub.empty:
            continue
        entry = {'rows': int(len(sub)), 'casts': int(sub.cast_id.nunique()),
                 'dates': int(sub.t_utc.dt.floor('D').nunique()), 'contrasts': {}}
        for a, b in (('A7o', 'A2_none'), ('A7od', 'A2_none'), ('A7on', 'A2_none'), ('A7od', 'A7o')):
            entry['contrasts'][f'{a} - {b}'] = {
                t: (contrast(cast_errors(sub, a, t), cast_errors(sub, b, t), draws)
                    if sub.t_utc.dt.floor('D').nunique() >= 2 else None) for t in TARGETS}
        block['depth_bands'][band] = entry
    return block


def published_read_gate(d, published):
    """The controls on the read reproduce the published nested errors and the A7o - A2_none contrast,
    both rounded once from full precision to the four decimals the published report stores."""
    rows = []
    for a in ('A2_none', 'A7o', 'background_A2_none', 'background_A7o', 'seasonal'):
        for t in TARGETS:
            full_precision = macro(cast_errors(d, a, t))
            here, want = round(full_precision, 4), published['nested_mae'][a][t]
            rows.append({'quantity': f'{a}|{t}', 'published': want, 'here': here, 'full_precision': full_precision,
                         'pass': bool(abs(here - want) < 5e-9)})
    for t in TARGETS:
        c = contrast(cast_errors(d, 'A7o', t), cast_errors(d, 'A2_none', t), decimals=4)
        want = published['contrasts']['A7o - A2_none'][t]
        rows.append({'quantity': f'A7o - A2_none|{t} contrast', 'published': want, 'here': c,
                     'pass': all(c[k] == want[k] for k in ('point', 'ci', 'jackknife_t', 'dates'))})
    return rows


def run_field2026(quick=False):
    out = (OUT / 'smoke' if quick else OUT) / 'field2026'
    out.mkdir(parents=True, exist_ok=True)
    dev, dev_preds = development2026(out, quick)
    report = {'review': 'R4', 'declared_in': DECLARATION, 'smoke_only': quick, 'cohort': 'Field-2026',
              'reading': 'a replication on a reused cohort (the 38 casts of January-June 2026)',
              'mirrors': 'src/experiments/v2_field_reserved2026.py (addendum 36)', 'development': dev}
    for arm, d in dev_preds.items():
        _csv(d, out / f'dev_{arm}_predictions.csv.gz')
    dev_rows = {a: dev_preds[a].assign(**{f'{a}|{t}': dev_preds[a]['pred_' + t] for t in TARGETS}) for a in FIELD2026}
    report['development']['contrasts'] = {
        f'{a} - {b}': {t: contrast(cast_errors(dev_rows[a], a, t), cast_errors(dev_rows[b], b, t), SMOKE_DRAWS if quick else DRAWS)
            for t in TARGETS}
        for a, b in (('A7od', 'A7o'), ('A7on', 'A7o'), ('A7od', 'A7on'), ('A7o', 'A2_none'), ('A7od', 'A2_none'))}
    gate = dev['gate_controls_reproduce_published']
    print('development controls reproduce the published studies:', gate, flush=True)
    if not quick and not all(gate.values()):
        _json(out / 'report.json', report)
        raise SystemExit('a development control did not reproduce its published study; 2026 not read.')
    alphas = {a: dev[a]['selection']['params']['alpha'] for a in FIELD2026}
    window = (pd.Timestamp('2025-07-01', tz='UTC'), START) if quick else (START, END)
    report['read_window'] = [str(window[0]), str(window[1])]
    published = json.loads(RESERVED_REPORT.read_text())
    reads = {}
    for label, held_out in (('held_station', True), ('station_seen', False)):
        d, diags, gates = reserved_depth(alphas, FIELD2026, held_out, *window)
        if not all(g['pass'] for g in gates):
            raise SystemExit(f'the neighbour records do not reproduce context_features: {gates}')
        block = read_block(d, FIELD2026, quick)
        block['context_gate'] = gates
        block['depth'] = diags
        if not quick:
            rows = published_read_gate(d, published[label])
            block['gate_controls_reproduce_published'] = {'pass': all(x['pass'] for x in rows), 'quantities': rows}
            if not block['gate_controls_reproduce_published']['pass']:
                report[label] = {'gate_controls_reproduce_published': block['gate_controls_reproduce_published']}
                _json(out / 'report.json', report)
                raise SystemExit(f'{label}: the controls do not reproduce the published read; depth arms not reported')
        block['prediction_sha256'] = _csv(d, out / f'reserved2026_{label}_predictions.csv.gz')
        reads[label] = block
        report[label] = block
        print(label, json.dumps({a: block['nested_mae'][a]['temp_c'] for a in FIELD2026}), flush=True)
    for arm in FIELD2026:
        _json(out / arm / 'report.json', {
            'review': 'R4', 'arm': arm, 'smoke_only': quick, 'depth_mode': MODE.get(arm),
            'development': {k: v for k, v in dev[arm].items()},
            'read_window': report['read_window'],
            **{label: {'nested_mae': reads[label]['nested_mae'][arm],
                       'background_nested_mae': reads[label]['nested_mae'][f'background_{arm}'],
                       **({'depth': reads[label]['depth'][arm]} if arm in MODE else {})} for label in reads}})
    _json(out / 'report.json', report)
    h = report['held_station']['contrasts']
    for k in ('A7od - A7o', 'A7od - A2_none', 'A7on - A7o', 'A7o - A2_none'):
        print(k, json.dumps(h[k]['temp_c']), flush=True)
    print('wrote', (out / 'report.json').relative_to(ROOT).as_posix())


# ------------------------------------------------------------------------------------------ check

def check(with_2026=False):
    """Gates that need no new study and score no depth arm: (1) the neighbour records reproduce
    context_features exactly at every held station; (2) A2 and A7 refitted here at the published per-fold
    penalties reproduce the published per-fold rows, and the per-fold objective at that penalty gives the
    study's best value; (3) the cast-date bootstrap reproduces the published per-fold A7 - A2 contrast;
    (4) the QC'd probe depths sit on the metrology medians. With --with-2026, (5) A2_none and A7o at the
    published penalties reproduce the published 2026 read (only already-published numbers are recomputed)."""
    out = {'review': 'R4', 'what': check.__doc__}
    s = field57_setup(['A2', 'A7'], False)
    exact = []
    for held in sorted({c['held_station'] for c in s['contracts']}):
        X = context_features(s['data'], s['meta'], s['frames'], held)
        r = weighted(deny(s['records'], s['data'], s['meta'], held))
        ref = X[['temperature_context', 'salinity_context', 'oxygen_context']].to_numpy()
        exact.append({'held_station': held, **context_gate(r, s['data'], ref, exact=True)})
    out['G1_records_reproduce_context_features'] = {'pass': all(x['pass'] for x in exact), 'stations': exact,
                                                    'float32_bundles': s['context_gate']}
    score = _score(s)
    g2 = {}
    for k in ('A2', 'A7'):
        pub_report, pub_rows = published_perfold(FIELD57[k])
        alpha = {int(f): v for f, v in pub_report['alpha_by_fold'].items()}
        d = _predict(s, k, alpha)
        objective = {}
        for f, sel in pub_report['selection'].items():
            value = float(np.mean([score(b, m, alpha[int(f)]) for b, m in zip(s['bundles'], s['mats'][k])
                                   if b['fold'] == int(f)]))
            objective[f] = {'published_best_value': sel['best_value'], 'here': value,
                            'abs_diff': abs(value - sel['best_value']), 'pass': bool(abs(value - sel['best_value']) <= 1e-9)}
        nm = nested_mae(d)
        g2[k] = {'alpha_by_fold': pub_report['alpha_by_fold'], **_compare_rows(d, pub_rows), 'objective': objective,
                 'nested_mae_here': nm, 'nested_mae_published': pub_report['nested_mae'],
                 'nested_mae_max_abs_diff': max(abs(nm[t] - pub_report['nested_mae'][t]) for t in TARGETS)}
        g2[k]['pass'] = bool(g2[k]['pass'] and all(o['pass'] for o in objective.values())
                             and g2[k]['nested_mae_max_abs_diff'] <= 1e-6)
        print(f'{k}: rows {g2[k]["pass"]} (max rel {g2[k].get("max_relative_prediction_diff")}), '
              f'nested MAE T {nm["temp_c"]:.6f} vs {pub_report["nested_mae"]["temp_c"]:.6f}', flush=True)
    out['G2_controls_at_published_perfold_penalties'] = g2
    rows ={k: pd.read_csv(PERFOLD / FIELD57[k] / 'selected_predictions.csv.gz')
            .sort_values(['fold', 'station', 'cast_id', 'depth_m']).reset_index(drop=True) for k in ('A2', 'A7')}
    keys = rows['A2'][['fold', 'station', 'cast_id', 'depth_m']].assign(
        day=pd.to_datetime(rows['A2'].t_utc, utc=True).dt.strftime('%Y-%m-%d').to_numpy())
    published = json.loads(PERFOLD_JSON.read_text())['field57']['contrasts']
    label = 'CTD prior, harmonic, vs none'
    g3 = []
    for t in TARGETS:
        e = {k: np.abs(rows[k][t].to_numpy() - rows[k]['pred_' + t].to_numpy()) for k in rows}
        got = CastDates(keys).boot({'a': e['A7'], 'b': e['A2']}, {'a': 1.0, 'b': -1.0}, f'a47|perfold|{label}|{t}|date', DRAWS)
        want = next(c['perfold']['value'] for c in published if c['contrast'] == label and c['target'] == t)
        g3.append({'target': t, 'published': want, 'here': got,
                   'pass': bool(np.max(np.abs(np.array(got) - np.array(want))) <= 1e-9)})
    out['G3_bootstrap_reproduces_published_perfold_contrast'] = {'pass': all(x['pass'] for x in g3), 'targets': g3}
    summary = json.loads(probe_depth.SUMMARY.read_text(encoding='utf-8'))
    diffs = {p: v.get('qc_median_minus_G1_median') for p, v in summary['probes'].items()}
    out['G4_qc_depth_medians_vs_metrology'] = {'max_abs_m': max(abs(x) for x in diffs.values() if x is not None),
                                              'by_probe': diffs, 'totals': summary['totals']}
    if with_2026:
        published = json.loads(RESERVED_REPORT.read_text())
        alphas = {a: published['development'][a]['selection']['params']['alpha'] for a in ('A2_none', 'A7o')}
        g5 = {}
        for label, held_out in (('held_station', True), ('station_seen', False)):
            d, _, gates = reserved_depth(alphas, ('A2_none', 'A7o'), held_out)
            rows5 = published_read_gate(d, published[label])
            g5[label] = {'pass': all(x['pass'] for x in rows5) and all(g['pass'] for g in gates),
                         'context_gate': gates, 'quantities': rows5}
            print(label, 'controls reproduce the published 2026 read:', g5[label]['pass'], flush=True)
        out['G5_controls_reproduce_published_2026_read'] = g5
    _json(OUT / 'check.json', out)
    for k, v in out.items():
        if isinstance(v, dict) and 'pass' in v:
            print(k, v['pass'])
    for k in ('A2', 'A7'):
        print('G2', k, g2[k]['pass'])
    print('wrote', (OUT / 'check.json').relative_to(ROOT).as_posix())


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest='cmd', required=True)
    c = sub.add_parser('check')
    c.add_argument('--with-2026', action='store_true')
    f = sub.add_parser('field57')
    f.add_argument('--arm', nargs='+', default=list(FIELD57), choices=list(FIELD57))
    f.add_argument('--quick', action='store_true')
    r = sub.add_parser('field57-report')
    r.add_argument('--quick', action='store_true')
    g = sub.add_parser('field2026')
    g.add_argument('--quick', action='store_true')
    a = ap.parse_args()
    if a.cmd == 'check':
        check(a.with_2026)
    elif a.cmd == 'field57':
        run_field57(a.arm, a.quick)
    elif a.cmd == 'field57-report':
        field57_report(a.quick)
    else:
        run_field2026(a.quick)


if __name__ == '__main__':
    main()
