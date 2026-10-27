"""Addendum 85: every forecaster, the silent-probe arms and every gap method on common rows.

Reads stored predictions only (and part B's 2026 trees from src/experiments/v2_common_rows.py); fits
nothing. Four parts, declared in config/bay_reserved_protocol_v2_addendum85.json:

  A  Bridge-Dev and Bridge-2026, 6-72 h: every forecast issued at onset against the arms that read the
     network after onset, on the same episodes; recursive minus direct at matched lead (addendum 4)
  B  Dev-2025 and Reserved-2026, 1-72 h: every forecaster on the distribution study's rows
  C  the six-hour Dev-2025 tree: shared and per fold on the reference grid, per fold over six grids
  D  every gap method and pretrained gap arm per cohort and length, against linear interpolation and
     the incumbent, episode macro

Within a cell, the arms covering at least 95 % of its base rows are its core arms and the common rows
are those on which every core arm is finite; core arms are scored and contrasted on the common rows,
with one bootstrap stream per cell applied to every arm's per-row error, so every contrast is paired
draw by draw (v2_probabilistic_report.joint_draws: weeks drawn jointly within each fold or quarter, the
scheme of v2_bridge.joint_week_draws). An arm below 95 % is scored on its own paired rows, each
contrast on its own stream. Gap cells use v2_fm_zoo_score_gaps.Cell (overlap clusters or probe x month,
jackknife beside). Oxygen is computed in the stored column and carried in mg/l beside.

Gates A85-G5 (rows), G6 (published intervals reproduced) and G7 (the printed six-hour values) run
before any new contrast is read; a failure stops the part it concerns.

Writes reports/bay_twin/runs/v2_addendum85.json and runs/v2_addendum85/{A,B,C,D}.json.
Usage: python -m scripts.v2_common_rows_report [--parts A,B,C,D]
"""
import argparse
import json
import sys
import time
import zlib
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import scripts.v2_fdr as FDR  # noqa: E402
import scripts.v2_fm_zoo_score_dist_bridge as Z  # noqa: E402
from scripts.v2_fm_zoo_io import ARMS as ZOO_ARMS, load_inputs, load_predictions  # noqa: E402
from src.experiments import v2_bridge72  # noqa: E402
from src.experiments import v2_common_rows as CR  # noqa: E402
from src.experiments import v2_probabilistic_report as R  # noqa: E402
from src.units import factor  # noqa: E402

RUNS = ROOT / 'reports/bay_twin/runs'
OUT = RUNS / 'v2_addendum85'
SUMMARY = RUNS / 'v2_addendum85.json'
DECLARED = 'config/bay_reserved_protocol_v2_addendum85.json'
C_MG = factor()
VARIABLES = ('temperature', 'oxygen')
B_COHORTS = ('Dev-2025', 'Reserved-2026')
A_COHORTS = ('Bridge-Dev', 'Bridge-2026')
TO_FORECAST = {'Bridge-Dev': 'Dev-2025', 'Bridge-2026': 'Reserved-2026'}
B_LEADS = (1, 3, 6, 12, 24, 36, 48, 72)
A_LEADS = (6, 12, 18, 24, 30, 36, 42, 48, 54, 60, 66, 72)
KEY = ['cohort', 'sensor', 'time', 'lead']
CORE_SHARE = 0.95
MIN_UNITS = 5
MIN_CELL_ROWS = 5
TOL = 1e-9

#: forecasts issued at onset (nothing read after issuance)
FORECASTS = ('persistence', 'clim_hour', 'clim_month_hour', 'tree', 'weather_tree', 'tree_shared',
             'weather_tree_shared', 'quantile_tree', 'quantile_tree_weather', 'clim_quantiles', 'linear_ar',
             'arima_ss', 'nbeats', 'chronos2_net', 'blended', 'learned_source', 'conservative_exchange',
             'chronos_bolt_small_uni', 'chronos_bolt_base_uni', 'chronos2_uni', 'timesfm25_uni', 'timesfm3_uni',
             'timesfm25_cov_past', 'timesfm3_cov_past')
#: arms that read other probes after onset (part A)
NETWORK = ('anchored_direct', 'neighbour_change', 'recursive_tree', 'graph_conservative', 'graph_unconstrained',
           'chronos2_cov', 'timesfm25_cov', 'timesfm3_cov')
#: arms that read other probes as covariates or after issuance: the co-located floor applies to their
#: temperature contrasts (the published convention of v2_fm_zoo_score_dist_bridge.floor_for)
READS_OTHERS = set(NETWORK) | {'chronos2_net', 'timesfm25_cov_past', 'timesfm3_cov_past'}
LABELS = {'persistence': 'last value', 'last_value': 'last value', 'clim_hour': 'hour-of-day increment',
          'clim_month_hour': 'month-by-hour increment', 'tree': 'LightGBM, per fold',
          'weather_tree': 'LightGBM + weather, per fold', 'tree_shared': 'LightGBM, shared selection',
          'weather_tree_shared': 'LightGBM + weather, shared selection',
          'quantile_tree': 'quantile LightGBM (median)', 'quantile_tree_weather': 'quantile LightGBM + weather (median)',
          'clim_quantiles': 'climatological quantiles (median)', 'linear_ar': 'linear autoregression',
          'arima_ss': 'state-space ARIMA', 'nbeats': 'N-BEATS', 'chronos2_net': 'Chronos-2 + network past',
          'blended': 'LightGBM + weather blended with the last value', 'learned_source': 'graph, learned source',
          'conservative_exchange': 'graph, conservative exchange',
          'chronos_bolt_small_uni': 'Chronos-Bolt small', 'chronos_bolt_base_uni': 'Chronos-Bolt base',
          'chronos2_uni': 'Chronos-2', 'timesfm25_uni': 'TimesFM 2.5', 'timesfm3_uni': 'TimesFM 3.0',
          'anchored_direct': 'network-aided forecast', 'neighbour_change': 'neighbour rule',
          'recursive_tree': 'rollout', 'graph_conservative': 'rolled graph, conservative',
          'graph_unconstrained': 'rolled graph, learned source', 'chronos2_cov': 'Chronos-2 + network',
          'timesfm25_cov': 'TimesFM 2.5 + network', 'timesfm3_cov': 'TimesFM 3.0 + network',
          'timesfm25_cov_past': 'TimesFM 2.5 + network past', 'timesfm3_cov_past': 'TimesFM 3.0 + network past'}


# --------------------------------------------------------------------------------------------------
# small helpers
# --------------------------------------------------------------------------------------------------
def r6(x):
    return None if x is None or not np.isfinite(x) else round(float(x), 6)


def utc(values):
    return pd.to_datetime(values, utc=True).astype('datetime64[ns, UTC]')


def rel(path):
    return Path(path).resolve().relative_to(ROOT).as_posix()


def write(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(json.dumps(obj, indent=1, default=lambda o: o.item() if hasattr(o, 'item') else str(o))
                     .encode('utf-8'))


_FLOORS = {}


def floors():
    if not _FLOORS:
        _FLOORS.update(Z.floors())
    return _FLOORS


def against(point, lo, hi, floor):
    """v2_summary_table.against: an interval spanning zero is not read against a floor."""
    if lo is not None and not (lo > 0 or hi < 0):
        return 'interval spans zero'
    if lo is not None and (lo > floor or hi < -floor):
        return 'clears floor'
    return 'exceeds floor' if abs(point) > floor else 'below floor'


def decorate(c, variable, x, y, reads_others):
    """Floors and p-value of a contrast stored in the variable's units."""
    point, ci = c['point'], c.get('ci')
    lo, hi = (None, None) if ci is None else ci
    fl = floors()
    if variable == 'oxygen':
        c['point_mgl'] = r6(point * C_MG)
        c['ci_mgl'] = None if ci is None else [r6(v * C_MG) for v in ci]
        c['floor'] = fl['oxygen_mgl']
        c['floor_basis'] = 'sonde (mg/l)'
        c['reading'] = against(point * C_MG, None if lo is None else lo * C_MG, None if hi is None else hi * C_MG,
                               fl['oxygen_mgl'])
    else:
        other = x in reads_others or y in reads_others
        main, alt = ((fl['temperature_colocated'], fl['temperature_quantisation']) if other
                     else (fl['temperature_quantisation'], fl['temperature_colocated']))
        c['floor'] = main
        c['floor_basis'] = ('co-located disagreement (an arm reads other probes)' if other
                            else 'quantisation (no arm reads another instrument after issuance)')
        c['reading'] = against(point, lo, hi, main)
        c['reading_against_other_floor'] = {'floor': alt, 'reading': against(point, lo, hi, alt)}
    if ci is None:
        c['reading'] = 'n.e. (no interval)'
        if 'reading_against_other_floor' in c:
            c['reading_against_other_floor']['reading'] = 'n.e. (no interval)'
    p = None if ci is None else FDR.p_side(point, lo, hi)
    c['p'] = None if p is None else float(f'{p:.9g}')
    return c


# --------------------------------------------------------------------------------------------------
# The forecasting sources (parts A and B)
# --------------------------------------------------------------------------------------------------
PROB = RUNS / 'v2_probabilistic'
ORACLE = RUNS / 'v2_weather_oracle'
HORIZON_RUNS = (RUNS / '20260920T083322Z_v2_horizon/results', RUNS / '20260920T133747Z_v2_horizon/results')
ADD83 = RUNS / 'v2_addendum83/perfold'
GRAPHS = RUNS / 'v2_perfold_neural/graph/perfold'
#: the blended weather tree's 2026 six-hour rows (the declared reading of the six-hour forecaster), scored
#: where anchor and target exist; the battery report's MODEL_v2 error is its published value
BLENDED = RUNS / 'v2_reserved_reading_v2/forecast_endpoints_2026.csv.gz'
BLENDED_REPORT = RUNS / 'v2_reserved_reading_v2/battery_full_v2.json'
RESERVED_H24 = RUNS / 'v2_horizon_reserved_h24'
QUANTILE_ARMS = ('quantile_tree', 'quantile_tree_weather', 'clim_quantiles', 'linear_ar', 'arima_ss', 'nbeats',
                 'chronos2_net')


def points(variable):
    """The base rows of part B: the distribution study's points rows, 1-48 h and 72 h."""
    p = pd.concat([pd.read_csv(PROB / 'seed42' / f'points_{variable}.csv.gz'),
                   pd.read_csv(ORACLE / 'seed42' / f'points72_{variable}.csv.gz')], ignore_index=True)
    p['time'] = utc(p.time)
    dev = (p.cohort == 'Dev-2025').to_numpy()
    month = p.time.dt.month.to_numpy()
    fold = np.where(dev, p.fold.fillna(-1).astype(int).astype(str), np.where(month <= 3, 'Q1', 'Q2'))
    p['fold'] = fold
    p['week'] = p.time.dt.isocalendar().week.astype(int).to_numpy()
    if p.duplicated(KEY).any():
        raise SystemExit(f'{variable}: duplicated points rows')
    return p.rename(columns={'last': 'persistence'})


def tree_frame(path, cohort, lead):
    f = pd.read_csv(path)
    if 'mode' in f:
        f = f[f['mode'] == 'forecast']
    return pd.DataFrame({'cohort': cohort, 'sensor': f.sensor.to_numpy(), 'time': utc(f.time).to_numpy(),
                         'lead': lead, 'pred': f.prediction.to_numpy(float), 'truth_src': f.truth.to_numpy(float),
                         'anchor_src': f.persistence.to_numpy(float)})


def dev_tree_path(variable, arm, lead):
    if lead == 72:
        return ADD83 / f'horizon_{variable}_{arm}_h72' / 'selected_predictions.csv.gz'
    if lead in (6, 24):
        return RUNS / 'v2_perfold/forecast/perfold' / f'{variable}_{arm}_h{lead}' / 'selected_predictions.csv.gz'
    return RUNS / 'v2_perfold_rest/forecast/perfold' / f'horizon_{variable}_{arm}_h{lead}' / 'selected_predictions.csv.gz'


def zoo_c(variable):
    """The eight pretrained arms of task C, median at each lead, keyed like the points rows."""
    parts = {arm: [] for arm in ZOO_ARMS}
    for cohort in B_COHORTS:
        stem = f'C_{variable}_{cohort}'
        inp = load_inputs(Z.INP / f'{stem}.npz')
        k = inp['keys']
        times = utc(pd.to_datetime(k.time.astype(np.int64), utc=True))
        for arm in ZOO_ARMS:
            med = load_predictions(Z.PRED / f'{stem}__{arm}.npz')['median'].astype(np.float64)
            for lead in B_LEADS:
                parts[arm].append(pd.DataFrame({'cohort': cohort, 'sensor': k.sensor.astype(str).to_numpy(),
                                                'time': times.to_numpy(), 'lead': lead,
                                                'pred': med[:, lead - 1]}))
    out = {}
    for arm, ps in parts.items():
        f = pd.concat(ps, ignore_index=True)
        f['time'] = utc(f.time)
        out[arm] = f.drop_duplicates(KEY)
    return out


def sources(variable):
    """Every forecasting arm as (KEY + pred [+ truth_src, anchor_src]) or (KEY + loss), and their origin."""
    src, origin = {}, {}

    def add(arm, frame, where):
        src.setdefault(arm, []).append(frame)
        origin.setdefault(arm, []).append(where)

    for arm in ('tree', 'weather_tree'):
        for lead in B_LEADS:
            p = dev_tree_path(variable, arm, lead)
            add(arm, tree_frame(p, 'Dev-2025', lead), rel(p))
            d, info = CR.predictions(variable, arm, lead)
            add(arm, pd.DataFrame({'cohort': 'Reserved-2026', 'sensor': d.sensor.to_numpy(), 'time': utc(d.time).to_numpy(),
                                   'lead': lead, 'pred': d.prediction.to_numpy(float),
                                   'truth_src': d.truth.to_numpy(float), 'anchor_src': d.persistence.to_numpy(float)}),
                info['report'])
        for lead in B_LEADS[:-1]:
            for base in HORIZON_RUNS:
                p = base / f'v2_horizon_{variable}_{arm}_h{lead}' / 'selected_predictions.csv.gz'
                if p.exists():
                    add(f'{arm}_shared', tree_frame(p, 'Dev-2025', lead), rel(p))
                    break
        r24 = pd.read_csv(RESERVED_H24 / f'reserved_h24_{variable}.csv.gz')
        r24 = r24[r24.arm == arm]
        add(f'{arm}_shared', pd.DataFrame({'cohort': 'Reserved-2026', 'sensor': r24.sensor.to_numpy(),
                                           'time': utc(r24.time).to_numpy(), 'lead': 24,
                                           'pred': r24.prediction.to_numpy(float), 'truth_src': r24.truth.to_numpy(float),
                                           'anchor_src': r24.persistence.to_numpy(float)}),
            rel(RESERVED_H24 / f'reserved_h24_{variable}.csv.gz'))
    long, files, absent = Z.published_quantiles(variable)
    for arm in QUANTILE_ARMS:
        f = long[arm]
        add(arm, pd.DataFrame({'cohort': f.cohort.to_numpy(), 'sensor': f.sensor.to_numpy(), 'time': utc(f.time).to_numpy(),
                               'lead': f.lead.to_numpy(int), 'pred': f.q50.to_numpy(float)}), ', '.join(files[arm]))
    for arm, f in zoo_c(variable).items():
        # task C's Chronos-2 with the network is addendum 65's chronos2_net to 1e-3 (its gate C-G1) and
        # is read once, as chronos2_net; the TimesFM covariate arms read the network's past at issuance
        if arm == 'chronos2_cov':
            continue
        name = arm.replace('_cov', '_cov_past')
        add(name, f, f'runs/v2_fm_zoo/preds/C_{variable}_<cohort>__{arm}.npz')
    b = pd.read_csv(BLENDED)
    b = b[(b.variable == variable) & b.has_anchor & b.has_target & b.prediction.notna()]
    add('blended', pd.DataFrame({'cohort': 'Reserved-2026', 'sensor': b.sensor.to_numpy(), 'time': utc(b.time).to_numpy(),
                                 'lead': 6, 'pred': b.prediction.to_numpy(float), 'truth_src': b.truth.to_numpy(float),
                                 'anchor_src': b.persistence.to_numpy(float)}), rel(BLENDED) + ' column prediction')
    for arm in ('learned_source', 'conservative_exchange'):
        for lead in B_LEADS[:-1]:
            p = GRAPHS / f'{variable}_{arm}_h{lead}' / 'selected_predictions.csv.gz'
            g = pd.read_csv(p)
            g['ae'] = (g.truth - g.prediction).abs()
            per = g.groupby(['sensor', 'time']).agg(loss=('ae', 'mean'), truth_src=('truth', 'first'),
                                                    anchor_src=('persistence', 'first'), seeds=('seed', 'nunique'))
            per = per.reset_index()
            if (per.seeds != 3).any():
                raise SystemExit(f'{p}: not three seeds per row')
            add(arm, pd.DataFrame({'cohort': 'Dev-2025', 'sensor': per.sensor.to_numpy(), 'time': utc(per.time).to_numpy(),
                                   'lead': lead, 'loss': per.loss.to_numpy(float), 'truth_src': per.truth_src.to_numpy(float),
                                   'anchor_src': per.anchor_src.to_numpy(float)}), rel(p))
    out = {}
    for arm, frames in src.items():
        f = pd.concat(frames, ignore_index=True)
        f['time'] = utc(f.time)
        if f.duplicated(KEY).any():
            raise SystemExit(f'{variable} {arm}: duplicated rows across its sources')
        out[arm] = f
    return out, origin, absent


def attach(base, srcs, keys, truth_col='truth', anchor_col='persistence', map_cohort=None):
    """Join every source onto `base` by keys; per-row absolute error columns `ae|<arm>`; a row check."""
    frame = base.copy()
    checks = {}
    left = frame[keys].copy()
    if map_cohort is not None:
        left['cohort'] = left['cohort'].map(map_cohort)
    for arm, f in srcs.items():
        cols = [c for c in ('pred', 'loss', 'truth_src', 'anchor_src') if c in f]
        j = left.merge(f[keys + cols], on=keys, how='left', validate='one_to_one')
        hit = (j['pred'] if 'pred' in j else j['loss']).notna().to_numpy()
        e = {'rows_joined': int(hit.sum())}
        if 'truth_src' in j:
            d = np.abs(j.truth_src.to_numpy(float) - frame[truth_col].to_numpy(float))[hit & j.truth_src.notna().to_numpy()]
            e['max_abs_truth_difference'] = float(d.max()) if len(d) else None
        if 'anchor_src' in j:
            d = np.abs(j.anchor_src.to_numpy(float) - frame[anchor_col].to_numpy(float))[hit & j.anchor_src.notna().to_numpy()]
            e['max_abs_anchor_difference'] = float(d.max()) if len(d) else None
        e['passed'] = bool(all(v is None or v <= TOL for k, v in e.items() if k.startswith('max_abs')))
        checks[arm] = e
        if 'pred' in j:
            frame[arm] = j['pred'].to_numpy(float)
            frame[f'ae|{arm}'] = np.abs(frame[truth_col].to_numpy(float) - frame[arm].to_numpy(float))
        else:
            frame[f'ae|{arm}'] = j['loss'].to_numpy(float)
    return frame, checks


# --------------------------------------------------------------------------------------------------
# Scoring one cell on common rows
# --------------------------------------------------------------------------------------------------
def own_contrast(sub, x, y, label, variable):
    s = sub.dropna(subset=[f'ae|{x}', f'ae|{y}'])
    if not len(s):
        return None
    units = int(s[['fold', 'week']].drop_duplicates().shape[0])
    c = {'point': r6(R.macro(s, f'ae|{x}') - R.macro(s, f'ae|{y}')), 'rows': int(len(s)), 'units_fold_week': units,
         'rows_kind': 'own paired rows'}
    if units < MIN_UNITS:
        c.update(ci=None, excludes_zero=None, not_estimable=f'{units} fold-week units')
    else:
        d = R.joint_draws(s, [f'ae|{x}', f'ae|{y}'], label)
        lo, hi = np.percentile(d[:, 0] - d[:, 1], [2.5, 97.5])
        c.update(ci=[r6(lo), r6(hi)], excludes_zero=bool(lo > 0 or hi < 0), stream=label)
    return c


def score_cell(cell, arms, refs, label, variable, extra_pairs=(), exclude_from_best=()):
    """MAE of every arm on the cell's common rows (core arms) or its own rows, and the contrasts."""
    n = len(cell)
    present = [a for a in arms if f'ae|{a}' in cell and cell[f'ae|{a}'].notna().any()]
    coverage = {a: float(cell[f'ae|{a}'].notna().mean()) for a in present}
    core = [a for a in present if coverage[a] >= CORE_SHARE]
    common = cell.dropna(subset=[f'ae|{a}' for a in core])
    out = {'base_rows': int(n), 'common_rows': int(len(common)), 'core_arms': core,
           'coverage': {a: r6(v) for a, v in coverage.items()},
           'fold_week_units': int(common[['fold', 'week']].drop_duplicates().shape[0]),
           'sensors': int(common.sensor.nunique()), 'mae': {}, 'contrasts': {}}
    small = common.groupby(['fold', 'sensor']).size()
    out['cells_below_five_rows'] = {f'{f}|{s}': int(v) for (f, s), v in small[small < MIN_CELL_ROWS].items()}
    for a in present:
        own = cell.dropna(subset=[f'ae|{a}'])
        e = {'own_rows': int(len(own)), 'mae_own': r6(R.macro(own, f'ae|{a}'))}
        if a in core:
            e['mae_common'] = r6(R.macro(common, f'ae|{a}'))
        if variable == 'oxygen':
            e['mgl'] = {k: r6(e[k] * C_MG) for k in ('mae_common', 'mae_own') if e.get(k) is not None}
        out['mae'][a] = e
    cols = [f'ae|{a}' for a in core]
    units = out['fold_week_units']
    draws = R.joint_draws(common, cols, label) if (units >= MIN_UNITS and len(cols) >= 2) else None
    out['stream'] = label

    def contrast(x, y):
        if x in core and y in core:
            point = R.macro(common, f'ae|{x}') - R.macro(common, f'ae|{y}')
            c = {'point': r6(point), 'rows': int(len(common)), 'units_fold_week': units, 'rows_kind': 'common rows'}
            if draws is None:
                c.update(ci=None, excludes_zero=None, not_estimable=f'{units} fold-week units')
            else:
                d = draws[:, cols.index(f'ae|{x}')] - draws[:, cols.index(f'ae|{y}')]
                lo, hi = np.percentile(d, [2.5, 97.5])
                c.update(ci=[r6(lo), r6(hi)], excludes_zero=bool(lo > 0 or hi < 0))
        else:
            c = own_contrast(cell, x, y, f'{label}|{x} - {y}', variable)
            if c is None:
                return None
        return decorate(c, variable, x, y, READS_OTHERS)

    pairs = [(a, r) for r in refs if r in present for a in present if a != r]
    pairs += [(x, y) for x, y in extra_pairs if x in present and y in present]
    ranked = sorted((out['mae'][a]['mae_common'], a) for a in core if a not in exclude_from_best)
    if len(ranked) >= 2:
        out['best'], out['second'] = ranked[0][1], ranked[1][1]
        pairs.append((ranked[0][1], ranked[1][1]))
    for x, y in dict.fromkeys(pairs):
        c = contrast(x, y)
        if c is not None:
            out['contrasts'][f'{x} - {y}'] = c
    out['_contrast'] = contrast
    return out, common


# --------------------------------------------------------------------------------------------------
# Part B
# --------------------------------------------------------------------------------------------------
B_REFS = ('persistence', 'clim_month_hour', 'clim_hour', 'weather_tree')


def part_b(variable, srcs):
    base = points(variable)
    for col in ('persistence', 'clim_hour', 'clim_month_hour'):
        base[f'ae|{col}'] = (base.truth - base[col]).abs()
    frame, checks = attach(base, srcs, KEY)
    # the blended weather tree's own rows reproduce its published 2026 six-hour error (battery_full_v2)
    published = json.loads(BLENDED_REPORT.read_text())[variable]['MODEL_v2']
    own = frame[(frame.cohort == 'Reserved-2026') & (frame.lead == 6)].dropna(subset=['ae|blended'])
    mine = R.macro(own, 'ae|blended')
    checks['blended']['published_mae'] = published
    checks['blended']['mae_here'] = mine
    checks['blended']['passed'] = bool(checks['blended']['passed'] and abs(mine - published) <= 1e-9)
    failed = [k for k, c in checks.items() if not c['passed']]
    if failed:
        raise SystemExit(f'{variable}: gate A85-G5 failed for part B: {failed}')
    arms = ['persistence', 'clim_hour', 'clim_month_hour'] + [a for a in srcs]
    res = {}
    for cohort in B_COHORTS:
        res[cohort] = {}
        for lead in B_LEADS:
            cell = frame[(frame.cohort == cohort) & (frame.lead == lead)]
            out, _ = score_cell(cell, arms, B_REFS, f'a85|B|{variable}|{cohort}|{lead}', variable)
            out.pop('_contrast', None)
            res[cohort][str(lead)] = out
            print(f'  B {variable} {cohort} {lead}: {out["common_rows"]}/{out["base_rows"]} common, '
                  f'best {out.get("best")}', flush=True)
    return res, checks, frame


# --------------------------------------------------------------------------------------------------
# Part A
# --------------------------------------------------------------------------------------------------
A_REFS = ('persistence', 'clim_hour', 'anchored_direct')


def bridge_frame(variable, srcs):
    import scripts.v2_rollout_a82_report as RA
    f, gates, _ = Z.d_frame(variable, list(ZOO_ARMS), Z.manifest())
    f = f.reset_index(drop=True)
    f['onset'] = utc(f.onset)
    roll_cols = ('recursive_tree', 'graph_conservative', 'graph_unconstrained')
    for c in roll_cols:
        f[c] = np.nan
    roll_info = {}
    for cohort in A_COHORTS:
        j, missing = RA.a82_frame(variable, cohort)
        j = j[['fold', 'sensor', 'onset_ns', 'lead'] + list(roll_cols)].rename(columns={'onset_ns': 't'})
        m = (f.cohort == cohort).to_numpy()
        k = f.loc[m, ['fold', 'sensor', 't', 'lead']].merge(j, on=['fold', 'sensor', 't', 'lead'], how='left',
                                                            validate='one_to_one')
        for c in roll_cols:
            f.loc[m, c] = k[c].to_numpy(float)
        roll_info[cohort] = {'rows_without_rollout': int(missing)}
    for col in ('persistence', 'clim_hour', 'clim_month_hour', 'neighbour_change', 'anchored_direct') + roll_cols + tuple(ZOO_ARMS):
        f[f'ae|{col}'] = (f.truth - f[col]).abs()
    base = f.rename(columns={'onset': 'time'})
    # the pretrained arms of part A are task D's (D-G2): their univariate medians equal task C's, and their
    # covariate arms read the other probes after onset; task C's covariate arms are not set beside them
    fc = {a: s for a, s in srcs.items() if a not in ('learned_source', 'conservative_exchange')
          and a not in ZOO_ARMS and not a.endswith('_cov_past')}
    base, checks = attach(base, fc, KEY, map_cohort=TO_FORECAST)
    return base.rename(columns={'time': 'onset'}), {'D-G2': gates.get('D-G2', {}).get('passed'),
                                                     'D-G1': gates.get('D-G1', {}).get('passed'),
                                                     'rollout': roll_info}, checks


def points_coverage(variable, frame):
    """A85-G5 for part A: every Bridge row at a forecasting lead is a points row, same truth and anchor."""
    p = points(variable)
    left = frame[['cohort', 'sensor', 'onset', 'lead', 'truth', 'persistence']].rename(columns={'onset': 'time'})
    left = left.assign(cohort=left.cohort.map(TO_FORECAST))
    left = left[left.lead.isin(B_LEADS)]
    j = left.merge(p[KEY + ['truth', 'persistence']], on=KEY, how='left', suffixes=('', '_p'), validate='one_to_one')
    hit = j.truth_p.notna()
    # A Bridge-Dev episode may run past its fold's end, while the forecasting rows score only targets
    # inside the fold (v2_horizon.folds): such rows are the only ones allowed to be missing.
    fold = frame.loc[frame.lead.isin(B_LEADS), 'fold'].to_numpy()
    target = j.time + pd.to_timedelta(j.lead, unit='h')
    ends = np.where(fold == 0, pd.Timestamp('2025-10-01', tz='UTC').value,
                    np.where(fold == 1, pd.Timestamp('2026-01-01', tz='UTC').value, np.iinfo(np.int64).max))
    beyond = (j.cohort == 'Dev-2025').to_numpy() & (pd.DatetimeIndex(target).as_unit('ns').asi8 >= ends)
    out = {'rows': int(len(j)), 'found_among_forecasting_rows': int(hit.sum()),
           'missing_with_target_past_fold_end': int((~hit & beyond).sum()),
           'missing_otherwise': int((~hit & ~beyond).sum()),
           'max_abs_truth_difference': float((j.truth - j.truth_p)[hit].abs().max()),
           'max_abs_anchor_difference': float((j.persistence - j.persistence_p)[hit].abs().max())}
    out['by_cohort_lead'] = {f'{c}|{l}': [int(g.truth_p.notna().sum()), int(len(g))]
                             for (c, l), g in j.groupby(['cohort', 'lead'])}
    out['passed'] = bool(out['missing_otherwise'] == 0
                         and out['max_abs_truth_difference'] <= TOL and out['max_abs_anchor_difference'] <= TOL)
    return out


def gate_bridge72():
    """A85-G6 (Bridge): the joint-draw path reproduces the published 72 h interval exactly."""
    rep = json.loads((RUNS / 'v2_bridge72' / 'report.json').read_text())
    out, ok = {}, True
    for variable in VARIABLES:
        f72 = v2_bridge72.with_weeks(pd.read_csv(RUNS / 'v2_bridge72' / f'bridge72_{variable}.csv.gz', parse_dates=['onset']))
        for cohort in A_COHORTS:
            sub = f72[(f72.cohort == cohort) & (f72.lead == 72)].dropna(subset=['anchored_direct', 'persistence']).copy()
            sub['ae|x'] = (sub.truth - sub.anchored_direct).abs()
            sub['ae|y'] = (sub.truth - sub.persistence).abs()
            label = f'bridge72|{variable}|{cohort}|72|anchored_direct|persistence'
            d = R.joint_draws(sub, ['ae|x', 'ae|y'], label)
            lo, hi = np.percentile(d[:, 0] - d[:, 1], [2.5, 97.5])
            point = round(R.macro(sub, 'ae|x') - R.macro(sub, 'ae|y'), 6)
            pub = rep[variable][cohort]['by_lead']['72']['contrasts']['anchored_direct - persistence']
            same = bool(point == pub['point'] and [round(float(lo), 6), round(float(hi), 6)] == pub['ci'])
            ok &= same
            out[f'{variable}|{cohort}'] = {'mine': [point, [round(float(lo), 6), round(float(hi), 6)]],
                                           'published': [pub['point'], pub['ci']], 'same': same}
    out['passed'] = bool(ok)
    return out


def part_a(variable, srcs):
    frame, info, checks = bridge_frame(variable, srcs)
    g5 = points_coverage(variable, frame)
    if not (g5['passed'] and all(c['passed'] for c in checks.values())):
        raise SystemExit(f'{variable}: gate A85-G5 failed for part A: {g5} '
                         f'{[k for k, c in checks.items() if not c["passed"]]}')
    arms =[a for a in FORECASTS + NETWORK if f'ae|{a}' in frame]
    res = {}
    for cohort in A_COHORTS:
        res[cohort] = {}
        for lead in A_LEADS:
            cell = frame[(frame.cohort == cohort) & (frame.lead == lead)]
            extra = [('recursive_tree', 'weather_tree')]
            out, common = score_cell(cell, arms, A_REFS, f'a85|A|{variable}|{cohort}|{lead}', variable,
                                     extra_pairs=extra)
            core = out['core_arms']
            fc = [a for a in core if a in FORECASTS]
            nw = [a for a in core if a in NETWORK]
            if fc and nw:
                best_f = min(fc, key=lambda a: out['mae'][a]['mae_common'])
                best_n = min(nw, key=lambda a: out['mae'][a]['mae_common'])
                out['best_forecast'], out['best_network_aided'] = best_f, best_n
                for x, y, name in (('anchored_direct', best_f, 'network_adds'), (best_n, best_f, 'best_network_vs_best_forecast')):
                    if x in core:
                        key = f'{x} - {y}'
                        if key not in out['contrasts']:
                            c = out['_contrast'](x, y)
                            if c is not None:
                                out['contrasts'][key] = c
                        out[name] = key
            out.pop('_contrast', None)
            res[cohort][str(lead)] = out
            print(f'  A {variable} {cohort} {lead}: {out["common_rows"]}/{out["base_rows"]} common, best '
                  f'forecast {out.get("best_forecast")}, best network {out.get("best_network_aided")}', flush=True)
    return res, {'A85-G5_rows': g5, 'sources': info, 'joins': checks}


# --------------------------------------------------------------------------------------------------
# Part C
# --------------------------------------------------------------------------------------------------
#: the sequence panels are stored in the grid's scaled units
from scripts.v2_sequence_table import SCALE as PANEL_SCALE  # noqa: E402
PRINTED = {('temperature', 'tree'): (0.1999, 4), ('temperature', 'weather_tree'): (0.1929, 4),
           ('oxygen', 'tree'): (0.20742, 5), ('oxygen', 'weather_tree'): (0.21454, 5)}


def fold_macro(frame, col):
    f = frame.dropna(subset=[col])
    return float(f.assign(e=(f.truth - f[col]).abs()).groupby(['fold', 'sensor']).e.mean().groupby('fold').mean().mean())


def part_c():
    out, ok = {}, True
    a83 = json.loads((RUNS / 'v2_addendum83.json').read_text())['levels']
    for variable in VARIABLES:
        k = C_MG if variable == 'oxygen' else 1.0
        e = {}
        for arm in ('tree', 'weather_tree'):
            p = RUNS / '20260919T144822Z_v2_forecast/results' / f'v2_forecast_{variable}_{arm}_nested' / 'selected_predictions.csv.gz'
            f = pd.read_csv(p)
            f = f[f['mode'] == 'forecast']
            shared = fold_macro(f, 'prediction')
            printed, dec = PRINTED[(variable, arm)]
            same = round(shared * k, dec) == printed
            ok &= same
            q = RUNS / 'v2_perfold/forecast/perfold' / f'{variable}_{arm}_h6' / 'selected_predictions.csv.gz'
            g = pd.read_csv(q)
            per_fold = fold_macro(g, 'prediction')
            e[arm] = {'shared_reference_grid': {'mae': r6(shared), 'rows': int(len(f)), 'source': rel(p),
                                                'printed': printed, 'reproduces_printed': bool(same)},
                      'per_fold_reference_grid': {'mae': r6(per_fold), 'rows': int(len(g)), 'source': rel(q)},
                      'shared_six_grids_addendum83': {
                          'by_offset': {str(o): a83[f'{variable}|6|{o}|42'][arm] for o in range(6)},
                          'mean': r6(np.mean([a83[f'{variable}|6|{o}|42'][arm] for o in range(6)])),
                          'source': 'runs/v2_addendum83.json levels'}}
        by_offset = {'0': e['tree']['per_fold_reference_grid']['mae']}
        srcs = {'0': e['tree']['per_fold_reference_grid']['source']}
        for o in range(1, 6):
            p = RUNS / 'v2_xgb_sweep' / f'{variable}_tree_h6_shift{o}_seed42' / 'selected_predictions.csv.gz'
            by_offset[str(o)] = r6(fold_macro(pd.read_csv(p), 'prediction'))
            srcs[str(o)] = rel(p)
        e['tree']['per_fold_six_grids'] = {'by_offset': by_offset, 'mean': r6(np.mean(list(by_offset.values()))),
                                           'sources': srcs}
        seq = {}
        for o in range(6):
            p = RUNS / 'v2_sequence_sweep/panel_perfold' / f'panel_{variable}_offset{o}.csv.gz'
            seq[str(o)] = r6(fold_macro(pd.read_csv(p), 'tree_seed42') * PANEL_SCALE[variable])
        e['tree']['sequence_cohort_per_fold_six_grids'] = {
            'by_offset': seq, 'mean': r6(np.mean(list(seq.values()))),
            'source': 'runs/v2_sequence_sweep/panel_perfold (the tree the pretrained zoo Dev-2025 column is set against; '
                      'rows with a 512 h context, fitted without per-probe weights)'}
        e['weather_tree']['per_fold_six_grids'] = 'not stored (no per-fold weather-tree study on the shifted grids)'
        if variable == 'oxygen':
            for arm in e:
                for block in e[arm].values():
                    if isinstance(block, dict):
                        if 'mae' in block:
                            block['mae_mgl'] = r6(block['mae'] * C_MG)
                        if 'mean' in block:
                            block['mean_mgl'] = r6(block['mean'] * C_MG)
        out[variable] = e
    return out, {'A85-G7_part_C': {'passed': bool(ok)}}


# --------------------------------------------------------------------------------------------------
# Part D
# --------------------------------------------------------------------------------------------------
GAP_COHORTS = ('Gap-Dev', 'Gap-2026', 'Gap-Long-Dev', 'Gap-Long-2026')
STRICT = ('Gap-Dev', 'Gap-2026')
STRICT_LENGTHS_MAX = 72
META = {'episode', 'sensor', 'has_twin', 'gap_length', 'position', 'fold', 'time', 'truth'}
OWN_GAP = {'linear', 'pchip', 'spline', 'persistence', 'climatology', 'polynomial', 'chronos_bidir', 'chronos_forward',
           'chronos2_bidir', 'arx_bridge', 'arx_bridge_exo', 'chronos2_exo_bidir', 'bolt_resid_bidir',
           'bolt_resid_lin_bidir'}


def gap_reads_others(arm):
    if arm in OWN_GAP:
        return False
    if '.' in arm:
        return not arm.split('.')[0].endswith('_uni')
    return True


def gap_frame(cohort, variable):
    from scripts.v2_gap_climatology import arm_dirs, load_predictions as gap_load
    from scripts.v2_gap_short2026_rows import append as append_short
    from scripts.v2_gap_short2026_score import ROW_FOLDERS, Rows
    from scripts.v2_gap_variants import COHORTS as GC
    bands = ('1-6', '12-72') if cohort in STRICT else ('168-1440',)
    pieces = []
    for band in bands:
        base = None
        for name in ROW_FOLDERS:
            f = pd.read_csv(Rows(name) / f'{cohort}__{variable}__{band}.csv.gz')
            if base is None:
                base = f
                continue
            if not (f[['episode', 'position']].equals(base[['episode', 'position']])
                    and (f.truth - base.truth).abs().max() == 0):
                raise SystemExit(f'{name} {cohort} {variable} {band}: rows differ from the variants rows')
            for col in f.columns:
                if col not in base:
                    base[col] = f[col].to_numpy()
        pieces.append(base)
    frame = pd.concat(pieces, ignore_index=True)
    run, results = GC[cohort][0], GC[cohort][1]
    info = {'row_folders': list(ROW_FOLDERS), 'run': run, 'arms_from_run': {}}
    for (v, arm), folder in sorted(arm_dirs(run, results).items()):
        if v != variable:
            continue
        pf = gap_load(run, results, variable, arm)
        if cohort == 'Gap-2026':
            try:
                pf = append_short(pf, variable, arm)
            except (SystemExit, FileNotFoundError) as err:
                info['arms_from_run'][f'{arm} (1-3 h rows)'] = f'not appended: {err}'
        new = [c for c in pf.columns if c not in frame.columns and c not in META]
        if not new:
            continue
        j = frame[['episode', 'position', 'truth']].merge(pf[['episode', 'position', 'truth'] + new],
                                                         on=['episode', 'position'], how='left',
                                                         suffixes=('', '_run'), validate='one_to_one')
        hit = j.truth_run.notna()
        diff = float((j.truth - j.truth_run)[hit].abs().max()) if hit.any() else None
        if diff is not None and diff > TOL:
            raise SystemExit(f'{cohort} {variable} {arm}: truth differs from the row folders ({diff})')
        for c in new:
            frame[c] = j[c].to_numpy()
        info['arms_from_run'][arm] = {'columns': new, 'rows_joined': int(hit.sum()), 'source': rel(folder)}
    if cohort in STRICT:
        frame = frame[frame.gap_length <= STRICT_LENGTHS_MAX].reset_index(drop=True)
    info['zoo'] = zoo_fills(frame, cohort, variable)
    if frame.duplicated(['episode', 'position']).any():
        raise SystemExit(f'{cohort} {variable}: duplicated rows')
    return frame, info


def zoo_fills(frame, cohort, variable):
    from scripts.v2_fm_zoo_score_gaps import INP as ZINP, PRED as ZPRED, stem as zstem
    rec = {}
    for arm in ZOO_ARMS:
        for mode in ('fwd', 'bidir'):
            frame[f'{arm}.{mode}'] = np.nan
    for length in sorted(frame.gap_length.unique()):
        rows = np.flatnonzero(frame.gap_length.to_numpy() == length)
        part = frame.iloc[rows]
        k = part.position.to_numpy(int) - 1
        w = (length - k) / (length + 1)
        traces = {}
        for direction in ('fwd', 'bwd'):
            name = zstem(variable, cohort, direction, int(length))
            path = ZINP / f'{name}.npz'
            if not path.exists():
                rec[f'{int(length)}|{direction}'] = 'no input file'
                continue
            z = np.load(path, allow_pickle=False)
            index = {str(e): i for i, e in enumerate(z['key_episode'])}
            e_idx = part.episode.map(index)
            ok = e_idx.notna().to_numpy()
            e_idx = e_idx.fillna(-1).astype(int).to_numpy()
            rec[f'{int(length)}|{direction}'] = {'rows': int(len(part)), 'rows_with_context': int(ok.sum())}
            for arm in ZOO_ARMS:
                med = load_predictions(ZPRED / f'{name}__{arm}.npz')['median'].astype(np.float64)
                if med.shape[1] != int(length):
                    raise SystemExit(f'{name} {arm}: horizon {med.shape[1]} != {length}')
                if direction == 'bwd':
                    med = med[:, ::-1]
                vals = np.full(len(part), np.nan)
                vals[ok] = med[e_idx[ok], k[ok]]
                traces[(arm, direction)] = vals
        for arm in ZOO_ARMS:
            fwd, bwd = traces.get((arm, 'fwd')), traces.get((arm, 'bwd'))
            if fwd is not None:
                frame.iloc[rows, frame.columns.get_loc(f'{arm}.fwd')] = fwd
            if fwd is not None and bwd is not None:
                frame.iloc[rows, frame.columns.get_loc(f'{arm}.bidir')] = w * fwd + (1 - w) * bwd
    return rec


def gap_arms(frame):
    skip = META | {'twin', 'twin_reports', 'nearest_live_probe_distance'}
    return [c for c in frame.columns if c not in skip and pd.api.types.is_numeric_dtype(frame[c])
            and frame[c].notna().any()]


def band_of(length):
    return '1-6' if length <= 6 else ('12-72' if length <= 72 else '168-1440')


def gap_entry(got, variable, arm, ref):
    if got is None:
        return None
    c = {'point': r6(got['delta']), 'rows': got['rows'], 'episodes': got['episodes'], 'units': got['units'],
         'ci': None if got['ci_low'] is None else [r6(got['ci_low']), r6(got['ci_high'])],
         'excludes_zero': got['excludes_zero'] if got['estimable'] else None,
         'jackknife_t': None if got['jackknife_t'] is None else [r6(v) for v in got['jackknife_t']],
         'jackknife_excludes_zero': got.get('jackknife_excludes_zero'), 'estimable': got['estimable']}
    if not got['estimable']:
        c['not_estimable'] = f'{got["units"]} units, fewer than {MIN_UNITS}'
    others = {a for a in (arm, ref) if gap_reads_others(a)}
    return decorate(c, variable, arm, ref, others)


def gap_gates(frames, whole):
    """A85-G6 (gaps): published intervals reproduced from the frames read here."""
    from scripts.v2_fm_zoo_score_gaps import Cell
    from scripts.v2_gap_climatology import load_predictions as gap_load
    from scripts.v2_gap_intervals import clusters
    from scripts.v2_gap_long import units as long_units
    from scripts.v2_gap_variants import COHORTS as GC
    out, ok = {}, True
    zoo = json.loads((RUNS / 'v2_fm_zoo/score_gaps.json').read_text())['results']
    long_pub = json.loads((RUNS / 'v2_gap_long.json').read_text())
    for variable in VARIABLES:
        for cohort, name in (('Gap-Dev', 'v2_gap_intervals.json'), ('Gap-2026', 'v2_gap_intervals_reserved2026.json')):
            pub = json.loads((RUNS / name).read_text())['variables'][variable]['by_gap_length']
            run, results = GC[cohort][0], GC[cohort][1]
            unit = clusters(gap_load(run, results, variable, 'chronos_bidir'))
            f = frames[(cohort, variable)]
            for length in sorted(f.gap_length.unique()):
                p = pub.get(str(int(length)), {}).get('chronos_bidir')
                part = f[f.gap_length == length]
                if p is not None and set(part.episode) <= set(unit):
                    got = Cell(part, unit, ['chronos_bidir', 'linear']).paired('chronos_bidir', 'linear',
                                                                              f'{variable}|chronos_bidir|{int(length)}')
                    mine = [round(got['delta'], 4), got['ci_low'] and round(got['ci_low'], 4),
                            got['ci_high'] and round(got['ci_high'], 4), got['units'], got['episodes']]
                    theirs = [p['delta'], p.get('ci_low'), p.get('ci_high'), p['clusters'], p['episodes']]
                    same = mine == theirs
                    ok &= same
                    out[f'intervals|{cohort}|{variable}|{int(length)}'] = {'mine': mine, 'published': theirs, 'same': same}
                z = zoo.get(cohort, {}).get(variable, {}).get('cells', {}).get(str(int(length)), {}).get('K1', {})
                zc = z.get('chronos_bolt_base_uni.bidir', {}).get('minus_linear')
                if zc is not None:
                    zunit = whole[('Gap-Dev', variable)] if cohort == 'Gap-Dev' else unit
                    got = Cell(part, zunit, ['chronos_bolt_base_uni.bidir', 'linear']).paired(
                        'chronos_bolt_base_uni.bidir', 'linear',
                        f'a81E|{cohort}|{variable}|{int(length)}|chronos_bolt_base_uni.bidir|linear')
                    mine = [r6(got['delta']), r6(got['ci_low']), r6(got['ci_high']), got['units']]
                    theirs = [zc['delta'], zc['ci_low'], zc['ci_high'], zc['units']]
                    same = mine == theirs
                    ok &= same
                    out[f'zoo|{cohort}|{variable}|{int(length)}'] = {'mine': mine, 'published': theirs, 'same': same}
        for cohort in ('Gap-Long-Dev', 'Gap-Long-2026'):
            f = frames[(cohort, variable)]
            _, cons = long_units(f)
            for length in sorted(f.gap_length.unique()):
                p = long_pub[cohort][variable]['by_gap_length'][str(int(length))]['minus_linear']['gpr']['conservative']
                part = f[f.gap_length == length]
                got = Cell(part, cons, ['gpr', 'linear']).paired(
                    'gpr', 'linear', f'long|{cohort}|{variable}|gpr|{int(length)}|conservative')
                mine = [round(got['delta'], 4), got['ci_low'] and round(got['ci_low'], 4),
                        got['ci_high'] and round(got['ci_high'], 4), got['units']]
                theirs = [p['delta'], p.get('ci_low'), p.get('ci_high'), p['clusters']]
                same = mine == theirs
                ok &= same
                out[f'long|{cohort}|{variable}|{int(length)}'] = {'mine': mine, 'published': theirs, 'same': same}
    out['passed'] = bool(ok)
    return out


def part_d():
    from scripts.v2_fm_zoo_score_gaps import Cell
    from scripts.v2_gap_long import units as long_units
    from scripts.v2_gap_short2026_score import whole_cohort_clusters
    from scripts.v2_gap_variants import INCUMBENT
    whole = whole_cohort_clusters()
    frames, infos = {}, {}
    for cohort in GAP_COHORTS:
        for variable in VARIABLES:
            frames[(cohort, variable)], infos[f'{cohort}|{variable}'] = gap_frame(cohort, variable)
            print(f'  D frame {cohort} {variable}: {len(frames[(cohort, variable)])} rows', flush=True)
    gates = gap_gates(frames, whole)
    print('  D gate G6:', gates['passed'], flush=True)
    res = {}
    if not gates['passed']:
        return {'stopped': 'gate A85-G6 (gaps) failed'}, gates, infos
    for (cohort, variable), frame in frames.items():
        unit = (whole[(cohort, variable)] if cohort in STRICT else long_units(frame)[1])
        unit_kind = ('overlap cluster (whole strict cohort)' if cohort in STRICT
                     else 'probe x calendar month of the gap start')
        arms = gap_arms(frame)
        block = {}
        for length in sorted(frame.gap_length.unique()):
            part = frame[frame.gap_length == length]
            present = [a for a in arms if part[a].notna().any()]
            coverage = {a: float(part[a].notna().mean()) for a in present}
            core = [a for a in present if coverage[a] >= CORE_SHARE]
            common = part.dropna(subset=core + ['truth'])
            incumbent = INCUMBENT[(variable, band_of(length))]
            cell_c = Cell(common, unit, core)
            cell_o = Cell(part, unit, [a for a in present if a not in core])
            e = {'rows': int(len(part)), 'common_rows': int(len(common)), 'episodes': int(part.episode.nunique()),
                 'common_episodes': int(common.episode.nunique()),
                 'units': int(len({unit[x] for x in common.episode.unique()})), 'unit': unit_kind,
                 'incumbent': incumbent, 'core_arms': core,
                 'coverage': {a: r6(v) for a, v in coverage.items()}, 'mae': {}, 'minus_linear': {},
                 'minus_incumbent': {}}
            for a in present:
                m = cell_c.mae(a) if a in core else cell_o.mae(a)
                e['mae'][a] = {'mae': r6(m), 'rows': 'common' if a in core else int(part[a].notna().sum())}
                if variable == 'oxygen' and m is not None:
                    e['mae'][a]['mae_mgl'] = r6(m * C_MG)
            for ref, key in (('linear', 'minus_linear'), (incumbent, 'minus_incumbent')):
                if ref not in present:
                    continue
                for a in present:
                    if a == ref:
                        continue
                    label = f'a85|D|{cohort}|{variable}|{int(length)}|{a} - {ref}'
                    cell = cell_c if (a in core and ref in core) else Cell(part, unit, [a, ref])
                    got = cell.paired(a, ref, label)
                    entry = gap_entry(got, variable, a, ref)
                    if entry is not None:
                        entry['rows_kind'] = 'common rows' if (a in core and ref in core) else 'own paired rows'
                        e[key][a] = entry
            ranked = sorted((v['mae'], a) for a, v in e['mae'].items() if a in core and v['mae'] is not None)
            e['best'] = ranked[0][1] if ranked else None
            block[str(int(length))] = e
        res.setdefault(cohort, {})[variable] = block
        print(f'  D {cohort} {variable}: {len(block)} lengths', flush=True)
    return res, gates, infos


# --------------------------------------------------------------------------------------------------
# Multiplicity and the summary
# --------------------------------------------------------------------------------------------------
def family(parts):
    out = []

    def walk(o, path):
        if isinstance(o, dict):
            if 'point' in o and 'p' in o and 'ci' in o:
                out.append((path, o))
                return
            for k, v in o.items():
                walk(v, f'{path}/{k}')
    for name, block in parts.items():
        walk(block, name)
    return out


def multiplicity(parts):
    entries = [(p, e) for p, e in family(parts) if e.get('p') is not None]
    ps = [e['p'] for _, e in entries]
    bh = FDR.step_up(ps)
    by = FDR.step_up(ps, dependence_robust=True)
    for (path, e), a, b in zip(entries, bh, by):
        e['bh'], e['by'] = bool(a), bool(b)
    return {'family': 'every new interval of parts A, B and D (reproduction gates excluded)',
            'intervals': len(entries), 'exclude_zero': int(sum(bool(e.get('excludes_zero')) for _, e in entries)),
            'bh_rejections': int(sum(bh)), 'by_rejections': int(sum(by)), 'q': FDR.Q}


def compact(variable, block, arms, kind):
    """MAE by lead of the named arms (common rows), the best arm, and the stored units' unit."""
    out = {}
    for cohort, leads in block.items():
        rows = {}
        for lead, cell in leads.items():
            row = {}
            for a in arms:
                m = cell['mae'].get(a, {})
                v = m.get('mae_common', m.get('mae_own'))
                if v is not None:
                    row[a] = r6(v * C_MG) if variable == 'oxygen' else v
            entry = {'mae': row, 'best': cell.get('best'), 'common_rows': cell['common_rows']}
            if kind == 'A':
                entry['best_forecast'] = cell.get('best_forecast')
                key = cell.get('network_adds')
                if key:
                    c = cell['contrasts'][key]
                    entry['network_adds'] = {'contrast': key, 'point': c.get('point_mgl', c['point']),
                                             'ci': c.get('ci_mgl', c['ci']), 'reading': c['reading']}
            rows[lead] = entry
        out[cohort] = rows
    return out


SLIDE_ARMS_B = ('persistence', 'clim_month_hour', 'clim_hour', 'tree', 'weather_tree', 'quantile_tree_weather',
                'arima_ss', 'linear_ar', 'nbeats', 'chronos2_net', 'chronos_bolt_base_uni', 'chronos2_uni', 'timesfm3_uni',
                'timesfm3_cov_past', 'blended')
SLIDE_ARMS_A = ('persistence', 'clim_hour', 'tree', 'weather_tree', 'quantile_tree_weather', 'arima_ss', 'nbeats',
                'chronos_bolt_base_uni', 'chronos2_uni', 'timesfm3_uni', 'blended', 'anchored_direct', 'neighbour_change',
                'recursive_tree', 'timesfm3_cov', 'chronos2_cov')


def main(parts):
    t0 = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    blocks, gates, extra = {}, {}, {}
    srcs = {}
    if 'A' in parts or 'B' in parts:
        gb = json.loads(CR.GATES.read_text())
        gates['A85-G1_to_G4'] = {k: gb[k].get('passed') if isinstance(gb[k], dict) and 'passed' in gb[k] else gb[k]
                                 for k in gb if k.startswith('A85')}
        if not gb.get('passed'):
            raise SystemExit('gates A85-G1 to G3 did not pass')
        for variable in VARIABLES:
            srcs[variable] = sources(variable)
    if 'A' in parts:
        g6 = gate_bridge72()
        gates['A85-G6_bridge'] = g6
        print('A gate G6:', g6['passed'], flush=True)
        if g6['passed']:
            blocks['A'] = {}
            extra['A'] = {}
            for variable in VARIABLES:
                blocks['A'][variable], extra['A'][variable] = part_a(variable, srcs[variable][0])
                gates[f'A85-G5_rows_A|{variable}'] = extra['A'][variable]['A85-G5_rows']['passed']
    if 'B' in parts:
        blocks['B'] = {}
        extra['B'] = {}
        for variable in VARIABLES:
            res, checks, _ = part_b(variable, srcs[variable][0])
            blocks['B'][variable] = res
            extra['B'][variable] = {'joins': checks, 'sources': srcs[variable][1], 'absent': srcs[variable][2]}
            gates[f'A85-G5_rows_B|{variable}'] = bool(all(c['passed'] for c in checks.values()))
    if 'C' in parts:
        blocks['C'], g7 = part_c()
        gates.update(g7)
    if 'D' in parts:
        blocks['D'], gd, infos = part_d()
        gates['A85-G6_gaps'] = gd['passed']
        extra['D'] = {'gate_cells': gd, 'frames': infos}
    mult = multiplicity({k: v for k, v in blocks.items() if k in ('A', 'B', 'D')})
    for name, block in blocks.items():
        write(OUT / f'{name}.json', {'declared_in': DECLARED, 'part': name, 'results': block,
                                     'records': extra.get(name), 'labels': LABELS})
    previous = json.loads(SUMMARY.read_text()) if SUMMARY.exists() else {}
    summary = {'declared_in': DECLARED, 'script': 'scripts/v2_common_rows_report.py',
               'unit': 'temperature degC; oxygen in mg/l in the compact tables (stored umol/kg x 0.032844 in the part files)',
               'gates': {**previous.get('gates', {}), **gates}, 'multiplicity': mult,
               'parts': {**previous.get('parts', {}), **{k: f'runs/v2_addendum85/{k}.json' for k in blocks}},
               'compact': previous.get('compact', {})}
    for variable in VARIABLES:
        if 'A' in blocks:
            summary['compact'].setdefault('A', {})[variable] = compact(variable, blocks['A'][variable], SLIDE_ARMS_A, 'A')
        if 'B' in blocks:
            summary['compact'].setdefault('B', {})[variable] = compact(variable, blocks['B'][variable], SLIDE_ARMS_B, 'B')
    if 'C' in blocks:
        summary['compact']['C'] = blocks['C']
    if 'D' in blocks and 'stopped' not in blocks['D']:
        dd = {}
        for cohort, vs in blocks['D'].items():
            for variable, lengths in vs.items():
                k = C_MG if variable == 'oxygen' else 1.0
                dd.setdefault(cohort, {})[variable] = {
                    L: {'best': e['best'], 'linear': r6(e['mae']['linear']['mae'] * k),
                        'incumbent': e['incumbent'],
                        'incumbent_mae': r6(e['mae'][e['incumbent']]['mae'] * k) if e['incumbent'] in e['mae'] else None,
                        'best_mae': r6(e['mae'][e['best']]['mae'] * k) if e['best'] else None,
                        'beat_linear': sorted(a for a, c in e['minus_linear'].items() if c.get('excludes_zero') and c['point'] < 0),
                        'beat_incumbent': sorted(a for a, c in e['minus_incumbent'].items()
                                                 if c.get('excludes_zero') and c['point'] < 0)}
                    for L, e in lengths.items()}
        summary['compact']['D'] = dd
    summary['seconds'] = round(time.time() - t0, 1)
    write(SUMMARY, summary)
    print('wrote', rel(SUMMARY), summary['seconds'], 's', flush=True)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--parts', default='A,B,C,D')
    a = ap.parse_args()
    main(set(a.parts.split(',')))
