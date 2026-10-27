"""Addendum 81, tasks C and D: the forecast distribution to 72 h and the continuation of a silent probe.

C  The forecast distribution at 1, 3, 6, 12, 24, 36, 48 and 72 h on Dev-2025 and the Reserved-2026 grid,
   on addendum 65's rows (the endpoints of its chronos2_net arm: 65,188 temperature, 15,028 oxygen).
   Quantile score: twice the pinball loss averaged over the nine levels 0.1 to 0.9, nested macro over
   sensors within each fold (Dev-2025) or quarter (2026), then over folds or quarters
   (v2_probabilistic_report.macro); the 80 % coverage (0.1 to 0.9) and the absolute error of the median
   with the same rule. Every arm of the declaration and every published arm whose quantiles are stored is
   scored at the same nine levels; beside each published arm the eleven-level score, recomputed here and
   as its own report printed it. The table of arms is read on the common rows (every scored new arm and
   both references present); each new arm is also scored on all of the task's rows. Contrasts K1-K4 on
   the nine-level score, paired on the rows where both arms exist, each drawn from its own stream of
   joint week draws (v2_probabilistic_report.joint_draws, 600 draws).

D  The continuation of a silent probe on Bridge-Dev and Bridge-2026 at 6 to 72 h. The prediction at lead
   h is position h of the arm's median from onset (v2_chronos_adapt_contrasts' rule). Absolute errors
   aggregated by the frozen (fold, sensor) macro (v2_bridge.macro) on the published rows (runs/v2_bridge
   at 6-48 h, runs/v2_bridge72 at 54-72 h); contrasts paired on the rows where both arms exist, with onset
   weeks drawn jointly within each fold (v2_bridge.joint_week_draws: the interval v2_bridge.week_blocks
   reports), each contrast on its own stream, and the sparse-cell reading of v2_bridge.sparse_cells.

Oxygen is scored in the sondes' stored column and given beside in mg/l (x c, src/units.py). Each contrast
carries its interval status and its reading against the floor that applies (v2_summary_table.against:
the quantisation scale for a contrast on one probe's own record, the co-located disagreement where an
arm reads other instruments, the sonde floor for oxygen), and a p-value from its interval
(v2_fdr.p_side) for the family adjustment. No interval on fewer than five independent units.

Gates
  C-G1  chronos2_cov reproduces addendum 65's chronos2_net at the nine levels within 1e-3 at every lead
  C-G2  the task's rows are the published chronos2_net rows, each joined to one input row; every input
        row scored somewhere; every arm finite on every row; prediction files match their inputs
  C-T   truth and last value rebuilt from the observations equal addendum 64/65's points (1e-9), and
        the fold rule (issuance month) reproduces every published fold
  C-R   the eleven-level score recomputed on each published report's own rows reproduces that report
  D-G1  chronos_bolt_base_uni reproduces the published zero-shot continuation at positions 1-48 (1e-4)
  D-G2  every episode of runs/v2_bridge and runs/v2_bridge72 has an input row, every input row a scored
        row, every arm finite where scored
  D-R   the contrast path reproduces published addendum-41 and addendum-63 intervals exactly

Writes reports/bay_twin/runs/v2_fm_zoo/score_dist_bridge.json only when preds/ holds the declared 1,128
files and preds/rerun_check_a3.json exists, with every arm; anything less is a partial reading and goes
to --out.

Usage:
  python -m scripts.v2_fm_zoo_score_dist_bridge
  python -m scripts.v2_fm_zoo_score_dist_bridge --arms chronos_bolt_small_uni,chronos_bolt_base_uni,chronos2_uni,chronos2_cov --out PATH
  python -m scripts.v2_fm_zoo_score_dist_bridge --gates-only --out PATH
"""
import argparse
import functools
import json
import sys
import time
import zlib
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import scripts.v2_fdr as FDR  # noqa: E402
import scripts.v2_summary_table as S  # noqa: E402
from scripts.v2_fm_zoo_io import ARMS, LEVELS, digest, load_inputs, load_predictions  # noqa: E402
from src.experiments import v2_arima_report as A  # noqa: E402
from src.experiments import v2_bridge, v2_bridge72  # noqa: E402
from src.experiments import v2_ensemble as E  # noqa: E402
from src.experiments import v2_probabilistic as P  # noqa: E402
from src.experiments import v2_probabilistic_report as R  # noqa: E402
from src.experiments import v2_weather_oracle_report as W  # noqa: E402
from src.units import factor  # noqa: E402

RUNS = ROOT / 'reports/bay_twin/runs'
ZOO = RUNS / 'v2_fm_zoo'
INP = ZOO / 'inputs'
PRED = ZOO / 'preds'
OUT = ZOO / 'score_dist_bridge.json'
MANIFEST = INP / 'manifest_dist_bridge.json'
RERUN_A3 = PRED / 'rerun_check_a3.json'
DECLARED_FILES = 1128
DECLARED_IN = 'config/bay_reserved_protocol_v2_addendum81.json'
AMENDMENTS = [f'config/bay_reserved_protocol_v2_addendum81_amendment{i}.json' for i in (1, 2, 3)]

PROB = RUNS / 'v2_probabilistic'
ORACLE = RUNS / 'v2_weather_oracle'
ARIMA = RUNS / 'v2_arima'
STATESPACE = RUNS / 'v2_statespace'
BAYES = RUNS / 'v2_bayes_ts'
ENSEMBLE = RUNS / 'v2_ensemble'
ADAPT = RUNS / 'v2_chronos_adapt'

VARIABLES = ('temperature', 'oxygen')
C_COHORTS = R.COHORTS
D_COHORTS = ('Bridge-Dev', 'Bridge-2026')
C_LEADS = list(W.LEADS)                                  # 1, 3, 6, 12, 24, 36, 48, 72
D_LEADS = list(v2_bridge.LEADS) + list(v2_bridge72.LEADS)
KEY = W.KEY
NINE = tuple(LEVELS)
Q9 = [P.qcol(a) for a in NINE]
IN11 = [P.QCOLS.index(c) for c in Q9]                    # the nine levels inside the eleven
MED11 = P.QCOLS.index('q50')
BASE = 'chronos_bolt_base_uni'
CHRONOS_ARMS = ARMS[:4]
C_REFS = ('clim_quantiles', 'quantile_tree')
D_REFS = {'temperature': ('persistence', 'anchored_direct'), 'oxygen': ('persistence', 'clim_hour')}
D_MAIN = {'temperature': 'anchored_direct', 'oxygen': 'clim_hour'}
D_PUBLISHED = ('persistence', 'anchored_direct', 'clim_hour', 'clim_month_hour', 'neighbour_change')
K3 = (('chronos2_cov', 'chronos2_uni'), ('timesfm25_cov', 'timesfm25_uni'), ('timesfm3_cov', 'timesfm3_uni'))
K4 = (('timesfm3_uni', 'timesfm25_uni'), ('chronos2_uni', 'chronos_bolt_base_uni'))
#: arms that read instruments other than the forecast probe while it is silent
READS_OTHERS = {'anchored_direct', 'neighbour_change'} | {a for a in ARMS if a.endswith('_cov')}
MIN_UNITS = 5
G1C_TOL, G1D_TOL, REPRO_TOL = 1e-3, 1e-4, 1e-6 + 1e-12
#: the oracle report's arms: its rows are addendum 64/65's points where all of them exist
ORACLE_ARMS = W.ARMS
#: addendum 64's main arms, renamed where addendum 65 reran them at prediction length 72
A64_RENAME = {'chronos2': 'chronos2_a64', 'chronos2_net': 'chronos2_net_a64', 'chronos2_net_tide': 'chronos2_net_tide_a64'}
A64_MAIN = ('clim_quantiles', 'quantile_tree', 'quantile_tree_tide', 'chronos2_a64', 'chronos2_net_a64',
            'chronos2_net_tide_a64')
C_FACTOR = factor()


def r(x):
    return R.r(x)


def rel(path):
    return Path(path).resolve().relative_to(ROOT).as_posix()


def mg(variable, x):
    """An oxygen quantity in mg/l (None for temperature)."""
    if variable != 'oxygen' or x is None:
        return None
    return r(x * C_FACTOR)


def jsonable(o):
    if isinstance(o, np.bool_):
        return bool(o)
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return r(float(o))
    if isinstance(o, Path):
        return o.as_posix()
    raise TypeError(type(o))


@functools.lru_cache(maxsize=1)
def floors():
    """The floors Table 6.1 reads (v2_summary_table.floors): degC and mg/l."""
    f = S.floors()
    return {'temperature_quantisation': float(f['quant']), 'temperature_colocated': float(f['coloc_low']),
            'oxygen_mgl': float(f['sonde'])}


def floor_for(variable, x, y):
    fl = floors()
    if variable == 'oxygen':
        return fl['oxygen_mgl'], 'sonde (mg/l)'
    if x in READS_OTHERS or y in READS_OTHERS:
        return fl['temperature_colocated'], 'co-located disagreement (an arm reads other instruments)'
    return fl['temperature_quantisation'], 'quantisation (one probe\'s own record)'


def decorate(c, variable, x, y):
    """Add mg/l values, the floor reading and the interval's p-value to a contrast in stored units."""
    floor, basis = floor_for(variable, x, y)
    point, ci = c['point'], c.get('ci')
    if variable == 'oxygen':
        c['point_mgl'] = mg(variable, point)
        c['ci_mgl'] = None if ci is None else [mg(variable, v) for v in ci]
        point, ci = c['point_mgl'], c['ci_mgl']
    c['floor'] = floor
    c['floor_basis'] = basis
    lo, hi = (None, None) if ci is None else ci
    c['reading'] = S.against(point, lo, hi, floor)
    if variable == 'temperature' and floor != floors()['temperature_quantisation']:
        c['reading_against_quantisation'] = S.against(point, lo, hi, floors()['temperature_quantisation'])
    # nine significant figures, not six decimals: the family adjustment ranks p-values near 1e-6
    p = None if ci is None else FDR.p_side(c['point'], *c['ci'])
    c['p'] = None if p is None else float(f'{p:.9g}')
    return c


def preds_state():
    files = sorted(p.name for p in PRED.glob('*.npz'))
    return {'npz_files': len(files), 'declared_files': DECLARED_FILES, 'rerun_check_a3': RERUN_A3.exists(),
            'ready': len(files) == DECLARED_FILES and RERUN_A3.exists()}


def prediction(stem, arm, sha):
    path = PRED / f'{stem}__{arm}.npz'
    p = load_predictions(path)
    m = p['meta']
    rec = {'file': rel(path), 'sha256': digest(path), 'input_sha256_matches': m.get('input_sha256') == sha,
           'amendments': [Path(a).name for a in m.get('amendments', [])], 'overrides': m.get('overrides'),
           'model_revision': m.get('model_revision'), 'created_utc': m.get('created_utc'),
           'shape_median': list(p['median'].shape), 'shape_q': list(p['q'].shape)}
    return p, rec


def manifest():
    return json.loads(MANIFEST.read_text())


@functools.lru_cache(maxsize=1)
def observations():
    _, frames = P.bay_reserved_evaluation.load_all_observations()
    return frames


def values_at(series, times_ns):
    """The hourly series' values at the given UTC times (int64 ns); NaN where the hour is not on it."""
    idx = pd.DatetimeIndex(series.index).as_unit('ns').asi8
    if not (np.diff(idx) > 0).all():
        raise ValueError('series index is not strictly increasing')
    vals = series.to_numpy(float)
    pos = np.searchsorted(idx, times_ns)
    ok = pos < len(idx)
    ok[ok] = idx[pos[ok]] == times_ns[ok]
    out = np.full(len(times_ns), np.nan)
    out[ok] = vals[pos[ok]]
    return out


def ns(times):
    return pd.DatetimeIndex(times).as_unit('ns').asi8


# ==================================================================================================
# Task C
# ==================================================================================================
def c_rows(variable):
    """The task's rows: addendum 65's chronos2_net endpoints, with truth and last value from the
    observations, the fold (Dev-2025: issuance in Jul-Sep or Oct-Dec) or quarter (2026) and ISO week."""
    pub = W.read(ORACLE / 'chronos' / f'chronos_{variable}.csv.gz')
    net = pub[pub.arm == 'chronos2_net'].drop(columns='arm').reset_index(drop=True)
    rows = net[KEY].copy()
    t = ns(rows.time)
    lead = rows.lead.to_numpy(np.int64) * 3_600_000_000_000
    truth, last = np.full(len(rows), np.nan), np.full(len(rows), np.nan)
    frames = observations()
    for sensor, idx in rows.groupby('sensor').indices.items():
        s = frames[sensor][variable]
        truth[idx] = values_at(s, t[idx] + lead[idx])
        last[idx] = values_at(s, t[idx])
    rows['truth'], rows['last'] = truth, last
    month = rows.time.dt.month.to_numpy()
    rows['fold'] = np.where(rows.cohort == 'Dev-2025', np.where(month <= 9, '0', '1'),
                            np.where(month <= 3, 'Q1', 'Q2'))
    rows['week'] = rows.time.dt.isocalendar().week.astype(int).to_numpy()
    return rows, net


def c_points(variable):
    p64 = W.read(PROB / 'seed42' / f'points_{variable}.csv.gz')
    p72 = W.read(ORACLE / 'seed42' / f'points72_{variable}.csv.gz')
    return p64, pd.concat([p64, p72], ignore_index=True)


def gate_truth(rows, points):
    j = rows[KEY + ['truth', 'last', 'fold']].merge(points[KEY + ['truth', 'last', 'fold']], on=KEY, how='outer',
                                                    suffixes=('', '_pub'), indicator=True)
    both = j[j._merge == 'both']
    pub_fold = np.where(both.cohort == 'Dev-2025', both.fold_pub.astype('float').astype('Int64').astype(str),
                        np.where(both.time.dt.month <= 3, 'Q1', 'Q2'))
    out = {'rows_task': int(len(rows)), 'rows_points': int(len(points)), 'points_in_task': int(len(both)),
           'points_not_in_task': int((j._merge == 'right_only').sum()),
           'task_rows_without_points': int((j._merge == 'left_only').sum()),
           'truth_missing': int(rows.truth.isna().sum()), 'last_missing': int(rows['last'].isna().sum()),
           'max_abs_truth_difference': float((both.truth - both.truth_pub).abs().max()),
           'max_abs_last_difference': float((both['last'] - both.last_pub).abs().max()),
           'fold_mismatches': int((both.fold.to_numpy() != pub_fold).sum()),
           'note': 'points rows not in the task are endpoints the quantile tree scored where the published '
                   'chronos2_net arm had no context (fewer than 48 finite hours); the oracle report never '
                   'scored them either'}
    out['passed'] = bool(out['truth_missing'] == 0
                         and out['max_abs_truth_difference'] <= 1e-9 and out['max_abs_last_difference'] <= 1e-9
                         and out['fold_mismatches'] == 0)
    return out


def published_quantiles(variable):
    """Every stored published arm of the task: name -> rows (KEY + the eleven quantile columns)."""
    parts, files, absent = {}, {}, []

    def add(path, name_of):
        if not path.exists():
            absent.append(rel(path))
            return
        f = W.read(path)
        for arm, g in f.groupby('arm'):
            name = name_of(arm)
            parts.setdefault(name, []).append(g[KEY + P.QCOLS])
            files.setdefault(name, []).append(rel(path))

    for seed in P.SEEDS:
        tag = functools.partial(lambda a, s: W.tag(a, s), s=seed)
        add(PROB / f'seed{seed}' / f'quantiles_{variable}_quantile_tree.csv.gz', tag)
        add(PROB / f'seed{seed}' / f'quantiles_{variable}_quantile_tree_tide.csv.gz', tag)
        for arm in ('quantile_tree_weather', 'quantile_tree_weather_future'):
            add(ORACLE / f'seed{seed}' / f'quantiles_{variable}_{arm}.csv.gz', tag)
        for arm in A.FITTED[variable]:
            add(ARIMA / f'seed{seed}' / f'quantiles_{variable}_{arm}.csv.gz', tag)
        for arm in ('arima_ss', 'nbeats'):
            add(STATESPACE / f'seed{seed}' / f'quantiles_{variable}_{arm}.csv.gz', tag)
        for arm in ('dlm', 'dlm_emp'):
            add(BAYES / f'seed{seed}' / f'quantiles_{variable}_{arm}.csv.gz', tag)
    add(STATESPACE / 'seed42' / f'quantiles_{variable}_auto_arima.csv.gz', lambda a: a)
    add(BAYES / 'seed42' / f'quantiles_{variable}_bsts.csv.gz', lambda a: a)
    add(PROB / 'chronos' / f'chronos_{variable}.csv.gz', lambda a: A64_RENAME[a])
    add(ORACLE / 'chronos' / f'chronos_{variable}.csv.gz', lambda a: a)
    out = {}
    for name, ps in parts.items():
        f = pd.concat(ps, ignore_index=True)
        if f.duplicated(KEY).any():
            raise ValueError(f'{variable} {name}: duplicated rows across {files[name]}')
        out[name] = f
    return out, files, absent


def align(rows, long):
    j = rows[KEY].merge(long, on=KEY, how='left', validate='one_to_one')
    return j[P.QCOLS].to_numpy(float)


def qscore(y, Q, levels):
    a = np.asarray(levels, float)[None, :]
    d = y[:, None] - Q
    return (2 * np.maximum(a * d, (a - 1) * d)).mean(1)


def put_metrics(cols, name, y, Q9_, med, Q11=None):
    cols[f'qs9|{name}'] = qscore(y, Q9_, NINE)
    ok = np.isfinite(Q9_).all(1)
    c80 = ((Q9_[:, 0] <= y) & (y <= Q9_[:, -1])).astype(float)
    cols[f'c80|{name}'] = np.where(ok, c80, np.nan)
    cols[f'ae|{name}'] = np.where(np.isfinite(med), np.abs(y - med), np.nan)
    if Q11 is not None:
        cols[f'qs11|{name}'] = R.quantile_score(y, Q11)
    crossing = ok & (np.diff(Q9_, axis=1) < 0).any(1)
    return int(crossing.sum())


def ensemble_q11(Q11s, members, rows_mask=None):
    """v2_ensemble.combine on the rows where every member exists (and rows_mask)."""
    if not all(m in Q11s for m in members):
        return None
    ok = np.logical_and.reduce([np.isfinite(Q11s[m]).all(1) for m in members])
    if rows_mask is not None:
        ok &= rows_mask
    out = np.full_like(Q11s[members[0]], np.nan)
    if ok.any():
        sub = pd.DataFrame({f'{m}|{c}': Q11s[m][ok, i] for m in members for i, c in enumerate(P.QCOLS)})
        E.combine(sub, members, 'X')
        out[ok] = sub[[f'X|{c}' for c in P.QCOLS]].to_numpy(float)
    return out


def c_frame(variable, arms, man):
    """One row per task endpoint with per-row scores of every arm; the gates C-G1, C-G2 and C-T."""
    rows, net = c_rows(variable)
    p64, points = c_points(variable)
    gates = {'C-T': gate_truth(rows, points)}
    k64 = p64[KEY].assign(_in64=True)
    kall = points[KEY].assign(_inpts=True)
    rows = rows.merge(kall, on=KEY, how='left', validate='one_to_one').merge(k64, on=KEY, how='left',
                                                                             validate='one_to_one')
    rows['in_points'] = rows._inpts.fillna(False).astype(bool)
    rows['in_points64'] = rows._in64.fillna(False).astype(bool)
    rows = rows.drop(columns=['_inpts', '_in64'])
    y = rows.truth.to_numpy(float)
    cols, crossings, records = {}, {}, {}
    # ---- the new arms: one input file per cohort, rows joined through the keys
    files = man['C_distribution']['files']
    lookup, g2 = [], {'rows_task': int(len(rows)), 'by_cohort': {}}
    published_rows = man['C_distribution']['gates'][variable]['C2_observed_leads']['rows_published']
    for cohort in C_COHORTS:
        stem = f'C_{variable}_{cohort}'
        inp = load_inputs(INP / f'{stem}.npz')
        k = inp['keys'].copy()
        k['row'] = np.arange(len(k))
        k['cohort_file'] = cohort
        lookup.append(k)
        g2['by_cohort'][cohort] = {'input_rows': int(len(k)), 'input_sha256_matches_manifest':
                                   digest(INP / f'{stem}.npz') == files[f'{stem}.npz']['sha256']}
    keys = pd.concat(lookup, ignore_index=True)
    keys['t'] = keys.time.astype(np.int64)
    rows['t'] = ns(rows.time)
    j = rows[['cohort', 'sensor', 't']].merge(keys[['cohort', 'sensor', 't', 'row']], on=['cohort', 'sensor', 't'],
                                             how='left', validate='many_to_one')
    rows['row'] = j.row.to_numpy()
    g2['task_rows_without_input'] = int(rows.row.isna().sum())
    used = rows.dropna(subset=['row']).groupby('cohort').row.nunique()
    for cohort in C_COHORTS:
        g2['by_cohort'][cohort]['input_rows_scored'] = int(used.get(cohort, 0))
    g2['rows_published_in_manifest'] = int(published_rows)
    if g2['task_rows_without_input']:
        g2['passed'] = False
        gates['C-G2'] = g2
        return None, gates, {}
    rows['row'] = rows.row.astype(int)
    lead_idx = rows.lead.to_numpy(int) - 1
    g1 = {}
    for arm in arms:
        Qn = np.full((len(rows), len(NINE)), np.nan)
        med = np.full(len(rows), np.nan)
        recs = {}
        for cohort in C_COHORTS:
            stem = f'C_{variable}_{cohort}'
            p, rec = prediction(stem, arm, files[f'{stem}.npz']['sha256'])
            recs[cohort] = rec
            m = (rows.cohort == cohort).to_numpy()
            ri = rows.row.to_numpy()[m]
            n_in = g2['by_cohort'][cohort]['input_rows']
            rec['rows_match_input'] = bool(p['median'].shape == (n_in, 72) and p['q'].shape == (n_in, len(NINE), 72))
            Qn[m] = p['q'][ri, :, lead_idx[m]]
            med[m] = p['median'][ri, lead_idx[m]]
        records[arm] = recs
        missing = int((~np.isfinite(Qn).all(1) | ~np.isfinite(med)).sum())
        g2.setdefault('arm_missing_rows', {})[arm] = missing
        crossings[arm] = put_metrics(cols, arm, y, Qn, med)
        cols[f'medq50|{arm}'] = np.abs(med - Qn[:, NINE.index(0.5)])
        if arm == 'chronos2_cov':
            pub = net[P.QCOLS].to_numpy(float)[:, IN11]
            d = np.abs(Qn - pub)
            for lead in C_LEADS:
                mm = rows.lead.to_numpy() == lead
                g1[str(lead)] = {'rows': int(mm.sum()), 'max_abs_difference': float(np.nanmax(d[mm])),
                                 'values_above_tolerance': int((d[mm] > G1C_TOL).sum()),
                                 'non_finite': int((~np.isfinite(d[mm])).sum())}
    if 'chronos2_cov' in arms:
        gates['C-G1'] = {'compared': 'chronos2_cov vs runs/v2_weather_oracle/chronos chronos2_net, levels 0.1-0.9',
                         'tolerance': G1C_TOL, 'by_lead': g1,
                         'passed': all(v['values_above_tolerance'] == 0 and v['non_finite'] == 0 for v in g1.values())}
    g2['passed'] = bool(g2['task_rows_without_input'] == 0 and len(rows) == published_rows
                        and all(c['input_rows_scored'] == c['input_rows'] and c['input_sha256_matches_manifest']
                                for c in g2['by_cohort'].values())
                        and all(v == 0 for v in g2.get('arm_missing_rows', {}).values())
                        and all(rec['input_sha256_matches'] and rec['rows_match_input']
                                for recs in records.values() for rec in recs.values()))
    gates['C-G2'] = g2
    # ---- the published arms
    long, src, absent = published_quantiles(variable)
    Q11s = {name: align(rows, f) for name, f in long.items()}
    ewma_gate = None
    try:
        frames, ewma_gate = A.banded(variable, A.LAMBDA)
        for name, f in frames.items():
            Q11s[name] = align(rows, f[KEY + P.QCOLS])
            src[name] = [f'computed by src/experiments/v2_arima_report.banded (lambda {A.LAMBDA}) from '
                         'runs/v2_weather_oracle/chronos and runs/v2_arima/seed42']
    except Exception as e:                                    # recorded, not fatal: the bands are a recalibration
        absent.append(f'EWMA bands not computed: {type(e).__name__}: {e}')
    dev = (rows.cohort == 'Dev-2025').to_numpy()
    combos = {'E3': (E.E3, None), 'E5': (E.E5, None)}
    combos.update({f'E3@{s}': ((f'quantile_tree@{s}', 'chronos2_net', f'arima_ss@{s}'), dev) for s in E.SEEDS})
    for name, (members, mask) in combos.items():
        q = ensemble_q11(Q11s, members, mask)
        if q is not None:
            Q11s[name] = q
            src[name] = [f'computed by src/experiments/v2_ensemble.combine from {", ".join(members)}']
    for name, Q11 in Q11s.items():
        crossings[name] = put_metrics(cols, name, y, Q11[:, IN11], Q11[:, MED11], Q11)
    frame = pd.concat([rows.reset_index(drop=True), pd.DataFrame(cols)], axis=1)
    info = {'published_arms': {k: sorted(set(v)) for k, v in sorted(src.items())}, 'absent_files': absent,
            'ewma_gate': ewma_gate, 'crossing_rows': crossings, 'prediction_files': records,
            'median_vs_q50_max_abs': {a: r(float(np.nanmax(frame[f'medq50|{a}']))) for a in arms}}
    return frame, gates, info


def published_reports(variable):
    """Each published report's eleven-level scores: report name -> (row set, name map, scores block).
    The reports score nine levels in their primary block and keep the eleven-level score beside it
    as `<block>_eleven_levels`; that one is read where it exists."""
    def scores(path, block, variable):
        node = json.loads(path.read_text())
        return node.get(f'{block}_eleven_levels', node[block])[variable]
    return {
        'v2_weather_oracle': ('oracle', {}, scores(ORACLE / 'report.json', 'scores', variable)),
        'v2_probabilistic': ('a64', A64_RENAME, scores(PROB / 'report.json', 'part_b', variable)),
        'v2_arima': ('oracle', {}, scores(ARIMA / 'report.json', 'scores', variable)),
        'v2_statespace': ('oracle', {}, scores(STATESPACE / 'report.json', 'scores', variable)),
        'v2_bayes_ts': ('oracle', {}, scores(BAYES / 'report.json', 'scores', variable)),
        'v2_ensemble': ('oracle', {}, scores(ENSEMBLE / 'report.json', 'scores', variable)),
    }


#: where each published arm's eleven-level score is first printed
HOME_ORDER = ('v2_weather_oracle', 'v2_probabilistic', 'v2_arima', 'v2_statespace', 'v2_bayes_ts', 'v2_ensemble')


def row_sets(frame):
    has = lambda a: frame[f'qs11|{a}'].notna() if f'qs11|{a}' in frame else pd.Series(False, index=frame.index)
    oracle = frame.in_points.copy()
    for a in ORACLE_ARMS:
        oracle &= has(a)
    a64 = frame.in_points64.copy()
    for a in A64_MAIN:
        a64 &= has(a)
    return {'oracle': oracle.to_numpy(), 'a64': a64.to_numpy()}


def reproduce(variable, frame, reports):
    """C-R: the eleven-level score (and the 80 % coverage where printed) of every published arm,
    recomputed on its report's own rows, against the report."""
    sets = row_sets(frame)
    out = {}
    for name, (rowset, rename, block) in reports.items():
        worst, compared, failed = {}, 0, []
        for cohort in C_COHORTS:
            for lead, cell_pub in block.get(cohort, {}).items():
                m = sets[rowset] & (frame.cohort == cohort).to_numpy() & (frame.lead == int(lead)).to_numpy()
                cell = frame[m]
                for parm, stats in cell_pub.get('arms', {}).items():
                    mine = rename.get(parm, parm)
                    col = f'qs11|{mine}'
                    if col not in cell or not len(cell) or cell[col].isna().any() or 'quantile_score' not in stats:
                        continue
                    pairs = [(R.macro(cell, col), stats['quantile_score'])]
                    if 'coverage80' in stats and f'c80|{mine}' in cell:
                        pairs.append((R.macro(cell, f'c80|{mine}'), stats['coverage80']))
                    for got, want in pairs:
                        if want is None:
                            continue
                        d = abs(r(got) - want)
                        compared += 1
                        worst[mine] = max(worst.get(mine, 0.0), d)
                        if d > REPRO_TOL:
                            failed.append([cohort, lead, mine, r(got), want])
        out[name] = {'rows': rowset, 'values_compared': compared, 'max_abs_difference_by_arm': {k: r(v) for k, v in worst.items()},
                     'failures': failed[:20], 'passed': bool(compared > 0 and not failed)}
    # the nine-level function equals the published eleven-level one when given the eleven levels
    rng = np.random.default_rng(0)
    yy, QQ = rng.normal(size=50), np.sort(rng.normal(size=(50, 11)), axis=1)
    same = float(np.abs(qscore(yy, QQ, P.LEVELS) - R.quantile_score(yy, QQ)).max())
    out['quantile_score_function'] = {'max_abs_difference_at_eleven_levels': same, 'passed': same == 0.0}
    return out


def c_contrast(cell, x, y, variable, cohort, lead):
    cx, cy = f'qs9|{x}', f'qs9|{y}'
    sub = cell.dropna(subset=[cx, cy])
    units = int(sub[['fold', 'week']].drop_duplicates().shape[0])
    c = {'rows': int(len(sub)), 'sensors': int(sub.sensor.nunique()), 'units_fold_week': units}
    if not len(sub):
        return None
    if units < MIN_UNITS:
        c.update(point=r(R.macro(sub, cx) - R.macro(sub, cy)), ci=None, excludes_zero=None,
                 not_estimable=f'{units} fold-week units, fewer than {MIN_UNITS}')
    else:
        label = f'fmzoo|C|{variable}|{cohort}|{lead}|{x} - {y}'
        draws = R.joint_draws(sub, [cx, cy], label)
        c.update(R.contrast(sub, draws, [cx, cy], cx, cy))
    return decorate(c, variable, x, y)


def arm_stats(sub, name, variable, published=False):
    col = f'qs9|{name}'
    if col not in sub:
        return None
    s = sub[sub[col].notna()]
    if not len(s):
        return None
    e = {'rows': int(len(s)), 'missing_rows': int(len(sub) - len(s)), 'sensors': int(s.sensor.nunique()),
         'quantile_score9': r(R.macro(s, col)), 'coverage80': r(R.macro(s, f'c80|{name}')),
         'median_mae': r(R.macro(s, f'ae|{name}'))}
    if published and f'qs11|{name}' in s:
        e['quantile_score11'] = r(R.macro(s, f'qs11|{name}'))
    if variable == 'oxygen':
        e['mgl'] = {k: mg(variable, e[k]) for k in ('quantile_score9', 'quantile_score11', 'median_mae') if k in e}
    return e


def c_families(arms):
    fam = {'K1': [(a, ref) for a in arms for ref in C_REFS],
           'K2': [(a, BASE) for a in arms if a != BASE and BASE in arms],
           'K3': [(x, y) for x, y in K3 if x in arms and y in arms],
           'K4': [(x, y) for x, y in K4 if x in arms and y in arms]}
    return fam


def c_score(variable, frame, arms, reports, published_names):
    home = {}
    for name in HOME_ORDER:
        rowset, rename, block = reports[name]
        inv = {v: k for k, v in rename.items()}
        for mine in published_names:
            if mine in home:
                continue
            for cohort in C_COHORTS:
                if any(inv.get(mine, mine) in cell.get('arms', {}) for cell in block.get(cohort, {}).values()):
                    home[mine] = (name, inv.get(mine, mine))
                    break
    fam = c_families(arms)
    pairs = list(dict.fromkeys(p for v in fam.values() for p in v))
    out = {}
    for (cohort, lead), cell in frame.groupby(['cohort', 'lead']):
        common = np.ones(len(cell), bool)
        for a in list(arms) + list(C_REFS):
            common &= cell[f'qs9|{a}'].notna().to_numpy()
        K = cell[common]
        block = {'rows_task': int(len(cell)), 'sensors_task': int(cell.sensor.nunique()),
                 'rows_common': int(len(K)), 'sensors_common': int(K.sensor.nunique()),
                 'units_common': int(K[['fold', 'week']].drop_duplicates().shape[0]),
                 'arms': {}, 'published_arms': {}, 'all_rows': {}, 'contrasts': {}}
        for a in arms:
            block['arms'][a] = arm_stats(K, a, variable)
            block['all_rows'][a] = arm_stats(cell, a, variable)
        for name in published_names:
            e = arm_stats(K, name, variable, published=True)
            if e is None:
                continue
            if name in home:
                rep, pname = home[name]
                pub = reports[rep][2].get(cohort, {}).get(str(lead), {}).get('arms', {}).get(pname, {})
                e['published_quantile_score11'] = pub.get('quantile_score')
                e['published_in'] = f'runs/{rep}/report.json'
                if variable == 'oxygen':
                    e['mgl']['published_quantile_score11'] = mg(variable, pub.get('quantile_score'))
            block['published_arms'][name] = e
        for x, y in pairs:
            c = c_contrast(cell, x, y, variable, cohort, lead)
            if c is not None:
                block['contrasts'][f'{x} - {y}'] = c
        block['families'] = {k: [f'{x} - {y}' for x, y in v] for k, v in fam.items()}
        out.setdefault(cohort, {})[str(lead)] = block
        print(f'  C {variable} {cohort} {lead} h: {len(cell)} rows, {len(K)} common', flush=True)
    return out


def task_c(arms, man):
    rep = {'leads': C_LEADS, 'levels': list(NINE), 'draws': R.DRAWS,
           'aggregation': 'nested macro: sensors within fold (Dev-2025) or quarter (2026), then folds or quarters',
           'gates': {}, 'info': {}, 'scores': {}}
    for variable in VARIABLES:
        t0 = time.time()
        frame, gates, info = c_frame(variable, arms, man)
        if frame is None:
            rep['gates'][variable] = gates
            rep['scores'][variable] = {'not_scored': 'a gate failed'}
            continue
        reports = published_reports(variable)
        published_names = sorted(c.split('|', 1)[1] for c in frame.columns if c.startswith('qs11|'))
        gates['C-R'] = reproduce(variable, frame, reports)
        gates['C-R']['passed'] = all(v['passed'] for v in gates['C-R'].values() if isinstance(v, dict) and 'passed' in v)
        rep['gates'][variable] = gates
        rep['info'][variable] = info
        print(f'C {variable}: gates', {k: v.get('passed') for k, v in gates.items()}, f'{time.time() - t0:.0f} s', flush=True)
        if not all(gates[k]['passed'] for k in ('C-G2', 'C-T', 'C-R')) or not gates.get('C-G1', {'passed': True})['passed']:
            rep['scores'][variable] = {'not_scored': 'a gate failed'}
            continue
        if GATES_ONLY:
            continue
        rep['scores'][variable] = c_score(variable, frame, arms, reports, published_names)
    rep['not_rescored'] = {
        'cqr and aci intervals (addendum 70)': 'a recalibration of the 80 % interval only; no quantile set is stored or defined',
        'EWMA bands at lambda 0.90, 0.97, 0.99 (addendum 66)': 'sensitivity settings of the declared 0.94 band; not part of the arm set',
        'bsts Gibbs attempt (runs/v2_bayes_ts/gibbs)': 'superseded sampler, not a declared output',
        'quantile_tree_weather_future, chronos2_net_weather_future': 'rescored, but they read the observed weather after issuance (a perfect weather forecast, addendum 65), not an operational arm',
        'chronos2_a64, chronos2_net_a64, chronos2_net_tide_a64': 'rescored, at prediction length 48 (addendum 64); compare with Chronos-2 arms at 72 only with care',
    }
    return rep


# ==================================================================================================
# Task D
# ==================================================================================================
def d_frame(variable, arms, man):
    a = pd.read_csv(RUNS / 'v2_bridge' / f'bridge_{variable}.csv.gz').assign(source='runs/v2_bridge')
    b = pd.read_csv(RUNS / 'v2_bridge72' / f'bridge72_{variable}.csv.gz').assign(source='runs/v2_bridge72')
    f = pd.concat([a, b], ignore_index=True)
    f['onset'] = pd.to_datetime(f.onset, utc=True).astype('datetime64[ns, UTC]')
    iso = f.onset.dt.isocalendar()
    f['week'] = iso.year.astype(str) + '-' + iso.week.astype(str)
    f['t'] = ns(f.onset)
    files = man['D_silent_probe']['files']
    keys, g2 = [], {'rows': int(len(f)), 'by_cohort': {}}
    for cohort in D_COHORTS:
        stem = f'D_{variable}_{cohort}'
        inp = load_inputs(INP / f'{stem}.npz')
        k = inp['keys'].copy()
        k['row'] = np.arange(len(k))
        keys.append(k)
        g2['by_cohort'][cohort] = {'input_rows': int(len(k)), 'input_sha256_matches_manifest':
                                   digest(INP / f'{stem}.npz') == files[f'{stem}.npz']['sha256']}
    keys = pd.concat(keys, ignore_index=True)
    keys['t'] = keys.onset.astype(np.int64)
    keys['fold'] = keys.fold.astype(int)
    j = f[['cohort', 'fold', 'sensor', 't']].merge(keys[['cohort', 'fold', 'sensor', 't', 'row']],
                                                   on=['cohort', 'fold', 'sensor', 't'], how='left', validate='many_to_one')
    f['row'] = j.row.to_numpy()
    g2['rows_without_input'] = int(f.row.isna().sum())
    g2['episodes'] = {src: int(g.groupby(['cohort', 'fold', 'sensor', 'onset']).ngroups) for src, g in f.groupby('source')}
    g2['episodes_without_input'] = {src: int(g[g.row.isna()].groupby(['cohort', 'fold', 'sensor', 'onset']).ngroups)
                                    for src, g in f.groupby('source')}
    used = f.dropna(subset=['row']).groupby('cohort').row.nunique()
    for cohort in D_COHORTS:
        g2['by_cohort'][cohort]['input_rows_scored'] = int(used.get(cohort, 0))
    step = f.lead.to_numpy(int) - 1
    records, medians = {}, {}
    for arm in arms:
        vals = np.full(len(f), np.nan)
        recs = {}
        for cohort in D_COHORTS:
            stem = f'D_{variable}_{cohort}'
            p, rec = prediction(stem, arm, files[f'{stem}.npz']['sha256'])
            n_in = g2['by_cohort'][cohort]['input_rows']
            rec['rows_match_input'] = bool(p['median'].shape == (n_in, 72) and p['q'].shape == (n_in, len(NINE), 72))
            recs[cohort] = rec
            m = (f.cohort == cohort).to_numpy()
            vals[m] = p['median'][f.row.to_numpy()[m].astype(int), step[m]]
            medians[(arm, cohort)] = p['median']
        records[arm] = recs
        f[arm] = vals
        g2.setdefault('arm_missing_rows', {})[arm] = int(np.isnan(vals).sum())
    g2['passed'] = bool(g2['rows_without_input'] == 0
                        and all(c['input_rows_scored'] == c['input_rows'] and c['input_sha256_matches_manifest']
                                for c in g2['by_cohort'].values())
                        and all(v == 0 for v in g2['arm_missing_rows'].values())
                        and all(rec['input_sha256_matches'] and rec['rows_match_input']
                                for recs in records.values() for rec in recs.values()))
    gates = {'D-G2': g2}
    if BASE in arms:
        gates['D-G1'] = gate_d1(variable, keys, medians)
    return f, gates, records


def gate_d1(variable, keys, medians):
    """chronos_bolt_base_uni's median at positions 1-48 against the published zero-shot traces."""
    inp = np.load(ADAPT / 'inputs' / f'continuation_{variable}.npz', allow_pickle=True)
    pub = pd.DataFrame({'cohort': inp['cohort'].astype(str), 'fold': inp['fold'].astype(int), 'sensor': inp['sensor'].astype(str),
                        't': ns(pd.to_datetime(inp['onset'], utc=True))})
    got = np.load(ADAPT / f'continuation_{variable}_zs.npz')
    traces = np.full((len(pub), got['traces'].shape[1]), np.nan)
    traces[got['rows']] = got['traces']
    k = keys[['cohort', 'fold', 'sensor', 't', 'row']].copy()
    k['sensor'] = k.sensor.astype(str)
    k['cohort'] = k.cohort.astype(str)
    j = pub.reset_index().merge(k, on=['cohort', 'fold', 'sensor', 't'], how='left', validate='one_to_one')
    new = np.full_like(traces, np.nan)
    for cohort in D_COHORTS:
        m = (j.cohort == cohort).to_numpy() & j.row.notna().to_numpy()
        new[j['index'].to_numpy()[m]] = medians[(BASE, cohort)][j.row.to_numpy()[m].astype(int), :traces.shape[1]]
    have = np.zeros(len(pub), bool)
    have[got['rows']] = True
    d = np.abs(new[have] - traces[have])
    out = {'compared': f'{BASE} median positions 1-{traces.shape[1]} vs runs/v2_chronos_adapt/continuation_{variable}_zs.npz',
           'episodes_published': int(have.sum()), 'episodes_without_new_row': int(j.row.isna().sum()),
           'max_abs_difference': float(np.nanmax(d)), 'values_above_tolerance': int((d > G1D_TOL).sum()),
           'non_finite': int((~np.isfinite(d)).sum()), 'tolerance': G1D_TOL}
    out['passed'] = bool(out['episodes_without_new_row'] == 0 and out['values_above_tolerance'] == 0 and out['non_finite'] == 0)
    return out


def d_contrast(sub, x, y, label, variable):
    f = sub.dropna(subset=[x, y])
    if not len(f):
        return None
    units = int(f[['fold', 'week']].drop_duplicates().shape[0])
    point = v2_bridge.macro(f, x) - v2_bridge.macro(f, y)
    c = {'point': round(point, 6), 'paired_episodes': int(len(f)), 'units_fold_week': units,
         'cells': int(f.groupby(['fold', 'sensor']).ngroups)}
    if units < MIN_UNITS:
        c.update(ci=None, excludes_zero=None, not_estimable=f'{units} onset weeks, fewer than {MIN_UNITS}')
    else:
        draws, redrawn = v2_bridge.joint_week_draws(f, x, y, np.random.default_rng([20260923, zlib.crc32(label.encode())]))
        lo, hi = np.percentile(draws, [2.5, 97.5])
        c.update(ci=[round(float(lo), 6), round(float(hi), 6)], excludes_zero=bool(lo > 0 or hi < 0), joint_redrawn=int(redrawn))
        sp = v2_bridge.sparse_cells(f, x, y, label)
        if sp['sparse_cells']:
            c['sparse_cells'] = sp['sparse_cells']
            c['without_sparse_cells'] = sp['without_sparse_cells']
    return decorate(c, variable, x, y)


def gate_dr():
    """D-R: the contrast path reproduces published intervals (addendum 41 at 6 and 48 h, addendum 63 at 72 h)."""
    out, ok = {}, True
    a41 = json.loads((ADAPT / 'contrasts.json').read_text())['continuation']
    b72 = json.loads((RUNS / 'v2_bridge72' / 'report.json').read_text())
    for variable in VARIABLES:
        fr = pd.read_csv(RUNS / 'v2_bridge' / f'bridge_{variable}.csv.gz')
        fr['onset'] = pd.to_datetime(fr.onset, utc=True)
        iso = fr.onset.dt.isocalendar()
        fr['week'] = iso.year.astype(str) + '-' + iso.week.astype(str)
        inp = np.load(ADAPT / 'inputs' / f'continuation_{variable}.npz', allow_pickle=True)
        key = pd.DataFrame({'cohort': inp['cohort'], 'fold': inp['fold'], 'sensor': inp['sensor'],
                            'onset': pd.to_datetime(inp['onset'], utc=True)})
        key['episode'] = np.arange(len(key))
        fr = fr.merge(key, on=['cohort', 'fold', 'sensor', 'onset'], how='left', validate='many_to_one')
        got = np.load(ADAPT / f'continuation_{variable}_zs.npz')
        trace = np.full((len(key), got['traces'].shape[1]), np.nan)
        trace[got['rows']] = got['traces']
        fr['chronos_zs'] = trace[fr.episode.to_numpy(), fr.lead.to_numpy() - 1]
        ref = 'anchored_direct'
        for cohort in D_COHORTS:
            for lead in (6, 48):
                sub = fr[(fr.cohort == cohort) & (fr.lead == lead)]
                mine = d_contrast(sub, 'chronos_zs', ref, f'a41|{variable}|{cohort}|{lead}|chronos_zs|{ref}', variable)
                pub = a41[variable][cohort]['by_lead'][str(lead)]['contrasts'][f'chronos_zs - {ref}']
                same = mine['point'] == pub['point'] and mine['ci'] == pub['ci']
                ok &= same
                out[f'a41|{variable}|{cohort}|{lead}|chronos_zs - {ref}'] = {'mine': [mine['point'], mine['ci']],
                                                                              'published': [pub['point'], pub['ci']], 'same': same}
        f72 = v2_bridge72.with_weeks(pd.read_csv(RUNS / 'v2_bridge72' / f'bridge72_{variable}.csv.gz', parse_dates=['onset']))
        for cohort in D_COHORTS:
            sub = f72[(f72.cohort == cohort) & (f72.lead == 72)]
            mine = d_contrast(sub, 'anchored_direct', 'persistence', f'bridge72|{variable}|{cohort}|72|anchored_direct|persistence',
                              variable)
            pub = b72[variable][cohort]['by_lead']['72']['contrasts']['anchored_direct - persistence']
            same = mine['point'] == pub['point'] and mine['ci'] == pub['ci']
            ok &= same
            out[f'bridge72|{variable}|{cohort}|72|anchored_direct - persistence'] = {
                'mine': [mine['point'], mine['ci']], 'published': [pub['point'], pub['ci']], 'same': same}
    out['passed'] = bool(ok)
    return out


def d_families(variable, arms):
    refs = [x for x in D_REFS[variable]] + ([BASE] if BASE in arms else [])
    return {'K1': [(a, ref) for a in arms for ref in refs if a != ref],
            'K2': [(a, BASE) for a in arms if a != BASE and BASE in arms],
            'K3': [(x, y) for x, y in K3 if x in arms and y in arms],
            'K4': [(x, y) for x, y in K4 if x in arms and y in arms]}


def d_score(variable, frame, arms):
    fam = d_families(variable, arms)
    pairs = list(dict.fromkeys(p for v in fam.values() for p in v))
    out = {}
    for cohort in D_COHORTS:
        fc = frame[frame.cohort == cohort]
        block = {'episodes': int(fc.groupby(['fold', 'sensor', 'onset']).ngroups), 'by_lead': {}}
        for lead in D_LEADS:
            sub = fc[fc.lead == lead]
            e = {'n': int(len(sub)), 'source': sorted(sub.source.unique()), 'mae': {}, 'contrasts': {}}
            for a in list(arms) + [x for x in D_PUBLISHED if x in sub]:
                n = int(sub[a].notna().sum())
                if n:
                    e['mae'][a] = {'rows': n, 'mae': round(v2_bridge.macro(sub, a), 6)}
                    if variable == 'oxygen':
                        e['mae'][a]['mae_mgl'] = mg(variable, e['mae'][a]['mae'])
            for x, y in pairs:
                c = d_contrast(sub, x, y, f'fmzoo|D|{variable}|{cohort}|{lead}|{x}|{y}', variable)
                if c is not None:
                    e['contrasts'][f'{x} - {y}'] = c
            e['families'] = {k: [f'{x} - {y}' for x, y in v] for k, v in fam.items()}
            block['by_lead'][str(lead)] = e
        out[cohort] = block
        print(f'  D {variable} {cohort} done', flush=True)
    return out


def task_d(arms, man):
    rep = {'leads': D_LEADS, 'draws': v2_bridge.DRAWS, 'aggregation': 'frozen (fold, sensor) macro',
           'references': {v: list(D_REFS[v]) + [BASE] for v in VARIABLES}, 'main_reference': D_MAIN,
           'gates': {'D-R': gate_dr()}, 'prediction_files': {}, 'scores': {}}
    print('D-R', rep['gates']['D-R']['passed'], flush=True)
    for variable in VARIABLES:
        frame, gates, records = d_frame(variable, arms, man)
        rep['gates'][variable] = gates
        rep['prediction_files'][variable] = records
        print(f'D {variable}: gates', {k: v.get('passed') for k, v in gates.items()}, flush=True)
        if not gates['D-G2']['passed'] or not gates.get('D-G1', {'passed': True})['passed'] or not rep['gates']['D-R']['passed']:
            rep['scores'][variable] = {'not_scored': 'a gate failed'}
            continue
        if GATES_ONLY:
            continue
        rep['scores'][variable] = d_score(variable, frame, arms)
    return rep


# ==================================================================================================
# Summary for the console
# ==================================================================================================
def summary(rep):
    lines = []
    C = rep.get('C', {}).get('scores', {})
    for v, blk in C.items():
        if not isinstance(blk, dict) or 'not_scored' in blk:
            continue
        unit = ' (mg/l)' if v == 'oxygen' else ''
        for cohort, leads in blk.items():
            for lead in ('6', '24', '72'):
                cell = leads.get(lead)
                if not cell:
                    continue
                row = []
                for a, e in cell['arms'].items():
                    if e is None:
                        continue
                    qs = e['mgl']['quantile_score9'] if v == 'oxygen' else e['quantile_score9']
                    k1 = cell['contrasts'].get(f'{a} - clim_quantiles')
                    pt = (k1.get('point_mgl') if v == 'oxygen' else k1['point']) if k1 else None
                    star = '*' if k1 and k1.get('excludes_zero') else ''
                    row.append(f'{a}={qs:.4f}[{pt:+.4f}{star}]' if pt is not None else f'{a}={qs:.4f}')
                for a in C_REFS:
                    e = cell['published_arms'].get(a)
                    if e:
                        qs = e['mgl']['quantile_score9'] if v == 'oxygen' else e['quantile_score9']
                        row.append(f'{a}={qs:.4f}')
                lines.append(f'C {v}{unit} {cohort} {lead}h: ' + '  '.join(row))
    D = rep.get('D', {}).get('scores', {})
    for v, blk in D.items():
        if not isinstance(blk, dict) or 'not_scored' in blk:
            continue
        main = D_MAIN[v]
        for cohort, cb in blk.items():
            for lead in ('6', '24', '48', '72'):
                cell = cb['by_lead'].get(lead)
                if not cell:
                    continue
                row = []
                for a, e in cell['mae'].items():
                    m = e['mae_mgl'] if v == 'oxygen' else e['mae']
                    k1 = cell['contrasts'].get(f'{a} - {main}')
                    pt = (k1.get('point_mgl') if v == 'oxygen' else k1['point']) if k1 else None
                    star = '*' if k1 and k1.get('excludes_zero') else ''
                    row.append(f'{a}={m:.4f}[{pt:+.4f}{star}]' if pt is not None else f'{a}={m:.4f}')
                lines.append(f'D {v} {cohort} {lead}h: ' + '  '.join(row))
    return lines


GATES_ONLY = False


def main():
    global GATES_ONLY
    ap = argparse.ArgumentParser()
    ap.add_argument('--arms', default=','.join(ARMS))
    ap.add_argument('--tasks', default='C,D')
    ap.add_argument('--out', default=None)
    ap.add_argument('--gates-only', action='store_true')
    a = ap.parse_args()
    GATES_ONLY = a.gates_only
    arms = tuple(x for x in a.arms.split(',') if x)
    unknown = [x for x in arms if x not in ARMS]
    if unknown:
        raise SystemExit(f'unknown arms {unknown}')
    state = preds_state()
    complete = state['ready'] and set(arms) == set(ARMS) and set(a.tasks.split(',')) == {'C', 'D'} and not GATES_ONLY
    out = Path(a.out) if a.out else OUT
    if out.resolve() == OUT.resolve() and not complete:
        raise SystemExit(f'not writing {rel(OUT)}: preds/ is {state} and arms {arms}, tasks {a.tasks}, '
                         f'gates only {GATES_ONLY}; give --out for a partial reading')
    man = manifest()
    rep = {'declared_in': DECLARED_IN, 'amendments': AMENDMENTS, 'producer': 'scripts/v2_fm_zoo_score_dist_bridge.py',
           'created_utc': datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ'), 'complete': complete,
           'preds': state, 'arms': list(arms), 'oxygen_mg_per_l_per_stored_unit': C_FACTOR, 'floors': floors(),
           'min_units': MIN_UNITS,
           'contrasts': {'K1': 'arm - each reference', 'K2': f'arm - {BASE}', 'K3': 'cov - uni',
                         'K4': 'timesfm3_uni - timesfm25_uni, chronos2_uni - chronos_bolt_base_uni'}}
    t0 = time.time()
    if 'C' in a.tasks.split(','):
        rep['C'] = task_c(arms, man)
    if 'D' in a.tasks.split(','):
        rep['D'] = task_d(arms, man)
    rep['seconds'] = round(time.time() - t0, 1)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes((json.dumps(rep, indent=1, default=jsonable) + '\n').encode('utf-8'))
    print('wrote', out)
    for line in summary(rep):
        print(line)


if __name__ == '__main__':
    main()
