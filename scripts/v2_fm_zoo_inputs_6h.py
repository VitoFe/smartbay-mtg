"""Addendum 81, the six-hour tasks: build locally, from the published code paths, the inputs of
tasks A (Dev-2025, six issuance phases), B (Reserved-2026 battery) and F (oxygen as apparent
utilisation, Reserved-2026), in the common format of scripts/v2_fm_zoo_io.py.

Task A, one file per variable and issuance offset (A_<variable>_offset<o>.npz). The rows and contexts
of addendum 51's panel (runs/v2_sequence_sweep/panel/panel_<variable>_offset<o>.csv.gz):
v2_sequence_sweep.setup (offset 0 is v2_horizon.build, offsets 1-5 v2_phase_shift.build_shifted),
then per outer fold v2_sequence_sweep.rows_of(..., 'va') and v2_sequence_arms.gather, i.e.
context_window (512 h ending at issuance, at least 48 finite values, interior and edge holes
interpolated with limit_direction='both') left-padded with its first value to 512. The contexts are
stored exactly as the published arms read them, standardised as v2_sequence_arms.hourly_panel does
((x - center) / scale; temperature (20, 10), oxygen (230, 50)); the panel's prediction columns
(chronos_zs, timesfm_zs, chronos2_zs) are in the same standardised units. Physical = y * scale + center.

Task B, one file per variable (B_<variable>_Reserved-2026.npz). The battery's endpoints
(runs/battery/results/chronos/chronos_predictions.csv.gz, keys variable, sensor, time, in its row
order, which is exp_chronos.ep's), contexts as exp_chronos.predict_at: the raw series up to and
including the issuance hour, its last 512 values, NaN kept, admissible with at least 48 finite
values. Gated against runs/v2_context_holes/inputs/A2_<variable>.npz ('raw', 'length').

Task F, one file (F_oxygen_aou_Reserved-2026.npz). The battery's oxygen rows, contexts as
exp_chronos.series_for(..., 'aou') read through predict_at's window. The unit is that of the
utilisation arms the thesis prints, addendum 60 amendment 4 (`--aou-unit mgl`, the default): the
sonde oxygen x the constant c (v2_oxygen_mgl_chronos_remote.mgl_hourly(constant=True)) and the
saturation as v2_oxygen_mgl_chronos.o2_sol in mode 'mgl' (Garcia-Gordon per litre at S = 38, T
clipped to [-2, 40], x 0.0319988). `--aou-unit l` gives addendum 52's per-litre form
(v2_aou_units_chronos.o2_sol_l) and `--aou-unit kg` the battery's original exp_chronos.o2_sol; either
writes F_oxygen_aou_Reserved-2026_<unit>.npz. A side file F_..._reconstruction.npz carries, row by
row, what the inversion o2_sol(T_hat) - AOU_hat needs besides the arm's forecast: the row of
B_temperature_Reserved-2026.npz that holds the same arm's six-hour temperature forecast at the same
(sensor, issuance), and the salinity (the constant 38 of exp_chronos.SAL).

Covariates (cov mode, every file): the battery's Chronos-2 covariate set, v2_chronos2.covs, past only,
for the hours of each context: net_T and net_sd (the mean and the standard deviation across every
temperature column of the hourly table, exp_chronos.NET, which includes the probe itself) and, for
oxygen, the probe's own temperature (own_T), each forward then backward filled inside the window and
any remaining hole set to the mean of the last 720 h. Physical units (degC) in every task.

exp_chronos, v2_chronos2 and v2_aou_units_chronos are written to run from a snapshot directory on the
compute host (they read SMARTBAY_BASE/review_20260919/inputs/ and create a results folder beside
themselves at import), so they are imported here from byte-identical copies in a temporary snapshot
with the two input files they read; nothing published is edited.

Gates (written to reports/bay_twin/runs/v2_fm_zoo/inputs/manifest_6h.json; the builder exits non-zero
if any fails): A, the keys equal the panel's rows in order at every offset; at offset 0 the contexts
equal runs/v2_context_holes/inputs/A1_<variable>.npz['linear'] to 1e-6 (standardised units) and A1's
keys equal the panel's; the standardisation reproduces the hourly table at the issuance hour. B, the
battery's row order equals exp_chronos.ep's; 3,451 and 799 rows; every context equals A2's raw window
exactly (values, holes, length). F, the rows, window lengths and covariates equal task B oxygen's;
each context equals o2_sol(T) - O2 recomputed from the table; every row finds its temperature row.
Every file is read back with load_inputs and compared with what was written.

Declared in config/bay_reserved_protocol_v2_addendum81.json.
Usage (repository root):  python -m scripts.v2_fm_zoo_inputs_6h [--tasks A B F] [--aou-unit mgl|l|kg]
"""
import argparse
import io
import json
import os
import shutil
import sys
import tempfile
import time
import warnings
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.v2_fm_zoo_io import digest, load_inputs, save_inputs  # noqa: E402

warnings.filterwarnings('ignore')
RUNS = ROOT / 'reports/bay_twin/runs'
OUT = RUNS / 'v2_fm_zoo' / 'inputs'
DECLARATION = ROOT / 'config/bay_reserved_protocol_v2_addendum81.json'
EXPERIMENTS = ROOT / 'scripts/experiments'
SNAPSHOT_CODE = ('exp_chronos.py', 'v2_chronos2.py', 'v2_aou_units_chronos.py')
HOURLY = ROOT / 'reports/bay_twin/hourly_observations.csv.gz'
ENDPOINTS = RUNS / '20260918_reserved_evaluation/forecast_endpoints_2026.csv.gz'
BATTERY = RUNS / 'battery/results/chronos/chronos_predictions.csv.gz'
BATTERY_CH2 = RUNS / 'battery/v2_chronos/v2_chronos2_predictions.csv.gz'
BATTERY_FORECAST = RUNS / 'battery/results/forecast/forecast_predictions.csv.gz'
PANEL = RUNS / 'v2_sequence_sweep/panel'
HOLES = RUNS / 'v2_context_holes/inputs'
AOU_L = RUNS / 'v2_aou_units_chronos/zeroshot_columns.csv.gz'
AOU_MGL = RUNS / 'v2_oxygen_mgl/C3/v2_aou_units_chronos/zeroshot_mgl_columns.csv.gz'
VARIABLES = ('temperature', 'oxygen')
OFFSETS = (0, 1, 2, 3, 4, 5)
KEY = ['variable', 'sensor', 'time']
HORIZON = 6
MIN_FINITE = 48
EXPECTED_B_ROWS = {'temperature': 3451, 'oxygen': 799}
#: v2_sequence_arms.hourly_panel, line 78
CENTER_SCALE = {'temperature': (20., 10.), 'oxygen': (230., 50.)}
#: per-litre oxygen to mg/l (v2_oxygen_mgl_chronos.MG_PER_UMOL_L, src.units.MG_PER_UMOL_L)
MG_PER_UMOL_L = 0.0319988
SALINITY = 38.0
AOU_UNITS = ('mgl', 'l', 'kg')
TOL_A1 = 1e-6
TOL_STANDARDISATION = 1e-4
#: the mg/l run read stored x c as CSV text (mgl_hourly); parsing that text back differs from the float
#: product by a unit in the last place (1.8e-15 mg/l), so the table is parsed from the text, as the run did
TOL_CSV_TEXT = 1e-12


def rel(path):
    return Path(path).resolve().relative_to(ROOT).as_posix()


def ns(values):
    """Times (tz-aware or ISO strings) as int64 ns UTC, the format the input files store."""
    t = pd.to_datetime(pd.Series(list(values) if not isinstance(values, pd.Series) else values), utc=True)
    return t.dt.tz_convert('UTC').dt.tz_localize(None).astype('datetime64[ns]').to_numpy().astype('int64')


def npz(path):
    """Every array of an npz file, read once (NpzFile decompresses on each access)."""
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


def write_json(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes((json.dumps(obj, indent=1, allow_nan=False) + '\n').encode('utf-8'))


# ---------------------------------------------------------------- the published host scripts

class Snapshot:
    """exp_chronos, v2_chronos2 and v2_aou_units_chronos imported from byte-identical copies in a
    temporary directory laid out as on the compute host (code beside base/review_20260919/inputs/)."""

    def __enter__(self):
        self.dir = Path(tempfile.mkdtemp(prefix='v2_fm_zoo_6h_'))
        inputs = self.dir / 'base/review_20260919/inputs'
        inputs.mkdir(parents=True)
        for name in SNAPSHOT_CODE:
            shutil.copyfile(EXPERIMENTS / name, self.dir / name)
            assert digest(self.dir / name) == digest(EXPERIMENTS / name)
        shutil.copyfile(HOURLY, inputs / 'hourly_observations.csv.gz')
        shutil.copyfile(ENDPOINTS, inputs / 'forecast_endpoints_2026.csv.gz')
        os.environ['SMARTBAY_BASE'] = str(self.dir / 'base')
        sys.path.insert(0, str(self.dir))
        import exp_chronos as xc
        import v2_chronos2 as c2m
        import v2_aou_units_chronos as au
        assert Path(xc.__file__).resolve().parent == self.dir.resolve(), xc.__file__
        self.xc, self.c2m, self.au = xc, c2m, au
        return self

    def __exit__(self, *exc):
        while str(self.dir) in sys.path:
            sys.path.remove(str(self.dir))
        shutil.rmtree(self.dir, ignore_errors=True)
        return False


def covariates(c2m, sensor, variable, t, n):
    """v2_chronos2.covs (lines 30-37) for the n hours ending at t: names and a [C, n] array."""
    c = c2m.covs(sensor, variable, t, n)
    names = list(c)
    arr = np.stack([np.asarray(c[k], float) for k in names])
    if arr.shape != (len(names), n):
        raise AssertionError(f'covariate window {arr.shape} for {sensor} {t}, context of {n} h')
    return names, arr


def battery_rows():
    """The battery's endpoints in its row order (variable, sensor, issuance time)."""
    rows = pd.read_csv(BATTERY, usecols=KEY)
    rows['time'] = pd.to_datetime(rows.time, utc=True)
    return rows


def published_window(series, t, ctx):
    """exp_chronos.predict_at, line 184: the series up to t, its last `ctx` values."""
    return series.loc[:t].to_numpy(float)[-ctx:]


def same_rows(a, b, columns):
    """Row-by-row equality of two key frames (times compared as ns)."""
    if len(a) != len(b):
        return False
    for col in columns:
        if col == 'time':
            if not (ns(a[col]) == ns(b[col])).all():
                return False
        elif not (a[col].to_numpy().astype(str) == b[col].to_numpy().astype(str)).all():
            return False
    return True


# ---------------------------------------------------------------- task A

def task_a(variable, offset, snap):
    from src.experiments import v2_sequence_arms as sa
    from src.experiments import v2_sequence_sweep as sw
    xc, c2m = snap.xc, snap.c2m
    data, z, q, outer, scale = sw.setup(variable, offset)          # v2_sequence_sweep.py:87-95
    center, scale_ = CENTER_SCALE[variable]
    assert scale == scale_, (scale, scale_)
    keys, contexts, past, names_seen, padded = [], [], [], set(), 0
    for fi, fold in enumerate(outer):                              # v2_sequence_sweep.run_panel, lines 266-267
        va_ctx, va_keep = sa.gather(data, z, q, sw.rows_of(data, fold, 'va'))   # v2_sequence_arms.py:93-119
        for k, (i, s) in enumerate(va_keep):
            t = pd.Timestamp(data['times'][i])
            sensor = data['names'][s]
            n_real = min(sa.CTX, int(q[i]) + 1)
            names, cov = covariates(c2m, sensor, variable, t, n_real)
            if n_real < sa.CTX:                                    # gather's left pad, mirrored
                padded += 1
                cov = np.concatenate([np.repeat(cov[:, :1], sa.CTX - n_real, axis=1), cov], axis=1)
            names_seen.add(tuple(names))
            keys.append((fi, sensor, t))
            contexts.append(va_ctx[k])
            past.append(cov)
    assert len(names_seen) == 1, names_seen
    cov_names = list(names_seen.pop())
    frame = pd.DataFrame(keys, columns=['fold', 'sensor', 'time'])
    frame['fold'] = frame.fold.astype(np.int64)

    panel_path = PANEL / f'panel_{variable}_offset{offset}.csv.gz'
    panel = pd.read_csv(panel_path, usecols=['fold', 'sensor', 'time'])
    gates = {'rows': int(len(frame)), 'panel_rows': int(len(panel)),
             'rows_equal_panel': same_rows(frame, panel, ['fold', 'sensor', 'time']),
             'contexts_left_padded': padded}
    # the standardisation: the last context hour back in physical units against the hourly table
    last = np.array([c[-1] for c in contexts], float) * scale + center
    table = np.array([xc.obs[f'{s}::{variable}'].get(t, np.nan) for _, s, t in keys], float)
    both = np.isfinite(table)
    worst = float(np.abs(last[both] - table[both]).max()) if both.any() else None
    gates['standardisation_check'] = {'compared': int(both.sum()), 'max_abs_diff_physical': worst,
                                      'passed': bool(worst is not None and worst <= TOL_STANDARDISATION)}
    if offset == 0:
        a1 = npz(HOLES / f'A1_{variable}.npz')
        lin = a1['linear']
        a1_keys = pd.DataFrame({'fold': a1['key_fold'], 'sensor': a1['key_sensor'], 'time': a1['key_time']})
        same_keys = same_rows(frame, a1_keys, ['fold', 'sensor', 'time'])
        stacked = np.stack(contexts).astype(np.float32)
        diff = (float(np.abs(stacked - lin).max())
                if lin.shape == stacked.shape and np.isfinite(lin).all() else None)
        gates['A1_linear'] = {'file': rel(HOLES / f'A1_{variable}.npz'), 'keys_equal': same_keys,
                              'scale_equal': bool(float(a1['scale']) == scale), 'shape': list(lin.shape),
                              'max_abs_diff_standardised': diff,
                              'passed': bool(same_keys and diff is not None and diff <= TOL_A1
                                             and float(a1['scale']) == scale)}
    gates['passed'] = bool(gates['rows_equal_panel'] and gates['standardisation_check']['passed'] and
                           (offset != 0 or gates['A1_linear']['passed']))

    path = OUT / f'A_{variable}_offset{offset}.npz'
    sources = [panel_path, ROOT / 'src/experiments/v2_sequence_arms.py', ROOT / 'src/experiments/v2_sequence_sweep.py',
               ROOT / 'src/experiments/v2_horizon.py', ROOT / 'src/experiments/v2_phase_shift.py',
               EXPERIMENTS / 'v2_chronos2.py', EXPERIMENTS / 'exp_chronos.py', HOURLY]
    if offset == 0:
        sources.append(HOLES / f'A1_{variable}.npz')
    standardisation = {
        'center': center, 'scale': scale,
        'contexts': 'standardised, (x - center) / scale, float32, exactly as the published panel arms read them '
                    '(v2_sequence_arms.hourly_panel, line 78)',
        'to_physical': 'y * scale + center', 'past_covariates': 'physical units (degC), not standardised',
        'published_columns': 'panel chronos_zs, timesfm_zs, chronos2_zs, truth and persistence are standardised; '
                             'an error in physical units is the standardised error x scale'}
    hours = sorted(set(int(h) for h in pd.DatetimeIndex(data['times']).hour))
    note = (f'Dev-2025 six-hour task at issuance offset {offset} h (issuance hours {hours} UTC), the rows and contexts '
            f'of {rel(panel_path)}. Contexts: v2_sequence_sweep.setup (lines 87-95) and v2_sequence_arms.gather and '
            'context_window (lines 93-119): 512 h ending at issuance, holes interpolated with limit_direction=both, '
            'left-padded with the first value. Covariates: v2_chronos2.covs (lines 30-37) on the hourly table '
            '(exp_chronos.obs and NET, lines 65-79), the same hours as the context, past only.')
    meta = save_inputs(path, task='A_dev_6h', variable=variable, cohort='Dev-2025', keys=frame,
                       contexts=contexts, horizon=HORIZON, units=units_of(variable, standardised=True),
                       past_cov=past, future_cov=None, cov_names=cov_names, standardisation=standardisation,
                       sources=sources, note=note)
    gates['roundtrip'] = roundtrip(path, frame, contexts, past)
    gates['passed'] = bool(gates['passed'] and gates['roundtrip']['passed'])
    refs = {'file': rel(panel_path),
            'columns': {'chronos_bolt_base_uni': 'chronos_zs', 'timesfm25_uni': 'timesfm_zs',
                        'chronos2_uni': 'chronos2_zs'},
            'units': 'standardised (x scale for physical)',
            'how': {'chronos_zs': 'v2_sequence_arms.chronos_predict (lines 130-143): amazon/chronos-bolt-base, '
                                  'predict_quantiles levels [0.1, 0.5, 0.9], prediction_length 6, median at step 6, '
                                  'batch 256',
                    'timesfm_zs': 'v2_sequence_arms.timesfm_predict (lines 146-157): TimesFM_2p5_200M_torch, '
                                  'ForecastConfig(max_context=512, max_horizon=6, normalize_inputs=True), point '
                                  'forecast at step 6, batch 256',
                    'chronos2_zs': 'v2_sequence_sweep.chronos2_predict (lines 215-232): amazon/chronos-2, predict '
                                   '[{target}], prediction_length 6, the 0.5 quantile at step 6, batch 128'}}
    return path, meta, gates, refs, {'offset': offset, 'issuance_hours_utc': hours}


# ---------------------------------------------------------------- task B

def task_b(variable, snap, rows):
    xc, c2m = snap.xc, snap.c2m
    part = rows[rows.variable == variable].reset_index(drop=True)
    keys, contexts, past, names_seen, short = [], [], [], set(), 0
    for r in part.itertuples(index=False):
        ser = xc.series_for(r.sensor, variable, 'raw')
        hist = published_window(ser, r.time, xc.CTX)
        if np.isfinite(hist).sum() < MIN_FINITE:
            short += 1
            continue
        names, cov = covariates(c2m, r.sensor, variable, r.time, len(hist))   # v2_chronos2.forecast, line 50
        names_seen.add(tuple(names))
        keys.append((variable, r.sensor, r.time))
        contexts.append(hist.astype(np.float32))
        past.append(cov)
    assert len(names_seen) == 1, names_seen
    cov_names = list(names_seen.pop())
    frame = pd.DataFrame(keys, columns=KEY)

    gates = {'rows': int(len(frame)), 'battery_rows': int(len(part)), 'inadmissible_rows': short,
             'rows_equal_battery': bool(short == 0 and same_rows(frame, part, KEY)),
             'expected_rows': EXPECTED_B_ROWS[variable]}
    gates['count_as_declared'] = bool(len(frame) == gates['expected_rows'])
    a2 = npz(HOLES / f'A2_{variable}.npz')
    index = {(s, int(t)): i for i, (s, t) in enumerate(zip(a2['key_sensor'], ns(a2['key_time'])))}
    worst, same_nan, same_len, missing = 0.0, True, True, 0
    for s, t, c in zip(frame.sensor, ns(frame.time), contexts):
        j = index.get((s, int(t)))
        if j is None:
            missing += 1
            continue
        n = int(a2['length'][j])
        ref = a2['raw'][j, a2['raw'].shape[1] - n:]
        if n != len(c):
            same_len = False
            continue
        same_nan &= bool((np.isfinite(ref) == np.isfinite(c)).all())
        fin = np.isfinite(ref) & np.isfinite(c)
        if fin.any():
            worst = max(worst, float(np.abs(ref[fin] - c[fin]).max()))
    gates['A2_raw'] = {'file': rel(HOLES / f'A2_{variable}.npz'), 'a2_rows': int(len(a2['length'])),
                       'keys_missing_in_a2': missing, 'same_lengths': same_len, 'same_holes': same_nan,
                       'max_abs_diff': worst,
                       'passed': bool(missing == 0 and len(a2['length']) == len(frame) and same_len and same_nan
                                      and worst == 0.0)}
    gates['passed'] = bool(gates['rows_equal_battery'] and gates['count_as_declared'] and gates['A2_raw']['passed'])

    path = OUT / f'B_{variable}_Reserved-2026.npz'
    note = (f'Reserved-2026 six-hour battery, the rows of {rel(BATTERY)} in its order (exp_chronos.ep, has_anchor and '
            'has_target). Contexts: exp_chronos.predict_at (lines 176-195), the raw hourly series up to and including '
            'the issuance hour, its last 512 values, NaN kept (Chronos-Bolt and Chronos-2 read the holes as missing), '
            'admissible with at least 48 finite values. Covariates: v2_chronos2.covs (lines 30-37), past only.')
    meta = save_inputs(path, task='B_reserved_6h', variable=variable, cohort='Reserved-2026', keys=frame,
                       contexts=contexts, horizon=HORIZON, units=units_of(variable), past_cov=past, future_cov=None,
                       cov_names=cov_names, standardisation=None,
                       sources=[BATTERY, HOURLY, ENDPOINTS, EXPERIMENTS / 'exp_chronos.py',
                                EXPERIMENTS / 'v2_chronos2.py', HOLES / f'A2_{variable}.npz'],
                       note=note)
    gates['roundtrip'] = roundtrip(path, frame, contexts, past)
    gates['passed'] = bool(gates['passed'] and gates['roundtrip']['passed'])
    refs = {'chronos_bolt_base_uni': {'file': rel(BATTERY), 'column': f'{variable}|ZS_raw',
                                      'how': 'exp_chronos.predict_at: amazon/chronos-bolt-base, predict_quantiles '
                                             'levels [0.1, 0.5, 0.9], prediction_length 6, median at step 6, batch 256'},
            'chronos2_uni': {'file': rel(BATTERY_CH2), 'column': f'CH2_{variable}_raw_uni',
                             'how': 'v2_chronos2.forecast (lines 40-57): amazon/chronos-2, predict, prediction_length '
                                    '6, the 0.5 quantile at step 6, batch 128'},
            'chronos2_cov': {'file': rel(BATTERY_CH2), 'column': f'CH2_{variable}_raw_cov',
                             'how': 'as chronos2_uni with past_covariates {name: float32 array} in cov_names order'},
            'kept_in_the_record': {'file': rel(BATTERY_FORECAST),
                                   'columns': ['CHRONOS_bolt_small', 'CHRONOS_bolt_base'],
                                   'note': 'producer not in the repository; their contexts differ from these'}}
    return path, meta, gates, refs, {}


# ---------------------------------------------------------------- task F

def o2_sol_mgl(T, S=SALINITY):
    """v2_oxygen_mgl_chronos.o2_sol in mode 'mgl' (lines 55-58)."""
    from src.data.thermodynamics import o2sol_garcia_gordon_umol_per_l
    T = np.clip(np.asarray(T, float), -2, 40)
    return o2sol_garcia_gordon_umol_per_l(T, S) * MG_PER_UMOL_L


def mgl_table():
    """The hourly table the mg/l run read (v2_oxygen_mgl_chronos_remote.mgl_hourly(constant=True), lines
    79-87), parsed as exp_chronos parses its own (lines 65-67)."""
    from scripts.v2_oxygen_mgl_chronos_remote import mgl_hourly
    obs = pd.read_csv(io.BytesIO(mgl_hourly(constant=True)), index_col=0)
    obs.index = pd.to_datetime(obs.index, utc=True)
    return obs.sort_index().reindex(pd.date_range(obs.index.min(), obs.index.max(), freq='h'))


def bind_unit(snap, unit):
    """Rebind exp_chronos's table and saturation for one unit, as v2_aou_units_chronos.set_unit and the
    mg/l snapshot do; returns the table and the saturation now bound, and a function restoring both."""
    xc, au = snap.xc, snap.au
    saved = (xc.obs, xc.o2_sol)
    if unit == 'kg':
        xc.o2_sol = au.O2_SOL_KG
    elif unit == 'l':
        au.set_unit('l')
    else:
        xc.obs, xc.o2_sol = mgl_table(), o2_sol_mgl

    def restore():
        xc.obs, xc.o2_sol = saved
    return xc.obs, xc.o2_sol, restore


def task_f(snap, rows, unit, b_oxygen, b_temperature_keys):
    from scripts.v2_oxygen_mgl_common import constant_c
    xc, c2m = snap.xc, snap.c2m
    part = rows[rows.variable == 'oxygen'].reset_index(drop=True)
    table0 = xc.obs
    gates = {}
    table, sol, restore = bind_unit(snap, unit)
    try:
        if unit == 'mgl':
            table = table[table0.columns]
            oxy = [c for c in table0.columns if c.endswith('::oxygen')]
            other = [c for c in table0.columns if c not in oxy]
            d_other = float(np.nanmax(np.abs(table[other].to_numpy() - table0[other].to_numpy())))
            d_oxy = float(np.nanmax(np.abs(table[oxy].to_numpy() - table0[oxy].to_numpy() * constant_c())))
            same = bool(len(table) == len(table0) and (table.index == table0.index).all() and
                        (np.isfinite(table.to_numpy()) == np.isfinite(table0.to_numpy())).all())
            gates['mgl_table'] = {'other_columns_max_abs_diff': d_other,
                                  'oxygen_minus_stored_x_c_max_abs_diff': d_oxy, 'c': constant_c(),
                                  'same_index_and_holes': same,
                                  'tolerance': TOL_CSV_TEXT,
                                  'passed': bool(d_other == 0.0 and d_oxy <= TOL_CSV_TEXT and same)}
        keys, contexts, short, worst, holes_ok = [], [], 0, 0.0, True
        for r in part.itertuples(index=False):
            ser = xc.series_for(r.sensor, 'oxygen', 'aou')                 # exp_chronos.py:82-93
            hist = published_window(ser, r.time, xc.CTX)
            if np.isfinite(hist).sum() < MIN_FINITE:
                short += 1
                continue
            T = published_window(table[f'{r.sensor}::temperature'], r.time, xc.CTX)
            O = published_window(table[f'{r.sensor}::oxygen'], r.time, xc.CTX)
            expect = sol(T) - O
            holes_ok &= bool((np.isfinite(expect) == np.isfinite(hist)).all())
            fin = np.isfinite(expect)
            if fin.any():
                worst = max(worst, float(np.abs(expect[fin] - hist[fin]).max()))
            keys.append(('oxygen', r.sensor, r.time))
            contexts.append(hist.astype(np.float32))
    finally:
        restore()
    # the covariates are temperatures, equal in every unit's table (gate mgl_table)
    past, names_seen = [], set()
    for (_, s, t), c in zip(keys, contexts):
        names, cov = covariates(c2m, s, 'oxygen', t, len(c))
        names_seen.add(tuple(names))
        past.append(cov)
    assert len(names_seen) == 1, names_seen
    cov_names = list(names_seen.pop())
    frame = pd.DataFrame(keys, columns=KEY)

    bk, bctx, bpast = b_oxygen
    n_ok = len(frame) == len(bk)
    gates.update({
        'rows': int(len(frame)), 'inadmissible_rows': short,
        'rows_equal_task_B_oxygen': bool(short == 0 and same_rows(frame, bk, ['sensor', 'time'])),
        'window_lengths_equal_task_B_oxygen': bool(n_ok and all(len(a) == len(b) for a, b in zip(contexts, bctx))),
        'covariates_equal_task_B_oxygen': bool(n_ok and all(
            np.array_equal(np.asarray(a, np.float32), np.asarray(b, np.float32), equal_nan=True)
            for a, b in zip(past, bpast))),
        'recomputed_from_table': {'max_abs_diff': worst, 'same_holes': holes_ok,
                                  'passed': bool(worst == 0.0 and holes_ok)}})
    where = {(s, int(t)): i for i, (s, t) in enumerate(zip(b_temperature_keys.sensor, ns(b_temperature_keys.time)))}
    temperature_row = np.array([where.get((s, int(t)), -1) for s, t in zip(frame.sensor, ns(frame.time))], np.int64)
    gates['temperature_rows_found'] = int((temperature_row >= 0).sum())
    gates['passed'] = bool(gates['rows_equal_task_B_oxygen'] and gates['window_lengths_equal_task_B_oxygen'] and
                           gates['covariates_equal_task_B_oxygen'] and gates['recomputed_from_table']['passed'] and
                           gates['temperature_rows_found'] == len(frame) and
                           gates.get('mgl_table', {'passed': True})['passed'])

    suffix = '' if unit == 'mgl' else f'_{unit}'
    path = OUT / f'F_oxygen_aou_Reserved-2026{suffix}.npz'
    solubility = {
        'mgl': 'o2sol_garcia_gordon_umol_per_l(clip(T, -2, 40), S) x 0.0319988 (src/data/thermodynamics.py; '
               'v2_oxygen_mgl_chronos.o2_sol in mode mgl, lines 55-58)',
        'l': 'o2sol_garcia_gordon_umol_per_l(clip(T, -2, 40), S) (src/data/thermodynamics.py; '
             'v2_aou_units_chronos.o2_sol_l, lines 47-50)',
        'kg': 'exp_chronos.o2_sol (lines 55-61): Garcia-Gordon per kilogram at S, T clipped to [-2, 40]'}[unit]
    context_unit = {'mgl': 'mg/l: saturation per litre x 0.0319988 minus the sonde oxygen x c = %.10g '
                           '(addendum 60 amendment 4)' % constant_c(),
                    'l': 'umol/l saturation minus the stored sonde column (addendum 52, unit l)',
                    'kg': 'umol/kg saturation minus the stored sonde column (the battery)'}[unit]
    result_unit = {'mgl': 'mg/l', 'l': 'umol/kg, the stored sonde column', 'kg': 'umol/kg, the stored sonde column'}[unit]
    reference = {'mgl': (AOU_MGL, {'chronos_bolt_base_uni': 'oxygen|ZS_aou|mgl',
                                   'chronos2_uni': 'CH2_oxygen_aou_uni|mgl', 'chronos2_cov': 'CH2_oxygen_aou_cov|mgl'}),
                 'l': (AOU_L, {'chronos_bolt_base_uni': 'oxygen|ZS_aou|l', 'chronos2_uni': 'CH2_oxygen_aou_uni|l',
                               'chronos2_cov': 'CH2_oxygen_aou_cov|l'}),
                 'kg': (BATTERY, {'chronos_bolt_base_uni': 'oxygen|ZS_aou'})}[unit]
    reconstruction = {
        'formula': 'oxygen_hat = o2_sol(T_hat, S) - aou_hat (v2_aou_units_chronos.column, lines 67-81)',
        'o2_sol': solubility, 'salinity': SALINITY,
        'salinity_source': 'exp_chronos.SAL (line 49), a constant: there is no hourly salinity',
        'T_hat': "the same arm's six-hour temperature median, row temperature_row of "
                 "B_temperature_Reserved-2026.npz (same sensor and issuance). The published columns invert "
                 "Chronos-Bolt at Chronos-Bolt base's temperature forecast and each Chronos-2 column at Chronos-2's "
                 "of the same mode (v2_aou_units_chronos.zeroshot, lines 120-121 and 163-170)",
        'aou_unit': unit, 'context_unit': context_unit, 'result_unit': result_unit,
        'published_reference': {'file': rel(reference[0]), 'columns': reference[1]}}
    note = ('Reserved-2026 oxygen as apparent utilisation, the battery\'s oxygen rows. Contexts: '
            'exp_chronos.series_for(sensor, oxygen, aou) (lines 82-93) through predict_at\'s window (lines 183-186), '
            f'saturation {solubility}; unit {context_unit}. Covariates: v2_chronos2.covs (lines 30-37), as task B '
            f'oxygen. The inversion is described in {path.stem}_reconstruction.npz.')
    sources = [BATTERY, HOURLY, ENDPOINTS, EXPERIMENTS / 'exp_chronos.py', EXPERIMENTS / 'v2_chronos2.py',
               EXPERIMENTS / 'v2_aou_units_chronos.py', EXPERIMENTS / 'v2_oxygen_mgl_chronos.py',
               ROOT / 'scripts/v2_oxygen_mgl_chronos_remote.py', ROOT / 'src/data/thermodynamics.py',
               RUNS / 'v2_oxygen_units.json', reference[0]]
    meta = save_inputs(path, task='F_aou', variable='oxygen', cohort='Reserved-2026', keys=frame,
                       contexts=contexts, horizon=HORIZON, units=context_unit, past_cov=past, future_cov=None,
                       cov_names=cov_names, standardisation=None, sources=sources, note=note)
    side = path.with_name(path.stem + '_reconstruction.npz')
    np.savez_compressed(side, meta=np.array(json.dumps(reconstruction)), temperature_row=temperature_row,
                        salinity=np.full(len(frame), SALINITY), key_variable=frame.variable.to_numpy().astype('U'),
                        key_sensor=frame.sensor.to_numpy().astype('U'), key_time=ns(frame.time))
    gates['roundtrip'] = roundtrip(path, frame, contexts, past)
    gates['passed'] = bool(gates['passed'] and gates['roundtrip']['passed'])
    refs = {'file': rel(reference[0]), 'columns': reference[1], 'reconstruction_file': rel(side),
            'reconstruction_sha256': digest(side), 'reconstruction': reconstruction}
    return path, meta, gates, refs, {'aou_unit': unit}


# ---------------------------------------------------------------- shared

def units_of(variable, standardised=False):
    base = {'temperature': 'degC',
            'oxygen': 'umol/kg, the stored sonde column (the thesis prints it x c, src/units.py factor())'}[variable]
    return f'standardised; {base} before standardisation' if standardised else base


def roundtrip(path, frame, contexts, past):
    """Read the file back with load_inputs and compare with what was written."""
    z = load_inputs(path)
    ok_ctx = len(z['contexts']) == len(contexts) and all(
        np.array_equal(a, np.asarray(b, np.float32), equal_nan=True) for a, b in zip(z['contexts'], contexts))
    ok_cov = past is None or (z['past_cov'] is not None and all(
        np.array_equal(a, np.asarray(b, np.float32), equal_nan=True) for a, b in zip(z['past_cov'], past)))
    ok_keys = list(z['keys'].columns) == list(frame.columns)
    for col in frame.columns:
        got = z['keys'][col].to_numpy()
        want = ns(frame[col]) if col == 'time' else frame[col].to_numpy().astype(got.dtype)
        ok_keys &= bool(len(got) == len(want) and (got == want).all())
    return {'contexts': bool(ok_ctx), 'past_covariates': bool(ok_cov), 'keys': bool(ok_keys),
            'passed': bool(ok_ctx and ok_cov and ok_keys)}


def describe(path, meta, gates, refs, extra):
    z = load_inputs(path)
    lengths = np.array([len(c) for c in z['contexts']])
    finite = np.array([np.isfinite(c).sum() for c in z['contexts']])
    pc = np.concatenate(z['past_cov'], axis=1) if z['past_cov'] is not None else np.zeros((0, 0))
    return {'task': meta['task'], 'variable': meta['variable'], 'cohort': meta['cohort'], **extra,
            'n': meta['n'], 'horizon': meta['horizon'], 'units': meta['units'],
            'standardisation': meta['standardisation'], 'cov_names': meta['cov_names'],
            'future_covariates': z['future_cov'] is not None, 'key_columns': meta['key_columns'],
            'context_length': {'min': int(lengths.min()), 'max': int(lengths.max())},
            'contexts_with_holes': int((finite < lengths).sum()), 'finite_values_min': int(finite.min()),
            'past_covariate_nan': int(np.isnan(pc).sum()),
            'sha256': digest(path), 'bytes': path.stat().st_size,
            'gate_references': refs, 'gates': gates}


DECISIONS = [
    'Task A contexts are stored standardised, exactly as the published panel arms read them, so that chronos_zs, '
    "timesfm_zs and chronos2_zs are reproducible from the file; center and scale are in each file's standardisation; "
    'the past covariates are in physical units.',
    "Covariates follow v2_chronos2.covs, the battery's Chronos-2 set: net_T and net_sd are taken across every "
    'temperature column of the hourly table, the probe itself included (exp_chronos.NET), not across the other '
    'probes only; that is what the CH2_*_cov columns read.',
    "Task A covariates come from the battery's hourly table (exp_chronos.obs), which equals load_observations over "
    '2024-2025 to 4e-14.',
    'Task F is built in the unit the thesis prints for the pretrained utilisation arms, mg/l under addendum 60 '
    'amendment 4 (--aou-unit mgl), with runs/v2_oxygen_mgl/C3/v2_aou_units_chronos/zeroshot_mgl_columns.csv.gz as '
    'its reference; --aou-unit l (addendum 52) and kg (the battery) are built and gated the same way on request.',
    'Task B oxygen stays in the stored sonde unit, as the battery read it; it is scored x c.']


def main():
    ap = argparse.ArgumentParser(description='Addendum 81: inputs of the six-hour tasks A, B and F.')
    ap.add_argument('--tasks', nargs='+', default=['A', 'B', 'F'], choices=['A', 'B', 'F'])
    ap.add_argument('--aou-unit', default='mgl', choices=AOU_UNITS)
    a = ap.parse_args()
    t0 = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    manifest_path = OUT / 'manifest_6h.json'
    manifest = json.loads(manifest_path.read_text(encoding='utf-8')) if manifest_path.exists() else {'files': {}}
    manifest.update({'declared_in': rel(DECLARATION), 'declaration_sha256': digest(DECLARATION),
                     'builder': rel(Path(__file__)), 'builder_sha256': digest(Path(__file__)),
                     'io': 'scripts/v2_fm_zoo_io.py', 'io_sha256': digest(ROOT / 'scripts/v2_fm_zoo_io.py'),
                     'decisions': DECISIONS})
    rows = battery_rows()
    failed = []

    def record(result):
        path, meta, gates, refs, extra = result
        manifest['files'][path.name] = describe(path, meta, gates, refs, extra)
        print(f'{path.name}: n={meta["n"]} cov={meta["cov_names"]} gates passed={gates["passed"]} '
              f'({time.time() - t0:.0f} s)', flush=True)
        if not gates['passed']:
            failed.append(path.name)

    with Snapshot() as snap:
        order = same_rows(snap.xc.ep[KEY].reset_index(drop=True), rows, KEY)
        manifest['battery_order_equals_exp_chronos_ep'] = bool(order)
        if not order:
            failed.append('battery order')
        if 'A' in a.tasks:
            for v in VARIABLES:
                for o in OFFSETS:
                    record(task_a(v, o, snap))
        if 'B' in a.tasks:
            for v in VARIABLES:
                record(task_b(v, snap, rows))
        if 'F' in a.tasks:
            parts = {}
            for v in VARIABLES:
                path = OUT / f'B_{v}_Reserved-2026.npz'
                if not path.exists():
                    raise SystemExit(f'task F reads {path.name}: build task B first (--tasks B F)')
                z = load_inputs(path)
                parts[v] = (z['keys'], z['contexts'], z['past_cov'])
            record(task_f(snap, rows, a.aou_unit, parts['oxygen'], parts['temperature'][0]))
    manifest['all_gates_passed'] = bool(not failed and all(f['gates']['passed'] for f in manifest['files'].values()))
    manifest['failed'] = failed
    manifest['created_utc'] = datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')
    manifest['seconds'] = round(time.time() - t0, 1)
    write_json(manifest_path, manifest)
    print(f'manifest: {rel(manifest_path)}; all gates passed: {manifest["all_gates_passed"]}', flush=True)
    if failed:
        raise SystemExit(f'gates failed: {failed}')


if __name__ == '__main__':
    main()
