"""Addendum 82 part G: the Dev-2025 inputs of the eight pretrained arms forecasting apparent oxygen
utilisation at six hours on the six issuance grids, and their two comparators.

Per issuance offset o (0-5), on the rows of addendum 81 task A oxygen (addendum 51's oxygen panel):
  G_oxygen_aou_offset<o>.npz        AOU(t) = o2_sol_mgl(T_sonde(t), 38) - c x DO(t) in mg/l, from the hourly
                                    table (reports/bay_twin/hourly_observations.csv.gz), unstandardised; the
                                    window of v2_sequence_arms.context_window (512 h ending at issuance, at least
                                    48 finite utilisation hours, holes interpolated in time with
                                    limit_direction='both') left-padded with its first value to 512, as gather.
                                    Rows admissible for oxygen but not for utilisation are left out and counted.
  G_oxygen_raw_offset<o>.npz        direct_raw: task A oxygen's stored contexts x 50 + 230 (the stored sonde
                                    unit, umol/kg), unstandardised.
  G_temperature_raw_offset<o>.npz   T_raw: the same sonde's temperature over the same window, task A
                                    temperature's stored contexts x 10 + 20 (degC), one row per oxygen row.
  G_oxygen_aou_offset<o>_reconstruction.npz   per utilisation row, its row in the T_raw file and S = 38.
Covariates: task A oxygen's (net_T, net_sd, own_T, past only, degC) for the utilisation and direct_raw files,
task A temperature's (net_T, net_sd) for T_raw, copied row by row.

Gate G-G2 (reports/bay_twin/runs/v2_addendum82/G/inputs/manifest_aou_dev.json; the builder exits non-zero if
any part fails): keys and order equal the oxygen panel; each utilisation hour before interpolation equals
o2_sol_mgl(T) - c x DO from the hourly table exactly, with the holes of the panel's own temperature and oxygen
(v2_sequence_arms.hourly_panel); the window positions reproduce task A oxygen's contexts (context_window on the
panel); direct_raw and T_raw equal task A's stored contexts x scale + centre (the float32 nearest to the exact
value); every oxygen row finds its temperature row; covariates equal task A's exactly; every file reads back.

Declared in config/bay_reserved_protocol_v2_addendum82.json, part G.
Usage: python -m scripts.v2_fm_zoo_inputs_aou_dev
"""
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from scripts.v2_fm_zoo_io import digest, load_inputs, save_inputs  # noqa: E402
from src.data.thermodynamics import o2sol_garcia_gordon_umol_per_l  # noqa: E402
from src.experiments import v2_sequence_arms as sa  # noqa: E402
from src.units import MG_PER_UMOL_L, factor  # noqa: E402

RUNS = ROOT / 'reports/bay_twin/runs'
ZOO_INPUTS = RUNS / 'v2_fm_zoo' / 'inputs'
PANEL = RUNS / 'v2_sequence_sweep' / 'panel'
HOURLY = ROOT / 'reports/bay_twin/hourly_observations.csv.gz'
OUT = RUNS / 'v2_addendum82' / 'G' / 'inputs'
DECLARATION = 'config/bay_reserved_protocol_v2_addendum82.json'
OFFSETS = (0, 1, 2, 3, 4, 5)
SALINITY = 38.0
CENTER_SCALE = {'temperature': (20., 10.), 'oxygen': (230., 50.)}
HORIZON = 6
TOL_PANEL = 1e-4                    # task A's standardisation tolerance (float32 panel against the table)
TOL_WINDOW = 1e-6                   # task A's A1 tolerance, standardised units


def o2_sol_mgl(T, S=SALINITY):
    """v2_oxygen_mgl_chronos.o2_sol in mode 'mgl': Garcia-Gordon per litre, T clipped to [-2, 40], x 0.0319988."""
    T = np.clip(np.asarray(T, float), -2, 40)
    return o2sol_garcia_gordon_umol_per_l(T, S) * MG_PER_UMOL_L


def rel(p):
    return Path(p).resolve().relative_to(ROOT).as_posix()


def hourly_table(index):
    obs = pd.read_csv(HOURLY, index_col=0)
    obs.index = pd.to_datetime(obs.index, utc=True)
    obs = obs.sort_index()
    return obs.reindex(index)


def aou_series(table, sensor, c):
    T = table[f'{sensor}::temperature'].to_numpy(float)
    DO = table[f'{sensor}::oxygen'].to_numpy(float)
    return o2_sol_mgl(T) - c * DO, T, DO


def window(series, q_i):
    """v2_sequence_arms.context_window and gather's left pad, on a physical series (float64)."""
    lo = max(0, q_i - sa.CTX + 1)
    hist = series[lo:q_i + 1]
    if np.isfinite(hist).sum() < sa.MIN_FINITE:
        return None, hist
    s_ = pd.Series(hist).interpolate(limit_direction='both').to_numpy(float)
    if not np.isfinite(s_).all():
        return None, hist
    if len(s_) < sa.CTX:
        s_ = np.concatenate([np.full(sa.CTX - len(s_), s_[0]), s_])
    return s_, hist


def nearest_float32(stored, exact):
    """Largest |stored - exact| and whether each stored value is the float32 nearest to the exact one."""
    stored = np.asarray(stored, np.float64)
    exact = np.asarray(exact, np.float64)
    d = np.abs(stored - exact)
    spacing = np.spacing(np.abs(exact).astype(np.float32)).astype(np.float64)
    return float(d.max()), bool((d <= spacing / 2 + 1e-15).all())


def same_keys(a, b):
    return bool(len(a) == len(b) and (a.fold.to_numpy() == b.fold.to_numpy()).all()
            and (a.sensor.astype(str).to_numpy() == b.sensor.astype(str).to_numpy()).all()
            and (np.asarray(a.time, np.int64) == np.asarray(b.time, np.int64)).all())


def roundtrip(path, keys, contexts, past):
    d = load_inputs(path)
    ok = same_keys(d['keys'], keys) and len(d['contexts']) == len(contexts)
    ok = ok and all(np.array_equal(a, np.asarray(b, np.float32), equal_nan=True) for a, b in zip(d['contexts'], contexts))
    if past is not None:
        ok = ok and all(np.array_equal(a, np.asarray(b, np.float32), equal_nan=True) for a, b in zip(d['past_cov'], past))
    return bool(ok)


def build_offset(o, c, panel_o, panel_t, table, index):
    a_o = load_inputs(ZOO_INPUTS / f'A_oxygen_offset{o}.npz')
    a_t = load_inputs(ZOO_INPUTS / f'A_temperature_offset{o}.npz')
    keys = a_o['keys'].copy()
    keys['fold'] = keys.fold.astype(np.int64)
    gates = {}
    panel = pd.read_csv(PANEL / f'panel_oxygen_offset{o}.csv.gz', usecols=['fold', 'sensor', 'time'])
    pk = pd.DataFrame({'fold': panel.fold.astype(np.int64), 'sensor': panel.sensor.astype(str),
                       'time': pd.to_datetime(panel.time, utc=True).dt.tz_convert('UTC').dt.tz_localize(None)
                       .astype('datetime64[ns]').to_numpy().astype(np.int64)})
    gates['keys_equal_oxygen_panel'] = same_keys(keys, pk)
    # temperature rows of the same (fold, sensor, issuance)
    tk = a_t['keys']
    where = {(int(f), str(s), int(t)): i for i, (f, s, t) in enumerate(zip(tk.fold, tk.sensor.astype(str), tk.time))}
    trow = np.array([where.get((int(f), str(s), int(t)), -1) for f, s, t in zip(keys.fold, keys.sensor.astype(str),
                                                                               keys.time)], np.int64)
    gates['oxygen_rows_with_temperature_row'] = int((trow >= 0).sum())
    gates['oxygen_rows'] = int(len(keys))
    # direct_raw and T_raw: affine transforms of task A's stored float32 contexts
    oc, os_ = CENTER_SCALE['oxygen']
    tc, ts = CENTER_SCALE['temperature']
    direct = [(np.asarray(x, np.float64) * os_ + oc).astype(np.float32) for x in a_o['contexts']]
    traw = [(np.asarray(a_t['contexts'][j], np.float64) * ts + tc).astype(np.float32) for j in trow]
    d_dir = [nearest_float32(s, np.asarray(x, np.float64) * os_ + oc) for s, x in zip(direct, a_o['contexts'])]
    d_t = [nearest_float32(s, np.asarray(a_t['contexts'][j], np.float64) * ts + tc) for s, j in zip(traw, trow)]
    gates['direct_raw_vs_task_A_affine'] = {'max_abs_diff': max(x[0] for x in d_dir),
                                            'every_value_the_nearest_float32': all(x[1] for x in d_dir)}
    gates['T_raw_vs_task_A_affine'] = {'max_abs_diff': max(x[0] for x in d_t),
                                       'every_value_the_nearest_float32': all(x[1] for x in d_t)}
    # utilisation windows
    names_o, z_o = panel_o
    names_t, z_t = panel_t
    pos = index.get_indexer(pd.to_datetime(keys.time.to_numpy(), utc=True))
    if (pos < 0).any():
        raise SystemExit(f'offset {o}: issuance outside the panel index')
    contexts, kept, dropped, hours_checked = [], [], [], 0
    worst_recompute, holes_same, window_worst = 0.0, True, 0.0
    aou_cache = {}
    for r, (s, q_i) in enumerate(zip(keys.sensor.astype(str), pos)):
        if s not in aou_cache:
            aou, T, DO = aou_series(table, s, c)
            si_o, si_t = names_o.index(s), names_t.index(s)
            zO, zT = z_o[:, si_o].astype(np.float64), z_t[:, si_t].astype(np.float64)
            aou_cache[s] = (aou, T, DO, zO, zT)
        aou, T, DO, zO, zT = aou_cache[s]
        w, hist = window(aou, q_i)
        # the window positions reproduce task A oxygen's context (context_window on the panel)
        wz, _ = window(zO, q_i)
        if wz is not None:
            window_worst = max(window_worst, float(np.abs(wz - np.asarray(a_o['contexts'][r], np.float64)).max()))
        else:
            window_worst = float('inf')
        if w is None:
            dropped.append(r)
            continue
        contexts.append(w.astype(np.float32))
        kept.append(r)
    # the utilisation hours from the table: recomputed independently and against the panel's holes
    for s, (aou, T, DO, zO, zT) in aou_cache.items():
        again = o2_sol_mgl(T) - c * DO
        both = np.isfinite(again)
        worst_recompute = max(worst_recompute, float(np.abs(again[both] - aou[both]).max()) if both.any() else 0.0)
        holes_same &= bool((np.isfinite(aou) == (np.isfinite(zO) & np.isfinite(zT))).all())
        fo, ft = np.isfinite(zO), np.isfinite(zT)
        holes_same &= bool((np.isfinite(DO) == fo).all() and (np.isfinite(T) == ft).all())
        gates.setdefault('table_vs_panel_max_abs', {})[s] = {
            'oxygen_umol_kg': float(np.abs(DO[fo] - (zO[fo] * os_ + oc)).max()),
            'temperature_degC': float(np.abs(T[ft] - (zT[ft] * ts + tc)).max())}
        hours_checked += int(len(aou))
    tp = gates['table_vs_panel_max_abs']
    gates['utilisation_from_table'] = {
        'hours_checked': hours_checked, 'recomputed_max_abs_diff': worst_recompute,
        'holes_equal_panel_temperature_and_oxygen': holes_same,
        'table_within_panel_tolerance': bool(all(v['oxygen_umol_kg'] <= TOL_PANEL and v['temperature_degC'] <= TOL_PANEL
                                                 for v in tp.values())),
        'passed': bool(worst_recompute == 0.0 and holes_same)}
    gates['window_positions_reproduce_task_A'] = {'max_abs_diff_standardised': window_worst,
                                                  'passed': bool(window_worst <= TOL_WINDOW)}
    gates['utilisation_rows'] = len(kept)
    gates['rows_admissible_for_oxygen_not_for_utilisation'] = len(dropped)
    # covariates copied row by row
    p_o = [np.asarray(a_o['past_cov'][i], np.float32) for i in range(len(keys))]
    p_t = [np.asarray(a_t['past_cov'][j], np.float32) for j in trow]
    names = {'oxygen': a_o['meta']['cov_names'], 'temperature': a_t['meta']['cov_names']}
    sources = [ZOO_INPUTS / f'A_oxygen_offset{o}.npz', ZOO_INPUTS / f'A_temperature_offset{o}.npz', HOURLY,
               PANEL / f'panel_oxygen_offset{o}.csv.gz', ROOT / 'src/experiments/v2_sequence_arms.py',
               ROOT / 'src/data/thermodynamics.py', RUNS / 'v2_oxygen_units.json']
    hours = sorted(set(int(h) for h in pd.to_datetime(keys.time.to_numpy(), utc=True).hour))
    files = {}
    kk = keys.iloc[kept].reset_index(drop=True)
    p_aou = [p_o[i] for i in kept]
    common = dict(cohort='Dev-2025', horizon=HORIZON, standardisation=None, sources=sources)
    spec = {
        f'G_oxygen_aou_offset{o}.npz': dict(
            task='G_dev_6h_aou', variable='oxygen', keys=kk, contexts=contexts, past_cov=p_aou, cov_names=names['oxygen'],
            units='mg/l: Garcia-Gordon saturation per litre at S = 38 (T clipped to [-2, 40]) x 0.0319988 minus the '
                  'sonde oxygen x c; unstandardised',
            note=f'Dev-2025, offset {o} (issuance hours {hours} UTC): apparent oxygen utilisation over the window of '
                 'v2_sequence_arms.context_window, the rows of task A oxygen admissible for utilisation; covariates '
                 'task A oxygen\'s.'),
        f'G_oxygen_raw_offset{o}.npz': dict(
            task='G_dev_6h_direct_raw', variable='oxygen', keys=keys, contexts=direct, past_cov=p_o,
            cov_names=names['oxygen'], units='umol/kg, the stored sonde column, unstandardised (scored x c)',
            note=f'Dev-2025, offset {o}: task A oxygen\'s contexts x 50 + 230; covariates task A oxygen\'s.'),
        f'G_temperature_raw_offset{o}.npz': dict(
            task='G_dev_6h_T_raw', variable='temperature', keys=keys, contexts=traw, past_cov=p_t,
            cov_names=names['temperature'], units='degC, unstandardised',
            note=f'Dev-2025, offset {o}: the oxygen sondes\' temperature, task A temperature\'s contexts x 10 + 20 at '
                 'the oxygen rows; covariates task A temperature\'s.')}
    ok_rt = True
    for name, sp in spec.items():
        path = OUT / name
        save_inputs(path, task=sp['task'], variable=sp['variable'], keys=sp['keys'], contexts=sp['contexts'],
                    units=sp['units'], past_cov=sp['past_cov'], cov_names=sp['cov_names'], note=sp['note'], **common)
        rt = roundtrip(path, sp['keys'], sp['contexts'], sp['past_cov'])
        ok_rt &= rt
        files[name] = {'sha256': digest(path), 'rows': int(len(sp['keys'])), 'roundtrip': rt, 'task': sp['task']}
    side = OUT / f'G_oxygen_aou_offset{o}_reconstruction.npz'
    np.savez_compressed(side, temperature_row=np.asarray(kept, np.int64), salinity=np.full(len(kept), SALINITY),
                        key_fold=kk.fold.to_numpy(), key_sensor=kk.sensor.astype(str).to_numpy().astype('U'),
                        key_time=kk.time.to_numpy().astype(np.int64),
                        meta=np.array(json.dumps({'formula': 'oxygen_hat = o2_sol_mgl(T_hat, 38) - aou_hat (mg/l)',
                                                  'T_hat': f'the same arm\'s six-hour median on G_temperature_raw_offset{o}.npz '
                                                           'row temperature_row, same mode'})))
    files[side.name] = {'sha256': digest(side), 'rows': len(kept)}
    cov_equal = all(np.array_equal(a, np.asarray(b, np.float32), equal_nan=True)
                    for a, b in zip(p_o, a_o['past_cov'])) and all(
        np.array_equal(a, np.asarray(a_t['past_cov'][j], np.float32), equal_nan=True) for a, j in zip(p_t, trow))
    gates['covariates_equal_task_A'] = bool(cov_equal)
    gates['roundtrip'] = bool(ok_rt)
    gates['passed'] = bool(gates['keys_equal_oxygen_panel'] and (trow >= 0).all()
                           and gates['direct_raw_vs_task_A_affine']['every_value_the_nearest_float32']
                           and gates['T_raw_vs_task_A_affine']['every_value_the_nearest_float32']
                           and gates['utilisation_from_table']['passed']
                           and gates['window_positions_reproduce_task_A']['passed']
                           and cov_equal and ok_rt)
    return files, gates


def main():
    t0 = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    c = factor()
    names_o, index, z_o, _ = sa.hourly_panel('oxygen')
    names_t, index_t, z_t, _ = sa.hourly_panel('temperature')
    assert (index == index_t).all()
    table = hourly_table(index)
    man = {'declared_in': DECLARATION, 'part': 'G', 'producer': 'scripts/v2_fm_zoo_inputs_aou_dev.py',
           'producer_sha256': digest(Path(__file__)), 'c_mg_per_l_per_umol_kg': c, 'salinity': SALINITY,
           'created_utc': datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ'),
           'gate_tolerance_note': ('direct_raw and T_raw are stored as float32 (the input format); the declared 1e-6 '
                                   'is read as "the stored value is the float32 nearest to task A\'s stored context x '
                                   'scale + centre", since float32 spacing at 230 umol/kg is 1.5e-5'),
           'files': {}, 'gates': {}}
    ok = True
    for o in OFFSETS:
        files, gates = build_offset(o, c, (names_o, z_o), (names_t, z_t), table, index)
        man['files'].update(files)
        man['gates'][f'offset{o}'] = gates
        ok &= gates['passed']
        print(f'offset {o}: rows {gates["oxygen_rows"]}, utilisation {gates["utilisation_rows"]}, dropped '
              f'{gates["rows_admissible_for_oxygen_not_for_utilisation"]}, passed {gates["passed"]}', flush=True)
    man['passed'] = bool(ok)
    man['seconds'] = round(time.time() - t0, 1)
    (OUT / 'manifest_aou_dev.json').write_bytes((json.dumps(man, indent=1) + '\n').encode('utf-8'))
    print('G-G2', ok)
    if not ok:
        sys.exit(1)


if __name__ == '__main__':
    main()
