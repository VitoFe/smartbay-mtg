"""Addendum 82 part G, Reserved-2026: the apparent-utilisation tree applied frozen on the reference grid.

aou_tree_2026 is the utilisation tree at its shared pre-2026 configuration (study oxygen_aou_tree_h6), refitted
exactly as the released model was (src/experiments/v2_release.refit_forecast: training issuances whose target
falls by 2025-11-14 minus 78 h, sensors with at least 100 valid training endpoints, per-sensor weights, the
released weather design, LightGBM regression_l1, random_state 42, deterministic), with the anchor and the
target replaced by utilisation in mg/l as the C3 tree builds them (AOU = Garcia-Gordon saturation per litre at
S = 38 x 0.0319988 at the sonde's forward-filled temperature, minus the sonde oxygen x c; z-scored with the
oxygen centre and scale x c). Its blend weight is chosen on its own calibration window (2025-11-14 to
2025-12-28) over {0, 0.25, 0.5, 0.75, 1} as the released pipeline chooses it, on its own target; weight 1 is
reported beside. On the 2026 grid it runs through the released inference loop (v2_reserved.evaluate_forecast,
causal_features at each six-hour issuance from the record up to it), with the filled value and the neighbour
mean computed from utilisation and every other input as the released design, and is inverted at the released
model's six-hour temperature forecast for the same (sensor, issuance). aou_tree_2026_fold1 is the same with
fold 1's per-fold configuration of addendum 48. thermodynamic_2026 persists utilisation and inverts it at the
same temperature forecast.

Gate G-G5: (a) with the oxygen target and the released configuration, the refit path here reproduces
runs/v2_release_v2/refit_forecast/refit_oxygen_model.json byte for byte; (b) through the inference loop here
that model reproduces the battery's released oxygen column on its 799 rows to 1e-9; (c) at every training and
calibration issuance with a valid utilisation anchor, the inference design equals the training design to 1e-9;
(d) the utilisation anchor and target equal the C3 construction (v2_horizon.build under the mg/l harmonisation)
on shared rows to 1e-12; (e) every battery oxygen row finds the released model's temperature forecast.

Run locally with OMP_NUM_THREADS unset and V2_THREADS=2:
  python -m scripts.v2_aou_frozen2026
Writes reports/bay_twin/runs/v2_addendum82/G/frozen2026/ (models as JSON, predictions and calibration rows as
npz) and its gates.json. Declared in config/bay_reserved_protocol_v2_addendum82.json, part G.
"""
import hashlib
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import lightgbm as lgb  # noqa: E402

from scripts.v2_chronos_lora_contrasts import battery_frame  # noqa: E402
from src.data.bay_forcing import weather  # noqa: E402
from src.data.thermodynamics import o2sol_garcia_gordon_umol_per_l  # noqa: E402
from src.experiments import bay_graph_dynamics as G  # noqa: E402
from src.experiments import bay_reserved_evaluation as v1  # noqa: E402
from src.experiments import v2_release as REL  # noqa: E402
from src.experiments.bay_dynamics_tree import features  # noqa: E402
from src.experiments.bay_robust_forecast import choose_weight  # noqa: E402
from src.experiments.bay_twin_sweep import save  # noqa: E402
from src.models.bay_sensor_inference import causal_features  # noqa: E402
from src.units import MG_PER_UMOL_L, factor  # noqa: E402

RUNS = ROOT / 'reports/bay_twin/runs'
OUT = RUNS / 'v2_addendum82' / 'G' / 'frozen2026'
RELEASED = RUNS / 'v2_release_v2' / 'refit_forecast'
C3 = RUNS / 'v2_oxygen_mgl' / 'C3'
SHARED_SELECTION = C3 / 'v2_aou_forecast' / 'oxygen_aou_tree_h6_selection.json'
PERFOLD_REPORT = C3 / 'v2_perfold_rest' / 'aou' / 'perfold' / 'report.json'
DECLARATION = 'config/bay_reserved_protocol_v2_addendum82.json'
SALINITY = 38.0
CENTRE, SCALE = 230.0, 50.0
CAL_START = pd.Timestamp('2025-11-14T00:00:00', tz='UTC')
CAL_END = pd.Timestamp('2025-12-28T18:00:00', tz='UTC')
GRID = [0.0, 0.25, 0.5, 0.75, 1.0]
H6 = pd.Timedelta(hours=6)


def sat_mgl(T):
    """The C3 saturation: Garcia-Gordon per litre at S = 38, x 0.0319988 (mg/l)."""
    return o2sol_garcia_gordon_umol_per_l(np.asarray(T, float), SALINITY) * MG_PER_UMOL_L


def now():
    return datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


# --------------------------------------------------------------------------------------------------
# Training data
# --------------------------------------------------------------------------------------------------
def utilisation_data(c):
    """The released oxygen build with anchor and target replaced by utilisation in mg/l (z-scored with the
    oxygen centre and scale x c); every other input as built. Also returns the plain oxygen build."""
    ox = G.build('oxygen')
    te = G.build('temperature')
    col = {n: i for i, n in enumerate(te['names'])}
    take = [col[n] for n in ox['names']]
    T_anchor = te['x'][:, take] * te['scale'] + te['center']
    T_obs = te['truth'][:, take] * te['scale'] + te['center']
    do_anchor = (ox['x'] * ox['scale'] + ox['center']) * c
    do_truth = (ox['truth'] * ox['scale'] + ox['center']) * c
    aou_anchor = sat_mgl(T_anchor) - do_anchor
    aou_truth = sat_mgl(T_obs) - do_truth
    nc, ns = CENTRE * c, SCALE * c
    data = dict(ox)
    data['x'] = (aou_anchor - nc) / ns
    data['truth'] = (aou_truth - nc) / ns
    data['valid'] = ox['valid'] & np.isfinite(aou_anchor) & np.isfinite(aou_truth)
    data['center'], data['scale'] = nc, ns
    return data, ox, {'aou_anchor': aou_anchor, 'aou_truth': aou_truth}


def refit(data, variable, params, out, tag, design_data=None):
    """v2_release.refit_forecast with the data passed in (the same statements, in the same order). With
    `design_data` (the sensitivity of amendment 1) the design and its fallback come from that build, as the
    Dev-2025 utilisation tree builds them from the oxygen data, while target, anchor and mask stay `data`'s."""
    out.mkdir(parents=True, exist_ok=True)
    t = data['times']
    met = weather(t).to_numpy()
    cal_end, cal_start = CAL_END, CAL_START
    train_end = cal_start - pd.Timedelta(hours=78)
    tr = np.flatnonzero(t + pd.Timedelta(hours=6) <= train_end)
    ca = np.flatnonzero((t >= cal_start) & (t + pd.Timedelta(hours=6) <= cal_end))
    eligible = data['valid'][tr].sum(0) >= 100
    src = data if design_data is None else design_data
    fallback = np.nan_to_num(pd.DataFrame(src['x'][tr]).median().to_numpy())
    base, _ = features(src, {'fallback': fallback}, 'forecast')
    x_weather = np.concatenate([base, np.broadcast_to(met[:, None], (len(t), len(data['names']), met.shape[1]))], axis=-1)
    mask = data['valid'] & eligible[None]
    tm = mask[tr]
    weight = 1 / np.maximum(tm.sum(0), 1)
    w = np.broadcast_to(weight, tm.shape)[tm]
    w /= w.mean()
    contract = {'model_family': 'weather_tree', 'variable': variable, 'train_start': str(t[tr].min()),
                'last_training_outcome': str(t[tr].max() + pd.Timedelta(hours=6)),
                'first_calibration': str(t[ca].min()), 'last_calibration_outcome': str(t[ca].max() + pd.Timedelta(hours=6)),
                'supported_sensors': [n for n, e in zip(data['names'], eligible) if e],
                'training_counts': {n: int(data['valid'][tr, s].sum()) for s, n in enumerate(data['names']) if eligible[s]}}
    model = REL.fit_tree(params, x_weather[tr][tm], (data['truth'][tr] - data['x'][tr])[tm], w)
    cm = mask[ca]
    pred_inc = model.predict(x_weather[ca][cm])
    cal_records = []
    for (j, s), val in zip(np.argwhere(cm), pred_inc):
        i = ca[j]
        anchor = float(data['x'][i, s] * data['scale'] + data['center'])
        cal_records.append({'sensor': data['names'][s], 'time': str(t[i]), 'variable': variable,
                            'truth': float(data['truth'][i, s] * data['scale'] + data['center']),
                            'prediction': float(anchor + val * data['scale']), 'persistence': anchor})
    cal_df = pd.DataFrame(cal_records)
    best_alpha, grid_scores = choose_weight(cal_df, GRID)
    cp = {'model_string': model.booster_.model_to_string(), 'params': params, 'fallback': fallback.tolist(),
          'names': data['names'], 'center': data['center'], 'scale': data['scale'], 'feature_count': x_weather.shape[-1],
          'contract': contract, 'global_blend_weight': best_alpha, 'calibration_grid_scores': grid_scores}
    save(out / f'refit_{tag}_model.json', cp)
    return cp, cal_df, {'tr': tr, 'ca': ca, 'design': x_weather, 'mask': mask}


# --------------------------------------------------------------------------------------------------
# Inference
# --------------------------------------------------------------------------------------------------
def window(frames, names, q):
    lo = q - pd.Timedelta(hours=8)
    return {n: f.loc[(f.index > lo) & (f.index <= q)] for n, f in frames.items() if n in names}


def latest(frames, names, q, col):
    index = pd.date_range(q - H6, q, freq='h')
    raw = pd.DataFrame({n: frames[n][col].reindex(index) if n in frames and col in frames[n] else np.nan
                        for n in names}, index=index)
    return raw.ffill(limit=6).iloc[-1].to_numpy(float), raw.iloc[-1].to_numpy(float)


def oxygen_checkpoint(cp):
    """The checkpoint fields causal_features reads, for the released oxygen design (z in umol/kg)."""
    return {'names': cp['names'], 'variable': 'oxygen', 'center': CENTRE, 'scale': SCALE,
            'fallback': cp['fallback'], 'feature_count': cp['feature_count']}


def aou_design(frames, meta, cp, q, w_row, c):
    """causal_features of the released oxygen design, with the filled value and the neighbour mean computed
    from utilisation (z with the checkpoint's mg/l centre and scale), as features() does on the training data:
    filled = utilisation where finite else the checkpoint's fallback; neighbour = adjacency-weighted mean over
    the probes with oxygen available (the build's `available`)."""
    names = cp['names']
    x, anchors, info = causal_features(frames, meta, oxygen_checkpoint(cp), q, w_row)
    T_latest, _ = latest(frames, names, q, 'temperature')
    DO_latest, _ = latest(frames, names, q, 'oxygen')
    aou = sat_mgl(T_latest) - c * DO_latest
    z = (aou - cp['center']) / cp['scale']
    available = np.isfinite(DO_latest)
    filled = np.where(np.isfinite(z), z, np.asarray(cp['fallback']))
    m = meta.loc[names]
    xy = m[['x_m', 'y_m']].to_numpy()
    distance = np.linalg.norm(xy[:, None] - xy[None, :], axis=-1)
    adjacency = np.exp(-distance / 1500)
    np.fill_diagonal(adjacency, 0)
    k = adjacency * available[:, None] * available[None, :]
    neighbor = np.einsum('ij,j->i', k, filled) / np.maximum(k.sum(-1), 1e-6)
    neighbor = np.where(k.sum(-1) > 0, neighbor, filled)
    x = x.copy()
    x[:, 0] = filled
    x[:, 1] = neighbor
    return x, anchors, aou


def infer(cp, kind, frames, meta, weather_df, c, eval_times):
    """The released inference loop for the oxygen sensors the checkpoint supports. kind 'oxygen': the released
    design and target (reproduction); kind 'aou': the utilisation design, prediction in mg/l utilisation."""
    model = lgb.Booster(model_str=cp['model_string'])
    alpha = cp['global_blend_weight']
    supported = set(cp['contract']['supported_sensors'])
    rows = []
    for q in eval_times:
        cur = window(frames, cp['names'], q)
        w_row = weather_df.loc[q].to_numpy()
        if kind == 'oxygen':
            x, anchors, _ = causal_features(cur, meta, dict(cp, variable='oxygen'), q, w_row)
            aou = None
        elif kind == 'aou_dev_inputs':
            # the oxygen design (filled value and neighbour from oxygen, oxygen fallback); utilisation anchor
            x, anchors, _ = causal_features(cur, meta, oxygen_checkpoint(cp), q, w_row)
            T_latest, _ = latest(cur, cp['names'], q, 'temperature')
            DO_latest, _ = latest(cur, cp['names'], q, 'oxygen')
            aou = sat_mgl(T_latest) - c * DO_latest
        else:
            x, anchors, aou = aou_design(cur, meta, cp, q, w_row, c)
        target_time = q + H6
        for s, sensor in enumerate(cp['names']):
            if sensor not in supported:
                continue
            anchor = anchors[s]
            if not np.isfinite(anchor):
                continue
            tf = frames[sensor]
            if not (target_time in tf.index and 'oxygen' in tf and pd.notna(tf.loc[target_time, 'oxygen'])):
                continue
            inc = float(model.predict(x[s:s + 1])[0] * cp['scale'])
            if kind == 'oxygen':
                raw_pred = float(anchor + inc)
                pred = float(anchor + alpha * (raw_pred - anchor))
                rows.append({'sensor': sensor, 'time': q, 'prediction': pred, 'raw_prediction': raw_pred})
            else:
                a = float(aou[s])
                if not np.isfinite(a):
                    rows.append({'sensor': sensor, 'time': q, 'aou_anchor': np.nan, 'aou_raw': np.nan, 'aou_blend': np.nan})
                    continue
                raw_pred = a + inc
                rows.append({'sensor': sensor, 'time': q, 'aou_anchor': a, 'aou_raw': raw_pred,
                             'aou_blend': a + alpha * (raw_pred - a)})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------------------------------
# Gates
# --------------------------------------------------------------------------------------------------
def gate_a(ox, params):
    scratch = OUT / 'reproduction'
    cp, _, _ = refit(ox, 'oxygen', params, scratch, 'oxygen')
    mine, pub = (scratch / 'refit_oxygen_model.json').read_bytes(), (RELEASED / 'refit_oxygen_model.json').read_bytes()
    pj = json.loads(pub)
    return cp, {'byte_identical': mine == pub, 'sha256_reproduction': sha(scratch / 'refit_oxygen_model.json'),
                'sha256_published': sha(RELEASED / 'refit_oxygen_model.json'),
                'model_string_equal': cp['model_string'] == pj['model_string'],
                'fallback_equal': cp['fallback'] == pj['fallback'],
                'blend_weight': cp['global_blend_weight'], 'blend_weight_published': pj['global_blend_weight'],
                'passed': bool(mine == pub)}


def gate_b(battery_ox, pred):
    j = battery_ox[['sensor', 'tkey', 'MODEL_v2']].merge(pred[['sensor', 'tkey', 'prediction']], on=['sensor', 'tkey'],
                                                         how='left', validate='one_to_one')
    d = (j.MODEL_v2 - j.prediction).abs()
    return {'rows': int(len(j)), 'found': int(j.prediction.notna().sum()), 'max_abs_diff': float(d.max()),
            'passed': bool(j.prediction.notna().all() and d.max() <= 1e-9)}


def gate_c(data, cp, built, frames, meta, c):
    """Inference design against training design at every training and calibration issuance with a valid
    utilisation anchor (the training weather row passed to both)."""
    t = data['times']
    met = weather(t).to_numpy()
    design = built['design']
    worst, rows, issuances, one_side, both_nan = 0.0, 0, 0, 0, 0
    idx = np.concatenate([built['tr'], built['ca']])
    names = cp['names']
    for i in idx:
        ok = np.isfinite(data['x'][i])
        if not ok.any():
            continue
        q = t[i]
        cur = window(frames, names, q)
        x, _, _ = aou_design(cur, meta, cp, q, met[i], c)
        a, b = x[ok].astype(float), design[i][ok].astype(float)
        na, nb = np.isnan(a), np.isnan(b)
        one_side += int((na != nb).sum())
        both_nan += int((na & nb).sum())
        d = np.where(na | nb, 0.0, np.abs(a - b))
        worst = max(worst, float(d.max()))
        rows += int(ok.sum())
        issuances += 1
    return {'issuances': issuances, 'rows': rows, 'max_abs_diff': worst,
            'values_missing_on_one_side_only': one_side,
            'values_missing_on_both_sides': both_nan,
            'note': 'a value missing on both sides (the weather station silent at that hour) is equal; on one side only, '
                    'a failure',
            'passed': bool(worst <= 1e-9 and one_side == 0)}


def gate_d(c, mine):
    """The C3 construction: v2_horizon.build under the mg/l harmonisation (scripts/v2_oxygen_mgl_aou_forecast)."""
    import scripts.v2_oxygen_mgl_aou_forecast as M3
    import src.experiments.v2_horizon as H
    from scripts.v2_oxygen_mgl_common import harmonised, sat_mgl as c3_sat
    with M3.sonde_unit():
        with harmonised(True):
            o = M3.oxygen_build(H.build)('oxygen', 6)
            te = H.build('temperature', 6)
    col = {n: i for i, n in enumerate(te['names'])}
    take = [col[n] for n in o['names']]
    t_anchor = te['x'][:, take] * te['scale'] + te['center']
    t_obs = te['truth'][:, take] * te['scale'] + te['center']
    do_anchor = o['x'] * o['scale'] + o['center']
    do_truth = o['truth'] * o['scale'] + o['center']
    c3_anchor = c3_sat(t_anchor, SALINITY) - do_anchor
    c3_truth = c3_sat(t_obs, SALINITY) - do_truth
    ox_times = {pd.Timestamp(x): i for i, x in enumerate(mine['times'])}
    worst_a = worst_t = 0.0
    same_finite, compared = True, 0
    for i, tt in enumerate(o['times']):
        j = ox_times.get(pd.Timestamp(tt))
        if j is None:
            continue
        for s, n in enumerate(o['names']):
            k = mine['names'].index(n)
            a, b = c3_anchor[i, s], mine['aou_anchor'][j, k]
            u, v = c3_truth[i, s], mine['aou_truth'][j, k]
            same_finite &= bool(np.isfinite(a) == np.isfinite(b) and np.isfinite(u) == np.isfinite(v))
            if np.isfinite(a) and np.isfinite(b):
                worst_a = max(worst_a, abs(a - b))
                compared += 1
            if np.isfinite(u) and np.isfinite(v):
                worst_t = max(worst_t, abs(u - v))
    return {'compared': compared, 'max_abs_diff_anchor': worst_a, 'max_abs_diff_target': worst_t,
            'same_finite_pattern': same_finite, 'passed': bool(compared > 0 and same_finite and worst_a <= 1e-12
                                                                and worst_t <= 1e-12)}


# --------------------------------------------------------------------------------------------------
def main():
    t0 = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    c = factor()
    gates = {'environment': {'OMP_NUM_THREADS': os.environ.get('OMP_NUM_THREADS'), 'V2_THREADS': os.environ.get('V2_THREADS'),
                             'lightgbm': lgb.__version__, 'python': sys.version.split()[0]}}
    params_released = REL.tree_params('oxygen', 'v2')
    data, ox, aou_arrays = utilisation_data(c)
    # (a) the refit path on the released oxygen design and target
    cp_ox, gates['G-G5a'] = gate_a(ox, params_released)
    print('G-G5a', gates['G-G5a']['passed'], f'{time.time() - t0:.0f} s', flush=True)
    meta, frames = v1.load_all_observations()
    eval_times = pd.date_range('2026-01-01T00:00:00', '2026-06-19T18:00:00', freq='6h', tz='UTC')
    weather_df = v1.load_weather_2026(eval_times)
    d = battery_frame()
    d['tkey'] = pd.to_datetime(d.time, utc=True)
    bo = d[d.variable == 'oxygen'].reset_index(drop=True)
    bt = d[d.variable == 'temperature'].reset_index(drop=True)
    # (b) the inference loop on the released oxygen model
    pub_cp = json.loads((RELEASED / 'refit_oxygen_model.json').read_text())
    pred_ox = infer(pub_cp, 'oxygen', frames, meta, weather_df, c, eval_times)
    pred_ox['tkey'] = pd.to_datetime(pred_ox.time, utc=True)
    gates['G-G5b'] = gate_b(bo, pred_ox)
    print('G-G5b', gates['G-G5b'], f'{time.time() - t0:.0f} s', flush=True)
    # the utilisation trees
    shared = json.loads(SHARED_SELECTION.read_text())
    perfold = json.loads(PERFOLD_REPORT.read_text())['selection']['1']
    configs = {'aou_tree_2026': {**shared['params'], 'subsample_freq': 1},
               'aou_tree_2026_fold1': {**perfold['params'], 'subsample_freq': 1}}
    cps, built_shared = {}, None
    for tag, params in configs.items():
        cp, cal, built = refit(data, 'oxygen_aou', params, OUT, tag)
        cps[tag] = cp
        np.savez_compressed(OUT / f'calibration_{tag}.npz', sensor=cal.sensor.to_numpy().astype('U'),
                            time=pd.to_datetime(cal.time, utc=True).astype('datetime64[ns, UTC]').dt.tz_convert(None)
                            .to_numpy().astype('datetime64[ns]').astype(np.int64),
                            truth=cal.truth.to_numpy(float), prediction=cal.prediction.to_numpy(float),
                            persistence=cal.persistence.to_numpy(float))
        if tag == 'aou_tree_2026':
            built_shared = built
        print(tag, 'blend weight', cp['global_blend_weight'], 'supported', cp['contract']['supported_sensors'],
              f'{time.time() - t0:.0f} s', flush=True)
    # (c) inference design against training design
    gates['G-G5c'] = gate_c(data, cps['aou_tree_2026'], built_shared, frames, meta, c)
    print('G-G5c', gates['G-G5c'], f'{time.time() - t0:.0f} s', flush=True)
    # (d) the utilisation anchor and target against the C3 construction
    mine = {'times': data['times'], 'names': data['names'], **aou_arrays}
    gates['G-G5d'] = gate_d(c, mine)
    print('G-G5d', gates['G-G5d'], f'{time.time() - t0:.0f} s', flush=True)
    # 2026: the trees, the thermodynamic baseline, the inversion at the released temperature forecast
    tmap = bt[['sensor', 'tkey', 'MODEL_v2']].rename(columns={'MODEL_v2': 'T_hat'})
    out = bo[['variable', 'sensor', 'time', 'tkey', 'quarter', 'week']].copy()
    out = out.merge(tmap, on=['sensor', 'tkey'], how='left', validate='one_to_one')
    gates['G-G5e'] = {'rows': int(len(out)), 'with_temperature_forecast': int(out.T_hat.notna().sum()),
                      'passed': bool(out.T_hat.notna().all())}
    sol_hat = sat_mgl(out.T_hat.to_numpy(float))
    for tag, cp in cps.items():
        p = infer(cp, 'aou', frames, meta, weather_df, c, eval_times)
        p['tkey'] = pd.to_datetime(p.time, utc=True)
        j = out[['sensor', 'tkey']].merge(p.drop(columns='time'), on=['sensor', 'tkey'], how='left', validate='one_to_one')
        out[tag] = sol_hat - j.aou_blend.to_numpy(float)
        out[f'{tag}_w1'] = sol_hat - j.aou_raw.to_numpy(float)
        if tag == 'aou_tree_2026':
            out['thermodynamic_2026'] = sol_hat - j.aou_anchor.to_numpy(float)
        gates[f'rows|{tag}'] = {'battery_rows': int(len(out)), 'predicted': int(np.isfinite(out[tag]).sum()),
                                'loop_rows': int(len(p)), 'blend_weight': cp['global_blend_weight']}
    gates['G-G5'] = {'passed': bool(all(gates[k]['passed'] for k in ('G-G5a', 'G-G5b', 'G-G5c', 'G-G5d', 'G-G5e')))}
    arrays = {'sensor': out.sensor.to_numpy().astype('U'),
              'time': out.tkey.dt.tz_convert(None).to_numpy().astype('datetime64[ns]').astype(np.int64),
              'T_hat_released': out.T_hat.to_numpy(float)}
    for col in ('aou_tree_2026', 'aou_tree_2026_w1', 'aou_tree_2026_fold1', 'aou_tree_2026_fold1_w1', 'thermodynamic_2026'):
        arrays[col] = out[col].to_numpy(float)
    np.savez_compressed(OUT / 'predictions_2026.npz', **arrays,
                        meta=np.array(json.dumps({'unit': 'mg/l oxygen', 'rows': 'the battery oxygen rows in order',
                                                  'inversion': 'sat_mgl(T_hat_released) - utilisation forecast'})))
    rep = {'declared_in': DECLARATION, 'part': 'G (Reserved-2026 half)', 'producer': 'scripts/v2_aou_frozen2026.py',
           'producer_sha256': sha(Path(__file__)), 'created_utc': now(), 'c_mg_per_l_per_umol_kg': c,
           'configurations': {'aou_tree_2026': {'source': str(SHARED_SELECTION.relative_to(ROOT).as_posix()),
                                                'params': shared['params'], 'best_trial': shared.get('best_trial')},
                              'aou_tree_2026_fold1': {'source': str(PERFOLD_REPORT.relative_to(ROOT).as_posix()) + ' selection 1',
                                                      'params': perfold['params']}},
           'blend_weights': {k: cp['global_blend_weight'] for k, cp in cps.items()},
           'calibration_grid_scores': {k: cp['calibration_grid_scores'] for k, cp in cps.items()},
           'supported_sensors': {k: cp['contract']['supported_sensors'] for k, cp in cps.items()},
           'gates': gates, 'files': sorted(p.name for p in OUT.iterdir()), 'seconds': round(time.time() - t0, 1)}
    (OUT / 'gates.json').write_bytes((json.dumps(rep, indent=1, default=str) + '\n').encode('utf-8'))
    print('G-G5', gates['G-G5']['passed'], f'{time.time() - t0:.0f} s')


# --------------------------------------------------------------------------------------------------
# Amendment 1: the sensitivity added after the first reading
# --------------------------------------------------------------------------------------------------
def sensitivity():
    """The 2026 tree with the Dev-2025 utilisation tree's input set: filled value, neighbour mean and fallback
    built from oxygen (the C3 tree's design), target and anchor utilisation in mg/l; same configurations, refit
    and blend procedure, inference loop and inversion as the declared reading. Gate: at every training and
    calibration issuance with a valid utilisation anchor, the inference design equals the training design."""
    t0 = time.time()
    out = OUT / 'sensitivity_dev_inputs'
    out.mkdir(parents=True, exist_ok=True)
    c = factor()
    data, ox, _ = utilisation_data(c)
    meta, frames = v1.load_all_observations()
    eval_times = pd.date_range('2026-01-01T00:00:00', '2026-06-19T18:00:00', freq='6h', tz='UTC')
    weather_df = v1.load_weather_2026(eval_times)
    d = battery_frame()
    d['tkey'] = pd.to_datetime(d.time, utc=True)
    bo = d[d.variable == 'oxygen'].reset_index(drop=True)
    bt = d[d.variable == 'temperature'].reset_index(drop=True)
    shared = json.loads(SHARED_SELECTION.read_text())
    perfold = json.loads(PERFOLD_REPORT.read_text())['selection']['1']
    configs = {'aou_tree_2026': {**shared['params'], 'subsample_freq': 1},
               'aou_tree_2026_fold1': {**perfold['params'], 'subsample_freq': 1}}
    tmap = bt[['sensor', 'tkey', 'MODEL_v2']].rename(columns={'MODEL_v2': 'T_hat'})
    rows = bo[['sensor', 'tkey']].merge(tmap, on=['sensor', 'tkey'], how='left', validate='one_to_one')
    sol_hat = sat_mgl(rows.T_hat.to_numpy(float))
    gates, arrays, weights = {}, {}, {}
    t = data['times']
    met = weather(t).to_numpy()
    for tag, params in configs.items():
        cp, cal, built = refit(data, 'oxygen_aou', params, out, tag, design_data=ox)
        weights[tag] = {'blend_weight': cp['global_blend_weight'], 'grid': cp['calibration_grid_scores'],
                        'supported_sensors': cp['contract']['supported_sensors']}
        if tag == 'aou_tree_2026':
            worst, one_side, n = 0.0, 0, 0
            ocp = oxygen_checkpoint(cp)
            for i in np.concatenate([built['tr'], built['ca']]):
                ok = np.isfinite(data['x'][i])
                if not ok.any():
                    continue
                x, _, _ = causal_features(window(frames, cp['names'], t[i]), meta, ocp, t[i], met[i])
                a, b = x[ok].astype(float), built['design'][i][ok].astype(float)
                na, nb = np.isnan(a), np.isnan(b)
                one_side += int((na != nb).sum())
                worst = max(worst, float(np.where(na | nb, 0.0, np.abs(a - b)).max()))
                n += int(ok.sum())
            gates['parity'] = {'rows': n, 'max_abs_diff': worst, 'values_missing_on_one_side_only': one_side,
                               'passed': bool(worst <= 1e-9 and one_side == 0)}
            print('parity', gates['parity'], f'{time.time() - t0:.0f} s', flush=True)
        p = infer(cp, 'aou_dev_inputs', frames, meta, weather_df, c, eval_times)
        p['tkey'] = pd.to_datetime(p.time, utc=True)
        j = rows[['sensor', 'tkey']].merge(p.drop(columns='time'), on=['sensor', 'tkey'], how='left', validate='one_to_one')
        arrays[f'{tag}_devinputs'] = sol_hat - j.aou_blend.to_numpy(float)
        arrays[f'{tag}_devinputs_w1'] = sol_hat - j.aou_raw.to_numpy(float)
        gates[f'rows|{tag}'] = {'battery_rows': int(len(rows)), 'predicted': int(np.isfinite(arrays[f'{tag}_devinputs']).sum())}
        print(tag, weights[tag]['blend_weight'], gates[f'rows|{tag}'], f'{time.time() - t0:.0f} s', flush=True)
    np.savez_compressed(out / 'predictions_2026.npz', sensor=rows.sensor.to_numpy().astype('U'),
                        time=rows.tkey.dt.tz_convert(None).to_numpy().astype('datetime64[ns]').astype(np.int64), **arrays)
    rep = {'declared_in': 'config/bay_reserved_protocol_v2_addendum82_amendment1.json (sensitivity_added_after_reading)',
           'what': 'the 2026 utilisation tree with the input set of the Dev-2025 tree (design from oxygen)',
           'blend': weights, 'gates': gates, 'seconds': round(time.time() - t0, 1)}
    (out / 'gates.json').write_bytes((json.dumps(rep, indent=1, default=str) + '\n').encode('utf-8'))


if __name__ == '__main__':
    if '--sensitivity' in sys.argv:
        sensitivity()
    else:
        main()
