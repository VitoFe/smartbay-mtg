"""Addendum 81 on the compute host: checks of the shared model layer before any task is run.

  G3   each of the eight arms run twice on the same 256 contexts per variable (the A1 contexts of
       addendum 53 with their holes, horizon 72) gives identical outputs, or differences below 1e-6.
       Covariate arms run on synthetic covariates that only exercise the code: the context lagged by
       72 h and the hour of day, both known over the horizon (schema 'past_future'), and the context
       lagged by 24 h with no future part (schema 'past_only'; TimesFM 2.5 XReg then holds its last
       value, since XReg cannot run without one).
  G1   (partial, task A) on every A1 context of offset 0, the adapter's chronos_bolt_base_uni,
       timesfm25_uni and chronos2_uni six-hour medians against the published chronos_zs, timesfm_zs
       and chronos2_zs columns of runs/v2_sequence_sweep/panel/panel_{v}_offset0.csv.gz, joined on
       (fold, sensor, time); tolerance 1e-4 in physical units.
  timing  contexts per second per arm at 6 h (G1 runs), 72 h (G3 runs) and 720 h (32 contexts),
       with the peak GPU memory.

Reads inputs/A1_{v}.npz and inputs/panel_{v}_offset0.csv.gz beside this script; writes
model_checks.json there.
  python v2_fm_zoo_checks.py [--quick]
"""
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import v2_fm_zoo_models as zm  # noqa: E402

INP = HERE / 'inputs'
QUICK = '--quick' in sys.argv
OUT = HERE / ('model_checks_quick.json' if QUICK else 'model_checks.json')
VARIABLES = ('temperature', 'oxygen')
G3_N, G3_H, LONG_N, LONG_H = (16, 72, 4, 720) if QUICK else (256, 72, 32, 720)
G1_TOL = 1e-4
G3_TOL = 1e-6
G1_ARMS = [('chronos_bolt_base_uni', 'chronos_zs', 256, {}), ('chronos_bolt_base_uni', 'chronos_zs', 64, {}),
           ('timesfm25_uni', 'timesfm_zs', None, {}),
           ('timesfm25_uni', 'timesfm_zs', None, {'tfm25_config': {'use_continuous_quantile_head': False,
                                                                  'fix_quantile_crossing': False}}),
           ('chronos2_uni', 'chronos2_zs', 128, {})]
KEEP_META = ('model_id', 'revision', 'library_levels', 'level_method', 'nan_treatment', 'nan_record',
             'horizon_method', 'native_horizon', 'autoregressive_blocks', 'autoregressive_steps',
             'context_limit', 'batch', 'per_core_batch_size', 'forecast_config', 'flags', 'model_config',
             'xreg', 'covariates', 'covariate_roles', 'positivity_clamp', 'quantile_crossing_share',
             'call', 'schemas', 'versions')


def a1(variable):
    z = np.load(INP / f'A1_{variable}.npz', allow_pickle=False)
    keys = pd.DataFrame({'fold': z['key_fold'].astype(int), 'sensor': z['key_sensor'].astype(str),
                         'time': z['key_time'].astype(str)})
    return z, keys


def hours_of(times):
    return pd.to_datetime(pd.Series(times), utc=True).dt.hour.to_numpy()


def synthetic(raw, times, horizon):
    """Covariates that exercise the code only. raw [n, L] with NaN; issuance hours from times."""
    n, L = raw.shape
    hour = hours_of(times)
    pf_past, pf_future, po_past = [], [], []
    for i in range(n):
        x = raw[i].astype(np.float32)
        lag72 = np.concatenate([np.full(72, np.nan, np.float32), x[:-72]])
        fut72 = x[L - 72:L - 72 + horizon] if horizon <= 72 else np.concatenate(
            [x[L - 72:], np.full(horizon - 72, np.nan, np.float32)])
        steps_past = (hour[i] - np.arange(L - 1, -1, -1)) % 24
        steps_fut = (hour[i] + np.arange(1, horizon + 1)) % 24
        pf_past.append({'lag72': lag72, 'hour_sin': np.sin(2 * np.pi * steps_past / 24).astype(np.float32)})
        pf_future.append({'lag72': fut72.astype(np.float32),
                          'hour_sin': np.sin(2 * np.pi * steps_fut / 24).astype(np.float32)})
        po_past.append({'lag24': np.concatenate([np.full(24, np.nan, np.float32), x[:-24]])})
    return {'past_future': (pf_past, pf_future), 'past_only': (po_past, None)}


def trim(meta):
    return {k: meta[k] for k in KEEP_META if k in meta}


def gpu_reset(torch):
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()


def gpu_peak(torch):
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        return round(torch.cuda.max_memory_allocated() / 2 ** 30, 3)
    return None


def run_twice(zoo, arm, ctx, horizon, past=None, future=None, **kw):
    out = []
    for _ in range(2):
        gpu_reset(zoo.torch)
        t0 = time.time()
        r = zoo.predict(arm, ctx, horizon, past_cov=past, future_cov=future, **kw)
        r['wall'] = time.time() - t0
        r['peak_gib'] = gpu_peak(zoo.torch)
        out.append(r)
    a, b = out
    dm = float(np.nanmax(np.abs(a['median'] - b['median']))) if a['median'].size else 0.0
    dq = float(np.nanmax(np.abs(a['q'] - b['q']))) if a['q'].size else 0.0
    nan_mismatch = int((np.isnan(a['q']) != np.isnan(b['q'])).sum())
    return a, b, {'max_abs_diff_median_z': dm, 'max_abs_diff_q_z': dq, 'nan_pattern_differs': nan_mismatch,
                  'identical': bool(np.array_equal(a['q'], b['q'], equal_nan=True)),
                  'pass': bool(max(dm, dq) < G3_TOL and nan_mismatch == 0),
                  'nan_outputs': int(np.isnan(a['q']).sum())}


def main():
    t_start = time.time()
    zoo = zm.Zoo()
    torch = zoo.torch
    report = {'declared_in': 'config/bay_reserved_protocol_v2_addendum81.json', 'quick': QUICK,
              'host_device': (torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu'),
              'cuda_visible_devices': os.environ.get('CUDA_VISIBLE_DEVICES'),
              'threads': torch.get_num_threads(), 'hf_hub_offline': os.environ.get('HF_HUB_OFFLINE'),
              'tf32': {'matmul': torch.backends.cuda.matmul.allow_tf32, 'cudnn': torch.backends.cudnn.allow_tf32},
              'versions': zm._versions(), 'G1': {}, 'G3': {}, 'timing': {}, 'meta': {}}
    data = {v: a1(v) for v in VARIABLES}

    # ---------------------------------------------------------------------------- G1 (task A, partial)
    for v in VARIABLES:
        z, keys = data[v]
        scale = float(z['scale'])
        ctx = z['linear'] if not QUICK else z['linear'][:64]
        k = keys if not QUICK else keys.iloc[:64]
        panel = pd.read_csv(INP / f'panel_{v}_offset0.csv.gz', float_precision='round_trip')
        panel['time'] = panel.time.astype(str)
        j = k.reset_index(names='row').merge(panel.reset_index(names='prow'), on=['fold', 'sensor', 'time'],
                                             how='left' if QUICK else 'outer', indicator=True)
        both = j[j['_merge'] == 'both']
        rec = {'scale': scale, 'contexts': int(len(ctx)), 'panel_rows': int(len(panel)),
               'matched': int(len(both)), 'only_in_a1': int((j['_merge'] == 'left_only').sum()),
               'only_in_panel': int((j['_merge'] == 'right_only').sum()),
               'same_order': bool(len(both) == len(k) and (both.row.to_numpy() == both.prow.to_numpy()).all()),
               'arms': {}}
        for arm, col, batch, kw in G1_ARMS:
            gpu_reset(torch)
            t0 = time.time()
            r = zoo.predict(arm, list(ctx), 6, batch_size=batch, **kw)
            wall = time.time() - t0
            pred = r['median'][both.row.to_numpy(), 5].astype(np.float64)
            pub = both[col].to_numpy(np.float64)
            d = np.abs(pred - pub) * scale
            tag = (f'{arm}|batch{batch or zm.BATCH[zm.arm_parts(arm)[0]]}'
                   + ('|published_config' if kw.get('tfm25_config') else ''))
            rec['arms'][tag] = {'published_column': col, 'max_abs_diff_physical': float(d.max()),
                                'mean_abs_diff_physical': float(d.mean()), 'rows': int(len(d)),
                                'pass': bool(d.max() <= G1_TOL), 'seconds': round(wall, 2),
                                'contexts_per_second': round(len(ctx) / wall, 1), 'peak_gib': gpu_peak(torch)}
            report['meta'][f'G1|{v}|{tag}'] = trim(r['meta'])
            print('G1', v, tag, json.dumps(rec['arms'][tag]), flush=True)
        report['G1'][v] = rec
        _save(report)

    # ---------------------------------------------------------------------------- G3 and timing at 72 h
    for v in VARIABLES:
        z, keys = data[v]
        n = len(keys)
        pick = np.unique(np.linspace(0, n - 1, G3_N).round().astype(int))
        raw = z['raw'][pick]
        times = keys.time.to_numpy()[pick]
        ctx = list(raw)
        cov = synthetic(raw, times, G3_H)
        rec = {'contexts': int(len(pick)), 'with_a_hole': int(z['hole_any'][pick].sum()),
               'nan_hours': int(np.isnan(raw).sum()), 'horizon': G3_H, 'arms': {}}
        for arm in zm.ARMS:
            family, mode = zm.arm_parts(arm)
            schemas = [('none', None, None, {})] if mode == 'uni' else [
                ('past_future',) + cov['past_future'] + ({},),
                ('past_only',) + cov['past_only'] + ({'xreg_future': 'hold_last'} if family == 'timesfm25' else {},)]
            for schema, past, future, kw in schemas:
                a, b, g = run_twice(zoo, arm, ctx, G3_H, past, future, **kw)
                tag = arm if schema == 'none' else f'{arm}|{schema}'
                g.update(seconds=round(a['wall'], 2), contexts_per_second=round(len(ctx) / a['wall'], 1),
                         peak_gib=a['peak_gib'], second_run_seconds=round(b['wall'], 2))
                rec['arms'][tag] = g
                report['meta'][f'G3|{v}|{tag}'] = trim(a['meta'])
                print('G3', v, tag, json.dumps(g), flush=True)
            _save(report)
        report['G3'][v] = rec
        _save(report)

    # ---------------------------------------------------------------------------- timing at 720 h
    z, keys = data['temperature']
    pick = np.arange(LONG_N)
    raw = z['raw'][pick]
    cov = synthetic(raw, keys.time.to_numpy()[pick], LONG_H)
    for arm in zm.ARMS:
        family, mode = zm.arm_parts(arm)
        past, future = (None, None) if mode == 'uni' else cov['past_future']
        gpu_reset(torch)
        t0 = time.time()
        try:
            r = zoo.predict(arm, list(raw), LONG_H, past_cov=past, future_cov=future)
            wall = time.time() - t0
            report['timing'][f'{arm}|h{LONG_H}'] = {
                'contexts': LONG_N, 'seconds': round(wall, 2), 'contexts_per_second': round(LONG_N / wall, 2),
                'peak_gib': gpu_peak(torch), 'nan_outputs': int(np.isnan(r['q']).sum()),
                'horizon_method': r['meta'].get('horizon_method'),
                'autoregressive': r['meta'].get('autoregressive_blocks', r['meta'].get('autoregressive_steps'))}
        except Exception as e:                                   # recorded, never hidden
            report['timing'][f'{arm}|h{LONG_H}'] = {'error': f'{type(e).__name__}: {e}'}
        print('T', arm, json.dumps(report['timing'][f'{arm}|h{LONG_H}']), flush=True)
        _save(report)

    report['G1_pass'] = all(a['pass'] for r in report['G1'].values() for a in r['arms'].values())
    report['G3_pass'] = all(a['pass'] for r in report['G3'].values() for a in r['arms'].values())
    report['load_seconds'] = getattr(zoo, 'load_seconds', {})
    report['revisions'] = zoo.revisions
    report['seconds'] = round(time.time() - t_start, 1)
    _save(report)
    print('done G1', report['G1_pass'], 'G3', report['G3_pass'], report['seconds'], 's', flush=True)


def _save(report):
    OUT.write_bytes((json.dumps(report, indent=2, default=str) + '\n').encode('utf-8'))


if __name__ == '__main__':
    main()
