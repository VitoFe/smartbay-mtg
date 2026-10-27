"""Addendum 52: the pretrained apparent-utilisation arms recomputed with saturation in umol/L.

The published arms form utilisation as exp_chronos.o2_sol (Garcia-Gordon in umol/kg, S = 38) minus
a concentration in umol/L, and invert a forecast through the same function. This driver runs the
published code path twice, once with that function (unit 'kg', the reproduction) and once with
src.data.thermodynamics.o2sol_garcia_gordon_umol_per_l at S = 38, T clipped to [-2, 40] as
exp_chronos clips it (unit 'l'). The switch rebinds `o2_sol` in the imported modules; no published
script is edited. Everything else (contexts, windows, trial seeds, selected configurations) is the
published one.

  python v2_aou_units_chronos.py zeroshot                  # Chronos-Bolt and Chronos-2, and the C2 study
  python v2_aou_units_chronos.py adapted --arm tv_full     # one adapted configuration, both units
  add --quick for a smoke run (a few endpoints, 20 training steps) into results_smoke/

Runs from a snapshot directory on the compute host with code/ (exp_chronos.py,
exp_chronos_valtest.py), the published v2_chronos*.py scripts and ref/ (published selections and
columns) beside it, and the repository snapshot on PYTHONPATH for src/.
Declared in config/bay_reserved_protocol_v2_addendum52.json.
"""
import hashlib
import json
import os
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE / 'code'))
os.environ.setdefault('HF_HOME', '/home/vito/smartbay_do/.hf')
QUICK = '--quick' in sys.argv
import torch  # noqa: E402
torch.set_num_threads(int(os.environ.get('V2_THREADS', '2')))
import exp_chronos as xc  # noqa: E402  (the snapshot's copy, imported before any published script)
from src.data.thermodynamics import o2sol_garcia_gordon_umol_per_l  # noqa: E402

OUT = HERE / ('results_smoke' if QUICK else 'results'); OUT.mkdir(exist_ok=True)
REF = HERE / 'ref'
O2_SOL_KG = xc.o2_sol
UNITS = ('kg', 'l')


def o2_sol_l(T, S=xc.SAL):
    """Garcia-Gordon saturation in umol/L at practical salinity S, T clipped as exp_chronos clips it."""
    T = np.clip(np.asarray(T, float), -2, 40)
    return o2sol_garcia_gordon_umol_per_l(T, S)


def set_unit(unit, modules=()):
    """Bind `o2_sol` in exp_chronos (series_for, the C2 target) and in every module whose to_column
    inverts through its own imported name."""
    f = O2_SOL_KG if unit == 'kg' else o2_sol_l
    for m in (xc, *modules):
        if hasattr(m, 'o2_sol'):
            m.o2_sol = f


def eval_times(ep, variables=('temperature', 'oxygen')):
    return {v: {s: pd.DatetimeIndex(sorted(ep.loc[(ep.variable == v) & (ep.sensor == s), 'time']))
                for s in xc.sensors_for(v)} for v in variables}


def column(ep, preds, temp, variable, space):
    """The prediction column in ep's row order, oxygen utilisation inverted at the given temperature."""
    col = np.full(len(ep), np.nan)
    for i, r in ep.iterrows():
        k = (r.sensor, r.time)
        if r.variable != variable or k not in preds:
            continue
        v = preds[k]
        if space == 'aou':
            T = temp.get(k, np.nan)
            if not np.isfinite(T):
                continue
            v = xc.o2_sol(T) - v
        col[i] = v
    return col


def mae(ep, col, variable='oxygen'):
    return float(xc.macro_mae(ep.assign(_c=col)[ep.variable == variable], '_c'))


def compare(col, ref):
    both = np.isfinite(col) & np.isfinite(ref)
    return {'max_abs_diff': float(np.max(np.abs(col[both] - ref[both]))) if both.any() else None,
            'same_missing': bool((np.isfinite(col) == np.isfinite(ref)).all())}


def gate_series():
    """G5: the umol/L utilisation differs from the umol/kg one only through saturation."""
    worst = 0.0
    for s in xc.sensors_for('oxygen'):
        set_unit('kg'); a = xc.series_for(s, 'oxygen', 'aou')
        set_unit('l'); b = xc.series_for(s, 'oxygen', 'aou')
        if a is None:
            continue
        T = xc.obs[f'{s}::temperature'].to_numpy(float)
        expect = o2_sol_l(T) - O2_SOL_KG(T)
        d = (b.to_numpy() - a.to_numpy()) - expect
        worst = max(worst, float(np.nanmax(np.abs(d))))
    return {'max_abs_residual': worst, 'passed': bool(worst <= 1e-9)}


# ---------------------------------------------------------------- zero-shot arms and the C2 corrector

def zeroshot():
    from chronos import BaseChronosPipeline
    import v2_chronos2 as c2m
    import v2_chronos_c2  # noqa: F401  (its samples() is the published C2 data assembly)
    from sklearn.linear_model import Ridge
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from src.tuning.protocol_v2 import run_study
    t0 = time.time()
    ep = xc.ep.groupby('variable').head(40).reset_index(drop=True) if QUICK else xc.ep
    times = eval_times(ep)
    report = {'declared_in': 'config/bay_reserved_protocol_v2_addendum52.json', 'smoke_only': QUICK,
              'G5': gate_series(), 'columns': {}}
    cols = {}
    # C2 needs the published random stream at the start of the utilisation step: v2_chronos_c2.main
    # drew the temperature and the raw-oxygen samples first. The draws are replayed without predicting.
    for variable in ('temperature', 'oxygen'):
        by = xc.pre2026_issuance(variable)
        {s: (t[xc.RNG.choice(len(t), min(len(t), 1500), replace=False)] if len(t) > 1500 else t)
         for s, t in by.items() if len(t) > 0}
    rng_state = xc.RNG.bit_generator.state

    zs = BaseChronosPipeline.from_pretrained(xc.REPO, device_map='cuda', torch_dtype=torch.float32)
    temp_zs = xc.predict_at(zs, 'temperature', 'raw', times['temperature'])
    cols['temperature|ZS_raw'] = column(ep, temp_zs, temp_zs, 'temperature', 'raw')
    cols['oxygen|ZS_raw'] = column(ep, xc.predict_at(zs, 'oxygen', 'raw', times['oxygen']), temp_zs, 'oxygen', 'raw')
    for unit in UNITS:
        set_unit(unit)
        cols[f'oxygen|ZS_aou|{unit}'] = column(ep, xc.predict_at(zs, 'oxygen', 'aou', times['oxygen']),
                                               temp_zs, 'oxygen', 'aou')
        # C2: v2_chronos_c2.main for ('oxygen', 'aou'), in this unit, from the replayed stream.
        xc.RNG.bit_generator.state = rng_state
        X, y, when = v2_chronos_c2.samples(zs, 'oxygen', 'aou')
        order = np.argsort(when.asi8); cut = int(len(order) * .75); itr, iva = order[:cut], order[cut:]

        def objective(trial):
            m = make_pipeline(StandardScaler(), Ridge(alpha=trial.suggest_float('alpha', 1e-3, 1e6, log=True))).fit(X[itr], y[itr])
            return float(np.abs(m.predict(X[iva]) - y[iva]).mean())
        name = 'c2_oxygen_aou' if unit == 'kg' else 'c2_oxygen_aou_umolL'
        sel = run_study(name, objective, OUT / 'studies', QUICK)
        c2 = make_pipeline(StandardScaler(), Ridge(alpha=sel['params']['alpha'])).fit(X, y)
        base = xc.predict_at(zs, 'oxygen', 'aou', times['oxygen'])
        corrected = {}
        for k, v in base.items():
            D = xc.c2_design(k[0], 'oxygen', [k[1]])
            corrected[k] = v + float(c2.predict(D)[0]) if np.isfinite(D).all() else v
        cols[f'V2C2_oxygen_aou|{unit}'] = column(ep, corrected, temp_zs, 'oxygen', 'aou')
        report[f'c2_{unit}'] = {'selection': sel, 'samples': int(len(y)), 'inner_validation': int(len(iva))}
    del zs; torch.cuda.empty_cache()

    pipe = BaseChronosPipeline.from_pretrained(c2m.REPO, device_map='cuda', torch_dtype=torch.float32)
    for tag, use_cov in (('uni', False), ('cov', True)):
        temp = c2m.forecast(pipe, 'temperature', 'raw', use_cov, times['temperature'])
        if tag == 'uni':
            cols['CH2_oxygen_raw_uni'] = column(ep, c2m.forecast(pipe, 'oxygen', 'raw', False, times['oxygen']),
                                                temp, 'oxygen', 'raw')
        for unit in UNITS:
            set_unit(unit)
            cols[f'CH2_oxygen_aou_{tag}|{unit}'] = column(ep, c2m.forecast(pipe, 'oxygen', 'aou', use_cov, times['oxygen']),
                                                          temp, 'oxygen', 'aou')
    del pipe; torch.cuda.empty_cache()
    set_unit('kg')

    frame = ep[['variable', 'sensor', 'time']].copy()
    for k, v in cols.items():
        frame[k] = v
        report['columns'][k] = {'predictions': int(np.isfinite(v).sum()),
                                'mae_2026': mae(ep, v, 'temperature' if k.startswith('temperature') else 'oxygen')}
    frame.to_csv(OUT / 'zeroshot_columns.csv.gz', index=False)
    if not QUICK:
        report['gates'] = gates_zeroshot(ep, cols, report)
    report['seconds'] = round(time.time() - t0, 1)
    (OUT / 'zeroshot_report.json').write_text(json.dumps(report, indent=2, default=str))
    print(json.dumps({k: v for k, v in report.items() if k != 'columns'}, indent=2, default=str)[:4000], flush=True)
    for k, v in report['columns'].items():
        print(f'  {k:32s} {v["mae_2026"]:.4f}  n={v["predictions"]}', flush=True)


def gates_zeroshot(ep, cols, report):
    """G1 and G2 against the published columns, joined on (variable, sensor, time)."""
    key = ['variable', 'sensor', 'time']
    here = ep[key].copy()
    here['time'] = pd.to_datetime(here.time, utc=True)
    refs = {'oxygen|ZS_aou|kg': ('chronos_predictions.csv.gz', 'oxygen|ZS_aou'),
            'oxygen|ZS_raw': ('chronos_predictions.csv.gz', 'oxygen|ZS_raw'),
            'temperature|ZS_raw': ('chronos_predictions.csv.gz', 'temperature|ZS_raw'),
            'V2C2_oxygen_aou|kg': ('v2_chronos_c2_predictions.csv.gz', 'V2C2_oxygen_aou'),
            'CH2_oxygen_aou_uni|kg': ('v2_chronos2_predictions.csv.gz', 'CH2_oxygen_aou_uni'),
            'CH2_oxygen_aou_cov|kg': ('v2_chronos2_predictions.csv.gz', 'CH2_oxygen_aou_cov'),
            'CH2_oxygen_raw_uni': ('v2_chronos2_predictions.csv.gz', 'CH2_oxygen_raw_uni')}
    out = {}
    for mine, (fname, col) in refs.items():
        pub = pd.read_csv(REF / fname, usecols=key + [col])
        pub['time'] = pd.to_datetime(pub.time, utc=True)
        j = here.assign(mine=cols[mine]).merge(pub, on=key, how='left', validate='one_to_one')
        c = compare(j['mine'].to_numpy(float), j[col].to_numpy(float))
        variable = 'temperature' if mine.startswith('temperature') else 'oxygen'
        c.update(mae_here=mae(ep, cols[mine], variable), mae_published=mae(ep, j[col].to_numpy(float), variable),
                 reference=f'{fname}:{col}', reference_sha256=hashlib.sha256((REF / fname).read_bytes()).hexdigest())
        c['passed'] = bool(c['max_abs_diff'] is not None and c['max_abs_diff'] <= 1e-3 and c['same_missing']
                           and abs(c['mae_here'] - c['mae_published']) <= 1e-4)
        out[mine] = c
    pub_sel = json.loads((REF / 'v2_chronos_c2_report.json').read_text())['c2_oxygen_aou']['selection']
    here_sel = report['c2_kg']['selection']
    out['G2_selection'] = {'alpha_published': pub_sel['params']['alpha'], 'alpha_here': here_sel['params']['alpha'],
                           'passed': bool(abs(here_sel['params']['alpha'] / pub_sel['params']['alpha'] - 1) <= 1e-6)}
    return out


# ---------------------------------------------------------------- adapted arms

ADAPTED = ('ft_random', 'ft_posthoc', 'lora_random', 'lora_posthoc', 'tv_full', 'tv_lora')


def adapted(arm):
    """One adapted configuration, retrained in both units; the tv_* arms also give the addendum-43 refit."""
    import exp_chronos_valtest as vt
    import v2_chronos as v2c
    import v2_chronos_lora as lora
    import v2_chronos_timeval as tv
    import v2_chronos_refit as rf
    from chronos import BaseChronosPipeline
    modules = (vt, v2c, lora, tv, rf)
    t0 = time.time()
    ep = xc.ep
    times = eval_times(ep)
    zs = BaseChronosPipeline.from_pretrained(xc.REPO, device_map='cuda', torch_dtype=torch.float32)
    temp_zs = xc.predict_at(zs, 'temperature', 'raw', times['temperature']); del zs; torch.cuda.empty_cache()
    report = {'declared_in': 'config/bay_reserved_protocol_v2_addendum52.json', 'arm': arm, 'smoke_only': QUICK,
              'units': {}}
    if arm.startswith('ft_'):
        rep = json.loads((REF / 'v2_chronos_report_oxygen.json').read_text())['chronos_oxygen_aou']
        curve = pd.read_csv(REF / 'chronos_oxygen_aou_curve.csv')
        if arm == 'ft_random':
            number, params = rep['selection']['best_trial'], rep['selection']['params']
        else:
            row = curve.loc[curve.mae_2026_posthoc.idxmin()]
            number = int(row.trial)
            params = {'lr': float(row.lr), 'max_steps': int(row.max_steps), 'weight_decay': float(row.weight_decay),
                      'batch': int(row.batch)}
        published = float(curve.set_index('trial').loc[number, 'mae_2026_posthoc'])
    elif arm.startswith('lora_'):
        rep = json.loads((REF / 'v2_chronos_lora_report.json').read_text())['lora_oxygen_aou']
        number = rep['selection']['best_trial'] if arm == 'lora_random' else rep['best_posthoc']['trial']
        curve = pd.read_csv(REF / 'lora_oxygen_aou_curve.csv').set_index('trial')
        if arm == 'lora_random':
            params = dict(rep['selection']['params'])
        else:
            keys = ('lr', 'max_steps', 'weight_decay', 'batch', 'rank', 'alpha_over_rank', 'dropout', 'targets')
            params = {k: curve.loc[number, k] for k in keys}
            params = {k: (int(v) if k in ('max_steps', 'batch', 'rank', 'alpha_over_rank') else
                          (str(v) if k == 'targets' else float(v))) for k, v in params.items()}
        published = float(curve.loc[number, 'mae_2026_posthoc'])
        ref_cols = np.load(REF / 'lora_oxygen_aou_2026_columns.npz')
        report['published_column_sha256'] = hashlib.sha256(ref_cols[str(number)].tobytes()).hexdigest()
    else:
        method = arm.removeprefix('tv_')
        sel = json.loads((HERE / 'selections.json').read_text())[f'timeval_{method}_oxygen_aou']
        number, params, published = sel['best_trial'], sel['params'], sel['selected_mae_2026']
        refit_published = json.loads((REF / f'refit_timeval_{method}_oxygen_aou.json').read_text())
        report['refit_published'] = {'restored_step': refit_published['restored_step'],
                                     'mae_refit_2026': refit_published['mae_refit_2026']}
    report.update(trial=int(number), params=params, published_mae_2026=published)
    frame = ep[['variable', 'sensor', 'time']].copy()
    for unit in UNITS:
        set_unit(unit, modules)
        u = {}
        t1 = time.time()
        if arm.startswith('ft_'):
            # v2_chronos.main's windows and split, drawn in its order from the shared stream.
            vt.RNG = np.random.default_rng(20260919)
            w = {}
            for variable, space in (('temperature', 'raw'), ('oxygen', 'aou')):
                ctxs, tgts = vt.windows(variable, space)
                n_val = max(128, int(.1 * len(ctxs))); idx = vt.RNG.permutation(len(ctxs))
                w[variable] = (ctxs, tgts, idx[n_val:], idx[:n_val])
            ctxs, tgts, ti, vi = w['oxygen']
            torch.manual_seed(int(number))     # declared: v2_chronos.train leaves torch unseeded
            pipe, val = v2c.train(ctxs, tgts, ti, vi, rng=np.random.default_rng(int(number)), **params)
            col = v2c.to_column(pipe, 'oxygen', 'aou', times['oxygen'], temp_zs)
            u.update(val_loss=val)
        elif arm.startswith('lora_'):
            vt.RNG = np.random.default_rng(20260919)
            ctxs, tgts, ti, vi = lora.draw_windows()['oxygen']
            pipe, val, info = lora.train(ctxs, tgts, ti, vi, rng=np.random.default_rng(int(number)), seed=int(number), **params)
            col = lora.to_column(pipe, 'oxygen', 'aou', times['oxygen'], temp_zs)
            u.update(val_loss=val, **info)
        else:
            w = tv.windows('oxygen', 'aou')
            pipe, best, step, curve_ = rf.rerun(method, params, w, int(number))
            col = tv.to_column(pipe, 'oxygen', 'aou', times['oxygen'], temp_zs)
            del pipe; torch.cuda.empty_cache()
            steps = max(step, 1) if QUICK else step
            pipe_r, n_windows = rf.refit(method, params, w, int(number), steps)
            col_r = tv.to_column(pipe_r, 'oxygen', 'aou', times['oxygen'], temp_zs)
            del pipe_r; torch.cuda.empty_cache()
            frame[f'refit_{method}|{unit}'] = col_r
            u.update(val_loss=best, restored_step=step, refit_windows=n_windows,
                     mae_refit_2026=mae(ep, col_r), curve=curve_)
            pipe = None
        if pipe is not None:
            del pipe; torch.cuda.empty_cache()
        frame[f'{arm}|{unit}'] = col
        u.update(mae_2026=mae(ep, col), seconds=round(time.time() - t1, 1))
        report['units'][unit] = u
        print(f'{arm} {unit}: MAE {u["mae_2026"]:.4f} (published {published:.4f}) in {u["seconds"]} s', flush=True)
    set_unit('kg', modules)
    kg = report['units']['kg']
    report['G3'] = {'mae_diff': abs(kg['mae_2026'] - published),
                    'gated': not arm.startswith('ft_'),
                    'passed': bool(abs(kg['mae_2026'] - published) <= 1e-3)}
    if arm.startswith('tv_'):
        report['G3_refit'] = {'restored_step_published': report['refit_published']['restored_step'],
                              'restored_step_here': kg['restored_step'],
                              'mae_diff': abs(kg['mae_refit_2026'] - report['refit_published']['mae_refit_2026']),
                              'passed': bool(kg['restored_step'] == report['refit_published']['restored_step'] and
                                             abs(kg['mae_refit_2026'] - report['refit_published']['mae_refit_2026']) <= 1e-3)}
    report['seconds'] = round(time.time() - t0, 1)
    frame.to_csv(OUT / f'adapted_{arm}_columns.csv.gz', index=False)
    (OUT / f'adapted_{arm}_report.json').write_text(json.dumps(report, indent=2, default=str))
    print(json.dumps({k: v for k, v in report.items() if k != 'units'}, indent=2, default=str), flush=True)


if __name__ == '__main__':
    if sys.argv[1] == 'zeroshot':
        zeroshot()
    elif sys.argv[1] == 'adapted':
        arm = sys.argv[sys.argv.index('--arm') + 1]
        if arm not in ADAPTED:
            raise SystemExit(f'unknown arm {arm}')
        adapted(arm)
    else:
        raise SystemExit('zeroshot | adapted --arm <name>')
