"""What the apparent-utilisation arms of the year-boundary comparison are worth once two faults in
their construction are removed.

The spatial and near-bottom experiments (scripts/experiments/exp_spatial.py, exp_bottom.py) compute
Garcia-Gordon solubility with the umol/kg coefficients (A0 = 5.80871) and subtract a concentration
in umol/L, so their target is about 0.973 C* - DO rather than C* - DO. They also evaluate that
solubility, both to form the target and to invert the prediction, at the cast's own measured
temperature and salinity at the scored depth, which the direct arm is never given. This reruns the
ridge arms of the year-boundary comparison on the published rows, with the published tuning (one
100-trial study per arm on the latest quarter of training casts), in four forms:

    direct_do      published direct arm (gate: must reproduce the stored predictions)
    aou            published utilisation arm, umol/kg solubility (gate: must reproduce)
    aou_l          utilisation with the umol/L solubility of src/data/thermodynamics.py
    direct_do_ts   the direct arm given the measured temperature and salinity as two more features

aou_l - aou is the unit fault; aou_l - direct_do is the published comparison done in one unit;
aou_l - direct_do_ts compares the transformation with a direct model holding the same information.
Intervals as in scripts/v2_summarize_aou.py: cast dates relabelled, delete-one-date jackknife.

For the pretrained arms that forecast utilisation at 38 PSU (exp_chronos.py and the v2_chronos*
scripts share the same umol/kg function; the utilisation tree of v2_aou_forecast uses umol/L and is
not affected) it measures two things on the oxygen sondes' own 2026 hourly temperatures, without
rerunning them: how far the fault moves a six-hour forecast that persists utilisation and inverts
at the target-hour temperature (0.027 times the six-hour change in solubility), and how far it
shifts the level of the utilisation series the pretrained model reads, against that series' own
mean absolute value. A pretrained model that scales its context by its mean absolute value is not
invariant to such a shift, so the second number is a caution, not a bound.

Diagnostic, not a declared addendum. Usage: python -m scripts.v2_aou_units
"""
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.v2_summarize_aou import cast_errors, cast_meta, contrast, summarise
from src.data.thermodynamics import o2sol_garcia_gordon_umol_per_l as solubility_l
from src.experiments import v2_bottom_aou as vr
from src.experiments.v2_field_reserved2026 import macro

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'reports/bay_twin/runs/v2_bottom_aou/aou_units'
PUBLISHED = ROOT / 'reports/bay_twin/runs/v2_bottom_aou/aou/predictions.csv.gz'
HOURLY = ROOT / 'reports/bay_twin/hourly_observations.csv.gz'
CONTRASTS = [('aou', 'direct_do'), ('aou_l', 'aou'), ('aou_l', 'direct_do'),
             ('direct_do_ts', 'direct_do'), ('aou_l', 'direct_do_ts')]


def arms(exp_spatial):
    feats = list(exp_spatial.FEATS)
    return {'direct_do': (feats, 'do_umol_l', lambda p, x: p),
            'aou': (feats, 'aou', lambda p, x: x.sol.to_numpy(float) - p),
            'aou_l': (feats, 'aou_l', lambda p, x: x.sol_l.to_numpy(float) - p),
            'direct_do_ts': (feats + ['temp_c', 'salinity_psu'], 'do_umol_l', lambda p, x: p)}


def forecast_bound():
    """The unit fault on the oxygen sondes' 2026 hourly record, at 38 PSU as the pretrained arms use."""
    obs = pd.read_csv(HOURLY, index_col=0)
    obs.index = pd.to_datetime(obs.index, utc=True)
    probes = [c.split('::')[0] for c in obs.columns if c.endswith('::oxygen')]
    out, pooled = {}, []
    for p in probes:
        frame = obs[[f'{p}::temperature', f'{p}::oxygen']]
        frame = frame[frame.index >= pd.Timestamp('2026-01-01', tz='UTC')].asfreq('1h')
        t, do = frame.iloc[:, 0].to_numpy(float), frame.iloc[:, 1].to_numpy(float)
        kg, lit = solubility_kg(t), solubility_l(t, 38.0)
        fault = pd.Series(kg - lit, index=frame.index)
        change = (fault.shift(-6) - fault).abs().dropna()
        u_l, u_kg = lit - do, kg - do
        ok = np.isfinite(u_l)
        if len(change) and ok.any():
            out[p] = {'hours': int(len(change)), 'six_hour_change_mean': round(float(change.mean()), 4),
                      'six_hour_change_p99': round(float(change.quantile(.99)), 4),
                      'six_hour_change_max': round(float(change.max()), 4),
                      'utilisation_hours': int(ok.sum()),
                      'level_shift_mean': round(float(np.mean(u_kg[ok] - u_l[ok])), 3),
                      'utilisation_l_mean': round(float(np.mean(u_l[ok])), 3),
                      'utilisation_l_mean_abs': round(float(np.mean(np.abs(u_l[ok]))), 3),
                      'utilisation_kg_mean': round(float(np.mean(u_kg[ok])), 3),
                      'utilisation_kg_mean_abs': round(float(np.mean(np.abs(u_kg[ok]))), 3)}
            pooled.append(change.to_numpy())
    allc = np.concatenate(pooled)
    ratio = solubility_kg(np.array([13., 28.])) / solubility_l(np.array([13., 28.]), 38.0)
    return {'what': 'six_hour_change: absolute change over six hours of (umol/kg solubility - umol/L '
                    'solubility) at 38 PSU, the amount by which the unit fault moves a six-hour forecast '
                    'that persists utilisation and inverts at the target-hour temperature; level_shift: '
                    'how far the fault moves the utilisation series a pretrained arm reads, against '
                    'that series\' mean and mean absolute value (umol/L); 2026 hourly record of each '
                    'oxygen sonde',
            'kg_over_l_at_13_and_28C': [round(float(r), 4) for r in ratio],
            'pooled_six_hour_change': {'hours': int(len(allc)), 'mean': round(float(allc.mean()), 4),
                                       'p99': round(float(np.quantile(allc, .99)), 4),
                                       'max': round(float(allc.max()), 4)},
            'by_probe': out}


def solubility_kg(T, S=38.0):
    """The function the review experiments use (Garcia and Gordon 1992, Benson-Krause fit, umol/kg)."""
    Ts = np.log((298.15 - T) / (273.15 + T))
    A = [5.80871, 3.20291, 4.17887, 5.10006, -9.86643e-2, 3.80369]
    B = [-7.01577e-3, -7.70028e-3, -1.13864e-2, -9.51519e-3]
    lnC = (A[0] + A[1] * Ts + A[2] * Ts ** 2 + A[3] * Ts ** 3 + A[4] * Ts ** 4 + A[5] * Ts ** 5
           + S * (B[0] + B[1] * Ts + B[2] * Ts ** 2 + B[3] * Ts ** 3) - 2.75915e-7 * S ** 2)
    return np.exp(lnC)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    _, exp_spatial = vr.import_experiments(OUT)
    if not np.allclose(exp_spatial.o2_solubility(np.array([18.]), np.array([38.])), solubility_kg(18.)):
        raise SystemExit('exp_spatial solubility is not the function reproduced here')
    c = exp_spatial.featurise(exp_spatial.load())
    c['sol_l'] = solubility_l(c.temp_c.to_numpy(float), c.salinity_psu.to_numpy(float))
    c['aou_l'] = c.sol_l - c.do_umol_l
    published = pd.read_csv(PUBLISHED)
    spec = arms(exp_spatial)
    report = {'what': __doc__.split('\n\n')[0], 'diagnostic': True, 'gates': {}, 'mae': {}, 'contrasts': [],
              'solubility_offset_on_scored_rows': {}}
    preds = []
    meta = cast_meta()
    for cname, cdf in {'gen2_164': c[c.t_utc >= '2024-07-24'], 'full_722': c}.items():
        for fold, tr, te in exp_spatial.folds(cdf, 'P2_chronological'):
            tr, te = tr.reset_index(drop=True), te.reset_index(drop=True)
            off = (te.sol - te.sol_l).to_numpy(float)
            report['solubility_offset_on_scored_rows'][cname] = {
                'kg_minus_l_mean': round(float(off.mean()), 3), 'min': round(float(off.min()), 3),
                'max': round(float(off.max()), 3), 'ratio_mean': round(float((te.sol / te.sol_l).mean()), 4)}
            errors = {}
            for arm, (cols, ycol, back) in spec.items():
                err = (lambda p, x, back=back: back(p, x) - x.do_umol_l.to_numpy(float))
                p, sel = vr.tuned_predict('ridge', tr, te, cols, ycol, err, 'P2_chronological', OUT / 'studies',
                                          f'units_{cname}_ridge_{arm}', False)
                do = back(p, te)
                frame = pd.DataFrame(dict(cohort=cname, space=arm, model='M2_ridge_tel', cast_id=te.cast_id.values,
                                          truth=te.do_umol_l.values, pred=do))
                preds.append(frame)
                errors[arm] = cast_errors(frame, meta, '2026')
                report['mae'][f'{cname}|{arm}'] = {'nested': round(macro(errors[arm]), 4),
                                                   'alpha': sel['params']['alpha'],
                                                   'completed_trials': sel['completed_trials']}
                if arm in ('direct_do', 'aou'):
                    ref = published[(published.cohort == cname) & (published.model == 'M2_ridge_tel')
                                     & (published.space == arm)]
                    same = ref.cast_id.to_numpy().tolist() == frame.cast_id.to_numpy().tolist()
                    gap = float(np.max(np.abs(ref.pred.to_numpy() - do))) if same else None
                    report['gates'][f'{cname}|{arm}'] = {'rows_match': same, 'max_abs_diff': gap,
                                                         'passed': bool(same and gap < 1e-6)}
                print(cname, arm, report['mae'][f'{cname}|{arm}'], flush=True)
            for a, b in CONTRASTS:
                entry = {'cohort': cname, 'a': a, 'b': b, **summarise(contrast(errors[a], errors[b]))}
                report['contrasts'].append(entry)
                print(f"{cname} {a} - {b}: {entry['diff']:+.3f} {entry['ci']} jk {entry['jackknife_t']}"
                      f"{'  robust' if entry['robust'] else ''}", flush=True)
    path = OUT / 'predictions.csv.gz'
    # mtime 0 so the gzip header, and with it the recorded digest, does not change between runs
    pd.concat(preds, ignore_index=True).to_csv(path, index=False, compression={'method': 'gzip', 'mtime': 0})
    report['prediction_sha256'] = hashlib.sha256(path.read_bytes()).hexdigest()
    report['gates_passed'] = bool(all(g['passed'] for g in report['gates'].values()))
    report['forecasting_arms'] = forecast_bound()
    (OUT / 'summary.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report['gates'], indent=1), json.dumps(report['forecasting_arms'], indent=1))


if __name__ == '__main__':
    main()
