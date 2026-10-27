"""F. Do the unused biological channels carry information about future oxygen?

A screen, not the protocol: plain ridge and a small untuned LightGBM, leave-one-month-out, on the
6 h and 24 h oxygen change at the two sondes with the longest records. Base features mimic what
the released tree already sees from the sonde itself (current O2, its recent change, T, S, hour and
season harmonics); the augmented set adds chlorophyll, pH and turbidity (level and 6 h change).
The quantity reported is the change in held-out MAE, with a month-block bootstrap interval.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from eda_lib import OUT, load

RNG = np.random.default_rng(20260923)


def frame(name):
    f, _ = load(name + '.txt')
    f = f.rename(columns={'Ossigeno(umol/l)': 'O2', 'Temperatura(°C)': 'T', "Salinita'(PSU)": 'S',
                          'Clorofilla-a(ug/l)': 'chl', "Torbidita'(FNU)": 'turb', 'pH_T': 'pH'})
    f = f.loc['2024-07-24':'2025-12-31']  # development years only; 2026 stays unread here
    x = pd.DataFrame(index=f.index)
    x['O2'] = f.O2
    x['dO2_6'] = f.O2.diff(6)
    x['dO2_24'] = f.O2.diff(24)
    x['T'], x['S'] = f['T'], f.S
    x['dT_6'], x['dS_6'] = f['T'].diff(6), f.S.diff(6)
    h = f.index.hour
    doy = f.index.dayofyear
    x['hs'], x['hc'] = np.sin(2 * np.pi * h / 24), np.cos(2 * np.pi * h / 24)
    x['ds'], x['dc'] = np.sin(2 * np.pi * doy / 365.25), np.cos(2 * np.pi * doy / 365.25)
    # night-time chlorophyll only: daytime fluorescence is quenched, so carry the last night value
    chl_night = f.chl.where(f.index.hour.isin(range(0, 5)))
    x['chl'] = chl_night.ffill(limit=24)
    x['chl_log'] = np.log1p(x.chl)
    x['pH'], x['dpH_6'] = f.pH, f.pH.diff(6)
    x['turb'] = f.turb
    return f, x


BASE = ['O2', 'dO2_6', 'dO2_24', 'T', 'S', 'dT_6', 'dS_6', 'hs', 'hc', 'ds', 'dc']
BIO = ['chl_log', 'pH', 'dpH_6', 'turb']


def cv(x, y, cols, model):
    d = pd.concat([x[cols], y.rename('y')], axis=1).dropna()
    month = d.index.to_period('M')
    err = pd.Series(index=d.index, dtype=float)
    for mo in month.unique():
        te = month == mo
        tr = ~te
        if te.sum() < 48 or tr.sum() < 500:
            continue
        if model == 'ridge':
            from sklearn.linear_model import RidgeCV
            from sklearn.pipeline import make_pipeline
            from sklearn.preprocessing import StandardScaler
            m = make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-2, 4, 20)))
        else:
            from lightgbm import LGBMRegressor
            m = LGBMRegressor(n_estimators=300, learning_rate=0.03, num_leaves=15, min_child_samples=50,
                              subsample=0.8, subsample_freq=1, colsample_bytree=0.8, n_jobs=4, verbose=-1)
        m.fit(d.loc[tr, cols], d.loc[tr, 'y'])
        err[te] = (m.predict(d.loc[te, cols]) - d.loc[te, 'y']).abs()
    return err.dropna()


out = {}
for name in ('TBaiaP_01m-CTD(calibrato)', 'TPortovenereS_02m-CTD(calibrato)', 'TMitiliE_05m-CTD(calibrato)'):
    f, x = frame(name)
    for h in (6, 24):
        y = f.O2.shift(-h) - f.O2
        rows = pd.concat([x[BASE + BIO], y.rename('y')], axis=1).dropna().index  # same rows for both
        xs, ys = x.loc[rows], y.loc[rows]
        for model in ('ridge', 'lgbm'):
            e0 = cv(xs, ys, BASE, model)
            e1 = cv(xs, ys, BASE + BIO, model)
            j = pd.concat([e0.rename('b'), e1.rename('a')], axis=1).dropna()
            month = j.index.to_period('M')
            per = j.groupby(month).mean()
            diffs = [(per.a - per.b).sample(len(per), replace=True, random_state=int(RNG.integers(1e9))).mean()
                     for _ in range(1000)]
            key = f"{name.split('_')[0]}|h{h}|{model}"
            out[key] = {'rows': int(len(j)), 'months': int(len(per)),
                        'mae_base': round(float(per.b.mean()), 3), 'mae_bio': round(float(per.a.mean()), 3),
                        'delta': round(float((per.a - per.b).mean()), 3),
                        'ci': [round(float(q), 3) for q in np.percentile(diffs, [2.5, 97.5])]}
            print(key, out[key], flush=True)
(OUT / 'eda_f_bio_features.json').write_text(json.dumps(out, indent=1))
