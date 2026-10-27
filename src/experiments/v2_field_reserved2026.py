"""Addendum 36: the vertical prior read on the 38 casts of 2026 (Field-2026).

The one physical-guidance gain above a measurement floor in the thesis is the vertical prior on the
spatial temperature estimate (Field-57, 2025). The Copernicus 3-D extraction on disk ends on
2026-01-16, so the released recipe cannot be scored on 2026; the prior rebuilt from the project's
own casts (addendum 14) can. Two learned arms, each with its own 100-trial study on the published
development folds, frozen, then read once on 2026:

    A2_none    background = causal moored context; ridge residual on the context features
    A7o        A2 plus the addendum-14 annual-harmonic cast prior as a depth-dependent offset

plus both backgrounds alone and a seasonal climatology of the pre-2026 archive. The A2 study must
reproduce the published penalty and development error before 2026 is read.

Declared in config/bay_reserved_protocol_v2_addendum36.json.
Usage: python -m src.experiments.v2_field_reserved2026 [--quick]

`--inner-prior` (review R3, 2026-10-05): A7o's development study scores each inner station fold
against the cast prior rebuilt without that split's validation stations
(`v2_field_ctdprior.inner_split_priors`), where the published study read the prior from all the
checkpoint's training stations; the refit at the selected penalty uses that full prior, as before.
A2 has no prior and is re-run unchanged as the reproduction gate. Everything goes to
`runs/v2_field_reserved2026/innerprior/` (`innerprior/smoke/` with `--quick`). By default the mode
stops after development: 2026 was declared to be read once, so a second reading with the new
penalty is a replication on a reused cohort and runs only with `--read-2026`.
  python -m src.experiments.v2_field_reserved2026 --inner-prior [--read-2026]
"""
import argparse
import hashlib
import json

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

from src.data.bay_observations import read_export
from src.data.bay_profile_context import ROOT, TARGETS, context_features, load_profiles
from src.experiments.bay_profile_field import SCALE, prepare, prediction_frame
from src.experiments.v2_field import fit_predict, inner_score, matrices, nested_mae
from src.experiments.v2_field_ctdprior import BIN_M, archive, harmonic_prior
from src.experiments.v2_field_naive import harmonic
from src.tuning.protocol_v2 import run_study

OUT = ROOT / 'reports/bay_twin/runs/v2_field_reserved2026'
START = pd.Timestamp('2026-01-01', tz='UTC')
END = pd.Timestamp('2026-07-01', tz='UTC')
GAP = pd.Timedelta(hours=72)
PUBLISHED_A2 = {'alpha': 19368.571933405394, 'temp_c': 0.7639235729023137}
DRAWS = 2000
RNG_SEED = 20260923


# --- development: the two studies on the published Field-57 folds ---------------------------------

def dev_matrices(arm, bundles, contracts, data, full):
    """Per-bundle (background, targets, design). A7o adds the cast prior built, as in addendum 14,
    from archive casts at the bundle's training stations up to its last training time."""
    query_bin = np.floor(data.depth_m.to_numpy() / BIN_M).astype(int)
    query_doy = data.t_utc.dt.dayofyear.to_numpy()
    mats = []
    for b, c in zip(bundles, contracts):
        extras = np.zeros((len(data), 8))
        if arm == 'A7o':
            cutoff = pd.Timestamp(c['last_training'])
            cutoff = cutoff.tz_localize('UTC') if cutoff.tzinfo is None else cutoff
            train = full[full.station.isin(set(c['training_stations'])) & (full.t_utc <= cutoff)]
            extras[:, 3:6] = harmonic_prior(train, query_bin, query_doy)[0]
        mats.append(matrices(b, extras, arm == 'A7o', False, False))
    return mats


def dev_split_matrices(bundles, contracts, data, full, stations):
    """A7o's matrices per checkpoint and inner split, the prior rebuilt without the split's
    validation stations (review R3), and each split's coverage."""
    from src.experiments.v2_field_ctdprior import inner_split_mats
    query_bin = np.floor(data.depth_m.to_numpy() / BIN_M).astype(int)
    query_doy = data.t_utc.dt.dayofyear.to_numpy()
    return inner_split_mats(bundles, contracts, full, stations, lambda t: harmonic_prior(t, query_bin, query_doy),
                            np.zeros((len(data), 8)), lambda b, e: matrices(b, e, True, False, False))


def development(quick, inner_prior=False):
    cfg = json.loads((ROOT / 'config/bay_profile_field.json').read_text())
    bundles, contracts, _ = prepare('held_site', cfg, quick)
    data, _ = load_profiles()
    full = archive()
    stations = data.station.to_numpy()
    casts = data.cast_id.to_numpy()
    truth = data[TARGETS].to_numpy(float)
    out, preds = {}, {}
    for arm in ('A2_none', 'A7o'):
        mats = dev_matrices(arm, bundles, contracts, data, full)
        if inner_prior and arm == 'A7o':
            split_mats, coverage = dev_split_matrices(bundles, contracts, data, full, stations)

            def objective(trial, mats=mats, split_mats=split_mats):
                alpha = trial.suggest_float('alpha', .1, 1e8, log=True)
                return float(np.mean([inner_score(b, m, alpha, False, stations, casts, truth, split_mats=sm)
                                      for b, m, sm in zip(bundles, mats, split_mats)]))
            sel = run_study(f'held_site_{arm}_dev_innerprior', objective, OUT / 'studies', quick)
            out['A7o_inner_prior'] = {'objective': 'each inner split scored against the cast prior rebuilt '
                                                   'without its validation stations', 'coverage': coverage}
        else:
            def objective(trial, mats=mats):
                alpha = trial.suggest_float('alpha', .1, 1e8, log=True)
                return float(np.mean([inner_score(b, m, alpha, False, stations, casts, truth)
                                      for b, m in zip(bundles, mats)]))
            sel = run_study(f'held_site_{arm}_dev', objective, OUT / 'studies', quick)
        rows = []
        for b, (base, y, feats) in zip(bundles, mats):
            pred, _, _ = fit_predict(feats, base, y, b['tr'], b['va'], np.asarray(b['weights'], float),
                                     sel['params']['alpha'], False)
            rows.append(prediction_frame(b, pred))
        d = pd.concat(rows, ignore_index=True)
        preds[arm] = d
        out[arm] = {'selection': sel, 'nested_mae': nested_mae(d), 'casts': int(d.cast_id.nunique())}
    a2 = out['A2_none']
    out['A2_reproduces_published'] = bool(
        np.isclose(a2['selection']['params']['alpha'], PUBLISHED_A2['alpha'], rtol=1e-6)
        and np.isclose(a2['nested_mae']['temp_c'], PUBLISHED_A2['temp_c'], rtol=1e-6))
    return out, preds


# --- 2026: frozen penalties, read once ------------------------------------------------------------

def profiles_window(start, end):
    """load_profiles' filters on the read window; load_profiles itself is not changed."""
    d = pd.read_csv(ROOT / 'data/processed/ctd_all_casts.csv.gz')
    d['t_utc'] = pd.to_datetime(d.t_utc, utc=True)
    d = d[(d.t_utc >= start) & (d.t_utc < end)].copy()
    audit = pd.read_csv(ROOT / 'reports/bay_twin/gen2_profile_inventory.csv')
    d = d[~d.cast_id.isin(audit.loc[audit.depth_exceeds_metadata_by_gt_1m, 'cast_id'])]
    d = d[np.isfinite(d[TARGETS]).all(axis=1) & (d.qc_flag == 0)]
    d = d[d.temp_c.between(-2, 40) & d.salinity_psu.between(0, 42) & d.do_umol_l.between(0, 600)]
    return d.sort_values(['t_utc', 'cast_id', 'depth_m']).reset_index(drop=True)


def observations_through(end):
    """The moored record through the end of the read; load_observations stops at 2026 by design."""
    meta = pd.read_csv(ROOT / 'reports/bay_twin/sensor_geometry.csv').set_index('sensor')
    frames = {}
    for name in meta.index:
        _, frame, _ = read_export(ROOT / 'data/raw' / (name + '.txt'), include_auxiliary=True)
        frames[name] = frame[frame.index < end]
    return meta, frames


def seasonal(train_rows, query):
    """Annual harmonic per 1 m bin of the absolute values; depth-pooled fallback (as naive_seasonal)."""
    day = train_rows.t_utc.dt.dayofyear.to_numpy()
    depth = np.floor(train_rows.depth_m.to_numpy()).astype(int)
    y = train_rows[TARGETS].to_numpy(float)

    def solve(mask):
        return np.linalg.lstsq(harmonic(day[mask]), y[mask], rcond=None)[0]
    pooled = solve(np.ones(len(y), bool))
    fits = {}
    for b in np.unique(depth):
        m = depth == b
        fits[b] = solve(m) if m.sum() >= 6 and len(np.unique(day[m])) >= 3 else pooled
    qd = np.floor(query.depth_m.to_numpy()).astype(int)
    coef = np.stack([fits.get(b, pooled) for b in qd])
    return np.einsum('ij,ijk->ik', harmonic(query.t_utc.dt.dayofyear.to_numpy()), coef)


def reserved(alphas, held_out=True, start=START, end=END):
    """The frozen arms on the casts of [start, end). The real read is 2026; a smoke run passes a
    2025 window, which the development folds have already used, so 2026 is read only once."""
    pre, _ = load_profiles()
    pre = pre[pre.t_utc < start]
    new = profiles_window(start, end)
    data = pd.concat([pre, new], ignore_index=True)
    meta, frames = observations_through(end)
    full = archive()
    y = data[TARGETS].to_numpy(float)
    query_bin = np.floor(data.depth_m.to_numpy() / BIN_M).astype(int)
    query_doy = data.t_utc.dt.dayofyear.to_numpy()
    held = sorted(new.station.unique()) if held_out else [None]
    frames_out = []
    for s in held:
        X = context_features(data, meta, frames, s)
        test_station = new.station.unique() if s is None else [s]
        tr = np.flatnonzero((data.t_utc < start - GAP).to_numpy() & (data.station != s).to_numpy())
        te = np.flatnonzero((data.t_utc >= start).to_numpy() & data.station.isin(test_station).to_numpy())
        fallback = np.median(y[tr], axis=0)
        base = X[['temperature_context', 'salinity_context', 'oxygen_context']].to_numpy().copy()
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
        clim_rows = full[(full.station != s) & (full.t_utc < start)] if s is not None else full[full.t_utc < start]
        frame = data.iloc[te][['cast_id', 'station', 't_utc', 'depth_m'] + TARGETS].reset_index(drop=True)
        frame['fold'] = 0
        for arm, offset in (('A2_none', 0.0), ('A7o', delta)):
            bs = base + offset
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
    return pd.concat(frames_out, ignore_index=True)


# --- metric and intervals -------------------------------------------------------------------------

def cast_errors(d, arm, target):
    e = (d[target] - d[f'{arm}|{target}'] if arm else d[target] - d['pred_' + target]).abs()
    return d.assign(e=e).groupby(['fold', 'station', 'cast_id']).agg(
        e=('e', 'mean'), day=('t_utc', lambda x: x.iloc[0].floor('D'))).reset_index()


def macro(c):
    return float(c.groupby(['fold', 'station']).e.mean().groupby('fold').mean().mean())


def contrast(ca, cb, draws=DRAWS):
    """a - b, paired over cast dates: bootstrap with each drawn date relabelled, and jackknife-t."""
    j = ca.merge(cb[['fold', 'station', 'cast_id', 'e']], on=['fold', 'station', 'cast_id'], suffixes=('_a', '_b'))
    point = macro(j.rename(columns={'e_a': 'e'})) - macro(j.rename(columns={'e_b': 'e'}))
    days = j.day.unique()
    groups = {d: g for d, g in j.groupby('day')}
    rng = np.random.default_rng(RNG_SEED)
    boot = []
    for _ in range(draws):
        pick = rng.choice(days, len(days), replace=True)
        s = pd.concat([groups[d].assign(cast_id=groups[d].cast_id + f'#{k}') for k, d in enumerate(pick)])
        boot.append(macro(s.rename(columns={'e_a': 'e'})) - macro(s.rename(columns={'e_b': 'e'})))
    lo, hi = np.percentile(boot, [2.5, 97.5])
    jack = []
    for d in days:
        s = j[j.day != d]
        jack.append(macro(s.rename(columns={'e_a': 'e'})) - macro(s.rename(columns={'e_b': 'e'})))
    n = len(days)
    se = np.sqrt((n - 1) / n * np.sum((np.array(jack) - np.mean(jack)) ** 2))
    t = stats.t.ppf(.975, n - 1)
    return {'point': round(point, 4), 'ci': [round(float(lo), 4), round(float(hi), 4)],
            'excludes_zero': bool(lo > 0 or hi < 0), 'jackknife_t': [round(point - t * se, 4), round(point + t * se, 4)],
            'jackknife_excludes_zero': bool(point - t * se > 0 or point + t * se < 0), 'dates': int(n)}


def main(quick=False, inner_prior=False, read_2026=True):
    """The addendum-36 study. `inner_prior` (module docstring) writes under OUT/'innerprior' and
    reads 2026 only if `read_2026`; the default reads it, as published."""
    global OUT
    if inner_prior:
        OUT = OUT / 'innerprior'
    if quick:
        OUT = OUT / 'smoke'
    OUT.mkdir(parents=True, exist_ok=True)
    dev, dev_preds = development(quick, inner_prior) if inner_prior else development(quick)
    report = {'declared_in': 'config/bay_reserved_protocol_v2_addendum36.json', 'smoke_only': quick,
              'development': dev}
    if inner_prior:
        report['review'] = 'R3 (2026-10-05): A7o selected with the cast prior rebuilt per inner split'
    for arm, d in dev_preds.items():
        d.to_csv(OUT / f'dev_{arm}_predictions.csv.gz', index=False)
    report['development']['A7o_minus_A2'] = {
        t: contrast(cast_errors(dev_preds['A7o'], None, t), cast_errors(dev_preds['A2_none'], None, t))
        for t in TARGETS}
    print('development', json.dumps({a: dev[a]['nested_mae'] for a in ('A2_none', 'A7o')}), flush=True)
    print('A2 reproduces the published study:', dev['A2_reproduces_published'], flush=True)
    if not dev['A2_reproduces_published'] and not quick:
        (OUT / 'report.json').write_text(json.dumps(report, indent=2, default=str))
        raise SystemExit('A2 did not reproduce the published study; 2026 not read (addendum 36).')
    if inner_prior and not read_2026:
        report['read_window'] = None
        report['note'] = '2026 not read: run with --read-2026 for a second reading with the new A7o penalty'
        (OUT / 'report.json').write_text(json.dumps(report, indent=2, default=str))
        print('development only; 2026 not read', flush=True)
        return
    alphas = {a: dev[a]['selection']['params']['alpha'] for a in ('A2_none', 'A7o')}
    # A smoke run exercises the read on the second half of 2025, never on 2026.
    window = (pd.Timestamp('2025-07-01', tz='UTC'), START) if quick else (START, END)
    report['read_window'] = [str(window[0]), str(window[1])]
    for label, held_out in (('held_station', True), ('station_seen', False)):
        d = reserved(alphas, held_out, *window)
        path = OUT / f'reserved2026_{label}_predictions.csv.gz'
        d.to_csv(path, index=False)
        arms = ['A2_none', 'A7o', 'background_A2_none', 'background_A7o', 'seasonal']
        block = {'casts': int(d.cast_id.nunique()), 'dates': int(d.t_utc.dt.floor('D').nunique()),
                 'stations': d.groupby('station').cast_id.nunique().to_dict(),
                 'prediction_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
                 'nested_mae': {a: {t: round(macro(cast_errors(d, a, t)), 4) for t in TARGETS} for a in arms},
                 'contrasts': {}}
        for a, b in (('A7o', 'A2_none'), ('A7o', 'seasonal'), ('A2_none', 'seasonal'),
                     ('background_A7o', 'A7o'), ('background_A2_none', 'A2_none'),
                     ('background_A7o', 'background_A2_none')):
            block['contrasts'][f'{a} - {b}'] = {t: contrast(cast_errors(d, a, t), cast_errors(d, b, t)) for t in TARGETS}
        block['per_station_A7o_minus_A2'] = {
            t: {s: round(float(g[f'A7o_e'].mean() - g['A2_e'].mean()), 4) for s, g in
                cast_errors(d, 'A7o', t).rename(columns={'e': 'A7o_e'}).merge(
                    cast_errors(d, 'A2_none', t).rename(columns={'e': 'A2_e'}),
                    on=['fold', 'station', 'cast_id', 'day']).groupby('station')}
            for t in TARGETS}
        report[label] = block
        print(label, json.dumps(block['nested_mae']), flush=True)
    h = report['held_station']['contrasts']
    report['predictions'] = {
        'P1': h['A7o - A2_none']['temp_c']['point'] < 0 and h['A7o - A2_none']['temp_c']['excludes_zero'],
        'P1b': h['A7o - A2_none']['temp_c']['point'] <= -0.10,
        'P2': h['A7o - A2_none']['do_umol_l']['point'] > 0,
        'P3': h['A7o - A2_none']['salinity_psu']['point'] < 0,
        'P4': h['A7o - seasonal']['temp_c']['point'] < 0,
        'P5': abs(h['background_A7o - A7o']['temp_c']['point']) <= 0.03}
    (OUT / 'report.json').write_text(json.dumps(report, indent=2, default=str))
    print(json.dumps(report['predictions']), flush=True)
    print(json.dumps(h['A7o - A2_none'], indent=1), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--quick', action='store_true')
    p.add_argument('--inner-prior', action='store_true',
                   help='A7o selected with the cast prior rebuilt per inner split (writes innerprior/)')
    p.add_argument('--read-2026', action='store_true',
                   help='with --inner-prior: also read 2026 at the new penalty (a replication on a reused cohort)')
    a = p.parse_args()
    if a.inner_prior:
        main(a.quick, inner_prior=True, read_2026=a.read_2026)
    else:
        main(a.quick)
