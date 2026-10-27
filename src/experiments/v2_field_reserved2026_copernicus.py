"""Addendum 38: the released Copernicus recipe read on the 38 casts of 2026.

Addendum 36 read the cast-derived prior on Field-2026 because the Copernicus 3-D extraction stopped
on 2026-01-16. With the product downloaded for the whole reserved half-year, the released recipe
(A1: offset in the background and values as regressors) and its offset-only form (A4) are read on
the same rows, with their published penalties, beside the two addendum-36 arms.

Three gates run before 2026 is scored with a Copernicus arm:
    G1  this module's boundary extraction reproduces profile_boundary_context.csv.gz pre-2026
    G2  A1 and A4 reproduce their published Field-57 nested MAE with their published penalties
    G3  A2 and A7o reproduce the addendum-36 Field-2026 nested MAE

Declared in config/bay_reserved_protocol_v2_addendum38.json.
Usage: python -m src.experiments.v2_field_reserved2026_copernicus
"""
import glob
import hashlib
import json

import numpy as np
import pandas as pd
import xarray as xr
from pyproj import Geod
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

from src.data.bay_profile_context import ROOT, TARGETS, context_features, load_profiles
from src.experiments.bay_field_export import EXTRAS
from src.experiments.bay_profile_field import prepare, prediction_frame
from src.experiments.v2_field import fit_predict, matrices, nested_mae
from src.experiments.v2_field_ctdprior import BIN_M, archive, harmonic_prior
from src.experiments.v2_field_reserved2026 import (END, GAP, START, cast_errors, contrast, macro,
                                                    observations_through, profiles_window, seasonal)

OUT = ROOT / 'reports/bay_twin/runs/v2_field_reserved2026_copernicus'
FITTED_ON = 'data/external/Copernicus/T2023_2026'
NEW = 'data/external/Copernicus/T2026'
PRODUCTS = (('phy-tem', 'thetao'), ('phy-sal', 'so'), ('bgc-bio', 'o2'))
PUBLISHED = {'A1_both': {'alpha': 31745.55331767915, 'temp_c': 0.5736325712127146},
             'A4_base_only': {'alpha': 26627.628008373962, 'temp_c': 0.5752518148214398}}
ADDENDUM36 = ROOT / 'reports/bay_twin/runs/v2_field_reserved2026/report.json'
GEOD = Geod(ellps='WGS84')


def products(folder):
    out = []
    for token, var in PRODUCTS:
        paths = glob.glob(str(ROOT / folder / f'*{token}*.nc'))
        if len(paths) != 1:
            raise ValueError(f'expected one {token} file in {folder}, found {len(paths)}')
        out.append((xr.open_dataset(paths[0]), var))
    return out


def extract(prods, lon, lat, depths, stamp):
    """The rule of bay_field_interface.boundary: field at or before t - 48 h, 48-72 h old."""
    cutoff = (stamp - pd.Timedelta(hours=48)).tz_localize(None).to_datetime64()
    absolute, delta, details = [], [], []
    for ds, var in prods:
        p = ds[var].sel(time=cutoff, method='pad').sel(latitude=lat, longitude=lon, method='nearest')
        t = pd.Timestamp(p.time.values).tz_localize('UTC')
        if not pd.Timedelta(hours=48) <= stamp - t <= pd.Timedelta(hours=72):
            raise ValueError('stale')
        z = ds.depth.to_numpy().astype(float)
        v = p.to_numpy().astype(float)
        a = np.interp(depths, z, v)
        absolute.append(a)
        delta.append(a - np.interp(3., z, v))
        details.append((GEOD.inv(lon, lat, float(p.longitude), float(p.latitude))[2], z.min(), z.max()))
    dist, zmin, zmax = details[0]
    clamped = ((np.asarray(depths) < zmin) | (np.asarray(depths) > zmax)).astype(float)
    return np.column_stack(absolute + delta + [np.full(len(depths), dist), clamped])


def boundary(data, prods):
    """EXTRAS for every row of `data`; NaN for a cast whose forcing is stale."""
    out = np.full((len(data), len(EXTRAS)), np.nan)
    stale = []
    for cast, idx in data.groupby('cast_id', sort=False).groups.items():
        g = data.loc[idx]
        try:
            out[data.index.get_indexer(idx)] = extract(prods, float(g.lon.iloc[0]), float(g.lat.iloc[0]),
                                                       g.depth_m.to_numpy(), g.t_utc.iloc[0])
        except (ValueError, KeyError):
            stale.append(cast)
    return out, stale


def gate_extraction():
    """G1: the published context, rebuilt from the files it was fitted on."""
    data, _ = load_profiles()
    published = data[['cast_id', 'depth_m']].merge(
        pd.read_csv(ROOT / 'reports/bay_twin/profile_boundary_context.csv.gz'),
        on=['cast_id', 'depth_m'], validate='one_to_one')[EXTRAS].to_numpy(float)
    rebuilt, stale = boundary(data, products(FITTED_ON))
    diff = np.nanmax(np.abs(rebuilt - published), axis=0)
    return {'rows': int(len(data)), 'stale_casts': stale,
            'max_abs_diff': {c: float(d) for c, d in zip(EXTRAS, diff)},
            'passed': bool(not stale and np.all(diff < 1e-6))}


def gate_development():
    """G2: A1 and A4 with their published penalties reproduce their published Field-57 error."""
    cfg = json.loads((ROOT / 'config/bay_profile_field.json').read_text())
    bundles, _, _ = prepare('held_site', cfg, False)
    data, _ = load_profiles()
    extras = data[['cast_id', 'depth_m']].merge(
        pd.read_csv(ROOT / 'reports/bay_twin/profile_boundary_context.csv.gz'),
        on=['cast_id', 'depth_m'], validate='one_to_one')[EXTRAS].to_numpy(float)
    out = {}
    for arm, design in (('A1_both', True), ('A4_base_only', False)):
        rows = []
        for b in bundles:
            base, y, feats = matrices(b, extras, True, design, False)
            pred, _, _ = fit_predict(feats, base, y, b['tr'], b['va'], np.asarray(b['weights'], float),
                                     PUBLISHED[arm]['alpha'], False)
            rows.append(prediction_frame(b, pred))
        mae = nested_mae(pd.concat(rows, ignore_index=True))
        out[arm] = {'nested_mae': mae, 'published_temp_c': PUBLISHED[arm]['temp_c'],
                    'passed': bool(abs(mae['temp_c'] - PUBLISHED[arm]['temp_c']) < 1e-6)}
    return out


def read(held_out=True):
    pre, _ = load_profiles()
    new = profiles_window(START, END)
    data = pd.concat([pre, new], ignore_index=True)
    published = pre[['cast_id', 'depth_m']].merge(
        pd.read_csv(ROOT / 'reports/bay_twin/profile_boundary_context.csv.gz'),
        on=['cast_id', 'depth_m'], validate='one_to_one')[EXTRAS].to_numpy(float)
    fresh, stale = boundary(new.reset_index(drop=True), products(NEW))
    extras = np.vstack([published, fresh])
    meta, frames = observations_through(END)
    full = archive()
    y = data[TARGETS].to_numpy(float)
    qbin = np.floor(data.depth_m.to_numpy() / BIN_M).astype(int)
    qdoy = data.t_utc.dt.dayofyear.to_numpy()
    alphas = {'A2_none': json.loads(ADDENDUM36.read_text())['development']['A2_none']['selection']['params']['alpha'],
              'A7o': json.loads(ADDENDUM36.read_text())['development']['A7o']['selection']['params']['alpha'],
              'A1_both': PUBLISHED['A1_both']['alpha'], 'A4_base_only': PUBLISHED['A4_base_only']['alpha']}
    frames_out = []
    for s in (sorted(new.station.unique()) if held_out else [None]):
        X = context_features(data, meta, frames, s)
        tested = new.station.unique() if s is None else [s]
        tr = np.flatnonzero((data.t_utc < START - GAP).to_numpy() & (data.station != s).to_numpy())
        te = np.flatnonzero((data.t_utc >= START).to_numpy() & data.station.isin(tested).to_numpy()
                            & np.isfinite(extras).all(axis=1))
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
        cast_delta = harmonic_prior(prior_rows, qbin, qdoy)[0]
        clim_rows = full[(full.station != s) & (full.t_utc < START)] if s is not None else full[full.t_utc < START]
        ex = np.nan_to_num(extras)
        frame = data.iloc[te][['cast_id', 'station', 't_utc', 'depth_m'] + TARGETS].reset_index(drop=True)
        frame['fold'] = 0
        arms = {'A2_none': (0.0, sx), 'A7o': (cast_delta, sx),
                'A4_base_only': (ex[:, 3:6], sx), 'A1_both': (ex[:, 3:6], np.concatenate([sx, ex], axis=1))}
        for arm, (offset, feats) in arms.items():
            bs = base + offset
            bs[:, 0] = np.clip(bs[:, 0], -2, 40)
            bs[:, 1] = np.clip(bs[:, 1], 0, 42)
            pred, _, _ = fit_predict(feats, bs, y, tr, te, w, alphas[arm], False)
            for j, t in enumerate(TARGETS):
                frame[f'{arm}|{t}'] = pred[:, j]
        clim = seasonal(clim_rows, data.iloc[te])
        for j, t in enumerate(TARGETS):
            frame[f'seasonal|{t}'] = clim[:, j]
        frames_out.append(frame)
        print('held' if held_out else 'seen', s, 'test casts', frame.cast_id.nunique(), flush=True)
    return pd.concat(frames_out, ignore_index=True), stale


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    report = {'declared_in': 'config/bay_reserved_protocol_v2_addendum38.json'}
    report['G1'] = gate_extraction()
    print('G1', report['G1']['passed'], report['G1']['max_abs_diff'], flush=True)
    report['G2'] = gate_development()
    print('G2', {a: (round(v['nested_mae']['temp_c'], 6), v['passed']) for a, v in report['G2'].items()}, flush=True)
    if not (report['G1']['passed'] and all(v['passed'] for v in report['G2'].values())):
        (OUT / 'report.json').write_text(json.dumps(report, indent=2, default=str))
        raise SystemExit('a gate failed; 2026 not read with the Copernicus arms (addendum 38)')
    a36 = json.loads(ADDENDUM36.read_text())
    for label, held_out in (('held_station', True), ('station_seen', False)):
        d, stale = read(held_out)
        path = OUT / f'reserved2026_{label}_predictions.csv.gz'
        d.to_csv(path, index=False)
        arms = ['A2_none', 'A7o', 'A4_base_only', 'A1_both', 'seasonal']
        block = {'casts': int(d.cast_id.nunique()), 'dates': int(d.t_utc.dt.floor('D').nunique()),
                 'stale_casts': stale, 'prediction_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
                 'nested_mae': {a: {t: round(macro(cast_errors(d, a, t)), 4) for t in TARGETS} for a in arms},
                 'contrasts': {}}
        for a, b in (('A1_both', 'A2_none'), ('A1_both', 'A7o'), ('A4_base_only', 'A1_both'),
                     ('A4_base_only', 'A2_none'), ('A1_both', 'seasonal')):
            block['contrasts'][f'{a} - {b}'] = {t: contrast(cast_errors(d, a, t), cast_errors(d, b, t)) for t in TARGETS}
        block['G3'] = {a: {t: [block['nested_mae'][a][t], a36[label]['nested_mae'][a][t]] for t in TARGETS}
                       for a in ('A2_none', 'A7o')}
        block['G3_passed'] = bool(all(abs(x - y) < 1e-9 for v in block['G3'].values() for x, y in v.values()))
        report[label] = block
        print(label, json.dumps(block['nested_mae']), 'G3', block['G3_passed'], flush=True)
    h = report['held_station']['contrasts']
    report['predictions'] = {
        'P1': h['A1_both - A2_none']['temp_c']['point'] < 0 and h['A1_both - A2_none']['temp_c']['excludes_zero'],
        'P2': not h['A1_both - A7o']['temp_c']['excludes_zero'],
        'P3': abs(h['A4_base_only - A1_both']['temp_c']['point']) < 0.02,
        'P4': not h['A1_both - A2_none']['do_umol_l']['excludes_zero']}
    (OUT / 'report.json').write_text(json.dumps(report, indent=2, default=str))
    print(json.dumps(report['predictions']), flush=True)
    for k in ('A1_both - A2_none', 'A1_both - A7o', 'A4_base_only - A1_both'):
        print(k, {t: (v['point'], v['ci']) for t, v in h[k].items()}, flush=True)


if __name__ == '__main__':
    main()
