"""Addendum 55 on the compute host: the sonde's own temperature as features of the tuned LightGBM
oxygen gap filler and forecaster (config/bay_reserved_protocol_v2_addendum55.json).

One job per process, from the uploaded snapshot's root with PYTHONPATH set to it:

  tfill    the sonde's temperature filled by addendum 54's method over the episodes addendum 54 did
           not fill (the strict design's development episodes past 72 h); run before rel_T_fill
  gap      --design strict|relaxed --period development|reserved --arm gate|rel_T_obs|rel_T_fill
           (with V2_GAP_DESIGN, V2_GAP_PERIOD and V2_GAP_LENGTHS set as in the published run)
  horizon  --lead H --arm gate|tree_T_obs|tree_T_fc

The gap jobs run v2_gap_recovery.addendum34 exactly as the published anchored_gbm_rel oxygen runs did,
with two edge-relative temperature features added after the relative ones (so the frozen 2026 read's
development build gets them too); the gate replaces each study with its published selection. The
horizon jobs run v2_horizon.run's tree branch with a temperature-ahead block appended to the design;
the gate refits the published selection on the unchanged design. Writes results_lgbm/<job>/.
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
IN = ROOT / 'reports/bay_twin/runs/v2_o2_temperature_lgbm/inputs'   # the published selections
A54 = ROOT / 'reports/bay_twin/runs/v2_o2_temperature'             # addendum 54's inputs and fills
OUT = Path(os.environ.get('V2_LGBM_OUT', ROOT / 'results_lgbm'))
CACHE = Path(os.environ.get('SMARTBAY_FULL_CACHE', ROOT / 'cache'))
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'companion'))


def log(msg):
    print(f'{time.strftime("%H:%M:%S")} {msg}', flush=True)


def save(path, obj):
    path.write_text(json.dumps(obj, indent=1, default=str) + '\n', encoding='utf-8')


# ------------------------------------------------------------------------------------------------ gaps
#: (design, period) -> the V2_GAP_LENGTHS of the published run
CONFIG_LENGTHS = {('strict', 'development'): [1, 2, 3, 6, 12, 18, 24, 48, 72, 168, 336, 504, 720, 1440],
                  ('strict', 'reserved'): [6, 12, 18, 24, 48, 72],
                  ('relaxed', 'development'): [168, 336, 504, 720, 1440],
                  ('relaxed', 'reserved'): [168, 336, 504, 720, 1440]}


def fill_tables():
    """Addendum 54's filled temperature (results/gaps_<cohort>.csv.gz, fetched from its run) and the
    episodes it did not cover, filled here by the same functions (tfill task)."""
    parts = [pd.read_csv(p, usecols=['episode', 'position', 'T_fill', 'T_pick'])
             for p in sorted((A54 / 'results').glob('gaps_*.csv.gz'))]
    extra = OUT / 'tfill_extra.csv.gz'
    if extra.exists():
        parts.append(pd.read_csv(extra, usecols=['episode', 'position', 'T_fill', 'T_pick']))
    return pd.concat(parts, ignore_index=True).drop_duplicates(['episode', 'position'])


def tfill_job(workers):
    """Fill the sonde's temperature, by addendum 54's method, over every oxygen episode of the four
    configurations (and of the development builds the frozen reads train on) that addendum 54 did
    not already fill: the strict design's episodes past 72 h, which Gap-Dev does not report."""
    from scripts.experiments import v2_o2_temperature as T54
    G = T54.gap_module()
    weights = json.loads((A54 / 'inputs/weights.json').read_text(encoding='utf-8'))
    have = set(fill_tables().episode)
    out, summary = [], {}
    for (design, period), lengths in CONFIG_LENGTHS.items():
        G.DESIGN = design
        frame = G.build('oxygen', np.random.default_rng(G.SEED), False, period=period, lengths=lengths)
        eps = frame.drop_duplicates('episode')[['episode', 'sensor', 'gap_length', 'fold']]
        missing = eps[~eps.episode.isin(have)].reset_index(drop=True)
        summary[f'{design}/{period}'] = {'episodes': int(len(eps)), 'filled_here': int(len(missing)),
                                         'lengths': sorted(int(x) for x in missing.gap_length.unique())}
        log(f'tfill {design}/{period}: {len(eps)} episodes, {len(missing)} not filled by addendum 54')
        if not len(missing):
            continue
        full = T54.episode_frame(missing)
        G.PERIOD = period
        members, truth = T54.temperature_members(G, full, period, workers)
        fill, pick, _ = T54.temperature_fill(members, full, missing, weights, period)
        out.append(full.assign(T_fill=fill, T_pick=pick, T_truth=truth))
        have |= set(missing.episode)
    if out:
        pd.concat(out, ignore_index=True).to_csv(OUT / 'tfill_extra.csv.gz', index=False)
    save(OUT / 'tfill_extra.json', summary)
    log('tfill: done')


def add_temperature(G, frame, period, mode, table):
    """own_T_dep: the sonde's temperature at the hour (observed, or filled) minus its straight line
    between the gap's edge hours; own_T_span: the far edge minus the near edge. In place."""
    meta, frames = G.observations(period)
    parts = frame.episode.str.rsplit('|', n=2)
    start = pd.to_datetime(parts.str[1], utc=True)
    length = parts.str[2].astype(int).to_numpy()
    pos = frame.position.to_numpy(int)
    hour = pd.Timedelta(hours=1)
    near = np.full(len(frame), np.nan)
    far = np.full(len(frame), np.nan)
    at = np.full(len(frame), np.nan)
    for sensor, rows in frame.groupby('sensor').groups.items():
        rows = frame.index.get_indexer(rows)
        s = frames[sensor]['temperature'] if 'temperature' in frames[sensor] else pd.Series(dtype=float)
        near[rows] = s.reindex(start.iloc[rows] - hour).to_numpy(float)
        far[rows] = s.reindex(start.iloc[rows] + pd.to_timedelta(length[rows], unit='h')).to_numpy(float)
        at[rows] = s.reindex(start.iloc[rows] + pd.to_timedelta(pos[rows] - 1, unit='h')).to_numpy(float)
    if mode == 'fill':
        got = frame[['episode', 'position']].merge(table, on=['episode', 'position'], how='left')
        at = got.T_fill.to_numpy(float)
        frame.attrs.setdefault('temperature_fill_missing', {})[period] = int(np.isnan(at).sum())
    frame['own_T_dep'] = at - (near + (far - near) * pos / (length + 1))
    frame['own_T_span'] = far - near
    return frame


def gap_job(design, period, arm, quick):
    from scripts.experiments import v2_o2_temperature as T54
    G = T54.gap_module()                          # the thesis module, its reader cached per period
    assert (G.DESIGN, G.PERIOD) == (design, period), 'the launcher sets V2_GAP_DESIGN and V2_GAP_PERIOD'
    assert G.LENGTHS == CONFIG_LENGTHS[(design, period)], 'the launcher sets V2_GAP_LENGTHS as published'
    mode = {'gate': None, 'rel_T_obs': 'obs', 'rel_T_fill': 'fill'}[arm]
    table = fill_tables() if mode == 'fill' else None
    original = G.add_relative
    missing = {}

    def add_relative(frame, variable, period_=None):
        original(frame, variable, period_)
        if mode:
            add_temperature(G, frame, period_ or G.PERIOD, mode, table)
            missing.update(frame.attrs.get('temperature_fill_missing', {}))
        return frame
    G.add_relative = add_relative
    folder = OUT / f'gap_{design}_{period}_{arm}'
    folder.mkdir(parents=True, exist_ok=True)
    if arm == 'gate':
        published = json.loads((IN / f'published_gap_{design}_{period}.json').read_text(encoding='utf-8'))
        chosen = {str(s['fold']): s['selection']['params'] for s in published['selections']}

        def run_study(name, objective, where, quick_):       # the published selection, no search
            return {'params': dict(chosen[name.rsplit('_f', 1)[1]]), 'published': True}
        G.run_study = run_study
    t0 = time.time()
    frame = G.build('oxygen', np.random.default_rng(G.SEED), quick)
    G.pad_neighbours(frame)
    G.add_relative(frame, 'oxygen')
    cols = G.relative_columns(frame, False) + (['own_T_dep', 'own_T_span'] if mode else [])
    log(f'gap {design} {period} {arm}: {frame.episode.nunique()} episodes, {len(cols)} features ({time.time() - t0:.0f} s)')
    if mode == 'fill' and not quick and any(missing.values()):
        raise RuntimeError(f'filled temperature absent on some rows: {missing}; run the tfill task first')
    predictions, selections = G.addendum34('oxygen', 'anchored_gbm_rel', frame, cols, folder, quick)
    if mode == 'fill' and not quick and any(missing.values()):
        raise RuntimeError(f'filled temperature absent on training rows: {missing}; run the tfill task first')
    frame[arm] = predictions
    keep = ['episode', 'sensor', 'gap_length', 'position', 'fold', 'time', 'truth', 'linear', arm] \
        + (['own_T_dep', 'own_T_span'] if mode else [])
    frame[keep].to_csv(folder / 'predictions.csv.gz', index=False)
    save(folder / 'report.json', {'design': design, 'period': period, 'arm': arm, 'features': cols,
                                  'rows': int(len(frame)), 'episodes': int(frame.episode.nunique()),
                                  'threads': G.THREADS, 'temperature_fill_missing_rows': missing,
                                  'own_T_dep_available': float(frame.own_T_dep.notna().mean()) if mode else None,
                                  'selections': selections, 'seconds': round(time.time() - t0)})
    log(f'gap {design} {period} {arm}: done ({time.time() - t0:.0f} s)')


# ------------------------------------------------------------------------------------------------ forecasts
def temperature_block(V, data, lead, mode):
    """[times, sonde, 3]: the temperature at the target hour (observed, or forecast at the issuance),
    scaled as the study scales temperature, its change since the latest reading, and a flag."""
    meta, frames = V.load_observations()
    index = pd.date_range('2024-07-24', '2025-12-31T23:00:00', freq='h', tz='UTC')   # v2_horizon.build
    raw = pd.DataFrame({n: frames[n]['temperature'].reindex(index) if 'temperature' in frames[n] else np.nan
                        for n in data['names']}, index=index)
    q = index.get_indexer(data['times'])
    latest = raw.ffill(limit=6).to_numpy()[q]
    if mode == 'obs':
        ahead = np.full(latest.shape, np.nan)
        ok = q + lead < len(index)
        ahead[ok] = raw.to_numpy()[q[ok] + lead]
    else:
        P = np.load(CACHE / 'panels.npz')
        T = np.load(CACHE / 'forecasts_temperature.npz')
        pindex = pd.DatetimeIndex(P['times']).tz_localize('UTC')
        p = pindex.get_indexer(data['times'])
        ahead = np.full(latest.shape, np.nan)
        for j, n in enumerate(data['names']):
            if f'{n}|t' not in T.files:
                continue
            row_of = {int(t): r for r, t in enumerate(T[f'{n}|t'])}
            trace = T[f'{n}|c2_net']
            for i, pos in enumerate(p):
                r = row_of.get(int(pos))
                if r is not None:
                    ahead[i, j] = trace[r, lead - 1]
    level, change = (ahead - 20) / 10, (ahead - latest) / 10
    flag = np.isfinite(level) & np.isfinite(change)
    return np.stack([np.nan_to_num(level), np.nan_to_num(change), flag.astype(float)], axis=-1).astype('float32')


def horizon_job(lead, arm, quick):
    from src.experiments import v2_horizon as V
    t0 = time.time()
    data = V.build('oxygen', lead)
    outer = V.folds(data, lead)
    if quick:
        for f in outer:
            f['tr'] = f['tr'][-300:]
            f['va'] = f['va'][:40]
    inner = [V.inner_fold(data, f, lead) for f in outer]
    if quick:
        for f in inner:
            f['tr'] = f['tr'][-300:]
            f['va'] = f['va'][:40]
    clim = {id(f): V.climatology(data, f) for f in outer + inner}
    block = None if arm == 'gate' else temperature_block(V, data, lead, 'obs' if arm == 'tree_T_obs' else 'fc')

    def design(fold):
        base, _ = V.features(data, {'fallback': fold['fallback']}, 'forecast')
        return base if block is None else np.concatenate([base, block], axis=-1)
    cache = {id(f): design(f) for f in outer + inner}
    folder = OUT / f'horizon_h{lead}_{arm}'
    folder.mkdir(parents=True, exist_ok=True)
    if arm == 'gate':
        sel = {'params': json.loads((IN / f'published_tree_h{lead}.json').read_text(encoding='utf-8'))['params'],
               'published': True}
    else:
        def objective(trial):
            p = V.tree_space(trial)
            return float(np.mean([V.ratio(V.tree_fit_score(data, cache[id(f)], f, p, clim[id(f)])[0]) for f in inner]))
        sel = V.run_study(f'oxygen_{arm}_h{lead}', objective, folder, quick)
    params = {**sel['params'], 'subsample_freq': 1}
    frames = [V.tree_fit_score(data, cache[id(f)], f, params, clim[id(f)])[0] for f in outer]
    d = pd.concat(frames, ignore_index=True)
    d.to_csv(folder / 'selected_predictions.csv.gz', index=False)
    used = np.concatenate([f['va'] for f in outer])
    flag = None if block is None else float(block[used][..., 2][data['valid'][used]].mean())
    save(folder / 'report.json', {'lead': lead, 'arm': arm, 'selection': sel, 'metrics': V.metrics(d),
                                  'features': int(cache[id(outer[0])].shape[-1]), 'threads': V.THREADS,
                                  'temperature_available_on_scored_rows': flag, 'seconds': round(time.time() - t0)})
    log(f'horizon h{lead} {arm}: done ({time.time() - t0:.0f} s)')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('task', choices=['tfill', 'gap', 'horizon'])
    ap.add_argument('--design')
    ap.add_argument('--period')
    ap.add_argument('--lead', type=int)
    ap.add_argument('--arm')
    ap.add_argument('--quick', action='store_true')
    ap.add_argument('--workers', type=int, default=8)
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if a.task == 'tfill':
        tfill_job(a.workers)
    elif a.task == 'gap':
        gap_job(a.design, a.period, a.arm, a.quick)
    else:
        horizon_job(a.lead, a.arm, a.quick)


if __name__ == '__main__':
    main()
