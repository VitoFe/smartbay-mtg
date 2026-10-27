"""Addendum 82, part A: the recursive rollout and both rolled graph operators on Bridge-Dev and
Bridge-2026, from 6 to 72 h.

The addendum-4 rollout (`v2_closed_loop`) exists on Bridge-Dev at 6-48 h only. This module rolls the
same arms through the episodes of both continuation cohorts for up to twelve six-hour steps, leaving
`v2_closed_loop` and its published folders untouched:

  recursive_tree        the six-hour weather tree fed its own output (for oxygen with the sonde's
                        temperature rolled by a companion temperature tree)
  graph_conservative    the conservation-constrained graph, real observations injected for every
                        probe but the silent one
  graph_unconstrained   the learned-source graph rolled the same way (addendum 76)

Episodes are chosen by the 48 h rule of addendum 4 (`v2_closed_loop.episodes`), never by a 72 h rule,
and restricted to the keys of the continuation files (`runs/v2_bridge/bridge_<variable>.csv.gz`). Each
is rolled min(12, rows left on the grid) steps.

Configurations: on Bridge-Dev each fold's own (addenda 47 and 49, `v2_perfold_select`); on Bridge-2026
the studies of `v2_rollout_a82_studies` on fold_2026's inner window.

Silencing. After onset every design column of the silent instrument is synthesised as in
`v2_closed_loop.roll_tree` (anchor, six-hour change, availability, own temperature or the companion's
rolled temperature, salinity held at onset), the rolled-neighbour switch is on for both variables,
and, new here (`held=True`), the neighbour column is recomputed from the other probes with the silent
probe's availability held at 1 wherever the real probe is unavailable at that step (features() builds
it with the real availability and falls back to the probe's own fold-median fill). Where the real
probe is available the design's own column is kept, which is the same formula, so the fix cannot act
there. The same holds for the oxygen sonde's temperature companion. Each run rolls the tree twice,
with the fix off (`*_off` columns, the reproduction of the published per-fold rollout, gate A-G1) and
on (the reading, gate A-G2).

Bridge-2026 (gate A-G8): the prediction file carries no truth column; truth enters only in the scorer,
after gates A-G1 to A-G7 pass. A quick run builds and rolls Bridge-2026 and writes no 2026 row.

Declared in config/bay_reserved_protocol_v2_addendum82.json (part_A) before anything here was run.

Usage (V2_THREADS=2, OMP_NUM_THREADS unset, on CPU, where the published per-fold rollouts ran):
  python -m src.experiments.v2_rollout_a82 --cohort dev --variable temperature [--quick]
  python -m src.experiments.v2_rollout_a82 --cohort reserved --variable oxygen
"""
import os
os.environ.setdefault('V2_THREADS', '2')

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from src.experiments import v2_closed_loop as cl
from src.experiments import v2_bridge as vb
from src.experiments import v2_horizon as vh
from src.experiments import v2_perfold_select as pfs
from src.experiments import v2_rollout_a82_studies as studies
from src.experiments.bay_dynamics_tree import features
from src.experiments.v2_forecast import NEURAL_ARMS, graphs, make_tree
from src.models.bay_graph_dynamics import BayGraphDynamics

ROOT = Path(__file__).resolve().parents[2]
RUNS = ROOT / 'reports/bay_twin/runs'
OUT = RUNS / 'v2_addendum82/A/closed_loop'
OUT_LS = RUNS / 'v2_addendum82/A/closed_loop_learned_source'
GATES = RUNS / 'v2_addendum82/A/runs'
DECLARED = 'config/bay_reserved_protocol_v2_addendum82.json part_A'
STEP = cl.STEP
EPISODE_LEAD = 48
MAX_STEPS = 12
COHORT = {'dev': 'Bridge-Dev', 'reserved': 'Bridge-2026'}
GRAPH_ARMS = {'graph_conservative': 'conservative_exchange', 'graph_unconstrained': 'learned_source'}
PUBLISHED = {'temperature': 'closed_loop_temperature.csv.gz', 'oxygen': 'closed_loop_oxygen_rolled_neighbour.csv.gz'}
GZIP = {'method': 'gzip', 'mtime': 0}
TOL_TREE, TOL_GRAPH = 1e-9, 1e-6


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(json.dumps(obj, indent=2, default=str, allow_nan=True).encode('utf-8'))


# --------------------------------------------------------------------------------------------------
# Grids, folds and configurations
# --------------------------------------------------------------------------------------------------
def setup(variable, cohort):
    if cohort == 'dev':
        from src.data.bay_forcing import weather
        data = vh.build(variable, STEP)
        outer = vh.folds(data, STEP)
        met = weather(data['times']).to_numpy()
        comp = vh.build('temperature', STEP) if variable == 'oxygen' else None
        cfolds = {f['number']: f for f in vh.folds(comp, STEP)} if comp is not None else None
    else:
        from src.experiments import v2_horizon_reserved as hr
        data = hr.build(variable, STEP)
        outer = [vb.fold_2026(data)]
        met = studies.reserved_weather(data['times'])
        comp = hr.build('temperature', STEP) if variable == 'oxygen' else None
        cfolds = {2: vb.fold_2026(comp)} if comp is not None else None
    if comp is not None:
        assert comp['times'].equals(data['times']), 'the companion grid is not aligned with the oxygen grid'
    return data, outer, met, comp, cfolds


class Configurations:
    def __init__(self, cohort):
        self.cohort, self.sources = cohort, {}

    def __call__(self, variable, arm, fold_number):
        rec = pfs.forecast(variable, arm, STEP, fold_number) if self.cohort == 'dev' else studies.selection(variable, arm)
        self.sources[f'{variable}|{arm}|fold{int(fold_number)}'] = {
            k: rec[k] for k in ('source', 'sha256', 'study', 'completed_trials', 'params')}
        return rec['params']


def design_for(data, fold, met):
    base, _ = features(data, {'fallback': fold['fallback']}, 'forecast')
    return np.concatenate([base, np.broadcast_to(met[:, None], (len(data['times']), len(data['names']),
                                                                met.shape[1]))], axis=-1)


def ns(values):
    """Timestamps as integer nanoseconds since the epoch, whatever their stored resolution."""
    return pd.to_datetime(pd.Series(values), utc=True).dt.as_unit('ns').astype('int64').tolist()


def bridge_keys(variable, cohort):
    f = pd.read_csv(RUNS / 'v2_bridge' / f'bridge_{variable}.csv.gz', usecols=['cohort', 'fold', 'sensor', 'onset'])
    f = f[f.cohort == COHORT[cohort]].drop_duplicates()
    return set(zip(f.fold.astype(int), f.sensor, ns(f.onset)))


# --------------------------------------------------------------------------------------------------
# The tree rollout, with the availability-held neighbour
# --------------------------------------------------------------------------------------------------
def held_neighbour(adjacency, available, x_filled, i, s):
    """features()' neighbour column of probe s at grid row i with s's own availability held at 1.
    adjacency[s, s] is 0, so s never enters its own neighbour mean."""
    k = adjacency[s] * available[i]
    return float(np.einsum('j,j->', k, x_filled[i]) / max(k.sum(), 1e-6))


def bump(counter, key, k):
    d = counter.setdefault(key, {})
    d[str(6 * k)] = d.get(str(6 * k), 0) + 1


def roll_tree(data, design, model, episode, steps, variable, companion, held, counter, adjacency, x_filled):
    """`v2_closed_loop.roll_tree` with the rolled-neighbour switch on, and with `held` the neighbour
    column recomputed with the silent probe's availability held at 1 where the real probe is silent.
    Counter keys name the input step by its hour after onset (6 * k)."""
    i0, s = episode
    if variable == 'oxygen' and companion is None:
        raise ValueError('the oxygen rollout needs the co-located temperature companion')
    anchor = data['x'][i0, s]
    previous = data['x'][i0 - 1, s] if np.isfinite(data['x'][i0 - 1, s]) else anchor
    state, prior = anchor, previous
    onset_row = design[i0, s]
    companion_state = data['forcing'][i0, s, cl.FORCING_TEMPERATURE] if companion is not None else None
    if companion is not None:
        cs = companion['sensor']
        conset_row = companion['design'][i0, cs]
        companion['prior'] = companion['x0']
    path = []
    for k in range(steps):
        row = design[i0 + k, s].copy()
        row[cl.COL_ANCHOR] = state
        row[cl.COL_TREND] = state - prior
        row[cl.COL_TREND_OK] = 1.0
        if companion is not None:
            row[cl.COL_TEMPERATURE] = companion_state
            row[cl.COL_TEMPERATURE_OK] = 1.0
        if k:
            cl.silence(row, onset_row, companion_state if companion is not None else state)
            lonely = cl.alone(data['available'], i0 + k, s)
            bump(counter, 'steps_after_onset', k)
            if lonely:
                bump(counter, 'no_other_probe', k)
                row[cl.COL_NEIGHBOUR] = state
            elif not data['available'][i0 + k, s]:
                bump(counter, 'probe_unavailable_with_others', k)
                if held:
                    row[cl.COL_NEIGHBOUR] = held_neighbour(adjacency, data['available'], x_filled, i0 + k, s)
        nxt = state + float(model.predict(row[None, :])[0])
        if companion is not None:
            crow = companion['design'][i0 + k, cs].copy()
            crow[cl.COL_ANCHOR] = companion_state
            crow[cl.COL_TREND] = 0.0 if k == 0 else companion_state - companion['prior']
            crow[cl.COL_TREND_OK] = 1.0
            if k:
                cl.silence(crow, conset_row, companion_state)
                if cl.alone(companion['available'], i0 + k, cs):
                    bump(counter, 'companion_no_other_probe', k)
                    crow[cl.COL_NEIGHBOUR] = companion_state
                elif not companion['available'][i0 + k, cs]:
                    bump(counter, 'companion_unavailable_with_others', k)
                    if held:
                        crow[cl.COL_NEIGHBOUR] = held_neighbour(companion['adjacency'], companion['available'],
                                                                companion['x_filled'], i0 + k, cs)
            companion['prior'] = companion_state
            companion_state = companion_state + float(companion['model'].predict(crow[None, :])[0])
        prior, state = state, nxt
        path.append((state, companion_state))
    return path


# --------------------------------------------------------------------------------------------------
# The rolled graph: trained once per fold exactly as v2_closed_loop.roll_graph, rolled per episode
# --------------------------------------------------------------------------------------------------
def train_graph(data, fold, params, exchange, relaxation, seed, device):
    torch.manual_seed(seed)
    arrays = {k: data[k] for k in ['available', 'forcing', 'truth']}
    arrays['x'] = np.where(np.isfinite(data['x']), data['x'], fold['fallback'])
    arrays['valid'] = data['valid'] & fold['eligible'][None, :]
    a = {k: torch.as_tensor(np.nan_to_num(v), device=device) for k, v in arrays.items()}
    a = {k: v.float() if v.dtype != torch.bool else v for k, v in a.items()}
    model = BayGraphDynamics(data['adjacency'], data['forcing'].shape[-1], params['hidden'],
                             exchange, relaxation).to(device).float()
    opt = torch.optim.AdamW(model.parameters(), lr=params['lr'], weight_decay=params['weight_decay'])
    weights = 1 / a['valid'][fold['tr']].sum(0).clamp_min(1).float(); weights /= weights.mean()
    rng = np.random.default_rng(seed)
    for _ in range(params['steps']):
        ids = rng.choice(fold['tr'], min(128, len(fold['tr'])), replace=False)
        pred, _ = model.rollout(a['x'][ids], a['available'][ids], a['forcing'][ids], hours=STEP)
        m = a['valid'][ids]
        loss = (torch.nn.functional.smooth_l1_loss(pred, a['truth'][ids], reduction='none') * m * weights).sum() / (m * weights).sum().clamp_min(1)
        opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 2.); opt.step()
    return model, arrays


def roll_graph(model, data, arrays, episode_steps, variable, temperature_paths, device):
    """The rollout loop of v2_closed_loop.roll_graph for each episode's own number of steps; after
    the grid's last row there is no observation left to inject."""
    x_all = torch.as_tensor(np.nan_to_num(arrays['x']), device=device).float()
    avail = torch.as_tensor(data['available'], device=device)
    forcing = torch.as_tensor(data['forcing'], device=device).float()
    n = len(data['times'])
    out = {}
    with torch.no_grad():
        for (i0, s), steps in episode_steps.items():
            state = x_all[i0:i0 + 1].clone()
            available = avail[i0:i0 + 1].clone()
            denied = torch.zeros_like(available); denied[0, s] = True
            temperatures = (temperature_paths or {}).get((i0, s))
            previous = state[0, s].clone()
            path = []
            for k in range(steps):
                f = forcing[i0 + k:i0 + k + 1]
                if k:
                    f = f.clone()
                    current = state[0, s].clone()
                    f[0, s, cl.FORCING_TREND] = current - previous
                    f[0, s, cl.FORCING_TREND_OK] = 1.0
                    f[0, s, cl.FORCING_AVAILABLE] = 1.0
                    if variable == 'temperature':
                        f[0, s, cl.FORCING_TEMPERATURE] = current
                    elif temperatures is not None:
                        f[0, s, cl.FORCING_TEMPERATURE] = float(temperatures[k - 1])
                    else:
                        f[0, s, cl.FORCING_TEMPERATURE] = forcing[i0, s, cl.FORCING_TEMPERATURE]
                    f[0, s, cl.FORCING_TEMPERATURE_OK] = 1.0
                    f[0, s, cl.FORCING_SALINITY] = forcing[i0, s, cl.FORCING_SALINITY]
                    f[0, s, cl.FORCING_SALINITY_OK] = forcing[i0, s, cl.FORCING_SALINITY_OK]
                    previous = current
                state, _ = model.rollout(state, available, f, hours=STEP)
                if i0 + k + 1 < n:
                    real = x_all[i0 + k + 1:i0 + k + 2]
                    seen = avail[i0 + k + 1:i0 + k + 2] & ~denied
                    state = torch.where(seen, real, state)
                    available = available | seen
                path.append(float(state[0, s]))
            out[(i0, s)] = path
    return out


# --------------------------------------------------------------------------------------------------
# Gate A-G7: the 2026 rollout's first step is the 2026 six-hour forecaster
# --------------------------------------------------------------------------------------------------
def open_loop_gate(data, design, fold, params, eps, first_step, variable):
    """The open-loop weather tree fitted as v2_horizon.tree_fit_score fits it, on fold_2026's training
    rows. (a) On the episodes whose first rolled row is the grid's own design row (its trend column is
    the grid's six-hour trend, compared at the design's float32 storage), the open-loop prediction at the
    onset equals the rollout's six-hour value. (b) On every episode the open-loop model given the
    rollout's own first row returns the rollout's six-hour value, so the two fits are one model."""
    mask = data['valid'] & fold['eligible'][None, :]; tr = fold['tr']; tm = mask[tr]
    w = np.broadcast_to(1 / np.maximum(tm.sum(0), 1), tm.shape)[tm]; w = w / w.mean()
    model = make_tree({**params, 'subsample_freq': 1})
    model.fit(design[tr][tm], (data['truth'][tr] - data['x'][tr])[tm], sample_weight=w)
    same, diffs, own = 0, [], []
    for (i0, s) in eps:
        anchor = data['x'][i0, s]
        previous = data['x'][i0 - 1, s] if np.isfinite(data['x'][i0 - 1, s]) else anchor
        row = design[i0, s].copy()
        row[cl.COL_ANCHOR] = anchor; row[cl.COL_TREND] = anchor - previous; row[cl.COL_TREND_OK] = 1.0
        if variable == 'oxygen':
            row[cl.COL_TEMPERATURE] = data['forcing'][i0, s, cl.FORCING_TEMPERATURE]
            row[cl.COL_TEMPERATURE_OK] = 1.0
        own.append(abs(anchor + float(model.predict(row[None, :])[0]) - first_step[(i0, s)]))
        if np.array_equal(row.astype(np.float32), design[i0, s].astype(np.float32)):
            same += 1
            diffs.append(abs(anchor + float(model.predict(design[i0, s][None, :])[0]) - first_step[(i0, s)]))
    worst = float(max(diffs)) if diffs else None
    worst_own = float(max(own)) if own else None
    return {'episodes': len(eps), 'episodes_whose_first_row_is_the_design_row': same,
            'max_abs_diff_on_those_scaled_units': worst,
            'max_abs_diff_same_row_every_episode_scaled_units': worst_own, 'tolerance': TOL_TREE,
            'passed': bool(same > 0 and worst is not None and worst <= TOL_TREE and worst_own <= TOL_TREE)}


# --------------------------------------------------------------------------------------------------
# One cohort and variable
# --------------------------------------------------------------------------------------------------
def run(variable, cohort, quick=False):
    started = time.time()
    torch.set_num_threads(int(os.environ.get('V2_THREADS', '2')))
    device = 'cpu'
    data, outer, met, comp, cfolds = setup(variable, cohort)
    keys = bridge_keys(variable, cohort)
    config = Configurations(cohort)
    adjacency_tree = data['adjacency'].copy()
    rows, counters = [], {'off': {}, 'on': {}}
    episode_keys, missing_companion, g7 = set(), [], {}
    t, names = data['times'], data['names']
    tns = ns(t)
    for fold in outer:
        number = int(fold['number'])
        design = design_for(data, fold, met)
        x_filled = np.where(np.isfinite(data['x']), data['x'], fold['fallback'])
        tree_params = config(variable, 'weather_tree', number)
        model = cl.fit_step_tree(data, design, fold, tree_params)
        month_hour = cl.lead_climatology(data, fold, MAX_STEPS)
        hour_only = vb.hour_climatology(data, fold, MAX_STEPS)
        if comp is not None:
            cfold = cfolds[number]
            cindex = {n: j for j, n in enumerate(comp['names'])}
            cdesign = design_for(comp, cfold, met)
            cmodel = cl.fit_step_tree(comp, cdesign, cfold, config('temperature', 'weather_tree', number))
            cx_filled = np.where(np.isfinite(comp['x']), comp['x'], cfold['fallback'])
        eps = [(i0, s) for i0, s in cl.episodes(data, fold, EPISODE_LEAD)
               if (number, names[s], tns[i0]) in keys]
        if quick:
            eps = eps[:40]
        episode_keys |= {(number, names[s], tns[i0]) for i0, s in eps}
        steps_of = {e: min(MAX_STEPS, len(t) - e[0]) for e in eps}
        paths = {'off': {}, 'on': {}}
        for (i0, s) in eps:
            companion = None
            if comp is not None:
                cs = cindex.get(names[s])
                if cs is None:
                    missing_companion.append((number, names[s], str(t[i0])))
                    continue
            for mode in ('off', 'on'):
                if comp is not None:
                    companion = {'design': cdesign, 'model': cmodel, 'sensor': cs, 'x0': comp['x'][i0, cs],
                                 'available': comp['available'], 'adjacency': comp['adjacency'], 'x_filled': cx_filled}
                paths[mode][(i0, s)] = roll_tree(data, design, model, (i0, s), steps_of[(i0, s)], variable, companion,
                                                 mode == 'on', counters[mode], adjacency_tree, x_filled)
        if cohort == 'reserved':
            first = {e: p[0][0] for e, p in paths['off'].items()}
            g7[f'fold{number}'] = open_loop_gate(data, design, fold, tree_params, list(paths['off']), first,
                                                 variable)
        rolled = [e for e in eps if e in paths['off']]
        graph_paths = {}
        for column, arm in GRAPH_ARMS.items():
            graph, exchange, relaxation = NEURAL_ARMS[arm]
            data['adjacency'] = graphs(data)[graph]
            gmodel, arrays = train_graph(data, fold, config(variable, arm, number), exchange, relaxation, 42, device)
            for mode in ('off', 'on'):
                tpaths = ({e: [c for _, c in p] for e, p in paths[mode].items()} if variable == 'oxygen' else None)
                if variable == 'temperature' and mode == 'on':
                    graph_paths[(column, 'on')] = graph_paths[(column, 'off')]
                    continue
                graph_paths[(column, mode)] = roll_graph(gmodel, data, arrays, {e: steps_of[e] for e in rolled},
                                                         variable, tpaths, device)
            data['adjacency'] = adjacency_tree.copy()
        sc, ce = data['scale'], data['center']
        for (i0, s) in rolled:
            x0 = data['x'][i0, s]
            for k in range(steps_of[(i0, s)]):
                row = {'cohort': COHORT[cohort], 'fold': number, 'sensor': names[s], 'time': str(t[i0]),
                       'lead': (k + 1) * STEP,
                       'persistence': float(x0 * sc + ce),
                       'climatology': float(cl.climatological_value(month_hour, data, i0, s, k) * sc + ce),
                       'clim_hour': float(vb.hour_value(hour_only, data, i0, s, k) * sc + ce)}
                if cohort == 'dev':
                    row['truth'] = float(data['truth'][i0 + k, s] * sc + ce)
                for mode, suffix in (('off', '_off'), ('on', '')):
                    row[f'recursive_tree{suffix}'] = float(paths[mode][(i0, s)][k][0] * sc + ce)
                    for column in GRAPH_ARMS:
                        row[f'{column}{suffix}'] = float(graph_paths[(column, mode)][(i0, s)][k] * sc + ce)
                    if comp is not None:
                        row[f'rolled_temperature{suffix}'] = float(paths[mode][(i0, s)][k][1] * 10 + 20)
                if comp is not None:
                    sal = data['forcing'][i0, s, cl.FORCING_SALINITY] * 5 + 35
                    row['salinity_at_onset'] = float(sal) if data['forcing'][i0, s, cl.FORCING_SALINITY_OK] > 0 else np.nan
                row['observed_at_target'] = bool(np.isfinite(data['truth'][i0 + k, s]))
                rows.append(row)

    frame = pd.DataFrame(rows)
    summary = {'declared_in': DECLARED, 'cohort': COHORT[cohort], 'variable': variable, 'smoke_only': bool(quick),
               'environment': studies.environment(), 'device': device,
               'episodes': len(episode_keys), 'rows': int(len(frame)), 'selection': config.sources,
               'counters': counters, 'missing_companion': missing_companion}
    summary['gate_A_G3_rows'] = rows_gate(frame, episode_keys, keys, variable, cohort, quick)
    summary['gate_A_G4_references'] = reference_gate(frame, data, variable, cohort, outer)
    if quick:
        summary['gate_A_G4_references']['passed_on_the_rows_rolled'] = bool(
            all(v <= 1e-9 for v in summary['gate_A_G4_references']['max_abs_diff'].values()))
    if cohort == 'dev':
        summary['gate_A_G1_reproduction'] = reproduction_gate(frame, variable, quick)
        summary['gate_A_G2_neighbour_fix'] = fix_gate(frame, variable, counters)
    else:
        summary['gate_A_G7_open_loop'] = g7
        summary['gate_A_G2_neighbour_fix'] = fix_gate(frame, variable, counters)
    folder = ('smoke/' if quick else '') + cohort
    target = OUT / folder / f'closed_loop_{variable}.csv.gz'
    target_ls = OUT_LS / folder / f'closed_loop_{variable}.csv.gz'
    if cohort == 'reserved':
        assert 'truth' not in frame, 'a 2026 truth column in the prediction file'
    if quick and cohort == 'reserved':
        summary['written'] = 'nothing: a quick run writes no 2026 row'
    else:
        target.parent.mkdir(parents=True, exist_ok=True); target_ls.parent.mkdir(parents=True, exist_ok=True)
        keep = [c for c in frame if not c.startswith('graph_unconstrained')]
        frame[keep].to_csv(target, index=False, lineterminator='\n', compression=GZIP)
        ls_cols = ['cohort', 'fold', 'sensor', 'time', 'lead'] + (['truth'] if 'truth' in frame else []) + \
                  ['persistence', 'recursive_tree_off', 'recursive_tree', 'graph_unconstrained_off', 'graph_unconstrained']
        frame[ls_cols].to_csv(target_ls, index=False, lineterminator='\n', compression=GZIP)
        summary['predictions'] = {target.relative_to(ROOT).as_posix(): _sha(target),
                                  target_ls.relative_to(ROOT).as_posix(): _sha(target_ls)}
    summary['seconds'] = round(time.time() - started, 1)
    _json(GATES / ('smoke' if quick else '') / f'run_{cohort}_{variable}.json', summary)
    gates = {k: v.get('passed') for k, v in summary.items() if k.startswith('gate_') and isinstance(v, dict)}
    print(json.dumps({'cohort': cohort, 'variable': variable, 'episodes': summary['episodes'], 'rows': summary['rows'],
                      'gates': gates, 'seconds': summary['seconds']}, default=str), flush=True)
    return summary


# --------------------------------------------------------------------------------------------------
# Gates computed in the run
# --------------------------------------------------------------------------------------------------
def rows_gate(frame, episode_keys, keys, variable, cohort, quick):
    """A-G3: the episodes are the continuation files' (and, on Dev, the published rollout's), every one
    rolled; the rows observed at 54-72 h are the 72 h continuation's rows."""
    out = {'episodes': len(episode_keys), 'continuation_episodes': len(keys)}
    out['equal_to_continuation_keys'] = bool(episode_keys == keys)
    passed = out['equal_to_continuation_keys'] or quick
    if cohort == 'dev':
        pub = pd.read_csv(RUNS / 'v2_closed_loop/perfold' / PUBLISHED[variable],
                          usecols=['fold', 'sensor', 'time']).drop_duplicates()
        pkeys = set(zip(pub.fold.astype(int), pub.sensor, ns(pub.time)))
        out['published_rollout_episodes'] = len(pkeys)
        out['equal_to_published_rollout_keys'] = bool(episode_keys == pkeys)
        passed = passed and (out['equal_to_published_rollout_keys'] or quick)
    b72 = pd.read_csv(RUNS / 'v2_bridge72' / f'bridge72_{variable}.csv.gz', usecols=['cohort', 'fold', 'sensor', 'onset', 'lead'])
    b72 = b72[b72.cohort == COHORT[cohort]]
    k72 = set(zip(b72.fold.astype(int), b72.sensor, ns(b72.onset), b72.lead.astype(int)))
    f = frame[(frame.lead >= 54) & frame.observed_at_target]
    mine = set(zip(f.fold.astype(int), f.sensor, ns(f.time), f.lead.astype(int)))
    out['rows_54_72_observed'] = len(mine)
    out['bridge72_rows'] = len(k72)
    out['equal_to_bridge72_rows'] = bool(mine == k72)
    out['by_lead'] = {int(L): int(n) for L, n in f.groupby('lead').size().items()}
    passed = passed and (out['equal_to_bridge72_rows'] or quick)
    out['passed'] = bool(passed)
    return out


def reference_gate(frame, data, variable, cohort, outer):
    """A-G4: on every joined row persistence, the hour-of-day and the month x hour increments equal the
    continuation files' columns, and so does the truth (on Bridge-2026 the grid's truth is compared with
    the continuation file's without being written or joined to a prediction), to 1e-9."""
    cont = pd.concat([
        pd.read_csv(RUNS / 'v2_bridge' / f'bridge_{variable}.csv.gz'),
        pd.read_csv(RUNS / 'v2_bridge72' / f'bridge72_{variable}.csv.gz')], ignore_index=True)
    cont = cont[cont.cohort == COHORT[cohort]].reset_index(drop=True)
    cont['onset'] = ns(cont.onset)
    cont = cont[['fold', 'sensor', 'onset', 'lead', 'truth', 'persistence', 'clim_hour', 'clim_month_hour']].rename(
        columns={c: f'{c}_cont' for c in ('truth', 'persistence', 'clim_hour', 'clim_month_hour')})
    f = frame.assign(onset=ns(frame.time))
    j = f.merge(cont, on=['fold', 'sensor', 'onset', 'lead'], how='inner')
    out = {'joined_rows': int(len(j)), 'continuation_rows': int(len(cont))}
    diffs = {'persistence': (j.persistence - j.persistence_cont).abs().max(),
             'clim_hour': (j.clim_hour - j.clim_hour_cont).abs().max(),
             'clim_month_hour': (j.climatology - j.clim_month_hour_cont).abs().max()}
    if cohort == 'dev':
        diffs['truth'] = (j.truth - j.truth_cont).abs().max()
    else:
        names = list(data['names']); index = {ts: i for i, ts in enumerate(ns(data['times']))}
        grid_truth = np.array([data['truth'][index[o] + L // STEP - 1, names.index(s)] * data['scale'] + data['center']
                               for o, s, L in zip(j.onset, j.sensor, j.lead)])
        diffs['truth'] = float(np.nanmax(np.abs(grid_truth - j.truth_cont.to_numpy())))
    out['max_abs_diff'] = {k: float(v) for k, v in diffs.items()}
    out['tolerance'] = 1e-9
    out['passed'] = bool(len(j) == len(cont) and all(v <= 1e-9 for v in out['max_abs_diff'].values()))
    return out


def reproduction_gate(frame, variable, quick):
    """A-G1: the fix-off columns reproduce the published per-fold rollouts at 6-48 h on every row."""
    key = ['fold', 'sensor', 'time', 'lead']
    pub = pd.read_csv(RUNS / 'v2_closed_loop/perfold' / PUBLISHED[variable])
    pls = pd.read_csv(RUNS / 'v2_closed_loop_learned_source/perfold' / PUBLISHED[variable])
    f = frame[frame.lead <= 48]
    j = pub.merge(f, on=key, how='outer', suffixes=('_pub', ''), indicator=True)
    jl = pls.merge(f, on=key, how='inner', suffixes=('_pub', ''))
    both = j[j._merge == 'both']
    out = {'published_rows': int(len(pub)), 'rows_6_48': int(len(f)), 'matched': int(len(both)),
           'unmatched': int((j._merge != 'both').sum())}
    checks = {'truth': ('truth_pub', 'truth', TOL_TREE), 'persistence': ('persistence_pub', 'persistence', TOL_TREE),
              'climatology': ('climatology_pub', 'climatology', TOL_TREE),
              'recursive_tree': ('recursive_tree_pub', 'recursive_tree_off', TOL_TREE),
              'graph_conservative': ('graph_denied', 'graph_conservative_off', TOL_GRAPH)}
    if variable == 'oxygen':
        checks['rolled_temperature'] = ('rolled_temperature_pub', 'rolled_temperature_off', TOL_TREE)
    diff = {}
    for name, (a, b, tol) in checks.items():
        d = (both[a] - both[b]).abs()
        diff[name] = {'max_abs_diff': float(d.max()), 'tolerance': tol,
                      'nan_mismatch': int((both[a].isna() != both[b].isna()).sum())}
    dl = (jl.graph_denied - jl.graph_unconstrained_off).abs()
    diff['graph_unconstrained'] = {'max_abs_diff': float(dl.max()), 'tolerance': TOL_GRAPH, 'rows': int(len(jl)),
                                   'published': 'runs/v2_closed_loop_learned_source/perfold/' + PUBLISHED[variable]}
    out['columns'] = diff
    ok = all(v['max_abs_diff'] <= v['tolerance'] and v.get('nan_mismatch', 0) == 0 for v in diff.values())
    out['tree_columns_passed'] = all(diff[c]['max_abs_diff'] <= TOL_TREE and diff[c]['nan_mismatch'] == 0
                                     for c in diff if c not in ('graph_conservative', 'graph_unconstrained'))
    out['graph_columns_passed'] = all(diff[c]['max_abs_diff'] <= TOL_GRAPH for c in ('graph_conservative', 'graph_unconstrained'))
    out['passed'] = bool(ok and (out['unmatched'] == 0 or quick) and len(jl) == len(both))
    return out


def fix_gate(frame, variable, counters):
    """A-G2: what the availability-held neighbour changes. A change on a temperature row at 6-48 h fails
    the gate; on oxygen it is the declared correction acting, counted with its largest change."""
    out = {'counters_fix_on': counters['on']}
    early = frame[frame.lead <= 48]
    cols = ['recursive_tree', 'graph_conservative', 'graph_unconstrained'] + (['rolled_temperature'] if variable == 'oxygen' else [])
    changes = {}
    for c in cols:
        d = (early[c] - early[f'{c}_off']).abs()
        late = (frame[frame.lead > 48][c] - frame[frame.lead > 48][f'{c}_off']).abs()
        changes[c] = {'rows_changed_6_48': int((d > 0).sum()), 'max_change_6_48': float(d.max()) if len(d) else 0.0,
                      'rows_changed_54_72': int((late > 0).sum()), 'max_change_54_72': float(late.max()) if len(late) else 0.0}
    out['changes'] = changes
    probe = counters['on'].get('probe_unavailable_with_others', {})
    out['probe_unavailable_input_steps_54_66'] = {h: probe.get(h, 0) for h in ('54', '60', '66')}
    out['probe_unavailable_input_steps_6_48'] = {h: n for h, n in probe.items() if int(h) <= 48}
    out['companion_unavailable_steps_by_input_hour'] = counters['on'].get('companion_unavailable_with_others', {})
    changed_early = changes['recursive_tree']['rows_changed_6_48']
    if variable == 'temperature':
        out['passed'] = bool(changed_early == 0 and not out['probe_unavailable_input_steps_6_48'])
        out['reading'] = 'no temperature row at 6-48 h changes' if out['passed'] else 'a temperature row at 6-48 h changed'
    else:
        out['passed'] = bool(not out['probe_unavailable_input_steps_6_48'])
        out['correction_acting'] = bool(any(v['rows_changed_6_48'] for v in changes.values()))
        out['reading'] = ('the declared correction acts on the published oxygen rows' if out['correction_acting']
                          else 'no oxygen row at 6-48 h changes')
    return out


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--cohort', choices=['dev', 'reserved'], required=True)
    p.add_argument('--variable', choices=['temperature', 'oxygen'], required=True)
    p.add_argument('--quick', action='store_true')
    a = p.parse_args()
    run(a.variable, a.cohort, quick=a.quick)
