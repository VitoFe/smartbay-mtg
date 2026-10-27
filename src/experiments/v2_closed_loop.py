"""Addendum-4: closed-loop rollout when one probe drops out.

Open-loop evaluation always re-anchors the model on a real observation. An operator bridging a
telemetry gap cannot: the model's own output becomes its next input. This measures how fast error
grows under that feedback, and whether a physically constrained arm degrades more gracefully than
an unconstrained tree.

Development data only; no 2026 endpoint is touched. Declared in
config/bay_reserved_protocol_v2_addendum4.json before anything here was run.

Arms, all scored on the identical episodes and the frozen (fold, sensor) macro rule:
  recursive_tree       the 6 h tree applied iteratively, fed its own predictions
  direct_tree          the tuned direct-horizon tree at the matched lead: the no-compounding control
  graph_denied         graph dynamics rolled with observations injected for every sensor but the
                       dropped one, through the architecture's existing denied-sensor path
  persistence          the delivered fallback, holding the last real value
  climatology          anchor plus the sensor/month/hour mean increment at that lead

After onset the dropped probe contributes nothing real to its own inputs: its anchor, trend,
availability and own temperature are the rolled state and its own salinity is held at onset, in
both the tree (and its temperature companion) and the graph path. Other probes and the weather stay
real. (Until 2026-09-26 the probe's own temperature, salinity and, in the graph, trend were read
from the real record after onset.)

The tree's neighbour column falls back to the probe's own reading when no other probe is available
(bay_dynamics_tree.features). With `--rolled-neighbour` that fallback is the rolled state after
onset, as features() would have set it had the probe's own reading been the rolled one; without it
the silent probe's real reading enters through that column at such a step (753 of the 3,269 rolled
oxygen steps; no temperature step). The switch writes `closed_loop_<variable>_rolled_neighbour*`
beside the published files, which it leaves untouched. `--replay` recomputes only the tree arm (and
its temperature companion) and takes every other arm from the published predictions, which the
neighbour column cannot reach; with the switch off it is the reproduction gate.

`--perfold` (review R3, 2026-10-05): the step tree, the temperature companion tree and the rolled
graph each take the configuration selected on that outer fold's own inner window (addenda 47 and
49, `v2_perfold_select`) instead of the shared one, which was chosen on both folds' inner windows
together. Everything else is the default run: the same episodes, silencing, seed and summary, and
`--rolled-neighbour` keeps its meaning (the thesis reads temperature without it, where it changes no
step, and oxygen with it). The graph is rolled again, since its configuration changes. Files go to
`runs/v2_closed_loop/perfold/` (`perfold_smoke/` with `--quick`) under the default names; the shared
files are not touched. Its gate: at the six-hour lead the rollout must equal the per-fold open-loop
weather tree on exactly the rows on which the shared rollout equals the shared one.
"""
import argparse, hashlib, json, os, zlib
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from src.experiments import v2_perfold_select as pfs
from src.experiments.bay_dynamics_tree import features
from src.experiments.bay_twin_sweep import save
from src.experiments.v2_horizon import ROOT, build, folds, graphs
from src.experiments.v2_forecast import NEURAL_ARMS, make_tree
from src.models.bay_graph_dynamics import BayGraphDynamics
from src.data.bay_forcing import weather

DEV = [ROOT / 'reports/bay_twin/runs/20260920T083322Z_v2_horizon/results',
       ROOT / 'reports/bay_twin/runs/20260920T133747Z_v2_horizon/results']
OUT = ROOT / 'reports/bay_twin/runs/v2_closed_loop'
STEP = 6
LEADS = [6, 12, 18, 24, 30, 36, 42, 48]
DIRECT_LEADS = [6, 12, 24, 36, 48]
DRAWS = 600

# Two different column layouts, and confusing them silently feeds salinity where temperature
# belongs. features() returns [anchor, neighbour] followed by the 14 forcing channels, so a design
# index is the forcing index plus two.
COL_ANCHOR, COL_NEIGHBOUR, COL_TREND, COL_TREND_OK = 0, 1, 2, 3
FORCING_TREND, FORCING_TREND_OK, FORCING_AVAILABLE = 0, 1, 2
FORCING_TEMPERATURE, FORCING_TEMPERATURE_OK, FORCING_SALINITY, FORCING_SALINITY_OK = 10, 11, 12, 13
COL_AVAILABLE = FORCING_AVAILABLE + 2
COL_TEMPERATURE, COL_TEMPERATURE_OK = FORCING_TEMPERATURE + 2, FORCING_TEMPERATURE_OK + 2
COL_SALINITY, COL_SALINITY_OK = FORCING_SALINITY + 2, FORCING_SALINITY_OK + 2
# v2_horizon.build gives every probe its own instrument's channels: its 6 h trend, the availability
# of its reading, and its own forward-filled temperature and salinity. After onset every one of them
# belongs to the silent instrument, so each is replaced by the rolled state (trend, availability,
# temperature) or held at its onset value (salinity, which nothing rolls). Neighbours, the calendar,
# the static position and the weather stay real (since 2026-09-26).


def selected(variable, arm, horizon):
    for root in DEV:
        p = root / f'v2_horizon_{variable}_{arm}_h{horizon}' / 'report.json'
        if p.exists():
            return json.loads(p.read_text())['selection']['params']
    raise FileNotFoundError(f'{variable} {arm} h{horizon}')


class Configurations:
    """The configuration of each arm at each outer fold: the shared selection (`selected`, the
    default), or with `perfold` the fold's own (`v2_perfold_select.forecast`), every per-fold record
    used being kept in `sources` for the report."""

    def __init__(self, perfold):
        self.perfold = perfold
        self.sources = {}

    def __call__(self, variable, arm, fold_number):
        if not self.perfold:
            return selected(variable, arm, STEP)
        rec = pfs.forecast(variable, arm, STEP, fold_number)
        self.sources.setdefault(f'{variable}|{arm}', {})[str(int(fold_number))] = {
            k: rec[k] for k in ('source', 'sha256', 'study', 'completed_trials', 'params')}
        return rec['params']

    def record(self):
        return {'design': 'perfold' if self.perfold else 'shared',
                'objective': ("fold-macro forecast MAE / persistence MAE on that outer fold's own inner window only"
                              if self.perfold else 'mean over both inner windows (the shared selection)'),
                'records': self.sources}


def fit_step_tree(data, design, fold, params):
    """Train the one-step (6 h) tree on the fold's training rows."""
    mask = data['valid'] & fold['eligible'][None, :]
    tr = fold['tr']; tm = mask[tr]
    w = np.broadcast_to(1 / np.maximum(tm.sum(0), 1), tm.shape)[tm]; w = w / w.mean()
    model = make_tree({**params, 'subsample_freq': 1})
    model.fit(design[tr][tm], (data['truth'][tr] - data['x'][tr])[tm], sample_weight=w)
    return model


def episodes(data, fold, max_lead):
    """Issuances whose sensor has a real anchor and a real observation at every scored lead.

    Truth at lead (k+1)*STEP from issuance i is data['truth'][i + k], because build() stores the
    observation one step ahead of each issuance and the issuance stride equals the step.
    """
    steps = max_lead // STEP
    va = fold['va']
    usable = va[(va + steps) < len(data['times'])]
    out = []
    for i in usable:
        for s in np.flatnonzero(fold['eligible']):
            if not (np.isfinite(data['x'][i, s]) and data['valid'][i, s]):
                continue
            if all(data['valid'][i + k, s] for k in range(steps)):
                out.append((i, s))
    return out


def lead_climatology(data, fold, steps):
    """Mean increment from the anchor to each lead, by sensor, month and issuance hour.

    Fitted on training rows only, and applied to the last real value before onset, so this arm
    never sees an observation from inside the simulated outage. Cells absent from training fall
    back to that sensor's mean increment at that lead and then to the global one.
    """
    t = data['times']; n = data['x'].shape[1]; tr = fold['tr']
    tr = tr[tr + steps < len(t)]
    tables = []
    for k in range(steps):
        delta = data['truth'][tr + k] - data['x'][tr]
        ok = data['valid'][tr + k] & np.isfinite(data['x'][tr])
        j, s = np.nonzero(ok)
        frame = pd.DataFrame({'sensor': s, 'month': t.month.to_numpy()[tr][j],
                              'hour': t.hour.to_numpy()[tr][j], 'd': delta[ok]})
        per_sensor = frame.groupby('sensor').d.mean().reindex(np.arange(n)).to_numpy()
        overall = float(frame.d.mean())
        fallback = np.where(np.isfinite(per_sensor), per_sensor, overall)
        cell = frame.groupby(['sensor', 'month', 'hour']).d.mean()
        tables.append((cell, fallback))
    return tables


def climatological_value(tables, data, i0, s, k):
    cell, fallback = tables[k]
    key = (s, int(data['times'][i0].month), int(data['times'][i0].hour))
    delta = cell.get(key, np.nan)
    if not np.isfinite(delta):
        delta = fallback[s]
    return data['x'][i0, s] + delta


def silence(row, onset_row, temperature):
    """Overwrite the silent instrument's own channels in one design row.

    The anchor and trend are set by the caller. Availability is 1, because the rolled state is
    passed as a value (the real flag is 1 on every scored episode too, since the probe did report).
    Temperature is the rolled temperature; salinity is held at its onset value with its onset flag.
    """
    row[COL_AVAILABLE] = 1.0
    row[COL_TEMPERATURE] = temperature
    row[COL_TEMPERATURE_OK] = 1.0
    row[COL_SALINITY] = onset_row[COL_SALINITY]
    row[COL_SALINITY_OK] = onset_row[COL_SALINITY_OK]
    return row


def alone(available, i, s):
    """True where no probe other than s is available at grid row i, so that features() set the
    neighbour column of s to s's own reading."""
    others = np.asarray(available[i], dtype=bool).copy()
    others[s] = False
    return not others.any()


def roll_tree(data, design, model, episode, steps, variable, companion=None, rolled_neighbour=False,
              counter=None):
    """Feed the tree its own output. Every channel of the dropped instrument is synthesised.

    For temperature the probe's own temperature channel is its rolled state (in the grid's units
    (T - 20) / 10, the same as the anchor's). For oxygen it is the co-located temperature rolled by
    the companion temperature tree, whose own row is silenced the same way. The first step reads the
    onset row, which is observed, so only steps after onset are silenced and the six-hour lead is
    the published one exactly.

    `rolled_neighbour`: after onset, at a step at which no other probe is available, the neighbour
    column (the probe's own reading in features()) is the rolled state, for the probe and for its
    temperature companion. `counter`, if given, counts the steps after onset and those steps.
    """
    i0, s = episode
    if variable == 'oxygen' and companion is None:
        raise ValueError('the oxygen rollout needs the co-located temperature companion')
    anchor = data['x'][i0, s]
    previous = data['x'][i0 - 1, s] if np.isfinite(data['x'][i0 - 1, s]) else anchor
    state, prior = anchor, previous
    onset_row = design[i0, s]
    companion_state = data['forcing'][i0, s, FORCING_TEMPERATURE] if companion is not None else None
    if companion is not None:
        conset_row = companion['design'][i0, companion['sensor']]
    path = []
    for k in range(steps):
        row = design[i0 + k, s].copy()
        row[COL_ANCHOR] = state
        row[COL_TREND] = state - prior
        row[COL_TREND_OK] = 1.0
        if companion is not None:
            row[COL_TEMPERATURE] = companion_state
            row[COL_TEMPERATURE_OK] = 1.0
        if k:
            silence(row, onset_row, companion_state if companion is not None else state)
            lonely = alone(data['available'], i0 + k, s)
            if counter is not None:
                counter['steps_after_onset'] = counter.get('steps_after_onset', 0) + 1
                counter['no_other_probe'] = counter.get('no_other_probe', 0) + int(lonely)
            if rolled_neighbour and lonely:
                row[COL_NEIGHBOUR] = state
        nxt = state + float(model.predict(row[None, :])[0])
        if companion is not None:
            crow = companion['design'][i0 + k, companion['sensor']].copy()
            crow[COL_ANCHOR] = companion_state
            crow[COL_TREND] = 0.0 if k == 0 else companion_state - companion['prior']
            crow[COL_TREND_OK] = 1.0
            if k:
                silence(crow, conset_row, companion_state)
                clonely = alone(companion['available'], i0 + k, companion['sensor'])
                if counter is not None:
                    counter['companion_no_other_probe'] = counter.get('companion_no_other_probe', 0) + int(clonely)
                if rolled_neighbour and clonely:
                    crow[COL_NEIGHBOUR] = companion_state
            companion['prior'] = companion_state
            companion_state = companion_state + float(companion['model'].predict(crow[None, :])[0])
        prior, state = state, nxt
        path.append((state, companion_state))
    return path


def roll_graph(data, fold, params, exchange, relaxation, episode_list, steps, seed, device, variable,
               temperature_paths=None):
    """Graph rollout with real observations injected for every sensor except the dropped one.

    The dropped sensor's own forcing channels are silenced after onset as in roll_tree: its trend is
    the rolled six-hour change, its availability 1, its temperature the rolled state (temperature)
    or the companion's rolled co-located temperature (oxygen, `temperature_paths`), and its
    salinity held at onset.
    """
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

    x_all = torch.as_tensor(np.nan_to_num(arrays['x']), device=device).float()
    avail = torch.as_tensor(data['available'], device=device)
    forcing = torch.as_tensor(data['forcing'], device=device).float()
    out = {}
    with torch.no_grad():
        for (i0, s) in episode_list:
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
                    f[0, s, FORCING_TREND] = current - previous
                    f[0, s, FORCING_TREND_OK] = 1.0
                    f[0, s, FORCING_AVAILABLE] = 1.0
                    if variable == 'temperature':
                        f[0, s, FORCING_TEMPERATURE] = current
                    elif temperatures is not None:
                        f[0, s, FORCING_TEMPERATURE] = float(temperatures[k - 1])
                    else:   # no co-located temperature: such an episode is not scored; hold onset
                        f[0, s, FORCING_TEMPERATURE] = forcing[i0, s, FORCING_TEMPERATURE]
                    f[0, s, FORCING_TEMPERATURE_OK] = 1.0
                    f[0, s, FORCING_SALINITY] = forcing[i0, s, FORCING_SALINITY]
                    f[0, s, FORCING_SALINITY_OK] = forcing[i0, s, FORCING_SALINITY_OK]
                    previous = current
                state, _ = model.rollout(state, available, f, hours=STEP)
                real = x_all[i0 + k + 1:i0 + k + 2]
                seen = avail[i0 + k + 1:i0 + k + 2] & ~denied
                state = torch.where(seen, real, state)
                available = available | seen
                path.append(float(state[0, s]))
            out[(i0, s)] = path
    return out


def macro(frame, col):
    f = frame.dropna(subset=[col])
    return float(f.assign(e=(f.truth - f[col]).abs()).groupby(['fold', 'sensor']).e.mean().groupby('fold').mean().mean())


def paired(frame, x, y, label):
    rng = np.random.default_rng([20260920, zlib.crc32(label.encode())])
    keys = frame[['fold', 'sensor']].drop_duplicates().values.tolist()
    groups = {tuple(k): g for k, g in frame.groupby(['fold', 'sensor'])}
    point = macro(frame, x) - macro(frame, y)
    draws = []
    for _ in range(DRAWS):
        s = pd.concat([groups[tuple(keys[i])] for i in rng.integers(0, len(keys), len(keys))])
        draws.append(macro(s, x) - macro(s, y))
    return [point, *np.percentile(draws, [2.5, 97.5]).tolist()]


def save_lf(path, obj):
    """JSON with LF line endings whatever the host (Path.write_text would write CRLF on Windows)."""
    path.write_bytes(json.dumps(obj, indent=2, allow_nan=False).encode('utf-8'))


GZIP = {'method': 'gzip', 'mtime': 0}


def run(variable, max_lead=48, quick=False, rolled_neighbour=False, replay=False, perfold=False,
        graph_arm='conservative_exchange'):
    """The addendum-4 rollout. `rolled_neighbour` is the switch described in the module docstring.

    `replay` recomputes the tree arm and its companion only and reads the graph arm from the
    published predictions: with the switch off it checks that the replay reproduces the published
    predictions (and writes nothing but the gate), with it on it writes the corrected files.

    `perfold` takes every configuration from the fold's own selection (module docstring) and writes
    under OUT/'perfold'; it cannot be combined with `replay`, which checks the shared run.

    `graph_arm` (addendum 76) rolls another graph operator in the graph_denied column; any arm but the
    published conservative_exchange writes to runs/v2_closed_loop_<arm>/, so no published folder changes.
    """
    if perfold and replay:
        raise ValueError('--replay checks the shared run; a per-fold run rolls every arm itself')
    if graph_arm != 'conservative_exchange' and (replay or not perfold):
        raise ValueError('another graph arm is rolled only in a per-fold run, which rolls every arm itself')
    out = OUT / ('perfold_smoke' if quick else 'perfold') if perfold else (OUT / 'smoke' if quick else OUT)
    if graph_arm != 'conservative_exchange':
        # its own folder beside OUT, so the published folders keep their digests
        out = OUT.parent / f'v2_closed_loop_{graph_arm}' / ('perfold_smoke' if quick else 'perfold')
    out.mkdir(parents=True, exist_ok=True)
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    torch.set_num_threads(int(os.environ.get('V2_THREADS', '4')))
    steps = max_lead // STEP
    config = Configurations(perfold)

    data = build(variable, STEP)
    outer = folds(data, STEP)
    met = weather(data['times']).to_numpy()
    companion_data = build('temperature', STEP) if variable == 'oxygen' else None

    rows = []
    counter = {}
    for fold in outer:
        base, _ = features(data, {'fallback': fold['fallback']}, 'forecast')
        design = np.concatenate([base, np.broadcast_to(met[:, None], (len(data['times']), len(data['names']), met.shape[1]))], axis=-1)
        model = fit_step_tree(data, design, fold, config(variable, 'weather_tree', fold['number']))
        clim_tables = lead_climatology(data, fold, steps)

        companion = None
        if companion_data is not None:
            # The companion build carries every temperature probe, so its sensor axis is wider than
            # the oxygen one; index by name, and use its own fold's fallback.
            cfold = folds(companion_data, STEP)[fold['number']]
            cindex = {n: j for j, n in enumerate(companion_data['names'])}
            cbase, _ = features(companion_data, {'fallback': cfold['fallback']}, 'forecast')
            cdesign = np.concatenate([cbase, np.broadcast_to(met[:, None], (len(companion_data['times']), len(companion_data['names']), met.shape[1]))], axis=-1)
            cmodel = fit_step_tree(companion_data, cdesign, cfold,
                                   config('temperature', 'weather_tree', cfold['number']))

        eps = episodes(data, fold, max_lead)
        if quick: eps = eps[:40]
        # The tree paths come first: the oxygen graph reads the companion's rolled temperature.
        tree_paths = {}
        for (i0, s) in eps:
            if companion_data is not None:
                cs = cindex.get(data['names'][s])
                if cs is None:
                    continue
                companion = {'design': cdesign, 'model': cmodel, 'sensor': cs,
                             'prior': companion_data['x'][i0, cs], 'available': companion_data['available']}
            tree_paths[(i0, s)] = roll_tree(data, design, model, (i0, s), steps, variable, companion,
                                            rolled_neighbour=rolled_neighbour, counter=counter)
        temperature_paths = ({e: [c for _, c in p] for e, p in tree_paths.items()}
                             if variable == 'oxygen' else None)
        graph_params = config(variable, graph_arm, fold['number'])
        graph, exchange, relaxation = NEURAL_ARMS[graph_arm]
        data['adjacency'] = graphs(data)[graph]
        graph_paths = None if replay else roll_graph(data, fold, graph_params, exchange, relaxation, eps,
                                                     steps, 42, device, variable, temperature_paths)

        for (i0, s) in eps:
            if (i0, s) not in tree_paths:
                continue
            path = tree_paths[(i0, s)]
            gpath = None if replay else graph_paths[(i0, s)]
            for k in range(steps):
                value, companion_state = path[k]
                row = {
                    'fold': fold['number'], 'sensor': data['names'][s], 'time': str(data['times'][i0]),
                    'lead': (k + 1) * STEP,
                    'truth': float(data['truth'][i0 + k, s] * data['scale'] + data['center']),
                    'recursive_tree': float(value * data['scale'] + data['center']),
                    'graph_denied': np.nan if replay else float(gpath[k] * data['scale'] + data['center']),
                    'persistence': float(data['x'][i0, s] * data['scale'] + data['center']),
                    'climatology': float(climatological_value(clim_tables, data, i0, s, k) * data['scale'] + data['center'])}
                if companion_data is not None:
                    # rolled co-located temperature, and the last real salinity before onset:
                    # both are needed to ask whether the synthesised oxygen is thermodynamically possible
                    row['rolled_temperature'] = float(companion_state * 10 + 20)
                    sal = data['forcing'][i0, s, FORCING_SALINITY] * 5 + 35
                    row['salinity_at_onset'] = float(sal) if data['forcing'][i0, s, FORCING_SALINITY_OK] > 0 else np.nan
                rows.append(row)

    frame = pd.DataFrame(rows)
    suffix = '_rolled_neighbour' if rolled_neighbour else ''
    if replay:
        frame, gate = against_published(frame, variable, rolled_neighbour)
        gate['steps'] = counter
        gates = OUT / 'rolled_neighbour_gate.json'
        record = json.loads(gates.read_text()) if gates.exists() else {}
        record[f"{variable}|{'on' if rolled_neighbour else 'off'}"] = gate
        save_lf(gates, record)
        print(json.dumps(gate, indent=1))
        if not rolled_neighbour or quick:
            return
        if not gate['passed']:
            raise SystemExit('the replay does not reproduce the published arms it keeps; nothing written')
    path = out / f'closed_loop_{variable}{suffix}.csv.gz'
    if suffix or perfold:
        frame.to_csv(path, index=False, lineterminator='\n', compression=GZIP)
    else:
        frame.to_csv(path, index=False)
    report = summarise(frame, variable, device, steps)
    if suffix:
        report['neighbour_column'] = ('after onset, at a step at which no other probe is available, the '
                                      'rolled state (the probe\'s own reading in training)')
        report['steps'] = counter
        if not perfold:
            report['graph_denied'] = f'read from runs/v2_closed_loop/closed_loop_{variable}.csv.gz (unchanged)'
    if perfold:
        report['selection'] = config.record()
        report['rolled_neighbour'] = bool(rolled_neighbour)
        report['steps'] = counter
        report['graph_denied'] = ('rolled in this run at initialisation seed 42, as the shared run, with '
                                  "each fold's own configuration")
        report['graph_arm'] = graph_arm
        report['smoke_only'] = bool(quick)
        report['threads'] = int(os.environ.get('V2_THREADS', '4'))
        report['gate_open_loop'] = open_loop_gate(frame, variable)
    report['prediction_sha256'] = hashlib.sha256(path.read_bytes()).hexdigest()
    (save_lf if (suffix or perfold) else save)(out / f'closed_loop_{variable}{suffix}_summary.json', report)
    show(report, variable, device)
    if perfold:
        print('gate (six-hour lead against the open-loop weather tree):', report['gate_open_loop'])


def open_loop_gate(frame, variable):
    """At the six-hour lead the rollout reads its onset row, so it is the open-loop weather tree on
    every episode whose trend column (state minus the previous issuance's value) equals the grid's own
    six-hour trend. The per-fold rollout must equal the per-fold open-loop rows (addendum 47) on
    exactly the episodes on which the shared rollout equals the shared open-loop rows: the set of
    episodes depends on the data alone, so any other difference is a fault in the per-fold path."""
    key = ['fold', 'sensor', 'time']
    first = frame[frame.lead == STEP][key + ['truth', 'recursive_tree']]
    path, sha = pfs.predictions('forecast', f'{variable}_weather_tree_h{STEP}')
    j = first.merge(pd.read_csv(path)[key + ['truth', 'prediction']], on=key, how='left', suffixes=('', '_open'))
    shared = pd.read_csv(OUT / f'closed_loop_{variable}.csv.gz')
    shared = shared[shared.lead == STEP][key + ['recursive_tree']]
    shared_open = next(root / f'v2_horizon_{variable}_weather_tree_h{STEP}' / 'selected_predictions.csv.gz'
                       for root in DEV if (root / f'v2_horizon_{variable}_weather_tree_h{STEP}').exists())
    s = shared.merge(pd.read_csv(shared_open)[key + ['prediction']], on=key, how='inner')
    s['equal_shared'] = (s.recursive_tree - s.prediction).abs() <= 1e-9
    j = j.merge(s[key + ['equal_shared']], on=key, how='left')
    j['equal'] = (j.recursive_tree - j.prediction).abs() <= 1e-9
    known = j.equal_shared.notna()
    agree = bool((j.equal[known] == j.equal_shared[known].astype(bool)).all())

    def largest(values):
        values = values.dropna()
        return float(values.max()) if len(values) else None
    truth_diff = largest((j.truth - j.truth_open).abs())
    return {'episodes': int(len(first)), 'with_open_loop_row': int(j.prediction.notna().sum()),
            'equal_to_open_loop': int(j.equal.sum()), 'equal_in_shared_run': int(j.equal_shared[known].astype(bool).sum()),
            'compared_with_shared': int(known.sum()),
            'max_abs_diff_where_equal_in_shared': largest((j.recursive_tree - j.prediction).abs()[
                known & j.equal_shared.fillna(False).astype(bool)]),
            'truth_max_abs_diff': truth_diff,
            'open_loop_predictions': path.relative_to(ROOT).as_posix(), 'open_loop_sha256': sha,
            'passed': bool(agree and known.all() and j.prediction.notna().all() and truth_diff == 0.0)}


def against_published(frame, variable, rolled_neighbour):
    """Join the replayed tree arm to the published predictions, gate the columns it recomputes and
    return the published frame with the replayed tree arm (and companion temperature) in place."""
    published = pd.read_csv(OUT / f'closed_loop_{variable}.csv.gz')
    key = ['fold', 'sensor', 'time', 'lead']
    joined = published.merge(frame, on=key, how='outer', suffixes=('', '_replay'), indicator=True)
    gate = {'rows_published': int(len(published)), 'rows_replayed': int(len(frame)),
            'unmatched': int((joined._merge != 'both').sum())}
    same = ['truth', 'persistence', 'climatology'] + (['salinity_at_onset'] if variable == 'oxygen' else [])
    moved = ['recursive_tree'] + (['rolled_temperature'] if variable == 'oxygen' else [])
    for c in same + moved:
        a, b = joined[c].to_numpy(float), joined[f'{c}_replay'].to_numpy(float)
        both = np.isfinite(a) & np.isfinite(b)
        gate[f'{c}_max_abs_diff'] = float(np.abs(a - b)[both].max())
        gate[f'{c}_nan_mismatch'] = int((np.isfinite(a) != np.isfinite(b)).sum())
    tolerance = 1e-9
    kept = all(gate[f'{c}_max_abs_diff'] <= tolerance and gate[f'{c}_nan_mismatch'] == 0 for c in same)
    if variable == 'oxygen':
        # the oxygen graph reads the companion's rolled temperature, so it is kept only if that is unchanged
        kept = kept and gate['rolled_temperature_max_abs_diff'] <= tolerance
    reproduced = all(gate[f'{c}_max_abs_diff'] <= tolerance and gate[f'{c}_nan_mismatch'] == 0 for c in moved)
    gate['tolerance'] = tolerance
    gate['passed'] = bool(gate['unmatched'] == 0 and kept and (rolled_neighbour or reproduced))
    if rolled_neighbour:
        d = (joined.recursive_tree_replay - joined.recursive_tree).abs()
        gate['recursive_tree_rows_changed'] = int((d > tolerance).sum())
        gate['recursive_tree_changed_by_lead'] = {int(L): int(n) for L, n in
                                                  joined.assign(c=d > tolerance).groupby('lead').c.sum().items()}
    out = published.copy()
    replay = joined.set_index(key)
    for c in moved:
        out[c] = replay.loc[pd.MultiIndex.from_frame(out[key]), f'{c}_replay'].to_numpy()
    return out, gate


def summarise(frame, variable, device, steps):
    report = {'declared_in': 'config/bay_reserved_protocol_v2_addendum4.json', 'variable': variable,
              'device': device, 'episodes': int(frame.groupby(['fold', 'sensor', 'time']).ngroups),
              'rows': len(frame), 'draws': DRAWS, 'leads': LEADS[:steps], 'by_lead': {}}
    arms = ['recursive_tree', 'graph_denied', 'persistence', 'climatology']
    for lead, sub in frame.groupby('lead'):
        entry = {'n': int(len(sub)), 'mae': {a: macro(sub, a) for a in arms}}
        entry['contrasts'] = {
            'recursive_tree - persistence': paired(sub, 'recursive_tree', 'persistence', f'{variable}|{lead}|rt-p'),
            'recursive_tree - climatology': paired(sub, 'recursive_tree', 'climatology', f'{variable}|{lead}|rt-c'),
            'graph_denied - recursive_tree': paired(sub, 'graph_denied', 'recursive_tree', f'{variable}|{lead}|g-rt'),
            'graph_denied - persistence': paired(sub, 'graph_denied', 'persistence', f'{variable}|{lead}|g-p')}
        report['by_lead'][int(lead)] = entry

    # amplitude diagnostic: does the synthesised trajectory flatten, or diverge?
    spread = frame.groupby('lead').agg(truth_sd=('truth', 'std'), tree_sd=('recursive_tree', 'std'),
                                       graph_sd=('graph_denied', 'std'), persistence_sd=('persistence', 'std'))
    report['amplitude'] = {int(k): {'truth_sd': float(v.truth_sd), 'recursive_tree_sd': float(v.tree_sd),
                                    'graph_denied_sd': float(v.graph_sd),
                                    'persistence_sd': float(v.persistence_sd)} for k, v in spread.iterrows()}
    # The pooled ratio above mixes episodes, so it mostly measures the spread of the anchors: holding
    # the last value scores about 1 on it too. Within each episode, the standard deviation of the
    # path over the scored leads against that of the observed path says whether the rollout keeps
    # the variation it has to follow (episodes with a flat observed path are left out).
    by_episode = frame.groupby(['fold', 'sensor', 'time'])
    truth_sd = by_episode.truth.std()
    report['amplitude_within_episode'] = {}
    for arm in ('recursive_tree', 'graph_denied', 'climatology'):
        ratio = (by_episode[arm].std() / truth_sd)[truth_sd > 0]
        report['amplitude_within_episode'][arm] = {
            'median_sd_ratio': float(ratio.median()), 'quartiles': [float(q) for q in ratio.quantile([.25, .75])],
            'episodes': int(len(ratio))}
    lo, hi = float(frame.truth.min()), float(frame.truth.max())
    report['out_of_range'] = {a: int(((frame[a] < lo) | (frame[a] > hi)).sum()) for a in ['recursive_tree', 'graph_denied']}
    report['observed_range'] = [lo, hi]

    # Declared diagnostic 1: does the released interval still mean anything once the inputs are
    # self-generated? An operator's dashboard keeps showing the six-hour band during a gap, so the
    # honest test is to hold that band fixed and measure coverage as the rollout advances.
    first = frame[frame.lead == STEP]
    half = float(np.percentile((first.recursive_tree - first.truth).abs(), 90))
    report['interval'] = {'nominal': 0.90, 'half_width_from_first_step': half,
                          'coverage_by_lead': {int(l): float((( g.recursive_tree - g.truth).abs() <= half).mean())
                                               for l, g in frame.groupby('lead')}}

    # Declared diagnostic 2: is the synthesised oxygen thermodynamically possible? Saturation uses
    # the rolled co-located temperature and the last real salinity before onset.
    if 'rolled_temperature' in frame:
        from src.data.thermodynamics import o2sol_garcia_gordon_umol_per_l as sol
        ok = frame.salinity_at_onset.notna()
        sub = frame[ok]
        ratio = sub.recursive_tree / sol(sub.rolled_temperature.to_numpy(), sub.salinity_at_onset.to_numpy())
        report['supersaturation'] = {
            'rows_checked': int(len(sub)),
            'above_140pct': int((ratio > 1.40).sum()),
            'max_ratio': float(ratio.max()), 'median_ratio': float(ratio.median()),
            'observed_truth_above_140pct': int(
                (sub.truth / sol(sub.rolled_temperature.to_numpy(), sub.salinity_at_onset.to_numpy()) > 1.40).sum())}
    return report


def show(report, variable, device):
    arms = ['recursive_tree', 'graph_denied', 'persistence', 'climatology']
    print(f"== {variable}  episodes={report['episodes']}  rows={report['rows']}  device={device}")
    print(f"{'lead':>5s} " + ' '.join(f'{a:>16s}' for a in arms) + f"{'sd(tree)/sd(obs)':>18s}")
    for lead in sorted(report['by_lead']):
        m = report['by_lead'][lead]['mae']; amp = report['amplitude'][lead]
        print(f'{lead:5d} ' + ' '.join(f'{m[a]:16.4f}' for a in arms) +
              f"{amp['recursive_tree_sd'] / max(amp['truth_sd'], 1e-9):18.3f}")
    print('out-of-observed-range predictions:', report['out_of_range'], 'range', report['observed_range'])
    iv = report['interval']
    print(f"fixed 90 per cent band +/- {iv['half_width_from_first_step']:.4f}, coverage by lead:",
          {k: round(v, 3) for k, v in iv['coverage_by_lead'].items()})
    if 'supersaturation' in report:
        print('supersaturation above 140 per cent:', report['supersaturation'])


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--variable', choices=['temperature', 'oxygen'], required=True)
    p.add_argument('--max-lead', type=int, default=48)
    p.add_argument('--quick', action='store_true')
    p.add_argument('--rolled-neighbour', action='store_true',
                   help='after onset, the neighbour column is the rolled state where no other probe reports')
    p.add_argument('--replay', action='store_true',
                   help='recompute the tree arm only; the graph arm is read from the published predictions')
    p.add_argument('--perfold', action='store_true',
                   help="every arm at its outer fold's own selection; writes runs/v2_closed_loop/perfold/")
    p.add_argument('--graph-arm', default='conservative_exchange', choices=['conservative_exchange', 'learned_source'],
                   help='the graph operator rolled in graph_denied; any other than the published one writes to its own folder')
    a = p.parse_args(); run(a.variable, a.max_lead, a.quick, a.rolled_neighbour, a.replay, a.perfold, a.graph_arm)
