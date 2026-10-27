"""Addendum-5: physics as a gate on the output rather than a constraint in the architecture.

Addenda 2 to 4 found that a conservation operator built into the model does not repay the capacity
it costs, at any horizon, in open loop or in rollout. A gate is the opposite bargain: it leaves the
estimator alone wherever its output is already physically possible and acts only where it is not.

Two regimes. In `single` one probe is silent and its neighbours stay real, which is addendum 4
extended to 120 h. In `network` every probe is silent at once, so the neighbour context is
synthetic too and every input after onset is the model's own output. The second is the regime in
which drift can actually compound, and it was not tested before.

The metric is the tail, not the mean. A gate that fires rarely cannot move a mean and can still
bound the worst case, and the worst case is what an operator is exposed to.

Development folds only. Declared in config/bay_reserved_protocol_v2_addendum5.json.

After onset no rolled probe reads its own record: its availability flag, temperature and salinity
channels are silenced as in addendum 4 (`silence`). Until 2026-09-26 they were read from the real
record after onset, in both regimes.

A probe with no reading at onset starts from its fold's fallback anchor, the median of its readings
over that fold's training rows (`fold['fallback']`, as every forecasting arm uses). Until 2026-09-27
it was the median over the whole development record, validation folds included.

The rate bound is the 99th percentile of six-hour increments between two readings exactly one
issuance step apart, both observed. Until 2026-09-28 it differenced whatever readings remained after
the missing rows were dropped, so increments across gaps of days to months entered it. For oxygen
the saturation bound is evaluated at the probe's temperature held at its onset value, the
companion temperature the rollout carries, and only where onset salinity is known; each gate's
evaluable steps are counted beside its firings.

`--issuance` reads the neighbours at each step's issuance hour, data['x'][i0 + k], as the tree was
trained, where the published runs read them at the step's target hour, data['x'][i0 + k + 1]; a
probe that is not rolled enters the neighbour mean only where it reports at that hour (in `single`
the published runs entered a silent one at its value held from onset), the rolled probes enter at
their rolled state, and a probe with no other probe in the mean gets its own rolled state, as
features() gives a probe with no available neighbour its own reading. In `network` every probe is
silent and nothing is read from the record after onset, so the switch changes nothing there. Its
runs are written as `gate_<variable>_<regime>_issuance*` beside the published files, which stay
untouched. `--check` replays without writing predictions: with the switch off it is the
reproduction gate; with it on it also compares the ungated six-hour step of the single-probe
rollout with the addendum-4 rollout's six-hour prediction, episode by episode. `--range-effect`
gives the range bound's effect a paired bootstrap over onset weeks drawn jointly within each fold.

`--perfold` (review R3, 2026-10-05) fits each fold's step tree at that fold's own selection
(addendum 47, `v2_perfold_select`) instead of the shared one, and works with every other switch:
the runs, their summaries and fires go to `runs/v2_physics_gate/perfold/` under the default names
(`perfold_smoke/` with `--quick`), and so do `--check` (a replay of the per-fold run),
`--six-hour-check` (against the per-fold addendum-4 rollout in `runs/v2_closed_loop/perfold/`) and
`--range-effect` (read from and written to the per-fold folder). The bounds come from the training
rows and do not change.
"""
import argparse, hashlib, json, os, zlib
from pathlib import Path

import numpy as np
import pandas as pd

from src.experiments import v2_perfold_select as pfs
from src.experiments.bay_dynamics_tree import features
from src.experiments.bay_twin_sweep import save
from src.experiments.v2_horizon import ROOT, build, folds
from src.experiments.v2_closed_loop import (STEP, DEV, selected, fit_step_tree, episodes,
                                            COL_ANCHOR, COL_TREND, COL_TREND_OK, COL_AVAILABLE,
                                            COL_TEMPERATURE, COL_TEMPERATURE_OK, COL_SALINITY, COL_SALINITY_OK,
                                            FORCING_TEMPERATURE, FORCING_SALINITY, FORCING_SALINITY_OK)
from src.data.bay_forcing import weather
from src.data.thermodynamics import o2sol_garcia_gordon_umol_per_l as solubility

OUT = ROOT / 'reports/bay_twin/runs/v2_physics_gate'
LEADS = [6, 12, 24, 48, 72, 96, 120]
GATES = ['none', 'range', 'rate', 'all']


def bounds(data, fold):
    """Per-sensor physical bounds taken from the training rows only.

    The rate bound uses only pairs of consecutive rows that are both observed and exactly one
    issuance step (six hours) apart, so no increment spans a gap in the record.
    """
    tr = fold['tr']; valid = data['valid'][tr]; truth = data['truth'][tr]
    times = pd.DatetimeIndex(data['times'])[tr]
    adjacent = np.diff(times.as_unit('ns').asi8) == pd.Timedelta(hours=STEP).value
    lo, hi, rate = [], [], []
    for s in range(data['x'].shape[1]):
        column = truth[valid[:, s], s]
        if len(column) < 20:
            lo.append(-np.inf); hi.append(np.inf); rate.append(np.inf); continue
        lo.append(float(column.min())); hi.append(float(column.max()))
        pair = valid[:-1, s] & valid[1:, s] & adjacent
        step = np.abs(np.diff(truth[:, s]))[pair]
        rate.append(float(np.percentile(step, 99)) if len(step) else np.inf)
    return np.array(lo), np.array(hi), np.array(rate)


def apply_gates(value, previous, s, gate, limits, saturation=None, units=(1.0, 0.0)):
    """Return the gated value and which gates fired. Order: rate, range, saturation.

    The state is z-scored. The saturation bound is a statement about physical concentration, so it
    has to be applied in physical units and converted back; comparing a z-score against a fraction
    of a physical saturation value is meaningless and clamps almost every step.
    """
    lo, hi, rate = limits
    scale, center = units
    fired = []
    if gate in ('rate', 'all') and np.isfinite(rate[s]):
        delta = value - previous
        if abs(delta) > rate[s]:
            value = previous + np.sign(delta) * rate[s]
            fired.append('rate')
    if gate in ('range', 'all') and np.isfinite(lo[s]):
        if value < lo[s]:
            value = lo[s]; fired.append('range')
        elif value > hi[s]:
            value = hi[s]; fired.append('range')
    if gate == 'all' and saturation is not None:
        physical = value * scale + center
        ceiling, floor = 1.40 * saturation, 0.0
        if physical > ceiling:
            value = (ceiling - center) / scale; fired.append('saturation')
        elif physical < floor:
            value = (floor - center) / scale; fired.append('saturation')
    return value, fired


def silence(row, onset_row, state, companion):
    """Overwrite a silent probe's own channels after onset with what a live system knows.

    v2_horizon.build gives every probe its availability flag and its own forward-filled temperature
    and salinity. After onset those belong to the silent instrument (in `network`, to every probe),
    so the flag is held at its onset value; the temperature channel is the rolled state wherever the
    probe had a reading at onset (for temperature it is the same series as the anchor), or its onset
    value otherwise; for oxygen it is the held companion temperature set by the caller; salinity is
    held at onset with its flag. On the `single` episodes every probe reports at onset, so this is
    addendum 4's rule (v2_closed_loop.silence), extended to addendum 5.
    """
    row[COL_AVAILABLE] = onset_row[COL_AVAILABLE]
    if not companion:
        known = onset_row[COL_TEMPERATURE_OK] > 0
        row[COL_TEMPERATURE] = state if known else onset_row[COL_TEMPERATURE]
        row[COL_TEMPERATURE_OK] = onset_row[COL_TEMPERATURE_OK]
    row[COL_SALINITY] = onset_row[COL_SALINITY]
    row[COL_SALINITY_OK] = onset_row[COL_SALINITY_OK]
    return row


def roll(data, design, model, i0, sensors, steps, gate, limits, regime, fallback, companion=None,
         evaluable=None, issuance=False):
    """Roll the chosen sensors forward. In `network` every sensor is synthetic, so the neighbour
    mean is recomputed from rolled states; in `single` the neighbours keep their real values.
    After onset every rolled sensor's own channels are silenced (`silence`). A probe with no reading
    at onset starts from `fallback`, the fold's training-row median. `evaluable`, if given, counts
    the steps on which each gate of the set could act (a finite bound; for saturation, a known
    onset salinity). `issuance` is the switch of the module docstring."""
    state = np.where(np.isfinite(data['x'][i0]), data['x'][i0], fallback)
    prior = np.where(np.isfinite(data['x'][i0 - 1]), data['x'][i0 - 1], state)
    synthetic = np.zeros(len(state), dtype=bool); synthetic[sensors] = True
    companion_state = data['forcing'][i0, :, FORCING_TEMPERATURE].copy() if companion is not None else None
    adjacency = data['adjacency']
    paths = {s: [] for s in sensors}
    fires = []
    for k in range(steps):
        if issuance:
            real = data['x'][i0 + k]      # the issuance hour of this step, as in training
        else:
            real = data['x'][i0 + k + 1] if (i0 + k + 1) < len(data['x']) else state
        current = state.copy()
        if regime == 'single':
            if issuance:
                # rolled probes at their rolled state; any other probe only where it reports
                current = np.where(synthetic, state, real)
            else:
                # neighbours keep their real values; only the dropped probes carry synthetic state
                current = np.where(synthetic, state, np.where(np.isfinite(real), real, state))
        weight = adjacency * np.isfinite(current)[None, :]
        neighbour = (weight * np.nan_to_num(current)[None, :]).sum(1) / np.maximum(weight.sum(1), 1e-6)
        if issuance:
            # features() gives a probe with no available neighbour its own reading
            neighbour = np.where(weight.sum(1) > 0, neighbour, state)
        nxt = state.copy()
        for s in sensors:
            row = design[i0 + k, s].copy()
            row[COL_ANCHOR] = state[s]
            row[1] = neighbour[s]
            row[COL_TREND] = state[s] - prior[s]
            row[COL_TREND_OK] = 1.0
            if companion is not None:
                row[COL_TEMPERATURE] = companion_state[s]
                row[COL_TEMPERATURE_OK] = 1.0
            if k:
                silence(row, design[i0, s], state[s], companion)
            raw = state[s] + float(model.predict(row[None, :])[0])
            sat = None
            if companion is not None and gate == 'all':
                temperature = companion_state[s] * 10 + 20
                sal = data['forcing'][i0, s, FORCING_SALINITY] * 5 + 35
                if data['forcing'][i0, s, FORCING_SALINITY_OK] > 0:
                    sat = float(solubility(np.array([temperature]), np.array([sal]))[0])
            if evaluable is not None:
                lo_s, _, rate_s = (limit[s] for limit in limits)
                evaluable['steps'] += 1
                evaluable['rate'] += int(gate in ('rate', 'all') and np.isfinite(rate_s))
                evaluable['range'] += int(gate in ('range', 'all') and np.isfinite(lo_s))
                evaluable['saturation'] += int(sat is not None)
            gated, fired = apply_gates(raw, state[s], s, gate, limits, sat,
                                       units=(data['scale'], data['center']))
            if fired:
                fires.append({'lead': (k + 1) * STEP, 'sensor': int(s), 'gates': fired,
                              'raw': float(raw), 'gated': float(gated)})
            nxt[s] = gated
            paths[s].append(gated)
        if companion is not None:
            companion_state = companion_state  # held: the companion loop is exercised in addendum 4
        prior, state = state, nxt
    return paths, fires


def tail(errors):
    e = np.abs(errors)
    return {'mae': float(e.mean()), 'p95': float(np.percentile(e, 95)),
            'p99': float(np.percentile(e, 99)), 'max': float(e.max())}


def save_lf(path, obj):
    """JSON with LF line endings whatever the host (Path.write_text would write CRLF on Windows)."""
    path.write_bytes(json.dumps(obj, indent=2, allow_nan=False).encode('utf-8'))


GZIP = {'method': 'gzip', 'mtime': 0}
CHECKS = OUT / 'issuance_gate.json'


def folder(perfold=False, quick=False):
    """Where a run writes: OUT for the shared selection, OUT/'perfold' per fold, and a smoke run beside
    them ('smoke', 'perfold_smoke'), never over a published file."""
    if perfold:
        return OUT / ('perfold_smoke' if quick else 'perfold')
    return OUT / 'smoke' if quick else OUT


def record_check(key, value, path=None):
    path = CHECKS if path is None else path
    path.parent.mkdir(parents=True, exist_ok=True)
    record = json.loads(path.read_text()) if path.exists() else {}
    record[key] = value
    save_lf(path, record)


def fold_models(data, outer, met, variable, perfold=False, sources=None):
    """Per fold: the design, the one-step tree and the bounds, exactly as run() builds them.

    With `perfold` the tree takes that fold's own selection, and each record used is added to
    `sources` (a dict) for the report."""
    for fold in outer:
        base, _ = features(data, {'fallback': fold['fallback']}, 'forecast')
        design = np.concatenate([base, np.broadcast_to(met[:, None], (len(data['times']), len(data['names']), met.shape[1]))], axis=-1)
        if perfold:
            rec = pfs.forecast(variable, 'weather_tree', STEP, fold['number'])
            if sources is not None:
                sources[str(fold['number'])] = {k: rec[k] for k in ('source', 'sha256', 'study', 'completed_trials', 'params')}
            params = rec['params']
        else:
            params = selected(variable, 'weather_tree', STEP)
        model = fit_step_tree(data, design, fold, params)
        yield fold, design, model


def six_hour_check(variable, perfold=False):
    """The ungated six-hour step of the corrected single-probe rollout against the addendum-4
    rollout's six-hour prediction, on every addendum-4 episode (both start from the onset row).
    With `perfold` both sides are the per-fold runs (the rollout from runs/v2_closed_loop/perfold/)."""
    data = build(variable, STEP)
    met = weather(data['times']).to_numpy()
    loop = ROOT / 'reports/bay_twin/runs/v2_closed_loop'
    if perfold:
        # either variant of the per-fold rollout: the neighbour switch acts only after onset
        loop = loop / 'perfold'
        candidates = [loop / f'closed_loop_{variable}{s}.csv.gz' for s in ('', '_rolled_neighbour')]
        source = next((p for p in candidates if p.exists()), candidates[0])
    else:
        source = loop / f'closed_loop_{variable}.csv.gz'
    published = pd.read_csv(source)
    published = published[published.lead == STEP].set_index(['fold', 'sensor', 'time']).recursive_tree
    companion = True if variable == 'oxygen' else None
    rows = []
    for fold, design, model in fold_models(data, folds(data, STEP), met, variable, perfold):
        limits = bounds(data, fold)
        for (i0, s) in episodes(data, fold, 48):
            key = (fold['number'], data['names'][s], str(data['times'][i0]))
            if key not in published.index:
                continue
            paths, _ = roll(data, design, model, i0, [int(s)], 1, 'none', limits, 'single',
                            fold['fallback'], companion=companion, issuance=True)
            rows.append({'fold': key[0], 'sensor': key[1], 'time': key[2],
                         'gate_rollout': float(paths[int(s)][0] * data['scale'] + data['center']),
                         'closed_loop': float(published.loc[key])})
    f = pd.DataFrame(rows)
    d = (f.gate_rollout - f.closed_loop).abs()
    out = {'episodes_compared': int(len(f)), 'episodes_in_closed_loop': int(len(published)),
           'max_abs_diff': float(d.max()), 'mean_abs_diff': float(d.mean()),
           'rows_above_1e-9': int((d > 1e-9).sum())}
    if perfold:
        out['closed_loop_source'] = source.relative_to(ROOT).as_posix()
        record_check(f'{variable}|six_hour_vs_closed_loop', out, folder(True) / 'issuance_gate.json')
    else:
        record_check(f'{variable}|six_hour_vs_closed_loop', out)
    print(variable, 'six-hour step against the addendum-4 rollout:', out)
    return out


def run(variable, regime, max_lead=120, quick=False, issuance=False, check=False, perfold=False):
    """`perfold`: each fold's step tree at that fold's own selection, everything written under
    folder(True, quick) (module docstring); the default writes under OUT, as published."""
    out_dir = folder(perfold, quick)
    out_dir.mkdir(parents=True, exist_ok=True)
    steps = max_lead // STEP
    data = build(variable, STEP)
    outer = folds(data, STEP)
    met = weather(data['times']).to_numpy()
    companion = True if variable == 'oxygen' else None
    sources = {}

    rows, fire_log = [], []
    evaluable = {g: {'steps': 0, 'rate': 0, 'range': 0, 'saturation': 0} for g in GATES}
    for fold, design, model in fold_models(data, outer, met, variable, perfold, sources):
        limits = bounds(data, fold)
        eps = episodes(data, fold, max_lead)
        if quick: eps = eps[:25]
        eligible = np.flatnonzero(fold['eligible'])

        seen = set()
        for (i0, s) in eps:
            if regime == 'network':
                if i0 in seen:
                    continue
                seen.add(i0)
                sensors = [int(j) for j in eligible]
            else:
                sensors = [int(s)]
            for gate in GATES:
                paths, fires = roll(data, design, model, i0, sensors, steps, gate, limits, regime,
                                    fold['fallback'], companion=companion, evaluable=evaluable[gate],
                                    issuance=issuance)
                for f in fires:
                    fire_log.append({**f, 'gate_set': gate, 'issuance': str(data['times'][i0])})
                for j in sensors:
                    for k in range(steps):
                        if not data['valid'][i0 + k, j]:
                            continue
                        rows.append({'fold': fold['number'], 'sensor': data['names'][j],
                                     'time': str(data['times'][i0]), 'lead': (k + 1) * STEP, 'gate': gate,
                                     'truth': float(data['truth'][i0 + k, j] * data['scale'] + data['center']),
                                     'prediction': float(paths[j][k] * data['scale'] + data['center'])})

    frame = pd.DataFrame(rows)
    frame['error'] = frame.prediction - frame.truth
    suffix = '_issuance' if issuance else ''
    if check:
        published = pd.read_csv(out_dir / f'gate_{variable}_{regime}.csv.gz')
        key = ['fold', 'sensor', 'time', 'lead', 'gate']
        joined = published.merge(frame, on=key, how='outer', suffixes=('', '_replay'), indicator=True)
        d = (joined.prediction - joined.prediction_replay).abs()
        out = {'rows_published': int(len(published)), 'rows_replayed': int(len(frame)),
               'unmatched': int((joined._merge != 'both').sum()), 'prediction_max_abs_diff': float(d.max()),
               'truth_max_abs_diff': float((joined.truth - joined.truth_replay).abs().max()),
               'fires_replayed': len(fire_log), 'evaluable_steps_replayed': evaluable}
        if issuance:
            out['rows_changed_above_1e-9'] = int((d > 1e-9).sum())
        else:
            out['passed'] = bool(out['unmatched'] == 0 and out['prediction_max_abs_diff'] <= 1e-9
                                 and out['truth_max_abs_diff'] <= 1e-9)
        if perfold:
            record_check(f"{variable}|{regime}|{'issuance' if issuance else 'published'}", out,
                         out_dir / 'issuance_gate.json')
        else:
            record_check(f"{variable}|{regime}|{'issuance' if issuance else 'published'}", out)
        print(variable, regime, 'replay against the published run:', out)
        return
    path = out_dir / f'gate_{variable}_{regime}{suffix}.csv.gz'
    if issuance or perfold:
        frame.to_csv(path, index=False, lineterminator='\n', compression=GZIP)
    else:
        frame.to_csv(path, index=False)

    report = {'declared_in': 'config/bay_reserved_protocol_v2_addendum5.json', 'variable': variable,
              'regime': regime, 'max_lead': max_lead,
              'episodes': int(frame.groupby(['fold', 'sensor', 'time']).ngroups),
              'rows_per_gate': int(len(frame) / len(GATES)), 'by_lead': {}}
    if perfold:
        report['selection'] = {'design': 'perfold',
                               'objective': "fold-macro forecast MAE / persistence MAE on that outer fold's own inner window only",
                               'records': {f'{variable}|weather_tree': sources}}
        report['smoke_only'] = bool(quick)
    for lead, sub in frame.groupby('lead'):
        report['by_lead'][int(lead)] = {g: tail(gg.error.to_numpy()) for g, gg in sub.groupby('gate')}

    fires = pd.DataFrame(fire_log)
    report['fire_counts'] = ({} if fires.empty else
                             {g: int(len(gg)) for g, gg in fires.groupby('gate_set')})
    report['fire_by_type'] = ({} if fires.empty else
                              {t: int(sum(t in f for f in fires[fires.gate_set == 'all'].gates))
                               for t in ['rate', 'range', 'saturation']})
    # the denominators: on how many rolled steps each gate of each set could act at all
    report['evaluable_steps'] = evaluable
    if issuance:
        report['neighbours'] = ('read at each step\'s issuance hour; a probe that is not rolled enters '
                                'only where it reports; rolled probes at their rolled state')
    report['prediction_sha256'] = hashlib.sha256(path.read_bytes()).hexdigest()
    (save_lf if (issuance or perfold) else save)(out_dir / f'gate_{variable}_{regime}{suffix}_summary.json', report)
    if not fires.empty:
        fires_path = out_dir / f'gate_{variable}_{regime}{suffix}_fires.csv.gz'
        if issuance or perfold:
            fires.to_csv(fires_path, index=False, lineterminator='\n', compression=GZIP)
        else:
            fires.to_csv(fires_path, index=False)

    print(f"== {variable} / {regime}  episodes={report['episodes']}  rows/gate={report['rows_per_gate']}")
    print(f"{'lead':>5s} " + ' '.join(f'{g:>10s}' for g in GATES) + '   (mae / p99 / max)')
    for lead in sorted(report['by_lead']):
        e = report['by_lead'][lead]
        print(f'{lead:5d} ' + ' '.join(f"{e[g]['mae']:10.3f}" for g in GATES) +
              '   p99 ' + ' '.join(f"{e[g]['p99']:7.2f}" for g in GATES) +
              '   max ' + ' '.join(f"{e[g]['max']:7.2f}" for g in GATES))
    print('gate fires:', report['fire_counts'], 'by type (all):', report['fire_by_type'])


RANGE_DRAWS = 600
RANGE_EFFECT = OUT / 'range_bound_effect.json'


def range_effect(suffix='_issuance', perfold=False):
    """The range bound against no gate, lead by lead, on the same rows of the same rollout.

    The statistic is the one the summaries report, the mean absolute error over the scored rows of
    a lead, range-gated minus ungated. Its interval is a paired bootstrap over onset weeks (ISO week
    of the issuance at onset), drawn with replacement jointly within each fold, every row of a drawn
    week entering together, with the folds fixed. Firings are counted over the steps on which the
    bound could act, from the run's own summary. With `perfold` the per-fold runs are read and the
    result is written beside them (the same draws: the seed depends on variable, regime and lead).
    """
    source = folder(perfold)
    out = {'statistic': 'mean absolute error over the scored rows of a lead, range bound minus no gate',
           'unit': 'onset week (ISO week of the issuance at onset), drawn jointly within each fold',
           'draws': RANGE_DRAWS, 'runs': suffix.strip('_') or 'published', 'results': {}}
    if perfold:
        out['selection'] = 'perfold'
    for variable in ('temperature', 'oxygen'):
        for regime in ('network', 'single'):
            name = f'gate_{variable}_{regime}{suffix}'
            frame = pd.read_csv(source / f'{name}.csv.gz')
            summary = json.loads((source / f'{name}_summary.json').read_text())
            wide = (frame[frame.gate.isin(['none', 'range'])]
                    .pivot_table(index=['fold', 'sensor', 'time', 'lead'], columns='gate', values='error')
                    .reset_index())
            assert wide[['none', 'range']].notna().all(axis=None), name
            iso = pd.to_datetime(wide.time, utc=True).dt.isocalendar()
            wide['week'] = iso.year.astype(str) + '-' + iso.week.astype(str)
            wide['d'] = wide.range.abs() - wide.none.abs()
            entry = {'firings': int(summary['fire_counts'].get('range', 0)),
                     'evaluable_steps': int(summary['evaluable_steps']['range']['range']),
                     'by_lead': {}}
            for lead, sub in wide.groupby('lead'):
                # seeded by variable, regime and lead only, so two runs give the same draws
                rng = np.random.default_rng([20261001, zlib.crc32(f'{variable}|{regime}|{lead}'.encode())])
                folds_ = []
                for _, f in sub.groupby('fold'):
                    g = f.groupby('week').d.agg(['sum', 'size'])
                    folds_.append(g[['sum', 'size']].to_numpy(float))
                draws = np.empty(RANGE_DRAWS)
                for i in range(RANGE_DRAWS):
                    tot = np.zeros(2)
                    for table in folds_:
                        tot += table[rng.integers(0, len(table), len(table))].sum(axis=0)
                    draws[i] = tot[0] / tot[1]
                lo, hi = np.percentile(draws, [2.5, 97.5])
                none_mae = float(sub.none.abs().mean()); range_mae = float(sub.range.abs().mean())
                entry['by_lead'][str(int(lead))] = {
                    'rows': int(len(sub)), 'weeks': int(sub.groupby('fold').week.nunique().sum()),
                    'rows_changed': int((sub.range != sub.none).sum()),
                    'mae_none': none_mae, 'mae_range': range_mae, 'point': range_mae - none_mae,
                    'ci': [float(lo), float(hi)], 'excludes_zero': bool(lo > 0 or hi < 0)}
            out['results'][f'{variable}|{regime}'] = entry
            six = entry['by_lead']['6']
            print(f"{variable:11s} {regime:7s} fires {entry['firings']}/{entry['evaluable_steps']}  6 h "
                  f"{six['mae_none']:.4f} -> {six['mae_range']:.4f}  {six['point']:+.4f} "
                  f"[{six['ci'][0]:+.4f}, {six['ci'][1]:+.4f}]")
    target = source / RANGE_EFFECT.name if perfold else RANGE_EFFECT
    save_lf(target, out)
    print('wrote', target)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--variable', choices=['temperature', 'oxygen'])
    p.add_argument('--regime', choices=['single', 'network'])
    p.add_argument('--max-lead', type=int, default=120)
    p.add_argument('--quick', action='store_true')
    p.add_argument('--issuance', action='store_true',
                   help='neighbours at the issuance hour, silent probes left out of their mean')
    p.add_argument('--check', action='store_true', help='replay against the published run; write no predictions')
    p.add_argument('--six-hour-check', action='store_true',
                   help='corrected single-probe six-hour step against the addendum-4 rollout')
    p.add_argument('--range-effect', action='store_true', help='paired onset-week bootstrap of the range bound')
    p.add_argument('--perfold', action='store_true',
                   help="each fold's tree at its own selection; reads and writes runs/v2_physics_gate/perfold/")
    a = p.parse_args()
    if a.range_effect:
        range_effect(perfold=a.perfold) if a.perfold else range_effect()
    elif a.six_hour_check:
        six_hour_check(a.variable, a.perfold) if a.perfold else six_hour_check(a.variable)
    else:
        run(a.variable, a.regime, a.max_lead, a.quick, a.issuance, a.check, a.perfold)
