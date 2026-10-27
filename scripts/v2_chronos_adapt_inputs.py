"""Addendum 41: build, locally, every array the cluster run needs, with the published code paths.

  continuation contexts   the probe's own hourly series over the 512 h ending at each addendum-39
                          onset, from the hourly frames v2_bridge.hourly reads (gate G1: the last
                          value equals the episode's persistence column)
  gap contexts            the forward and reversed 512 h windows of every Gap-2026 episode, built
                          line for line as v2_gap_recovery.chronos_fill builds them, in its order
  training panels         the development panel of each variable (load_observations, pre-2026),
                          from which the cluster samples the fine-tuning windows (gate G3)

The cluster receives numbers only, so what it scores cannot drift from what the comparands saw.
Writes reports/bay_twin/runs/v2_chronos_adapt/inputs/*.npz and a manifest of their digests.

Usage: python -m scripts.v2_chronos_adapt_inputs
"""
import hashlib
import json

import numpy as np
import pandas as pd

from src.data.bay_profile_context import ROOT, load_observations
from src.experiments import v2_bridge
from src.experiments import v2_gap_recovery as gap

RUNS = ROOT / 'reports/bay_twin/runs'
OUT = RUNS / 'v2_chronos_adapt' / 'inputs'
GAP_RUN = RUNS / '20260922T180814Z_v2_gaprecovery' / 'v2_gap_recovery_reserved2026'
CTX = 512
CUT = pd.Timestamp('2026-01-01', tz='UTC')
VARIABLES = ('temperature', 'oxygen')


def continuation(variable):
    """512 h of the probe's own record ending at each onset; NaN where the probe did not report."""
    rows = pd.read_csv(RUNS / 'v2_bridge' / f'bridge_{variable}.csv.gz')
    rows['onset'] = pd.to_datetime(rows.onset, utc=True)
    episodes = rows.drop_duplicates(['cohort', 'fold', 'sensor', 'onset'])[
        ['cohort', 'fold', 'sensor', 'onset', 'persistence']].reset_index(drop=True)
    contexts = np.full((len(episodes), CTX), np.nan, dtype=np.float64)
    for cohort, key in (('Bridge-Dev', 'dev'), ('Bridge-2026', 'reserved')):
        part = episodes[episodes.cohort == cohort]
        names = sorted(part.sensor.unique())
        index = v2_bridge.hourly_index(key)
        frame, _, _ = v2_bridge.hourly(names, variable, index, key)
        values = frame.to_numpy(float)
        column = {n: i for i, n in enumerate(names)}
        position = index.get_indexer(part.onset)
        assert (position >= 0).all(), 'onset outside the hourly index'
        for row, (i, p) in zip(part.index, zip(part.sensor.map(column), position)):
            window = values[max(0, p - CTX + 1):p + 1, i]
            contexts[row, CTX - len(window):] = window
    last = contexts[:, -1]
    g1 = bool(np.all(np.abs(last - episodes.persistence.to_numpy()) <= 1e-6))
    if not g1:
        bad = np.flatnonzero(~(np.abs(last - episodes.persistence.to_numpy()) <= 1e-6))
        raise SystemExit(f'G1 failed for {variable}: {len(bad)} episodes, first {episodes.iloc[bad[0]].to_dict()}')
    return episodes, contexts.astype(np.float32), g1


def gap_contexts(variable):
    """chronos_fill's windows for the Gap-2026 episodes, in chronos_fill's order."""
    frame = pd.read_csv(GAP_RUN / f'{variable}_chronos_bidir' / 'predictions.csv.gz')
    meta, frames = gap.observations('reserved')
    names = [n for n, f in frames.items() if variable in f and f[variable].notna().any()]
    index = pd.date_range(min(frames[n].index.min() for n in names),
                          max(frames[n].index.max() for n in names), freq='h', tz='UTC')
    panel = pd.DataFrame({n: frames[n][variable].reindex(index) for n in names}, index=index)
    position_of = {t: i for i, t in enumerate(index)}
    keys, sensors, starts, lengths, fwd, bwd, fwd_len, bwd_len = [], [], [], [], [], [], [], []
    for key in frame.episode.unique():
        sensor, start, length = key.rsplit('|', 2)
        g0, length = position_of[pd.Timestamp(start)], int(length)
        series = panel[sensor].to_numpy(float)
        pair = []
        for direction in ('fwd', 'bwd'):
            window = (series[max(0, g0 - CTX):g0] if direction == 'fwd'
                      else series[g0 + length:g0 + length + CTX][::-1])
            filled = pd.Series(window).interpolate(limit_direction='both').to_numpy()
            pair.append(np.nan_to_num(filled).astype(np.float32))
        padded = [np.concatenate([np.full(CTX - len(w), np.nan, np.float32), w]) for w in pair]
        keys.append(key); sensors.append(sensor); starts.append(g0); lengths.append(length)
        fwd.append(padded[0]); bwd.append(padded[1]); fwd_len.append(len(pair[0])); bwd_len.append(len(pair[1]))
    return dict(keys=np.array(keys), sensors=np.array(sensors), lengths=np.array(lengths),
                fwd=np.stack(fwd), bwd=np.stack(bwd), fwd_len=np.array(fwd_len), bwd_len=np.array(bwd_len))


def training_panel(variable):
    _, frames = load_observations()
    names = [n for n, f in frames.items() if variable in f and f[variable].notna().any()]
    index = pd.date_range(min(frames[n].index.min() for n in names),
                          max(frames[n].index.max() for n in names), freq='h', tz='UTC')
    panel = pd.DataFrame({n: frames[n][variable].reindex(index) for n in names}, index=index)
    assert panel.index.max() < CUT, 'the development panel reaches 2026'
    return dict(names=np.array(names), times=panel.index.as_unit('ns').asi8, values=panel.to_numpy(np.float32))


def configurations():
    """The configurations the six-hour studies selected on validation loss, reused on Vito's instruction."""
    ft = json.loads((ROOT / 'reports/bay_twin/runs/battery/v2_chronos/ft_diagnostic.json').read_text())
    lora = json.loads((RUNS / 'v2_chronos_lora/v2_chronos_lora_report.json').read_text())
    out = {'full': {}, 'lora': {}}
    for variable, space in (('temperature', 'raw'), ('oxygen', 'aou')):
        sel = ft[f'chronos_{variable}_{space}']['selected']
        out['full'][variable] = {'source': f'ft_diagnostic.json chronos_{variable}_{space} trial {sel["trial"]}',
                                 'params': sel['params']}
        s = lora[f'lora_{variable}_{space}']['selection']
        out['lora'][variable] = {'source': f'v2_chronos_lora_report.json lora_{variable}_{space} trial {s["best_trial"]}',
                                 'params': s['params']}
    return out


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    manifest = {'declared_in': 'config/bay_reserved_protocol_v2_addendum41.json', 'files': {}, 'gates': {}}
    (OUT / 'configurations.json').write_text(json.dumps(configurations(), indent=2))
    for variable in VARIABLES:
        episodes, contexts, g1 = continuation(variable)
        np.savez_compressed(OUT / f'continuation_{variable}.npz', contexts=contexts,
                            cohort=episodes.cohort.to_numpy(), fold=episodes.fold.to_numpy(),
                            sensor=episodes.sensor.to_numpy(), onset=episodes.onset.astype(str).to_numpy())
        np.savez_compressed(OUT / f'gap_{variable}.npz', **gap_contexts(variable))
        panel = training_panel(variable)
        np.savez_compressed(OUT / f'panel_{variable}.npz', **panel)
        manifest['gates'][f'G1_{variable}'] = {'passed': g1, 'episodes': int(len(episodes))}
        manifest['gates'][f'G3_{variable}'] = {'passed': True, 'panel_last_hour': str(pd.Timestamp(int(panel['times'].max()), tz='UTC')),
                                               'probes': int(len(panel['names']))}
        print(variable, len(episodes), 'continuation episodes;', len(panel['names']), 'training probes')
    for path in sorted([*OUT.glob('*.npz'), OUT / 'configurations.json']):
        manifest['files'][path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
    (OUT / 'manifest.json').write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest['gates'], indent=2))


if __name__ == '__main__':
    main()
