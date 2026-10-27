"""Addendum 81, task D: the inputs of the silent-probe continuation (Bridge-Dev and Bridge-2026, 6 to 72 h).

Contexts are addendum 41's continuation contexts, runs/v2_chronos_adapt/inputs/continuation_{variable}.npz:
the probe's own hourly series over the 512 h ending at each addendum-39 onset, raw, NaN kept, last value
the reading at onset. They are rebuilt here by the published function
(scripts/v2_chronos_adapt_inputs.py: continuation, lines 34-58) and must equal the stored file exactly.
Lead h is horizon position h - 1 of a 72 h horizon, i.e. the hour onset + h.

Covariates (cov mode), as the declaration states for task D: the mean and the spread of the OTHER
probes' readings of the same variable, hour by hour. The other probes are the probes of the cohort's
six-hour grid that the anchored continuation reads (v2_bridge.Context.names, from v2_bridge.grid), less
the silent probe; their readings are the hourly frames v2_bridge.hourly reads (never forward-filled).
The spread is pandas' sample standard deviation across probes, the definition of the network spread of
the published Chronos-2 covariate arms (exp_chronos.NET.net_sd, v2_weather_oracle.WeatherPanel). An
hour where no other probe reports has a NaN mean; an hour with fewer than two has a NaN spread.
Nothing is filled. Past covariates are aligned to the context (the 512 h ending at onset); future
covariates run over the hours onset + 1 to onset + 72, NaN past the end of the cohort's hourly index.

A silent instrument is silent in every channel. The channels that come from it, and how each is kept out
after onset:
  its own target series   the context ends at onset (gate D1: rebuilt, last value = the persistence)
  its readings in the mean and spread   dropped at every hour, before and after onset (gate D4:
                          the covariates are unchanged when the silent column is replaced by a sentinel)
  its own temperature and salinity (oxygen sondes)   not in this covariate set
  a duplicate of the same instrument under another name   checked (gate D5: no other column agrees
                          with the silent probe's exactly on 95 % or more of their common hours, which
                          a second export of one instrument would; co-located twins quantised at 0.1 degC
                          agree on some hours and are separate instruments)

Gates, all required:
  D1  the published builder reproduces the stored contexts and keys exactly (and its own G1)
  D2  every episode of runs/v2_bridge/bridge_{v}.csv.gz and runs/v2_bridge72/bridge72_{v}.csv.gz is
      present, and the file holds no other episode
  D3  every scored row's truth equals the hourly frame at onset + lead (horizon alignment)
  D4  the silent probe never enters a covariate (sentinel test over every episode)
  D5  no other probe duplicates the silent probe's record

Writes reports/bay_twin/runs/v2_fm_zoo/inputs/D_{variable}_{cohort}.npz and the task's section of
manifest_dist_bridge.json in the same folder.

Usage: python -m scripts.v2_fm_zoo_inputs_bridge
"""
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.v2_fm_zoo_io import digest, load_inputs, save_inputs  # noqa: E402

RUNS = ROOT / 'reports/bay_twin/runs'
OUT = RUNS / 'v2_fm_zoo/inputs'
MANIFEST = OUT / 'manifest_dist_bridge.json'
STORED = RUNS / 'v2_chronos_adapt/inputs'
CTX, HORIZON = 512, 72
COHORTS = {'Bridge-Dev': 'dev', 'Bridge-2026': 'reserved'}
KEY = ['cohort', 'fold', 'sensor', 'onset']
COV_NAMES = ['others_mean', 'others_sd']
UNITS = {'temperature': 'degC', 'oxygen': 'umol/L (the sondes stored column)'}
SENTINEL = 1.0e6

OPENED = set()


def _trace(event, args):
    if event == 'open' and args and isinstance(args[0], (str, Path)):
        OPENED.add(str(args[0]))


def data_sources():
    """Every data file the build opened, relative to ROOT; the builder's own outputs excluded."""
    out = []
    data_target = Path(os.path.realpath(ROOT / 'data'))     # the junction resolves outside the repository
    for p in sorted(OPENED):
        q = Path(p)
        if not q.is_absolute():
            q = ROOT / q
        if q.suffix in ('.py', '.pyc', '.pyd', '.dll', '.so') or not q.is_file():
            continue
        try:
            rel = q.relative_to(ROOT)
        except ValueError:
            try:
                rel = Path('data') / q.relative_to(data_target)
            except ValueError:
                continue
        if rel.parts[0] in ('data', 'reports') and 'v2_fm_zoo' not in rel.parts:
            out.append(rel.as_posix())
    return sorted(set(out))


def stored(variable):
    z = np.load(STORED / f'continuation_{variable}.npz', allow_pickle=True)
    keys = pd.DataFrame({'cohort': z['cohort'].astype(str), 'fold': z['fold'].astype('int64'),
                         'sensor': z['sensor'].astype(str),
                         'onset': pd.to_datetime(pd.Series(z['onset'].astype(str)), utc=True)})
    keys['onset'] = keys.onset.astype('datetime64[ns, UTC]')
    return keys, np.asarray(z['contexts'], dtype=np.float32)


def bridge_rows(variable):
    out = []
    for path in (RUNS / 'v2_bridge' / f'bridge_{variable}.csv.gz', RUNS / 'v2_bridge72' / f'bridge72_{variable}.csv.gz'):
        f = pd.read_csv(path, usecols=KEY + ['lead', 'truth', 'persistence'])
        f['onset'] = pd.to_datetime(f.onset, utc=True).astype('datetime64[ns, UTC]')
        f['source'] = path.relative_to(ROOT).as_posix()
        out.append(f)
    return pd.concat(out, ignore_index=True)


def gate_rebuild(variable, keys, contexts):
    from scripts.v2_chronos_adapt_inputs import continuation
    episodes, rebuilt, g1 = continuation(variable)
    episodes = episodes.copy()
    episodes['onset'] = episodes.onset.astype('datetime64[ns, UTC]')
    same_keys = bool((episodes[KEY].reset_index(drop=True).astype(str) == keys[KEY].astype(str)).all().all())
    same_ctx = bool(np.array_equal(rebuilt, contexts, equal_nan=True))
    last = contexts[:, -1].astype(float)
    persist = episodes.persistence.to_numpy(float)
    out = {'episodes': int(len(keys)), 'keys_identical_in_order': same_keys, 'contexts_identical': same_ctx,
           'published_G1_last_value_is_persistence': bool(g1),
           'max_abs_last_minus_persistence': float(np.max(np.abs(last - persist))),
           'context_length': int(contexts.shape[1])}
    out['passed'] = bool(same_keys and same_ctx and g1 and contexts.shape[1] == CTX)
    return out


def gate_episodes(keys, rows):
    have = set(map(tuple, keys[KEY].astype(str).to_numpy()))
    out, ok = {}, True
    for source, part in rows.groupby('source'):
        eps = set(map(tuple, part[KEY].drop_duplicates().astype(str).to_numpy()))
        missing = len(eps - have)
        out[source] = {'episodes': len(eps), 'missing_from_contexts': missing, 'rows': int(len(part)),
                       'leads': sorted(int(h) for h in part.lead.unique())}
        ok &= missing == 0
    every = set(map(tuple, rows[KEY].drop_duplicates().astype(str).to_numpy()))
    out['contexts_without_a_scored_row'] = len(have - every)
    out['passed'] = bool(ok and out['contexts_without_a_scored_row'] == 0)
    return out


class Others:
    """Per cohort: the grid's probes, their hourly frame, and per silent probe the mean and spread of the rest."""

    def __init__(self, variable, key):
        from src.experiments import v2_bridge
        self.names = list(v2_bridge.grid(variable, key)['names'])        # v2_bridge.Context.names
        self.index = v2_bridge.hourly_index(key)
        self.frame, _, _ = v2_bridge.hourly(self.names, variable, self.index, key)
        self._cache = {}

    @staticmethod
    def stats(frame, silent):
        others = frame.drop(columns=[silent])
        return np.stack([others.mean(axis=1).to_numpy(float), others.std(axis=1, ddof=1).to_numpy(float)])

    def of(self, silent):
        if silent not in self._cache:
            self._cache[silent] = self.stats(self.frame, silent)
        return self._cache[silent]

    def windows(self, series, p):
        """[C, 512] ending at position p, and [C, 72] over p + 1 .. p + 72, NaN outside the index."""
        c, n = series.shape[0], series.shape[1]
        past = np.full((c, CTX), np.nan)
        a = max(0, p - CTX + 1)
        past[:, CTX - (p + 1 - a):] = series[:, a:p + 1]
        future = np.full((c, HORIZON), np.nan)
        b = min(n, p + 1 + HORIZON)
        future[:, :b - (p + 1)] = series[:, p + 1:b]
        return past.astype(np.float32), future.astype(np.float32)


def build(variable):
    keys, contexts = stored(variable)
    rows = bridge_rows(variable)
    gates = {'D1_rebuild': gate_rebuild(variable, keys, contexts), 'D2_episodes': gate_episodes(keys, rows)}
    past, future = [None] * len(keys), np.full((len(keys), len(COV_NAMES), HORIZON), np.nan, np.float32)
    align = {'rows': 0, 'max_abs_truth_difference': 0.0, 'missing_truth': 0}
    sentinel = {'episodes': 0, 'changed': 0}
    duplicates = {}
    coverage = {}
    for cohort, key in COHORTS.items():
        others = Others(variable, key)
        sel = np.flatnonzero(keys.cohort.to_numpy() == cohort)
        pos = others.index.get_indexer(pd.DatetimeIndex(keys.onset.iloc[sel]))
        if (pos < 0).any():
            raise SystemExit(f'{variable} {cohort}: {int((pos < 0).sum())} onsets outside the hourly index')
        missing = sorted(set(keys.sensor.iloc[sel]) - set(others.names))
        if missing:
            raise SystemExit(f'{variable} {cohort}: silent probes not on the grid: {missing}')
        values = others.frame.to_numpy(float)
        col = {n: i for i, n in enumerate(others.names)}
        for r, p in zip(sel, pos):
            s = keys.sensor.iloc[r]
            past[r], future[r] = others.windows(others.of(s), p)
            # D1 again, from the frame the covariates read: the context is this probe's column ending at p
            own = np.full(CTX, np.nan)
            a = max(0, p - CTX + 1)
            own[CTX - (p + 1 - a):] = values[a:p + 1, col[s]]
            if not np.array_equal(own.astype(np.float32), contexts[r], equal_nan=True):
                raise SystemExit(f'{variable} {cohort} {s} {keys.onset.iloc[r]}: context differs from the covariate frame')
        # D3: horizon position h - 1 is the hour onset + h (the truth of every scored row)
        part = rows[rows.cohort == cohort].reset_index(drop=True)
        p0 = others.index.get_indexer(pd.DatetimeIndex(part.onset))
        hours = p0 + part.lead.to_numpy()
        inside = hours < len(others.index)
        got = np.full(len(part), np.nan)
        got[inside] = values[hours[inside], part.sensor.map(col).to_numpy()[inside]]
        d = np.abs(got - part.truth.to_numpy(float))
        align['rows'] += int(len(part))
        align['missing_truth'] += int((~np.isfinite(got)).sum())
        align['max_abs_truth_difference'] = max(align['max_abs_truth_difference'], float(np.nanmax(d)) if len(d) else 0.0)
        # D4: the silent column replaced by a sentinel at every hour leaves every covariate unchanged
        for s in sorted(set(keys.sensor.iloc[sel])):
            poisoned = others.frame.copy()
            poisoned[s] = SENTINEL
            again = others.stats(poisoned, s)
            sentinel['episodes'] += int((keys.sensor.iloc[sel] == s).sum())
            sentinel['changed'] += int(not np.array_equal(again, others.of(s), equal_nan=True))
            # D5: share of common hours on which another column equals the silent one exactly
            x = values[:, col[s]]
            shares = {}
            for n in others.names:
                if n == s:
                    continue
                y = values[:, col[n]]
                both = np.isfinite(x) & np.isfinite(y)
                if both.sum() >= 24:
                    shares[n] = float(np.mean(x[both] == y[both]))
            top = max(shares.items(), key=lambda kv: kv[1]) if shares else (None, 0.0)
            duplicates[f'{cohort}|{s}'] = {'most_similar': top[0], 'share_equal': round(top[1], 6)}
        fp = np.stack([past[r] for r in sel], axis=0)
        coverage[cohort] = {'episodes': int(len(sel)), 'other_probes_on_grid': len(others.names) - 1,
                            'past_mean_nan_share': float(np.isnan(fp[:, 0]).mean()),
                            'past_sd_nan_share': float(np.isnan(fp[:, 1]).mean()),
                            'future_mean_nan_share': float(np.isnan(future[sel, 0]).mean()),
                            'future_sd_nan_share': float(np.isnan(future[sel, 1]).mean()),
                            'episodes_whose_horizon_passes_the_index_end': int(np.sum(pos + HORIZON >= len(others.index))),
                            'index_end': str(others.index[-1]), 'grid_probes': others.names}
    align['passed'] = bool(align['missing_truth'] == 0 and align['max_abs_truth_difference'] <= 1e-6)
    gates['D3_horizon_alignment'] = align
    sentinel['passed'] = sentinel['changed'] == 0
    gates['D4_silent_probe_never_a_covariate'] = sentinel
    worst = max(v['share_equal'] for v in duplicates.values())
    gates['D5_no_duplicate_instrument'] = {'max_share_equal': worst, 'per_probe': duplicates, 'passed': bool(worst < 0.95)}
    return keys, contexts, past, future, gates, coverage


def main():
    sys.addaudithook(_trace)
    OUT.mkdir(parents=True, exist_ok=True)
    section = {'task': 'D_silent_probe', 'declared_in': 'config/bay_reserved_protocol_v2_addendum81.json',
               'builder': 'scripts/v2_fm_zoo_inputs_bridge.py',
               'published_functions': ['scripts/v2_chronos_adapt_inputs.py:continuation (34-58)',
                                       'src/experiments/v2_bridge.py:grid (69-73), hourly (76-83), hourly_index (86-88)'],
               'silent_instrument_channels': {
                   'target series': 'context ends at onset (D1)',
                   'its readings in the mean and spread': 'dropped at every hour (D4)',
                   'own temperature and salinity (oxygen)': 'not in this covariate set',
                   'same instrument under another name': 'none found (D5)',
                   'co-located twin probes': 'other instruments, kept as other probes (the anchored continuation reads them too)'},
               'files': {}, 'gates': {}, 'coverage': {}}
    for variable in ('temperature', 'oxygen'):
        keys, contexts, past, future, gates, coverage = build(variable)
        section['gates'][variable] = gates
        section['coverage'][variable] = coverage
        print(variable, json.dumps({k: v.get('passed') for k, v in gates.items()}), flush=True)
        if not all(v['passed'] for v in gates.values()):
            raise SystemExit(f'task D gates failed for {variable}: {json.dumps(gates, default=str)[:3000]}')
        sources = sorted(set([f'reports/bay_twin/runs/v2_chronos_adapt/inputs/continuation_{variable}.npz',
                              f'reports/bay_twin/runs/v2_bridge/bridge_{variable}.csv.gz',
                              f'reports/bay_twin/runs/v2_bridge72/bridge72_{variable}.csv.gz',
                              'scripts/v2_chronos_adapt_inputs.py', 'src/experiments/v2_bridge.py',
                              'scripts/v2_fm_zoo_inputs_bridge.py'] + data_sources()))
        for cohort in COHORTS:
            sel = np.flatnonzero(keys.cohort.to_numpy() == cohort)
            path = OUT / f'D_{variable}_{cohort}.npz'
            meta = save_inputs(
                path, task='D_silent_probe', variable=variable, cohort=cohort,
                keys=keys.iloc[sel][KEY].reset_index(drop=True), contexts=[contexts[i] for i in sel],
                horizon=HORIZON, units=UNITS[variable], past_cov=[past[i] for i in sel], future_cov=future[sel],
                cov_names=COV_NAMES, standardisation=None, sources=sources,
                note=('Addendum 41 continuation contexts: 512 h of the silent probe ending at onset (UTC), raw, '
                      'NaN kept; lead h (6..72 in six-hour steps, scored where observed) is horizon position '
                      'h - 1, the hour onset + h. Covariates: mean and sample standard deviation (ddof 1) of the '
                      'other grid probes of the same variable, hour by hour, the silent probe excluded at every '
                      'hour; NaN where no other probe reports (mean) or fewer than two (spread); nothing filled. '
                      'Past covariates align with the context; future covariates cover onset + 1 .. onset + 72 '
                      'h. A live system has the value at hour onset + k only from that hour on: a forecast for '
                      'lead h that reads future covariates beyond position h - 1 reads hours after its target. '
                      'Rows in the order of the stored continuation file.'))
            back = load_inputs(path)
            assert back['meta']['n'] == len(sel) and back['future_cov'].shape == (len(sel), len(COV_NAMES), HORIZON)
            section['files'][path.name] = {'n': int(len(sel)), 'horizon': HORIZON, 'cov_names': COV_NAMES,
                                           'context_length': CTX, 'sha256': digest(path),
                                           'sensors': int(keys.sensor.iloc[sel].nunique())}
            print(' ', path.name, len(sel), 'episodes', flush=True)
    manifest = json.loads(MANIFEST.read_text(encoding='utf-8')) if MANIFEST.exists() else {}
    manifest['declared_in'] = 'config/bay_reserved_protocol_v2_addendum81.json'
    manifest['D_silent_probe'] = section
    with open(MANIFEST, 'w', encoding='utf-8', newline='\n') as f:
        f.write(json.dumps(manifest, indent=2, default=str) + '\n')
    print('manifest', MANIFEST.relative_to(ROOT).as_posix())


if __name__ == '__main__':
    main()
