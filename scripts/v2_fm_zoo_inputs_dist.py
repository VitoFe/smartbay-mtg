"""Addendum 81, task C: the inputs of the forecast distribution to 72 h (Dev-2025 and the Reserved-2026 grid).

Rows and contexts are those of addendum 65's Chronos-2 arms, built by the published class
(src/experiments/v2_weather_oracle.py: WeatherPanel and its `items`) on the issuance grid and the hourly
index that run_chronos builds (the same file, lines 301-311). One context per issuance kept by the
published rule (the raw target with NaN kept, the 512 h ending at the issuance, at least 48 finite
values, a finite value at the issuance, the index reaching 72 h past it) and scored at one lead at
least, which is the set of issuances of the published prediction file
runs/v2_weather_oracle/chronos/chronos_{variable}.csv.gz. Leads are horizon positions (lead h is
position h - 1 of a 72 h horizon).

Covariates (cov mode) are those of the published arm chronos2_net: the network's mean temperature and
its spread across probes (pandas' sample standard deviation over every probe with a temperature
record, the forecast probe included) and, for oxygen, the probe's own temperature, past only, each
filled as v2_probabilistic._filled fills them. No covariate is known after issuance.

Gates, all required:
  C1  the issuance keys equal the published file's chronos2_net issuances (count and keys, both ways)
  C2  for every issuance, the leads whose target is observed equal the published file's leads
  C3  every context ends at its issuance, is 512 h long, has a finite value there and >= 48 finite
  C4  two contexts per variable and cohort rebuilt by hand from the observation frames, with their
      covariates, equal the published function's output exactly

Writes reports/bay_twin/runs/v2_fm_zoo/inputs/C_{variable}_{cohort}.npz and the task's section of
manifest_dist_bridge.json in the same folder.

Usage: python -m scripts.v2_fm_zoo_inputs_dist
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

OUT = ROOT / 'reports/bay_twin/runs/v2_fm_zoo/inputs'
MANIFEST = OUT / 'manifest_dist_bridge.json'
PUBLISHED = ROOT / 'reports/bay_twin/runs/v2_weather_oracle/chronos'
ARM = 'chronos2_net'
COHORTS = ('Dev-2025', 'Reserved-2026')
UNITS = {'temperature': 'degC', 'oxygen': 'umol/L (the sondes stored column)'}

OPENED = set()


def _trace(event, args):
    if event == 'open' and args and isinstance(args[0], (str, Path)):
        OPENED.add(str(args[0]))


def data_sources():
    """Every data file the build opened (raw exports, station, published artifacts), relative to ROOT."""
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


def published_rows(variable):
    f = pd.read_csv(PUBLISHED / f'chronos_{variable}.csv.gz', usecols=['cohort', 'sensor', 'time', 'lead', 'arm'])
    f = f[f.arm == ARM].drop(columns='arm').reset_index(drop=True)
    f['time'] = pd.to_datetime(f.time, utc=True).astype('datetime64[ns, UTC]')
    return f


def build(variable):
    from src.experiments import v2_probabilistic as P
    from src.experiments import v2_weather_oracle as W
    # run_chronos, v2_weather_oracle.py lines 301-311, without the model
    meta, frames = P.bay_reserved_evaluation.load_all_observations()
    temps = {n: f['temperature'] for n, f in frames.items() if 'temperature' in f and f['temperature'].notna().any()}
    names = [n for n in meta.index if n in frames and variable in frames[n] and frames[n][variable].notna().any()]
    index = pd.date_range(pd.Timestamp('2024-07-24', tz='UTC'), P.RESERVED_WINDOW[1] + pd.Timedelta(hours=W.MAX_LEAD + 1),
                          freq='h')
    panel = W.WeatherPanel({n: frames[n][variable] for n in names}, temps, index, W.MAX_LEAD)
    grid = np.flatnonzero(np.isin(index.hour, (0, 6, 12, 18))
                          & (((index >= P.DEV_WINDOW[0]) & (index < P.DEV_WINDOW[1]))
                             | ((index >= P.RESERVED_WINDOW[0]) & (index <= P.RESERVED_WINDOW[1]))))
    cov_names = ['net_T', 'net_sd'] + (['own_T'] if variable == 'oxygen' else [])
    missing_own = [n for n in names if variable == 'oxygen' and n not in panel.own_t]
    if missing_own:
        raise SystemExit(f'{variable}: no own temperature for {missing_own}; the published covariate set varies by probe')
    rows, contexts, past, seen_rows = [], [], [], []
    kept_unscored = 0
    for name in names:
        items, kept = panel.items(name, grid, variable, (ARM,))
        if not kept:
            continue
        kept = np.asarray(kept)
        seen = np.stack([np.isfinite(panel.series[name][kept + h]) for h in W.LEADS], axis=1)
        for it, p, s in zip(items[ARM], kept, seen):
            if not s.any():                               # the published file holds no row for it
                kept_unscored += 1
                continue
            if list(it['past_covariates']) != cov_names:
                raise SystemExit(f'{variable} {name}: covariates {list(it["past_covariates"])} != {cov_names}')
            t = index[p]
            rows.append((np.where(t < P.RESERVED_WINDOW[0], 'Dev-2025', 'Reserved-2026').item(), name, t, int(p)))
            contexts.append(it['target'])
            past.append(np.stack([it['past_covariates'][k] for k in cov_names]))
            seen_rows.append([h for h, ok in zip(W.LEADS, s) if ok])
    keys = pd.DataFrame(rows, columns=['cohort', 'sensor', 'time', 'position'])
    keys['time'] = pd.to_datetime(keys.time, utc=True).astype('datetime64[ns, UTC]')
    return dict(keys=keys, contexts=contexts, past=past, seen=seen_rows, cov_names=cov_names, names=names,
                index=index, frames=frames, temps=temps, panel=panel, kept_unscored=kept_unscored,
                grid_issuances=int(len(grid)), ctx=P.CTX, min_finite=P.MIN_FINITE, leads=list(W.LEADS))


def gate_keys(b, published):
    mine = b['keys'][['cohort', 'sensor', 'time']]
    pub = published.drop_duplicates(['cohort', 'sensor', 'time'])[['cohort', 'sensor', 'time']]
    j = mine.merge(pub, how='outer', indicator=True)
    out = {'issuances_builder': int(len(mine)), 'issuances_published': int(len(pub)),
           'only_builder': int((j._merge == 'left_only').sum()), 'only_published': int((j._merge == 'right_only').sum()),
           'duplicates_builder': int(mine.duplicated().sum()),
           'by_cohort_builder': mine.groupby('cohort').size().to_dict(),
           'by_cohort_published': pub.groupby('cohort').size().to_dict()}
    out['passed'] = bool(out['only_builder'] == 0 and out['only_published'] == 0 and out['duplicates_builder'] == 0
                         and out['issuances_builder'] == out['issuances_published'])
    return out


def gate_leads(b, published):
    mine = b['keys'].assign(lead=b['seen']).explode('lead')[['cohort', 'sensor', 'time', 'lead']]
    mine['lead'] = mine.lead.astype('int64')
    j = mine.merge(published, how='outer', indicator=True)
    out = {'rows_builder': int(len(mine)), 'rows_published': int(len(published)),
           'only_builder': int((j._merge == 'left_only').sum()), 'only_published': int((j._merge == 'right_only').sum())}
    out['passed'] = bool(out['only_builder'] == 0 and out['only_published'] == 0 and out['rows_builder'] == out['rows_published'])
    return out


def gate_contexts(b):
    ctx, need = b['ctx'], b['min_finite']
    bad_len = bad_end = bad_finite = bad_last = 0
    for (cohort, name, t, p), c in zip(b['keys'].itertuples(index=False), b['contexts']):
        s = b['panel'].series[name]
        bad_len += len(c) != ctx
        bad_end += not np.array_equal(c, s[p - len(c) + 1:p + 1].astype(np.float32), equal_nan=True)
        bad_finite += np.isfinite(c).sum() < need
        bad_last += not np.isfinite(c[-1])
    out = {'contexts': len(b['contexts']), 'length_not_512': int(bad_len), 'not_ending_at_issuance': int(bad_end),
           'fewer_than_48_finite': int(bad_finite), 'issuance_value_missing': int(bad_last)}
    out['passed'] = bool(bad_len == bad_end == bad_finite == bad_last == 0)
    return out


def _fill_by_hand(x):
    """ffill, bfill, then the window's own mean (zero if it has none), written out again."""
    s = pd.Series(np.asarray(x, float)).ffill().bfill()
    m = float(np.nanmean(x)) if np.isfinite(x).any() else 0.0
    return s.fillna(m).to_numpy(np.float32)


def gate_by_hand(b, variable):
    """Rebuild two contexts per cohort straight from the observation frames, independently of the panel."""
    out, ok = [], True
    keys = b['keys']
    for cohort in COHORTS:
        sel = np.flatnonzero(keys.cohort.to_numpy() == cohort)
        for r in (sel[len(sel) // 3], sel[(2 * len(sel)) // 3]):
            name, t = keys.sensor.iloc[r], keys.time.iloc[r]
            hours = pd.date_range(t - pd.Timedelta(hours=b['ctx'] - 1), t, freq='h')
            target = b['frames'][name][variable].reindex(hours).to_numpy(np.float32)
            temps = pd.DataFrame({n: s.reindex(hours) for n, s in b['temps'].items()})
            cov = {'net_T': _fill_by_hand(temps.mean(axis=1).to_numpy(float)),
                   'net_sd': _fill_by_hand(temps.std(axis=1, ddof=1).to_numpy(float))}
            if variable == 'oxygen':
                cov['own_T'] = _fill_by_hand(b['temps'][name].reindex(hours).to_numpy(float))
            same_target = bool(np.array_equal(target, b['contexts'][r], equal_nan=True))
            diffs = {k: float(np.max(np.abs(cov[k] - b['past'][r][i]))) for i, k in enumerate(b['cov_names'])}
            row_ok = same_target and all(d == 0.0 for d in diffs.values())
            ok &= row_ok
            out.append({'cohort': cohort, 'sensor': name, 'issuance': str(t), 'target_identical': same_target,
                        'finite_in_context': int(np.isfinite(target).sum()),
                        'max_abs_covariate_difference': diffs, 'passed': row_ok})
    return {'rebuilt': out, 'passed': bool(ok)}


def main():
    sys.addaudithook(_trace)
    OUT.mkdir(parents=True, exist_ok=True)
    section = {'task': 'C_distribution', 'declared_in': 'config/bay_reserved_protocol_v2_addendum81.json',
               'builder': 'scripts/v2_fm_zoo_inputs_dist.py',
               'published_functions': ['src/experiments/v2_weather_oracle.py:WeatherPanel (225-263)',
                                       'src/experiments/v2_weather_oracle.py:run_chronos grid and index (301-311)',
                                       'src/experiments/v2_probabilistic.py:_filled (370-375), CTX, MIN_FINITE'],
               'files': {}, 'gates': {}}
    for variable in ('temperature', 'oxygen'):
        b = build(variable)
        published = published_rows(variable)
        gates = {'C1_issuance_keys': gate_keys(b, published), 'C2_observed_leads': gate_leads(b, published),
                 'C3_contexts': gate_contexts(b), 'C4_by_hand': gate_by_hand(b, variable)}
        gates['kept_by_the_published_rule_but_scored_at_no_lead'] = b['kept_unscored']
        section['gates'][variable] = gates
        print(variable, json.dumps({k: v.get('passed') for k, v in gates.items() if isinstance(v, dict)}), flush=True)
        if not all(v['passed'] for v in gates.values() if isinstance(v, dict)):
            raise SystemExit(f'task C gates failed for {variable}: {json.dumps(gates, default=str)[:2000]}')
        sources = [str((PUBLISHED / f'chronos_{variable}.csv.gz').relative_to(ROOT).as_posix()),
                   'src/experiments/v2_weather_oracle.py', 'src/experiments/v2_probabilistic.py',
                   'scripts/v2_fm_zoo_inputs_dist.py'] + data_sources()
        sources = sorted(set(sources))
        for cohort in COHORTS:
            sel = np.flatnonzero(b['keys'].cohort.to_numpy() == cohort)
            keys = b['keys'].iloc[sel][['cohort', 'sensor', 'time']].reset_index(drop=True)
            path = OUT / f'C_{variable}_{cohort}.npz'
            meta = save_inputs(
                path, task='C_distribution', variable=variable, cohort=cohort, keys=keys,
                contexts=[b['contexts'][i] for i in sel], horizon=72, units=UNITS[variable],
                past_cov=[b['past'][i] for i in sel], future_cov=None, cov_names=b['cov_names'],
                standardisation=None, sources=sources,
                note=('Addendum 65 chronos2_net rows: one context per issuance (time = issuance, UTC) on the '
                      '00/06/12/18 UTC grid, 512 h of the raw hourly series ending at the issuance, NaN kept; '
                      'lead h is horizon position h - 1 (leads 1, 3, 6, 12, 24, 36, 48, 72 are scored where the '
                      'target is observed). Past covariates exactly as the published chronos2_net arm: net_T and '
                      'net_sd over every probe with a temperature record (the forecast probe included), own_T '
                      'for oxygen, filled forward, backward and by the window mean; no future covariates. '
                      'Rows ordered by probe, then issuance.'))
            back = load_inputs(path)
            assert back['meta']['n'] == len(sel) and len(back['contexts']) == len(sel)
            section['files'][path.name] = {'n': int(len(sel)), 'horizon': 72, 'cov_names': b['cov_names'],
                                           'context_length': int(b['ctx']), 'sha256': digest(path),
                                           'sensors': int(keys.sensor.nunique())}
            print(' ', path.name, len(sel), 'contexts', flush=True)
    manifest = json.loads(MANIFEST.read_text(encoding='utf-8')) if MANIFEST.exists() else {}
    manifest['declared_in'] = 'config/bay_reserved_protocol_v2_addendum81.json'
    manifest['C_distribution'] = section
    with open(MANIFEST, 'w', encoding='utf-8', newline='\n') as f:
        f.write(json.dumps(manifest, indent=2, default=str) + '\n')
    print('manifest', MANIFEST.relative_to(ROOT).as_posix())


if __name__ == '__main__':
    main()
