"""Addendum 54: build, locally, what the compute host needs (config/bay_reserved_protocol_v2_addendum54.json).

Writes reports/bay_twin/runs/v2_o2_temperature/inputs/:

  episodes.csv.gz  every published oxygen episode of Gap-Dev, Gap-2026, Gap-Long-Dev and Gap-Long-2026
                   (reports/bay_twin/runs/v2_gap_variants/<cohort>__oxygen__<band>.csv.gz): cohort, band,
                   period, episode key, sensor, gap length and calendar fold
  weights.json     the temperature stacked ensemble's frozen weights per band: each development fold's
                   out-of-fold weights and the weights frozen before 2026 (addendum 33's studies)
  issuances.json   the oxygen forecasts' issuances per sonde, as positions of the companion precompute's
                   hourly index: the horizon study's oxygen endpoints (development) and every 2026
                   issuance at 00/06/12/18 UTC with an oxygen reading

Usage: python -m scripts.v2_o2_temperature_inputs
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / 'reports/bay_twin/runs'
OUT = RUNS / 'v2_o2_temperature' / 'inputs'
CACHE = ROOT / 'companion/.cache/full'
COHORTS = {'Gap-Dev': ('development', ('1-6', '12-72')), 'Gap-2026': ('reserved', ('1-6', '12-72')),
           'Gap-Long-Dev': ('development', ('168-1440',)), 'Gap-Long-2026': ('reserved', ('168-1440',))}
STACK_MEMBERS = ('linear', 'chronos_bidir', 'gpr', 'chronos2_cov_bidir')


def episodes():
    parts = []
    for cohort, (period, bands) in COHORTS.items():
        for band in bands:
            f = pd.read_csv(RUNS / 'v2_gap_variants' / f'{cohort}__oxygen__{band}.csv.gz',
                            usecols=['episode', 'sensor', 'gap_length', 'fold'])
            e = f.drop_duplicates('episode').assign(cohort=cohort, band=band, period=period)
            parts.append(e[['cohort', 'band', 'period', 'episode', 'sensor', 'gap_length', 'fold']])
    return pd.concat(parts, ignore_index=True)


def weights():
    out = {}
    for band in ('1-6', '12-72', '168-1440'):
        out[band] = {}
        for tag in ('f0', 'f1', 'f2', 'f3', 'all'):
            p = json.loads((RUNS / 'v2_gap_variants_studies' / f'stack_temperature_{band}_{tag}_selection.json')
                           .read_text(encoding='utf-8'))['params']
            raw = np.array([p[f'w_{m}'] for m in STACK_MEMBERS], float)
            out[band][tag] = (raw / raw.sum()).tolist()
    return {'members': list(STACK_MEMBERS), 'by_band': out}


def issuances():
    sys.path.insert(0, str(ROOT / 'companion'))
    from smartbay_companion.forecasting import published_horizon
    P = np.load(CACHE / 'panels.npz')
    index = pd.DatetimeIndex(P['times']).tz_localize('UTC')
    pos = pd.Series(np.arange(len(index)), index=index)
    names = [str(n) for n in P['names_oxygen']]
    values = P['values_oxygen']
    out = {n: set() for n in names}
    for (variable, _, _), g in published_horizon().items():
        if variable == 'oxygen':
            for s, t in zip(g.sensor, g.time):
                if s in out:
                    out[s].add(int(pos[t]))
    grid = np.flatnonzero((index >= pd.Timestamp('2026-01-01', tz='UTC')) & np.isin(index.hour, (0, 6, 12, 18)))
    for j, n in enumerate(names):
        out[n] |= {int(p) for p in grid if np.isfinite(values[p, j])}
    return {n: sorted(v) for n, v in out.items()}


def manifest():
    """The digests of the inputs the compute host received."""
    import hashlib
    files = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(OUT.glob('*')) if p.name != 'manifest.json'}
    (OUT / 'manifest.json').write_text(json.dumps({'addendum': 54, 'files': files}, indent=1) + '\n', encoding='utf-8')


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    e = episodes()
    e.to_csv(OUT / 'episodes.csv.gz', index=False)
    (OUT / 'weights.json').write_text(json.dumps(weights(), indent=1) + '\n', encoding='utf-8')
    iss = issuances()
    (OUT / 'issuances.json').write_text(json.dumps(iss) + '\n', encoding='utf-8')
    manifest()
    print(e.groupby(['cohort', 'band']).episode.nunique().to_string())
    print('issuances', {k: len(v) for k, v in iss.items()})


#: Addendum 59: the temperature fill reads the Gaussian process (directly at 336 and 1,440 h, through
#: the stacked weights elsewhere), so the fill was rerun with the corrected process and the stack
#: weights refitted on it. The published inputs stay as they are; the rerun's are written beside them.
RERUN = RUNS / 'v2_o2_temperature' / 'inputs_gp_by_pattern'


def rerun_inputs():
    """The rerun's inputs: the published episodes and issuances byte for byte, and the stack weights
    of the refitted studies (v2_gap_variants.STUDIES)."""
    import hashlib
    import shutil
    from scripts.v2_gap_variants import STUDIES
    RERUN.mkdir(parents=True, exist_ok=True)
    for name in ('episodes.csv.gz', 'issuances.json'):
        shutil.copyfile(OUT / name, RERUN / name)
    published = json.loads((OUT / 'weights.json').read_text(encoding='utf-8'))
    out = {'members': list(STACK_MEMBERS), 'by_band': {}}
    for band in ('1-6', '12-72', '168-1440'):
        out['by_band'][band] = {}
        for tag in ('f0', 'f1', 'f2', 'f3', 'all'):
            p = json.loads((STUDIES / f'stack_temperature_{band}_{tag}_selection.json')
                           .read_text(encoding='utf-8'))['params']
            raw = np.array([p[f'w_{m}'] for m in STACK_MEMBERS], float)
            out['by_band'][band][tag] = (raw / raw.sum()).tolist()
    assert published['members'] == out['members']
    (RERUN / 'weights.json').write_bytes((json.dumps(out, indent=1) + '\n').encode('utf-8'))
    files = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(RERUN.glob('*'))
             if p.name != 'manifest.json'}
    (RERUN / 'manifest.json').write_bytes((json.dumps(
        {'addendum': 54, 'rerun_declared_in': 'config/bay_reserved_protocol_v2_addendum59.json',
         'stack_studies': STUDIES.relative_to(RUNS).as_posix(), 'files': files}, indent=1)
        + '\n').encode('utf-8'))
    print('wrote', RERUN.relative_to(ROOT), files)


if __name__ == '__main__':
    # `--gp-by-pattern` writes only the rerun's inputs and never touches the published ones
    if sys.argv[1:2] == ['--gp-by-pattern']:
        rerun_inputs()
    else:
        main()
