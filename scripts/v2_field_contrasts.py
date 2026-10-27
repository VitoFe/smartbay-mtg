"""Every paired Field-57 contrast the thesis reports, with a bootstrap that counts what it draws.

Two earlier producers bootstrapped casts without counting repeats. `v2_summarize_field` kept the rows
of each cast once (`isin` of the drawn ids), so a draw was a subsample of about 63 % of the casts;
`v2_field_naive` concatenated repeated casts but its nested metric grouped them back into one by
`cast_id`. Both intervals were therefore too narrow. The placebo and CTD-prior comparisons had no
producer at all. This replaces all of them without touching the contract-bound originals.

A repeated unit is relabelled, so a cast drawn twice counts twice inside its (fold, station) cell.
Two units are reported:

  date   the calendar day of the cast. Casts on one day share the boundary product's profile, the
         weather and the moored context, so the day is the independent unit (primary)
  cast   the cast, as the declared protocol names it

Addendum 69: by default the cast-prior arms A6 and A7 are the inner-split-prior runs
(results/v2_field_held_site_A{6,7}_ctdprior_direct_innerprior/: each inner station split scored against
a prior built without its validation stations) and the naive comparators are those of the per-fold
model and no-boundary ridge (runs/v2_field_naive/perfold/); the report is runs/v2_field_contrasts_a69.json.
Addendum 71: the fitted depth control P1 is read the same way, from its inner-split run
(results/v2_field_held_site_P1_placebo_direct_innerprior/).
`--shared` reads the earlier files and writes runs/v2_field_contrasts.json.

Usage: python -m scripts.v2_field_contrasts [--shared]
"""
import json
import sys
import zlib
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / 'reports/bay_twin/runs'
SHARED = '--shared' in sys.argv[1:]
OUT = RUNS / ('v2_field_contrasts.json' if SHARED else 'v2_field_contrasts_a69.json')
DRAWS = 2000
TARGETS = ('temp_c', 'salinity_psu', 'do_umol_l')
V2_FIELD = RUNS / '20260919T145045Z_v2_field/results'
LOCAL = ROOT / 'results'
#: addendum 69: the inner-split prior (A6, A7) and the per-fold naive comparators
PRIOR_SUFFIX = '' if SHARED else '_innerprior'
NAIVE = RUNS / ('v2_field_naive' if SHARED else 'v2_field_naive/perfold')

SOURCES = {
    **{a: V2_FIELD / f'v2_field_held_site_{a}/selected_predictions.csv.gz'
       for a in ('A1_both_direct', 'A2_none_direct', 'A3_design_only_direct',
                 'A4_base_only_direct', 'A1_both_aou')},
    **{a: LOCAL / f'v2_field_held_site_{a}/selected_predictions.csv.gz'
       for a in ('A5_intercept_direct', 'P2_placebo_direct')},
    # addendum 71: the fitted depth control, like the cast prior, fitted inside each inner split
    **{a: LOCAL / f'v2_field_held_site_{a}{PRIOR_SUFFIX}/selected_predictions.csv.gz'
       for a in ('A6_ctdprior_direct', 'A7_ctdprior_direct', 'P1_placebo_direct')},
    **{f'naive_{n}': NAIVE / f'pred_{n}.csv.gz'
       for n in ('model', 'background', 'climatology', 'seasonal', 'oracle', 'no_copernicus')},
}

#: (label, arm, reference, targets). Each is a contrast a Field-57 table or sentence reports.
CONTRASTS = [
    ('boundary product, both uses, vs none', 'A1_both_direct', 'A2_none_direct', TARGETS),
    ('background offset only vs none', 'A4_base_only_direct', 'A2_none_direct', TARGETS),
    ('regressors only vs none', 'A3_design_only_direct', 'A2_none_direct', TARGETS),
    ('background offset only vs both', 'A4_base_only_direct', 'A1_both_direct', TARGETS),
    ('apparent utilisation vs direct, released family', 'A1_both_aou', 'A1_both_direct',
     ('do_umol_l',)),
    ('fitted intercept vs released', 'A5_intercept_direct', 'A1_both_direct', TARGETS),
    ('CTD prior, month x depth, vs Copernicus', 'A6_ctdprior_direct', 'A1_both_direct', TARGETS),
    ('CTD prior, harmonic, vs Copernicus', 'A7_ctdprior_direct', 'A1_both_direct', TARGETS),
    ('CTD prior, month x depth, vs none', 'A6_ctdprior_direct', 'A2_none_direct', TARGETS),
    ('CTD prior, harmonic, vs none', 'A7_ctdprior_direct', 'A2_none_direct', TARGETS),
    ('fitted depth basis vs Copernicus', 'P1_placebo_direct', 'A1_both_direct', ('temp_c',)),
    ('fitted depth basis vs none', 'P1_placebo_direct', 'A2_none_direct', ('temp_c',)),
    ('random depth basis vs none', 'P2_placebo_direct', 'A2_none_direct', ('temp_c',)),
    ('random depth basis vs Copernicus', 'P2_placebo_direct', 'A1_both_direct', ('temp_c',)),
    ('model vs month x depth climatology', 'naive_model', 'naive_climatology', TARGETS),
    ('model vs harmonic climatology', 'naive_model', 'naive_seasonal', TARGETS),
    ('model vs oracle climatology', 'naive_model', 'naive_oracle', TARGETS),
    ('model vs background alone', 'naive_model', 'naive_background', TARGETS),
    ('no boundary product vs oracle climatology', 'naive_no_copernicus', 'naive_oracle', TARGETS),
]


def load():
    frames = {}
    keys = None
    for arm, path in SOURCES.items():
        f = pd.read_csv(path)
        f = f.sort_values(['fold', 'station', 'cast_id', 'depth_m']).reset_index(drop=True)
        k = f[['fold', 'station', 'cast_id', 'depth_m']]
        if keys is None:
            keys = k
            truth = f[list(TARGETS)].to_numpy(float)
            day = pd.to_datetime(f.t_utc, utc=True).dt.strftime('%Y-%m-%d').to_numpy()
        elif not k.equals(keys) or not np.allclose(f[list(TARGETS)].to_numpy(float), truth,
                                                    equal_nan=True):
            raise SystemExit(f'{arm} does not score the rows of the others')
        frames[arm] = f[[f'pred_{t}' for t in TARGETS]].to_numpy(float)
    return keys.assign(day=day), truth, frames


def nested(keys, error):
    """Mean over depth bins within cast, casts within (fold, station), stations, then folds."""
    z = keys[['fold', 'station', 'cast_id']].assign(e=error)
    return float(z.groupby(['fold', 'station', 'cast_id']).e.mean()
                 .groupby(['fold', 'station']).mean().groupby('fold').mean().mean())


def contrast(keys, truth, frames, arm, ref, t, unit, label):
    j = TARGETS.index(t)
    ea = np.abs(truth[:, j] - frames[arm][:, j])
    eb = np.abs(truth[:, j] - frames[ref][:, j])
    ok = np.isfinite(ea) & np.isfinite(eb)
    k = keys[ok].reset_index(drop=True)
    ea, eb = ea[ok], eb[ok]
    point = nested(k, ea) - nested(k, eb)
    groups = k.groupby(unit).indices
    names = list(groups)
    rng = np.random.default_rng([20260923, zlib.crc32(f'{label}|{t}|{unit}'.encode())])
    draws = np.empty(DRAWS)
    for i in range(DRAWS):
        picked = rng.integers(0, len(names), len(names))
        rows = np.concatenate([groups[names[p]] for p in picked])
        # Relabel each drawn unit so a repeat is a second cast, not the same cast again.
        tag = np.concatenate([np.full(len(groups[names[p]]), n) for n, p in enumerate(picked)])
        kk = k.iloc[rows].assign(cast_id=k.cast_id.to_numpy()[rows].astype(str) + '#'
                                 + tag.astype(str))
        draws[i] = nested(kk, ea[rows]) - nested(kk, eb[rows])
    lo, hi = np.percentile(draws, [2.5, 97.5])
    # Six decimals: the tables print three, and rounding a four-decimal value to three can land on
    # the wrong side of a 5 (-1.01348 stored as -1.0135 printed as -1.014).
    return {'point': round(point, 6), 'ci': [round(float(lo), 6), round(float(hi), 6)],
            'excludes_zero': bool(lo > 0 or hi < 0), 'units': len(names)}


def main():
    keys, truth, frames = load()
    report = {'rows': int(len(keys)), 'casts': int(keys.cast_id.nunique()),
              'days': int(keys.day.nunique()), 'draws': DRAWS,
              'metric': 'nested macro MAE: depth bins within cast, casts within (fold, station), '
                        'stations within fold, folds',
              'units': {'date': 'calendar day of the cast (primary)', 'cast': 'the cast'},
              'contrasts': []}
    for label, arm, ref, targets in CONTRASTS:
        for t in targets:
            entry = {'contrast': label, 'arm': arm, 'reference': ref, 'target': t}
            for unit, column in (('date', 'day'), ('cast', 'cast_id')):
                entry[unit] = contrast(keys, truth, frames, arm, ref, t, column, label)
            report['contrasts'].append(entry)
            d, c = entry['date'], entry['cast']
            print(f"{label:48s} {t:12s} {d['point']:+.4f}  date {d['ci']}{'*' if d['excludes_zero'] else ' '}"
                  f"  cast {c['ci']}{'*' if c['excludes_zero'] else ' '}")
    OUT.write_bytes(json.dumps(report, indent=2).encode('utf-8'))  # LF on every host
    print('wrote', OUT, f"({report['casts']} casts on {report['days']} days)")


if __name__ == '__main__':
    main()
