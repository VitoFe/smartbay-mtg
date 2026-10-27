"""Numbers Chapter 5 states in running text that no other producer writes.

Each block recomputes a figure from the inputs or the saved predictions it came from, so the
sentence can cite this file instead of the audit trail:

  field_prior            the vertical prior per calendar fold and with each station left out
                         (Section 5.4.2; saved Field-57 predictions, nested rule)
  river_correlation      the Magra and Arno stages on shared hours before 2026 (Section 5.4.1)
  envelope               how often a development-fold validation target lies outside its sensor's
                         training minimum and maximum, the reason the range gate fails (Section 5.4.3)
  observability          the moored and near-bottom oxygen record against the thresholds, the cast
                         start hours and the July-October surface-to-bottom difference (Section 5.5.1)
  stratified_naive       the naive near-bottom error on casts whose surface and bottom differ by at
                         least 20 umol/L (Section 5.5.2)

Filters are stated in the output. The cast archive is read with qc_flag == 0, the rule every
analysis of it uses; without it the near-bottom minimum would be lower, and both are recorded.
Near-bottom means the three deepest bins of a cast, as in scripts/experiments/exp_bottom.py.

Usage: python -m scripts.v2_section5_numbers
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'reports/bay_twin/runs/v2_section5_numbers.json'
CASTS = ROOT / 'data/processed/ctd_all_casts.csv.gz'
HOURLY = ROOT / 'reports/bay_twin/hourly_observations.csv.gz'
BOTTOM = ROOT / 'reports/bay_twin/runs/v2_bottom_aou/bottom/predictions.csv.gz'
THRESHOLDS = {'hypoxia_62.5': 62.5, 'mean_sublethal_82': 82.0, 'sublethal_p90_156': 156.0, 'six_mg_l_186': 186.0}


def field_prior():
    """Per calendar fold and leave-one-station-out, from the saved Field-57 predictions."""
    from scripts import v2_field_contrasts as fc
    keys, truth, frames = fc.load()
    j = fc.TARGETS.index('temp_c')
    err = {a: np.abs(truth[:, j] - frames[a][:, j]) for a in ('A1_both_direct', 'A2_none_direct', 'A7_ctdprior_direct')}
    ok = np.isfinite(err['A1_both_direct']) & np.isfinite(err['A2_none_direct']) & np.isfinite(err['A7_ctdprior_direct'])
    k = keys[ok].reset_index(drop=True)
    e = {a: v[ok] for a, v in err.items()}

    def diff(mask, a, b):
        kk = k[mask].reset_index(drop=True)
        return round(fc.nested(kk, e[a][mask]) - fc.nested(kk, e[b][mask]), 4)
    everything = np.ones(len(k), bool)
    per_fold = {}
    for f in sorted(k.fold.unique()):
        m = (k.fold == f).to_numpy()
        per_fold[str(f)] = {'scored': [str(k.day[m].min()), str(k.day[m].max())],
                            'copernicus_minus_none': diff(m, 'A1_both_direct', 'A2_none_direct'),
                            'cast_prior_minus_none': diff(m, 'A7_ctdprior_direct', 'A2_none_direct')}
    loo = {s: diff((k.station != s).to_numpy(), 'A1_both_direct', 'A2_none_direct') for s in sorted(k.station.unique())}
    z = k[['fold', 'station', 'cast_id']].assign(d=e['A1_both_direct'] - e['A2_none_direct'])
    cells = z.groupby(['fold', 'station', 'cast_id']).d.mean().groupby(['fold', 'station']).mean()
    return {'what': 'temperature, nested MAE difference (degC), negative favours the first named; '
                    'folds are the two Field-57 calendar folds, checkpoints the (fold, station) cells',
            'all': {'copernicus_minus_none': diff(everything, 'A1_both_direct', 'A2_none_direct'),
                    'cast_prior_minus_none': diff(everything, 'A7_ctdprior_direct', 'A2_none_direct')},
            'per_fold': per_fold,
            'copernicus_minus_none_leave_one_station_out': loo,
            'leave_one_station_out_smallest_gain': round(-max(loo.values()), 4),
            'checkpoints': int(len(cells)), 'checkpoints_favouring_copernicus': int((cells < 0).sum())}


def river_correlation():
    from src.data.bay_bocca import ameglia, bocca
    a = ameglia().resample('1h').mean()
    b = bocca()
    j = pd.concat([a.rename("magra"), b.rename("arno")], axis=1, sort=True).dropna()
    return {'what': 'Pearson correlation of the Magra stage at Ameglia (hourly mean) and the Arno stage at '
                    "Bocca d'Arno (daily, held forward causally to hourly, src.data.bay_bocca) on the hours "
                    'both report before 2026',
            'hours': int(len(j)), 'first': str(j.index.min()), 'last': str(j.index.max()),
            'r': round(float(j.magra.corr(j.arno)), 3)}


def envelope():
    """The empirical range gate's own bounds (v2_physics_gate.bounds) against the scored targets."""
    from src.experiments.v2_closed_loop import STEP
    from src.experiments.v2_horizon import build, folds
    from src.experiments.v2_physics_gate import bounds
    out = {'what': 'share of valid validation targets, at the eligible sensors of each development fold '
                   'on the six-hour issuance grid, outside that sensor\'s training minimum and maximum'}
    for variable in ('temperature', 'oxygen'):
        data = build(variable, STEP)
        outside = total = 0
        for f in folds(data, STEP):
            lo, hi, _ = bounds(data, f)
            truth = data['truth'][f['va']]
            valid = data['valid'][f['va']] & np.asarray(f['eligible'], bool)[None, :]
            outside += int((((truth < lo) | (truth > hi)) & valid).sum())
            total += int(valid.sum())
        out[variable] = {'outside': outside, 'targets': total, 'percent': round(100 * outside / total, 2)}
    return out


def bottom_bins(casts):
    ordered = casts.sort_values(['cast_id', 'depth_m'])
    return ordered.groupby('cast_id').tail(3)


def below(values):
    return {k: round(100 * float((values < v).mean()), 3) for k, v in THRESHOLDS.items()}


def observability():
    obs = pd.read_csv(HOURLY, index_col=0)
    obs.index = pd.to_datetime(obs.index, utc=True)
    oxygen = obs[[c for c in obs.columns if c.endswith('::oxygen')]].stack().dropna()
    raw = pd.read_csv(CASTS).dropna(subset=['temp_c', 'do_umol_l', 'depth_m'])
    casts = raw[raw.qc_flag == 0]
    near = bottom_bins(casts)
    start = pd.to_datetime(casts.drop_duplicates('cast_id').t_utc, utc=True).dt.hour
    rows = []
    for cid, g in casts.groupby('cast_id'):
        g = g.sort_values('depth_m')
        surface, bottom = g[g.depth_m <= 2.0], g.tail(3)
        if len(surface) and len(bottom):
            rows.append((pd.Timestamp(g.t_utc.iloc[0]).month, surface.do_umol_l.mean() - bottom.do_umol_l.mean()))
    d = pd.DataFrame(rows, columns=['month', 'surface_minus_bottom'])
    summer = d[d.month.isin([7, 8, 9, 10])].surface_minus_bottom
    return {'thresholds_umol_l': THRESHOLDS,
            'moored': {'source': 'reports/bay_twin/hourly_observations.csv.gz, every ::oxygen column',
                       'first': str(obs.index.min()), 'last': str(obs.index.max()),
                       'n': int(len(oxygen)), 'min': round(float(oxygen.min()), 2),
                       'percent_below': below(oxygen)},
            'ctd_near_bottom': {'filter': 'qc_flag == 0; the three deepest bins of each cast',
                                'casts': int(casts.cast_id.nunique()), 'bins': int(len(near)),
                                'min': round(float(near.do_umol_l.min()), 2),
                                'min_mg_l': round(float(near.do_umol_l.min()) / 31.2512, 2),
                                'percent_below': below(near.do_umol_l),
                                'min_without_qc_filter': round(float(bottom_bins(raw).do_umol_l.min()), 2)},
            'cast_start_hour_utc': {str(h): int(n) for h, n in start.value_counts().sort_index().items()},
            'cast_start_07_12_percent': round(100 * float(start.between(7, 11).mean()), 2),
            'surface_minus_bottom_july_october': {
                'what': 'mean of bins at or above 2 m minus mean of the three deepest bins, umol/L',
                'casts': int(len(summer)), 'median': round(float(summer.median()), 2),
                'quartiles': [round(float(summer.quantile(q)), 2) for q in (.25, .75)],
                'max': round(float(summer.max()), 2)}}


def stratified_naive():
    b = pd.read_csv(BOTTOM, dtype={'fold': str})
    s = b[(b.protocol == 'P1_held_station') & (b.model == 'B_surface') & (b.abs_delta >= 20)]
    e = (s.pred - s.truth).abs()
    return {'what': 'naive assumption (bottom = surface) on held-station casts with |surface - bottom| >= 20 umol/L',
            'casts': int(len(s)), 'mean_abs_error': round(float(e.mean()), 2),
            'nested_over_stations': round(float(s.assign(e=e).groupby('fold').e.mean().mean()), 2)}


def main():
    report = {'what': __doc__.split('\n\n')[0], 'field_prior': field_prior(),
              'river_correlation': river_correlation(), 'envelope': envelope(),
              'observability': observability(), 'stratified_naive': stratified_naive()}
    OUT.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=1))


if __name__ == '__main__':
    main()
