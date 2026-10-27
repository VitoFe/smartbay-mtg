"""Paired contrasts for the v2 Field-57 decomposition (same bootstrap as for v1: casts, nested metric, 2000)."""
import json, sys
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / 'reports/bay_twin/runs' / sys.argv[1] / 'results'
V1 = ROOT / 'reports/bay_twin/runs/20260917T011757Z_field_export_pilot/results/field_export_held_site/replay_predictions.csv.gz'
RNG = np.random.default_rng(20260919)
TARGETS = ['temp_c', 'salinity_psu', 'do_umol_l']


def agg(frame, col):
    return frame.groupby(['fold', 'station', 'cast_id'])[col].mean().groupby(['fold', 'station']).mean().groupby('fold').mean().mean()


def paired(a, b, target, nboot=2000):
    keys = ['fold', 'station', 'cast_id', 'depth_m']
    j = a[keys + [target, 'pred_' + target]].merge(b[keys + ['pred_' + target]], on=keys, suffixes=('_a', '_b'), validate='one_to_one')
    assert len(j) == len(a) == len(b)
    j['ea'] = (j[target] - j['pred_' + target + '_a']).abs(); j['eb'] = (j[target] - j['pred_' + target + '_b']).abs()
    obs = agg(j, 'ea') - agg(j, 'eb'); casts = j.cast_id.unique(); ds = []
    for _ in range(nboot):
        s = j[j.cast_id.isin(RNG.choice(casts, len(casts), replace=True))]
        ds.append(agg(s, 'ea') - agg(s, 'eb'))
    lo, hi = np.percentile(ds, [2.5, 97.5]); return float(obs), float(lo), float(hi)


preds = {p.parent.name.removeprefix('v2_field_held_site_'): pd.read_csv(p) for p in RUN.glob('v2_field_held_site_*/selected_predictions.csv.gz')}
preds['v1_release'] = pd.read_csv(V1)
arms = {k: {**json.loads((RUN / f'v2_field_held_site_{k}/report.json').read_text())['nested_mae'],
            'alpha': json.loads((RUN / f'v2_field_held_site_{k}/report.json').read_text())['selection']['params']['alpha'],
            'trials': json.loads((RUN / f'v2_field_held_site_{k}/report.json').read_text())['selection']['completed_trials']}
        for k in preds if k != 'v1_release'}
contrasts = [('Copernicus (both) vs none', 'A1_both_direct', 'A2_none_direct', ['temp_c', 'do_umol_l']),
             ('background only vs none', 'A4_base_only_direct', 'A2_none_direct', ['temp_c', 'do_umol_l']),
             ('design only vs none', 'A3_design_only_direct', 'A2_none_direct', ['temp_c', 'do_umol_l']),
             ('background only vs both', 'A4_base_only_direct', 'A1_both_direct', ['temp_c', 'do_umol_l']),
             ('AOU vs direct (released family)', 'A1_both_aou', 'A1_both_direct', ['do_umol_l']),
             ('v2 release vs v1 frozen release', 'A1_both_direct', 'v1_release', ['temp_c', 'salinity_psu', 'do_umol_l'])]
out = []
for label, a, b, targets in contrasts:
    for t in targets:
        d, lo, hi = paired(preds[a], preds[b], t)
        out.append({'contrast': label, 'a': a, 'b': b, 'target': t, 'diff': d, 'ci_low': lo, 'ci_high': hi, 'excludes_zero': bool(lo > 0 or hi < 0)})
        print(f'{label:34s} {t:13s} {d:+.4f} [{lo:+.4f}, {hi:+.4f}]' + ('  <- excludes zero' if lo > 0 or hi < 0 else ''))
(RUN.parent / 'v2_field_summary.json').write_text(json.dumps({'arms': arms, 'contrasts': out, 'bootstrap': 'casts, nested metric, 2000, seed 20260919'}, indent=2))
