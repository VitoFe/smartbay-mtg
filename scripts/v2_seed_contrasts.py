"""The physics contrasts of the horizon sweep under three sampler seeds, and the table that shows them.

Every study in this project draws its trials from the protocol's TPE sampler, seeded 42. The
contrasts that carry the physical-guidance conclusions were re-tuned under seeds 7 and 123 with
everything else identical: the trees through `v2_phase_shift --seed` on the
published grid, the graph operators through `v2_seed_replicate`. Each arm is compared with its
reference tuned under the same seed (the tree, or for the constraint the learned-source graph), on
identical rows, with the (fold, ISO week) block bootstrap of the published sweep.

Each graph arm is the mean of three networks that differ only in their initialisation seed, compared
with a single deterministic tree, so an ensemble contrast includes the effect of the averaging.
`single_network` records the same contrast for each network alone (for the constraint, networks
paired by initialisation seed), point estimates only, and the table prints their range at the
protocol's sampler seed.

Usage: python -m scripts.v2_seed_contrasts
"""
import json
import zlib
from pathlib import Path

import numpy as np
import pandas as pd

from src.units import OXYGEN_UNIT_TEX, column_decimals, factor

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / 'reports/bay_twin/runs'
OUT = RUNS / 'v2_seed_contrasts.json'
TABLE = ROOT / 'thesis_template/content/tables/table_bay_seed_spread.tex'
DRAWS = 600
KEY = ['fold', 'sensor', 'time']
PUBLISHED = RUNS / '20260920T083322Z_v2_horizon/results'
NEURAL = {7: RUNS / '20260923T110726Z_v2_seedrep/results',
          123: RUNS / '20260923T110752Z_v2_seedrep/results'}
PHASE = RUNS / '20260923T103430Z_v2_phasefix/reports/bay_twin/runs'
#: (variable, horizon, arm minus reference, kind as Chapter 4 names it)
CONTRASTS = [
    ('temperature', 6, ('weather_tree', 'tree'), 'features'),
    ('oxygen', 6, ('weather_tree', 'tree'), 'features'),
    ('temperature', 6, ('learned_source', 'tree'), 'operator'),
    ('oxygen', 6, ('learned_source', 'tree'), 'operator'),
    ('temperature', 6, ('conservative_exchange', 'learned_source'), 'constraint'),
    ('oxygen', 6, ('conservative_exchange', 'learned_source'), 'constraint'),
    ('temperature', 6, ('conservative_exchange', 'tree'), 'constraint and operator'),
    ('oxygen', 6, ('conservative_exchange', 'tree'), 'constraint and operator'),
    ('oxygen', 24, ('weather_tree', 'tree'), 'features'),
    ('oxygen', 24, ('learned_source', 'tree'), 'operator'),
]
SEEDS = (42, 7, 123)


def horizon_arm(folder, variable, arm, h):
    path = folder / f'v2_horizon_{variable}_{arm}_h{h}/selected_predictions.csv.gz'
    if not path.exists():
        return None
    f = pd.read_csv(path)
    if 'mode' in f:
        f = f[f['mode'] == 'forecast']
    if 'seed' in f:
        # Neural arms: the mean over the three initialisation seeds, as every published table uses.
        f = f.groupby(KEY, as_index=False).agg(truth=('truth', 'first'),
                                               prediction=('prediction', 'mean'))
    return f[KEY + ['truth', 'prediction']].rename(columns={'prediction': arm})


def networks(folder, variable, arm, h):
    """The single networks behind a neural arm, keyed by initialisation seed; None for a tree."""
    path = folder / f'v2_horizon_{variable}_{arm}_h{h}/selected_predictions.csv.gz'
    if not path.exists():
        return None
    f = pd.read_csv(path)
    if 'mode' in f:
        f = f[f['mode'] == 'forecast']
    if 'seed' not in f:
        return None
    return {int(s): g[KEY + ['prediction']].rename(columns={'prediction': arm})
            for s, g in f.groupby('seed')}


def single_network(variable, h, a, b, seed):
    """Point contrast a - b with each neural arm taken one network at a time."""
    folder = PUBLISHED if seed == 42 else NEURAL[seed]
    nets = {arm: networks(folder, variable, arm, h) for arm in (a, b)
            if arm in ('learned_source', 'conservative_exchange')}
    nets = {k: v for k, v in nets.items() if v is not None}
    if not nets:
        return None
    frame = tree_arms(variable, h, seed)
    inits = sorted(set.intersection(*(set(v) for v in nets.values())))
    out = {}
    for init in inits:
        f = frame
        for arm, by_init in nets.items():
            f = f.merge(by_init[init], on=KEY, how='left', validate='one_to_one')
        f = f.dropna(subset=[a, b])
        out[str(init)] = round(macro(f, a) - macro(f, b), 4)
    return out


def tree_arms(variable, h, seed):
    """Plain and weather trees under `seed`: the published run for 42, the phase sweep otherwise."""
    if seed == 42:
        parts = [horizon_arm(PUBLISHED, variable, a, h) for a in ('tree', 'weather_tree')]
        return parts[0].merge(parts[1].drop(columns='truth'), on=KEY, validate='one_to_one')
    folder = PHASE / ('v2_phase_shift' if h == 6 else f'v2_phase_shift_h{h}')
    f = pd.read_csv(folder / f'phase_shift_{variable}_offset0_ratio_seed{seed}.csv.gz')
    return f[KEY + ['truth', 'tree', 'weather_tree']]


def frame_for(variable, h, seed):
    frame = tree_arms(variable, h, seed)
    folder = PUBLISHED if seed == 42 else NEURAL[seed]
    for arm in ('learned_source', 'conservative_exchange'):
        part = horizon_arm(folder, variable, arm, h)
        if part is not None:
            frame = frame.merge(part.drop(columns='truth'), on=KEY, how='left',
                                validate='one_to_one')
    return frame


def macro(frame, col):
    f = frame.dropna(subset=[col])
    return float(f.assign(e=(f.truth - f[col]).abs()).groupby(['fold', 'sensor']).e.mean()
                 .groupby('fold').mean().mean())


def paired(frame, a, b, label):
    f = frame.dropna(subset=[a, b]).assign(
        week=lambda d: pd.to_datetime(d.time, utc=True).dt.isocalendar().week.astype(int))
    rng = np.random.default_rng([20260923, zlib.crc32(label.encode())])
    blocks = f[['fold', 'week']].drop_duplicates().values.tolist()
    groups = {tuple(k): g for k, g in f.groupby(['fold', 'week'])}
    draws = []
    for _ in range(DRAWS):
        s = pd.concat([groups[tuple(blocks[i])] for i in rng.integers(0, len(blocks), len(blocks))])
        draws.append(macro(s, a) - macro(s, b))
    lo, hi = np.percentile(draws, [2.5, 97.5])
    point = macro(f, a) - macro(f, b)
    return {'point': round(point, 4), 'ci': [round(float(lo), 4), round(float(hi), 4)],
            'excludes_zero': bool(lo > 0 or hi < 0), 'endpoints': int(len(f))}


def cell(e, d, k=1.0):
    text = f"{k * e['point']:+.{d}f}".replace('+-', '-')
    return f"${text}^{{\\ast}}$" if e['excludes_zero'] else f'${text}$'


def oxygen_digits(contrasts):
    """Decimals of each column over the oxygen rows (printed at three in the artifact's unit): the
    most any of their values needs in mg/l."""
    rows = [e for key, e in contrasts.items() if key.startswith('oxygen|')]
    dig = {str(s): column_decimals([e['by_seed'][str(s)]['point'] for e in rows if str(s) in e['by_seed']], 3)
           for s in SEEDS}
    dig['range'] = column_decimals([e['range'] for e in rows], 3)
    spreads = [e['single_network']['42'] for e in rows if e.get('single_network', {}).get('42')]
    dig['single'] = column_decimals([f(s.values()) for s in spreads for f in (min, max)], 3)
    return dig


def main():
    report = {'draws': DRAWS, 'seeds': list(SEEDS),
              'note': 'each arm against its reference (the tree, or for the constraint the '
                      'learned-source graph) tuned under the same sampler seed, identical rows',
              'single_network_note': 'point contrasts with each graph arm taken one network at a '
                                     'time, keyed by initialisation seed; the constraint pairs '
                                     'networks by initialisation seed',
              'contrasts': {}}
    for variable, h, (a, b), kind in CONTRASTS:
        key = f'{variable}|{h}|{a} - {b}'
        entry = {'kind': kind, 'by_seed': {}}
        for seed in SEEDS:
            frame = frame_for(variable, h, seed)
            if a not in frame or b not in frame:
                continue
            entry['by_seed'][str(seed)] = paired(frame, a, b, f'{key}|{seed}')
        points = [v['point'] for v in entry['by_seed'].values()]
        signs = {np.sign(p) for p in points}
        entry['range'] = round(max(points) - min(points), 4)
        entry['same_sign'] = len(signs) == 1
        entry['starred'] = sum(v['excludes_zero'] for v in entry['by_seed'].values())
        singles = {str(seed): single_network(variable, h, a, b, seed) for seed in SEEDS}
        if any(v for v in singles.values()):
            entry['single_network'] = {k: v for k, v in singles.items() if v}
        report['contrasts'][key] = entry
        print(f"{key:48s} " + '  '.join(f"{s}: {v['point']:+.4f}{'*' if v['excludes_zero'] else ' '}"
                                      for s, v in entry['by_seed'].items())
              + f"   range {entry['range']}"
              + (f"   single networks {entry['single_network']}" if 'single_network' in entry else ''))
    OUT.write_text(json.dumps(report, indent=2))

    names = {'weather_tree': 'weather', 'tree': 'tree', 'learned_source': 'learned',
             'conservative_exchange': 'conservative'}
    rows = []
    # oxygen is stored per kilogram (the sondes' unit) and printed in mg/l, c times the stored value
    o2_digits = oxygen_digits(report['contrasts'])
    for key, e in report['contrasts'].items():
        variable, h, pair = key.split('|')
        a, b = pair.split(' - ')
        if variable == 'temperature':
            k, dig = 1.0, dict.fromkeys(o2_digits, 4)
        else:
            k, dig = factor(), o2_digits
        cells = [cell(e['by_seed'][str(s)], dig[str(s)], k) if str(s) in e['by_seed'] else '---' for s in SEEDS]
        single = e.get('single_network', {}).get('42')
        if single:
            lo, hi = min(single.values()), max(single.values())
            d = dig['single']
            spread = (f"${k * lo:+.{d}f}$ to ${k * hi:+.{d}f}$").replace('+-', '-')
        else:
            spread = '---'
        short = 'T' if variable == 'temperature' else r'O$_2$'
        kind = (r'\makecell[l]{constraint and\\operator}' if e['kind'] == 'constraint and operator'
                else e['kind'])
        rows.append(f"{h}\\,h & {short} & {kind} & {names[a]} $-$ {names[b]} & "
                    + ' & '.join(cells) + f" & ${k * e['range']:.{dig['range']}f}$ & {spread} \\\\")
    TABLE.write_text(rf"""\begin{{table}}[htbp]
\centering
\caption{{The same contrasts under three sampler seeds. Each cell is a contrast between two arms
tuned by their own $100$-trial studies, identical in everything but the seed of the TPE sampler
that draws their trials; $42$ is the protocol's seed and the one every other table uses. Every arm
is compared with its reference tuned under the same seed (the tree, or for the constraint the
learned-source graph), on identical \textsc{{Dev-2025}} endpoints on the reference issuance grid.
Negative favours the first named.}}
\label{{tab:bay_seed_spread}}
\begin{{adjustbox}}{{max width=\textwidth}}
\begin{{tabular}}{{@{{}}lllllllrl@{{}}}}
\toprule
\textbf{{Lead}} & \textbf{{Variable}} & \textbf{{Kind}} & \textbf{{Contrast}} & \textbf{{Seed 42}} &
\textbf{{Seed 7}} & \textbf{{Seed 123}} & \textbf{{Range}} & \makecell[l]{{\textbf{{Single}}\\\textbf{{networks}}}} \\
\midrule
{chr(10).join(rows)}
\bottomrule
\end{{tabular}}
\end{{adjustbox}}

\vspace{{2pt}}
\footnotesize\justifying\noindent T in $^\circ$C, O$_2$ in {OXYGEN_UNIT_TEX}. $^{{\ast}}$ interval excludes zero; paired bootstrap over (fold,
ISO week) blocks, {DRAWS} draws. Graph arms are the mean of three networks that differ only in their
initialisation seed, compared with a single tree, so their contrasts include the effect of the
averaging; the last column gives the same contrast at sampler seed 42 with each network taken alone (for
the constraint, networks paired by initialisation seed), point estimates only.
Table~\ref{{tab:bay_perfold_graph}} gives the graph rows with every arm selected per outer fold.\artifact{{runs/v2_seed_contrasts.json}}
\end{{table}}
""", encoding='utf-8')
    print('wrote', OUT, 'and', TABLE.name)


if __name__ == '__main__':
    main()
