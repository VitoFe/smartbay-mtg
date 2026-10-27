"""Addendum 32: long-gap recovery on windows whose interior may hold real short gaps.

Reads the two relaxed-design runs (Gap-Long-Dev and Gap-Long-2026) and applies the declared
analysis, and nothing else:

  per length, never pooled across lengths   the tiling only guarantees independence within one
  two resampling units                      the tiled window (primary), and probe x calendar month
                                            of the gap start (conservative); decisions use the
                                            conservative one, fewer than five is not estimable
  the Baia EXO2 subset and per-probe rows   oxygen rests on two to four probes
  a sensitivity                             scored hours within 6 h of a real gap dropped
  the four registered predictions           each checked against its declared falsifier

Only observed hours are ever scored; the build drops unobserved positions and this script asserts
it. Climatology is the declared one, which the relaxed build computes directly.

The co-located twin is scored only where it reports (`twin`). The declared comparand
`nearest_probe` is persistence wherever the twin is dark, which on this cohort is a third to a half
of the twin probes' rows, so its result is kept for the registered prediction but says little about
a twin. The twin rule falls back to bidirectional Chronos, its declared method, wherever there is no
twin reading, not only on probes without a twin. Two sensitivities of the unit are added for the
temperature arms: probe x calendar month is the declared unit, and site x calendar month merges
the co-located probes of one site, which read the same water.

Usage: python -m scripts.v2_gap_long
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.v2_gap_climatology import arm_dirs, fix_neighbours, read_arm
from scripts.v2_gap_intervals import MIN_CLUSTERS, paired
from scripts.v2_gap_neighbours import with_twin

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / 'reports/bay_twin/runs'
OUT = RUNS / 'v2_gap_long.json'
COHORTS = {'Gap-Long-Dev': ('20260922T195727Z_v2_gaprecovery', 'v2_gap_recovery_long', 'development'),
           'Gap-Long-2026': ('20260922T195745Z_v2_gaprecovery', 'v2_gap_recovery_long_reserved2026',
                             'reserved')}
ARMS = ('gpr', 'chronos_bidir')
COMPARANDS = ('pchip', 'climatology', 'nearest_probe', 'nearest_live_probe', 'gpr', 'chronos_bidir')
FLOOR = {'temperature': 0.20, 'oxygen': 1.8}
EXO2 = 'TBaiaP_01m-CTD(calibrato)'
NEAR_REAL_GAP = 6
#: What the pretrained fillers forecast in one pass, so that a long-gap row can say whether it
#: describes the model at its trained horizon or its own continuation. Read
#: from `chronos_config` in each model's config.json on the Hugging Face hub on 2026-09-26; asked
#: for more, the chronos-forecasting pipeline warns that quality may degrade and appends its own
#: forecast to the context, block by block. v2_gap_recovery gives both directions CHRONOS_CTX hours.
MODEL_HORIZONS = {
    'amazon/chronos-bolt-base': {'native_output_h': 64, 'longest_context_h': 2048},
    'amazon/chronos-2': {'native_output_h': 1024, 'output_patch_size': 16, 'max_output_patches': 64,
                         'longest_context_h': 8192},
    'context_given_h': 512,
    'source': "chronos_config in each model's config.json on the Hugging Face hub, read 2026-09-26",
}


def load(run, results, variable, period):
    merged = None
    for arm in ARMS:
        # the Gaussian process is read from its addendum-59 rerun, which must score these rows
        frame = fix_neighbours(read_arm(run, results, variable, arm), run, variable)
        report = json.loads((arm_dirs(run, results)[(variable, arm)] / 'report.json').read_text())
        if report.get('design') != 'relaxed':
            raise SystemExit(f'{run}/{variable}_{arm} is not a relaxed-design run')
        if merged is None:
            merged = frame
        else:
            if not merged[['episode', 'position']].equals(frame[['episode', 'position']]):
                raise SystemExit(f'{run}: arms scored different rows')
            merged[arm] = frame[arm].to_numpy()
    if merged.truth.isna().any():
        raise SystemExit(f'{run} {variable}: an unobserved hour was scored')
    if variable == 'temperature':
        with_twin(merged, variable, period)
        merged['twin_rule'] = merged.twin.where(merged.twin_reports, merged.chronos_bidir)
    return merged


SITE = pd.read_csv(ROOT / 'reports/bay_twin/sensor_geometry.csv').set_index('sensor').site


def units(frame):
    """episode -> primary unit (itself) and -> conservative unit (probe, month of gap start)."""
    start = frame.episode.str.rsplit('|', n=2).str[1].str[:7]
    conservative = dict(zip(frame.episode, frame.sensor + '|' + start))
    return {e: e for e in frame.episode.unique()}, conservative


def site_units(frame):
    """episode -> (site, month of gap start): co-located probes of one site are one unit."""
    start = frame.episode.str.rsplit('|', n=2).str[1].str[:7]
    return dict(zip(frame.episode, frame.sensor.map(SITE) + '|' + start))


def near_real_gap(frame, period, variable):
    """Hours from each scored row to the nearest unobserved hour of the same probe."""
    if period == 'development':
        from src.data.bay_profile_context import load_observations
        _, frames = load_observations()
    else:
        from src.data.bay_observations import read_export
        frames = {}
        for name in frame.sensor.unique():
            _, f, _ = read_export(ROOT / 'data/raw' / (name + '.txt'), include_auxiliary=True)
            frames[name] = f[f.index >= pd.Timestamp('2026-01-01', tz='UTC')]
    out = np.full(len(frame), np.inf)
    times = pd.to_datetime(frame.time, utc=True)
    for name, idx in frame.groupby('sensor').groups.items():
        s = frames[name][variable]
        grid = pd.date_range(s.dropna().index.min(), s.dropna().index.max(), freq='h')
        missing = grid[s.reindex(grid).isna().to_numpy()]
        if not len(missing):
            continue
        m = missing.as_unit('ns').asi8
        t = pd.DatetimeIndex(times.loc[idx]).as_unit('ns').asi8
        pos = np.searchsorted(m, t)
        left = np.abs(t - m[np.clip(pos - 1, 0, len(m) - 1)])
        right = np.abs(m[np.clip(pos, 0, len(m) - 1)] - t)
        out[frame.index.get_indexer(idx)] = np.minimum(left, right) / 3.6e12
    return out


def macro(frame, column):
    d = frame.dropna(subset=[column, 'truth'])
    if not len(d):
        return None
    return round(float(d.assign(e=(d.truth - d[column]).abs()).groupby('episode').e.mean().mean()), 4)


def contrast(frame, arm, label, mappings, reference='linear'):
    """`arm` minus `reference` under each unit. The seed label of a contrast with linear is the one
    the published artifact used, so adding units or references moves no existing interval."""
    out = {}
    for name, mapping in mappings.items():
        tag = f'{label}|{name}' if reference == 'linear' else f'{label}|{reference}|{name}'
        got = paired(frame, arm, reference, tag, mapping)
        if got:
            out[name] = got
    return out


def block(frame, variable, label, twin=False):
    """Every comparand against linear at every length, by every unit.

    With `twin`, the twin probes' block: the twin scored where it reports, against linear and
    against the Gaussian process, with every error also given on those rows alone.
    """
    primary, conservative = units(frame)
    mappings = {'primary': primary, 'conservative': conservative}
    if variable == 'temperature':
        mappings['site'] = site_units(frame)
    arms = COMPARANDS + (('twin_rule',) if variable == 'temperature' else ()) \
        + (('twin',) if twin else ())
    rows = {}
    for length, part in frame.groupby('gap_length'):
        row = {'episodes': int(part.episode.nunique()), 'probes': int(part.sensor.nunique()),
               'conservative_units': int(len({conservative[e] for e in part.episode.unique()})),
               'mae': {c: macro(part, c) for c in ('linear',) + arms},
               'minus_linear': {}}
        if variable == 'temperature':
            row['site_units'] = int(len({mappings['site'][e] for e in part.episode.unique()}))
        for arm in arms:
            got = contrast(part, arm, f'{label}|{variable}|{arm}|{length}', mappings)
            if got:
                for unit in got.values():
                    unit['exceeds_floor'] = bool(-unit['delta'] > FLOOR[variable])
                row['minus_linear'][arm] = got
        if twin:
            live = part[part.twin_reports]
            row['where_twin_reports'] = {
                'rows': int(len(live)), 'share_of_rows': round(float(part.twin_reports.mean()), 3),
                'episodes': int(live.episode.nunique()), 'probes': int(live.sensor.nunique()),
                'conservative_units': int(len({conservative[e] for e in live.episode.unique()})),
                'mae': {c: macro(live, c) for c in ('linear', 'twin', 'gpr', 'chronos_bidir')}}
            row['twin_minus_gpr'] = contrast(part, 'twin', f'{label}|{variable}|twin|{length}',
                                             mappings, reference='gpr')
        rows[str(int(length))] = row
    return rows


def per_probe(frame):
    out = {}
    for (sensor, length), part in frame.groupby(['sensor', 'gap_length']):
        out.setdefault(sensor, {})[str(int(length))] = {
            'episodes': int(part.episode.nunique()),
            **{c: macro(part, c) for c in ('linear', 'chronos_bidir', 'gpr', 'climatology')}}
    return out


def verdicts(result):
    """The four registered predictions against their declared falsifiers, on Gap-Long-Dev."""
    dev = result['Gap-Long-Dev']
    ox, te = dev['oxygen']['by_gap_length'], dev['temperature']['twin_probes']

    def cons(row, arm):
        return row['minus_linear'].get(arm, {}).get('conservative', {})

    c168 = cons(ox['168'], 'chronos_bidir')
    p1a = bool(c168.get('estimable') and c168.get('excludes_zero') and c168['delta'] < 0
               and -c168['delta'] < FLOOR['oxygen'])
    beyond = [a for L in ('720', '1440') for a in COMPARANDS
              if cons(ox[L], a).get('estimable') and cons(ox[L], a).get('excludes_zero')
              and -cons(ox[L], a)['delta'] > FLOOR['oxygen']]
    def p2_of(arm):
        rows = {L: cons(r, arm) for L, r in te.items()}
        clears = all(u.get('estimable') and u.get('excludes_zero') and u['delta'] < 0
                     for u in rows.values())
        floor = all(rows[L].get('delta') is not None and -rows[L]['delta'] > FLOOR['temperature']
                    for L in ('720', '1440') if L in rows)
        return rows, bool(clears and floor)

    # The declared comparand, and the twin scored only where it reports.
    p2_rows, p2_held = p2_of('nearest_probe')
    p2_twin_rows, p2_twin_held = p2_of('twin')
    exo = dev['oxygen']['exo2']['720']['mae']
    best = min((v, k) for k, v in exo.items() if v is not None and k != 'linear')
    te_all = dev['temperature']['by_gap_length']
    clim_temp = [L for L, r in te_all.items()
                 if not (cons(r, 'climatology').get('excludes_zero') and cons(r, 'climatology')['delta'] > 0)]
    clim_ox = cons(ox['1440'], 'climatology')
    return {
        '1': {'held': bool(p1a and not beyond), 'chronos_168_conservative': c168,
              'beyond_floor_at_720_or_1440': beyond},
        '2': {'held': p2_held, 'twin_probe_nearest_minus_linear': p2_rows,
              'held_with_the_twin_scored_where_it_reports': p2_twin_held,
              'twin_minus_linear_where_it_reports': p2_twin_rows},
        '3': {'held': bool(best[0] > 5.0), 'best_method_at_720_on_exo2': {best[1]: best[0]},
              'linear': exo.get('linear')},
        '4': {'held': bool(not clim_temp and not (clim_ox.get('estimable') and clim_ox.get('excludes_zero'))),
              'temperature_lengths_where_climatology_is_not_worse': clim_temp,
              'oxygen_1440_climatology_minus_linear': clim_ox},
    }


def main():
    lengths = (168, 336, 504, 720, 1440)
    horizons = {**MODEL_HORIZONS, 'forecast_blocks_by_length': {
        model: {str(L): -(-L // MODEL_HORIZONS[model]['native_output_h']) for L in lengths}
        for model in ('amazon/chronos-bolt-base', 'amazon/chronos-2')}}
    result = {'declared_in': 'config/bay_reserved_protocol_v2_addendum32.json',
              'min_units': MIN_CLUSTERS, 'near_real_gap_hours': NEAR_REAL_GAP,
              'cohorts': {k: {'run': v[0], 'results': v[1]} for k, v in COHORTS.items()},
              'model_horizons': horizons}
    for cohort, (run, results, period) in COHORTS.items():
        result[cohort] = {}
        for variable in ('oxygen', 'temperature'):
            frame = load(run, results, variable, period)
            label = f'long|{cohort}'
            entry = {'rows': int(len(frame)), 'episodes': int(frame.episode.nunique()),
                     'by_gap_length': block(frame, variable, label)}
            frame['near_gap_h'] = near_real_gap(frame, period, variable)
            away = frame[frame.near_gap_h > NEAR_REAL_GAP]
            entry['sensitivity_far_from_real_gaps'] = {
                'rows_kept': int(len(away)), 'rows_dropped': int(len(frame) - len(away)),
                'by_gap_length': block(away, variable, label + '|far')}
            entry['per_probe'] = per_probe(frame)
            if variable == 'oxygen':
                entry['exo2'] = block(frame[frame.sensor == EXO2], variable, label + '|exo2')
            else:
                entry['twin_probes'] = block(frame[frame.has_twin], variable, label + '|twin',
                                             twin=True)
            result[cohort][variable] = entry
    result['registered_predictions'] = verdicts(result)
    result['arm_sources'] = {cohort: {f'{v}/{a}': arm_dirs(run, results)[(v, a)].relative_to(RUNS).as_posix()
                                      for v in ('oxygen', 'temperature') for a in ARMS}
                             for cohort, (run, results, _) in COHORTS.items()}
    OUT.write_bytes((json.dumps(result, indent=2) + '\n').encode('utf-8'))

    for cohort in COHORTS:
        for variable in ('oxygen', 'temperature'):
            print(f'\n== {cohort} {variable}  (floor {FLOOR[variable]}; conservative unit; * excludes 0)')
            for L, row in result[cohort][variable]['by_gap_length'].items():
                cells = []
                for arm in ('chronos_bidir', 'gpr', 'nearest_live_probe', 'climatology', 'pchip'):
                    u = row['minus_linear'].get(arm, {}).get('conservative')
                    if not u:
                        continue
                    mark = ('*' if u['excludes_zero'] else '') if u['estimable'] else ' n.e.'
                    cells.append(f"{arm[:8]} {u['delta']:+.3f}{mark}")
                print(f"  {L:>5} h  n={row['episodes']:3d} units={row['conservative_units']:3d}  "
                      f"linear {row['mae']['linear']:.3f}  " + '  '.join(cells))
    print('\nregistered predictions:',
          {k: v['held'] for k, v in result['registered_predictions'].items()})
    print('wrote', OUT)


if __name__ == '__main__':
    main()
