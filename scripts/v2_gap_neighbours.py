"""The nearest-probe comparand, rebuilt without the self-neighbour leak.

`v2_gap_recovery.build` took a probe's neighbours as `argsort(distance)[1:]`, which assumes the
probe itself sorts first. Six probe pairs share coordinates, and on a tie at zero the sort placed
the twin first for five temperature probes (Nodo_S09, TMitiliE, TMitiliO, TPortovenereP,
TPortovenereS). Slicing off position 0 then dropped the twin and kept the probe, so neighbour 0 was
the probe's own unmasked reading and `nearest_probe` returned the truth on every row of those
probes. Oxygen is untouched: no two sondes share a site.

The runs are left exactly as they were written. This rebuilds each run's episodes under the
environment recorded in its transport.json with the ordering fixed, checks they match the published
predictions row for row, checks that `nearest_probe` is unchanged on every probe the leak did not
touch, and writes the corrected column to `runs/v2_gap_neighbours/`, which
`v2_gap_climatology.load_predictions` and `fix_neighbours` read in place of the published one.

The fitted arms that read the `nb*` columns as features were given the truth as well on those
probes; they are rerun, not patched, and mapped in `v2_gap_climatology.REPLACED_ARMS`.

`neighbour0_reports` answers a second question about the same comparand. `nearest_probe` is the
offset-corrected reading of neighbour 0 where that probe reports and persistence where it is dark,
so on a probe with a co-located twin it mixes two fills. On the long development cohort the twin is
dark on 34 to 48 % of the twin probes' scored rows, and three of the six twin probes never have a
reporting twin there; scored over all rows, the persistence rows decide the twin's result. The
function marks the rows where neighbour 0 was observed both at the hour before the gap and at the
scored hour, which are exactly the rows on which `nearest_probe` is the twin's reading, so a twin
can be scored only where it reports and a twin rule can fall back to its declared method where it
does not.

Usage: python -m scripts.v2_gap_neighbours [<run-id> ...]   (default: every run the thesis reads)
"""
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / 'reports/bay_twin/runs'
SIDECARS = RUNS / 'v2_gap_neighbours'
SUMMARY = RUNS / 'v2_gap_neighbours.json'
GZIP = {'method': 'gzip', 'mtime': 0}

#: Every temperature gap-recovery run whose baseline columns a thesis table, figure or decision
#: reads: addendum 30 (strict, both periods, second seed), 32 (long, both periods), 33, 34 and 35.
RUNS_READ = (
    '20260922T171859Z_v2_gaprecovery', '20260922T180814Z_v2_gaprecovery',
    '20260922T180829Z_v2_gaprecovery', '20260922T195727Z_v2_gaprecovery',
    '20260922T195745Z_v2_gaprecovery', '20260922T204658Z_v2_gaprecovery',
    '20260922T204715Z_v2_gaprecovery', '20260922T204732Z_v2_gaprecovery',
    '20260922T204749Z_v2_gaprecovery', '20260922T215308Z_v2_gaprecovery',
    '20260922T215328Z_v2_gaprecovery', '20260922T215413Z_v2_gaprecovery',
    '20260922T215431Z_v2_gaprecovery', '20260922T223901Z_v2_gaprecovery',
    '20260922T223918Z_v2_gaprecovery', '20260922T224158Z_v2_gaprecovery',
    '20260922T224215Z_v2_gaprecovery',
)
VARIABLE = 'temperature'


def sidecar(run):
    return SIDECARS / f'{run}__{VARIABLE}.csv.gz'


_PANELS = {}


def _panel(variable, period):
    """The hourly panel and inter-probe distances `v2_gap_recovery.build` used for this period."""
    if (variable, period) not in _PANELS:
        from src.experiments.v2_gap_recovery import observations
        meta, frames = observations(period)
        names = [n for n, f in frames.items() if variable in f and f[variable].notna().any()]
        index = pd.date_range(min(frames[n].index.min() for n in names),
                              max(frames[n].index.max() for n in names), freq='h', tz='UTC')
        values = pd.DataFrame({n: frames[n][variable].reindex(index) for n in names},
                              index=index).to_numpy(float)
        xy = meta.loc[names, ['x_m', 'y_m']].to_numpy()
        distance = np.linalg.norm(xy[:, None] - xy[None, :], axis=-1)
        _PANELS[(variable, period)] = (names, index, values, distance)
    return _PANELS[(variable, period)]


def neighbour0_reports(frame, variable, period):
    """True where the nearest other probe was observed at the hour before the gap and at that hour.

    These are the rows on which `nearest_probe` is neighbour 0's offset-corrected reading; on the
    others it is persistence (neighbour 0 dark at that hour) or undefined (dark at the gap's edge).
    `period` is 'development' or 'reserved', the record the run was built from. Neighbour 0 is
    chosen by `nearest_others`, as in `build`, so on a probe with a co-located twin it is the twin.
    """
    from src.experiments.v2_gap_recovery import nearest_others
    names, index, values, distance = _panel(variable, period)
    column = {n: i for i, n in enumerate(names)}
    where = {t: k for k, t in enumerate(index)}
    out = np.zeros(len(frame), bool)
    for key, block in frame.groupby('episode', sort=False):
        sensor, start, _ = key.rsplit('|', 2)
        i, g0 = column[sensor], where[pd.Timestamp(start)]
        j = nearest_others(distance, i, 1)[0]
        t = g0 + block.position.to_numpy(int) - 1
        out[frame.index.get_indexer(block.index)] = (np.isfinite(values[t, j])
                                                     & np.isfinite(values[g0 - 1, j]))
    reading = frame.nearest_probe.to_numpy(float)
    if not np.isfinite(reading[out]).all():
        raise SystemExit('nearest_probe undefined on a row where neighbour 0 reports')
    return out


def with_twin(frame, variable, period):
    """Add `twin_reports` (a twin reading exists on this row) and `twin` (that reading, else NaN).

    Only probes with a co-located twin get a reading. A twin rule is then
    `frame.twin.where(frame.twin_reports, <its declared method>)`.
    """
    live = neighbour0_reports(frame, variable, period) & frame.has_twin.to_numpy(bool)
    frame['twin_reports'] = live
    frame['twin'] = frame.nearest_probe.where(live)
    return frame


def worker(run):
    """Runs under the run's own environment, so the module-level switches match what it used."""
    from src.experiments.v2_gap_recovery import OUT, SEED, build
    published = sorted((RUNS / run / OUT.name).glob(f'{VARIABLE}_*/predictions.csv.gz'))
    if not published:
        raise SystemExit(f'no {VARIABLE} predictions under {run}/{OUT.name}')
    pub = pd.read_csv(published[0], usecols=['episode', 'position', 'sensor', 'truth',
                                              'nearest_probe'])
    frame = build(VARIABLE, np.random.default_rng(SEED))
    new = frame[['episode', 'position', 'gap_length', 'sensor', 'has_twin', 'truth',
                 'nearest_probe', 'nb0_dist']]
    joined = pub.merge(new, on=['episode', 'position'], how='outer',
                       suffixes=('_published', ''), indicator=True)
    unmatched = int((joined._merge != 'both').sum())
    drift = float(np.nanmax(np.abs(joined.truth - joined.truth_published)))
    if unmatched or drift > 1e-9:
        raise SystemExit(f'{run}: rebuilt episodes do not match the published ones '
                         f'({unmatched} unmatched rows, truth drift {drift})')
    joined['leaked'] = np.isclose(joined.nearest_probe_published, joined.truth)
    report = {'results': OUT.name, 'rows': int(len(joined)), 'sensors': {}}
    for sensor, block in joined.groupby('sensor'):
        changed = ~np.isclose(block.nearest_probe_published, block.nearest_probe,
                              rtol=0, atol=1e-12, equal_nan=True)
        report['sensors'][sensor] = {
            'rows': int(len(block)), 'has_twin': bool(block.has_twin.iloc[0]),
            'published_equals_truth': round(float(block.leaked.mean()), 4),
            'corrected_equals_truth': round(float(np.isclose(block.nearest_probe,
                                                             block.truth).mean()), 4),
            'rows_changed': int(changed.sum()),
            'mae_published': round(float((block.nearest_probe_published - block.truth)
                                         .abs().mean()), 4),
            'mae_corrected': round(float((block.nearest_probe - block.truth).abs().mean()), 4)}
    # The fix may change a probe only where it changes the probe's neighbour 0. That is every probe
    # whose published fill was its own reading, and TMitiliE, whose published order dropped one of
    # its two zero-distance twins (Nodo_S02) as if it were itself and kept the other. Anything else
    # moving means the rebuild differs from the run for a reason other than the leak.
    from src.experiments.v2_gap_recovery import nearest_others, observations
    meta, frames = observations()
    names = [n for n, f in frames.items() if VARIABLE in f and f[VARIABLE].notna().any()]
    xy = meta.loc[names, ['x_m', 'y_m']].to_numpy()
    distance = np.linalg.norm(xy[:, None] - xy[None, :], axis=-1)
    reordered = {names[i] for i in range(len(names))
                 if int(np.argsort(distance[i])[1]) != nearest_others(distance, i, 1)[0]}
    for sensor, r in report['sensors'].items():
        r['neighbour0_changed'] = sensor in reordered
    moved = [s for s, r in report['sensors'].items() if r['rows_changed'] and s not in reordered]
    if moved:
        raise SystemExit(f'{run}: nearest_probe changed on probes the leak did not reach: {moved}')
    SIDECARS.mkdir(parents=True, exist_ok=True)
    out = joined[['episode', 'position', 'sensor', 'has_twin', 'nearest_probe']]
    out.sort_values(['episode', 'position']).to_csv(sidecar(run), index=False, compression=GZIP)
    print(json.dumps(report))


def main():
    if sys.argv[1:2] == ['--worker']:
        return worker(sys.argv[2])
    runs = sys.argv[1:] or list(RUNS_READ)
    summary = {'definition': 'offset-corrected reading of the nearest OTHER probe, persistence '
                             'when it is dark', 'variable': VARIABLE, 'runs': {}}
    for run in runs:
        recorded = json.loads((RUNS / run / 'transport.json').read_text()).get('forwarded_env', {})
        env = {k: v for k, v in os.environ.items() if not k.startswith('V2_GAP_')}
        env.update(recorded)
        env['PYTHONHASHSEED'] = '0'
        done = subprocess.run([sys.executable, '-m', 'scripts.v2_gap_neighbours', '--worker', run],
                              cwd=ROOT, env=env, capture_output=True, text=True)
        if done.returncode:
            raise SystemExit(done.stderr[-2000:] or done.stdout[-2000:])
        report = json.loads(done.stdout.strip().splitlines()[-1])
        summary['runs'][run] = {'forwarded_env': recorded, **report}
        leaked = {s: r['mae_published'] for s, r in report['sensors'].items()
                  if r['published_equals_truth'] > 0.99}
        print(f'{run}: {report["rows"]} rows matched; leaked probes {sorted(leaked)}')
    SUMMARY.write_text(json.dumps(summary, indent=2))
    print('wrote', SUMMARY)


if __name__ == '__main__':
    main()
