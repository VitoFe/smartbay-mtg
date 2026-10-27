"""Addendum 60 amendment 3: the moorings' daily cycle with oxygen saturation in one unit.

scripts/v2_daily_cycle.py reports oxygen saturation as the sondes' per-kilogram oxygen over the
per-litre Garcia-Gordon solubility, a ratio of two units, low by the density ratio (about 2.6 %);
multiplying a ratio by c does not repair it. This reruns that producer unchanged except for the unit
of what it reads: the sonde oxygen in mg/l through its hourly factor (c where a sonde-hour has no
density) and the solubility per litre times 0.0319988, both installed by
scripts.v2_oxygen_mgl_common.harmonised. It first runs the producer in the published unit and
requires the published runs/v2_daily_cycle.json back value for value; only then does it run in mg/l.

The rerun keeps the published schema, with every value rounded two decimals further than published
(the oxygen swings are some thirty times smaller in mg/l), and draws its figure to a scratch folder:
the thesis figure is drawn by scripts/v2_daily_cycle.py, whose oxygen rows read this rerun.

Writes runs/v2_oxygen_mgl/C5/v2_daily_cycle.json and runs/v2_oxygen_mgl/C5/daily_cycle_reproduction.json.
Usage: python -m scripts.v2_oxygen_mgl_daily_cycle
"""
import builtins
import contextlib
import json
import math

import scripts.v2_daily_cycle as D
from scripts.v2_oxygen_mgl_common import OUT, RUNS, harmonised, write_json

C5 = OUT / 'C5'
SCRATCH = OUT / '_reproduction' / 'daily_cycle'
PUBLISHED = RUNS / 'v2_daily_cycle.json'


def finer(x, n=None):
    return builtins.round(x, None if n is None else n + 2)


@contextlib.contextmanager
def patched(obj, **attrs):
    saved = {k: obj.__dict__[k] for k in attrs if k in obj.__dict__}
    for k, v in attrs.items():
        setattr(obj, k, v)
    try:
        yield
    finally:
        for k in attrs:
            if k in saved:
                setattr(obj, k, saved[k])
            else:
                delattr(obj, k)


def run(out, figure, mgl, log=None):
    """scripts.v2_daily_cycle.main with its outputs redirected; in mg/l when `mgl`."""
    extra = dict(OUT=out, FIGURE=figure, OXYGEN_FROM=None)
    if mgl:
        extra['round'] = finer
    out.parent.mkdir(parents=True, exist_ok=True)
    figure.parent.mkdir(parents=True, exist_ok=True)
    with patched(D, **extra), harmonised(mgl, log=log):
        D.main()
    return json.loads(out.read_text())


def differences(got, want, path=''):
    """Every place two parsed reports differ (NaN equal to NaN)."""
    if isinstance(want, dict) and isinstance(got, dict):
        out = [f'{path}/{k}: missing' for k in want if k not in got]
        out += [f'{path}/{k}: extra' for k in got if k not in want]
        for k in want:
            if k in got:
                out += differences(got[k], want[k], f'{path}/{k}')
        return out
    if isinstance(want, list) and isinstance(got, list):
        if len(got) != len(want):
            return [f'{path}: length {len(got)} against {len(want)}']
        return [d for i, (a, b) in enumerate(zip(got, want)) for d in differences(a, b, f'{path}[{i}]')]
    if isinstance(want, float) and isinstance(got, float) and math.isnan(want) and math.isnan(got):
        return []
    return [] if got == want else [f'{path}: {got!r} against {want!r}']


def main():
    record = {'declared_in': ['config/bay_reserved_protocol_v2_addendum60.json',
                              'config/bay_reserved_protocol_v2_addendum60_amendment3.json'],
              'producer': 'scripts/v2_daily_cycle.py, unchanged except for the unit of its inputs',
              'published': 'runs/v2_daily_cycle.json'}
    got = run(SCRATCH / 'v2_daily_cycle.json', SCRATCH / 'fig_bay_daily_cycle_published_unit.png', False)
    diff = differences(got, json.loads(PUBLISHED.read_text()))
    record['reproduction'] = {'unit': 'published (sonde oxygen per kilogram, solubility per litre)',
                              'pass': not diff, 'differences': diff[:20]}
    print('reproduce v2_daily_cycle:', 'ok' if not diff else diff[:4], flush=True)
    if diff:
        write_json(C5 / 'daily_cycle_reproduction.json', record)
        raise SystemExit('the published unit does not reproduce runs/v2_daily_cycle.json; nothing rerun')
    seen = {}
    mgl = run(C5 / 'v2_daily_cycle.json', SCRATCH / 'fig_bay_daily_cycle_mgl.png', True, log=seen)
    write_json(C5 / 'v2_daily_cycle.json', mgl)          # LF on every host
    if not seen.get('read_export') or not seen.get('o2sol'):
        raise SystemExit(f'the mg/l conversion was not applied: {seen}')
    baia = 'Baia S. Teresa, 1 m'
    record['rerun'] = {
        'unit': 'mg/l: sonde oxygen by its hourly factor, solubility per litre x 0.0319988',
        'conversions_applied': seen,
        'rounding': 'two decimals further than published',
        'baia_1m_saturation_percent': {s: {k: mgl['saturation_percent'][s][baia][k] for k in ('mean', 'max', 'min')}
                                       for s in D.SEASONS if baia in mgl['saturation_percent'].get(s, {})},
        'baia_1m_saturation_percent_published': {
            s: {k: got['saturation_percent'][s][baia][k] for k in ('mean', 'max', 'min')}
            for s in D.SEASONS if baia in got['saturation_percent'].get(s, {})},
    }
    write_json(C5 / 'daily_cycle_reproduction.json', record)
    print('wrote', (C5 / 'v2_daily_cycle.json').relative_to(RUNS.parents[2]),
          (C5 / 'daily_cycle_reproduction.json').relative_to(RUNS.parents[2]))


if __name__ == '__main__':
    main()
