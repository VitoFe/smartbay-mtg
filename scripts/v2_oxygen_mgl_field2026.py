"""Addendum 60 parts C1 and C2: the Field-2026 reading (addenda 36 and 38) with oxygen in mg/l.

The 38 casts of 2026 are per kilogram; the cast-derived prior and the seasonal climatology read the
722-cast archive (219 per-kilogram casts, 502 per-litre), and the Copernicus arms read o2 in mmol/m3
(per litre). Both producers are rerun as published, with their published penalties (the addendum-36
studies are not run again: their selections are handed back to them), first in the published unit,
where each must give its published report, then with every oxygen input in mg/l
(scripts/v2_oxygen_mgl_common.py; the 2026 Copernicus extraction, read from netCDF, is converted by
0.0319988 where it is extracted).

The published reports round every error and contrast to four decimals, which in mg/l would keep three
significant figures of an oxygen contrast; the mg/l reports keep six (the reproduction is checked at
the published four).

Outputs: C2/v2_field_reserved2026/report.json (+ predictions), C1/v2_field_reserved2026_copernicus/
report.json (+ predictions), C1/field2026_reproduction.json.

Usage: python -m scripts.v2_oxygen_mgl_field2026
"""
import builtins
import json
import shutil
import time
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.v2_oxygen_mgl_common import MG_PER_UMOL_L, OUT, RUNS, harmonised, write_csv, write_json
from scripts.v2_oxygen_mgl_field_contrasts import same

C1, C2 = OUT / 'C1', OUT / 'C2'
SCRATCH = OUT / '_reproduction'
PUB36 = RUNS / 'v2_field_reserved2026/report.json'
PUB38 = RUNS / 'v2_field_reserved2026_copernicus/report.json'


def finer(x, n=None):
    return builtins.round(x, None if n is None else n + 2)


def relf(path):
    """Rewrite the producer's CSVs with LF and no gzip time, and its report as LF JSON."""
    path = Path(path)
    report = json.loads((path / 'report.json').read_text())
    for p in sorted(path.glob('*.csv.gz')):
        write_csv(pd.read_csv(p), p)
    import hashlib
    for label in ('held_station', 'station_seen'):
        if label in report:
            report[label]['prediction_sha256'] = hashlib.sha256(
                (path / f'reserved2026_{label}_predictions.csv.gz').read_bytes()).hexdigest()
    write_json(path / 'report.json', report)
    return report


def run36(out, mgl):
    from src.experiments import v2_field_reserved2026 as R
    published = json.loads(PUB36.read_text())
    selections = {f'held_site_{a}_dev': published['development'][a]['selection'] for a in ('A2_none', 'A7o')}
    saved = (R.OUT, R.run_study, R.PUBLISHED_A2)
    R.OUT = out
    R.run_study = lambda name, objective, folder, quick=False: selections[name]
    if mgl:
        R.round = finer
        # The study's own gate (A2 at its penalty gives the published Field-57 temperature) is the
        # published-unit reproduction above. In mg/l the oxygen context features move the shared
        # ridge design, and so temperature, at the 1e-5 level; the gate then checks that this
        # producer's A2 gives the mg/l A2 of scripts/v2_oxygen_mgl_field.py, an independent path.
        R.PUBLISHED_A2 = {'alpha': R.PUBLISHED_A2['alpha'], 'temp_c': field57_mgl('A2_none_direct')}
    try:
        with harmonised(mgl):
            R.main(False)
    finally:
        R.OUT, R.run_study, R.PUBLISHED_A2 = saved
        if hasattr(R, 'round'):
            del R.round
    return relf(out)


def run38(out, a36, mgl):
    from src.experiments import v2_field_reserved2026 as R
    from src.experiments import v2_field_reserved2026_copernicus as RC
    saved = (RC.OUT, RC.ADDENDUM36, RC.boundary, RC.PUBLISHED)
    original = RC.boundary

    def boundary_mgl(data, prods):
        out_, stale = original(data, prods)
        out_ = out_.copy()
        out_[:, 2] *= MG_PER_UMOL_L          # boundary_oxygen, mmol/m3
        out_[:, 5] *= MG_PER_UMOL_L          # boundary_oxygen_vertical_delta
        return out_, stale
    RC.OUT, RC.ADDENDUM36 = out, a36
    if mgl:
        RC.boundary = boundary_mgl
        R.round = finer
        RC.round = finer
        RC.PUBLISHED = {k: {'alpha': v['alpha'], 'temp_c': field57_mgl(f'{k}_direct')} for k, v in RC.PUBLISHED.items()}
    try:
        with harmonised(mgl):
            RC.main()
    finally:
        RC.OUT, RC.ADDENDUM36, RC.boundary, RC.PUBLISHED = saved
        for m in (R, RC):
            if hasattr(m, 'round'):
                del m.round
    return relf(out)


def field57_mgl(arm):
    """The mg/l Field-57 temperature error of an arm, from scripts/v2_oxygen_mgl_field.py."""
    r = json.loads((C1 / f'20260919T145045Z_v2_field/results/v2_field_held_site_{arm}/report.json').read_text())
    return r['nested_mae']['temp_c']


def strip(report):
    """The parts of a report that must reproduce (hashes of CSV bytes and smoke flags excepted)."""
    r = json.loads(json.dumps(report))
    for label in ('held_station', 'station_seen'):
        if label in r:
            r[label].pop('prediction_sha256', None)
    return r


def main():
    t0 = time.time()
    record = {'what': 'the Field-2026 producers rerun in the published unit with their published penalties, '
                      'against the published reports, before the mg/l run', 'steps': {}}
    # a reproduction already on disk is checked again, not recomputed (it takes forty minutes)
    done = ((SCRATCH / 'v2_field_reserved2026/report.json').exists()
            and (SCRATCH / 'v2_field_reserved2026_copernicus/report.json').exists())
    got = (json.loads((SCRATCH / 'v2_field_reserved2026/report.json').read_text()) if done
           else run36(SCRATCH / 'v2_field_reserved2026', False))
    diff = same(strip(got), strip(json.loads(PUB36.read_text())), tol=0)
    record['steps']['v2_field_reserved2026'] = {'pass': not diff, 'differences': diff[:20]}
    print('addendum 36 reproduction', 'ok' if not diff else diff[:5], flush=True)
    if diff:
        write_json(C1 / 'field2026_reproduction.json', record)
        raise SystemExit('addendum 36 does not reproduce')
    got = (json.loads((SCRATCH / 'v2_field_reserved2026_copernicus/report.json').read_text()) if done
           else run38(SCRATCH / 'v2_field_reserved2026_copernicus', PUB36, False))
    diff = same(strip(got), strip(json.loads(PUB38.read_text())), tol=0)
    record['steps']['v2_field_reserved2026_copernicus'] = {'pass': not diff, 'differences': diff[:20]}
    print('addendum 38 reproduction', 'ok' if not diff else diff[:5], flush=True)
    if diff:
        write_json(C1 / 'field2026_reproduction.json', record)
        raise SystemExit('addendum 38 does not reproduce')
    print(f'reproduced in {time.time() - t0:.0f} s', flush=True)
    record['reproduction_reused_from_disk'] = bool(done)
    write_json(C1 / 'field2026_reproduction.json', record)
    r36 = run36(C2 / 'v2_field_reserved2026', True)
    r36['harmonised'] = {'declared_in': 'config/bay_reserved_protocol_v2_addendum60.json',
                         'unit': 'mg/l (columns keep their published names)',
                         'rounding': 'six decimals where the published report keeps four',
                         'penalties': 'the published addendum-36 selections, not re-selected'}
    write_json(C2 / 'v2_field_reserved2026/report.json', r36)
    r38 = run38(C1 / 'v2_field_reserved2026_copernicus', C2 / 'v2_field_reserved2026/report.json', True)
    r38['harmonised'] = {**r36['harmonised'], 'penalties': 'the published Field-57 and addendum-36 penalties'}
    write_json(C1 / 'v2_field_reserved2026_copernicus/report.json', r38)
    shutil.rmtree(C2 / 'v2_field_reserved2026/studies', ignore_errors=True)
    record['pass'] = True
    write_json(C1 / 'field2026_reproduction.json', record)
    print(f'done in {time.time() - t0:.0f} s')


if __name__ == '__main__':
    main()
