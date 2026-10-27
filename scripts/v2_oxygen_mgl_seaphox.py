"""Addendum 60 part C4: the in-situ check of the Copernicus product at the Tinetto buoy, in mg/l.

Units, read from the raw file headers (data/raw):
  TBoaT_16m-SeapHOx.txt   'Ossigeno(mg/l)' beside 'Ossigeno(ml/l)' (ratio 1.4290, the gas-volume
                          conversion of a per-litre quantity) and no micromolar column: per litre, so
                          in mg/l it is read as written (the published script multiplied it by 31.2512)
  TBoaT_01m/16m-CTD+pH.txt 'Ossigeno(umol/l)' = 'Ossigeno(mg/l)' x 31.2512 x 1000 / 'Densita'(Kg/m^3)'
                          row by row: per kilogram, converted by 0.0319988 x rho/1000 with that row's
                          density
  Copernicus o2           mmol/m3, per litre: 0.0319988
  bay sondes              per kilogram, by their hour's factor (runs/v2_oxygen_units/hourly_factor.csv.gz)

scripts/eda_opportunities/eda_b_seaphox.py is run unchanged with those conversions installed where it
reads (eda_lib.load, eda_lib.MG_TO_UMOL and xarray.open_dataset), first in the published unit,
where it must give runs/eda_opportunities/eda_b_seaphox.json, then in mg/l. Its rounding (one to
three decimals) is widened by two decimals in the mg/l run.

Output: reports/bay_twin/runs/v2_oxygen_mgl/C4/eda_opportunities/eda_b_seaphox.json and
C4/seaphox_reproduction.json.

Usage: python -m scripts.v2_oxygen_mgl_seaphox
"""
import builtins
import contextlib
import json
import runpy
import sys

import numpy as np

from scripts.v2_oxygen_mgl_common import MG_PER_UMOL_L, OUT, ROOT, RUNS, sonde_factor, sonde_factors, write_json
from scripts.v2_oxygen_mgl_field_contrasts import same

C4 = OUT / 'C4'
SCRATCH = OUT / '_reproduction' / 'seaphox'
SCRIPT = ROOT / 'scripts/eda_opportunities/eda_b_seaphox.py'


def finer(x, n=None):
    return builtins.round(x, None if n is None else n + 2)


def header_units():
    """The oxygen columns each Tinetto file declares, and the per-row identity that fixes their unit."""
    sys.path.insert(0, str(ROOT / 'scripts/eda_opportunities'))
    import eda_lib
    out = {}
    for name in ('TBoaT_16m-SeapHOx.txt', 'TBoaT_01m-CTD+pH.txt', 'TBoaT_16m-CTD+pH.txt'):
        cols, _ = eda_lib.header(eda_lib.RAW / name)
        f, _ = eda_lib.load(name, hourly=False)
        entry = {'oxygen_columns': [c for c in cols if c.startswith('Ossigeno')]}
        ok = f[['Ossigeno(mg/l)', 'Ossigeno(ml/l)']].dropna()
        entry['mg_over_ml_median'] = round(float((ok['Ossigeno(mg/l)'] / ok['Ossigeno(ml/l)']).median()), 5)
        if 'Ossigeno(umol/l)' in f:
            ok = f[['Ossigeno(mg/l)', 'Ossigeno(umol/l)', "Densita'(Kg/m^3)"]].dropna()
            per_kg = ok['Ossigeno(mg/l)'] * 31.2512 * 1000 / ok["Densita'(Kg/m^3)"]
            rel = (ok['Ossigeno(umol/l)'] / per_kg - 1).abs()
            entry['umol_equals_mg_x_31.2512_x_1000_over_rho'] = {'rows': int(len(ok)),
                                                                  'within_0.1pct': round(float((rel < 1e-3).mean()), 4),
                                                                  'median_abs_relative': float(rel.median())}
            entry['unit'] = 'umol/kg (per kilogram)'
        else:
            entry['unit'] = 'mg/l per litre (mg/l over ml/l is the gas-volume factor 1.42905)'
        out[name] = entry
    return out


@contextlib.contextmanager
def installed(mgl):
    sys.path.insert(0, str(ROOT / 'scripts/eda_opportunities'))
    import eda_lib
    import xarray as xr
    saved = (eda_lib.load, eda_lib.MG_TO_UMOL, eda_lib.OUT, xr.open_dataset)
    load, open_dataset = saved[0], saved[3]
    gen2 = set(sonde_factors())

    def load_mgl(name, hourly=True, how='mean'):
        f, meta = load(name, hourly, how)
        stem = name[:-4]
        if 'Ossigeno(umol/l)' in f:
            f = f.copy()
            if stem in gen2:
                if not hourly:
                    raise SystemExit(f'{name}: a Gen-2 sonde read at its own timestamps')
                f['Ossigeno(umol/l)'] = f['Ossigeno(umol/l)'].to_numpy(float) * sonde_factor(stem, f.index)
            elif "Densita'(Kg/m^3)" in f:
                rho = f["Densita'(Kg/m^3)"].where(f["Densita'(Kg/m^3)"].between(1000, 1040))
                f['Ossigeno(umol/l)'] = f['Ossigeno(umol/l)'] * MG_PER_UMOL_L * rho / 1000
            else:
                raise SystemExit(f'{name}: no density to convert its micromolar oxygen')
        return f, meta

    def open_mgl(*a, **k):
        ds = open_dataset(*a, **k)
        if 'o2' in ds:
            ds = ds.assign(o2=ds.o2 * MG_PER_UMOL_L)
        return ds
    try:
        if mgl:
            eda_lib.load, eda_lib.MG_TO_UMOL, xr.open_dataset = load_mgl, 1.0, open_mgl
        yield eda_lib
    finally:
        eda_lib.load, eda_lib.MG_TO_UMOL, eda_lib.OUT, xr.open_dataset = saved


def run(folder, mgl):
    folder.mkdir(parents=True, exist_ok=True)
    with installed(mgl) as eda_lib:
        eda_lib.OUT = folder
        runpy.run_path(str(SCRIPT), init_globals={'round': finer} if mgl else {}, run_name='__main__')
    out = json.loads((folder / 'eda_b_seaphox.json').read_text())
    write_json(folder / 'eda_b_seaphox.json', out)
    return out


def main():
    record = {'what': 'eda_b_seaphox rerun in the published unit against its published artifact, then in mg/l',
              'units_from_headers': header_units()}
    got = run(SCRATCH, False)
    pub = json.loads((RUNS / 'eda_opportunities/eda_b_seaphox.json').read_text())
    diff = same(got, pub, tol=0)
    record['reproduction'] = {'pass': not diff, 'differences': diff[:20]}
    print('reproduce eda_b_seaphox', 'ok' if not diff else diff[:4], flush=True)
    if diff:
        write_json(C4 / 'seaphox_reproduction.json', record)
        raise SystemExit('eda_b_seaphox does not reproduce')
    mgl = run(C4 / 'eda_opportunities', True)
    mgl['harmonised'] = {'declared_in': 'config/bay_reserved_protocol_v2_addendum60.json',
                         'unit': 'mg/l: SeapHOx as written (per litre), buoy casts by their row density, '
                                 'Copernicus by 0.0319988, bay sondes by their hour factor',
                         'rounding': 'two more decimals than published'}
    write_json(C4 / 'eda_opportunities/eda_b_seaphox.json', mgl)
    record['pass'] = True
    write_json(C4 / 'seaphox_reproduction.json', record)
    print(json.dumps(mgl['copernicus_vs_seaphox_16m'], indent=1))


if __name__ == '__main__':
    main()
