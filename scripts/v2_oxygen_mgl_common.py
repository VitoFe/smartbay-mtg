"""Addendum 60 part C: every oxygen input in mg/l, converted by its own source unit.

The part-C reruns (scripts/v2_oxygen_mgl_*.py) call the published producers with their published
configuration and change one thing, the unit of the oxygen they read. This module holds that one
change, so that every rerun applies the same conversion (addendum 60 part A, amendment 1):

  sondes        the pipeline's micromolar column is per kilogram; each hour converts by its own
                factor 0.0319988 x rho/1000 (runs/v2_oxygen_units/hourly_factor.csv.gz), an hour
                with no density by the constant c of runs/v2_oxygen_units.json
  casts         do_unit_source 'umol/l' bins are per kilogram and convert by 0.0319988 x rho/1000
                with rho the TEOS-10 in-situ density of the bin (temperature, practical salinity,
                depth as pressure in dbar); 'mg/l' and '%' bins are per litre (the one '%' cast was
                turned into micromoles with the per-litre Garcia-Gordon solubility, to 0.006 umol/l)
                and convert by 0.0319988
  Copernicus    o2 in mmol/m3 is per litre: 0.0319988
  saturation    the Garcia-Gordon per-litre form (thermodynamics.o2sol_garcia_gordon_umol_per_l)
                times 0.0319988

`harmonised()` installs the conversion where the published code reads its inputs: pandas.read_csv
of the cast archive, the hourly observation table and the Copernicus boundary context;
bay_observations.read_export for the sonde exports; and the Garcia-Gordon function wherever a module
bound it. With `enabled=False` it installs nothing, which is how every rerun first reproduces the
published values. Column names keep their published spelling (`do_umol_l`, `::oxygen`) so the outputs
have the published schema; their values are mg/l.

Normalisation constants written in micromoles (the forecasting z-score centre 230 and scale 50, the
Field-57 objective scale 20) are part of a configuration and are carried into mg/l by c, which leaves
every z-score as it was; `OXYGEN_SCALE` and `scale_constant` do that.
"""
import contextlib
import json
import sys
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
RUNS = ROOT / 'reports/bay_twin/runs'
OUT = RUNS / 'v2_oxygen_mgl'
MG_PER_UMOL_L = 0.0319988          # mg/l per umol/l (molar mass of O2, 31.9988 g/mol)
UNITS = RUNS / 'v2_oxygen_units.json'
HOURLY = RUNS / 'v2_oxygen_units/hourly_factor.csv.gz'
PER_KG_SOURCES = ('umol/l',)       # the cast files' micromolar column, per kilogram
PER_L_SOURCES = ('mg/l', '%')


@lru_cache(maxsize=1)
def constant_c():
    """c, mg/l per (umol/kg): the mean hourly factor over the five Gen-2 oxygen sondes."""
    return float(json.loads(UNITS.read_text())['c_mg_per_l_per_umol_kg'])


@lru_cache(maxsize=1)
def sonde_factors():
    """{sensor: Series of the hourly factor indexed by the hour label (UTC)}."""
    f = pd.read_csv(HOURLY)
    f['time'] = pd.to_datetime(f.time, utc=True)
    return {s: g.set_index('time').factor.sort_index() for s, g in f.groupby('sensor')}


def sonde_factor(sensor, index):
    """The factor of each hour of `index` for one sonde; c where the table has no density for it."""
    table = sonde_factors().get(sensor)
    if table is None:
        return np.full(len(index), constant_c())
    idx = pd.DatetimeIndex(index)
    idx = idx.tz_localize('UTC') if idx.tz is None else idx.tz_convert('UTC')
    out = table.reindex(idx.as_unit(table.index.unit)).to_numpy(float)
    return np.where(np.isfinite(out), out, constant_c())


def cast_factor(frame):
    """mg/l per unit of `do_umol_l` for each cast bin, by the bin's source unit."""
    from src.data.thermodynamics import seawater_density_kg_per_m3
    source = frame['do_unit_source'].astype(str).to_numpy()
    unknown = sorted(set(source) - set(PER_KG_SOURCES) - set(PER_L_SOURCES))
    if unknown:
        raise ValueError(f'cast oxygen with an unknown unit source: {unknown}')
    rho = seawater_density_kg_per_m3(frame.temp_c.to_numpy(float), frame.salinity_psu.to_numpy(float),
                                     frame.depth_m.to_numpy(float))
    per_kg = np.isin(source, PER_KG_SOURCES)
    if (per_kg & ~np.isfinite(rho)).any():
        raise ValueError('a per-kilogram cast bin has no density')
    return np.where(per_kg, MG_PER_UMOL_L * rho / 1000.0, MG_PER_UMOL_L)


def convert_casts(frame):
    """The cast archive with oxygen, its per-litre solubility and its utilisation in mg/l."""
    out = frame.copy()
    if 'do_umol_l' not in out:
        return out                      # read without its oxygen column: nothing to convert
    f = cast_factor(out)
    out['do_umol_l'] = out.do_umol_l.to_numpy(float) * f
    for col in ('sigma_bin_umol_l', 'sigma_obs_umol_l'):
        if col in out:
            out[col] = out[col].to_numpy(float) * f
    if 'o2_sol_umol_l' in out:
        out['o2_sol_umol_l'] = out.o2_sol_umol_l.to_numpy(float) * MG_PER_UMOL_L
        if 'aou_umol_l' in out:
            out['aou_umol_l'] = out.o2_sol_umol_l - out.do_umol_l
    return out


def convert_hourly(frame):
    """The wide hourly observation table with every `<sonde>::oxygen` column in mg/l."""
    out = frame.copy()
    index = pd.to_datetime(out.index if not isinstance(out.index, pd.RangeIndex) else out.iloc[:, 0], utc=True)
    for col in [c for c in out.columns if str(c).endswith('::oxygen')]:
        out[col] = out[col].to_numpy(float) * sonde_factor(col.split('::')[0], index)
    return out


def convert_boundary(frame):
    """Copernicus boundary context: o2 is mmol/m3, per litre."""
    out = frame.copy()
    for col in ('boundary_oxygen', 'boundary_oxygen_vertical_delta'):
        if col in out:
            out[col] = out[col].to_numpy(float) * MG_PER_UMOL_L
    return out


def _original_o2sol():
    from src.data.thermodynamics import o2sol_garcia_gordon_umol_per_l
    return o2sol_garcia_gordon_umol_per_l


#: The Garcia-Gordon per-litre function itself, captured when this module is imported. Inside
#: harmonised() the name is rebound, in thermodynamics too, to a version already in mg/l; a container
#: is not reached by that rebinding, so sat_mgl converts once whether or not harmonised() is active.
_ORIGINAL = {'o2sol': _original_o2sol()}


def sat_mgl(temperature_c, salinity_psu):
    return _ORIGINAL['o2sol'](temperature_c, salinity_psu) * MG_PER_UMOL_L


def scale_constant(value_umol):
    """A normalisation constant written in micromoles, carried into mg/l by c."""
    return float(value_umol) * constant_c()


_CONVERTERS = {'ctd_all_casts.csv.gz': convert_casts,
               'hourly_observations.csv.gz': convert_hourly,
               'profile_boundary_context.csv.gz': convert_boundary}


def _name(path):
    try:
        return Path(str(path)).name
    except Exception:
        return ''


def _replace_everywhere(original, replacement):
    """Rebind every module attribute that is `original` (a `from x import f` binding) to `replacement`."""
    done = []
    for mod in list(sys.modules.values()):
        d = getattr(mod, '__dict__', None)
        if not d:
            continue
        for k, v in list(d.items()):
            if v is original:
                setattr(mod, k, replacement)
                done.append((mod, k))
    return done


@contextlib.contextmanager
def harmonised(enabled=True, log=None):
    """Install the mg/l conversion for the duration of the block (nothing when `enabled` is False).

    Import the producer modules before entering, so that their `from ... import` bindings exist
    and are rebound; modules imported inside the block are patched at the source module too."""
    if not enabled:
        yield
        return
    import src.data.bay_observations as obs
    import src.data.thermodynamics as thermo
    read_csv = pd.read_csv
    read_export = obs.read_export
    o2sol = thermo.o2sol_garcia_gordon_umol_per_l
    seen = {}

    def read_csv_mgl(path, *a, **k):
        frame = read_csv(path, *a, **k)
        conv = _CONVERTERS.get(_name(path))
        if conv is not None and not k.get('nrows'):
            seen[_name(path)] = seen.get(_name(path), 0) + 1
            if conv is convert_hourly and isinstance(frame.index, pd.RangeIndex):
                frame = frame.set_index(frame.columns[0])
                frame = conv(frame).reset_index()
            else:
                frame = conv(frame)
        return frame

    def read_export_mgl(path, *a, **k):
        meta, frame, info = read_export(path, *a, **k)
        if 'oxygen' in frame:
            frame = frame.copy()
            frame['oxygen'] = frame.oxygen.to_numpy(float) * sonde_factor(Path(path).stem, frame.index)
            seen['read_export'] = seen.get('read_export', 0) + 1
        return meta, frame, info

    def o2sol_mgl(*a, **k):
        seen['o2sol'] = seen.get('o2sol', 0) + 1
        return o2sol(*a, **k) * MG_PER_UMOL_L

    pd.read_csv = read_csv_mgl
    rebound = _replace_everywhere(read_export, read_export_mgl) + _replace_everywhere(o2sol, o2sol_mgl)
    try:
        yield seen
    finally:
        pd.read_csv = read_csv
        for mod, k in rebound:
            setattr(mod, k, read_export if getattr(mod, k) is read_export_mgl else o2sol)
        if log is not None:
            log.update(seen)


def write_json(path, obj):
    """LF on every host."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(json.dumps(obj, indent=2, allow_nan=False, default=_default).encode('utf-8'))


def _default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (pd.Timestamp,)):
        return str(o)
    raise TypeError(type(o))


def lf_rewrite(path):
    """Rewrite a CSV (gzip or not) or a text file written by a published code path with the host's line
    ending, with LF (and, for gzip, no write time); returns the new SHA-256 of the bytes."""
    import hashlib
    path = Path(path)
    if path.name.endswith('.csv.gz'):
        write_csv(pd.read_csv(path, dtype=str, keep_default_na=False), path)
    else:
        path.write_bytes(path.read_bytes().replace(bytes([13, 10]), bytes([10])))
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_csv(frame, path):
    """A gzip CSV written with LF and without a write time in its header, so reruns are byte-stable."""
    import gzip
    import io
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    text = frame.to_csv(index=False, lineterminator='\n')
    buf = io.BytesIO()
    with gzip.GzipFile(filename='', mode='wb', fileobj=buf, mtime=0) as gz:
        gz.write(text.encode('utf-8'))
    path.write_bytes(buf.getvalue())
