"""Causal ISPRA La Spezia station context: sea level and external water temperature.

Ten-minute multi-parameter record, 2021-05 to 2026-06, semicolon separated with comma decimals.
Two columns are used, both at about 98.8 per cent hourly coverage of the development window:

    LIVELLO_IDROMETRICO   sea level, the forcing behind tidal exchange through the breakwater
    TEMPERATURA_ACQUA     water temperature at the station, an in-situ external reference

The repository holds no coordinates for this station and nothing establishing its position relative
to the breakwater, so it is treated as an external reference rather than as a verified outer-gulf
boundary. The precipitation column is deliberately not used: it covers only 27.8 per cent of the
development window and its distribution (median 13, 99th percentile and maximum both 42) is that of
a saturating or resetting counter rather than an hourly rate.

All features use only observations strictly before the issuance timestamp. The station stamps each
reading in UTC (no daylight-saving gap or repeat in the record), and the hourly mean labelled q holds
the readings stamped in [q - 1 h, q), labelled at the interval end as `bay_observations.read_export`
and `bay_forcing.weather` label theirs, so a feature reindexed at issuance q reads nothing stamped at
or after q.

V2_ISPRA_BINNING=start labels each hourly mean at its start instead, so that the mean at q holds the
readings stamped q to q + 50 min, which follow issuance. It exists only to reproduce the addendum-7
runs, which were made with it (config/bay_reserved_protocol_v2_addendum59.json, part B).
"""
import os

import numpy as np
import pandas as pd

from src.data.bay_profile_context import ROOT

SOURCE = 'data/external/Ispra/ISPRA_La_Spezia_2026_06_10.csv'
LEVEL_COLUMNS = ['sea_level', 'sea_level_change_6h', 'sea_level_range_24h']
BOUNDARY_COLUMNS = ['ext_water_temperature', 'ext_water_change_24h']
BINNINGS = {'end': {'closed': 'left', 'label': 'right'},     # [q - 1 h, q) -> q: causal at issuance q
            'start': {'closed': 'left', 'label': 'left'}}    # [q, q + 1 h) -> q: reads past issuance
_CACHE = {}


def binning():
    """The hourly binning in force: 'end' unless V2_ISPRA_BINNING says otherwise."""
    name = os.environ.get('V2_ISPRA_BINNING', 'end')
    if name not in BINNINGS:
        raise ValueError(f'V2_ISPRA_BINNING must be one of {sorted(BINNINGS)}, not {name!r}')
    return name


def _hourly(cutoff='2026-01-01'):
    key = (str(cutoff), binning())
    if key in _CACHE:
        return _CACHE[key]
    frame = pd.read_csv(ROOT / SOURCE, sep=';', decimal=',', encoding='utf-8', low_memory=False)
    stamp = pd.to_datetime(frame.DATA.astype(str) + ' ' + frame.ORA.astype(str), errors='coerce', utc=True)
    frame = frame.assign(t=stamp).dropna(subset=['t']).set_index('t').sort_index()
    if cutoff is not None:
        frame = frame[frame.index < pd.Timestamp(cutoff, tz='UTC')]
    rule = BINNINGS[key[1]]
    out = pd.DataFrame({
        'level': pd.to_numeric(frame['LIVELLO_IDROMETRICO'], errors='coerce').resample('1h', **rule).mean(),
        'water': pd.to_numeric(frame['TEMPERATURA_ACQUA'], errors='coerce').resample('1h', **rule).mean()})
    _CACHE[key] = out
    return out


def sea_level(times, cutoff='2026-01-01'):
    """Level, its 6 h change, and its 24 h range as a tidal-amplitude proxy."""
    h = _hourly(cutoff)['level']
    smooth = h.rolling(6, min_periods=3).mean()
    features = pd.DataFrame({
        'sea_level': smooth,
        'sea_level_change_6h': smooth - smooth.shift(6),
        'sea_level_range_24h': h.rolling(24, min_periods=12).max() - h.rolling(24, min_periods=12).min()})
    return features.reindex(times)


def external_boundary(times, anchor=None, cutoff='2026-01-01'):
    """External water temperature and its 24 h change.

    `anchor`, when given, is the target probe's own last observed value in physical units; the
    difference between the external station and that anchor is then supplied as a third feature,
    which is the quantity a boundary condition is actually informative about.
    """
    h = _hourly(cutoff)['water']
    smooth = h.rolling(6, min_periods=3).mean()
    features = pd.DataFrame({
        'ext_water_temperature': smooth,
        'ext_water_change_24h': smooth - smooth.shift(24)}).reindex(times)
    if anchor is None:
        return features
    gap = features['ext_water_temperature'].to_numpy()[:, None] - anchor
    return features, gap


def coverage(times, cutoff='2026-01-01'):
    h = _hourly(cutoff).reindex(times)
    return {c: float(h[c].notna().mean()) for c in h.columns}
