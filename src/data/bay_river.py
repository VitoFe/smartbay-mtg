"""Causal Magra river-level context for the known-sensor forecasting task.

ARPAL gauging station AMEGLIA FOCE MAGRA, hourly, 2021-10 to 2026-06. The files are named
`portata` but the recorded parameter is 'PORTATA - LIVELLO MEDIO DEL TORRENTE (m)', a stage height
in metres rather than a volumetric flow, so it is handled and described as a level.

Each monthly export carries a preamble of station metadata before a header row beginning
'Inizio rilevazione', then rows of (interval start, interval end, value, dataset, validity). The
interval end is the causal timestamp: a row labelled 03:00 summarises the hour that ended at 03:00
and is therefore usable at an 03:00 issuance.

Three features, all strictly from observations before the issuance timestamp:
    level        6 h rolling mean, matching the convention used for the meteorological channels
    change_24h   level minus level 24 h earlier, a rising-limb indicator
    max_72h      maximum hourly level in the 72 h before issuance, as event memory
"""
import csv
import glob

import numpy as np
import pandas as pd

from src.data.bay_profile_context import ROOT

SOURCE = 'data/external/Rivers/arpa_magra_ameglia_portata_*.csv'
COLUMNS = ['river_level', 'river_change_24h', 'river_max_72h']


def _read():
    records = []
    for path in sorted(glob.glob(str(ROOT / SOURCE))):
        with open(path, encoding='latin-1', newline='') as stream:
            started = False
            for row in csv.reader(stream):
                if not row:
                    continue
                if row[0].strip() == 'Inizio rilevazione':
                    started = True
                    continue
                if started and len(row) >= 3:
                    records.append((row[1].strip(), row[2].strip().replace(',', '.')))
    frame = pd.DataFrame(records, columns=['end', 'value'])
    frame['t'] = pd.to_datetime(frame.end, format='%d/%m/%Y %H:%M', errors='coerce', utc=True)
    frame['v'] = pd.to_numeric(frame.value, errors='coerce')
    series = frame.dropna(subset=['t', 'v']).set_index('t').v.sort_index()
    return series[~series.index.duplicated(keep='last')]


def river(times, cutoff='2026-01-01'):
    """Causal river features on `times`. 2026 is excluded by default, as the loaders elsewhere do."""
    series = _read()
    if cutoff is not None:
        series = series[series.index < pd.Timestamp(cutoff, tz='UTC')]
    hourly = series.resample('1h').mean()
    features = pd.DataFrame({
        'river_level': hourly.rolling(6, min_periods=4).mean(),
        'river_change_24h': hourly.rolling(6, min_periods=4).mean() - hourly.shift(24).rolling(6, min_periods=4).mean(),
        'river_max_72h': hourly.rolling(72, min_periods=24).max()})
    return features.reindex(times)


def high_flow(times, quantile=0.90, cutoff='2026-01-01'):
    """Boolean mask of issuances whose 72 h maximum level exceeds its own quantile.

    Declared in addendum 6 before any contrast was computed, so the restricted analysis is
    pre-registered rather than a subgroup chosen after seeing the result.
    """
    column = river(times, cutoff)['river_max_72h']
    threshold = float(np.nanquantile(column.to_numpy(), quantile))
    return (column > threshold).to_numpy(), threshold
