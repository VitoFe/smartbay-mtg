"""Arno stage at Bocca d'Arno, and lagged river features for the addendum-26 sweep.

Two catchments reach this coast on different scales. The Magra gauge at Ameglia is at the mouth of
the river that drains into this gulf, hourly, and is what addendum 6 used. Bocca d'Arno is the
mouth of the Arno at Pisa, 43.681 N 10.280 E, 54-55 km south-east of the probes and a different catchment
entirely; it is recorded daily since 1998.

The physical argument for using it, from the supervisor, is conditional: the Arno signal is too
distant to mean anything on its own, and reaches this bay only when the local river is also
running, since otherwise the currents arriving are not river-driven. That makes it a gated feature
rather than an additional one, and the gate is declared in addendum 26.

Causality for a daily series. At an issuance timestamp only the last *completed* day is knowable,
so the daily value is held forward from the day after it was recorded. Interpolating between
daily values would use the following day's reading, which at issuance has not happened.
"""
import csv

import numpy as np
import pandas as pd

from src.data.bay_profile_context import ROOT
from src.data.bay_river import _read as _read_ameglia

BOCCA = 'data/external/Rivers/liv_TOS_Arno_Bocca.csv'
LAG_COLUMNS = ['level', 'change_24h', 'max_72h']


def bocca(cutoff='2026-01-01'):
    """Daily Arno stage at Bocca d'Arno, on an hourly grid, held forward causally."""
    rows = []
    with open(ROOT / BOCCA, encoding='latin-1', newline='') as stream:
        for row in csv.reader(stream, delimiter=';'):
            if len(row) >= 2 and row[0].count('/') == 2:
                rows.append((row[0].strip(), row[1].strip().replace(',', '.')))
    frame = pd.DataFrame(rows, columns=['d', 'v'])
    frame['t'] = pd.to_datetime(frame.d, format='%d/%m/%Y', errors='coerce', utc=True)
    frame['x'] = pd.to_numeric(frame.v, errors='coerce')
    series = frame.dropna(subset=['t', 'x']).set_index('t').x.sort_index()
    series = series[~series.index.duplicated(keep='last')]
    if cutoff is not None:
        series = series[series.index < pd.Timestamp(cutoff, tz='UTC')]
    # Shift a full day before holding forward: the value dated D is not available until D+1.
    return series.shift(1, freq='D').resample('1h').ffill()


def features_at_lag(series, times, lag_hours):
    """The addendum-6 feature triple, computed on a series shifted back by `lag_hours`.

    A lag is applied to the *source* rather than to the features, so every window still ends at or
    before the issuance and causality is preserved whatever the lag.
    """
    hourly = series.resample('1h').mean().shift(int(lag_hours))
    smooth = hourly.rolling(6, min_periods=4).mean()
    out = pd.DataFrame({
        'level': smooth,
        'change_24h': smooth - hourly.shift(24).rolling(6, min_periods=4).mean(),
        'max_72h': hourly.rolling(72, min_periods=24).max()})
    return out.reindex(times)


def ameglia(cutoff='2026-01-01'):
    series = _read_ameglia()
    if cutoff is not None:
        series = series[series.index < pd.Timestamp(cutoff, tz='UTC')]
    return series


def gate_from_ameglia(times, quantile, cutoff='2026-01-01'):
    """Boolean mask: is the local river high at issuance?

    The Ameglia gauge records mean level at a river mouth, which mixes catchment discharge with
    tidal stage and cannot separate them. The gate is therefore 'the local water is high', which is
    what the physical argument requires, and not specifically 'the river is in flood'. A tidal gate
    built from the ISPRA sea-level record would test the narrower reading and is not run here.
    """
    level = features_at_lag(ameglia(cutoff), times, 0)['max_72h']
    threshold = float(np.nanquantile(level.to_numpy(), quantile))
    return (level > threshold).to_numpy(), threshold
