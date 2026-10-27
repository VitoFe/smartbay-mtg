"""Addendum-24: does the biofouling detector distinguish a manually cleaned record from a wiped one?

Section 3.8 states that the detector's thresholds were set by inspection and never validated
against documented servicing dates. Those dates are not in this repository: the Gen. 1 oxygen
exports carry a header and then an uninterrupted data block, with no maintenance column, no
annotation rows and no separate log anywhere in data/.

The record supplies a substitute. The same two sites carry both deployment generations under
opposite cleaning regimes: Aanderaa optodes cleaned by hand from October 2021 to August 2023, and
EXO2 sondes whose wiper clears the optical surface daily from July 2024. If the statistic
integrates biofilm accumulation, it has far more to integrate on the first.

The detector is the published one, recovered unchanged from the retired qc.py: the deficit against
Garcia-Gordon equilibrium solubility, smoothed over 48 h, accumulated by a one-sided CUSUM with
slack 3.0 and threshold 45.0 umol/L. As published it never resets, so `alert_triggered` is sticky
and alerts cannot be counted; a reset-on-alert variant is computed alongside and labelled, because
the declared primary statistic needs it.

Salinity is absent from the Gen. 1 export, so both generations use the constant 38 PSU the thesis
already defends, keeping the solubility input identical across the contrast.

Declared in config/bay_reserved_protocol_v2_addendum24.json.
Usage: python -m src.experiments.v2_biofouling
"""
import json
import re

import numpy as np
import pandas as pd

from src.data.bay_profile_context import ROOT
from src.data.thermodynamics import o2sol_garcia_gordon_umol_per_l as solubility

OUT = ROOT / 'reports/bay_twin/runs/v2_biofouling.json'
RAW = ROOT / 'data/raw'
K_SLACK, H_DECISION, WINDOW = 3.0, 45.0, 48      # exactly as published
SALINITY_PSU = 38.0

#: site -> (Gen.1 export, Gen.2 export). Same location and nominal depth on both sides.
PAIRS = {
    'Baia S. Teresa 1 m': ("TBaia_01m-Ossigeno(calibrato)", "TBaiaP_01m-CTD(calibrato)"),
    'Mitili Est 5 m': ("TMitiliE_05m-Ossigeno(calibrato)", "TMitiliE_05m-CTD(calibrato)"),
}


def read_export(stem):
    """Hourly oxygen and temperature from a raw export, whichever generation wrote it."""
    path = RAW / f'{stem}.txt'
    header, rows = None, []
    with path.open(encoding='latin-1') as fh:
        for line in fh:
            if line.startswith('#'):
                if 'Data' in line and 'Ora' in line:
                    header = line.lstrip('#').split()
                continue
            rows.append(line.split())
    assert header, f'no column header in {stem}'
    frame = pd.DataFrame(rows).iloc[:, :len(header)]
    frame.columns = header[:frame.shape[1]]
    time = pd.to_datetime(frame['Data'] + ' ' + frame['Ora(UTC)'], dayfirst=True, errors='coerce',
                          utc=True)

    def column(pattern):
        for c in frame.columns:
            if re.search(pattern, c, re.I):
                return pd.to_numeric(frame[c], errors='coerce')
        return pd.Series(np.nan, index=frame.index)

    # .to_numpy() matters: passing Series positionally indexed 0..N together with index=time
    # makes pandas reindex them onto the timestamps, which silently yields an all-NaN frame.
    out = pd.DataFrame({'oxygen': column(r"Ossigeno\(umol").to_numpy(),
                        'temperature': column(r'Temperatura').to_numpy()},
                       index=time).dropna(how='all')
    # The exports carry -9999 sentinels. src/data/bay_observations.py screens them with these
    # gates; the same ones are applied here so this analysis sees the record the thesis sees.
    out['temperature'] = out.temperature.where(out.temperature.between(-2, 40))
    out['oxygen'] = out.oxygen.where(out.oxygen.between(0, 600))
    out = out[out.index.notna()].sort_index()
    out = out[~out.index.duplicated(keep='first')]
    # A uniform hourly grid, so a 48-hour window spans 48 hours rather than 48 observations.
    grid = pd.date_range(out.index[0].ceil('h'), out.index[-1].floor('h'), freq='h', tz='UTC')
    return out.resample('1h').mean().reindex(grid)


def deficit(frame):
    """Smoothed departure from equilibrium solubility, the quantity the CUSUM integrates."""
    sol = solubility(frame.temperature.to_numpy(), np.full(len(frame), SALINITY_PSU))
    sol = pd.Series(sol, index=frame.index).rolling(WINDOW, min_periods=12).median().bfill().ffill()
    obs = frame.oxygen.rolling(WINDOW, min_periods=12).median().bfill().ffill()
    return (sol - obs)


def cusum(values, reset):
    """One-sided CUSUM. `reset` returns the statistic to zero on each alert, which the published
    form does not do; without it the statistic is monotone and alerts cannot be counted."""
    s, series, alerts = 0.0, [], []
    for i, r in enumerate(values):
        if np.isfinite(r):
            s = max(0.0, s + (r - K_SLACK))
            if s >= H_DECISION:
                alerts.append(i)
                if reset:
                    s = 0.0
        series.append(s)
    return np.asarray(series), alerts


def assess(frame, label):
    d = deficit(frame)
    published, _ = cusum(d.to_numpy(), reset=False)
    _, alerts = cusum(d.to_numpy(), reset=True)
    hours = int(frame.oxygen.notna().sum())
    years = hours / (24 * 365.25)
    # A one-sided CUSUM only stays near zero if its slack exceeds the mean of what it integrates.
    # If the mean deficit is above k, the statistic has positive drift by construction and reaching
    # any threshold is a matter of time rather than of an episode, so the implied spacing is
    # reported next to the observed one.
    drift = float(d.mean()) - K_SLACK
    implied_hours = (H_DECISION / drift) if drift > 0 else None
    return {
        'label': label,
        'first': str(frame.index[0].date()), 'last': str(frame.index[-1].date()),
        'observed_hours': hours, 'probe_years': round(years, 2),
        'mean_deficit_umol_l': round(float(d.mean()), 3),
        'max_cusum_published': round(float(np.nanmax(published)), 1),
        'ever_triggers_published': bool(np.nanmax(published) >= H_DECISION),
        'fraction_hours_above_threshold': round(float((published >= H_DECISION).mean()), 4),
        'alerts_reset_variant': len(alerts),
        'alerts_per_probe_year': round(len(alerts) / years, 2) if years > 0 else None,
        'slack_k': K_SLACK,
        'drift_per_hour': round(drift, 3),
        'implied_hours_to_alert': round(implied_hours, 1) if implied_hours else None,
        'observed_hours_between_alerts': round(hours / len(alerts), 1) if alerts else None,
    }


def shared_months(a, b):
    """Fouling is temperature-driven, so the contrast is repeated on months both records cover."""
    months = sorted(set(a.index.month) & set(b.index.month))
    return a[a.index.month.isin(months)], b[b.index.month.isin(months)], months


def main():
    report = {'declared_in': 'config/bay_reserved_protocol_v2_addendum24.json',
              'detector': {'k_slack': K_SLACK, 'h_decision': H_DECISION,
                           'smooth_window_hours': WINDOW, 'salinity_psu': SALINITY_PSU,
                           'source': 'recovered unchanged from the retired src/data/qc.py'},
              'cleaning_dates': 'NOT PRESENT in this repository; see the declaration',
              'sites': {}}

    for site, (gen1_stem, gen2_stem) in PAIRS.items():
        g1, g2 = read_export(gen1_stem), read_export(gen2_stem)
        entry = {'gen1_file': gen1_stem, 'gen2_file': gen2_stem,
                 'full_record': {'gen1': assess(g1, 'Gen.1 manual'),
                                 'gen2': assess(g2, 'Gen.2 wiper')}}
        r1, r2, months = shared_months(g1, g2)
        entry['shared_months'] = {'months': months,
                                  'gen1': assess(r1, 'Gen.1 manual, shared months'),
                                  'gen2': assess(r2, 'Gen.2 wiper, shared months')}
        report['sites'][site] = entry

    OUT.write_text(json.dumps(report, indent=2))
    for site, e in report['sites'].items():
        print(f'== {site}')
        for scope in ('full_record', 'shared_months'):
            print(f'   -- {scope}' + (f" (months {e[scope]['months']})" if scope == 'shared_months' else ''))
            for gen in ('gen1', 'gen2'):
                a = e[scope][gen]
                print(f"      {a['label']:28s} {a['first']}..{a['last']} "
                      f"{a['probe_years']:5.2f} yr  deficit {a['mean_deficit_umol_l']:8.3f}  "
                      f"maxCUSUM {a['max_cusum_published']:10.1f}  "
                      f"above {a['fraction_hours_above_threshold']:.3f}  "
                      f"alerts/yr {a['alerts_per_probe_year']}")
                print(f"        drift/h {a['drift_per_hour']:+.3f} (k={a['slack_k']})  "
                      f"implied {a['implied_hours_to_alert']} h between alerts, "
                      f"observed {a['observed_hours_between_alerts']} h")
    print('\nwrote', OUT)


if __name__ == '__main__':
    main()
