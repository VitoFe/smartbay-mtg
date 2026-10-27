"""Addendum-12 part A: is the physical background biased, and is the bias common across stations?

Section 5.3.2 observes that a penalty re-selected for temperature alone runs to 9.9e7 and infers
that the estimator's whole temperature skill is its physical guidance. Equation 4.1 fits
Ridge(fit_intercept=False) on standardised features, so a constant offset in (y - background) is
not representable and the penalty running away is consistent with two different situations: the
features carry nothing, or the background carries a bias the model class cannot remove.

This measures which. The per-station mean signed residual is a diagnostic only. Field-57 holds out
whole stations, so a per-station intercept does not exist for the station being scored and could
never be a model term; what decides whether a GLOBAL intercept could help is whether these
per-station biases share a sign.

Declared in config/bay_reserved_protocol_v2_addendum12.json.
Usage: python scripts/v2_background_bias.py
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'reports/bay_twin/runs/v2_background_bias.json'
TARGETS = ['temp_c', 'salinity_psu', 'do_umol_l']
UNITS = {'temp_c': 'degC', 'salinity_psu': 'PSU', 'do_umol_l': 'umol/L'}


def main():
    import sys
    sys.path.insert(0, str(ROOT))
    from src.experiments.bay_profile_field import prepare
    from src.experiments.bay_field_export import EXTRAS
    from src.data.bay_profile_context import load_profiles

    cfg = json.loads((ROOT / 'config/bay_profile_field.json').read_text())
    bundles, contracts, _ = prepare('held_site', cfg, False)
    data, _ = load_profiles()
    context = data[['cast_id', 'depth_m']].merge(
        pd.read_csv(ROOT / 'reports/bay_twin/profile_boundary_context.csv.gz'),
        on=['cast_id', 'depth_m'], validate='one_to_one')
    extras = context[EXTRAS].to_numpy(dtype=float)
    stations = data.station.to_numpy()

    report = {'declared_in': 'config/bay_reserved_protocol_v2_addendum12.json',
              'quantity': 'mean signed (observation - background) on training rows, per station',
              'note': 'diagnostic only; a per-station intercept cannot exist for a held-out station',
              'by_fold': {}}

    pooled = {t: [] for t in TARGETS}
    for k, b in enumerate(bundles):
        # The released A1 background: base plus the Copernicus depth-dependent offset.
        base = b['base'].astype(float).copy() + extras[:, 3:6]
        base[:, 0] = np.clip(base[:, 0], -2, 40)
        base[:, 1] = np.clip(base[:, 1], 0, 42)
        y = b['y'].astype(float)
        tr = b['tr']
        residual = pd.DataFrame(y[tr] - base[tr], columns=TARGETS)
        residual['station'] = stations[tr]
        entry = {'train_rows': int(len(tr)), 'per_station': {}, 'summary': {}}
        for station, g in residual.groupby('station'):
            entry['per_station'][station] = {t: round(float(g[t].mean()), 4) for t in TARGETS}
        for t in TARGETS:
            per = np.array([v[t] for v in entry['per_station'].values()])
            entry['summary'][t] = {
                'pooled_mean_signed': round(float(residual[t].mean()), 4),
                'mean_of_station_means': round(float(per.mean()), 4),
                'sd_of_station_means': round(float(per.std(ddof=1)), 4),
                'stations_positive': int((per > 0).sum()), 'stations_negative': int((per < 0).sum()),
                'min': round(float(per.min()), 4), 'max': round(float(per.max()), 4)}
            pooled[t].append(residual[t].mean())
        report['by_fold'][f'fold{k}'] = entry

        print(f'== fold {k}  train rows {len(tr)}  stations {len(entry["per_station"])}')
        for t in TARGETS:
            s = entry['summary'][t]
            print(f'   {t:14s} pooled {s["pooled_mean_signed"]:+8.4f} {UNITS[t]:7s} | '
                  f'station means {s["min"]:+.4f} to {s["max"]:+.4f}, sd {s["sd_of_station_means"]:.4f}, '
                  f'{s["stations_positive"]}+ / {s["stations_negative"]}-')

    report['across_folds'] = {t: {'pooled_mean_signed': round(float(np.mean(v)), 4)} for t, v in pooled.items()}
    OUT.write_text(json.dumps(report, indent=2))
    print('\nwrote', OUT)


if __name__ == '__main__':
    main()
