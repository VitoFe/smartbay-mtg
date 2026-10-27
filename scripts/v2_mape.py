"""Mean absolute percentage error (MAPE) beside the headline mean absolute errors of the deck.

For each headline estimate the deck prints as an MAE, the MAPE is computed from the same stored
per-row predictions under the same averaging rule, with |prediction - truth| / |truth| x 100 in place
of |prediction - truth|. Before a MAPE is written, the MAE recomputed from the same rows and the same
rule must equal the stored one (gate G1), so the percentage describes exactly the rows and the
weighting of the printed error. Descriptive only: nothing is selected or fitted, and no interval is
attached.

  spatial   held-station error, casts within (fold, station), stations within fold, folds
            (src/experiments/v2_field.nested_mae): Field-57 per fold, the released model (temperature
            from runs/v2_perfold, oxygen from its mg/l rerun); Field-2026 temperature for the cast prior
            fitted inside each inner split, the released Copernicus configuration and no prior
  forecast  Reserved-2026 at six hours, sensors within quarter, quarters (the battery's rule):
            the released model, persistence, the month-by-hour climatological increment
  silent    Bridge-2026, sensors within fold, folds, by lead (src/experiments/v2_bridge.macro): the
            anchored continuation and the last value for temperature, the hour-of-day increment and
            the last value for oxygen
  gaps      Gap-2026, 12 to 72 h, the mean over masked episodes (scripts/v2_gap_variants.macro):
            the Gaussian process and linear interpolation for temperature, bidirectional Chronos-Bolt
            and linear interpolation for oxygen

Oxygen needs no unit: a percentage of the sonde record is the same in umol/kg and in mg/l. Temperature
is in degrees Celsius, whose zero is a convention, so its percentage depends on that choice; each
temperature entry also carries its MAPE in kelvin, and `temperature_in_kelvin` gives the range of both
and of their ratio. The deck says so.

Usage: python -m scripts.v2_mape   -> reports/bay_twin/runs/v2_mape.json
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / 'reports/bay_twin/runs'
OUT = RUNS / 'v2_mape.json'
MGL = RUNS / 'v2_oxygen_mgl'
TOL = 1e-9


def ape(truth, pred, kelvin=False):
    """Absolute percentage error; with `kelvin` the temperature is referred to absolute zero."""
    return np.abs(pred - truth) / np.abs(truth + (273.15 if kelvin else 0.0)) * 100.0


def check(name, got, want, tol=TOL):
    if want is None or abs(got - want) > tol:
        raise SystemExit(f'gate G1 failed for {name}: recomputed MAE {got!r}, stored {want!r}')
    return {'mae_recomputed': got, 'mae_stored': want}


# ------------------------------------------------------------------ spatial

def nested(frame, column):
    """v2_field.nested_mae's rule applied to any per-row column."""
    return float(frame.groupby(['fold', 'station', 'cast_id'])[column].mean()
                 .groupby(['fold', 'station']).mean().groupby('fold').mean().mean())


def field57():
    out = {}
    for target, path, source in (
            ('temp_c', RUNS / 'v2_perfold/field/perfold/A1_both_direct', 'runs/v2_perfold/field/perfold/A1_both_direct'),
            ('do_umol_l', MGL / 'C1/v2_perfold/field/perfold/A1_both_direct',
             'runs/v2_oxygen_mgl/C1/v2_perfold/field/perfold/A1_both_direct')):
        d = pd.read_csv(path / 'selected_predictions.csv.gz')
        d = d.dropna(subset=[target, f'pred_{target}'])
        d = d.assign(e=(d[target] - d[f'pred_{target}']).abs(), p=ape(d[target], d[f'pred_{target}']),
                     k=ape(d[target], d[f'pred_{target}'], kelvin=True))
        want = json.loads((path / 'report.json').read_text())['nested_mae'][target]
        out[target] = {'source': source, **check(f'Field-57 {target}', nested(d, 'e'), want),
                       'mape_percent': nested(d, 'p'), 'rows': int(len(d))}
        if target == 'temp_c':
            out[target]['mape_kelvin_percent'] = nested(d, 'k')
    return out


def field2026():
    from src.experiments.v2_field_reserved2026 import macro as macro26
    inner = pd.read_csv(RUNS / 'v2_field_reserved2026/innerprior/reserved2026_held_station_predictions.csv.gz')
    cop = pd.read_csv(MGL / 'C1/v2_field_reserved2026_copernicus/reserved2026_held_station_predictions.csv.gz')
    stored_inner = json.loads((RUNS / 'v2_field_reserved2026/innerprior/report.json').read_text())['held_station']['nested_mae']
    stored_cop = json.loads((RUNS / 'v2_field_reserved2026_copernicus/report.json').read_text())['held_station']['nested_mae']
    out = {}
    # The record-unit Copernicus predictions for 2026 were not kept; the mg/l rerun's are used. Its moored
    # oxygen inputs are converted too, so its temperature predictions are not bit-identical: on every arm of
    # that report its temperature MAE is within 8e-5 degC of the published one, hence the 1e-4 tolerance.
    for label, frame, arm, want, source, tol in (
            ('cast prior, inner split', inner, 'A7o', stored_inner['A7o']['temp_c'],
             'runs/v2_field_reserved2026/innerprior/reserved2026_held_station_predictions.csv.gz', 0.0),
            ('Copernicus, released', cop, 'A1_both', stored_cop['A1_both']['temp_c'],
             'runs/v2_oxygen_mgl/C1/v2_field_reserved2026_copernicus/reserved2026_held_station_predictions.csv.gz',
             1e-4),
            ('no prior', inner, 'A2_none', stored_inner['A2_none']['temp_c'],
             'runs/v2_field_reserved2026/innerprior/reserved2026_held_station_predictions.csv.gz', 0.0)):
        d = frame.dropna(subset=['temp_c', f'{arm}|temp_c'])
        d = d.assign(e=(d['temp_c'] - d[f'{arm}|temp_c']).abs(), p=ape(d['temp_c'], d[f'{arm}|temp_c']),
                     k=ape(d['temp_c'], d[f'{arm}|temp_c'], kelvin=True))
        cast = d.groupby(['fold', 'station', 'cast_id'])[['e', 'p', 'k']].mean().reset_index()
        mae = macro26(cast)
        mape = macro26(cast.drop(columns='e').rename(columns={'p': 'e'}))
        mape_k = macro26(cast.drop(columns='e').rename(columns={'k': 'e'}))
        # the reports round to four decimals
        got = mae if tol else round(mae, 4)
        out[label] = {'source': source, **check(f'Field-2026 {label}', got, want, tol or TOL),
                      'gate_tolerance_degC': tol or TOL, 'mape_percent': mape, 'mape_kelvin_percent': mape_k,
                      'casts': int(len(cast))}
    return out


# ------------------------------------------------------------------ forecast

def forecast2026():
    from scripts.v2_chronos_lora_contrasts import battery_frame
    d = battery_frame()
    bat = json.loads((RUNS / 'v2_reserved_reading_v2/battery_full_v2.json').read_text())
    out = {}
    for variable in ('temperature', 'oxygen'):
        s0 = d[d.variable == variable]
        stored = bat[variable]
        out[variable] = {}
        for arm, want in (('MODEL_v2', stored['MODEL_v2']),
                          ('B0_persistence', stored['methods']['B0_persistence']['mae']),
                          ('B1_clim_increment', stored['methods']['B1_clim_increment']['mae'])):
            s = s0.dropna(subset=[arm, 'truth'])
            s = s.assign(e=(s[arm] - s.truth).abs(), p=ape(s.truth, s[arm]), k=ape(s.truth, s[arm], kelvin=True))
            rule = lambda col: float(s.groupby(['quarter', 'sensor'])[col].mean().groupby('quarter').mean().mean())
            out[variable][arm] = {**check(f'Reserved-2026 {variable} {arm}', rule('e'), want),
                                  'mape_percent': rule('p'), 'endpoints': int(len(s))}
            if variable == 'temperature':
                out[variable][arm]['mape_kelvin_percent'] = rule('k')
    out['source'] = 'the battery frame of runs/v2_reserved_reading_v2 (scripts/v2_chronos_lora_contrasts.battery_frame)'
    return out


# ------------------------------------------------------------------ silent probe

def bridge2026():
    from src.experiments.v2_bridge import macro
    br = json.loads((RUNS / 'v2_bridge/report_a69.json').read_text())
    b72 = json.loads((RUNS / 'v2_bridge72/report.json').read_text())
    arms = {'temperature': ('anchored_direct', 'persistence'), 'oxygen': ('clim_hour', 'persistence')}
    out = {}
    for variable, (best, last) in arms.items():
        out[variable] = {}
        frames = [(RUNS / f'v2_bridge/bridge_{variable}.csv.gz', br, (6, 24, 48))]
        if variable == 'temperature':
            frames.append((RUNS / f'v2_bridge72/bridge72_{variable}.csv.gz', b72, (72,)))
        for path, report, leads in frames:
            d = pd.read_csv(path)
            d = d[d.cohort == 'Bridge-2026']
            for lead in leads:
                f = d[d.lead == lead]
                entry = {}
                for arm in (best, last):
                    want = report[variable]['Bridge-2026']['by_lead'][str(lead)]['mae'][arm]
                    g = f.dropna(subset=[arm]).assign(pct=lambda x: ape(x.truth, x[arm]),
                                                      pk=lambda x: ape(x.truth, x[arm], kelvin=True))
                    rule = lambda col: float(g.groupby(['fold', 'sensor'])[col].mean().groupby('fold').mean().mean())
                    # the bridge reports store six decimals
                    entry[arm] = {**check(f'Bridge-2026 {variable} {lead} h {arm}', macro(f, arm), want, 5e-7),
                                  'mape_percent': rule('pct'),
                                  'source': path.relative_to(ROOT / 'reports/bay_twin').as_posix()}
                    if variable == 'temperature':
                        entry[arm]['mape_kelvin_percent'] = rule('pk')
                out[variable][str(lead)] = entry
    return out


# ------------------------------------------------------------------ gaps

def gaps2026():
    va = json.loads((RUNS / 'v2_gap_variants.json').read_text())['results']['Gap-2026']
    out = {}
    for variable, arms in (('temperature', ('gpr', 'linear')), ('oxygen', ('chronos_bidir', 'linear'))):
        d = pd.read_csv(RUNS / f'v2_gap_variants/Gap-2026__{variable}__12-72.csv.gz')
        out[variable] = {}
        for arm in arms:
            f = d.dropna(subset=[arm, 'truth'])
            f = f.assign(e=(f.truth - f[arm]).abs(), p=ape(f.truth, f[arm]), k=ape(f.truth, f[arm], kelvin=True))
            mae = round(float(f.groupby('episode').e.mean().mean()), 4)   # the producer rounds to four decimals
            out[variable][arm] = {**check(f'Gap-2026 {variable} {arm}', mae, va[variable]['12-72'][arm]['mae']),
                                  'mape_percent': float(f.groupby('episode').p.mean().mean()),
                                  'episodes': int(f.episode.nunique()),
                                  'source': f'runs/v2_gap_variants/Gap-2026__{variable}__12-72.csv.gz'}
            if variable == 'temperature':
                out[variable][arm]['mape_kelvin_percent'] = float(f.groupby('episode').k.mean().mean())
    return out


def kelvin_summary(report):
    """The range of the temperature MAPEs in degrees Celsius and in kelvin, and of their ratio."""
    pairs = []

    def walk(x):
        if isinstance(x, dict):
            if 'mape_kelvin_percent' in x:
                pairs.append((x['mape_percent'], x['mape_kelvin_percent']))
            for v in x.values():
                walk(v)
    walk(report)
    return {'entries': len(pairs),
            'celsius_percent': [min(p for p, _ in pairs), max(p for p, _ in pairs)],
            'kelvin_percent': [min(k for _, k in pairs), max(k for _, k in pairs)],
            'ratio_celsius_over_kelvin': [min(p / k for p, k in pairs), max(p / k for p, k in pairs)]}


def main():
    report = {
        'what': 'MAPE (percent) beside the headline MAEs of the deck, from the same rows under the same averaging '
                'rule; gate G1 recomputes each MAE from those rows and requires the stored value',
        'definition': 'mean of |prediction - truth| / |truth| x 100, averaged as the MAE it accompanies',
        'temperature_scale': 'degrees Celsius; a percentage of a Celsius reading depends on where the scale puts zero',
        'oxygen_unit': 'none needed: the ratio is the same in umol/kg and mg/l',
        'spatial': {'Field-57 per fold, released model': field57(), 'Field-2026 temperature': field2026()},
        'forecast': {'Reserved-2026, 6 h': forecast2026()},
        'silent_probe': {'Bridge-2026': bridge2026()},
        'gaps': {'Gap-2026, 12 to 72 h': gaps2026()},
        'gate_G1': 'passed',
    }
    report['temperature_in_kelvin'] = kelvin_summary(report)
    OUT.write_bytes((json.dumps(report, indent=2) + '\n').encode('utf-8'))
    print(json.dumps(report, indent=1)[:6000])


if __name__ == '__main__':
    main()
