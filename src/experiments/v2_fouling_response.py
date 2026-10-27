"""Addendum-27: fouling as a diffusion barrier, tested as a dose-response on time since cleaning.

Addendum 25 established that the deficit against equilibrium solubility carries no cleaning
signature. That result is about one observable, and the obstacle it identified is a confound: the
deficit cannot separate a fouled optical window from a genuinely undersaturated bay, and this bay
is undersaturated by 5.8 to 9.9 umol/l on average. A cleaning log does not repair that.

Two things here are new rather than a retuning of the same test.

The observable. Biofilm on an optode is a diffusion barrier, and a barrier attenuates the amplitude
of a real change and delays the response to it. Those are properties of the dynamics, not of the
level, so the confound that defeated the deficit does not apply. Each export carries temperature
from the same instrument, and a thermistor does not sit behind the optical window, so temperature
gives the forcing as it was and oxygen gives the forcing as the window saw it. Everything below is
an oxygen-to-temperature ratio or phase difference, which removes the weather without an external
record.

The design. The logbook gives, for every hour of the first-generation record, the time since the
last cleaning. That converts a two-sample comparison into a dose-response over 75 intervals at
node1 and 16 at node5, and a monotone seasonal trend does not reset on a maintenance calendar.

Four tests, each against a placebo:

    A  the step in the RAW unsmoothed series at a cleaning, so the smoothing objection to
       addendum 25 is settled by measurement (a trailing median preserves an edge, so it should
       find the same nothing)
    B  daily oxygen-to-temperature amplitude ratio against days since cleaning
    C  daily oxygen-minus-temperature phase, in hours, against days since cleaning
    D  the paired 10-day recovery at each cleaning, the step form of B and C

The strict control is a fake calendar: 400 draws of cleaning times with the same interval
distribution, asking whether any periodic reset produces the same slope. The wiped second
generation is run through it as a negative control, where a slope would mean the method invents
one.

Declared in config/bay_reserved_protocol_v2_addendum27.json.
Usage: python -m src.experiments.v2_fouling_response
"""
import json

import numpy as np
import pandas as pd

from src.data.bay_profile_context import ROOT
from src.data.thermodynamics import o2sol_garcia_gordon_umol_per_l as solubility
from src.experiments.v2_biofouling import PAIRS, SALINITY_PSU, read_export
from src.experiments.v2_logbook import NODES, read_logbook

OUT = ROOT / 'reports/bay_twin/runs/v2_fouling_response.json'
EPOCH_DAYS = 10          # half-width of the paired recovery window, as addendum 25 used
DRAWS = 400              # placebo draws, as addendum 25 used
SEED = 20260922
MIN_HOURS_PER_DAY = 18   # a day with fewer finite hours cannot carry a fitted diurnal harmonic
MAX_DOSE_DAYS = 60       # intervals longer than this are redeployment gaps, not accumulation

#: node -> the wiped Gen. 2 export at the same site, for the negative control.
WIPED = {'node1': PAIRS['Baia S. Teresa 1 m'][1], 'node5': PAIRS['Mitili Est 5 m'][1]}


def harmonic(values):
    """Amplitude and phase of the 24 h harmonic of one day, by least squares.

    Returns (amplitude, phase_hours) with phase measured as the hour of the peak. Fitting rather
    than taking a daily range matters: a range is set by the two most extreme hours and so is
    dominated by noise, while the fitted amplitude uses every hour of the day.
    """
    y = np.asarray(values, float)
    ok = np.isfinite(y)
    if ok.sum() < MIN_HOURS_PER_DAY:
        return np.nan, np.nan
    h = np.arange(len(y))[ok]
    w = 2 * np.pi / 24.0
    design = np.column_stack([np.ones(ok.sum()), np.cos(w * h), np.sin(w * h)])
    coef, *_ = np.linalg.lstsq(design, y[ok], rcond=None)
    amplitude = float(np.hypot(coef[1], coef[2]))
    phase = float(np.arctan2(coef[2], coef[1]) / w) % 24.0
    return amplitude, phase


def daily_table(frame):
    """Per-day fitted amplitude and phase for oxygen and temperature, and their contrasts."""
    rows = []
    for day, block in frame.groupby(frame.index.floor('D')):
        # Reindexing onto the full day keeps the harmonic's hour axis aligned with the clock,
        # so a day with missing hours is not silently compressed into a shorter period.
        full = block.reindex(pd.date_range(day, periods=24, freq='h', tz='UTC'))
        ao, po = harmonic(full.oxygen)
        at, pt = harmonic(full.temperature)
        rows.append({'day': day, 'amp_oxygen': ao, 'amp_temperature': at,
                     'phase_oxygen': po, 'phase_temperature': pt})
    out = pd.DataFrame(rows).set_index('day')
    # The ratio is the barrier's attenuation; guard against a near-zero thermal day inflating it.
    out['amp_ratio'] = np.where(out.amp_temperature > 0.02,
                                out.amp_oxygen / out.amp_temperature, np.nan)
    # Wrapped to (-12, 12] so a lag of one hour is not recorded as a lead of twenty-three.
    out['phase_lag'] = (out.phase_oxygen - out.phase_temperature + 12) % 24 - 12
    return out


def dose(days, cleanings):
    """Days since the most recent cleaning, NaN before the first and beyond MAX_DOSE_DAYS."""
    if not len(cleanings):
        return np.full(len(days), np.nan)
    marks = np.sort(np.asarray([c.value for c in cleanings]))
    idx = np.searchsorted(marks, np.asarray([d.value for d in days]), side='right') - 1
    elapsed = np.where(idx >= 0,
                       (np.asarray([d.value for d in days]) - marks[np.clip(idx, 0, None)])
                       / 86400e9, np.nan)
    return np.where(elapsed <= MAX_DOSE_DAYS, elapsed, np.nan)


def slope(table, column, cleanings, monthly):
    """OLS slope of `column` on days since cleaning, optionally with a month fixed effect."""
    y = table[column].to_numpy(float)
    x = dose(table.index, cleanings)
    ok = np.isfinite(y) & np.isfinite(x)
    if ok.sum() < 30:
        return None
    blocks = [np.ones(ok.sum()), x[ok]]
    if monthly:
        months = table.index.month.to_numpy()[ok]
        # One column per month present, first dropped: the intercept carries it.
        for m in sorted(set(months))[1:]:
            blocks.append((months == m).astype(float))
    design = np.column_stack(blocks)
    coef, *_ = np.linalg.lstsq(design, y[ok], rcond=None)
    resid = y[ok] - design @ coef
    dof = ok.sum() - design.shape[1]
    var = float(resid @ resid) / dof * np.linalg.pinv(design.T @ design)[1, 1]
    return {'slope_per_day': round(float(coef[1]), 5),
            'std_error': round(float(np.sqrt(max(var, 0.0))), 5),
            'n_days': int(ok.sum()), 'month_fixed_effect': monthly}


def fake_calendars(cleanings, first, last, rng, draws=DRAWS):
    """Draw cleaning calendars by circularly shifting the real one within the record.

    This is the stricter placebo. Random timestamps ask whether the signal is special at cleanings;
    a fake calendar asks whether any reset on the same rhythm would produce the same slope, which
    is what a seasonal or maintenance-correlated artefact would do.

    The shift has to be circular. Laying the observed gaps end to end from a random start instead
    leaves an uncovered prefix, because the dose is undefined before the first cleaning, so each
    draw is fitted over fewer days, a narrower dose range and a different part of the year than the
    real calendar. That made the null distribution for node1's amplitude slope sit at +1.38 rather
    than at zero and would have been read as an effect. A circular shift preserves every interval
    but one, the number of scorable days and the seasonal coverage, and changes only the phase of
    the calendar against the data.
    """
    if len(cleanings) < 2:
        return []
    span = (last - first).total_seconds() / 86400
    if span <= 0:
        return []
    offsets = np.array([(c - first).total_seconds() / 86400 for c in sorted(cleanings)])
    out = []
    for _ in range(draws):
        shifted = np.sort((offsets + rng.uniform(0, span)) % span)
        out.append([first + pd.Timedelta(days=float(o)) for o in shifted])
    return out


def transplanted(cleanings, source_first, target_first):
    """The same cleaning rhythm, moved onto another record's dates.

    The wiped generation has no cleanings of its own, so the negative control needs a calendar to
    test against. Translating the hand-cleaned node's calendar preserves its rhythm exactly.
    """
    return [target_first + (c - source_first) for c in sorted(cleanings)]


def placebo_slopes(table, column, calendars, monthly):
    values = [slope(table, column, c, monthly) for c in calendars]
    values = [v['slope_per_day'] for v in values if v]
    if not values:
        return None
    return {'draws': len(values), 'p02_5': round(float(np.percentile(values, 2.5)), 5),
            'p97_5': round(float(np.percentile(values, 97.5)), 5),
            'mean': round(float(np.mean(values)), 5)}


def outside(observed, band):
    if observed is None or band is None:
        return None
    return bool(observed['slope_per_day'] < band['p02_5']
                or observed['slope_per_day'] > band['p97_5'])


def raw_step(frame, cleanings, hours=24):
    """Median of the `hours` after a cleaning minus the median before, on the RAW series."""
    out = {}
    for name, series in [('oxygen', frame.oxygen), ('deficit', raw_deficit(frame))]:
        deltas = []
        for c in cleanings:
            before = series[(series.index > c - pd.Timedelta(hours=hours)) & (series.index <= c)]
            after = series[(series.index > c) & (series.index <= c + pd.Timedelta(hours=hours))]
            if before.notna().sum() >= hours // 2 and after.notna().sum() >= hours // 2:
                deltas.append(float(after.median() - before.median()))
        out[name] = {'n': len(deltas),
                     'mean_step': round(float(np.mean(deltas)), 3) if deltas else None}
    return out


def raw_deficit(frame):
    """Departure from equilibrium solubility with no smoothing at all."""
    sol = solubility(frame.temperature.to_numpy(), np.full(len(frame), SALINITY_PSU))
    return pd.Series(sol, index=frame.index) - frame.oxygen


def placebo_steps(frame, n, rng, hours=24, draws=DRAWS):
    span = frame.index
    bands = {}
    for name, series in [('oxygen', frame.oxygen), ('deficit', raw_deficit(frame))]:
        means = []
        for _ in range(draws):
            picks = pd.to_datetime(rng.choice(span[hours:-hours or None], size=max(n, 1),
                                              replace=False))
            deltas = []
            for c in picks:
                before = series[(series.index > c - pd.Timedelta(hours=hours)) & (series.index <= c)]
                after = series[(series.index > c) & (series.index <= c + pd.Timedelta(hours=hours))]
                if before.notna().sum() >= hours // 2 and after.notna().sum() >= hours // 2:
                    deltas.append(float(after.median() - before.median()))
            if deltas:
                means.append(float(np.mean(deltas)))
        bands[name] = {'draws': len(means), 'p02_5': round(float(np.percentile(means, 2.5)), 3),
                       'p97_5': round(float(np.percentile(means, 97.5)), 3)} if means else None
    return bands


def recovery(table, cleanings, column):
    """Paired before/after over EPOCH_DAYS, the step form of the dose-response."""
    pairs = []
    for c in cleanings:
        day = c.floor('D')
        before = table[column][(table.index >= day - pd.Timedelta(days=EPOCH_DAYS))
                               & (table.index < day)]
        after = table[column][(table.index > day)
                              & (table.index <= day + pd.Timedelta(days=EPOCH_DAYS))]
        if before.notna().sum() >= 4 and after.notna().sum() >= 4:
            pairs.append(float(after.mean() - before.mean()))
    if not pairs:
        return None
    return {'n_cleanings': len(pairs), 'mean_change': round(float(np.mean(pairs)), 4),
            'sd': round(float(np.std(pairs, ddof=1)), 4) if len(pairs) > 1 else None}


def analyse(stem, cleanings, rng, label):
    frame = read_export(stem)
    table = daily_table(frame)
    first, last = table.index[0], table.index[-1]
    if label == 'wiped' and cleanings:
        # The wiped record has no cleanings of its own, so without this the control is empty and
        # says nothing. Transplanting the rhythm asks the sharper question: does the method find a
        # dose-response on a record where, by construction, there is no dose?
        cleanings = transplanted(cleanings, min(cleanings).floor('D'), first)
    inside = [c for c in cleanings if first <= c <= last]
    calendars = fake_calendars(inside, first, last, rng) if len(inside) > 1 else []
    entry = {'probe': stem, 'role': label, 'days': int(len(table)),
             'first': str(first.date()), 'last': str(last.date()),
             'cleanings_in_series': len(inside),
             'usable_days_amplitude': int(table.amp_ratio.notna().sum()),
             'median_amp_ratio': round(float(table.amp_ratio.median()), 4),
             'median_amp_oxygen_umol_l': round(float(table.amp_oxygen.median()), 3),
             'median_phase_lag_hours': round(float(table.phase_lag.median()), 3)}
    for key, column in [('B_amplitude_ratio', 'amp_ratio'), ('C_phase_lag', 'phase_lag')]:
        observed = slope(table, column, inside, monthly=False)
        seasonal = slope(table, column, inside, monthly=True)
        band = placebo_slopes(table, column, calendars, monthly=False)
        entry[key] = {'observed': observed, 'observed_with_month_effect': seasonal,
                      'fake_calendar_band': band, 'outside_placebo': outside(observed, band),
                      'recovery_at_cleaning': recovery(table, inside, column)}
    if label == 'hand_cleaned':
        entry['A_raw_step'] = raw_step(frame, inside)
        entry['A_placebo'] = placebo_steps(frame, len(inside), rng)
    return entry


#: sign a diffusion barrier predicts for each test, from addendum 27.
PREDICTED_SIGN = {'B_amplitude_ratio': -1, 'C_phase_lag': +1}


def verdict(nodes):
    """Apply addendum 27's falsifier, which is a conjunction of four conditions.

    Declared verbatim: a slope in the predicted direction "that sits outside the fake-calendar
    placebo range at either node, survives the month fixed effect, and shows recovery at the
    cleaning in test D, with no corresponding slope in the wiped second generation".

    It is evaluated here rather than in prose so the decision cannot drift once the numbers are in
    front of someone. The only quantity the declaration left open is how large a recovery counts;
    two standard errors is applied, which is the conventional choice and is recorded as an addition
    rather than something the declaration specified.
    """
    rows = []
    for name, entry in nodes.items():
        if entry['role'] != 'hand_cleaned':
            continue
        control = nodes.get(f'{name}_wiped_control', {})
        for key, sign in PREDICTED_SIGN.items():
            observed, band = entry[key]['observed'], entry[key]['fake_calendar_band']
            month, rec = entry[key]['observed_with_month_effect'], entry[key]['recovery_at_cleaning']
            if not observed:
                continue
            t = None
            if rec and rec.get('sd') and rec['n_cleanings'] > 1:
                t = rec['mean_change'] / (rec['sd'] / np.sqrt(rec['n_cleanings']))
            checks = {
                'direction_and_outside_placebo': bool(
                    entry[key]['outside_placebo'] and np.sign(observed['slope_per_day']) == sign),
                'survives_month_fixed_effect': bool(
                    month and np.sign(month['slope_per_day']) == sign),
                # Recovery restores what the dose removed, so it points the other way.
                'recovery_at_cleaning': bool(
                    t is not None and np.sign(t) == -sign and abs(t) >= 2.0),
                'silent_in_wiped_control':
                    not bool(control.get(key, {}).get('outside_placebo')),
            }
            rows.append({'node': name, 'test': key,
                         'slope_per_day': observed['slope_per_day'],
                         'recovery_t': None if t is None else round(float(t), 2),
                         'checks': checks, 'passes': bool(all(checks.values()))})
    supported = [r for r in rows if r['passes']]
    return {
        'rule': 'addendum 27: outside placebo in the predicted direction, surviving the month '
                'fixed effect, with recovery at cleaning, and silent in the wiped control',
        'recovery_threshold_t': 2.0,
        'tests': rows,
        'any_test_supported': bool(supported),
        'decision': ('rewrite Section 3.8 around the surviving observable' if supported
                     else 'remove Section 3.8 in full, per the declared decision rule'),
    }


def main():
    rng = np.random.default_rng(SEED)
    log = read_logbook()
    report = {'declared_in': 'config/bay_reserved_protocol_v2_addendum27.json',
              'epoch_days': EPOCH_DAYS, 'placebo_draws': DRAWS,
              'max_dose_days': MAX_DOSE_DAYS, 'seed': SEED, 'nodes': {}}
    for node, stem in NODES.items():
        cleanings = sorted(log[(log.node == node) & (log.action == 'Pulizia')].time)
        report['nodes'][node] = analyse(stem, cleanings, rng, 'hand_cleaned')
        # Negative control: the same fake calendars against the wiped generation, which has none.
        report['nodes'][f'{node}_wiped_control'] = analyse(WIPED[node], cleanings, rng, 'wiped')

    report['verdict'] = verdict(report['nodes'])
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
