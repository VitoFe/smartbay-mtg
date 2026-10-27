"""Addendum 81, task E: every pretrained arm as a gap filler, scored on the published rows.

Reads the zero-shot traces of the eight arms of scripts/v2_fm_zoo_io.ARMS
(reports/bay_twin/runs/v2_fm_zoo/preds/E_<variable>_<cohort>_<fwd|bwd>_L<length>__<arm>.npz, rows in
the order of the input file of the same stem, backward traces in the model's own reversed time) and
assembles, per arm, two fills on the rows of the published bidirectional Chronos arms:

  <arm>.fwd     the forward trace at the scored position (step k = position - 1)
  <arm>.bidir   v2_gap_recovery.chronos_fill's blend: w * fwd[k] + (1 - w) * bwd[k], with the
                backward trace reversed into time order and w = (L - k) / (L + 1)

The published rows, linear interpolation and the incumbents come from v2_gap_variants.load (the
published runs, arms rerun since read through v2_gap_climatology.read_arm and REPLACED_ARMS).

Scoring, as published. Error of a fill: the mean over episodes of the episode's mean absolute error
(v2_gap_recovery.score). Strict cohorts (Gap-Dev, Gap-2026): per gap length on overlap clusters of
the whole cohort (v2_gap_intervals), and over the bands 1-6 and 12-72 h on clusters of the band
(v2_gap_variants). Long cohorts (Gap-Long-Dev, Gap-Long-2026): per length only, never pooled, on
probe x calendar month of the gap start (v2_gap_long, the unit its decisions read). Every contrast is
paired on the rows where both columns exist, 600 bootstrap draws on the contrast's own stream
(seeded by the crc32 of its label), the delete-one-unit jackknife beside it, no interval on fewer
than five units. `paired` below is v2_gap_intervals.paired kept to six decimals; gate R checks that
it agrees with the published function to that function's four.

Contrasts (declaration, contrasts K1-K4), in the published units, oxygen also in mg/l:
  K1  each fill minus linear interpolation, minus the incumbent of its variable and length
      (v2_gap_variants.INCUMBENT: linear at 1-6 h; the Gaussian process over the four nearest probes
      for temperature and bidirectional Chronos-Bolt for oxygen from 12 h), and minus
      chronos_bolt_base_uni.bidir
  K2  each forward fill minus chronos_bolt_base_uni.fwd (for a bidirectional fill K2 is K1's third)
  K3  cov minus uni for chronos2, timesfm25 and timesfm3, in each direction
  K4  timesfm3_uni minus timesfm25_uni and chronos2_uni minus chronos_bolt_base_uni, each direction

Floors (Chapter 4): temperature 0.029 degC where neither column reads another instrument, 0.20 degC
where one does (a covariate arm, the Gaussian process, the published Chronos-2 with neighbours);
oxygen 0.059 mg/l. A contrast is flagged beyond its floor by the point estimate and, separately, by
the whole interval.

Gates:
  G1  chronos_bolt_base_uni.bidir reproduces the published chronos_bidir, chronos2_cov.bidir the
      published chronos2_cov_bidir and chronos_bolt_base_uni.fwd the published chronos_forward (strict
      cohorts) to 1e-4 on every row; for Chronos-Bolt past 168 h, unrolled autoregressively beyond its
      64 native steps, the episode-macro error of each length must equal the published one within
      1e-3 degC or 0.03 umol/kg, the row differences recorded beside it (amendment 4); the published 12-72 h
      interval of chronos_bidir against linear (runs/v2_gap_variants.json) is reproduced exactly,
      by v2_gap_intervals.paired under the published label, from the new fill (decisive on
      Gap-2026: -0.0771 degC and -1.8197 umol/l; Gap-Dev reported beside it)
  G2  every prediction file was computed from the input file on disk (digest), has one row per
      input context and the gap length as horizon; every published episode has a context in both
      directions at its length, in the published order; no fill is missing on a published row
  R   the six-decimal `paired` agrees with v2_gap_intervals.paired on every K1 contrast of
      chronos_bolt_base_uni.bidir against linear

The full scoring runs only when preds/ holds the 1,128 declared files and preds/rerun_check_a3.json
exists (amendment 3: TimesFM 3.0 rerun one series per call, TimesFM 2.5 at 1440 h with the head
off). With --arms, a subset is scored and written where --out says (never the declared artifact).

Writes reports/bay_twin/runs/v2_fm_zoo/score_gaps.json.

Usage: python -m scripts.v2_fm_zoo_score_gaps [--gates-only] [--arms a,b,...] [--out PATH]
"""
import argparse
import contextlib
import json
import sys
import time
import zlib
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import t as student

from scripts.v2_fm_zoo_io import ARMS, digest, load_predictions
from scripts.v2_gap_climatology import read_arm
from scripts.v2_gap_intervals import DRAWS, MIN_CLUSTERS, clusters
from scripts.v2_gap_intervals import paired as published_paired
from scripts.v2_gap_long import units as long_units
from scripts.v2_gap_variants import BANDS, COHORTS, INCUMBENT, load
from src.units import factor

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / 'reports/bay_twin/runs'
ZOO = RUNS / 'v2_fm_zoo'
INP = ZOO / 'inputs'
PRED = ZOO / 'preds'
OUT = ZOO / 'score_gaps.json'
MANIFEST = INP / 'manifest_gaps.json'
VARIANTS = RUNS / 'v2_gap_variants.json'
DECLARED_FILES = 1128
RERUN_CHECK = PRED / 'rerun_check_a3.json'
VARIABLES = ('temperature', 'oxygen')
STRICT = ('Gap-Dev', 'Gap-2026')
LONG = ('Gap-Long-Dev', 'Gap-Long-2026')
STRICT_BANDS = ('1-6', '12-72')
MODES = ('fwd', 'bidir')
TOLERANCE = 1e-4
HARDWARE_TOLERANCE = 1e-3
#: the published 12-72 h Gap-2026 contrast chronos_bidir - linear (runs/v2_gap_variants.json; the
#: values v2_chronos_adapt_contrasts.PUBLISHED_BAND checks)
PUBLISHED_BAND = {'temperature': -0.0771, 'oxygen': -1.8197}
#: the published columns each new fill must reproduce (G1)
G1_PAIRS = (('chronos_bolt_base_uni.bidir', 'chronos_bidir'),
            ('chronos2_cov.bidir', 'chronos2_cov_bidir'),
            ('chronos_bolt_base_uni.fwd', 'chronos_forward'))
#: published columns the same traces also reproduce, recorded beside G1 but not deciding it
G1_INFORMATIVE = (('chronos2_uni.bidir', 'chronos2_bidir'),)
#: amendment 4: past this gap length Chronos-Bolt is unrolled autoregressively past its 64 native
#: steps, which amplifies floating-point differences block by block, so G1 reads the scored error
BOLT_ROW_RULE_TO = 168
ERROR_TOLERANCE = {'temperature': 1e-3, 'oxygen': 0.03}
FLOOR_T_PAIRED = 0.029
FLOOR_T_INSTRUMENTS = 0.20
FLOOR_O_MGL = 0.059
#: columns whose fill reads instruments other than the probe's own record
OTHER_INSTRUMENTS = ('gpr', 'chronos2_cov_bidir')
BOLT = 'chronos_bolt_base_uni'
FAMILIES_K3 = ('chronos2', 'timesfm25', 'timesfm3')
PAIRS_K4 = (('timesfm3_uni', 'timesfm25_uni'), ('chronos2_uni', 'chronos_bolt_base_uni'))


def r6(x):
    return None if x is None or not np.isfinite(x) else round(float(x), 6)


def reads_others(column):
    return column in OTHER_INSTRUMENTS or column.split('.')[0].endswith('_cov')


def band_name(length):
    return next(b for b, (lo, hi) in BANDS.items() if lo <= length <= hi)


@contextlib.contextmanager
def gap_period_unchanged():
    """v2_gap_recovery.PERIOD is a module global the published loaders may read; nothing here sets
    it, and it is restored and checked on the way out."""
    from src.experiments import v2_gap_recovery as gap
    saved = gap.PERIOD
    try:
        yield
    finally:
        changed = gap.PERIOD != saved
        gap.PERIOD = saved
        if changed:
            raise SystemExit('v2_gap_recovery.PERIOD was changed while loading the published rows')


# ----------------------------------------------------------------------------------------- inputs

def completeness():
    npz = sorted(p.name for p in PRED.glob('*.npz'))
    return {'npz_files': len(npz), 'declared': DECLARED_FILES,
            'rerun_check_a3': RERUN_CHECK.exists(),
            'complete': len(npz) == DECLARED_FILES and RERUN_CHECK.exists()}


def stem(variable, cohort, direction, length):
    return f'E_{variable}_{cohort}_{direction}_L{int(length)}'


def file_summary(meta):
    a = meta.get('adapter', {})
    calls = a.get('rows_per_call') or []
    clamp = a.get('positivity_clamp') or {}
    return {'model_revision': meta.get('model_revision'), 'horizon_method': a.get('horizon_method'),
            'max_rows_per_call': max(calls) if calls else None,
            'per_core_batch_size': a.get('per_core_batch_size', a.get('batch')),
            'zero_outputs_under_clamp': clamp.get('of_them_with_a_zero_output'),
            'deviation_flagged_in_file': 'deviation_from_declaration' in meta,
            'created_utc': meta.get('created_utc')}


def assemble(frame, cohort, variable, arms, manifest, g2, record):
    """Add the forward and bidirectional fill of every arm to `frame` (published rows, in order)."""
    files = manifest['files']
    for length in sorted(frame.gap_length.unique()):
        rows = np.flatnonzero((frame.gap_length == length).to_numpy())
        part = frame.iloc[rows]
        expected = list(dict.fromkeys(part.episode))
        k = part.position.to_numpy(int) - 1
        w = (length - k) / (length + 1)
        traces = {}
        for direction in ('fwd', 'bwd'):
            name = stem(variable, cohort, direction, length)
            path = INP / f'{name}.npz'
            if not path.exists():
                g2['problems'].append(f'{name}: no input file')
                continue
            sha = digest(path)
            if files.get(path.name, {}).get('sha256') != sha:
                g2['problems'].append(f'{name}: input digest differs from manifest_gaps.json')
            z = np.load(path, allow_pickle=False)
            episodes = [str(e) for e in z['key_episode']]
            if episodes != expected:
                g2['problems'].append(f'{name}: episodes differ from the published rows '
                                      f'({len(episodes)} vs {len(expected)})')
                continue
            if not bool((z['reverse'] == (direction == 'bwd')).all()):
                g2['problems'].append(f'{name}: reverse flags do not match the direction')
            index = {e: i for i, e in enumerate(episodes)}
            e_idx = part.episode.map(index).to_numpy(int)
            for arm in arms:
                pfile = PRED / f'{name}__{arm}.npz'
                if not pfile.exists():
                    g2['problems'].append(f'{pfile.name}: missing')
                    continue
                p = load_predictions(pfile)
                meta = p['meta']
                med = np.asarray(p['median'], np.float64)
                ok = (meta.get('input_sha256') == sha and meta.get('arm') == arm
                      and med.shape == (len(episodes), int(length)))
                if not ok:
                    g2['problems'].append(f'{pfile.name}: digest, arm or shape does not match '
                                          f'its input ({med.shape})')
                    continue
                g2['files_checked'] += 1
                record.setdefault(arm, []).append(file_summary(meta))
                if direction == 'bwd':
                    med = med[:, ::-1]      # the model's reversed time back into time order
                traces[(arm, direction)] = med[e_idx, k]
        for arm in arms:
            fwd, bwd = traces.get((arm, 'fwd')), traces.get((arm, 'bwd'))
            for mode, values in (('fwd', fwd),
                                 ('bidir', None if fwd is None or bwd is None
                                  else w * fwd + (1 - w) * bwd)):
                col = f'{arm}.{mode}'
                if col not in frame:
                    frame[col] = np.nan
                if values is not None:
                    frame.iloc[rows, frame.columns.get_loc(col)] = values
    return frame


# ---------------------------------------------------------------------------------------- scoring

class Cell:
    """One scored cell: per-episode errors per column, and the resampling unit of each episode."""

    def __init__(self, part, unit_of, columns):
        self.part = part
        self.unit_of = unit_of
        self.per = {}
        self.complete = {}
        for col in columns:
            d = part[[col, 'truth', 'episode']].dropna(subset=[col, 'truth'])
            self.complete[col] = len(d) == len(part)
            self.per[col] = (d.assign(e=(d.truth - d[col]).abs()).groupby('episode').e.mean()
                             if len(d) else None)

    def mae(self, col):
        s = self.per.get(col)
        return None if s is None or not len(s) else float(s.mean())

    def paired(self, arm, reference, label):
        """v2_gap_intervals.paired at six decimals: arm minus reference on shared rows."""
        if self.complete.get(arm) and self.complete.get(reference) and self.part.truth.notna().all():
            ea, eb = self.per[arm], self.per[reference]
            rows = len(self.part)
        else:
            d = self.part.dropna(subset=[arm, reference, 'truth'])
            if not len(d):
                return None
            per = (d.assign(ea=(d.truth - d[arm]).abs(), eb=(d.truth - d[reference]).abs())
                   .groupby('episode')[['ea', 'eb']].mean())
            ea, eb, rows = per.ea, per.eb, len(d)
        n = len(ea)
        units = pd.Series(ea.index.map(self.unit_of), index=ea.index)
        codes, uniques = pd.factorize(units, sort=True)
        k = len(uniques)
        a, b = ea.to_numpy(float), eb.to_numpy(float)
        point = float(a.mean() - b.mean())
        out = {'delta': point, 'rows': int(rows), 'episodes': int(n), 'units': int(k)}
        if k < MIN_CLUSTERS:
            return {**out, 'ci_low': None, 'ci_high': None, 'jackknife_t': None,
                    'excludes_zero': False, 'jackknife_excludes_zero': False, 'estimable': False}
        sums = np.stack([np.bincount(codes, weights=a, minlength=k),
                         np.bincount(codes, weights=b, minlength=k)], axis=1)
        sizes = np.bincount(codes, minlength=k).astype(float)
        rng = np.random.default_rng([20260922, zlib.crc32(label.encode())])
        counts = np.empty((DRAWS, k))
        for i in range(DRAWS):
            counts[i] = np.bincount(rng.integers(0, k, k), minlength=k)
        total = counts @ sizes
        draws = (counts @ sums[:, 0]) / total - (counts @ sums[:, 1]) / total
        lo, hi = np.percentile(draws, [2.5, 97.5])
        leave = (sums.sum(axis=0) - sums) / (n - sizes)[:, None]
        theta = leave[:, 0] - leave[:, 1]
        se = float(np.sqrt((k - 1) / k * ((theta - theta.mean()) ** 2).sum()))
        half = float(student.ppf(0.975, k - 1)) * se
        return {**out, 'ci_low': float(lo), 'ci_high': float(hi),
                'jackknife_t': [point - half, point + half],
                'excludes_zero': bool(lo > 0 or hi < 0),
                'jackknife_excludes_zero': bool(point - half > 0 or point + half < 0),
                'estimable': True}


def finish(c, variable, arm, reference, c_mg):
    """Status, floors, rounding and, for oxygen, mg/l."""
    if c is None:
        return None
    if not c['estimable']:
        status = 'n.e.'
    else:
        status = 'excludes zero' if c['excludes_zero'] else 'not distinguishable'
    robust = bool(c['estimable'] and c['excludes_zero'] and c['jackknife_excludes_zero'])
    out = {'delta': r6(c['delta']), 'ci_low': r6(c['ci_low']), 'ci_high': r6(c['ci_high']),
           'jackknife_t': None if c['jackknife_t'] is None else [r6(x) for x in c['jackknife_t']],
           'status': status, 'excludes_zero': c['excludes_zero'],
           'jackknife_excludes_zero': c['jackknife_excludes_zero'],
           'robust': robust,
           'robust_chapter4': bool(c['estimable'] and c['excludes_zero']
                                   and (c['units'] >= 20 or c['jackknife_excludes_zero'])),
           'rows': c['rows'], 'episodes': c['episodes'], 'units': c['units'],
           'estimable': c['estimable']}
    if variable == 'oxygen':
        scale = lambda x: None if x is None else r6(x * c_mg)
        mg = {'delta': scale(c['delta']), 'ci_low': scale(c['ci_low']), 'ci_high': scale(c['ci_high']),
              'jackknife_t': None if c['jackknife_t'] is None else [scale(x) for x in c['jackknife_t']]}
        out['mgl'] = mg
        floor, value = FLOOR_O_MGL, mg
    else:
        floor = FLOOR_T_INSTRUMENTS if (reads_others(arm) or reads_others(reference)) else FLOOR_T_PAIRED
        value = c
    out['floor'] = floor
    out['beyond_floor'] = bool(abs(value['delta']) > floor)
    out['interval_beyond_floor'] = bool(c['estimable'] and value['ci_low'] is not None
                                        and (value['ci_high'] < -floor or value['ci_low'] > floor))
    return out


def score_cell(part, unit_of, unit_kind, variable, length_or_band, label, fills, arms, c_mg, replica):
    incumbent = INCUMBENT[(variable, length_or_band if isinstance(length_or_band, str)
                           else band_name(length_or_band))]
    refs = ['linear', incumbent, f'{BOLT}.bidir'] if incumbent != 'linear' else ['linear', f'{BOLT}.bidir']
    columns = list(dict.fromkeys(fills + ['linear', incumbent, 'chronos_bidir', 'chronos2_cov_bidir']))
    columns = [c for c in columns if c in part]
    cell = Cell(part, unit_of, columns)
    out = {'gap_lengths': sorted(int(x) for x in part.gap_length.unique()),
           'rows': int(len(part)), 'episodes': int(part.episode.nunique()),
           'units': int(len({unit_of[e] for e in part.episode.unique()})), 'unit': unit_kind,
           'incumbent': incumbent, 'mae': {}, 'missing_rows': {}, 'K1': {}, 'K2': {}, 'K3': {}, 'K4': {}}
    for col in columns:
        m = cell.mae(col)
        entry = {'mae': r6(m)}
        if variable == 'oxygen':
            entry['mae_mgl'] = None if m is None else r6(m * c_mg)
        out['mae'][col] = entry
        out['missing_rows'][col] = int(part[col].isna().sum())

    def contrast(arm, ref):
        got = cell.paired(arm, ref, f'a81E|{label}|{arm}|{ref}')
        return finish(got, variable, arm, ref, c_mg)

    for fill in fills:
        out['K1'][fill] = {f'minus_{ref}': contrast(fill, ref) for ref in refs if ref != fill}
        if incumbent == 'linear':
            out['K1'][fill]['minus_incumbent'] = 'minus_linear (linear is the incumbent here)'
        else:
            out['K1'][fill]['minus_incumbent'] = f'minus_{incumbent}'
        if fill.endswith('.fwd') and fill != f'{BOLT}.fwd':
            out['K2'][fill] = contrast(fill, f'{BOLT}.fwd')
    for family in FAMILIES_K3:
        for mode in MODES:
            a, b = f'{family}_cov.{mode}', f'{family}_uni.{mode}'
            if a in fills and b in fills:
                out['K3'][f'{a} - {b}'] = contrast(a, b)
    for new, old in PAIRS_K4:
        for mode in MODES:
            a, b = f'{new}.{mode}', f'{old}.{mode}'
            if a in fills and b in fills:
                out['K4'][f'{a} - {b}'] = contrast(a, b)
    # gate R: the six-decimal replica against the published function, on one contrast per cell
    a = f'{BOLT}.bidir'
    if a in fills:
        tag = f'a81E|{label}|{a}|linear'
        mine = cell.paired(a, 'linear', tag)
        theirs = published_paired(part, a, 'linear', tag, unit_of)
        if mine and theirs:
            agree = (round(mine['delta'], 4) == theirs['delta'] and mine['episodes'] == theirs['episodes']
                     and mine['units'] == theirs['clusters'])
            if theirs['estimable']:
                agree &= (round(mine['ci_low'], 4) == theirs['ci_low']
                          and round(mine['ci_high'], 4) == theirs['ci_high']
                          and all(abs(x - y) <= 1.0001e-4
                                  for x, y in zip(mine['jackknife_t'], theirs['jackknife_t'])))
            replica.append({'cell': label, 'agree': bool(agree), 'published': theirs,
                            'six_decimals': {k: (r6(v) if isinstance(v, float) else v)
                                             for k, v in mine.items() if k != 'jackknife_t'}})
    return out


# ------------------------------------------------------------------------------------------ gates

def reproduction(frame, new, published, variable):
    """Agreement of a new fill with a published column per gap length, with what the difference does
    to the error (episode-macro MAE, new minus published).

    Row by row to TOLERANCE, except (amendment 4) for a Chronos-Bolt fill at a gap longer than
    BOLT_ROW_RULE_TO h, unrolled autoregressively past its 64 native steps: there the scored error
    must equal the published one within ERROR_TOLERANCE, and the row differences are recorded."""
    a, b = frame[new].to_numpy(float), frame[published].to_numpy(float)
    both = np.isfinite(a) & np.isfinite(b)
    nan_mismatch = int((np.isfinite(a) != np.isfinite(b)).sum())
    diff = np.where(both, np.abs(a - b), np.nan)
    worst = float(np.nanmax(diff)) if both.any() else float('nan')
    bolt = new.startswith('chronos_bolt_')
    by_length = {}
    for length, idx in frame.groupby('gap_length').indices.items():
        d = diff[idx]
        part = frame.iloc[idx]
        m_new, m_pub = macro_mae(part, new), macro_mae(part, published)
        delta = m_new - m_pub if m_new is not None and m_pub is not None else None
        complete = bool(np.isfinite(d).all())
        rule = 'error' if bolt and length > BOLT_ROW_RULE_TO else 'row'
        if rule == 'row':
            passed = bool(complete and np.nanmax(d) <= TOLERANCE)
        else:
            passed = bool(complete and delta is not None
                          and abs(delta) <= ERROR_TOLERANCE[variable])
        by_length[str(int(length))] = {
            'rule': rule,
            'tolerance': TOLERANCE if rule == 'row' else ERROR_TOLERANCE[variable],
            'mae_new': r6(m_new), 'mae_published': r6(m_pub),
            'mae_new_minus_published': None if delta is None else float(f'{delta:.9g}'),
            'max_abs_difference': float(np.nanmax(d)) if np.isfinite(d).any() else None,
            'median_abs_difference': float(np.nanmedian(d)) if np.isfinite(d).any() else None,
            'rows': int(len(idx)), 'rows_above_1e-4': int(np.nansum(d > TOLERANCE)),
            'rows_above_1e-3': int(np.nansum(d > HARDWARE_TOLERANCE)),
            'passed': passed}
    return {'rows': int(len(frame)), 'compared': int(both.sum()), 'nan_mismatch': nan_mismatch,
            'max_abs_difference': worst, 'rows_above_1e-4': int(np.nansum(diff > TOLERANCE)),
            'rows_above_1e-3': int(np.nansum(diff > HARDWARE_TOLERANCE)),
            'passed': bool(nan_mismatch == 0 and both.sum() == len(frame)
                           and all(e['passed'] for e in by_length.values())),
            'passed_row_by_row_everywhere': bool(nan_mismatch == 0 and both.sum() == len(frame)
                                                 and worst <= TOLERANCE),
            'lengths_failing': [L for L, e in by_length.items() if not e['passed']],
            'lengths_read_on_the_error': [L for L, e in by_length.items() if e['rule'] == 'error'],
            'by_gap_length': by_length}


def macro_mae(frame, column):
    d = frame.dropna(subset=[column, 'truth'])
    if not len(d):
        return None
    return float(d.assign(e=(d.truth - d[column]).abs()).groupby('episode').e.mean().mean())


def gate_g1(frame, cohort, variable, fills):
    """The declared pairs (decisive, with amendment 4's forward Chronos-Bolt fill, which exists
    published on the strict cohorts only) and, for information, Chronos-2 univariate."""
    out, extra = {}, {}
    for pairs, target in ((G1_PAIRS, out), (G1_INFORMATIVE, extra)):
        for new, published in pairs:
            if new in fills and published in frame:
                target[f'{new} vs {published}'] = reproduction(frame, new, published, variable)
    return out, extra


def gate_band(frame, cohort, variable, variants):
    """The published 12-72 h interval of chronos_bidir - linear, from the published column (harness)
    and from the new fill under the published label (the gate)."""
    band = frame[(frame.gap_length >= 12) & (frame.gap_length <= 72)]
    cluster_of = clusters(band)
    label = f'{cohort}|{variable}|12-72|chronos_bidir|linear'
    published = variants['results'][cohort][variable]['12-72']['chronos_bidir']['minus_linear']
    harness = published_paired(band, 'chronos_bidir', 'linear', label, cluster_of)
    swapped = band.copy()
    swapped['chronos_bidir'] = band[f'{BOLT}.bidir'].to_numpy()
    reproduced = published_paired(swapped, 'chronos_bidir', 'linear', label, cluster_of)
    keys = ('delta', 'ci_low', 'ci_high', 'clusters', 'episodes', 'jackknife_t', 'excludes_zero',
            'jackknife_excludes_zero')
    same = {k: reproduced.get(k) == published.get(k) for k in keys}
    entry = {'published': published, 'from_published_column': harness, 'from_new_fill': reproduced,
             'fields_equal': same, 'reproduced_exactly': all(same.values()),
             'harness_exact': all(harness.get(k) == published.get(k) for k in keys)}
    if cohort == 'Gap-2026':
        entry['delta_matches_cited'] = bool(abs(reproduced['delta'] - PUBLISHED_BAND[variable]) < 5e-5)
        entry['passed'] = bool(entry['reproduced_exactly'] and entry['delta_matches_cited'])
    else:
        entry['passed'] = bool(entry['reproduced_exactly'])
    return entry


# ------------------------------------------------------------------------------------------- main

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--arms', default=','.join(ARMS),
                        help='comma-separated subset of the eight arms (a test; needs --out)')
    parser.add_argument('--gates-only', action='store_true', help='G1, G2 and the band gate only')
    parser.add_argument('--out', default=None)
    args = parser.parse_args()
    arms = [a for a in args.arms.split(',') if a]
    unknown = [a for a in arms if a not in ARMS]
    if unknown:
        raise SystemExit(f'unknown arms {unknown}')
    full = arms == list(ARMS) and not args.gates_only
    status = completeness()
    if full and not status['complete']:
        raise SystemExit(f'preds/ is not complete ({status}); the full scoring waits for it. '
                         'Use --arms with --out for a test on finished arms.')
    out_path = Path(args.out) if args.out else OUT
    if not full and out_path.resolve() == OUT.resolve():
        raise SystemExit('a subset or gates-only run is never written to the declared artifact; '
                         'pass --out')
    t0 = time.time()
    c_mg = factor()
    manifest = json.loads(MANIFEST.read_text())
    variants = json.loads(VARIANTS.read_text())
    fills = [f'{arm}.{mode}' for arm in arms for mode in MODES]
    g2 = {'problems': [], 'files_checked': 0, 'files_expected': 0}
    record, gates_g1, informative, gates_band, missing = {}, {}, {}, {}, {}
    replica, result = [], {}
    for cohort in STRICT + LONG:
        for variable in VARIABLES:
            with gap_period_unchanged():
                frame = load(cohort, variable).reset_index(drop=True)
                if cohort in STRICT:
                    # Chronos-Bolt forward-only, reproduced by the forward fill under G1 (amendment 4)
                    run, results = COHORTS[cohort][0], COHORTS[cohort][1]
                    fwd = read_arm(run, results, variable, 'chronos_forward')
                    if not fwd[['episode', 'position']].equals(frame[['episode', 'position']]):
                        raise SystemExit(f'{cohort} {variable}: chronos_forward scores other rows')
                    frame['chronos_forward'] = fwd['chronos_forward'].to_numpy()
            if cohort in LONG and frame.truth.isna().any():
                raise SystemExit(f'{cohort} {variable}: an unobserved hour is among the rows')
            lengths = sorted(frame.gap_length.unique())
            g2['files_expected'] += 2 * len(lengths) * len(arms)
            frame = assemble(frame, cohort, variable, arms, manifest, g2, record)
            key = f'{cohort} {variable}'
            missing[key] = {f: int(frame[f].isna().sum()) for f in fills}
            gates_g1[key], informative[key] = gate_g1(frame, cohort, variable, fills)
            if cohort in STRICT and f'{BOLT}.bidir' in fills:
                gates_band[key] = gate_band(frame, cohort, variable, variants)
            print(f'{key}: {len(frame)} rows; G1',
                  {k: (round(v['max_abs_difference'], 7), v['lengths_failing'])
                   for k, v in gates_g1[key].items()},
                  '; band', gates_band.get(key, {}).get('passed'), flush=True)
            if args.gates_only:
                continue
            cells = {}
            if cohort in STRICT:
                whole = clusters(frame)
                for length in lengths:
                    part = frame[frame.gap_length == length]
                    cells[str(int(length))] = score_cell(
                        part, whole, 'overlap cluster (whole cohort)', variable, int(length),
                        f'{cohort}|{variable}|{int(length)}', fills, arms, c_mg, replica)
                for band in STRICT_BANDS:
                    lo, hi = BANDS[band]
                    part = frame[(frame.gap_length >= lo) & (frame.gap_length <= hi)]
                    if not len(part):
                        continue
                    cells[band] = score_cell(
                        part, clusters(part), 'overlap cluster (band)', variable, band,
                        f'{cohort}|{variable}|{band}', fills, arms, c_mg, replica)
            else:
                _, conservative = long_units(frame)
                for length in lengths:
                    part = frame[frame.gap_length == length]
                    cells[str(int(length))] = score_cell(
                        part, conservative, 'probe x calendar month of the gap start (within length)',
                        variable, int(length), f'{cohort}|{variable}|{int(length)}', fills, arms,
                        c_mg, replica)
            result.setdefault(cohort, {})[variable] = {
                'unit_of_values': 'degC' if variable == 'temperature' else 'umol/L as stored (per kg); mgl = x c',
                'rows': int(len(frame)), 'episodes': int(frame.episode.nunique()), 'cells': cells}
            print(f'   scored {len(cells)} cells, {time.time() - t0:.0f} s', flush=True)

    g2_missing = sum(v for m in missing.values() for v in m.values())
    g2['fills_missing_on_published_rows'] = missing
    g2['passed'] = bool(not g2['problems'] and g2['files_checked'] == g2['files_expected']
                        and g2_missing == 0)
    g1_passed = all(v['passed'] for cell in gates_g1.values() for v in cell.values())
    g1_failing = {f'{key} | {pair}': v['lengths_failing'] for key, cell in gates_g1.items()
                  for pair, v in cell.items() if not v['passed']}
    band_passed = all(v['passed'] for k, v in gates_band.items() if k.startswith('Gap-2026'))
    replica_state = ('not run (gates only)' if args.gates_only
                     else bool(replica and all(r['agree'] for r in replica)))
    passed = g1_passed and band_passed and g2['passed']
    if passed and replica_state is True:
        reading = 'allowed: every gate passed'
    elif passed and args.gates_only:
        reading = 'G1, the band interval and G2 passed; gate R runs with the scoring (gates only)'
    else:
        reading = ('stopped: a gate failed; the declaration (gates.on_failure) stops the reading of '
                   'task E until an amendment explains it')
    files = {}
    for arm, entries in record.items():
        files[arm] = {
            'files': len(entries),
            'model_revisions': sorted({e['model_revision'] for e in entries if e['model_revision']}),
            'horizon_methods': sorted({e['horizon_method'] for e in entries if e['horizon_method']}),
            'max_rows_per_call': max((e['max_rows_per_call'] or 0) for e in entries),
            'zero_outputs_under_clamp': int(sum(e['zero_outputs_under_clamp'] or 0 for e in entries)),
            'files_flagging_a_deviation': int(sum(e['deviation_flagged_in_file'] for e in entries)),
            'created_utc': [min(e['created_utc'] or '' for e in entries),
                            max(e['created_utc'] or '' for e in entries)]}
    report = {
        'declared_in': 'config/bay_reserved_protocol_v2_addendum81.json',
        'amendments': [f'config/bay_reserved_protocol_v2_addendum81_amendment{i}.json'
                       for i in (1, 2, 3, 4)],
        'task': 'E_gaps', 'producer': 'scripts/v2_fm_zoo_score_gaps.py',
        'arms': arms, 'fills': fills, 'complete_run': full, 'preds_status': status,
        'inputs': {'manifest': 'runs/v2_fm_zoo/inputs/manifest_gaps.json',
                   'manifest_sha256': digest(MANIFEST),
                   'published_rows': 'v2_gap_variants.load (runs in v2_gap_variants.COHORTS, arms '
                                     'rerun since through v2_gap_climatology.REPLACED_ARMS)',
                   'variants_sha256': digest(VARIANTS)},
        'rules': {
            'fill': 'fwd: forward trace at step position - 1; bidir: w * fwd + (1 - w) * bwd '
                    '(backward trace reversed into time order), w = (L - k) / (L + 1), '
                    'k = position - 1 (v2_gap_recovery.chronos_fill)',
            'error': 'mean over episodes of the episode mean absolute error on the published rows '
                     '(long cohorts: observed hours only)',
            'strict_cohorts': 'per length on overlap clusters of the whole cohort '
                              '(v2_gap_intervals); bands 1-6 and 12-72 h on clusters of the band '
                              '(v2_gap_variants)',
            'long_cohorts': 'per length only, on probe x calendar month of the gap start '
                            '(v2_gap_long conservative unit); never pooled across lengths',
            'interval': f'paired on rows where both columns exist; {DRAWS} bootstrap draws over the '
                        'unit, stream seeded [20260922, crc32(label)], label '
                        "'a81E|<cohort>|<variable>|<cell>|<arm>|<reference>'; delete-one-unit "
                        f'jackknife with t(k - 1) beside it; n.e. below {MIN_CLUSTERS} units',
            'status': "'excludes zero' or 'not distinguishable' (bootstrap interval), 'n.e.'; "
                      'robust: bootstrap and jackknife both exclude zero; robust_chapter4: '
                      'bootstrap excludes zero and (20 units or more, or the jackknife agrees)',
            'incumbent': 'v2_gap_variants.INCUMBENT by band of the length: linear at 1-6 h; from '
                         '12 h the Gaussian process over the four nearest probes (temperature) and '
                         'bidirectional Chronos-Bolt (oxygen), as Table 5 and v2_gap_tables name them',
            'K2': 'forward fills only: minus chronos_bolt_base_uni.fwd; for a bidirectional fill '
                  'K2 is K1 minus_chronos_bolt_base_uni.bidir',
            'floors': {'temperature_one_probe': FLOOR_T_PAIRED,
                       'temperature_other_instruments': FLOOR_T_INSTRUMENTS,
                       'oxygen_mgl': FLOOR_O_MGL,
                       'reads_other_instruments': 'any *_cov fill, gpr, chronos2_cov_bidir',
                       'beyond_floor': '|point| > floor (oxygen in mg/l)',
                       'interval_beyond_floor': 'the whole interval beyond the floor on one side'},
            'oxygen_factor_mg_per_l_per_umol_kg': c_mg},
        'reading': reading,
        'gates': {'G1_reproduction': {'tolerance': TOLERANCE,
                                      'bolt_error_rule_above_h': BOLT_ROW_RULE_TO,
                                      'bolt_error_tolerance': ERROR_TOLERANCE,
                                      'cells': gates_g1, 'passed': g1_passed,
                                      'failing_lengths': g1_failing},
                  'G1_informative': {'note': 'recorded beside G1, not deciding it', 'cells': informative},
                  'G1_band_12_72': {'cells': gates_band, 'passed': band_passed},
                  'G2_rows': g2,
                  'R_paired_replica': {'contrasts': len(replica), 'passed': replica_state,
                                       'disagreeing': [r for r in replica if not r['agree']]}},
        'prediction_files': files,
        'results': result,
        'seconds': round(time.time() - t0, 1),
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_bytes((json.dumps(report, indent=1) + '\n').encode('utf-8'))
    print(json.dumps({'G1': g1_passed, 'G1_failing_lengths': g1_failing, 'band': band_passed,
                      'G2': g2['passed'], 'R': replica_state, 'reading': reading,
                      'problems': g2['problems'][:10]}, indent=1))
    if not args.gates_only:
        summarise(report)
    print('wrote', out_path, f'({report["seconds"]} s)')


def summarise(report):
    """The cells the thesis reads: the 12-72 h band on the strict cohorts, 168/336/720 h past a week."""
    for cohort, by_var in report['results'].items():
        for variable, block in by_var.items():
            keys = ['12-72'] if cohort in STRICT else ['168', '336', '720']
            for cell_key in keys:
                cell = block['cells'].get(cell_key)
                if not cell:
                    continue
                inc = cell['incumbent']
                print(f'\n== {cohort} {variable} {cell_key} h  (units {cell["units"]}, incumbent {inc})')
                lin = cell['mae']['linear']
                print(f"   {'linear':<30} {lin.get('mae_mgl', lin['mae'])}")
                for fill, k1 in cell['K1'].items():
                    m = cell['mae'][fill]
                    def fmt(c):
                        if not isinstance(c, dict):
                            return '-'
                        v = c.get('mgl', c)
                        mark = {'excludes zero': '*', 'not distinguishable': '', 'n.e.': ' n.e.'}[c['status']]
                        return f"{v['delta']:+.4f}{mark}{'R' if c['robust'] else ''}"
                    print(f"   {fill:<30} {m.get('mae_mgl', m['mae'])}  lin {fmt(k1.get('minus_linear'))}"
                          f"  inc {fmt(k1.get('minus_' + inc) if inc != 'linear' else k1.get('minus_linear'))}")


if __name__ == '__main__':
    sys.exit(main())
