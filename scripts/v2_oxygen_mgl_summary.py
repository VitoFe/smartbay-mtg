"""Addendum 60 part C: every part-C rerun against c x published, for each oxygen number the thesis prints.

For each published artifact the analyses of part C mixed per-kilogram and per-litre oxygen in, this
maps the artifact to its mg/l rerun (written by scripts/v2_oxygen_mgl_*.py under
reports/bay_twin/runs/v2_oxygen_mgl/) and lists every oxygen number the thesis prints from it (found
through the \\path citations of the chapters, tables and appendix that thesis.tex reaches): the
published value, c x published, the rerun value, the relative change, and for a contrast whether
its interval's exclusion of zero, or its sign, changes. It then evaluates the declared predictions
of addendum 60 part C:

  PC1  no contrast the thesis reports changes whether its interval excludes zero, or its sign where
       it excludes zero
  PC2  no reported oxygen error changes by more than 3 % between c x published and the rerun

An 'error' is a model's or a baseline's MAE (PC2 is evaluated on these); a 'floor' is a
disagreement statistic between instruments; a 'statistic' is a descriptive oxygen quantity (a
minimum, a median difference); a 'contrast' is a difference of two errors with its interval. A
contrast's relative change is reported against the reference arm's c x published MAE where the
artifact has one, as part B's criterion is written, and against the contrast itself otherwise.

Usage: python -m scripts.v2_oxygen_mgl_summary   ->  reports/bay_twin/runs/v2_oxygen_mgl/harmonised_summary.json
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.v2_oxygen_mgl_common import MG_PER_UMOL_L, OUT, RUNS, constant_c, write_json

C = constant_c()
DST = OUT / 'harmonised_summary.json'
PC2_LIMIT = 0.03


def J(path):
    return json.loads(Path(path).read_text())


def P(rel):
    return RUNS / rel


def M(rel):
    return OUT / rel


ENTRIES = []


def add(artifact, rerun, part, where, label, kind, pub, mgl, printed=None, reference=None, per_litre_only=False,
        jackknife=None, hourly=None, amendment4=False):
    """pub/mgl: a float, or [point, lo, hi] for a contrast. reference: (pub, mgl) MAE of the reference arm."""
    ENTRIES.append(dict(artifact=artifact, rerun=rerun, part=part, where=where, label=label, kind=kind,
                        pub=pub, mgl=mgl, printed=printed, reference=reference, per_litre_only=per_litre_only,
                        jackknife=jackknife, hourly=hourly, amendment4=amendment4))


def excl(v):
    return bool(v[1] > 0 or v[2] < 0)


def evaluate(e):
    factor = C                     # c x published is what a uniform conversion would print
    pub, mgl = e['pub'], e['mgl']
    out = {k: e[k] for k in ('artifact', 'rerun', 'part', 'where', 'label', 'kind', 'printed')}
    if isinstance(pub, (list, tuple)):
        cp = [factor * x for x in pub]
        out.update(published=list(pub), c_times_published=cp, rerun_value=list(mgl),
                   change=mgl[0] - cp[0])
        if e['reference']:
            rp, rm = e['reference']
            out['relative_change_of_reference_mae'] = (mgl[0] - cp[0]) / abs(factor * rp)
        out['relative_change'] = (mgl[0] - cp[0]) / abs(cp[0]) if cp[0] else None
        out['excludes_zero'] = {'published': excl(pub), 'rerun': excl(mgl)}
        out['exclusion_changed'] = excl(pub) != excl(mgl)
        out['sign_changed'] = bool(np.sign(pub[0]) != np.sign(mgl[0]))
        out['sign_changed_where_excluding'] = bool(out['sign_changed'] and (excl(pub) or excl(mgl)))
        if e['jackknife']:
            jp, jm = e['jackknife']
            out['jackknife'] = {'published': jp, 'rerun': jm, 'c_times_published': [factor * x for x in jp],
                                'excludes_zero_published': bool(jp[0] > 0 or jp[1] < 0),
                                'excludes_zero_rerun': bool(jm[0] > 0 or jm[1] < 0)}
            out['jackknife']['exclusion_changed'] = out['jackknife']['excludes_zero_published'] != \
                out['jackknife']['excludes_zero_rerun']
    else:
        cp = factor * pub
        out.update(published=pub, c_times_published=cp, rerun_value=mgl, change=mgl - cp,
                   relative_change=(mgl - cp) / abs(cp) if cp else None)
    if e['amendment4']:
        out['method'] = ('amendment 4: sonde oxygen x c, saturation per litre x 0.0319988, only the utilisation arms '
                         'refitted, every reference its published value x c')
        h = e['hourly']
        out['hourly_factor_run'] = h
        if isinstance(pub, (list, tuple)):
            out['vs_hourly_factor_run'] = {'excludes_zero_hourly': excl(h),
                                           'exclusion_differs': excl(h) != excl(mgl),
                                           'sign_differs': bool(np.sign(h[0]) != np.sign(mgl[0]))}
        else:
            out['vs_hourly_factor_run'] = {'relative_difference': (mgl - h) / abs(h) if h else None}
    if e['per_litre_only']:
        out['converted_by'] = ('per-litre source only: the harmonised value is 0.0319988 x published exactly (not '
                               'rerun); c x published overstates it by c/0.0319988 - 1 = 2.6 %')
    return out


# ------------------------------------------------------------------ C1: Field-57 and Field-2026

def field57():
    a = '20260919T145045Z_v2_field'
    pub, mgl = J(P(f'{a}/v2_field_summary.json'))['arms'], J(M(f'C1/{a}/v2_field_summary.json'))['arms']
    for arm, printed in (('A1_both_direct', '12.060'), ('A2_none_direct', '12.182'), ('A3_design_only_direct', '12.024'),
                         ('A4_base_only_direct', '11.978')):
        add(f'runs/{a}', f'runs/v2_oxygen_mgl/C1/{a}', 'C1', 'table_bay_boundary_ablation', f'{arm} nested MAE',
            'error', pub[arm]['do_umol_l'], mgl[arm]['do_umol_l'], printed)
    pc, mc = J(P('v2_field_contrasts.json')), J(M('C1/v2_field_contrasts.json'))

    def fc(rep, label):
        x = next(c for c in rep['contrasts'] if c['contrast'] == label and c['target'] == 'do_umol_l')['date']
        return [x['point'], *x['ci']]
    ref = {'A2_none_direct': (pub['A2_none_direct']['do_umol_l'], mgl['A2_none_direct']['do_umol_l']),
           'A1_both_direct': (pub['A1_both_direct']['do_umol_l'], mgl['A1_both_direct']['do_umol_l'])}
    for label, where, printed, r in (
            ('boundary product, both uses, vs none', 'table_bay_boundary_ablation; table_bay_perfold (published)',
             '-0.123 [-1.013, +0.743]', 'A2_none_direct'),
            ('background offset only vs none', 'table_bay_boundary_ablation', '-0.204 [-1.057, +0.606]', 'A2_none_direct'),
            ('regressors only vs none', 'table_bay_boundary_ablation', '-0.158 [-0.575, +0.146]', 'A2_none_direct'),
            ('background offset only vs both', 'table_bay_boundary_ablation', '-0.082 [-0.185, +0.023]', 'A1_both_direct'),
            ('model vs oracle climatology', 'table_bay_field_comparators; table_bay_summary', '-0.173 [-3.834, +3.521]', None),
            ('model vs background alone', 'table_bay_field_comparators', '-0.556* [-1.041, -0.107]', None),
            ('no boundary product vs oracle climatology', 'table_bay_field_comparators', '-0.051 [-3.465, +3.749]', None),
            ('CTD prior, harmonic, vs none', 'chapter5 (one penalty: worse by about 1, interval excluding zero)',
             '+0.971* [+0.168, +1.703]', 'A2_none_direct'),
            ('CTD prior, harmonic, vs Copernicus', 'chapter5 (one penalty: worse by about 1, interval excluding zero)',
             '+1.094* [+0.273, +1.775]', 'A1_both_direct')):
        add('runs/v2_field_contrasts.json', 'runs/v2_oxygen_mgl/C1/v2_field_contrasts.json',
            'C2' if 'CTD prior' in label else 'C1', where, label, 'contrast', fc(pc, label), fc(mc, label), printed,
            reference=ref.get(r))
    pn, mn = J(P('v2_field_naive/field_naive_summary.json'))['nested_mae'], \
        J(M('C1/v2_field_naive/field_naive_summary.json'))['nested_mae']
    for k, printed in (('climatology', '18.5338'), ('seasonal', '23.8338'), ('oracle', '12.2331'),
                       ('no_copernicus', '12.1824'), ('background', '12.6155'), ('model', '12.0597')):
        add('runs/v2_field_naive/field_naive_summary.json', 'runs/v2_oxygen_mgl/C1/v2_field_naive/field_naive_summary.json',
            'C2' if k in ('climatology', 'seasonal', 'oracle') else 'C1', 'table_bay_field_comparators',
            f'{k} nested MAE', 'error', pn[k]['do_umol_l'], mn[k]['do_umol_l'], printed)
    # addendum 47 (per fold) and 48
    p47, rerun47 = J(P('v2_perfold.json'))['field57'], J(M('C1/v2_perfold.json'))['field57']
    add('runs/v2_perfold.json', 'runs/v2_oxygen_mgl/C1/v2_perfold.json', 'C1', 'table_bay_field_comparators note',
        'A1 per-fold nested MAE', 'error', p47['arms']['A1_both_direct']['nested_mae']['perfold']['do_umol_l'],
        rerun47['arms']['A1_both_direct']['nested_mae']['perfold']['do_umol_l'], '12.462')
    add('runs/v2_perfold.json', 'runs/v2_oxygen_mgl/C1/v2_perfold.json', 'C1', 'table_bay_field_comparators note',
        'model - background, per fold', 'contrast', p47['descriptive_not_declared']['model_minus_background']['do_umol_l']['perfold'],
        rerun47['descriptive_not_declared']['model_minus_background']['do_umol_l']['perfold'], '-0.154 [-0.960, +0.479]')
    printed47 = {('boundary product, both uses, vs none', 'shared'): '-0.123 [-0.974, +0.668]',
                 ('boundary product, both uses, vs none', 'perfold'): '+0.006 [-0.698, +0.770] (chapter5: +0.01)',
                 ('boundary product, both uses, vs none', 'movement'): '+0.129 [-0.090, +0.345]',
                 ('CTD prior, harmonic, vs none', 'shared'): '+0.971* [+0.178, +1.698]',
                 ('CTD prior, harmonic, vs none', 'perfold'): '+0.335 [-0.322, +1.088] (chapter5: +0.34)',
                 ('CTD prior, harmonic, vs none', 'movement'): '-0.636* [-0.817, -0.094]',
                 ('CTD prior, harmonic, vs Copernicus', 'shared'): '+1.094* [+0.299, +1.756]',
                 ('CTD prior, harmonic, vs Copernicus', 'perfold'): '+0.329 [-0.346, +1.161] (chapter5: +0.33)',
                 ('CTD prior, harmonic, vs Copernicus', 'movement'): '-0.765* [-0.937, -0.089]'}
    for (label, design), printed in printed47.items():
        def get(rep):
            e = next(x for x in rep['contrasts'] if x['contrast'] == label and x['target'] == 'do_umol_l')
            return e['movement perfold - published'] if design == 'movement' else e[design]['value']
        add('runs/v2_perfold.json', 'runs/v2_oxygen_mgl/C1/v2_perfold.json', 'C2' if 'CTD' in label else 'C1',
            'table_bay_perfold', f'{label}, {design}', 'contrast', get(p47), get(rerun47), printed)
    p48, rerun48 = J(P('v2_perfold_rest.json'))['field57'], J(M('C1/v2_perfold_rest.json'))['field57']
    e = lambda rep: next(x for x in rep['contrasts'] if x['contrast'] == 'CTD prior, month x depth, vs none'
                         and x['target'] == 'do_umol_l')['perfold']['value']
    add('runs/v2_perfold_rest.json', 'runs/v2_oxygen_mgl/C1/v2_perfold_rest.json', 'C2', 'chapter5',
        'CTD prior, month x depth, vs none, per fold', 'contrast', e(p48), e(rerun48), '+1.03 [0.09, 1.75]')


def field2026():
    p36, m36 = J(P('v2_field_reserved2026/report.json')), J(M('C2/v2_field_reserved2026/report.json'))
    p38, m38 = J(P('v2_field_reserved2026_copernicus/report.json')), J(M('C1/v2_field_reserved2026_copernicus/report.json'))
    a36, a38 = 'runs/v2_field_reserved2026/report.json', 'runs/v2_field_reserved2026_copernicus/report.json'
    r36, r38 = 'runs/v2_oxygen_mgl/C2/v2_field_reserved2026/report.json', 'runs/v2_oxygen_mgl/C1/v2_field_reserved2026_copernicus/report.json'
    for arm, printed in (('A2_none', '12.18'), ('A7o', '13.18')):
        add(a36, r36, 'C2', 'table_bay_field_2026 (Field-57 column)', f'{arm} development nested MAE', 'error',
            p36['development'][arm]['nested_mae']['do_umol_l'], m36['development'][arm]['nested_mae']['do_umol_l'], printed)
    for arm, printed in (('A2_none', '12.71'), ('A7o', '13.03'), ('seasonal', '13.91')):
        add(a36, r36, 'C2', 'table_bay_field_2026 (Field-2026 column)', f'{arm} held-station nested MAE', 'error',
            p36['held_station']['nested_mae'][arm]['do_umol_l'], m36['held_station']['nested_mae'][arm]['do_umol_l'], printed)
    for arm, printed in (('A4_base_only', '11.98'), ('A1_both', '12.06')):
        add(a38, r38, 'C1', 'table_bay_field_2026 (Field-57 column)', f'{arm} Field-57 nested MAE (gate G2)', 'error',
            p38['G2'][arm]['nested_mae']['do_umol_l'], m38['G2'][arm]['nested_mae']['do_umol_l'], printed)
    for arm, printed in (('A4_base_only', '12.84'), ('A1_both', '12.95')):
        add(a38, r38, 'C1', 'table_bay_field_2026 (Field-2026 column)', f'{arm} held-station nested MAE', 'error',
            p38['held_station']['nested_mae'][arm]['do_umol_l'], m38['held_station']['nested_mae'][arm]['do_umol_l'], printed)

    def cv(rep, key):
        x = rep['held_station']['contrasts'][key]['do_umol_l']
        return [x['point'], *x['ci']], x['jackknife_t']
    ref36 = (p36['held_station']['nested_mae']['A2_none']['do_umol_l'], m36['held_station']['nested_mae']['A2_none']['do_umol_l'])
    (pv, pj), (mv, mj) = cv(p36, 'A7o - A2_none'), cv(m36, 'A7o - A2_none')
    add(a36, r36, 'C2', 'table_bay_field_2026', 'cast prior - no prior, oxygen, Field-2026', 'contrast', pv, mv,
        '+0.318 [-1.191, +1.498]', reference=ref36, jackknife=(pj, mj))
    for key, printed in (('A4_base_only - A2_none', '+0.134 [-0.081, +0.315]'),
                         ('A1_both - A2_none', '+0.245* [+0.035, +0.453] (table_bay_summary)')):
        (pv, pj), (mv, mj) = cv(p38, key), cv(m38, key)
        add(a38, r38, 'C1', 'table_bay_field_2026', f'{key}, oxygen, Field-2026', 'contrast', pv, mv, printed,
            reference=ref36, jackknife=(pj, mj))


# ------------------------------------------------------------------ C2: the cast archive

def casts():
    po, mo = J(P('v2_oxygen_floor.json')), J(M('C2/v2_oxygen_floor.json'))
    a, r = 'runs/v2_oxygen_floor.json', 'runs/v2_oxygen_mgl/C2/v2_oxygen_floor.json'
    pre = lambda rep: rep['by_cohort']['pre_2026']['30min']
    for st, printed in (('PBaia', '2.065 / 4.246 / -0.318'), ('PPortovenereS', '0.809 / 1.056 / +0.370'),
                        ('PMitiliE', '0.959 / 0.959 / -0.959')):
        for stat in ('median_abs', 'mean_abs', 'mean_signed'):
            add(a, r, 'C2', 'table_bay_oxygen_floor', f'{st} {stat}', 'floor',
                pre(po)['per_site'][st]['oxygen'][stat], pre(mo)['per_site'][st]['oxygen'][stat], printed)
    for stat, printed in (('median_abs', '1.803 (the 1.8 floor)'), ('mean_abs', '3.772'), ('p90_abs', '7.234'),
                          ('mean_signed', '-0.260')):
        add(a, r, 'C2', 'table_bay_oxygen_floor; chapter3', f'pooled before 2026 {stat}', 'floor',
            pre(po)['pooled']['oxygen'][stat], pre(mo)['pooled']['oxygen'][stat], printed)
    bp, bm = po['checks_2026_09_26']['bootstrap_over_cast_dates']['oxygen'], mo['checks_2026_09_26']['bootstrap_over_cast_dates']['oxygen']
    for stat, printed in (('median_abs_95', '[0.97, 2.37]'), ('mean_abs_95', '[1.87, 6.27]'), ('p90_abs_95', '[3.02, 11.73]')):
        for i, end in enumerate(('lower', 'upper')):
            add(a, r, 'C2', 'table_bay_oxygen_floor', f'pooled before 2026 {stat} {end}', 'floor', bp[stat][i], bm[stat][i], printed)
    allp, allm = po['by_cohort']['all_years']['30min']['pooled']['oxygen'], mo['by_cohort']['all_years']['30min']['pooled']['oxygen']
    for stat, printed in (('median_abs', '1.433'), ('mean_abs', '3.444'), ('p90_abs', '7.110'), ('mean_signed', '-0.499')):
        add(a, r, 'C2', 'table_bay_oxygen_floor', f'all years {stat}', 'floor', allp[stat], allm[stat], printed)
    tp = po['checks_2026_09_26']['timing']['true_time']['30min']['oxygen']
    tm = mo['checks_2026_09_26']['timing']['true_time']['30min']['oxygen']
    for stat, printed in (('median_abs', '1.49'), ('mean_abs', '4.02')):
        add(a, r, 'C2', 'table_bay_oxygen_floor note', f'nearest reading in true time {stat}', 'floor',
            tp[stat], tm[stat], printed)
    pm, mm = J(P('v2_metrology_review.json'))['G2_sonde_adjustment']['oxygen_floor'], \
        J(M('C2/v2_metrology_review.json'))['G2_sonde_adjustment']['oxygen_floor']
    a, r = 'runs/v2_metrology_review.json', 'runs/v2_oxygen_mgl/C2/v2_metrology_review.json'
    for stat, printed in (('median_abs', '2.138'), ('mean_abs', '3.093'), ('p90_abs', '7.549'), ('mean_signed', '+1.046')):
        add(a, r, 'C2', 'table_bay_oxygen_floor', f'adjusted months left out {stat}', 'floor',
            pm['without_flagged_months'][stat], mm['without_flagged_months'][stat], printed)
    for stat, printed in (('median_abs_95', '[1.42, 2.54]'), ('mean_abs_95', '[1.85, 4.60]'), ('p90_abs_95', '[3.21, 11.73]')):
        for i, end in enumerate(('lower', 'upper')):
            add(a, r, 'C2', 'table_bay_oxygen_floor', f'adjusted months left out {stat} {end}', 'floor',
                pm['without_flagged_months_bootstrap'][stat][i], mm['without_flagged_months_bootstrap'][stat][i], printed)
    nd, ndm = pm['not_declared_cast_against_read_and_native'], mm['not_declared_cast_against_read_and_native']
    for block, stat, printed in (('flagged_months', 'median_abs_cast_minus_read', '0.77'),
                                 ('flagged_months', 'median_abs_cast_minus_native', '6.40'),
                                 ('other_months', 'median_abs_cast_minus_read', '2.14'),
                                 ('other_months', 'median_abs_cast_minus_native', '2.05')):
        add(a, r, 'C2', 'chapter3 (the adjusted sonde-months)', f'{block} {stat}', 'floor', nd[block][stat], ndm[block][stat], printed)
    pc, mc = J(P('v2_ctd_floor.json'))['cohorts'], J(M('C2/v2_ctd_floor.json'))['cohorts']
    a, r = 'runs/v2_ctd_floor.json', 'runs/v2_oxygen_mgl/C2/v2_ctd_floor.json'
    for cohort, printed in (('CTD-722', '1.54 [1.35, 1.74]'), ('gen2_before_2026', '2.02 [1.34, 2.95]')):
        add(a, r, 'C2', 'chapter3; table_bay_summary', f'cast-to-cast mean, {cohort}', 'floor',
            [pc[cohort]['oxygen']['mean_over_station_days'], *pc[cohort]['oxygen']['mean_95_over_cast_dates']],
            [mc[cohort]['oxygen']['mean_over_station_days'], *mc[cohort]['oxygen']['mean_95_over_cast_dates']], printed)
    ps, ms = J(P('v2_ctd_sections.json')), J(M('C2/v2_ctd_sections.json'))
    a, r = 'runs/v2_ctd_sections.json', 'runs/v2_oxygen_mgl/C2/v2_ctd_sections.json'
    for st, label in (('PBaia', 'PBaia'), ('PBoaT_deep', 'PBoaT 19-21 m')):
        for season in ('DJF', 'MAM', 'JJA', 'SON'):
            add(a, r, 'C2', 'table_bay_ctd_stratification', f'{label} {season} surface - bottom oxygen, median', 'statistic',
                ps['seasons'][st][season]['dO_median'], ms['seasons'][st][season]['dO_median'],
                str(ps['seasons'][st][season]['dO_median']))
    ss, sm = ps['oxygen_under_salinity_stratification'], ms['oxygen_under_salinity_stratification']
    for block, key, printed in (('salinity_stratified', 'dO_median', '+13.0'), ('mixed', 'dO_median', '+0.6'),
                                ('salinity_stratified', 'dSol_median', '4.3 (solubility)')):
        add(a, r, 'C2', 'table_bay_ctd_stratification note; chapter5', f'{block} {key}', 'statistic',
            ss[block][key], sm[block][key], printed)
    for q, printed in ((0, '+3.9'), (1, '+29.2')):
        for k in ('dO_iqr', 'dO_quartiles'):
            if k in ss['salinity_stratified']:
                add(a, r, 'C2', 'table_bay_ctd_stratification note; chapter5', f'salinity_stratified {k}[{q}]', 'statistic',
                    ss['salinity_stratified'][k][q], sm['salinity_stratified'][k][q], printed)
    add(a, r, 'C2', 'table_bay_ctd_stratification note', 'salinity_stratified dSat_median (percentage points, not a '
        'concentration: compared as published, not times c)', 'statistic_percent',
        ss['salinity_stratified']['dSat_median'], sm['salinity_stratified']['dSat_median'], '4.3 percentage points')
    cast = lambda rep: next(x for x in rep['declines']['casts_within_7_days_before'] if x['cast_date'] == '2025-03-26')
    add(a, r, 'C2', 'chapter5 (the cast before two water-mass declines)', 'PBaia 2025-03-26 surface - bottom oxygen',
        'statistic', cast(ps)['cast_dO'], cast(ms)['cast_dO'], '40.8')
    p5, m5 = J(P('v2_section5_numbers.json'))['observability'], J(M('C2/v2_section5_numbers.json'))['observability']
    a, r = 'runs/v2_section5_numbers.json', 'runs/v2_oxygen_mgl/C2/v2_section5_numbers.json'
    add(a, r, 'C2', 'chapter5', 'moored oxygen minimum', 'statistic', p5['moored']['min'], m5['moored']['min'], '141.3')
    add(a, r, 'C2', 'chapter5', 'near-bottom cast minimum', 'statistic', p5['ctd_near_bottom']['min'],
        m5['ctd_near_bottom']['min'], '149.7 (approximately 4.8 mg/l)')
    sp, sm_ = p5['surface_minus_bottom_july_october'], m5['surface_minus_bottom_july_october']
    for k, i, printed in (('median', None, '+1.3'), ('quartiles', 0, '-5.0'), ('quartiles', 1, '+8.4'), ('max', None, '64.4')):
        add(a, r, 'C2', 'chapter5', f'surface - bottom July-October {k}{"" if i is None else f"[{i}]"}', 'statistic',
            sp[k] if i is None else sp[k][i], sm_[k] if i is None else sm_[k][i], printed)
    ph = pd.read_csv(P('v2_placement/placement_horizontal_summary.csv'))
    mh = pd.read_csv(M('C2/v2_placement/placement_horizontal_summary.csv'))
    add('runs/v2_placement', 'runs/v2_oxygen_mgl/C2/v2_placement', 'C2', 'chapter5',
        'largest station-removal degradation, oxygen', 'statistic', float(ph[ph.variable == 'do_umol_l'].mean_degradation.max()),
        float(mh[mh.variable == 'do_umol_l'].mean_degradation.max()), '0.44')
    po_, mo_ = J(P('v2_observability_reanalysis.json')), J(M('C2/v2_observability_reanalysis.json'))
    a, r = 'runs/v2_observability_reanalysis.json', 'runs/v2_oxygen_mgl/C2/v2_observability_reanalysis.json'
    f2p, f2m = po_['F2_placement'], mo_['F2_placement']
    for h, printed in (('PBaia', '4.38 [2.33, 6.61]'), ('PBoaS', '5.72 [1.33, 10.13]'), ('PMitiliE', '-3.02 [-7.48, +1.29]'),
                       ('PBoaT', 'interval spans zero'), ('PPortovenereP', 'interval spans zero')):
        add(a, r, 'C2', 'chapter5 (vertical transfer)', f'donor transfer - naive, held {h}', 'contrast',
            [f2p['held_stations'][h]['diff'], *f2p['held_stations'][h]['ci']],
            [f2m['held_stations'][h]['diff'], *f2m['held_stations'][h]['ci']], printed,
            reference=(f2p['held_stations'][h]['naive'], f2m['held_stations'][h]['naive']))
    x, y = f2p['all_held_stations_mean_over_donor_sets'], f2m['all_held_stations_mean_over_donor_sets']
    add(a, r, 'C2', 'chapter5 (vertical transfer)', 'donor transfer - naive, all held stations', 'contrast',
        [x['diff'], *x['ci']], [y['diff'], *y['ci']], '2.08 [0.15, 4.01]', reference=(x['naive'], y['naive']))
    for d_, printed in (('PBoaT', '8.53 [4.72, 12.44]'), ('PMitiliE', '6.05 [3.19, 8.89]')):
        x, y = f2p['single_donors'][d_], f2m['single_donors'][d_]
        add(a, r, 'C2', 'chapter5 (single donors)', f'single donor {d_} - naive', 'contrast',
            [x['diff'], *x['ci']], [y['diff'], *y['ci']], printed, reference=(x['naive'], y['naive']))
    for k in ('PMitiliE', 'PBaia'):
        pass
    cv = lambda rep: rep['F1_near_bottom']['contrasts']['P1_held_station|covered']['C2_moored_only|lightgbm|aou_ref']
    x, y = cv(po_), cv(mo_)
    add(a, r, 'C2/C3', 'chapter5 (covered casts)', 'best mean-state arm reading nothing at the bed - naive, covered',
        'contrast', [x['diff'], *x['ci']], [y['diff'], *y['ci']], 'gains 0.37 [-2.52, +3.57]', reference=(x['naive'], y['naive']))
    # near-bottom models
    pb, mb = J(P('v2_bottom_aou/bottom/contrasts.json')), J(M('C2/v2_bottom_aou/bottom/contrasts.json'))
    a, r = 'runs/v2_bottom_aou/bottom/contrasts.json', 'runs/v2_oxygen_mgl/C2/v2_bottom_aou/bottom/contrasts.json'
    g = lambda rep, pr, m: next(e for e in rep if e['protocol'] == pr and e['model'] == m)
    x, y = g(pb, 'P1_held_station', 'C1_cast_surface|ridge|gradient'), g(mb, 'P1_held_station', 'C1_cast_surface|ridge|gradient')
    add(a, r, 'C2', 'chapter5', 'ridge on the surface-to-bottom difference, nested MAE', 'error', x['mae'], y['mae'], '11.93')
    add(a, r, 'C2', 'chapter5', 'naive (bottom = surface), nested MAE', 'error', x['naive'], y['naive'], '14.50')
    add(a, r, 'C2', 'chapter5', 'ridge gradient - naive', 'contrast', [x['diff'], *x['ci']], [y['diff'], *y['ci']],
        'reduction 2.6 (18 %) [0.1, 4.8]', reference=(x['naive'], y['naive']), jackknife=(x['jackknife_t'], y['jackknife_t']))
    for e in pb:
        if e['protocol'] == 'P2_chronological' and e['model'].startswith('C1_cast_surface'):
            y = g(mb, 'P2_chronological', e['model'])
            add(a, r, 'C2' if not e['model'].endswith('aou') else 'C3', 'chapter5 (2026: profile models by 3.7 to 5.5)',
                f'{e["model"]} - naive, 2026', 'contrast', [e['diff'], *e['ci']], [y['diff'], *y['ci']], '3.7 to 5.5',
                reference=(e['naive'], y['naive']), jackknife=(e['jackknife_t'], y['jackknife_t']))
    pr_, mr_ = J(P('v2_bottom_aou/bottom_ref/contrasts.json')), J(M('C2/v2_bottom_aou/bottom_ref/contrasts.json'))
    a, r = 'runs/v2_bottom_aou/bottom_ref/contrasts.json', 'runs/v2_oxygen_mgl/C2/v2_bottom_aou/bottom_ref/contrasts.json'

    def best_ref(rep):
        pool = [e for e in rep['against_naive'] if e['protocol'] == 'P1_held_station' and e['model'].startswith('C1_cast_surface')]
        return min(pool, key=lambda e: e['mae'])
    bp_, bm_ = best_ref(pr_), best_ref(mr_)
    add(a, r, 'C3', 'chapter5', f'best utilisation arm reading nothing at the bed ({bp_["model"]} / {bm_["model"]}), nested MAE',
        'error', bp_['mae'], bm_['mae'], '12.92')
    cc = lambda rep, m: next(e for e in rep['contrasts'] if e['protocol'] == 'P1_held_station' and e['a'] == m
                             and e['b'] == 'C1_cast_surface|ridge|gradient')
    x, y = cc(pr_, bp_['model']), cc(mr_, bp_['model'])
    add(a, r, 'C3', 'chapter5', f'{bp_["model"]} - ridge gradient', 'contrast', [x['diff'], *x['ci']], [y['diff'], *y['ci']],
        'worse by 1.0 [0.4, 1.7]', reference=(g(pb, 'P1_held_station', 'C1_cast_surface|ridge|gradient')['mae'],
                                             g(mb, 'P1_held_station', 'C1_cast_surface|ridge|gradient')['mae']))


# ------------------------------------------------------------------ C3: utilisation

def utilisation():
    """The C3 rows. The sonde-only utilisation arms follow amendment 4 (sonde oxygen x c, references
    published x c); each such row also carries the value of the first part-C3 run (every sonde hour by its
    own factor, every arm refitted), kept in C3/_hourly_factor/."""
    H = 'C3/_hourly_factor/'
    pa, ma, ha = (J(P('v2_aou_forecast/aou_forecast_h6_summary.json')), J(M('C3/v2_aou_forecast/aou_forecast_h6_summary.json')),
                  J(M(H + 'v2_aou_forecast/aou_forecast_h6_summary.json')))
    a, r = 'runs/v2_aou_forecast/', 'runs/v2_oxygen_mgl/C3/v2_aou_forecast/'
    for arm, printed in (('plain_tree', '6.3155'), ('aou_tree', '6.4339'), ('weather_tree', '6.5322'),
                         ('climatology', '7.2468'), ('persistence', '8.2166'), ('thermodynamic', '8.4808')):
        add(a, r, 'C3', 'table_bay_aou_forecast', f'{arm} MAE', 'error', pa['mae'][arm], ma['mae'][arm], printed,
            hourly=ha['mae'][arm], amendment4=True)
    for k, printed in (('aou_tree - weather_tree', '-0.098 [-0.244, +0.025]'), ('aou_tree - plain_tree', '+0.118 [-0.077, +0.274]'),
                       ('aou_tree - persistence', '-1.783* [-2.348, -1.241] (the 1.78 margin)'),
                       ('thermodynamic - persistence', '+0.264* [+0.199, +0.339]'),
                       ('thermodynamic - climatology', '+1.234* [+0.747, +1.678]'),
                       ('thermodynamic - weather_tree', '+1.949* [+1.503, +2.479]')):
        refarm = k.split(' - ')[1]
        add(a, r, 'C3', 'table_bay_aou_forecast', k, 'contrast', pa['contrasts'][k], ma['contrasts'][k], printed,
            reference=(pa['mae'][refarm], ma['mae'][refarm]), hourly=ha['contrasts'][k], amendment4=True)
    p48, rerun48, h48 = J(P('v2_perfold_rest.json'))['aou'], J(M('C3/v2_perfold_rest.json'))['aou'], J(M(H + 'v2_perfold_rest.json'))['aou']
    cv = lambda rep, k, d: (lambda e: e['movement perfold - published'] if d == 'movement' else e[d]['value'])(
        next(e for e in rep['contrasts'] if e['contrast'] == k))
    for k in ('aou_tree - weather_tree', 'aou_tree - plain_tree', 'aou_tree - persistence', 'thermodynamic - persistence',
              'thermodynamic - weather_tree', 'thermodynamic - climatology'):
        for d in ('published', 'shared', 'perfold', 'movement'):
            where = ('table_bay_aou_forecast note; ' if d == 'perfold' and k in ('aou_tree - weather_tree', 'aou_tree - plain_tree')
                     else '') + 'table_bay_perfold_rest'
            printed = {('aou_tree - weather_tree', 'perfold'): '+0.083 [-0.117, +0.239]',
                       ('aou_tree - plain_tree', 'perfold'): '+0.174 [-0.009, +0.328]'}.get((k, d), 'table_bay_perfold_rest')
            add('runs/v2_perfold_rest.json', 'runs/v2_oxygen_mgl/C3/v2_perfold_rest.json', 'C3', where,
                f'{k}, {d}' + (' design' if d != 'movement' else ' (per fold - published)'), 'contrast',
                cv(p48, k, d), cv(rerun48, k, d), printed, hourly=cv(h48, k, d), amendment4=True)
    pu, mu = J(P('v2_bottom_aou/aou_units/summary.json')), J(M('C3/v2_bottom_aou/aou_units/summary.json'))
    a, r = 'runs/v2_bottom_aou/aou_units/summary.json', 'runs/v2_oxygen_mgl/C3/v2_bottom_aou/aou_units/summary.json'
    g = lambda rep, co, x, y: next(e for e in rep['contrasts'] if e['cohort'] == co and e['a'] == x and e['b'] == y)
    for co, x_, y_, printed in (('gen2_164', 'aou_l', 'direct_do', 'helps by 3.94 [2.04, 5.67] (table_bay_summary)'),
                                ('gen2_164', 'direct_do_ts', 'direct_do', 'improves by 1.20'),
                                ('gen2_164', 'aou_l', 'direct_do_ts', 'beats it by 2.74 [0.84, 4.45]'),
                                ('full_722', 'aou_l', 'direct_do', 'not distinguishable from zero'),
                                ('full_722', 'aou_l', 'direct_do_ts', 'the two measurements account for all of the gain')):
        x, y = g(pu, co, x_, y_), g(mu, co, x_, y_)
        add(a, r, 'C3', 'chapter5', f'{co} ridge {x_} - {y_}', 'contrast', [x['diff'], *x['ci']], [y['diff'], *y['ci']], printed,
            reference=(pu['mae'][f'{co}|{y_}']['nested'], mu['mae'][f'{co}|{y_}']['nested']),
            jackknife=(x['jackknife_t'], y['jackknife_t']))
    ps, ms = J(P('v2_bottom_aou/aou_summary.json')), J(M('C3/v2_bottom_aou/aou_summary.json'))
    a, r = 'runs/v2_bottom_aou/aou_summary.json', 'runs/v2_oxygen_mgl/C3/v2_bottom_aou/aou_summary.json'
    for co, model, printed in (('gen2_164', 'M3_lightgbm', 'gains 2.8'), ('gen2_164', 'M3b_xgboost', 'gains 3.9'),
                               ('full_722', 'M3_lightgbm', 'neutral'), ('full_722', 'M3b_xgboost', 'costs 5.1')):
        x, y = g(ps, co, f'{model}|aou', f'{model}|direct_do'), g(ms, co, f'{model}|aou', f'{model}|direct_do')
        add(a, r, 'C3', 'chapter5', f'{co} {model} aou - direct', 'contrast', [x['diff'], *x['ci']], [y['diff'], *y['ci']],
            printed, reference=(ps['mae'][f'{co}|{model}|direct_do'], ms['mae'][f'{co}|{model}|direct_do']),
            jackknife=(x['jackknife_t'], y['jackknife_t']))
    # pretrained arms: the thesis prints the 'l' unit
    pc, mc, hc = (J(P('v2_aou_units_chronos.json')), J(M('C3/v2_aou_units_chronos.json')),
                  J(M(H + 'v2_aou_units_chronos.json')))
    a, r = 'runs/v2_aou_units_chronos.json', 'runs/v2_oxygen_mgl/C3/v2_aou_units_chronos.json'
    model_v2 = (pc['arms']['bolt_aou']['l']['mae'] - pc['arms']['bolt_aou']['l']['minus_v2'][0],
                mc['references_mae']['MODEL_v2'])
    rows = (('bolt_aou', '5.3935 [-0.862, -0.102]*; chapter5 0.46 [0.10, 0.86]'), ('bolt_aou_c2', '5.3995 [-0.864, -0.091]*'),
            ('ch2_aou_cov', '5.4644 [-0.672, -0.016]*'), ('ch2_aou_uni', '5.5497 [-0.570, +0.086]'),
            ('lora_posthoc', '5.4089 (post-hoc, no interval printed)'), ('ft_posthoc', '5.5109 (post-hoc, no interval printed)'),
            ('tv_full', '5.5995 [-0.666, +0.396]'), ('tv_lora', '5.6236 [-0.612, +0.229]'),
            ('ft_random', '6.2994 [+0.060, +1.370]*'), ('lora_random', '6.4128 [+0.141, +1.339]*'),
            ('refit_full', 'chapter5 (refit after the held-out quarter)'), ('refit_lora', 'chapter5 (refit after the held-out quarter)'))
    for n, printed in rows:
        x, y, h = pc['arms'][n]['l'], mc['arms'][n]['mgl'], hc['arms'][n]['mgl']
        add(a, r, 'C3', 'table_bay_chronos', f'{n} MAE (mgl)', 'error', x['mae'], y['mae'], printed, hourly=h['mae'],
            amendment4=True)
        add(a, r, 'C3', 'table_bay_chronos', f'{n} - released model (mgl)', 'contrast', x['minus_v2'], y['minus_v2'],
            printed, reference=model_v2, hourly=h['minus_v2'], amendment4=True)
    for ref, printed in (('MODEL_v2', '5.8489 (released model)'), ('oxygen|ZS_raw', '5.5320 (Chronos-Bolt zs)'),
                         ('B0_persistence', '7.4817 (persistence)')):
        add(a, r, 'C3', 'table_bay_chronos (reference row)', f'{ref} MAE (published x c)', 'error',
            pc['arms']['bolt_aou']['l']['mae'] - pc['arms']['bolt_aou']['l']['minus_v2'][0] if ref == 'MODEL_v2' else
            mc['gates']['G3_references']['by_arm'][ref]['published_x_c'] / C,
            mc['references_mae'][ref], printed, hourly=hc['references_mae'][ref], amendment4=True)
    add(a, r, 'C3', 'chapter5', 'Chronos-Bolt utilisation - Chronos-Bolt raw', 'contrast',
        pc['contrasts']['bolt_aou@l - oxygen|ZS_raw'], mc['contrasts']['bolt_aou@mgl - oxygen|ZS_raw'], '-0.14 [-0.81, +0.22]',
        hourly=hc['contrasts']['bolt_aou@mgl - oxygen|ZS_raw'], amendment4=True)
    for n in ('bolt_aou', 'bolt_aou_c2', 'ch2_aou_uni'):
        x, y, h = pc['sensor_axis'][n]['l'], mc['sensor_axis'][n]['mgl'], hc['sensor_axis'][n]['mgl']
        for scheme in ('published', 'two_way'):
            add(a, r, 'C3', 'table_bay_sensor_axis', f'{n} - released, {scheme} resampling', 'contrast',
                [x['point'], *x[scheme]['ci']], [y['point'], *y[scheme]['ci']],
                {'bolt_aou': '-0.455', 'bolt_aou_c2': '-0.449', 'ch2_aou_uni': '-0.299'}[n], reference=model_v2,
                hourly=[h['point'], *h[scheme]['ci']], amendment4=True)
        for s_, v in x['per_sensor'].items():
            add(a, r, 'C3', 'table_bay_sensor_axis; chapter5', f'{n} - released, {s_}', 'statistic', v, y['per_sensor'][s_],
                {'bolt_aou': '-0.486 / -0.425'}.get(n, 'per-sensor range'), hourly=h['per_sensor'][s_], amendment4=True)
    # per-litre only: a multiple of 0.0319988, not of c
    sc = J(P('v2_salinity_check.json'))['residual_after_cancellation']['mean_abs']
    add('runs/v2_salinity_check.json', '(not rerun: per-litre saturation only)', 'C3', 'chapter5',
        'constant-salinity residual, mean absolute', 'statistic', sc, sc * MG_PER_UMOL_L, '0.016', per_litre_only=True)


# ------------------------------------------------------------------ C4: the Tinetto check

def tinetto():
    pt, mt = J(P('eda_opportunities/eda_b_seaphox.json')), J(M('C4/eda_opportunities/eda_b_seaphox.json'))
    a, r = 'runs/eda_opportunities/eda_b_seaphox.json', 'runs/v2_oxygen_mgl/C4/eda_opportunities/eda_b_seaphox.json'
    for k, printed in (('corr_daily', 'not printed'), ('corr_anomaly_30d', 'no skill on variation shorter than a month')):
        add(a, r, 'C4', 'chapter5', f'Copernicus against SeapHOx at 16 m, oxygen {k} (unit-free)', 'statistic_unitfree',
            pt['copernicus_vs_seaphox_16m']['O2'][k], mt['copernicus_vs_seaphox_16m']['O2'][k], printed)
    for k in ('bias_cm_minus_obs', 'mae'):
        add(a, r, 'C4', '(not printed)', f'Copernicus against SeapHOx at 16 m, oxygen {k}', 'statistic',
            pt['copernicus_vs_seaphox_16m']['O2'][k], mt['copernicus_vs_seaphox_16m']['O2'][k], None)
    lows = J(P('eda_opportunities/eda_b2_lows.json'))['minimum']['O2']
    add('runs/eda_opportunities/eda_b2_lows.json', '(not rerun: the SeapHOx alone, per litre)', 'C4', 'chapter5',
        'SeapHOx minimum at 16 m', 'statistic', lows, lows * MG_PER_UMOL_L, '111.3', per_litre_only=True)


def main():
    field57()
    field2026()
    casts()
    utilisation()
    tinetto()
    rows = [evaluate(e) for e in ENTRIES]
    for row in rows:
        if row['kind'] in ('statistic_unitfree', 'statistic_percent'):
            row['c_times_published'] = row['published']
            row['change'] = row['rerun_value'] - row['published']
            row['relative_change'] = row['change'] / abs(row['published']) if row['published'] else None
            row['note'] = 'not a concentration: compared with the published value itself'
    errors = [r_ for r_ in rows if r_['kind'] == 'error']
    contrasts = [r_ for r_ in rows if r_['kind'] == 'contrast']
    pc2_fail = [r_ for r_ in errors if abs(r_['relative_change']) > PC2_LIMIT]
    pc1_fail = [r_ for r_ in contrasts if r_['exclusion_changed'] or r_['sign_changed_where_excluding']]
    jk_changes = [r_ for r_ in contrasts if r_.get('jackknife', {}).get('exclusion_changed')]
    floors = [r_ for r_ in rows if r_['kind'] == 'floor']
    largest = sorted(errors + floors, key=lambda r_: -abs(r_['relative_change']))[:12]
    report = {
        'declared_in': ['config/bay_reserved_protocol_v2_addendum60.json',
                        'config/bay_reserved_protocol_v2_addendum60_amendment1.json',
                        'config/bay_reserved_protocol_v2_addendum60_amendment4.json'],
        'c_mg_per_l_per_umol_kg': C, 'mg_per_l_per_umol_l': MG_PER_UMOL_L,
        'amendment4': amendment4(rows),
        'artifacts': ARTIFACTS,
        'not_rerun': NOT_RERUN,
        'reproduction_records': REPRODUCTION,
        'numbers': rows,
        'PC1': {'statement': 'No contrast the thesis reports changes whether its interval excludes zero, or its sign '
                             'where it excludes zero.',
                'contrasts_checked': len(contrasts), 'held': not pc1_fail,
                'changed': [{k: r_[k] for k in ('artifact', 'where', 'label', 'published', 'c_times_published',
                                                 'rerun_value', 'excludes_zero', 'sign_changed')} for r_ in pc1_fail],
                'jackknife_exclusion_changed': [{k: r_[k] for k in ('artifact', 'where', 'label', 'jackknife')}
                                                for r_ in jk_changes],
                'sign_changed_without_exclusion': [{k: r_[k] for k in ('artifact', 'where', 'label', 'c_times_published',
                                                                        'rerun_value')}
                                                   for r_ in contrasts if r_['sign_changed'] and not r_['sign_changed_where_excluding']]},
        'PC2': {'statement': 'No reported oxygen error changes by more than 3 % between c x published and the harmonised rerun.',
                'errors_checked': len(errors), 'held': not pc2_fail,
                'largest_relative_change': max((abs(r_['relative_change']) for r_ in errors), default=None),
                'exceeding_3_percent': [{k: r_[k] for k in ('artifact', 'where', 'label', 'published', 'c_times_published',
                                                             'rerun_value', 'relative_change')} for r_ in pc2_fail],
                'floors_largest_relative_change': max((abs(r_['relative_change']) for r_ in floors), default=None)},
        'largest_changes': [{k: r_[k] for k in ('artifact', 'label', 'kind', 'c_times_published', 'rerun_value',
                                                'relative_change')} for r_ in largest],
        'refit_noise_control': REFIT_NOISE,
    }
    write_json(DST, json.loads(json.dumps(report, default=float)))
    print(f'{len(rows)} numbers; PC1 held {report["PC1"]["held"]} ({len(contrasts)} contrasts, {len(pc1_fail)} changed); '
          f'PC2 held {report["PC2"]["held"]} ({len(errors)} errors, largest {report["PC2"]["largest_relative_change"]:.4f})')
    for r_ in pc1_fail:
        print('  PC1:', r_['artifact'], r_['label'], r_['c_times_published'], r_['rerun_value'])
    for r_ in pc2_fail:
        print('  PC2:', r_['artifact'], r_['label'], round(r_['relative_change'], 4))


def amendment4(rows):
    """The sonde-only utilisation rows under amendment 4 against the first part-C3 run: stars, signs, order."""
    a4 = [r_ for r_ in rows if r_.get('method')]
    diffs = [{k: r_[k] for k in ('where', 'label', 'c_times_published', 'rerun_value', 'hourly_factor_run',
                                  'vs_hourly_factor_run')}
             for r_ in a4 if isinstance(r_['rerun_value'], list)
             and (r_['vs_hourly_factor_run']['exclusion_differs'] or r_['vs_hourly_factor_run']['sign_differs'])]

    def order(where, key):
        errs = [r_ for r_ in a4 if r_['kind'] == 'error' and r_['where'].startswith(where)]
        return [r_['label'].replace(' MAE', '') for r_ in sorted(errs, key=lambda r_: r_[key])]
    orderings = {}
    for where in ('table_bay_aou_forecast', 'table_bay_chronos'):
        orderings[where] = {'c_x_published': order(where, 'c_times_published'), 'amendment4': order(where, 'rerun_value'),
                            'hourly_factor_run': order(where, 'hourly_factor_run')}
        orderings[where]['amendment4_equals_c_x_published'] = orderings[where]['amendment4'] == orderings[where]['c_x_published']
        orderings[where]['amendment4_equals_hourly_factor_run'] = \
            orderings[where]['amendment4'] == orderings[where]['hourly_factor_run']
    return {'what': 'the sonde-only utilisation rows (forecasting tree, thermodynamic baseline, per-fold designs, '
                    'pretrained arms and the sensor-axis rows read from them) follow amendment 4: sonde oxygen x c, '
                    'saturation per litre x 0.0319988, only the utilisation arms refitted, every reference its '
                    'published value x c; the first part-C3 run (every sonde hour by its own factor, every arm '
                    'refitted) is kept in runs/v2_oxygen_mgl/C3/_hourly_factor/',
            'rows': len(a4),
            'star_or_sign_differs_from_hourly_factor_run': diffs,
            'orderings_by_mae': orderings}


ARTIFACTS = {
    'runs/20260919T145045Z_v2_field': {'rerun': 'runs/v2_oxygen_mgl/C1/20260919T145045Z_v2_field', 'part': 'C1',
                                       'producer': 'scripts/v2_oxygen_mgl_field.py, scripts/v2_oxygen_mgl_field_contrasts.py'},
    'results/v2_field_held_site_A5_intercept_direct': {'rerun': 'runs/v2_oxygen_mgl/C1/results/v2_field_held_site_A5_intercept_direct', 'part': 'C1', 'producer': 'scripts/v2_oxygen_mgl_field.py'},
    'results/v2_field_held_site_A6_ctdprior_direct': {'rerun': 'runs/v2_oxygen_mgl/C2/results/v2_field_held_site_A6_ctdprior_direct', 'part': 'C2', 'producer': 'scripts/v2_oxygen_mgl_field.py'},
    'results/v2_field_held_site_A7_ctdprior_direct': {'rerun': 'runs/v2_oxygen_mgl/C2/results/v2_field_held_site_A7_ctdprior_direct', 'part': 'C2', 'producer': 'scripts/v2_oxygen_mgl_field.py'},
    'results/v2_field_held_site_P1_placebo_direct/': {'rerun': 'runs/v2_oxygen_mgl/C2/results/v2_field_held_site_P1_placebo_direct', 'part': 'C2', 'producer': 'scripts/v2_oxygen_mgl_field.py'},
    'results/v2_field_held_site_P2_placebo_direct/': {'rerun': 'runs/v2_oxygen_mgl/C2/results/v2_field_held_site_P2_placebo_direct', 'part': 'C2', 'producer': 'scripts/v2_oxygen_mgl_field.py'},
    'runs/v2_field_naive/field_naive_summary.json': {'rerun': 'runs/v2_oxygen_mgl/C1/v2_field_naive/field_naive_summary.json', 'part': 'C1/C2', 'producer': 'scripts/v2_oxygen_mgl_field_contrasts.py'},
    'runs/v2_field_contrasts.json': {'rerun': 'runs/v2_oxygen_mgl/C1/v2_field_contrasts.json', 'part': 'C1/C2', 'producer': 'scripts/v2_oxygen_mgl_field_contrasts.py'},
    'runs/v2_perfold.json': {'rerun': 'runs/v2_oxygen_mgl/C1/v2_perfold.json', 'part': 'C1/C2',
                             'section': 'field57 only (the forecasting cells read the sondes alone)', 'producer': 'scripts/v2_oxygen_mgl_field_contrasts.py'},
    'runs/v2_perfold_rest.json': {'rerun': ['runs/v2_oxygen_mgl/C1/v2_perfold_rest.json (field57)',
                                            'runs/v2_oxygen_mgl/C3/v2_perfold_rest.json (aou)'], 'part': 'C1/C2/C3',
                                  'section': 'field57 and aou (the forecasting families read the sondes alone)',
                                  'producer': 'scripts/v2_oxygen_mgl_field_contrasts.py, scripts/v2_oxygen_mgl_aou_forecast.py',
                                  'aou_section': 'amendment 4 (hourly-factor record: runs/v2_oxygen_mgl/C3/_hourly_factor/v2_perfold_rest.json)'},
    'runs/v2_background_bias.json': {'rerun': 'runs/v2_oxygen_mgl/C1/v2_background_bias.json', 'part': 'C1', 'producer': 'scripts/v2_oxygen_mgl_field_contrasts.py'},
    'runs/v2_field_reserved2026/report.json': {'rerun': 'runs/v2_oxygen_mgl/C2/v2_field_reserved2026/report.json', 'part': 'C2', 'producer': 'scripts/v2_oxygen_mgl_field2026.py'},
    'runs/v2_field_reserved2026_copernicus/report.json': {'rerun': 'runs/v2_oxygen_mgl/C1/v2_field_reserved2026_copernicus/report.json', 'part': 'C1', 'producer': 'scripts/v2_oxygen_mgl_field2026.py'},
    'runs/v2_oxygen_floor.json': {'rerun': 'runs/v2_oxygen_mgl/C2/v2_oxygen_floor.json', 'part': 'C2', 'producer': 'scripts/v2_oxygen_mgl_casts.py'},
    'runs/v2_ctd_floor.json': {'rerun': 'runs/v2_oxygen_mgl/C2/v2_ctd_floor.json', 'part': 'C2', 'producer': 'scripts/v2_oxygen_mgl_casts.py'},
    'runs/v2_metrology_review.json': {'rerun': 'runs/v2_oxygen_mgl/C2/v2_metrology_review.json', 'part': 'C2',
                                      'section': 'G2 oxygen floor only (the rest is temperature, depth or unit-free)', 'producer': 'scripts/v2_oxygen_mgl_casts.py'},
    'runs/v2_ctd_sections.json': {'rerun': 'runs/v2_oxygen_mgl/C2/v2_ctd_sections.json', 'part': 'C2',
                                  'also': 'runs/v2_oxygen_mgl/C2/v2_ctd_sections/ (the two tables and the figure, redirected)', 'producer': 'scripts/v2_oxygen_mgl_casts.py'},
    'runs/eda_opportunities/eda_c2_strat.json': {'rerun': 'runs/v2_oxygen_mgl/C2/eda_opportunities/eda_c2_strat.json', 'part': 'C2', 'producer': 'scripts/v2_oxygen_mgl_casts.py'},
    'runs/v2_placement': {'rerun': 'runs/v2_oxygen_mgl/C2/v2_placement', 'part': 'C2', 'producer': 'scripts/v2_oxygen_mgl_casts.py'},
    'runs/v2_observability_reanalysis.json': {'rerun': 'runs/v2_oxygen_mgl/C2/v2_observability_reanalysis.json', 'part': 'C2',
                                              'section': 'F1 near-bottom and F2 placement (F3 and the warning blocks read the sondes alone)', 'producer': 'scripts/v2_oxygen_mgl_casts.py'},
    'runs/v2_section5_numbers.json': {'rerun': 'runs/v2_oxygen_mgl/C2/v2_section5_numbers.json', 'part': 'C2', 'producer': 'scripts/v2_oxygen_mgl_casts.py'},
    'runs/v2_bottom_aou/bottom/contrasts.json': {'rerun': 'runs/v2_oxygen_mgl/C2/v2_bottom_aou/bottom/contrasts.json', 'part': 'C2/C3', 'producer': 'scripts/v2_oxygen_mgl_bottom.py'},
    'runs/v2_bottom_aou/bottom_ref/contrasts.json': {'rerun': 'runs/v2_oxygen_mgl/C2/v2_bottom_aou/bottom_ref/contrasts.json', 'part': 'C3', 'producer': 'scripts/v2_oxygen_mgl_bottom.py'},
    'runs/v2_bottom_aou/aou_summary.json': {'rerun': 'runs/v2_oxygen_mgl/C3/v2_bottom_aou/aou_summary.json', 'part': 'C3', 'producer': 'scripts/v2_oxygen_mgl_bottom.py'},
    'runs/v2_bottom_aou/aou_units/summary.json': {'rerun': 'runs/v2_oxygen_mgl/C3/v2_bottom_aou/aou_units/summary.json', 'part': 'C3', 'producer': 'scripts/v2_oxygen_mgl_bottom.py'},
    'runs/v2_aou_forecast/': {'rerun': 'runs/v2_oxygen_mgl/C3/v2_aou_forecast/', 'part': 'C3', 'producer': 'scripts/v2_oxygen_mgl_aou_forecast.py',
                              'method': 'amendment 4: sonde oxygen x c, the utilisation arms refitted, references published x c', 'hourly_factor_record': 'runs/v2_oxygen_mgl/C3/_hourly_factor/v2_aou_forecast/'},
    'runs/v2_aou_units_chronos.json': {'rerun': 'runs/v2_oxygen_mgl/C3/v2_aou_units_chronos.json', 'part': 'C3',
                                       'host_columns': 'runs/v2_oxygen_mgl/C3/v2_aou_units_chronos/',
                                       'method': 'amendment 4: sonde oxygen x c, the utilisation arms refitted, references published x c', 'hourly_factor_record': 'runs/v2_oxygen_mgl/C3/_hourly_factor/v2_aou_units_chronos.json',
                                       'producer': 'scripts/experiments/v2_oxygen_mgl_chronos.py, scripts/experiments/v2_oxygen_mgl_chronos_adapted.py '
                                                   '(compute host), scripts/v2_oxygen_mgl_chronos_contrasts.py'},
    'runs/eda_opportunities/eda_b_seaphox.json': {'rerun': 'runs/v2_oxygen_mgl/C4/eda_opportunities/eda_b_seaphox.json', 'part': 'C4', 'producer': 'scripts/v2_oxygen_mgl_seaphox.py'},
}

NOT_RERUN = {
    'runs/v2_salinity_check.json': 'reads the per-litre Garcia-Gordon saturation alone (oxygen enters only as a validity '
                                   'mask): its numbers convert exactly by 0.0319988, not by c',
    'runs/eda_opportunities/eda_b2_lows.json': 'the SeapHOx record alone, whose oxygen column is mg/l per litre (header: '
                                               'mg/l beside ml/l, ratio 1.42905): its oxygen numbers convert by 0.0319988 '
                                               'per micromole, i.e. they are the mg/l column itself (111.3 umol/l = 3.561 mg/l)',
    'Field-57 arms A2-A4 with the utilisation target (runs/20260919T145045Z_v2_field)': 'printed nowhere in the thesis',
    'runs/v2_sensor_axis.json': 'its utilisation rows are read from runs/v2_aou_units_chronos.json (rerun above); its '
                                'other oxygen rows read the sondes alone (part B)',
    'runs/v2_oxygen_floor.json and runs/v2_metrology_review.json (casts of 2024-2025)': 'rerun although nearly per-kilogram '
                                'only, because Field-57 and the floors include the one per-litre (\'%\') cast of 2025-12-16 at PMitiliE',
    'adapted pretrained arms (ft, lora, held-out quarter, refits)': 'retrained in mg/l on the host (see runs/v2_oxygen_mgl/'
                                'C3/v2_aou_units_chronos.json, units mgl); kg_x_factor is shown beside them',
}

REPRODUCTION = {k: f'runs/v2_oxygen_mgl/{v}' for k, v in (
    ('Field-57 arms', 'C1/field57_reproduction.json'), ('Field-57 comparators and contrasts', 'C1/field57_contrasts_reproduction.json'),
    ('Field-2026', 'C1/field2026_reproduction.json'), ('near-bottom and year-boundary', 'C2/bottom_reproduction.json'),
    ('cast-archive descriptive analyses', 'C2/casts_reproduction.json'), ('utilisation forecast', 'C3/aou_forecast_reproduction.json'),
    ('pretrained arms', 'C3/v2_aou_units_chronos.json (gates)'), ('Tinetto check', 'C4/seaphox_reproduction.json'))}

REFIT_NOISE = {
    'what': 'the utilisation forecast rerun with every sonde hour converted by the constant c instead of its own '
            'factor (scratchpad diagnostic, scripts/v2_oxygen_mgl_aou_forecast.run_aou_forecast with '
            'sonde_factor = c): persistence and climatology then equal c x published exactly, but the '
            'L1 LightGBM trees move by up to 2.2 % (weather tree -2.2 %, plain tree -0.6 %, utilisation '
            'tree +0.7 %), because their exact-zero residuals flip sign under rounding; a tree refit in '
            'any unit has this much numerical spread before any unit effect',
    'weather_tree_relative': -0.0223, 'plain_tree_relative': -0.0061, 'aou_tree_relative': 0.0070,
    'thermodynamic_relative': -0.0008,
    'near_bottom_and_year_boundary': {
        'what': 'the near-bottom and year-boundary arms with every oxygen input and the published saturation times '
                'the constant c, a pure change of scale (scratchpad diagnostic over scripts/v2_oxygen_mgl_bottom.Units): '
                'the largest relative departure of an arm\'s MAE from c x published',
        'ridge_arms': 2.2e-15, 'near_bottom_lightgbm': {'C1_cast_surface|lightgbm|direct': 0.0108,
                                                         'C2_moored_only|lightgbm|direct': 0.0030,
                                                         'C1_cast_surface|lightgbm|aou': 0.0003,
                                                         'C1_cast_surface|lightgbm|gradient': 0.0002},
        'year_boundary_lightgbm': 1.0e-08},
    'field57_ridge': 'linear in the target: a pure change of scale reproduces c x published to rounding'}


if __name__ == '__main__':
    main()
