"""Addendum 82, part H: the seabed features at a held-out site, every arm with its own studies.

Arms, each selected per (fold, held site) without the held site (src/experiments/v2_held_site_nested.py):
HB the seabed tree (weather tree plus the three seabed features of v2_bathy), HP the plain tree,
HW the weather tree (temperature: the published nested run; oxygen: part D's arm). References: the
all-site seabed, weather and plain trees selected per outer fold (addenda 47 and 48).

  H1  HB - HW  the seabed features at an unseen site (principal)
  H2  HB - HP
  H3  HW - HP  so that H2 can be read
  H4  HB - AB  the transfer penalty with the features, beside part D's HW - AW
  H5  (HB - AB) - (HW - AW), computed in every draw

Intervals as part D (scripts/v2_transfer_perfold.py, whose scheme functions are used): temperature whole
sites with the jackknife and joint weeks beside; oxygen joint weeks with the sondes fixed, site level n.e.
Labels 'a82|H|...'. The support count (H-G4): how many distinct values each seabed feature takes over
the probes each held-site model trains on. The sweep trigger: H1 or H5 excluding zero on its primary
scheme on either variable. Writes reports/bay_twin/runs/v2_addendum82_H.json.
Usage: python -m scripts.v2_seabed_site
"""
import json

import numpy as np
import pandas as pd

import scripts.v2_transfer_perfold as D

RUNS = D.RUNS
A82 = D.A82
OUT = RUNS / 'v2_addendum82_H.json'
KEY = D.KEY
CONTRASTS = [('H1', [('HB', 1.0), ('HW', -1.0)]), ('H2', [('HB', 1.0), ('HP', -1.0)]),
             ('H3', [('HW', 1.0), ('HP', -1.0)]), ('H4', [('HB', 1.0), ('AB_perfold', -1.0)]),
             ('H5', [('HB', 1.0), ('AB_perfold', -1.0), ('HW', -1.0), ('AW_perfold', 1.0)]),
             ('D_penalty_beside', [('HW', 1.0), ('AW_perfold', -1.0)])]


def frames(o=0, s=42):
    published = D.published_frames()
    out = {}
    for variable in ('temperature', 'oxygen'):
        base = published[variable][KEY + ['site', 'truth', 'persistence']]
        if variable == 'temperature' and (o, s) == (0, 42):
            hw = published['temperature'][KEY + ['held_station_nested']].rename(columns={'held_station_nested': 'HW'})
        else:
            tag = f'{variable}_weather_tree_h6_o{o}_s{s}'
            hw = D.combine_cells(A82 / 'D/cells' / tag, base)[KEY + [tag]].rename(columns={tag: 'HW'})
        f = base.merge(hw, on=KEY, validate='one_to_one')
        for arm, col in (('bathy_tree', 'HB'), ('tree', 'HP')):
            tag = f'{variable}_{arm}_h6_o{o}_s{s}'
            c = D.combine_cells(A82 / 'H/cells' / tag, base)
            m = c[KEY + ['truth']].merge(base[KEY + ['truth']], on=KEY, suffixes=('', '_b'), validate='one_to_one')
            if len(m) != len(base) or not (m.truth == m.truth_b).all():
                raise SystemExit(f'{tag}: truth differs from the published frame')
            f = f.merge(c[KEY + [tag]].rename(columns={tag: col}), on=KEY, validate='one_to_one')
        refs = {'AB_perfold': RUNS / f'v2_perfold_rest/forecast/perfold/bathy_{variable}_bathy_tree_h6',
                'AW_perfold': RUNS / f'v2_perfold/forecast/perfold/{variable}_weather_tree_h6',
                'AP_perfold': RUNS / f'v2_perfold/forecast/perfold/{variable}_tree_h6'}
        for col, folder in refs.items():
            r = pd.read_csv(folder / 'selected_predictions.csv.gz')
            f = f.merge(r[KEY + ['prediction']].rename(columns={'prediction': col}), on=KEY, validate='one_to_one')
        out[variable] = f
    return out


def support(variable):
    """H-G4: distinct values of each seabed feature over the probes each held-site model trains on."""
    from src.experiments import v2_held_site_nested as M
    from src.experiments.v2_horizon import folds
    data = M.build_data(variable, 0)
    sites, _ = M.geometry(data['names'])
    block = M.bathymetric_block(data)[0]
    out = {}
    for fold in folds(data, 6):
        tr_valid = data['valid'][fold['tr']].sum(0) > 0
        for site in pd.unique(sites[fold['eligible']]):
            train = fold['eligible'] & (sites != site) & tr_valid
            out[f'{fold["number"]}|{site}'] = {
                'training_probes': int(train.sum()),
                'distinct_values': {n: int(len(np.unique(block[train, k])))
                                    for k, n in enumerate(['seabed_m', 'clearance_m', 'relative_depth'])}}
    return out


def sweep_frame(variable, o, s):
    tag = f'h6_o{o}_s{s}'
    aw = pd.read_csv(A82 / f'D/allsite/{variable}_weather_tree_{tag}/predictions.csv.gz')
    ab = pd.read_csv(A82 / f'H/allsite/{variable}_bathy_tree_{tag}/predictions.csv.gz')
    sites = D.site_map()
    f = aw[KEY + ['truth', 'persistence', 'prediction']].rename(columns={'prediction': 'AW_perfold'})
    f['site'] = f.sensor.map(sites)
    m = f[KEY + ['truth']].merge(ab[KEY + ['truth']], on=KEY, suffixes=('', '_b'), validate='one_to_one')
    if len(m) != len(f) or len(ab) != len(f) or not (m.truth == m.truth_b).all():
        raise SystemExit(f'{variable} {tag}: the all-site seabed and weather frames differ')
    f = f.merge(ab[KEY + ['prediction']].rename(columns={'prediction': 'AB_perfold'}), on=KEY, validate='one_to_one')
    for part, arm, col in (('D', 'weather_tree', 'HW'), ('H', 'bathy_tree', 'HB'), ('H', 'tree', 'HP')):
        name = f'{variable}_{arm}_{tag}'
        c = D.combine_cells(A82 / part / 'cells' / name, f)
        f = f.merge(c[KEY + [name]].rename(columns={name: col}), on=KEY, validate='one_to_one')
    return f


def sweep(report, floors):
    """H1 and H5 at the seven further replicates, and their verdicts over the eight."""
    from src.experiments.sweep_verdict import label as vlabel, verdict
    out, members = {'replicates': {}, 'verdicts': {}}, []
    for variable in ('temperature', 'oxygen'):
        cells = {'H1': [], 'H5': []}
        for name in cells:
            r = report['reference_grid'][variable]['contrasts'][name]
            cells[name].append(((0, 42), (r['point'], *r['decision']['ci'])))
        for o, s in D.REPLICATES:
            frame = sweep_frame(variable, o, s)
            entry = {'endpoints': int(len(frame)), 'sites': int(frame.site.nunique()),
                     'held_site_cells': {arm: D.records(A82 / f'H/cells/{variable}_{arm}_h6_o{o}_s{s}')
                                         for arm in ('bathy_tree', 'tree')},
                     'mae': D.mae_table(frame, ['HB', 'HW', 'HP', 'AB_perfold', 'AW_perfold']), 'contrasts': {}}
            for name, combo in CONTRASTS:
                if name not in cells:
                    continue
                r = D.contrast(frame, combo, f'a82|H|sweep|o{o}|s{s}|{variable}|{name}', variable, floors[variable])
                entry['contrasts'][name] = r
                members += D.family_members(f'sweep|o{o}|s{s}|{variable}|{name}', r, variable)
                cells[name].append(((o, s), (r['point'], *r['decision']['ci'])))
            out['replicates'][f'{variable}|o{o}|s{s}'] = entry
        for name, cs in cells.items():
            d = dict(cs)
            seeds = [d[(0, 42)], d[(0, 7)], d[(0, 123)]]
            phases = [d[(o, 42)] for o in range(6)]
            code = verdict(seeds, phases)
            pts = [p for p, _, _ in phases]
            out['verdicts'][f'{variable}|{name}'] = {
                'code': code, 'label': vlabel(code),
                'seeds': [[D.r6(x) for x in c_] for c_ in seeds], 'phases': [[D.r6(x) for x in c_] for c_ in phases],
                'phase_mean': D.r6(np.mean(pts)), 'phase_range': D.r6(max(pts) - min(pts)),
                'cells_excluding_zero': int(sum(D.excludes(c_[1:]) for _, c_ in cs))}
    return out, members


def main():
    c = D.c_mgl()
    floors = {'temperature': D.FLOOR_T, 'oxygen': 0.059 / c}
    report = {'declared_in': D.DECLARED, 'part': 'H', 'draws': D.DRAWS,
              'units': {'temperature': 'degC', 'oxygen': 'umol/kg as stored, with mg/l copies by c = %.6f' % c},
              'floors': {'temperature_degC': D.FLOOR_T, 'oxygen_mgl': 0.059},
              'gates': {}}
    report['gate_notes'] = ['A first local gate run compared design arrays with numpy.array_equal, which counts '
                           'NaN as unequal (the weather channels have missing hours) and so reported H-G1 and '
                           'H-G2 as failed while every prediction agreed to 6e-14; the gates were rerun with a '
                           'bit-for-bit comparison and passed. The rerun overwrote the record of the first run.']
    local = json.loads((A82 / 'gates_local.json').read_text())
    for k in ('H-G1_seabed_block', 'H-G2_designs', 'H-G3_references', 'D-G3_inner_window'):
        report['gates'][k] = {'pass': local[k]['pass'], 'source': 'reports/bay_twin/runs/v2_addendum82/gates_local.json'}
    report['gates']['D-G2_study_reproduction'] = json.loads((A82 / 'D/gate_DG2.json').read_text())
    report['gates']['H-G2_arm_identity_weather'] = {'pass': local['D-G1_refit']['pass'],
                                                    'note': 'the weather arm reproduces held_station_nested (D-G1)'}
    report['gates']['H-G4_support'] = {'pass': True, 'reported_not_a_stop': True,
                                       'temperature': support('temperature'), 'oxygen': support('oxygen')}
    if not all(g['pass'] for g in report['gates'].values()):
        OUT.write_bytes(json.dumps(report, indent=1).encode())
        raise SystemExit('a gate failed; part H is not read')
    fs = frames()
    family = []
    report['reference_grid'] = {}
    for variable, frame in fs.items():
        block = {'endpoints': int(len(frame)), 'sites': int(frame.site.nunique()),
                 'mae': D.mae_table(frame, ['HB', 'HW', 'HP', 'AB_perfold', 'AW_perfold', 'AP_perfold', 'persistence']),
                 'selections': {'HB': D.records(A82 / f'H/cells/{variable}_bathy_tree_h6_o0_s42'),
                                'HP': D.records(A82 / f'H/cells/{variable}_tree_h6_o0_s42')},
                 'contrasts': {}}
        for name, combo in CONTRASTS:
            # the part-D penalty printed beside H4 draws under part D's own label, so the two agree exactly
            label = (f'a82|D|ref|{variable}|HW|AW_perfold' if name == 'D_penalty_beside'
                     else f'a82|H|ref|{variable}|{name}')
            r = D.contrast(frame, combo, label, variable, floors[variable])
            block['contrasts'][name] = r
            if name != 'D_penalty_beside':
                family += D.family_members(f'ref|{variable}|{name}', r, variable)
            print(variable, name, r['point'], r['weeks']['ci'], r.get('sites', {}).get('ci'),
                  r.get('jackknife', {}).get('ci'), r['status'], flush=True)
        block['per_site'] = {name: D.per_site(frame, combo[0][0], combo[1][0])
                             for name, combo in CONTRASTS[:4]}
        report['reference_grid'][variable] = block
    fired = {}
    for variable in fs:
        for name in ('H1', 'H5'):
            r = report['reference_grid'][variable]['contrasts'][name]
            primary = r['robust_over_sites'] if variable == 'temperature' else r['weeks']['excludes_zero']
            fired[f'{variable}|{name}'] = bool(primary)
    report['sweep_trigger'] = {'rule': 'H1 or H5 excludes zero on its primary scheme (temperature: both the site '
                                       'bootstrap and the jackknife; oxygen: weeks with the sondes fixed) on the '
                                       'reference grid, on either variable',
                               'cells': fired, 'fired': bool(any(fired.values()))}
    report['sweep'] = None
    if report['sweep_trigger']['fired']:
        if all((A82 / f'H/allsite/{v}_bathy_tree_h6_o{o}_s{s}/predictions.csv.gz').exists()
               for v in fs for o, s in D.REPLICATES):
            report['sweep'], members = sweep(report, floors)
            family += members
        else:
            report['sweep'] = 'pending: the triggered replicates have not all been fetched'
    bh = D.step_up([m['p'] for m in family])
    by = D.step_up([m['p'] for m in family], dependence_robust=True)
    for m, a, b in zip(family, bh, by):
        m['bh'] = bool(a and m['excludes_zero'])
        m['by'] = bool(b and m['excludes_zero'])
    report['multiplicity'] = {
        'family': 'part H: every interval on a difference computed by this part (H1-H5 on both variables; the '
                  'part-D penalty printed beside H4 belongs to part D\'s family)',
        'p_value': 'addendum 79: from the estimate and the interval end towards zero; for the hull of the site '
                   'bootstrap and the jackknife the larger of the two',
        'tests': len(family), 'exclude_zero': int(sum(m['excludes_zero'] for m in family)),
        'bh_retained': int(sum(m['bh'] for m in family)), 'by_retained': int(sum(m['by'] for m in family)),
        'members': family}
    report['compute'] = D.compute_record()
    OUT.write_bytes(json.dumps(report, indent=1, allow_nan=False).encode())
    print('wrote', OUT, 'trigger', report['sweep_trigger'])


if __name__ == '__main__':
    main()
