"""Addendum 82 part G, the reading: apparent oxygen utilisation in both cohorts.

Dev-2025 (six issuance grids, addendum 51's oxygen panel rows): each of the eight pretrained arms X forecasts
utilisation (G_oxygen_aou), turned back into oxygen at its own six-hour forecast of the sonde's temperature in
the same mode (G_temperature_raw): oxygen_hat = o2_sol_mgl(T_hat_X, 38) - AOU_hat. Comparators: the same arm on
the sonde's oxygen unstandardised (direct_raw_X, G_oxygen_raw), the published z-scored direct arm (task A), the
tree of each phase (per fold primary, shared beside), the weather tree of each phase (phasefix, seed 42), the
month-by-hour climatological increment (hour-of-day beside), Chronos-Bolt base on utilisation, and at offset 0
the utilisation tree (per fold primary, shared beside). Fold macro MAE in mg/l; paired (fold, ISO week) block
bootstrap, 600 draws, one stream per contrast seeded [20260921, crc32('a82G|<offset>|<x>|<y>')]; a contrast at
all six offsets gets the verdict over phases only (sweep_verdict; the models are deterministic).

Reserved-2026 (the battery's 799 oxygen rows): the frozen utilisation tree (scripts/v2_aou_frozen2026.py) and
its fold-1 replicate, the thermodynamic baseline, against the released model, the month-by-hour increment,
persistence and the task-F utilisation arms; the battery's own resamples replayed (score_6h.BatteryDraws).

Gates: G-G0 (the published Dev tree numbers), G-G1 (addendum 81's task F block), G-G4 (reconstruction),
G-G6 (rows); G-G2 is the input builder's manifest, G-G3 the compute host's check_f.json and determinism.json,
G-G5 the frozen tree's gates.json.

Usage: python -m scripts.v2_aou_symmetric_score
Writes reports/bay_twin/runs/v2_addendum82_G.json and reports/bay_twin/runs/v2_addendum82/G/dev_rows_offset<o>.npz.
Declared in config/bay_reserved_protocol_v2_addendum82.json, part G.
"""
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import scripts.v2_fdr as FDR  # noqa: E402
import scripts.v2_fm_zoo_score_6h as Z  # noqa: E402
from scripts import v2_fm_zoo_io as io  # noqa: E402
from scripts.v2_chronos_lora_contrasts import battery_frame  # noqa: E402
from src.experiments import v2_aou_forecast as A15  # noqa: E402
from src.units import factor  # noqa: E402

RUNS = ROOT / 'reports/bay_twin/runs'
G = RUNS / 'v2_addendum82' / 'G'
GIN = G / 'inputs'
GPRED = G / 'preds'
FROZEN = G / 'frozen2026'
OUT = RUNS / 'v2_addendum82_G.json'
C3 = RUNS / 'v2_oxygen_mgl' / 'C3'
PHASEFIX = RUNS / '20260923T103430Z_v2_phasefix/reports/bay_twin/runs/v2_phase_shift'
SWEEP = RUNS / 'v2_sequence_sweep'
DECLARATION = 'config/bay_reserved_protocol_v2_addendum82.json'
ARMS = io.ARMS
OFFSETS = tuple(range(6))
STEP = 5
CENTER, SCALE = 230.0, 50.0
TC, TS = 20.0, 10.0
SALINITY = 38.0
KG1_CAVEAT = ('The utilisation arm reads the sonde\'s temperature twice (inside utilisation and at the inversion) and '
              'a constant salinity of 38; in uni mode direct_raw reads oxygen only, so the uni contrast includes that '
              'information; the cov contrast, in which both read the sonde\'s temperature, is the '
              'information-matched one.')


def r6(x):
    return Z.r6(x)


def now():
    return datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')


def ns(series):
    return Z.ns(series)


# --------------------------------------------------------------------------------------------------
# Gates G-G0 and G-G1
# --------------------------------------------------------------------------------------------------
def gate_g0():
    frame = pd.read_csv(C3 / 'v2_aou_forecast' / 'aou_forecast_h6.csv.gz')
    pub = json.loads((C3 / 'v2_aou_forecast' / 'aou_forecast_h6_summary.json').read_text())
    worst, n = 0.0, 0
    for a, v in pub['mae'].items():
        worst = max(worst, abs(A15.macro(frame, a) - v))
        n += 1
    for k, v in pub['contrasts'].items():
        a, b = [x.strip() for x in k.split(' - ')]
        got = A15.paired(frame, a, b, f'{a}|{b}')
        worst = max(worst, max(abs(x - y) for x, y in zip(got, v)))
        n += 1
    out = {'published_summary': {'values': n, 'max_abs_diff': worst, 'passed': bool(worst <= 1e-12)}}
    import scripts.v2_perfold_rest_contrasts as Q
    saved = (Q.RUNS, Q.WORK)
    Q.RUNS, Q.WORK = C3, C3 / 'v2_perfold_rest'
    try:
        got = json.loads(json.dumps(Q.aou()))
    finally:
        Q.RUNS, Q.WORK = saved
    pub48 = json.loads((C3 / 'v2_perfold_rest.json').read_text())['aou']
    w48 = 0.0
    for d, m in pub48['mae'].items():
        for a, v in m.items():
            w48 = max(w48, abs(got['mae'][d][a] - v))
    pf = {}
    for e_got, e_pub in zip(got['contrasts'], pub48['contrasts']):
        assert e_got['contrast'] == e_pub['contrast']
        for d in ('published', 'shared', 'perfold'):
            w48 = max(w48, max(abs(x - y) for x, y in zip(e_got[d]['value'], e_pub[d]['value'])))
        pf[e_got['contrast']] = e_got['perfold']['value']
    out['per_fold'] = {'max_abs_diff': w48, 'perfold_values': pf, 'passed': bool(w48 <= 1e-12)}
    out['passed'] = bool(out['published_summary']['passed'] and out['per_fold']['passed'])
    return out


def gate_g1():
    """Addendum 81's 6 h scorer, tasks B and F, on the stored predictions, against score_6h.json's F block."""
    manifest = json.loads((Z.INPUTS / 'manifest_6h.json').read_text())
    bf = Z.task_bf(Z.ARMS, manifest, factor(), True)
    pub = json.loads(Z.OUT.read_text())['F']
    mine = json.loads(json.dumps({k: v for k, v in bf['F'].items() if k != 'gates'}))
    worst, n, mismatch = 0.0, 0, []

    def walk(a, b, path):
        nonlocal worst, n
        if isinstance(a, dict) and isinstance(b, dict):
            for k in set(a) | set(b):
                walk(a.get(k), b.get(k), f'{path}/{k}')
        elif isinstance(a, list) and isinstance(b, list) and len(a) == len(b):
            for i, (x, y) in enumerate(zip(a, b)):
                walk(x, y, f'{path}[{i}]')
        elif isinstance(a, (int, float)) and isinstance(b, (int, float)) and not isinstance(a, bool):
            n += 1
            worst = max(worst, abs(a - b))
            if abs(a - b) > 1e-9:
                mismatch.append(path)
        elif a != b:
            mismatch.append(path)
    walk(mine, pub, '')
    harness = bf['F']['gates']['harness']['oxygen']
    return {'values_compared': n, 'max_abs_diff': worst, 'mismatches': mismatch[:20], 'harness': harness,
            'passed': bool(n > 0 and not mismatch and harness['passed'])}


# --------------------------------------------------------------------------------------------------
# Dev-2025
# --------------------------------------------------------------------------------------------------
def load_median(folder, stem, arm, n, sha):
    v, info = Z.load_arm(stem, arm, n, sha, folder=folder)
    return v, info


def dev_frame(o, c, man):
    """One offset's rows (task A oxygen panel order) with every arm's oxygen forecast in mg/l."""
    pan = pd.read_csv(SWEEP / 'panel' / f'panel_oxygen_offset{o}.csv.gz')
    per = pd.read_csv(SWEEP / 'panel_perfold' / f'panel_oxygen_offset{o}.csv.gz')
    n = len(pan)
    f = pan[['fold', 'sensor', 'time']].copy()
    f['tkey'] = ns(f.time)
    to_mgl = lambda z: (np.asarray(z, float) * SCALE + CENTER) * c
    f['truth'] = to_mgl(pan.truth)
    f['persistence'] = to_mgl(pan.persistence)
    f['clim_hour'] = to_mgl(pan.clim_hour)
    f['tree'] = to_mgl(per.tree_seed42)
    f['tree_shared'] = to_mgl(pan.tree_seed42)
    g6 = {'rows': n, 'arms': {}}
    # the weather tree and the month-by-hour increment of the phase
    ps = pd.read_csv(PHASEFIX / f'phase_shift_oxygen_offset{o}_ratio.csv.gz')
    ps['tkey'] = ns(ps.time)
    j = f[['fold', 'sensor', 'tkey']].merge(ps[['fold', 'sensor', 'tkey', 'truth', 'weather_tree', 'climatology']],
                                            on=['fold', 'sensor', 'tkey'], how='left', validate='one_to_one')
    f['weather_tree'] = j.weather_tree.to_numpy(float) * c
    f['clim_month_hour'] = j.climatology.to_numpy(float) * c
    tdiff = np.abs(j.truth.to_numpy(float) * c - f.truth.to_numpy(float))
    g6['phase_shift_rows_found'] = int(j.weather_tree.notna().sum())
    g6['phase_shift_truth_max_abs_diff_mgl'] = float(np.nanmax(tdiff))
    # task A's published z-scored direct arms
    a_man = json.loads((Z.INPUTS / 'manifest_6h.json').read_text())
    a_sha = a_man['files'][f'A_oxygen_offset{o}.npz']['sha256']
    for arm in ARMS:
        v, info = load_median(Z.PREDS, f'A_oxygen_offset{o}', arm, n, a_sha)
        f[f'z_{arm}'] = to_mgl(v)
    # the new files
    files = man['files']
    n_aou = files[f'G_oxygen_aou_offset{o}.npz']['rows']
    rec = np.load(GIN / f'G_oxygen_aou_offset{o}_reconstruction.npz')
    trow = rec['temperature_row']
    g4 = {}
    for arm in ARMS:
        d, info_d = load_median(GPRED, f'G_oxygen_raw_offset{o}', arm, n, files[f'G_oxygen_raw_offset{o}.npz']['sha256'])
        t, info_t = load_median(GPRED, f'G_temperature_raw_offset{o}', arm, n,
                                files[f'G_temperature_raw_offset{o}.npz']['sha256'])
        u, info_u = load_median(GPRED, f'G_oxygen_aou_offset{o}', arm, n_aou,
                                files[f'G_oxygen_aou_offset{o}.npz']['sha256'])
        g6['arms'][arm] = {'direct_raw': info_d, 'T_raw': info_t, 'aou': info_u}
        f[f'direct_raw_{arm}'] = None if d is None else np.asarray(d, float) * c
        f[f'T_hat_{arm}'] = t
        aou = np.full(n, np.nan)
        if u is not None and t is not None:
            aou[trow] = Z.o2_sol_mgl(np.asarray(t, float)[trow], rec['salinity']) - np.asarray(u, float)
        f[f'aou_{arm}'] = aou
        # G-G4: every utilisation row finds its temperature forecast of the same arm, mode, sonde and issuance
        g4[arm] ={'aou_rows': int(len(trow)), 'aou_rows_with_T_hat': int(np.isfinite(np.asarray(t, float)[trow]).sum())
                   if t is not None else 0}
    return f, g6, g4


def g4_information(o, f, c):
    """T_raw's median against task A's published temperature arm on the same rows, direct_raw against the
    published z-scored direct arm (largest and mean absolute difference)."""
    a_man = json.loads((Z.INPUTS / 'manifest_6h.json').read_text())
    tkeys = io.load_inputs(Z.INPUTS / f'A_temperature_offset{o}.npz')['keys']
    where = {(int(a), str(s), int(t)): i for i, (a, s, t) in enumerate(zip(tkeys.fold, tkeys.sensor.astype(str), tkeys.time))}
    rows = np.array([where[(int(a), str(s), int(t))] for a, s, t in zip(f.fold, f.sensor.astype(str), f.tkey)])
    out = {}
    for arm in ARMS:
        v, _ = Z.load_arm(f'A_temperature_offset{o}', arm, len(tkeys), a_man['files'][f'A_temperature_offset{o}.npz']['sha256'])
        tpub = np.asarray(v, float)[rows] * TS + TC
        dt = np.abs(f[f'T_hat_{arm}'].to_numpy(float) - tpub)
        dd = np.abs(f[f'direct_raw_{arm}'].to_numpy(float) - f[f'z_{arm}'].to_numpy(float))
        out[arm] = {'T_raw_vs_published_temperature_degC': [r6(np.nanmax(dt)), r6(np.nanmean(dt))],
                    'direct_raw_vs_published_z_direct_mgl': [r6(np.nanmax(dd)), r6(np.nanmean(dd))]}
    return out


def dev_pairs():
    out = []
    for X in ARMS:
        a = f'aou_{X}'
        out += [(a, f'direct_raw_{X}', 'KG1'), (a, f'z_{X}', 'KG1b'), (a, 'tree', 'KG2'), (a, 'tree_shared', 'KG2'),
                (a, 'weather_tree', 'KG2'), (a, 'clim_month_hour', 'KG3'), (a, 'clim_hour', 'KG3')]
        if X != 'chronos_bolt_base_uni':
            out.append((a, 'aou_chronos_bolt_base_uni', 'KG4'))
    for m in ('chronos2', 'timesfm25', 'timesfm3'):
        out.append((f'aou_{m}_cov', f'aou_{m}_uni', 'KG4'))
    return out


def dev_scoring(c, man):
    res = {'offsets': {}, 'gates': {'G-G4': {}, 'G-G6': {}}}
    raw = {}
    for o in OFFSETS:
        f, g6, g4 = dev_frame(o, c, man)
        g4['information'] = g4_information(o, f, c)
        res['gates']['G-G4'][f'offset{o}'] = g4
        g6['passed'] = bool(all(all(x['passed'] for x in a.values()) for a in g6['arms'].values())
                            and g6['phase_shift_rows_found'] == g6['rows'] and g6['phase_shift_truth_max_abs_diff_mgl'] <= 1e-9)
        res['gates']['G-G6'][f'offset{o}'] = g6
        cols = (['persistence', 'clim_hour', 'clim_month_hour', 'tree', 'tree_shared', 'weather_tree'] +
                [f'{p}_{X}' for p in ('aou', 'direct_raw', 'z') for X in ARMS])
        if o == 0:
            pf = pd.read_csv(C3 / 'v2_perfold_rest' / 'aou' / 'perfold' / 'aou_forecast_h6.csv.gz')
            sh = pd.read_csv(C3 / 'v2_aou_forecast' / 'aou_forecast_h6.csv.gz')
            for name, src, col in (('aou_tree', pf, 'aou_tree'), ('aou_tree_shared', sh, 'aou_tree'),
                                   ('thermodynamic', sh, 'thermodynamic')):
                s = src[['fold', 'sensor', 'time', 'truth', col]].copy()
                s['tkey'] = ns(s.time)
                jj = f[['fold', 'sensor', 'tkey']].merge(s[['fold', 'sensor', 'tkey', 'truth', col]],
                                                         on=['fold', 'sensor', 'tkey'], how='left', validate='one_to_one')
                f[name] = jj[col].to_numpy(float)
                if name == 'aou_tree':
                    g6['utilisation_tree_rows_on_panel'] = int(jj[col].notna().sum())
                    g6['utilisation_tree_truth_max_abs_diff_mgl'] = float(np.nanmax(np.abs(jj.truth - f.truth)))
            cols += ['aou_tree', 'aou_tree_shared', 'thermodynamic']
        mae = {k: r6(Z.fold_point(f, k)) for k in cols if f[k].notna().any()}
        contrasts = {}
        pairs = dev_pairs() + ([(f'aou_{X}', 'aou_tree', 'KG5') for X in ARMS] +
                               [(f'aou_{X}', 'aou_tree_shared', 'KG5') for X in ARMS] if o == 0 else [])
        for a, b, k in pairs:
            lab = f'a82G|{o}|{a}|{b}'
            p, lo, hi, n_rows, n_units = Z.a_contrast(f, a, b, lab)
            e = Z.entry([k], p, lo, hi, n_rows, n_units, 'oxygen', 'mg/l')
            e['stream'] = lab
            if k == 'KG1':
                e['information'] = 'cov: information-matched' if a.endswith('_cov') else 'uni: includes the sonde temperature the comparator does not read'
            contrasts[f'{a} - {b}'] = e
            if k != 'KG5':
                raw.setdefault(f'{a} - {b}', []).append((p, lo, hi) if n_units >= Z.MIN_UNITS else (p, None, None))
        res['offsets'][str(o)] = {'rows': int(len(f)), 'rows_utilisation': int(f[f'aou_{ARMS[0]}'].notna().sum()),
                                  'mae_mgl': mae, 'contrasts': contrasts}
        keep = {k: f[k].to_numpy(float) for k in ['truth'] + cols}
        np.savez_compressed(G / f'dev_rows_offset{o}.npz', fold=f.fold.to_numpy(), sensor=f.sensor.to_numpy().astype('U'),
                            time=f.tkey.to_numpy(np.int64), **keep)
        print(f'G Dev offset {o}: {len(f)} rows, {len(contrasts)} contrasts', flush=True)
    res['verdicts'] = {k: Z.phase_verdict(cells, 'oxygen', 1.0) for k, cells in raw.items()}
    return res


# --------------------------------------------------------------------------------------------------
# Reserved-2026
# --------------------------------------------------------------------------------------------------
def reserved_scoring(c):
    d = battery_frame()
    d['tkey'] = ns(d.time)
    draws = Z.BatteryDraws(d)
    s0 = d[d.variable == 'oxygen'].reset_index(drop=True)
    for col in ('truth', 'MODEL_v2', 'B0_persistence', 'B1_clim_increment'):
        s0[col] = s0[col].to_numpy(float) * c
    # the task-F utilisation arms, reconstructed as addendum 81's scorer does
    manifest = json.loads((Z.INPUTS / 'manifest_6h.json').read_text())
    rec = np.load(Z.INPUTS / 'F_oxygen_aou_Reserved-2026_reconstruction.npz', allow_pickle=False)
    bt = d[d.variable == 'temperature'].reset_index(drop=True)
    for arm in ARMS:
        tv, _ = Z.load_arm('B_temperature_Reserved-2026', arm, len(bt), manifest['files']['B_temperature_Reserved-2026.npz']['sha256'])
        uv, _ = Z.load_arm('F_oxygen_aou_Reserved-2026', arm, len(s0), manifest['files']['F_oxygen_aou_Reserved-2026.npz']['sha256'])
        s0[f'aou_{arm}'] = Z.o2_sol_mgl(np.asarray(tv)[rec['temperature_row']], rec['salinity']) - uv
    z = np.load(FROZEN / 'predictions_2026.npz')
    if not ((z['sensor'] == s0.sensor.to_numpy().astype('U')).all() and (z['time'] == s0.tkey.to_numpy()).all()):
        raise SystemExit('the frozen predictions are not in the battery oxygen rows order')
    cols = ['aou_tree_2026', 'aou_tree_2026_w1', 'aou_tree_2026_fold1', 'aou_tree_2026_fold1_w1', 'thermodynamic_2026']
    for col in cols:
        s0[col] = z[col]
    mae = {k: r6(Z.battery_point(s0, k)) for k in cols + ['MODEL_v2', 'B0_persistence', 'B1_clim_increment'] +
           [f'aou_{a}' for a in ARMS]}
    pairs = []
    for t in ('aou_tree_2026', 'aou_tree_2026_fold1', 'aou_tree_2026_w1', 'aou_tree_2026_fold1_w1'):
        pairs += [(t, 'MODEL_v2', 'KH1'), (t, 'B1_clim_increment', 'KH2'), (t, 'B0_persistence', 'KH2'),
                  (t, 'aou_chronos_bolt_base_uni', 'KH3'), (t, 'aou_chronos2_cov', 'KH3')]
    pairs += [('thermodynamic_2026', 'B0_persistence', 'KH4'), ('thermodynamic_2026', 'B1_clim_increment', 'KH4'),
              ('thermodynamic_2026', 'MODEL_v2', 'KH4')]
    contrasts = {}
    for a, b, k in pairs:
        p, lo, hi, n_rows, n_units = draws.contrast(s0, 'oxygen', a, b)
        contrasts[f'{a} - {b}'] = Z.entry([k], p, lo, hi, n_rows, n_units, 'oxygen', 'mg/l')
    rows = {k: int(s0[k].notna().sum()) for k in cols}
    return {'rows': int(len(s0)), 'rows_predicted': rows, 'mae_mgl': mae, 'contrasts': contrasts}


# --------------------------------------------------------------------------------------------------
def family(dev, res26):
    entries = []
    for o, b in dev['offsets'].items():
        for k, e in b['contrasts'].items():
            entries.append((('Dev-2025', o, k), e))
    for k, e in res26['contrasts'].items():
        entries.append((('Reserved-2026', None, k), e))
    ps, keep = [], []
    for key, e in entries:
        p, lo, hi = e['value']
        if lo is None:
            continue
        q = FDR.p_side(p, lo, hi)
        e['p'] = float(f'{(q if q is not None else FDR.p_mid(lo, hi)):.9g}')
        ps.append(e['p'])
        keep.append(e)
    bh, by = FDR.step_up(ps), FDR.step_up(ps, dependence_robust=True)
    for e, a, b in zip(keep, bh, by):
        e['bh'], e['by'] = bool(a), bool(b)
    return {'q': FDR.Q, 'intervals': len(ps), 'excluding_zero': int(sum(e['status'] == 'excludes zero' for e in keep)),
            'bh_retained': int(sum(bh)), 'by_retained': int(sum(by)),
            'p_rule': 'scripts/v2_fdr.p_side; p_mid where the estimate lies outside its own interval'}


def main():
    t0 = time.time()
    c = factor()
    man = json.loads((GIN / 'manifest_aou_dev.json').read_text())
    gates = {'G-G2': {'passed': man['passed'], 'source': 'reports/bay_twin/runs/v2_addendum82/G/inputs/manifest_aou_dev.json',
                      'note': man['gate_tolerance_note']}}
    for name in ('check_f.json', 'determinism.json'):
        p = G / name
        gates[f'G-G3|{name}'] = json.loads(p.read_text()) if p.exists() else {'passed': None, 'missing': True}
    g3 = gates['G-G3|check_f.json'].get('passed') and gates['G-G3|determinism.json'].get('all_below_1e-6')
    gates['G-G3'] = {'passed': bool(g3)}
    fz = json.loads((FROZEN / 'gates.json').read_text())
    gates['G-G5'] = {k: fz['gates'][k] for k in ('G-G5a', 'G-G5b', 'G-G5c', 'G-G5d', 'G-G5e', 'G-G5')}
    gates['G-G0'] = gate_g0()
    print('G-G0', gates['G-G0']['passed'], flush=True)
    gates['G-G1'] = gate_g1()
    print('G-G1', gates['G-G1']['passed'], f'{time.time() - t0:.0f} s', flush=True)
    dev_ok = gates['G-G2']['passed'] and gates['G-G3']['passed'] and gates['G-G0']['passed'] and gates['G-G1']['passed']
    res26_ok = fz['gates']['G-G5']['passed'] and gates['G-G1']['passed']
    dev = dev_scoring(c, man) if dev_ok else None
    if dev is not None:
        gates['G-G4'] = dev['gates']['G-G4']
        gates['G-G6'] = dev['gates']['G-G6']
        dev_ok = all(g['passed'] for g in gates['G-G6'].values())
    res26 = reserved_scoring(c) if res26_ok else None
    fam = family(dev, res26) if (dev is not None and res26 is not None) else None
    rep = {'declared_in': DECLARATION, 'declaration_sha256': io.digest(ROOT / DECLARATION), 'part': 'G',
           'producer': 'scripts/v2_aou_symmetric_score.py', 'producer_sha256': io.digest(Path(__file__)),
           'created_utc': now(), 'c_mg_per_l_per_umol_kg': c, 'units': 'mg/l throughout',
           'floor_mg_per_l': Z.FLOOR['oxygen_mgl'], 'KG1_caveat': KG1_CAVEAT,
           'gates': gates, 'stopped': {'Dev-2025': not dev_ok, 'Reserved-2026': not res26_ok},
           'dev_2025': ({k: v for k, v in dev.items() if k != 'gates'} if dev is not None else None),
           'reserved_2026': res26, 'frozen_tree': {k: fz[k] for k in ('configurations', 'blend_weights',
                                                                      'calibration_grid_scores', 'supported_sensors')},
           'multiplicity': fam, 'seconds': round(time.time() - t0, 1)}
    OUT.write_bytes((json.dumps(rep, indent=1, default=lambda o: o.item() if hasattr(o, 'item') else str(o)) + '\n')
                    .encode('utf-8'))
    print('wrote', OUT, rep['stopped'], f'{rep["seconds"]} s')


def sensitivity_scoring(c):
    """Amendment 1, sensitivity_added_after_reading: the 2026 utilisation tree with the Dev-2025 tree's input set
    (scripts/v2_aou_frozen2026.py --sensitivity), with the contrasts of the primary 2026 reading on the same
    battery draws. Added to v2_addendum82_G.json; nothing else in that file is recomputed."""
    d = battery_frame()
    d['tkey'] = ns(d.time)
    draws = Z.BatteryDraws(d)
    s0 = d[d.variable == 'oxygen'].reset_index(drop=True)
    for col in ('truth', 'MODEL_v2', 'B0_persistence', 'B1_clim_increment'):
        s0[col] = s0[col].to_numpy(float) * c
    manifest = json.loads((Z.INPUTS / 'manifest_6h.json').read_text())
    rec = np.load(Z.INPUTS / 'F_oxygen_aou_Reserved-2026_reconstruction.npz', allow_pickle=False)
    bt = d[d.variable == 'temperature'].reset_index(drop=True)
    for arm in ('chronos_bolt_base_uni', 'chronos2_cov'):
        tv, _ = Z.load_arm('B_temperature_Reserved-2026', arm, len(bt), manifest['files']['B_temperature_Reserved-2026.npz']['sha256'])
        uv, _ = Z.load_arm('F_oxygen_aou_Reserved-2026', arm, len(s0), manifest['files']['F_oxygen_aou_Reserved-2026.npz']['sha256'])
        s0[f'aou_{arm}'] = Z.o2_sol_mgl(np.asarray(tv)[rec['temperature_row']], rec['salinity']) - uv
    folder = FROZEN / 'sensitivity_dev_inputs'
    z = np.load(folder / 'predictions_2026.npz')
    if not ((z['sensor'] == s0.sensor.to_numpy().astype('U')).all() and (z['time'] == s0.tkey.to_numpy()).all()):
        raise SystemExit('the sensitivity predictions are not in the battery oxygen rows order')
    cols = ['aou_tree_2026_devinputs', 'aou_tree_2026_devinputs_w1', 'aou_tree_2026_fold1_devinputs',
            'aou_tree_2026_fold1_devinputs_w1']
    for col in cols:
        s0[col] = z[col]
    contrasts = {}
    for t in cols:
        for b, k in (('MODEL_v2', 'KH1'), ('B1_clim_increment', 'KH2'), ('B0_persistence', 'KH2'),
                     ('aou_chronos_bolt_base_uni', 'KH3'), ('aou_chronos2_cov', 'KH3')):
            p, lo, hi, n_rows, n_units = draws.contrast(s0, 'oxygen', t, b)
            e = Z.entry([k], p, lo, hi, n_rows, n_units, 'oxygen', 'mg/l')
            q = FDR.p_side(p, lo, hi)
            e['p'] = float(f'{(q if q is not None else FDR.p_mid(lo, hi)):.9g}')
            contrasts[f'{t} - {b}'] = e
    g = json.loads((folder / 'gates.json').read_text())
    return {'declared_in': 'config/bay_reserved_protocol_v2_addendum82_amendment1.json',
            'status': 'a sensitivity added after the first reading; the declared reading (reserved_2026) stays primary',
            'what': ('the 2026 utilisation tree with the Dev-2025 tree\'s input set: filled value, neighbour mean and '
                     'fallback built from oxygen; target and anchor utilisation in mg/l; same configurations, refit, '
                     'blend procedure, inference loop and inversion at the released temperature forecast'),
            'producer': 'scripts/v2_aou_frozen2026.py --sensitivity', 'gates': g['gates'], 'blend': g['blend'],
            'rows': int(len(s0)), 'mae_mgl': {k: r6(Z.battery_point(s0, k)) for k in cols}, 'contrasts': contrasts,
            'multiplicity': 'not part of the declared family of part G (added after reading); p-values stored'}


def add_sensitivity():
    rep = json.loads(OUT.read_text())
    rep['sensitivity_added_after_reading'] = sensitivity_scoring(factor())
    OUT.write_bytes((json.dumps(rep, indent=1, default=lambda o: o.item() if hasattr(o, 'item') else str(o)) + '\n')
                    .encode('utf-8'))
    print('added the sensitivity to', OUT)


if __name__ == '__main__':
    if '--sensitivity' in sys.argv:
        add_sensitivity()
    else:
        main()
