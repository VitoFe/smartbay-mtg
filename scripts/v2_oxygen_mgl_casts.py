"""Addendum 60 part C2: the descriptive analyses of the 722-cast archive, with oxygen in mg/l.

The cast archive mixes 219 per-kilogram casts with 502 per-litre ones (and one '%' cast, per litre),
so a statistic pooled over casts is not a constant multiple of its mg/l form. Each producer below
is rerun unchanged except for the unit of what it reads (scripts/v2_oxygen_mgl_common.py), first in
the published unit, where it must give its published artifact, then in mg/l:

  v2_oxygen_floor        cast against co-located sonde (raw sonde readings by their hour's factor)
  v2_ctd_floor           cast against cast, minutes apart
  v2_metrology_review    the cast-against-sonde floor without the adjusted sonde-months (its oxygen
                         block only; the native column is per litre and is read in mg/l as written)
  v2_ctd_sections        the sections and the stratification statistics (percent saturation and
                         solubility from the per-litre Garcia-Gordon form in mg/l)
  eda_c2_strat           cast stratification against the bottom deficit
  exp_placement          the placement experiment (horizontal and vertical)
  v2_observability_reanalysis   its near-bottom (F1) and placement (F2) blocks
  v2_section5_numbers    the observability block and the stratified naive error

Oxygen thresholds and selection margins written in micromoles enter in mg/l as addendum 60 part D
states them: hypoxia 2 mg/l, the sub-lethal 2.61 and 5.00 mg/l, 6 mg/l; a 2 umol/L prediction
threshold and the 20 umol/L stratified-cast selection by c. Figures and the thesis tables that some
of these producers also write are redirected into the run folder, never into thesis_template/.
The six-decimal rounding of the mg/l outputs replaces the published two to four where the
published one would leave fewer than three significant figures.

Outputs under reports/bay_twin/runs/v2_oxygen_mgl/C2/, each named as the artifact it mirrors, and
C2/casts_reproduction.json.

Usage: python -m scripts.v2_oxygen_mgl_casts [ctd_sections]
"""
import builtins
import contextlib
import json
import runpy
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.v2_oxygen_mgl_common import (OUT, ROOT, RUNS, constant_c, harmonised, lf_rewrite, sat_mgl, sonde_factor,
                                           write_csv, write_json)
from scripts.v2_oxygen_mgl_field_contrasts import same

C2 = OUT / 'C2'
SCRATCH = OUT / '_reproduction' / 'casts'
THRESHOLDS_MGL = {'hypoxia_62.5': 2.0, 'mean_sublethal_82': 2.61, 'sublethal_p90_156': 5.00, 'six_mg_l_186': 6.0}


def finer(x, n=None):
    return builtins.round(x, None if n is None else n + 2)


@contextlib.contextmanager
def patched(obj, **attrs):
    saved = {k: obj.__dict__[k] for k in attrs if k in obj.__dict__}
    for k, v in attrs.items():
        setattr(obj, k, v)
    try:
        yield
    finally:
        for k in attrs:
            if k in saved:
                setattr(obj, k, saved[k])
            else:
                delattr(obj, k)


def hour_label(index):
    """read_export's hour label of a raw reading: bins closed on the left, labelled at their end."""
    idx = pd.DatetimeIndex(index)
    return idx.floor('h') + pd.Timedelta(hours=1)


def relf(path):
    path = Path(path)
    write_json(path, json.loads(path.read_text()))


# ---------------------------------------------------------------- the floors

def oxygen_floor(out, mgl):
    from src.experiments import v2_oxygen_floor as O
    original = O.raw_readings

    def raw_mgl(sonde):
        r = original(sonde)
        r = r.copy()
        r['oxygen'] = r.oxygen.to_numpy(float) * sonde_factor(sonde, hour_label(r.index))
        return r
    extra = dict(OUT=out)
    if mgl:
        extra.update(raw_readings=raw_mgl, PREDICTED_OXYGEN_ABOVE=2.0 * constant_c(), round=finer)
    with patched(O, **extra), harmonised(mgl):
        O.main()
    relf(out)
    return json.loads(out.read_text())


def ctd_floor(out, mgl):
    from scripts import v2_ctd_floor as F
    extra = dict(OUT=out)
    if mgl:
        extra['round'] = finer
    with patched(F, **extra), harmonised(mgl):
        F.main()
    relf(out)
    return json.loads(out.read_text())


def metrology_oxygen(out, mgl, declared):
    """The oxygen block of v2_metrology_review (G2): the floor with and without adjusted sonde-months."""
    from scripts import v2_metrology_review as M
    from src.experiments import v2_oxygen_floor as O
    adjustment, ratios = {}, {}
    for s in O.SITES:                       # the read/native ratio is unit-free: flags as published
        adjustment[s], ratios[s] = M.sonde_adjustment(s)
    original = M.oxygen_columns

    def columns_mgl(sonde):
        f, read, mg, native = original(sonde)
        return f, read, mg, mg                  # the native reading is per litre: in mg/l it is mg
    extra = {}
    if mgl:
        extra.update(oxygen_columns=columns_mgl, DECLARED_OXYGEN_FLOOR=declared)
    with patched(M, **extra), harmonised(mgl):
        block = M.floor_without_flagged(adjustment, ratios)
    report = {'declared_in': 'config/bay_reserved_protocol_v2_addendum59.json (part_G_metrology)',
              'G2_sonde_adjustment': {'oxygen_floor': block, 'PG2': block['PG2'],
                                      'months_flagged': {s: a['months_flagged'] for s, a in adjustment.items()}}}
    write_json(out, json.loads(json.dumps(report, default=float)))
    return report


# ---------------------------------------------------------------- the sections and stratification

def ctd_sections(folder, mgl):
    from scripts import v2_ctd_sections as S
    folder.mkdir(parents=True, exist_ok=True)
    extra = dict(OUT=folder / 'v2_ctd_sections.json', FIGURE=folder / 'fig_bay_ctd_sections.png',
                 TABLE=folder / 'table_bay_ctd_stratification.tex',
                 STATION_TABLE=folder / 'table_bay_ctd_stations.tex')
    if mgl:
        extra['round'] = finer
    # The decline episodes it pairs with casts are the published set (addendum 37), found on the sondes
    # with the decline threshold in umol/L; they are found first, in that unit, and handed back.
    import scripts.v2_decline_types as types
    episodes = types.episodes(*types.DEV)
    with patched(S, **extra), patched(types, episodes=lambda *a, **k: episodes), harmonised(mgl):
        S.main()
    relf(folder / 'v2_ctd_sections.json')
    for name in ('table_bay_ctd_stratification.tex', 'table_bay_ctd_stations.tex'):
        lf_rewrite(folder / name)
    return json.loads((folder / 'v2_ctd_sections.json').read_text())


def strat(folder, mgl):
    sys.path.insert(0, str(ROOT / 'scripts/eda_opportunities'))
    import eda_lib
    folder.mkdir(parents=True, exist_ok=True)
    init = {'round': finer} if mgl else {}
    with patched(eda_lib, OUT=folder), harmonised(mgl):
        runpy.run_path(str(ROOT / 'scripts/eda_opportunities/eda_c2_strat.py'), init_globals=init,
                       run_name='__main__')
    relf(folder / 'eda_c2_strat.json')
    return json.loads((folder / 'eda_c2_strat.json').read_text())


# ---------------------------------------------------------------- placement

def placement(folder, mgl):
    """exp_placement's __main__ block, with its output folder redirected."""
    folder.mkdir(parents=True, exist_ok=True)
    from scripts.experiments import exp_placement as P
    extra = dict(OUT=folder, RNG=np.random.default_rng(20260919))
    if mgl:                                   # stored beside the pairs, not used by either analysis
        extra['o2_sol'] = lambda T, S=38.0: sat_mgl(np.clip(np.asarray(T, float), -2, 40), S)
    with patched(P, **extra), harmonised(mgl):
        init = {}
        code = (ROOT / 'scripts/experiments/exp_placement.py').read_text(encoding='utf-8')
        main = code[code.index('if __name__ == "__main__":'):]
        glb = dict(P.__dict__)
        glb['__name__'] = '__main__'
        glb.update(init)
        exec(compile(main, str(ROOT / 'scripts/experiments/exp_placement.py'), 'exec'), glb)
    for p in folder.glob('*.csv'):
        pd.read_csv(p).to_csv(p, index=False, lineterminator='\n')
    relf(folder / 'placement_summary.json')
    return {p.name: pd.read_csv(p) for p in folder.glob('*.csv')}


# ---------------------------------------------------------------- observability and section 5

def scaled_bottom(src, dst):
    """The near-bottom predictions with abs_delta in units of c, so that the published 20 umol/L
    selection selects the casts whose surface and bottom differ by 20 x c mg/l (part D)."""
    for sub in ('bottom', 'bottom_ref'):
        (dst / sub).mkdir(parents=True, exist_ok=True)
        for name in ('contrasts.json',):
            shutil.copyfile(src / sub / name, dst / sub / name)
        p = pd.read_csv(src / sub / 'predictions.csv.gz', dtype={'fold': str})
        if 'abs_delta' in p:
            p['abs_delta'] = p.abs_delta / constant_c()
        write_csv(p, dst / sub / 'predictions.csv.gz')
    return dst


def observability(out, mgl, bottom_root, placement_dir):
    from scripts import v2_observability_reanalysis as R
    from scripts import v2_summarize_aou as SA
    import src.experiments.v2_field_reserved2026 as FR
    from src.experiments import v2_bottom_aou as vr
    bottom = scaled_bottom(bottom_root, SCRATCH / 'bottom_scaled') if mgl else bottom_root
    extra = dict(BOTTOM=bottom / 'bottom', BOTTOM_REF=bottom / 'bottom_ref', PLACEMENT=placement_dir)
    eb, _ = vr.import_experiments(SCRATCH / 'stage')     # the module R.near_bottom imports again
    if mgl:
        extra['round'] = finer
    rounding = (patched(SA, round=finer), patched(FR, round=finer)) if mgl else ()
    with contextlib.ExitStack() as stack:
        if mgl:      # saturation of exp_bottom.build in mg/l, as scripts/v2_oxygen_mgl_bottom.py sets it
            stack.enter_context(patched(eb, o2_solubility=lambda T, S=38.0: sat_mgl(np.asarray(T, float),
                                                                                   np.asarray(S, float))))
        stack.enter_context(patched(R, **extra))
        for r in rounding:
            stack.enter_context(r)
        stack.enter_context(harmonised(mgl))
        report = {'declared_in': 'config/bay_reserved_protocol_v2_addendum59.json (part F)',
                  'F1_near_bottom': R.near_bottom(), 'F2_placement': R.placement()}
    if mgl:
        report['harmonised'] = ('F1 and F2 only (F3 and the warning blocks read the sondes alone, per kilogram); '
                                'the 20 umol/L stratified selection is applied as 20 x c mg/l')
    report = json.loads(json.dumps(report, default=str))
    write_json(out, report)
    return report


def section5(out, mgl, bottom_root):
    from scripts import v2_section5_numbers as N
    from scripts import v2_field_contrasts as fc
    extra = dict(OUT=out, BOTTOM=bottom_root / 'bottom/predictions.csv.gz')
    stack = contextlib.ExitStack()
    if mgl:
        from scripts.v2_oxygen_mgl_field_contrasts import sources, C1
        from scripts import v2_oxygen_mgl_aou_forecast as AF
        import src.experiments.v2_horizon as H
        mgl_sources = sources(C1 / '20260919T145045Z_v2_field/results', C1 / 'results', C2 / 'results', C1 / 'v2_field_naive')
        extra.update(THRESHOLDS=THRESHOLDS_MGL, round=finer)
        threshold = 20.0 * constant_c()

        def stratified_mgl():
            b = pd.read_csv(N.BOTTOM, dtype={'fold': str})
            s = b[(b.protocol == 'P1_held_station') & (b.model == 'B_surface') & (b.abs_delta >= threshold)]
            e = (s.pred - s.truth).abs()
            return {'what': f'naive assumption (bottom = surface) on held-station casts with |surface - bottom| >= '
                            f'{threshold:.4f} mg/l (20 umol/L x c)',
                    'casts': int(len(s)), 'mean_abs_error': finer(float(e.mean()), 2),
                    'nested_over_stations': finer(float(s.assign(e=e).groupby('fold').e.mean().mean()), 2)}
        extra['stratified_naive'] = stratified_mgl
        stack.enter_context(patched(fc, SOURCES=mgl_sources))
        stack.enter_context(patched(H, build=AF.oxygen_build(H.build)))
    with stack, patched(N, **extra), harmonised(mgl):
        N.main()
    report = json.loads(out.read_text())
    if mgl:
        ob = report['observability']
        ob['ctd_near_bottom']['min_mg_l'] = ob['ctd_near_bottom']['min']
        ob['thresholds_umol_l'] = THRESHOLDS_MGL
        report['harmonised'] = ('oxygen in mg/l; thresholds at their mg/l definitions (keys keep the published '
                                'names); min_mg_l equals min')
    write_json(out, report)
    return report


# ---------------------------------------------------------------- driver

def compare(got, pub, keys=None, tol=1e-9, rel=True):
    if keys:
        got, pub = {k: got.get(k) for k in keys}, {k: pub.get(k) for k in keys}
    return same(got, pub, tol=tol, rel=rel)


def sections_only():
    """Rerun v2_ctd_sections alone (both units), keeping the record of the other steps."""
    record = json.loads((C2 / 'casts_reproduction.json').read_text())
    pub = json.loads((RUNS / 'v2_ctd_sections.json').read_text())
    diff = compare(ctd_sections(SCRATCH / 'ctd_sections', False), pub)
    record['steps']['v2_ctd_sections'] = {'pass': not diff, 'differences': diff[:20]}
    print('reproduce v2_ctd_sections', 'ok' if not diff else diff[:4], flush=True)
    ctd_sections(C2 / 'v2_ctd_sections', True)
    shutil.move(str(C2 / 'v2_ctd_sections/v2_ctd_sections.json'), str(C2 / 'v2_ctd_sections.json'))
    record['pass'] = all(s_['pass'] for s_ in record['steps'].values())
    write_json(C2 / 'casts_reproduction.json', record)


def main():
    if sys.argv[1:] == ['ctd_sections']:
        return sections_only()
    t0 = time.time()
    record = {'what': 'each cast-archive producer rerun in the published unit, against its published artifact, '
                      'before mg/l', 'steps': {}}

    def gate(name, diff):
        record['steps'][name] = {'pass': not diff, 'differences': diff[:20]}
        print('reproduce', name, 'ok' if not diff else diff[:4], f'({time.time() - t0:.0f} s)', flush=True)

    pub = json.loads((RUNS / 'v2_oxygen_floor.json').read_text())
    gate('v2_oxygen_floor', compare(oxygen_floor(SCRATCH / 'v2_oxygen_floor.json', False), pub))
    pub = json.loads((RUNS / 'v2_ctd_floor.json').read_text())
    gate('v2_ctd_floor', compare(ctd_floor(SCRATCH / 'v2_ctd_floor.json', False), pub))
    pub = json.loads((RUNS / 'v2_metrology_review.json').read_text())['G2_sonde_adjustment']
    got = metrology_oxygen(SCRATCH / 'v2_metrology_review.json', False, None)['G2_sonde_adjustment']
    gate('v2_metrology_review.G2', compare(got, pub, keys=('oxygen_floor', 'PG2')))
    pub = json.loads((RUNS / 'v2_ctd_sections.json').read_text())
    gate('v2_ctd_sections', compare(ctd_sections(SCRATCH / 'ctd_sections', False), pub))
    pub = json.loads((RUNS / 'eda_opportunities/eda_c2_strat.json').read_text())
    gate('eda_c2_strat', compare(strat(SCRATCH / 'eda', False), pub))
    got = placement(SCRATCH / 'v2_placement', False)
    diff = []
    for name, frame in got.items():
        p = pd.read_csv(RUNS / 'v2_placement' / name)
        num = frame.select_dtypes('number').columns
        if list(p.columns) != list(frame.columns) or len(p) != len(frame) or \
                np.nanmax(np.abs(p[num].to_numpy(float) - frame[num].to_numpy(float))) > 1e-9:
            diff.append(name)
    gate('v2_placement', diff)
    pub = json.loads((RUNS / 'v2_observability_reanalysis.json').read_text())
    got = observability(SCRATCH / 'v2_observability_reanalysis.json', False, RUNS / 'v2_bottom_aou', RUNS / 'v2_placement')
    gate('v2_observability_reanalysis.F1_F2', compare(got, pub, keys=('F1_near_bottom', 'F2_placement')))
    pub = json.loads((RUNS / 'v2_section5_numbers.json').read_text())
    gate('v2_section5_numbers', compare(section5(SCRATCH / 'v2_section5_numbers.json', False, RUNS / 'v2_bottom_aou'), pub))
    record['pass'] = all(s['pass'] for s in record['steps'].values())
    write_json(C2 / 'casts_reproduction.json', record)
    failed = [k for k, s in record['steps'].items() if not s['pass']]
    print('reproduction', 'passed' if not failed else f'FAILED: {failed}', flush=True)

    # mg/l (a producer that did not reproduce is still run, and is reported as not to be used)
    of = oxygen_floor(C2 / 'v2_oxygen_floor.json', True)
    ctd_floor(C2 / 'v2_ctd_floor.json', True)
    declared = of['by_cohort']['pre_2026']['30min']['pooled']['oxygen']['median_abs']
    metrology_oxygen(C2 / 'v2_metrology_review.json', True, declared)
    ctd_sections(C2 / 'v2_ctd_sections', True)
    shutil.move(str(C2 / 'v2_ctd_sections/v2_ctd_sections.json'), str(C2 / 'v2_ctd_sections.json'))
    strat(C2 / 'eda_opportunities', True)
    placement(C2 / 'v2_placement', True)
    bottom = C2 / 'v2_bottom_aou'
    if not (bottom / 'bottom/predictions.csv.gz').exists():
        raise SystemExit('run scripts/v2_oxygen_mgl_bottom.py first: the near-bottom predictions in mg/l are missing')
    observability(C2 / 'v2_observability_reanalysis.json', True, bottom, C2 / 'v2_placement')
    section5(C2 / 'v2_section5_numbers.json', True, bottom)
    record['mgl_not_to_be_used'] = failed
    write_json(C2 / 'casts_reproduction.json', record)
    print(f'done in {time.time() - t0:.0f} s')


if __name__ == '__main__':
    main()
