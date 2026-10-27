"""Near-bottom oxygen (v2): every model against the naive assumption that the bottom equals the
surface, from runs/v2_bottom_aou/bottom/predictions.csv.gz (one near-bottom value per cast).

Metric: the nested macro, casts within (fold, station), stations within fold, folds. Under the
held-station protocol the fold is the held station, so this is the per-station mean averaged over
stations; under the chronological protocol it weights the five stations of 2026 equally.
Intervals: the paired bootstrap over cast *dates*, each drawn date relabelled (2000 draws), and the
delete-one-date jackknife-t (scripts.v2_summarize_aou.contrast). Casts on one day
share the moored context and the weather, so the day is the independent unit.

The utilisation arms (`|aou`) invert through Garcia-Gordon solubility in umol/kg at the cast's own
measured near-bottom temperature and salinity (scripts/experiments/exp_bottom.py), which no
other arm and no operator without a bottom sensor has; `reads_bottom_ts` marks them.

Reference arms (2026-09-27). `rerun()` refits the utilisation arms mapping back
at what each regime may read (exp_bottom.with_reference): the mean-state regime at the network's
mean temperature and 38, the profile regime at the deepest reading of its surface profile, with
the target formed at that reference (`aou_ref`) or the published near-bottom utilisation
(`aou_bed_ref`). Tuning as published (src/experiments/v2_bottom_aou.py: one TPE study of at least 100
trials per protocol, fold, regime, model and target, on inner splits of the outer training casts),
the studies persisted under runs/v2_bottom_aou/bottom_ref/studies, so a second call resumes them. The
published `aou` arms are refitted beside them as a control and gated against the stored
predictions. contrasts.json keeps the published arms only, unchanged; the reference arms against the
naive assumption, their paired contrasts with the published arms, and the best model of each regime
over both are in runs/v2_bottom_aou/bottom_ref/contrasts.json.

Usage: python -m scripts.v2_summarize_bottom [--summary-only]
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.v2_summarize_aou import cast_meta, contrast
from src.experiments.v2_field_reserved2026 import DRAWS, RNG_SEED, macro

ROOT = Path(__file__).resolve().parents[1]
D = ROOT / 'reports/bay_twin/runs/v2_bottom_aou/bottom'
REF = ROOT / 'reports/bay_twin/runs/v2_bottom_aou/bottom_ref'
PROTOCOLS = ('P1_held_station', 'P2_chronological')
MAPS_BACK = {'C2_moored_only': 'the moored network mean temperature (training mean where dark) and 38',
             'C1_cast_surface': 'the deepest reading of the surface profile (within 2 m)'}


def cast_errors(frame, meta):
    c = frame.assign(e=(frame.pred - frame.truth).abs())[['fold', 'cast_id', 'e']].join(meta, on='cast_id')
    if c.station.isna().any() or c.cast_id.duplicated().any():
        raise SystemExit('expected one row per archived cast')
    return c[['fold', 'station', 'cast_id', 'e', 'day']].reset_index(drop=True)


def rerun(quick=False):
    """Fit the reference arms and the published `aou` arms (control), one study each."""
    from src.experiments import v2_bottom_aou as vr
    REF.mkdir(parents=True, exist_ok=True)
    eb, _ = vr.import_experiments(REF)
    d = eb.build().join(eb.profile_reading(), on='cast_id')
    published = pd.read_csv(D / 'predictions.csv.gz', dtype={'fold': str})
    rows, preds, sels, gates = [], [], [], {}
    for protocol in PROTOCOLS:
        for fold, tr, te in eb.folds(d, protocol):
            tr, te = eb.with_reference(tr.reset_index(drop=True), te.reset_index(drop=True))
            truth = te.bot_DO.to_numpy(float)
            for rg, cols in eb.REGIMES.items():
                ref = f'ref_sol_{rg}'
                # (target name, target column, saturation it is inverted at); `aou` is the
                # published arm, refitted here as the control
                targets = [('aou', 'bot_AOU', 'bot_sol'), ('aou_ref', f'aou_ref_{rg}', ref),
                           ('aou_bed_ref', 'bot_AOU', ref)]
                for mdl in ('ridge', 'lightgbm'):
                    for tname, ycol, sat in targets:
                        def back(p, x, sat=sat):
                            return x[sat].to_numpy(float) - p

                        def err(p, x, back=back):
                            return back(p, x) - x.bot_DO.to_numpy(float)
                        key = f'{rg}|{mdl}|{tname}'
                        p, sel = vr.tuned_predict(mdl, tr, te, cols, ycol, err, protocol, REF / 'studies',
                                                  f'bottom_{protocol}_{fold}_{rg}_{mdl}_{tname}', quick)
                        pred = back(p, te)
                        sels.append({'protocol': protocol, 'fold': str(fold), 'model': key, **sel})
                        rows.append(dict(protocol=protocol, fold=str(fold), model=key, n=len(te),
                                         mae=float(np.abs(pred - truth).mean())))
                        preds.append(pd.DataFrame(dict(protocol=protocol, fold=str(fold), model=key,
                                                       cast_id=te.cast_id.values, truth=truth, pred=pred,
                                                       ref_sol=te[sat].to_numpy(float),
                                                       bed_sol=te.bot_sol.to_numpy(float))))
                        if tname == 'aou':
                            pub = published[(published.protocol == protocol) & (published.fold == str(fold))
                                            & (published.model == key)]
                            same = pub.cast_id.tolist() == te.cast_id.tolist()
                            gap = float(np.max(np.abs(pub.pred.to_numpy(float) - pred))) if same else None
                            gates[f'{protocol}|{fold}|{key}'] = {'rows_match': same, 'max_abs_diff': gap,
                                                                 'passed': bool(same and gap < 1e-6)}
                        print(f'{protocol}/{fold} {key}: {rows[-1]["mae"]:.3f}', flush=True)
    P = pd.concat(preds, ignore_index=True)
    path = REF / 'predictions.csv.gz'
    P.to_csv(path, index=False, compression={'method': 'gzip', 'mtime': 0})
    pd.DataFrame(rows).to_csv(REF / 'folds.csv', index=False)
    (REF / 'selections.json').write_text(json.dumps(sels, indent=2, default=str))
    short = [s['study'] for s in sels if s['completed_trials'] < 100]
    status = {'state': 'COMPLETED', 'studies': len(sels), 'studies_below_minimum': short, 'smoke_only': quick,
              'gates': gates, 'gates_passed': bool(all(g['passed'] for g in gates.values())),
              'prediction_sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
    (REF / 'status.json').write_text(json.dumps(status, indent=2))
    return status


def entry(protocol, model, arm, base, **extra):
    c = contrast(arm, base)
    return dict(protocol=protocol, model=model, mae=round(macro(arm), 4), naive=round(macro(base), 4),
                diff=c['point'], ci=c['ci'], excludes_zero=c['excludes_zero'],
                jackknife_t=c['jackknife_t'], jackknife_excludes_zero=c['jackknife_excludes_zero'],
                robust=bool(c['excludes_zero'] and c['jackknife_excludes_zero']),
                casts=int(len(arm)), dates=c['dates'], **extra,
                metric='nested macro: casts within (fold, station), stations, folds',
                unit=f'cast date, relabelled, {DRAWS} draws, seed {RNG_SEED}; jackknife-t over dates')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--summary-only', action='store_true',
                        help='read runs/v2_bottom_aou/bottom_ref/predictions.csv.gz instead of refitting')
    args = parser.parse_args()
    if not args.summary_only:
        status = rerun()
        print('control gates passed:', status['gates_passed'], flush=True)
    P = pd.read_csv(D / 'predictions.csv.gz', dtype={'fold': str})
    meta = cast_meta()
    out, errors = [], {}
    for pr in PROTOCOLS:
        sub = P[P.protocol == pr]
        base = cast_errors(sub[sub.model == 'B_surface'], meta)
        errors[(pr, 'B_surface')] = base
        for m in sorted(set(sub.model) - {'B_surface'}):
            arm = cast_errors(sub[sub.model == m], meta)
            if not arm[['fold', 'cast_id']].equals(base[['fold', 'cast_id']]):
                raise SystemExit(f'{pr} {m}: rows differ from the naive arm')
            errors[(pr, m)] = arm
            out.append(entry(pr, m, arm, base, reads_bottom_ts=m.endswith('|aou')))
            r = out[-1]
            print(f"{pr:17s} {m:34s} {r['mae']:7.3f} vs naive {r['diff']:+.3f} {r['ci']} jk {r['jackknife_t']}"
                  f"{'  *' if r['excludes_zero'] else ''}", flush=True)
    (D / 'contrasts.json').write_text(json.dumps(out, indent=2))
    # The reference arms go to their own artifact, so the published list above is unchanged (the
    # deck and its near-bottom figure read it).
    R = pd.read_csv(REF / 'predictions.csv.gz', dtype={'fold': str})
    extra = []
    new = sorted(m for m in set(R.model) if m.rsplit('|', 1)[1] in ('aou_ref', 'aou_bed_ref'))
    for pr in PROTOCOLS:
        base = errors[(pr, 'B_surface')]
        for m in new:
            arm = cast_errors(R[(R.protocol == pr) & (R.model == m)], meta)
            if not arm[['fold', 'cast_id']].equals(base[['fold', 'cast_id']]):
                raise SystemExit(f'{pr} {m}: rows differ from the naive arm')
            errors[(pr, m)] = arm
            rg = m.split('|')[0]
            extra.append(entry(pr, m, arm, base, reads_bottom_ts=False, maps_back_at=MAPS_BACK[rg]))
            r = extra[-1]
            print(f"{pr:17s} {m:34s} {r['mae']:7.3f} vs naive {r['diff']:+.3f} {r['ci']} jk {r['jackknife_t']}"
                  f"{'  *' if r['excludes_zero'] else ''}", flush=True)
    paired(errors, out, extra)


def paired(errors, published, extra):
    """The reference arms against the published arms they replace, and the best models by regime."""
    report = {'what': 'near-bottom utilisation arms mapped back at what each regime may read',
              'maps_back_at': MAPS_BACK, 'salinity': 38.0,
              'targets': {'aou_ref': 'saturation at the reference minus near-bottom oxygen, inverted there',
                          'aou_bed_ref': 'near-bottom utilisation at the measured bed T and S (training '
                                         'label), inverted at the reference'},
              'status': json.loads((REF / 'status.json').read_text()),
              'against_naive': extra, 'contrasts': [], 'best': {}}
    entries = published + extra
    for pr in PROTOCOLS:
        for rg in ('C1_cast_surface', 'C2_moored_only'):
            for mdl in ('ridge', 'lightgbm'):
                refs = [f'{rg}|{mdl}|aou', f'{rg}|{mdl}|direct'] + (
                    [f'{rg}|{mdl}|gradient'] if rg == 'C1_cast_surface' else [])
                for t in ('aou_ref', 'aou_bed_ref'):
                    a = f'{rg}|{mdl}|{t}'
                    for b in refs:
                        c = contrast(errors[(pr, a)], errors[(pr, b)])
                        report['contrasts'].append(dict(
                            protocol=pr, a=a, b=b, diff=c['point'], ci=c['ci'],
                            excludes_zero=c['excludes_zero'], jackknife_t=c['jackknife_t'],
                            jackknife_excludes_zero=c['jackknife_excludes_zero'],
                            robust=bool(c['excludes_zero'] and c['jackknife_excludes_zero']),
                            dates=c['dates']))
                        x = report['contrasts'][-1]
                        print(f"{pr:17s} {a:34s} - {b:32s} {x['diff']:+.3f} {x['ci']} jk {x['jackknife_t']}"
                              f"{'  robust' if x['robust'] else ''}", flush=True)
            # the best model of the regime: over every arm, and over the arms that read nothing
            # measured at the bed when they predict
            pool = [e for e in entries if e['protocol'] == pr and e['model'].startswith(rg)]
            honest = [e for e in pool if not e['reads_bottom_ts']]
            pick = {'overall': min(pool, key=lambda e: e['mae']), 'reads_nothing_at_the_bed':
                    min(honest, key=lambda e: e['mae'])}
            report['best'][f'{pr}|{rg}'] = {k: {f: e[f] for f in ('model', 'mae', 'naive', 'diff', 'ci',
                                                               'jackknife_t', 'robust', 'dates',
                                                               'reads_bottom_ts')}
                                            for k, e in pick.items()}
    (REF / 'contrasts.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report['best'], indent=1))


if __name__ == '__main__':
    main()
