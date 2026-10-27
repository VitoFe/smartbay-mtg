"""Addendum 82, part C: the gap-recovery arms on Gap-2026 at 1, 2 and 3 h, applied frozen.

Gap-2026 was built at 6 to 72 h only, so its 1-6 h band held 6 h alone while Gap-Dev's held 1, 2, 3
and 6 h. This runs `src.experiments.v2_gap_recovery` unchanged on the reserved record at lengths 1,
2 and 3 (`--kind short`), and at the published 6-72 h lengths (`--kind gate`) so that the same code
can first be shown to reproduce every published Gap-2026 column. The builder draws lengths in order
from one generator seeded 20260922, so the 1-3 h episodes are those a joint 1-72 h build would draw.

Nothing is fitted, tuned or selected on 2026 and no Optuna study is created. The tuned arms of
addenda 34 and 35 (the anchored trees, the ARX bridges, the covariate regressors) are fitted on the
development cohort exactly as published, with `run_study` replaced, inside this module only, by a
reader of the published selections (`--selections`): it returns the selection the published run
recorded, marked `frozen_from`, and refuses any study it does not hold. Zero-shot models and the
per-episode Gaussian process need no selection.

The module globals of v2_gap_recovery that the run reads (PERIOD, LENGTHS, OUT, build, run_study)
are set for each job and restored afterwards. `build` is deterministic given its seed, so each
(variable, period, lengths) frame is built once per process and handed out as a deep copy.

Declared in config/bay_reserved_protocol_v2_addendum82.json, part C.
Usage, from the snapshot root on the compute host:
  python -m src.experiments.v2_gap_short2026 --jobs gate:temperature:gpr,short:oxygen:gpr \
      --selections published_selections.json --out results [--quick]
"""
import argparse
import copy
import json
import os
import platform
import socket
import sys
import time
import traceback
from pathlib import Path

import numpy as np

from src.experiments import v2_gap_recovery as gap

KINDS = {'gate': [6, 12, 18, 24, 48, 72], 'short': [1, 2, 3]}
#: the declared seed of every Gap-2026 build (addenda 30 and 31)
SEED = 20260922

_frozen = {}
_cache = {}
_original_build = gap.build


def frozen_study(name, objective, folder, quick=False, n_trials=None):
    """The published selection of study `name`; never creates, resumes or extends a study."""
    if name not in _frozen:
        raise RuntimeError(f'{name}: no published selection is held; refusing to create a study')
    record = _frozen[name]
    summary = {'study': name, 'protocol': record['protocol'],
               'completed_trials': record['completed_trials'],
               'failed_trials': record['failed_trials'], 'best_trial': record['best_trial'],
               'best_value': record['best_value'], 'params': dict(record['params']),
               'smoke_only': bool(quick), 'frozen_from': record['source_run'],
               'trials_run_here': 0}
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    (folder / f'{name}_selection.json').write_bytes(
        (json.dumps(summary, indent=2) + '\n').encode('utf-8'))
    return summary


def cached_build(variable, rng, quick=False, period=None, lengths=None):
    """`gap.build`, memoised on (variable, quick, period, lengths) for a generator at its seed."""
    fresh = np.random.default_rng(gap.SEED).integers(0, 2 ** 62)
    if copy.deepcopy(rng).integers(0, 2 ** 62) != fresh:
        return _original_build(variable, rng, quick, period, lengths)
    used = lengths if lengths is not None else (gap.LENGTHS[:4] if quick else gap.LENGTHS)
    key = (variable, bool(quick), period or gap.PERIOD, tuple(int(x) for x in used))
    if key not in _cache:
        _cache[key] = _original_build(variable, rng, quick, period, lengths)
    frame = _cache[key]
    out = frame.copy(deep=True)
    out.attrs = copy.deepcopy(frame.attrs)
    return out


def run_job(kind, variable, arm, out, quick):
    """One arm on one kind of cohort, with the module globals set and restored."""
    saved = {k: getattr(gap, k) for k in ('PERIOD', 'LENGTHS', 'OUT', 'build', 'run_study')}
    try:
        gap.PERIOD = 'reserved'
        gap.LENGTHS = list(KINDS[kind])
        gap.OUT = Path(out) / kind / 'v2_gap_recovery_reserved2026'
        gap.build = cached_build
        gap.run_study = frozen_study
        started = time.time()
        gap.run(variable, arm, quick)
        return round(time.time() - started, 1)
    finally:
        for k, v in saved.items():
            setattr(gap, k, v)


def runtime_record(out):
    record = {'host': socket.gethostname(), 'python': sys.version.split()[0],
              'platform': platform.platform(),
              'threads': {k: os.environ.get(k) for k in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS',
                                                          'MKL_NUM_THREADS', 'V2_THREADS')},
              'cuda_visible_devices': os.environ.get('CUDA_VISIBLE_DEVICES'),
              'hf_home': 'set' if os.environ.get('HF_HOME') else None,
              'hf_hub_offline': os.environ.get('HF_HUB_OFFLINE')}
    import importlib.metadata as md
    record['packages'] = {}
    for name in ('numpy', 'pandas', 'scipy', 'scikit-learn', 'lightgbm', 'optuna', 'torch',
                 'chronos-forecasting', 'transformers'):
        try:
            record['packages'][name] = md.version(name)
        except md.PackageNotFoundError:
            record['packages'][name] = None
    try:
        import torch
        record['gpu'] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    except Exception:  # torch missing or no driver: the record says so
        record['gpu'] = None
    return record


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--jobs', required=True, help='comma-separated kind:variable:arm')
    p.add_argument('--selections', required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--worker', default='worker')
    p.add_argument('--quick', action='store_true')
    a = p.parse_args()
    if gap.SEED != SEED or gap.DESIGN != 'strict':
        raise SystemExit('v2_gap_recovery must run at its default seed and strict design')
    _frozen.update(json.loads(Path(a.selections).read_text(encoding='utf-8'))['studies'])
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    jobs = [tuple(j.split(':')) for j in a.jobs.split(',') if j]
    for kind, variable, arm in jobs:
        if kind not in KINDS or arm not in gap.ARMS:
            raise SystemExit(f'unknown job {kind}:{variable}:{arm}')
    log = {'worker': a.worker, 'quick': a.quick, 'runtime': runtime_record(out), 'jobs': []}
    path = out / f'worker_{a.worker}.json'
    for kind, variable, arm in jobs:
        entry = {'kind': kind, 'variable': variable, 'arm': arm}
        try:
            entry['seconds'] = run_job(kind, variable, arm, out, a.quick)
            entry['state'] = 'COMPLETED'
        except BaseException as exc:  # recorded, then the next job runs
            entry['state'] = 'FAILED'
            entry['error'] = f'{type(exc).__name__}: {exc}'[:500]
            traceback.print_exc()
        log['jobs'].append(entry)
        path.write_bytes((json.dumps(log, indent=2) + '\n').encode('utf-8'))
        print(json.dumps(entry), flush=True)


if __name__ == '__main__':
    main()
