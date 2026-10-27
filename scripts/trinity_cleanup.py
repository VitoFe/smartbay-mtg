"""Free disk space on Trinity by removing only files that have a verified copy elsewhere.

Run it ON Trinity, as vito. It previews by default; nothing is deleted without --apply.

  python trinity_cleanup.py                 # preview: what would go, and how much space
  python trinity_cleanup.py --apply         # delete what the preview lists
  python trinity_cleanup.py --apply --smoke # also delete smoke-run outputs (results_smoke/)

What it removes, each only after the check in brackets passes:
  1. <run>/data/ in every campaign run under ~/smartbay_do/bay_twin/
     [every file in it is inside that run's snapshot.tar.gz with the same size; the tarball stays,
      so the inputs can be unpacked again]
  2. <run>/results_download.tar.gz
     [every file in the tarball is already unpacked under the run with the same size]
  3. <run>/results/columns/ in the standalone experiment runs of trial_columns() (one .npy per trial)
     [every trial number in it is also a key of the study's *_2026_columns.npz]
  4. with --smoke only: <run>/results_smoke/ in those same runs (smoke outputs; the logs stay)

It never touches: snapshot tarballs, manifests, results, logs, studies, the virtual environment,
the model caches, anything outside ~/smartbay_do, or anything that is a symbolic link. It prints
the package-cache commands instead of running them.
"""
import shutil, sys, tarfile
from pathlib import Path

import numpy as np

BASE = Path.home() / 'smartbay_do'
APPLY = '--apply' in sys.argv
SMOKE = '--smoke' in sys.argv


def size(path):
    return sum(p.stat().st_size for p in path.rglob('*') if p.is_file() and not p.is_symlink())


def inside(path):
    """True only for a real path under BASE."""
    try:
        return not path.is_symlink() and path.resolve().is_relative_to(BASE.resolve())
    except OSError:
        return False


def remove(path, why, plan):
    if not inside(path):
        print('  SKIP (outside the project or a link):', path); return
    n = size(path) if path.is_dir() else path.stat().st_size
    plan.append((path, n, why))


def unpacked_inputs(plan, kept):
    for run in sorted((BASE / 'bay_twin').glob('*')):
        data, tar = run / 'data', run / 'snapshot.tar.gz'
        if not data.is_dir():
            continue
        if not tar.is_file():
            kept.append((data, 'no snapshot.tar.gz to restore it from')); continue
        with tarfile.open(tar) as t:
            members = {m.name: m.size for m in t.getmembers() if m.isfile()}
        files = [p for p in data.rglob('*') if p.is_file()]
        missing = [p for p in files if members.get(p.relative_to(run).as_posix()) != p.stat().st_size]
        if missing:
            kept.append((data, f'{len(missing)} files not in the tarball, e.g. {missing[0].relative_to(run)}'))
        else:
            remove(data, 'unpacked copy of snapshot.tar.gz', plan)


def downloaded_results(plan, kept):
    for tar in sorted((BASE / 'bay_twin').glob('*/results_download.tar.gz')):
        run = tar.parent
        with tarfile.open(tar) as t:
            members = [(m.name, m.size) for m in t.getmembers() if m.isfile()]
        missing = [name for name, n in members
                   if not (run / name).is_file() or (run / name).stat().st_size != n]
        if missing:
            kept.append((tar, f'{len(missing)} members not unpacked, e.g. {missing[0]}'))
        else:
            remove(tar, 'every member already unpacked in the run', plan)


def trial_columns(plan, kept):
    for columns in sorted((BASE / 'review_20260919').glob('*/results/columns')):
        for study in sorted(p for p in columns.iterdir() if p.is_dir()):
            packed = columns.parent / f'{study.name}_2026_columns.npz'
            if not packed.is_file():
                kept.append((study, 'no packed .npz beside it')); continue
            keys = set(np.load(packed).files)
            trials = {p.stem for p in study.glob('*.npy')}
            if trials - keys:
                kept.append((study, f'{len(trials - keys)} trials not in {packed.name}'))
            else:
                remove(study, f'all {len(trials)} trials are in {packed.name}', plan)


def smoke_outputs(plan):
    for smoke in sorted((BASE / 'review_20260919').glob('*/results_smoke')):
        remove(smoke, 'smoke-run outputs', plan)


def main():
    plan, kept = [], []
    unpacked_inputs(plan, kept)
    downloaded_results(plan, kept)
    trial_columns(plan, kept)
    if SMOKE:
        smoke_outputs(plan)
    total = sum(n for _, n, _ in plan)
    print(f"{'Deleting' if APPLY else 'Would delete'} {len(plan)} items, {total / 1e9:.1f} GB:")
    for path, n, why in plan:
        print(f'  {n / 1e6:9.1f} MB  {path.relative_to(BASE)}  ({why})')
    for path, why in kept:
        print(f'  KEEP {path.relative_to(BASE)}  ({why})')
    if APPLY:
        for path, _, _ in plan:
            shutil.rmtree(path) if path.is_dir() else path.unlink()
        print(f'freed {total / 1e9:.1f} GB')
    else:
        print('Preview only. Run again with --apply to delete (add --smoke to include smoke outputs).')
    print('Package caches are not touched; to clear them yourself: pip cache purge ; uv cache prune')


if __name__ == '__main__':
    main()
