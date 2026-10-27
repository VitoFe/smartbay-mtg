"""Digest manifest of the pretrained-forecaster comparison's arrays, which git does not carry.

The contexts under runs/v2_fm_zoo/inputs are observed values and the prediction arrays under
runs/v2_fm_zoo/preds* are large, so both are git-ignored. This script writes
runs/v2_fm_zoo/npz_manifest.json with the path, size and sha256 of every .npz there, so the record
binds them by digest; `--check` verifies the files on disk against it. `scripts/v2_add82_npz_manifest.py`
does the same for runs/v2_addendum82 through `write` and `check` below.

usage: python scripts/v2_fm_zoo_npz_manifest.py [--check]
"""
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / 'reports' / 'bay_twin' / 'runs' / 'v2_fm_zoo'
OUT = RUN / 'npz_manifest.json'
WHAT = ('sha256 and size of every .npz under runs/v2_fm_zoo (git-ignored: contexts are observed '
        'values, prediction arrays are large)')


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def scan(run=RUN):
    files = {}
    for p in sorted(run.rglob('*.npz')):
        files[p.relative_to(run).as_posix()] = {'bytes': p.stat().st_size, 'sha256': sha256(p)}
    return files


def check(run=RUN, out=OUT):
    """0 when every file the manifest lists is on disk with its digest and no other .npz is there."""
    files = scan(run)
    stored = json.loads(out.read_text(encoding='utf-8'))['files']
    missing = sorted(set(stored) - set(files))
    extra = sorted(set(files) - set(stored))
    changed = sorted(k for k in set(stored) & set(files) if stored[k]['sha256'] != files[k]['sha256'])
    print(f'{len(stored)} in manifest, {len(files)} on disk; missing {len(missing)}, extra {len(extra)}, '
          f'changed {len(changed)}')
    for k in (missing + extra + changed)[:20]:
        print('  ', k)
    return 1 if missing or extra or changed else 0


def write(run=RUN, out=OUT, what=WHAT, extra=None):
    """Write the manifest of `run` to `out` (LF, sorted keys); `extra` adds top-level entries."""
    files = scan(run)
    folders = {}
    for k, v in files.items():
        top = k.split('/')[0]
        folders.setdefault(top, {'files': 0, 'bytes': 0})
        folders[top]['files'] += 1
        folders[top]['bytes'] += v['bytes']
    doc = {'what': what, 'folders': folders, 'files': files, **(extra or {})}
    data = (json.dumps(doc, indent=1, sort_keys=True) + '\n').encode('utf-8')
    out.write_bytes(data)
    print(f'wrote {out.relative_to(ROOT)}: {len(files)} files, ' +
          ', '.join(f'{k} {v["files"]}' for k, v in sorted(folders.items())))
    return files


def main():
    if '--check' in sys.argv:
        sys.exit(check())
    write()


if __name__ == '__main__':
    main()
