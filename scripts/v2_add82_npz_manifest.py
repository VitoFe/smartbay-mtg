"""Digest manifest of addendum 82's arrays and downloads, which git does not carry.

Parts C and G of addendum 82 keep their pretrained-forecaster contexts (observed values) and prediction
arrays, and part G its 2026 utilisation arrays, as .npz under runs/v2_addendum82; they are git-ignored. The
cluster fetch of parts D and H downloaded runs/v2_addendum82/remote/{D,H}/fetch.tar.gz and unpacked every
member (results under runs/v2_addendum82/<part>/, the run's other files under remote/<part>/); the archives
are git-ignored too. This script writes runs/v2_addendum82/npz_manifest.json with the path, size and sha256
of every .npz and every fetch.tar.gz there, so the record binds them by digest (the same format as
runs/v2_fm_zoo/npz_manifest.json, scripts/v2_fm_zoo_npz_manifest.py; the .npz entries are exactly those the
.npz-only scan gives), and checks it against the two manifests the parts wrote themselves: part C's
runs/v2_addendum82/C/npz_manifest.json (every file it lists) and part G's input manifest,
runs/v2_addendum82/G/inputs/manifest_aou_dev.json (its sha256 per input). `--check` verifies the files on
disk against the manifest.

usage: python -m scripts.v2_add82_npz_manifest [--check]
"""
import json
import sys

from scripts.v2_fm_zoo_npz_manifest import ROOT, scan, sha256

RUN = ROOT / 'reports' / 'bay_twin' / 'runs' / 'v2_addendum82'
OUT = RUN / 'npz_manifest.json'
WHAT = ('sha256 and size of every .npz under runs/v2_addendum82 (git-ignored: contexts are observed values, '
        'prediction arrays are model output) and of the fetch.tar.gz downloads of parts D and H (git-ignored: '
        'unpacked in full)')
#: the downloads the cluster fetch left beside each part's run record
ARCHIVE = 'fetch.tar.gz'
#: the manifests the parts wrote themselves, each relative to its own folder
PART_MANIFESTS = {'C': RUN / 'C' / 'npz_manifest.json', 'G/inputs': RUN / 'G' / 'inputs' / 'manifest_aou_dev.json'}


def scan_all():
    """Every .npz (exactly the .npz-only scan) and every fetch archive, keyed by path under RUN."""
    files = scan(RUN)
    for p in sorted(RUN.rglob(ARCHIVE)):
        files[p.relative_to(RUN).as_posix()] = {'bytes': p.stat().st_size, 'sha256': sha256(p)}
    return dict(sorted(files.items()))


def write():
    """Write the manifest (LF, sorted keys), in the format of scripts/v2_fm_zoo_npz_manifest.write."""
    files = scan_all()
    folders = {}
    for k, v in files.items():
        top = k.split('/')[0]
        folders.setdefault(top, {'files': 0, 'bytes': 0})
        folders[top]['files'] += 1
        folders[top]['bytes'] += v['bytes']
    doc = {'what': WHAT, 'folders': folders, 'files': files}
    OUT.write_bytes((json.dumps(doc, indent=1, sort_keys=True) + '\n').encode('utf-8'))
    print(f'wrote {OUT.relative_to(ROOT)}: {len(files)} files, ' +
          ', '.join(f'{k} {v["files"]}' for k, v in sorted(folders.items())))
    return files


def check():
    """0 when every file the manifest lists is on disk with its digest and no other .npz or archive is there."""
    files = scan_all()
    stored = json.loads(OUT.read_text(encoding='utf-8'))['files']
    missing = sorted(set(stored) - set(files))
    extra = sorted(set(files) - set(stored))
    changed = sorted(k for k in set(stored) & set(files) if stored[k]['sha256'] != files[k]['sha256'])
    print(f'{len(stored)} in manifest, {len(files)} on disk; missing {len(missing)}, extra {len(extra)}, '
          f'changed {len(changed)}')
    for k in (missing + extra + changed)[:20]:
        print('  ', k)
    return 1 if missing or extra or changed else 0


def agree(files):
    """Every digest a part's own manifest records equals the one scanned here."""
    bad, compared = [], {}
    for prefix, path in PART_MANIFESTS.items():
        listed = json.loads(path.read_text(encoding='utf-8'))['files']
        compared[prefix] = len(listed)
        for name, entry in listed.items():
            key = f'{prefix}/{name}'
            if key not in files or files[key]['sha256'] != entry['sha256']:
                bad.append(key)
    return bad, compared


def main():
    if '--check' in sys.argv:
        sys.exit(check())
    files = write()
    bad, compared = agree(files)
    print('agrees with the parts\' own manifests:', compared, 'disagreements:', len(bad))
    if bad:
        raise SystemExit('v2_add82_npz_manifest: these differ from a part\'s own manifest: ' + ', '.join(bad[:10]))


if __name__ == '__main__':
    main()
