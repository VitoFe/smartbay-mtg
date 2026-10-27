"""Addendum 81 on the compute host: put every checkpoint in the Hugging Face cache and record its revision.

Downloads google/timesfm-3.0-pytorch (the one checkpoint not yet cached) at the revision recorded
in v2_fm_zoo_models.PINNED, then, offline, resolves every model of the addendum to the snapshot its
name loads and writes weights.json beside this script: repository, revision, the files of the
snapshot with their sizes and the sha256 of each weights file. Every later job runs with
HF_HUB_OFFLINE=1 and v2_fm_zoo_models asserts that the name still resolves to that revision.

  python v2_fm_zoo_fetch.py            download what is missing, then write weights.json
  python v2_fm_zoo_fetch.py --record   offline: write weights.json only
"""
import hashlib
import json
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
os.environ.setdefault('HF_HOME', '/home/vito/smartbay_do/.hf')
sys.path.insert(0, str(HERE))
import v2_fm_zoo_models as zoo  # noqa: E402

DOWNLOAD = {'google/timesfm-3.0-pytorch': ['config.json', 'model.safetensors', 'LICENSE', 'README.md']}


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1 << 22), b''):
            h.update(block)
    return h.hexdigest()


def download():
    from huggingface_hub import snapshot_download
    for repo, patterns in DOWNLOAD.items():
        t0 = time.time()
        path = snapshot_download(repo, revision='main', allow_patterns=patterns)
        got = Path(path).name
        print(repo, 'snapshot', got, f'{time.time() - t0:.0f} s', flush=True)
        if got != zoo.PINNED[repo]:
            raise SystemExit(f'{repo}: main resolves to {got}, the declaration pins {zoo.PINNED[repo]}')


def record():
    os.environ['HF_HUB_OFFLINE'] = '1'
    out = {'hf_home': os.environ['HF_HOME'], 'written_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
           'models': {}}
    for repo in zoo.REPOS.values():
        revision, snapshot = zoo.cached_revision(repo)
        files = {}
        for p in sorted(Path(snapshot).iterdir()):
            real = p.resolve()
            entry = {'bytes': real.stat().st_size}
            if p.suffix in ('.safetensors', '.json'):
                entry['sha256'] = sha256(real)
            files[p.name] = entry
        out['models'][repo] = {'revision': revision, 'pinned': zoo.PINNED[repo],
                               'matches_pin': revision == zoo.PINNED[repo], 'files': files}
        print(repo, revision, 'pin ok' if revision == zoo.PINNED[repo] else 'PIN MISMATCH', flush=True)
    (HERE / 'weights.json').write_bytes((json.dumps(out, indent=2) + '\n').encode('utf-8'))
    print('wrote', HERE / 'weights.json', flush=True)
    if not all(m['matches_pin'] for m in out['models'].values()):
        raise SystemExit('a cached revision differs from its pin')


if __name__ == '__main__':
    if '--record' not in sys.argv:
        download()
    record()
