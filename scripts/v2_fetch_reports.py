"""Fetch what a campaign wrote under reports/bay_twin/runs/ on Trinity, hash-verified.

`v2_campaign.py fetch` collects `results/`, and the phase-sweep modules write beside the published
artifacts under `reports/bay_twin/runs/<dir>` instead. This bundles those directories of one run,
together with its logs, and extracts them into the run's local folder, so the published copies are
never overwritten.

Usage: python scripts/v2_fetch_reports.py <run-id> <dir> [<dir> ...]
"""
import hashlib
import json
import re
import shlex
import sys
import tarfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from scientific_campaign import connect, execute  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def main():
    run, dirs = sys.argv[1], sys.argv[2:]
    if not re.fullmatch(r'\d{8}T\d{6}Z_v2_[a-z0-9_]+', run) or not dirs:
        raise SystemExit(__doc__)
    client, base = connect()
    remote = f'{base}/bay_twin/{run}'
    py = f'{base}/.venv/bin/python'
    bundle = f"""import pathlib,tarfile,hashlib,json
p=pathlib.Path.cwd()
paths=list(p.glob('*.log'))+[f for d in {dirs!r} for f in (p/'reports/bay_twin/runs'/d).rglob('*') if f.is_file()]
with tarfile.open('reports_download.tar.gz','w:gz',compresslevel=1) as a:
 for f in paths: a.add(f,arcname=f.relative_to(p).as_posix())
f=p/'reports_download.tar.gz'; print(json.dumps({{'sha256':hashlib.sha256(f.read_bytes()).hexdigest(),'bytes':f.stat().st_size,'files':len(paths)}}))
"""
    code, out = execute(client, f'cd {shlex.quote(remote)} && {py} -c {shlex.quote(bundle)}')
    if code:
        raise SystemExit(out)
    expected = json.loads(out.strip().splitlines()[-1])
    local = ROOT / 'reports/bay_twin/runs' / run
    local.mkdir(parents=True, exist_ok=True)
    target = local / 'reports_download.tar.gz'
    sftp = client.open_sftp()
    try:
        sftp.get(f'{remote}/reports_download.tar.gz', str(target))
    finally:
        sftp.close()
    if hashlib.sha256(target.read_bytes()).hexdigest() != expected['sha256']:
        raise SystemExit('download checksum mismatch')
    with tarfile.open(target) as archive:
        archive.extractall(local, filter='data')
    (local / 'reports_download_manifest.json').write_text(json.dumps(expected, indent=2))
    print('fetched and hash-verified', expected['files'], 'files into', local)


if __name__ == '__main__':
    main()
