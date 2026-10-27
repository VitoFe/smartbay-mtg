"""Progress of the addendum-30 gap-recovery campaign, and fetch when it is done.

`v2_campaign.py status` reads `results/status_*.json`, a convention this experiment does not use,
so it reports nothing for it. This reads the per-arm status files the experiment does write.

Usage: python scripts/v2_gap_status.py <run-id> [--fetch] [--results DIR]
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from scientific_campaign import connect, execute  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]



def main():
    p = argparse.ArgumentParser()
    p.add_argument('run_id')
    p.add_argument('--fetch', action='store_true')
    p.add_argument('--results', default='v2_gap_recovery',
                   help='results directory the run writes, e.g. v2_gap_recovery_long_reserved2026')
    args = p.parse_args()
    REMOTE_RESULTS = f'reports/bay_twin/runs/{args.results}'
    # The number of jobs the run launched, from its own launch record; a fixed count here once
    # meant a narrowed launch could never be fetched.
    expected = len(json.loads((ROOT / 'reports/bay_twin/runs' / args.run_id / 'transport.json')
                              .read_text())['jobs'])

    client, base = connect()
    remote = f'{base}/bay_twin/{args.run_id}'
    code, alive = execute(client, 'ps -eo args | grep -c "[g]ap_recovery"')
    # One file at a time. Concatenating them and splitting on the brace boundary is fragile: the
    # writer pretty-prints and ends with a newline, so the separator is not the `}{` it looks like.
    code, names = execute(client, f'ls {remote}/{REMOTE_RESULTS}/status_*.json 2>/dev/null')
    states = []
    for name in names.split():
        code, text = execute(client, f'cat {name}')
        try:
            states.append(json.loads(text))
        except json.JSONDecodeError:
            pass
    done = [s for s in states if s.get('state') == 'COMPLETED']
    failed = [s for s in states if s.get('state') == 'FAILED']
    print(f'{len(done)} completed, {len(failed)} failed, {alive.strip()} process(es) alive')
    for s in sorted(states, key=lambda x: (x.get('variable', ''), x.get('arm', ''))):
        line = f"  {s.get('variable',''):<12} {s.get('arm',''):<20} {s.get('state','')}"
        if s.get('state') == 'COMPLETED':
            best = s.get('overall', {})
            arm = s.get('arm')
            line += f"   arm {best.get(arm, float('nan')):.4f}  linear {best.get('linear', float('nan')):.4f}"
        if s.get('state') == 'FAILED':
            line += '   ' + str(s.get('error'))[:70]
        print(line)

    if args.fetch and len(done) + len(failed) == expected:
        dest = ROOT / 'reports/bay_twin/runs' / args.run_id / args.results
        dest.mkdir(parents=True, exist_ok=True)
        code, digests = execute(client, f'cd {remote}/{REMOTE_RESULTS} && find . -type f '
                                        f"! -name '*.sqlite3' -exec sha256sum {{}} +")
        want = {}
        for line in digests.strip().splitlines():
            h, _, name = line.partition('  ')
            want[name.strip().lstrip('./')] = h.strip()
        sftp = client.open_sftp()
        try:
            for name, digest in want.items():
                target = dest / name
                target.parent.mkdir(parents=True, exist_ok=True)
                sftp.get(f'{remote}/{REMOTE_RESULTS}/{name}', str(target))
                got = hashlib.sha256(target.read_bytes()).hexdigest()
                assert got == digest, f'{name}: digest mismatch'
        finally:
            sftp.close()
        print(f'fetched and hash-verified {len(want)} files into {dest}')


if __name__ == '__main__':
    main()
