"""Emit one line each time a gap-recovery arm finishes, then exit when the run is done.

Written for a background watcher: every line on stdout becomes a notification, so it prints only
on a change of state rather than on every poll. Silence means nothing has finished yet.

It covers failure as well as success. A watcher that only reports completions is indistinguishable
from a watcher whose jobs have all crashed, so a FAILED arm emits a line too, and the loop ends if
every process has exited even when arms are missing.

The SSH tunnel to Trinity has dropped twice in a session; a failed poll is reported once and then
retried rather than ending the watch.

Usage: python scripts/v2_gap_watch.py <run-id> [--expect 4] [--interval 180]
"""
import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from scientific_campaign import connect, execute  # noqa: E402

RESULTS = 'reports/bay_twin/runs/v2_gap_recovery'


def poll(run_id, results=RESULTS):
    """(states, processes_alive). Raises if the tunnel is down."""
    client, base = connect()
    try:
        remote = f'{base}/bay_twin/{run_id}/{results}'
        code, alive = execute(client, 'ps -eo args | grep -c "[g]ap_recovery"')
        code, names = execute(client, f'ls {remote}/status_*.json 2>/dev/null')
        states = {}
        for name in names.split():
            code, text = execute(client, f'cat {name}')
            try:
                d = json.loads(text)
            except json.JSONDecodeError:
                continue
            states[f"{d.get('variable')}/{d.get('arm')}"] = d
        return states, int(alive.strip() or 0)
    finally:
        client.close()


def describe(key, state):
    if state.get('state') == 'FAILED':
        return f'FAILED  {key}: {str(state.get("error"))[:90]}'
    overall = state.get('overall', {})
    arm = state.get('arm')
    value, linear = overall.get(arm), overall.get('linear')
    if value is None or linear is None:
        return f'done    {key}'
    verdict = 'beats linear' if value < linear else 'loses to linear'
    return f'done    {key}: {value:.4f} vs linear {linear:.4f}  ({verdict})'


def main():
    p = argparse.ArgumentParser()
    p.add_argument('run_id')
    p.add_argument('--expect', type=int, default=4)
    p.add_argument('--interval', type=int, default=180)
    p.add_argument('--results', default=RESULTS,
                   help='results directory under the run, e.g. a _reserved2026 or _seed variant')
    args = p.parse_args()

    seen, complained = set(), False
    while True:
        try:
            states, alive = poll(args.run_id, args.results)
            complained = False
        except Exception as exc:
            if not complained:
                print(f'poll failed ({type(exc).__name__}); retrying', flush=True)
                complained = True
            time.sleep(args.interval)
            continue

        finished = {k: v for k, v in states.items()
                    if v.get('state') in ('COMPLETED', 'FAILED')}
        for key in sorted(finished):
            if key not in seen:
                seen.add(key)
                print(f'[{len(seen)}/{args.expect}] {describe(key, finished[key])}', flush=True)

        if len(seen) >= args.expect:
            print(f'all {args.expect} arms finished', flush=True)
            return 0
        if alive == 0 and seen:
            print(f'no processes left with {len(seen)}/{args.expect} finished; run ended early',
                  flush=True)
            return 1
        time.sleep(args.interval)


if __name__ == '__main__':
    raise SystemExit(main())
