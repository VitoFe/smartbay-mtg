"""Addendum 83's job queue on the compute host: run every job of queue.txt at most once, keeping the load down.

A job starts only while fewer than --max jobs of this queue run and the one-minute load average, other users'
work included, plus two for each job started in the last minute (the load average lags), stays under
--load. Jobs start at most one every --gap seconds. Each job script writes status/<name>.exit; this runner
samples the load every 30 s into load.log and writes queue.done at the end.

usage (from the run folder, detached): python add83_queue.py --max 48 --load 100 --gap 3
"""
import argparse
import os
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--max', type=int, default=48)
    p.add_argument('--load', type=float, default=100.0)
    p.add_argument('--gap', type=float, default=3.0)
    a = p.parse_args()
    here = Path.cwd()
    names = [n.strip() for n in (here / 'queue.txt').read_text().splitlines() if n.strip()]
    running, started, peak = {}, [], 0.0
    log = open(here / 'load.log', 'a', buffering=1)
    last_sample = 0.0
    todo = [n for n in names if not (here / 'status' / f'{n}.exit').exists()]
    while todo or running:
        now = time.time()
        for n, proc in list(running.items()):
            if proc.poll() is not None:
                del running[n]
        started = [t for t in started if now - t < 60]
        load1 = os.getloadavg()[0]
        peak = max(peak, load1)
        if now - last_sample >= 30:
            log.write(f'{datetime.now(timezone.utc).isoformat()} load1 {load1:.1f} running {len(running)} '
                      f'left {len(todo)} peak {peak:.1f}\n')
            last_sample = now
        width = int((here / 'width.txt').read_text().strip()) if (here / 'width.txt').exists() else a.max
        if todo and len(running) < min(a.max, width) and load1 + 2 * len(started) + 2 < a.load:
            n = todo.pop(0)
            running[n] = subprocess.Popen(['bash', str(here / 'jobs' / f'{n}.sh')], stdin=subprocess.DEVNULL,
                                          stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            started.append(now)
            time.sleep(a.gap)
            continue
        time.sleep(2)
    log.write(f'{datetime.now(timezone.utc).isoformat()} done peak {peak:.1f}\n')
    (here / 'queue.done').write_text(datetime.now(timezone.utc).isoformat() + f' peak_load1 {peak:.1f}\n')


if __name__ == '__main__':
    main()
