"""Lane T1 job queue on the compute host (addenda 49, 51, 52).

Runs the jobs of jobs.json in order, at most T1_MAX_GPU (default 2) on GPU 1 and T1_MAX_CPU
(default 1) on the CPU at once. Before a GPU job starts it checks that GPU 1 has room (less than
T1_GPU_BUSY_MB in use, default 30000) and that the load average is below T1_MAX_LOAD (default 90);
otherwise it waits. A job whose logs/<name>.done exists is skipped, so a relaunch continues. A file
logs/STOP stops new launches; running jobs finish. Every PID it starts is recorded in
logs/queue_status.json; it never signals a process it did not start.

A second queue for the same lane can run beside the first: T1_AFTER names the first queue's
queue_status.json, and this queue starts nothing until that file shows no pending GPU job; and every
queue counts the lane's GPU processes on the whole host (T1_GPU_PATTERN), not only its own, before it
starts one, so that two queues never put more than T1_MAX_GPU jobs on the card.

  cd <run dir> && setsid nohup .venv-python -u v2_t1_queue.py > logs/queue.log 2>&1 < /dev/null & disown
"""
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
LOGS = HERE / 'logs'
LOGS.mkdir(exist_ok=True)
STATUS = LOGS / 'queue_status.json'
BASE = os.environ['T1_BASE']
MAX_GPU = int(os.environ.get('T1_MAX_GPU', '2'))
MAX_CPU = int(os.environ.get('T1_MAX_CPU', '1'))
BUSY_MB = float(os.environ.get('T1_GPU_BUSY_MB', '30000'))
MAX_LOAD = float(os.environ.get('T1_MAX_LOAD', '90'))
AFTER = os.environ.get('T1_AFTER')
GPU_PATTERN = os.environ.get('T1_GPU_PATTERN', 'v2_perfold_neural graph|v2_sequence_sweep|v2_aou_units_chronos')


def lane_gpu_processes():
    """GPU jobs of this lane running on the host, from any queue (matched on their command line)."""
    import re
    try:
        out = subprocess.run(['ps', '-eo', 'args'], capture_output=True, text=True, timeout=20).stdout
    except (OSError, subprocess.TimeoutExpired):
        return 99
    rx = re.compile(GPU_PATTERN)
    return sum(1 for line in out.splitlines() if rx.search(line) and 'python' in line and 'grep' not in line)


def other_queue_busy():
    """True while the queue named by T1_AFTER still has GPU jobs pending."""
    if not AFTER:
        return False
    try:
        s = json.loads(Path(AFTER).read_text())
    except (OSError, ValueError):
        return True
    return any(v['state'] == 'pending' and v.get('gpu') for v in s['jobs'].values())


def now():
    return datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')


def gpu1_used_mb():
    try:
        out = subprocess.run(['nvidia-smi', '--query-gpu=index,memory.used', '--format=csv,noheader,nounits'],
                             capture_output=True, text=True, timeout=20).stdout
        for line in out.strip().splitlines():
            i, used = [x.strip() for x in line.split(',')]
            if i == '1':
                return float(used)
    except (OSError, subprocess.TimeoutExpired, ValueError):
        pass
    return 1e9


def env_for(job):
    env = os.environ.copy()
    t = str(job.get('threads', 2))
    env.update(OMP_NUM_THREADS=t, OPENBLAS_NUM_THREADS=t, MKL_NUM_THREADS=t, V2_THREADS=t,
               CUDA_VISIBLE_DEVICES='1' if job['gpu'] else '', PYTHONPATH=str(HERE), PYTHONDONTWRITEBYTECODE='1',
               HF_HOME=f'{BASE}/.hf', V2PN_PUBLISHED_ROOT=f'{BASE}/bay_twin', SMARTBAY_BASE=BASE)
    return env


def main():
    jobs = json.loads((HERE / 'jobs.json').read_text())
    state = {j['name']: {'state': 'done' if (LOGS / f"{j['name']}.done").exists() else 'pending', 'gpu': j['gpu']}
             for j in jobs}
    running = {}
    started_at = now()
    while True:
        for name, p in list(running.items()):
            rc = p.poll()
            if rc is not None:
                state[name].update(state='done' if rc == 0 else 'failed', rc=rc, end=now())
                if rc == 0:
                    (LOGS / f'{name}.done').write_text(now())
                del running[name]
        stop = (LOGS / 'STOP').exists()
        for j in jobs:
            if stop or state[j['name']]['state'] != 'pending':
                continue
            n_gpu = sum(1 for n in running if state[n]['gpu'])
            n_cpu = len(running) - n_gpu
            if j['gpu'] and (n_gpu >= MAX_GPU or gpu1_used_mb() > BUSY_MB or other_queue_busy()
                             or lane_gpu_processes() >= MAX_GPU):
                continue
            if not j['gpu'] and n_cpu >= MAX_CPU:
                continue
            if os.getloadavg()[0] > MAX_LOAD:
                break
            log = open(LOGS / f"{j['name']}.log", 'ab')
            p = subprocess.Popen(j['cmd'], cwd=HERE / j.get('cwd', '.'), env=env_for(j), stdin=subprocess.DEVNULL,
                                 stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            running[j['name']] = p
            state[j['name']].update(state='running', pid=p.pid, start=now())
            time.sleep(20)           # let a GPU job allocate before the next memory check
        STATUS.write_text(json.dumps({'queue_pid': os.getpid(), 'started': started_at, 'updated': now(),
                                      'stop_requested': stop, 'load': os.getloadavg(), 'gpu1_used_mb': gpu1_used_mb(),
                                      'jobs': state}, indent=1))
        if not running and (stop or all(s['state'] != 'pending' for s in state.values())):
            break
        time.sleep(15)
    print('queue finished', now(), flush=True)


if __name__ == '__main__':
    main()
