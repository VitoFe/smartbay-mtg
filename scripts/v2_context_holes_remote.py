"""Addendum 53 on the compute host: check it, upload the inputs, launch, poll and fetch.

Credentials are read by scientific_campaign.connect from the repository's own configuration and
are never printed. The job runs on GPU 1 with three threads, detached.

  python scripts/v2_context_holes_remote.py check | upload | smoke | full | status | fetch-smoke | fetch
"""
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / 'scripts'))
from scientific_campaign import connect, execute  # noqa: E402

RUN = 'review_20260919/v2_context_holes'
LOCAL_INPUTS = REPO / 'reports/bay_twin/runs/v2_context_holes/inputs'
SCRIPT = REPO / 'scripts/experiments/v2_context_holes.py'
FETCH = REPO / 'reports/bay_twin/runs/v2_context_holes'


def sh(client, cmd):
    code, out = execute(client, cmd)
    print(out.strip())
    return code


def main(action):
    client, base = connect()
    run = f'{base}/{RUN}'
    py = f'{base}/.venv/bin/python'
    try:
        if action == 'check':
            sh(client, 'hostname; uptime; nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv')
            sh(client, f"{py} -c \"import chronos, torch; print('chronos', chronos.__version__, 'torch', torch.__version__)\"")
        elif action == 'upload':
            sh(client, f'mkdir -p {run}/inputs {run}/logs')
            sftp = client.open_sftp()
            try:
                sftp.put(str(SCRIPT), f'{run}/v2_context_holes.py')
                for p in sorted(LOCAL_INPUTS.glob('*')):
                    sftp.put(str(p), f'{run}/inputs/{p.name}')
                    print('put', p.name)
            finally:
                sftp.close()
            sh(client, f'cd {run} && sha256sum inputs/*.npz v2_context_holes.py')
        elif action in ('smoke', 'full'):
            flag = '--quick' if action == 'smoke' else ''
            sh(client, f'cd {run} && setsid nohup env CUDA_VISIBLE_DEVICES=1 OMP_NUM_THREADS=3 MKL_NUM_THREADS=3 '
                       f'OPENBLAS_NUM_THREADS=3 V2_THREADS=3 HF_HOME={base}/.hf {py} -u v2_context_holes.py {flag} '
                       f'> logs/{action}.log 2>&1 < /dev/null & disown')
        elif action == 'status':
            sh(client, f'cd {run} && for f in logs/*.log; do echo == $f; tail -n 6 $f; done; '
                       'ps -eo pid,etime,nlwp,pcpu,args | grep "[v]2_context_holes"; uptime; '
                       'nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv')
        elif action in ('fetch-smoke', 'fetch'):
            remote = f'{run}/results_smoke' if action == 'fetch-smoke' else f'{run}/results'
            target = FETCH / ('results_smoke' if action == 'fetch-smoke' else 'results')
            target.mkdir(parents=True, exist_ok=True)
            sftp = client.open_sftp()
            try:
                for item in sftp.listdir_attr(remote):
                    sftp.get(f'{remote}/{item.filename}', str(target / item.filename))
                    print('got', item.filename)
                (FETCH / 'logs').mkdir(exist_ok=True)
                for log in sftp.listdir_attr(f'{run}/logs'):
                    sftp.get(f'{run}/logs/{log.filename}', str(FETCH / 'logs' / log.filename))
            finally:
                sftp.close()
    finally:
        client.close()


if __name__ == '__main__':
    main(sys.argv[1])
