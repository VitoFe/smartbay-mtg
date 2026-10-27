"""Addendum 83, amendment 1: the per-fold studies of the ISPRA-based arms on the compute host (Optuna 4.9.0).

Twenty cells (sealevel_tree and extboundary_tree at 1, 3, 12, 36 and 72 h, both variables; two fold studies each)
run inside the shared sweep's run folder, from the same hash-verified snapshot, through a second copy of the
queue runner (add83_queue.py) in the subfolder perfold_ispra/. The sweep's own queue is narrowed by --main-width
so that both together stay within one budget, and each runner holds the load average under --load. Outputs go to
the run folder's reports/bay_twin/runs/v2_addendum83/perfold/, which the sweep's fetch collects; logs to its
logs/perfold_<cell>.log.

usage: python scripts/launchers/add83_perfold_host.py start  <run_id> [--max 8] [--main-width 40] [--load 100]
       python scripts/launchers/add83_perfold_host.py status <run_id>
       python scripts/launchers/add83_perfold_host.py record <run_id>   -> the queue's status files, locally
Connection settings are read by scientific_campaign.connect() and never printed.
"""
import hashlib
import io
import json
import shlex
import sys
import tarfile
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / 'scripts'))
sys.path.insert(0, str(REPO))
from scientific_campaign import connect, execute  # noqa: E402

PY = '<base>/.venv/bin/python'
LEADS = (1, 3, 12, 36, 72)
ARMS = ('sealevel_tree', 'extboundary_tree')
VARIABLES = ('temperature', 'oxygen')
TARGET = REPO / 'reports/bay_twin/runs/v2_addendum83'


def remote_dir(run):
    return f'<base>/review_20260919/v2_add83/{run}'


def session(tries=6):
    for i in range(tries):
        try:
            return connect()
        except Exception as e:
            print(f'[connect attempt {i + 1} failed: {type(e).__name__}]', flush=True)
            time.sleep(4 + 3 * i)
    sys.exit('unreachable')


def jobs():
    return [(f'ispra_{v}_{a}_h{h}', ['src.experiments.v2_lead_family', 'perfold', '--arm', a, '--variable', v,
                                     '--horizon', str(h)]) for v in VARIABLES for h in LEADS for a in ARMS]


def job_script(run_root, name, args):
    cmd = ' '.join(shlex.quote(x) for x in args)
    return ('#!/bin/bash\n'
            f'cd {run_root}\n'
            'unset V2_ISPRA_BINNING V2_ADD82_OUT V2_ADD83_OUT\n'
            "export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 V2_THREADS=2 "
            "CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1\n"
            f'{PY} -u -m {cmd} > logs/perfold_{name}.log 2>&1\n'
            f'echo $? > perfold_ispra/status/{name}.exit\n')


def start(run, n, main_width, load):
    c, base = session()
    try:
        root = remote_dir(run).replace('<base>', base)
        js = jobs()
        buf = io.BytesIO()
        with tarfile.open(fileobj=buf, mode='w:gz') as tar:
            files = {'perfold_ispra/queue.txt': ('\n'.join(n_ for n_, _ in js) + '\n').encode()}
            for name, args in js:
                files[f'perfold_ispra/jobs/{name}.sh'] = job_script(root, name, args).replace('<base>', base).encode()
            for name, content in files.items():
                info = tarfile.TarInfo(name)
                info.size = len(content)
                info.mode = 0o755 if name.endswith('.sh') else 0o644
                tar.addfile(info, io.BytesIO(content))
        sftp = c.open_sftp()
        try:
            sftp.putfo(io.BytesIO(buf.getvalue()), root + '/perfold_ispra.tar.gz')
        finally:
            sftp.close()
        record = {'launcher': 'scripts/launchers/add83_perfold_host.py',
                  'launcher_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  'jobs': [{'name': n_, 'args': a} for n_, a in js], 'max': n, 'main_width': main_width, 'load': load}
        (TARGET / 'launch' / run).mkdir(parents=True, exist_ok=True)
        (TARGET / 'launch' / run / 'perfold_ispra_record.json').write_bytes(json.dumps(record, indent=2).encode('utf-8'))
        q = shlex.quote(root)
        code, out = execute(c, f"cd {q} && tar xzf perfold_ispra.tar.gz && mkdir -p perfold_ispra/status && "
                               f"echo {int(main_width)} > width.txt && "
                               f"setsid -f bash -c 'cd {q}/perfold_ispra && {PY.replace('<base>', base)} ../add83_queue.py "
                               f"--max {int(n)} --load {float(load)} --gap 3 > queue.log 2>&1' < /dev/null > /dev/null 2>&1; "
                               f"sleep 6; ls perfold_ispra/jobs | wc -l; cat width.txt; uptime")
        print(out.replace(base, '<base>'))
    finally:
        c.close()


def status(run):
    c, base = session()
    try:
        root = shlex.quote(remote_dir(run).replace('<base>', base))
        code, out = execute(c, f"cd {root}/perfold_ispra && echo perfold jobs $(wc -l < queue.txt) done $(ls status | wc -l) "
                               f"failed $(grep -L '^0$' status/*.exit 2>/dev/null | wc -l); cat queue.done 2>/dev/null; "
                               f"tail -n 1 load.log 2>/dev/null")
        print(out.replace(base, '<base>'))
    finally:
        c.close()


def record(run):
    c, base = session()
    try:
        root = remote_dir(run).replace('<base>', base)
        code, out = execute(c, f"cd {shlex.quote(root)}/perfold_ispra && for f in status/*.exit; do echo $(basename $f) $(cat $f); done; "
                               f"cat queue.done 2>/dev/null; tail -n 3 load.log")
        out = out.replace(base, '<base>')
        (TARGET / 'launch' / run / 'perfold_ispra_status.txt').write_bytes(out.encode('utf-8'))
        print(out[-400:])
    finally:
        c.close()


if __name__ == '__main__':
    op, run = sys.argv[1], sys.argv[2]
    if op == 'start':
        arg = lambda k, d: type(d)(sys.argv[sys.argv.index(k) + 1]) if k in sys.argv else d
        start(run, arg('--max', 8), arg('--main-width', 40), arg('--load', 100.0))
    elif op == 'status':
        status(run)
    elif op == 'record':
        record(run)
    else:
        raise SystemExit(op)
