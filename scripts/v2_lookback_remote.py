"""Addendum 46: upload the Chronos look-back snapshot to the compute host, launch it, poll it, fetch it.

The snapshot is the remote script, its launcher and the contexts scripts/v2_lookback_inputs.py built
locally; the host only runs the model. Connection settings are read by scientific_campaign.connect
and never printed.

  python -m scripts.v2_lookback_remote upload
  python -m scripts.v2_lookback_remote launch [--quick]     refuses if GPU 1 is busy or a job runs
  python -m scripts.v2_lookback_remote status
  python -m scripts.v2_lookback_remote fetch                results into runs/v2_lookback/chronos/
Declared in config/bay_reserved_protocol_v2_addendum46.json.
"""
import io
import sys
import tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
from scientific_campaign import connect, execute  # noqa: E402

LOCAL = ROOT / 'reports/bay_twin/runs/v2_lookback'
CODE = ROOT / 'scripts/experiments'
FILES = ('v2_chronos_lookback.py', 'v2_chronos_lookback_launch.sh')


def remote_dir(base):
    return f'{base}/review_20260919/v2_lookback'


def upload(client, base):
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode='w:gz') as tar:
        for name in FILES:
            data = (CODE / name).read_bytes().replace(b'\r\n', b'\n')
            info = tarfile.TarInfo(name); info.size = len(data); info.mode = 0o755
            tar.addfile(info, io.BytesIO(data))
        for path in sorted((LOCAL / 'inputs').glob('*')):
            tar.add(path, arcname=f'inputs/{path.name}')
    remote = remote_dir(base)
    print(execute(client, f'mkdir -p {remote}/logs')[1].strip())
    sftp = client.open_sftp()
    try:
        with sftp.open(f'{remote}/snapshot.tar.gz', 'wb') as f:
            f.write(buf.getvalue())
    finally:
        sftp.close()
    code, out = execute(client, f'cd {remote} && tar -xzf snapshot.tar.gz && ls inputs | wc -l && sha256sum inputs/*.npz | head -3')
    print('unpacked:', code, out.strip())


def launch(client, base, quick):
    remote = remote_dir(base)
    running = execute(client, 'ps -eo pid,args | grep "[v]2_chronos_lookback.py" || true')[1].strip()
    if running:
        raise SystemExit(f'a look-back job is already running:\n{running}')
    gpu = execute(client, 'nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader')[1]
    print('GPUs:', gpu.strip())
    line = [g for g in gpu.strip().splitlines() if g.startswith('1,')]
    used = float(line[0].split(',')[1].strip().split()[0]) if line else 1e9
    if used > 20000:
        raise SystemExit('GPU 1 has more than 20 GB in use; not launching')
    flag = ' --quick' if quick else ''
    code, out = execute(client, f'cd {remote} && setsid nohup bash v2_chronos_lookback_launch.sh{flag} > /dev/null 2>&1 < /dev/null & disown; sleep 2; cat {remote}/logs/campaign.status')
    print('launched:', code, out.strip())


def status(client, base):
    remote = remote_dir(base)
    print(execute(client, f'cat {remote}/logs/campaign.status 2>/dev/null; tail -n 12 {remote}/logs/*lookback.log 2>/dev/null; '
                          f'ls {remote}/results 2>/dev/null | wc -l; uptime')[1])


def fetch(client, base):
    remote = remote_dir(base)
    target = LOCAL / 'chronos'
    target.mkdir(parents=True, exist_ok=True)
    sftp = client.open_sftp()
    try:
        for item in sftp.listdir_attr(f'{remote}/results'):
            sftp.get(f'{remote}/results/{item.filename}', str(target / item.filename))
        (target / 'logs').mkdir(exist_ok=True)
        for item in sftp.listdir_attr(f'{remote}/logs'):
            sftp.get(f'{remote}/logs/{item.filename}', str(target / 'logs' / item.filename))
    finally:
        sftp.close()
    print('fetched into', target, sorted(p.name for p in target.glob('*.npz')))


if __name__ == '__main__':
    action = sys.argv[1]
    client, base = connect()
    try:
        {'upload': lambda: upload(client, base), 'launch': lambda: launch(client, base, '--quick' in sys.argv),
         'status': lambda: status(client, base), 'fetch': lambda: fetch(client, base)}[action]()
    finally:
        client.close()
