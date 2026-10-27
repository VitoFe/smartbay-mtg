"""Addendum 60 part C3: upload, launch, poll and fetch the pretrained utilisation arms in mg/l.

One run directory <base>/review_20260919/mgl/<run id> holds a hash-verified snapshot of what the host
drivers read: the published Chronos scripts, the published records of addendum 52 (ref/),
thermodynamics.py, and two input folders, base_published/ (the inputs the published run read, copied
on the host from <base>/review_20260919/inputs/, their digests checked against the local files) and
base_mgl/ (the same with every sonde oxygen value in mg/l). The drivers run on GPU 1 with two
threads; connection settings are read by scientific_campaign.connect and never printed.

Five kinds of run, each with its own record:

  zeroshot            scripts/experiments/v2_oxygen_mgl_chronos.py, published then mg/l, sondes by their
                      hour's factor (the first part-C3 run; record kept in C3/_hourly_factor/)
  adapted             scripts/experiments/v2_oxygen_mgl_chronos_adapted.py, six arms, published then mg/l,
                      hour's factor (record in C3/_hourly_factor/)
  zeroshot-const      as zeroshot, the sondes x the constant c (addendum 60 amendment 4)
  adapted-const       as adapted, sondes x c, the mg/l retrains only: the published-unit retrains of the
                      adapted run reproduced the published columns bit for bit on this host
  zeroshot-bk         as zeroshot-const, then mode mgl_bk with the consistent Benson and Krause solubility
                      (addendum 71 part B); record and columns in runs/v2_solubility_bk/C3/v2_aou_units_chronos/

  python -m scripts.v2_oxygen_mgl_chronos_remote upload|launch|status|fetch <kind>
"""
import hashlib
import io
import json
import sys
import tarfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
sys.path.insert(0, str(ROOT))
from scientific_campaign import connect, execute  # noqa: E402
from scripts.v2_oxygen_mgl_common import OUT, constant_c, convert_hourly, write_json  # noqa: E402

RUNS = ROOT / 'reports/bay_twin/runs'
CODE = ROOT / 'scripts/experiments'
HOURLY = ROOT / 'reports/bay_twin/hourly_observations.csv.gz'
ENDPOINTS = RUNS / '20260918_reserved_evaluation/forecast_endpoints_2026.csv.gz'
FILES = {'v2_oxygen_mgl_chronos.py': CODE / 'v2_oxygen_mgl_chronos.py',
         'code/exp_chronos.py': CODE / 'exp_chronos.py', 'code/exp_chronos_valtest.py': CODE / 'exp_chronos_valtest.py',
         'v2_chronos2.py': CODE / 'v2_chronos2.py', 'v2_chronos_c2.py': CODE / 'v2_chronos_c2.py',
         'v2_aou_units_chronos.py': CODE / 'v2_aou_units_chronos.py',
         'ref/zeroshot_report.json': RUNS / 'v2_aou_units_chronos/zeroshot_report.json',
         'ref/zeroshot_columns.csv.gz': RUNS / 'v2_aou_units_chronos/zeroshot_columns.csv.gz',
         'src/data/thermodynamics.py': ROOT / 'src/data/thermodynamics.py',
         # imported by v2_chronos_c2.py; no study is run (the published penalty is used)
         'src/tuning/protocol_v2.py': ROOT / 'src/tuning/protocol_v2.py',
         'config/tuning_protocol_v2.json': ROOT / 'config/tuning_protocol_v2.json'}
ADAPTED = ('ft_posthoc', 'lora_posthoc', 'tv_full', 'tv_lora', 'ft_random', 'lora_random')
KINDS = {  # kind -> (record folder, record name, constant c, adapted)
    'zeroshot': (OUT / 'C3/_hourly_factor/v2_aou_units_chronos', 'remote.json', False, False),
    'adapted': (OUT / 'C3/_hourly_factor/v2_aou_units_chronos', 'remote_adapted.json', False, True),
    'zeroshot-const': (OUT / 'C3/v2_aou_units_chronos', 'remote_const.json', True, False),
    'adapted-const': (OUT / 'C3/v2_aou_units_chronos', 'remote_adapted_const.json', True, True),
    # addendum 71 part B: as zeroshot-const, then again with the consistent Benson and Krause solubility
    'zeroshot-bk': (RUNS / 'v2_solubility_bk/C3/v2_aou_units_chronos', 'remote_bk.json', True, False)}


def adapted_files():
    from v2_t1_remote import A52
    files = {k: v for k, v in A52.items() if k != 'v2_aou_units_chronos.py'}
    files.update({k: v for k, v in FILES.items() if not k.startswith(('ref/zeroshot', 'v2_oxygen_mgl_chronos.py'))})
    files['v2_oxygen_mgl_chronos_adapted.py'] = CODE / 'v2_oxygen_mgl_chronos_adapted.py'
    for arm in ADAPTED:
        files[f'ref/adapted_{arm}_columns.csv.gz'] = RUNS / f'v2_aou_units_chronos/adapted_{arm}_columns.csv.gz'
    return files


def sha(data):
    return hashlib.sha256(data).hexdigest()


def mgl_hourly(constant):
    """The hourly table with every `::oxygen` column in mg/l: by each hour's factor, or by c (amendment 4)."""
    frame = pd.read_csv(HOURLY, index_col=0)
    if constant:
        for col in [c for c in frame.columns if str(c).endswith('::oxygen')]:
            frame[col] = frame[col].to_numpy(float) * constant_c()
    else:
        frame = convert_hourly(frame)
    return frame.to_csv(lineterminator='\n').encode('utf-8')


def record_path(kind):
    folder, name, _, _ = KINDS[kind]
    return folder / name


def upload(kind):
    folder, _, constant, adapted = KINDS[kind]
    files = adapted_files() if adapted else FILES
    client, base = connect()
    try:
        run_id = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ') + '_mgl_' + kind.replace('-', '_')
        remote = f'{base}/review_20260919/mgl/{run_id}'
        manifest = {'run_id': run_id, 'created_utc': datetime.now(timezone.utc).isoformat(), 'kind': kind,
                    'declared_in': ['config/bay_reserved_protocol_v2_addendum60.json',
                                    'config/bay_reserved_protocol_v2_addendum60_amendment1.json'] +
                                   (['config/bay_reserved_protocol_v2_addendum60_amendment4.json'] if constant else []) +
                                   (['config/bay_reserved_protocol_v2_addendum71.json'] if kind == 'zeroshot-bk' else []),
                    'sonde_oxygen': 'x the constant c' if constant else 'x the hour factor',
                    'files_sha256': {}}
        buf = io.BytesIO()
        with tarfile.open(fileobj=buf, mode='w:gz', compresslevel=6) as tar:
            items = [(k, v.read_bytes()) for k, v in files.items()]
            import gzip
            items.append(('base_mgl/review_20260919/inputs/hourly_observations.csv.gz',
                          gzip.compress(mgl_hourly(constant), mtime=0)))
            for name, data in items:
                if name.endswith('.py'):
                    data = data.replace(b'\r\n', b'\n')
                manifest['files_sha256'][name] = sha(data)
                info = tarfile.TarInfo(name)
                info.size, info.mode = len(data), 0o644
                tar.addfile(info, io.BytesIO(data))
            data = json.dumps(manifest, indent=1).encode()
            info = tarfile.TarInfo('snapshot_manifest.json')
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))
        code, out = execute(client, f'mkdir -p {remote}/logs')
        if code:
            raise RuntimeError(out)
        sftp = client.open_sftp()
        try:
            with sftp.open(f'{remote}/snapshot.tar.gz', 'wb') as f:
                f.set_pipelined(True)
                f.write(buf.getvalue())
        finally:
            sftp.close()
        inputs = f'{base}/review_20260919/inputs'
        want = {'hourly_observations.csv.gz': sha(HOURLY.read_bytes()),
                'forecast_endpoints_2026.csv.gz': sha(ENDPOINTS.read_bytes())}
        cmd = (f'cd {remote} && tar xzf snapshot.tar.gz && mkdir -p base_published/review_20260919/inputs '
               f'&& cp {inputs}/hourly_observations.csv.gz {inputs}/forecast_endpoints_2026.csv.gz '
               f'base_published/review_20260919/inputs/ && cp {inputs}/forecast_endpoints_2026.csv.gz '
               f'base_mgl/review_20260919/inputs/ && sha256sum base_published/review_20260919/inputs/*')
        code, out = execute(client, cmd)
        if code:
            raise RuntimeError(out)
        got = {line.split()[1].rsplit('/', 1)[1]: line.split()[0] for line in out.strip().splitlines()}
        if got != want:
            raise SystemExit(f'the host inputs differ from the local files: {got} != {want}')
        code, out = execute(client, f'cd {remote} && sha256sum ' + ' '.join(manifest['files_sha256']))
        remote_sha = {line.split()[1]: line.split()[0] for line in out.strip().splitlines()}
        if remote_sha != manifest['files_sha256']:
            raise SystemExit('the unpacked snapshot does not match its manifest')
        write_json(record_path(kind), {'run_id': run_id, 'kind': kind, 'remote_dir': remote.replace(base, '<base>'),
                                       'inputs_sha256': want, 'snapshot_sha256': manifest['files_sha256'],
                                       'sonde_oxygen': manifest['sonde_oxygen']})
        print('uploaded', run_id, f'{len(buf.getvalue()) / 1e6:.2f} MB')
    finally:
        client.close()


def launch(kind):
    _, _, constant, adapted = KINDS[kind]
    rec = json.loads(record_path(kind).read_text())
    client, base = connect()
    try:
        remote = rec['remote_dir'].replace('<base>', base)
        code, out = execute(client, 'nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader')
        gpu1 = [line for line in out.strip().splitlines() if line.startswith('1,')]
        print('GPU:', out.strip().replace('\n', ' | '))
        if not gpu1 or int(gpu1[0].split(',')[1].strip().split()[0]) > 2000:
            raise SystemExit('GPU 1 is busy; not launched')
        py = f'{base}/.venv/bin/python'
        env = (f'CUDA_VISIBLE_DEVICES=1 V2_THREADS=2 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 '
               f'PYTHONPATH={remote} HF_HOME={base}/.hf')
        if adapted:
            modes = ('mgl',) if constant else ('published', 'mgl')
            steps = ' && '.join(f'env {env} {py} -u v2_oxygen_mgl_chronos_adapted.py {arm} {mode} > logs/{arm}_{mode}.log 2>&1'
                                for arm in ADAPTED for mode in modes)
            job = f'cd {remote} && {steps}; echo $? > logs/exit_code'
        else:
            bk = (f'&& env {env} {py} -u v2_oxygen_mgl_chronos.py mgl_bk > logs/mgl_bk.log 2>&1 '
                  if kind == 'zeroshot-bk' else '')
            job = (f'cd {remote} && env {env} {py} -u v2_oxygen_mgl_chronos.py published > logs/published.log 2>&1 '
                   f'&& env {env} {py} -u v2_oxygen_mgl_chronos.py mgl > logs/mgl.log 2>&1 {bk}; '
                   f'echo $? > logs/exit_code')
        code, out = execute(client, f"setsid -f bash -c '{job}' < /dev/null > /dev/null 2>&1; echo launched")
        print(out.strip())
        rec['launched_utc'] = datetime.now(timezone.utc).isoformat()
        write_json(record_path(kind), rec)
    finally:
        client.close()


def status(kind):
    rec = json.loads(record_path(kind).read_text())
    client, base = connect()
    try:
        remote = rec['remote_dir'].replace('<base>', base)
        code, out = execute(client, f'uptime; ps -eo pid,nlwp,pcpu,args | grep "[v]2_oxygen_mgl_chronos"; '
                                    f'nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader; '
                                    f'ls {remote}/results 2>/dev/null; cat {remote}/logs/exit_code 2>/dev/null; '
                                    f'tail -n 3 {remote}/logs/*.log 2>/dev/null')
        print(out)
    finally:
        client.close()


def fetch(kind):
    folder, _, _, adapted = KINDS[kind]
    rec = json.loads(record_path(kind).read_text())
    client, base = connect()
    try:
        remote = rec['remote_dir'].replace('<base>', base)
        folder.mkdir(parents=True, exist_ok=True)
        sftp = client.open_sftp()
        try:
            for name in sftp.listdir(f'{remote}/results'):
                sftp.get(f'{remote}/results/{name}', str(folder / name))
                print('fetched', name)
            for name in sftp.listdir(f'{remote}/logs'):
                sftp.get(f'{remote}/logs/{name}', str(folder / f'host_{"adapted_" if adapted else ""}{name}'))
        finally:
            sftp.close()
    finally:
        client.close()


if __name__ == '__main__':
    if len(sys.argv) != 3 or sys.argv[2] not in KINDS:
        raise SystemExit(f'upload|launch|status|fetch {"|".join(KINDS)}')
    {'upload': upload, 'launch': launch, 'status': status, 'fetch': fetch}[sys.argv[1]](sys.argv[2])
