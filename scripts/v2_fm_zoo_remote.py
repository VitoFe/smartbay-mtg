"""Addendum 81 on the compute host: upload, smoke, launch, status, rerun, inspect, fetch.

  python scripts/v2_fm_zoo_remote.py upload            code and inputs, checked on the host by sha256
  python scripts/v2_fm_zoo_remote.py smoke             one worker, the first 32 contexts of one file per task
  python scripts/v2_fm_zoo_remote.py launch [--workers 3] [--tasks ABFCDE]
  python scripts/v2_fm_zoo_remote.py status
  python scripts/v2_fm_zoo_remote.py rerun [--n 64]    the determinism check, after the full run
  python scripts/v2_fm_zoo_remote.py inspect [--smoke | --folder F]
  python scripts/v2_fm_zoo_remote.py fetch [--smoke | --folder F]   predictions to reports/bay_twin/runs/v2_fm_zoo/preds/
  python scripts/v2_fm_zoo_remote.py deviation         NOT declared: TimesFM 2.5 on the 1440 h gap files without
                                                       the continuous quantile head (the library refuses it beyond
                                                       1024 steps), into preds_tfm25_no_cqh/

Amendment 3 (config/bay_reserved_protocol_v2_addendum81_amendment3.json):
  a3-arrange   on the host: the batched TimesFM 3.0 outputs to preds_superseded_tfm3_batched/, the clamped
               TimesFM 2.5 outputs of tasks A and F to preds_superseded_tfm25_clamped/ (both kept), the 1440 h
               TimesFM 2.5 outputs of preds_tfm25_no_cqh/ adopted into preds/ (the originals untouched)
  a3-launch    three workers on GPU 1, each through the three amendment-3 plans (A3_PLANS) in its own order
  a3-rerun     the determinism check of the amended outputs, preds/rerun_check_a3.json
  a3-fetch     the amended outputs, the reports and the logs
  a3-mirror    sha256 of every prediction file on the host and here, folder by folder (run_logs/a3_mirror_check.json)

The runner is scripts/experiments/v2_fm_zoo.py; it sits flat in the run directory beside the model layer
(v2_fm_zoo_models.py), the input and output format (v2_fm_zoo_io.py) and the model revisions
(weights.json, written on the host when the checkpoints were cached). Inputs are the files the three
manifests of reports/bay_twin/runs/v2_fm_zoo/inputs/ list, and the manifests themselves; nothing else is
uploaded, and the pack refuses credentials. The host checks every uploaded file against the digests
computed here (sha256sum -c) before anything runs; the result is kept in upload_manifest.json.

Every job runs on GPU 1 with four threads per library and the Hugging Face cache offline. `launch`
refuses while the model checks or another runner is still running, or when GPU 1 has less free memory
than the workers need.
"""
import argparse
import hashlib
import json
import shutil
import sys
import tarfile
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
HERE = ROOT / 'reports/bay_twin/runs/v2_fm_zoo'
INPUTS = HERE / 'inputs'
PREDS = HERE / 'preds'
RUN = 'review_20260919/v2_fm_zoo'
CODE = {'v2_fm_zoo.py': ROOT / 'scripts/experiments/v2_fm_zoo.py',
        'v2_fm_zoo_models.py': ROOT / 'scripts/experiments/v2_fm_zoo_models.py',
        'v2_fm_zoo_io.py': ROOT / 'scripts/v2_fm_zoo_io.py'}
MANIFESTS = ('manifest_6h.json', 'manifest_dist_bridge.json', 'manifest_gaps.json')
ENV = ('CUDA_VISIBLE_DEVICES=1 HF_HOME={base}/.hf HF_HUB_OFFLINE=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 '
       'OPENBLAS_NUM_THREADS=4 V2_THREADS=4 PYTHONDONTWRITEBYTECODE=1')
GIB_PER_WORKER = 8          # all five checkpoints loaded plus the largest batch seen in the checks
DEVIATION = 'preds_tfm25_no_cqh'   # TimesFM 2.5 at 1440 h without the continuous quantile head (amendment 3 adopts)
A3 = 'config/bay_reserved_protocol_v2_addendum81_amendment3.json'
SUP_TFM3 = 'preds_superseded_tfm3_batched'
SUP_TFM25 = 'preds_superseded_tfm25_clamped'
#: amendment 3: TimesFM 3.0 one series per call everywhere, positivity clamps off in tasks A and F
A3_PLANS = {'tfm3BCDE': '--arms timesfm3_uni,timesfm3_cov --tasks BCDE --batch 1',
            'tfm3AF': '--arms timesfm3_uni,timesfm3_cov --tasks AF --batch 1 --tfm3-flags make_positive=false',
            'tfm25AF': '--arms timesfm25_uni,timesfm25_cov --tasks AF --tfm25-config infer_is_positive=false'}
A3_ORDERS = (('tfm3BCDE', 'tfm3AF', 'tfm25AF'), ('tfm3AF', 'tfm3BCDE', 'tfm25AF'), ('tfm25AF', 'tfm3BCDE', 'tfm3AF'))


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def model_inputs():
    """The input files the manifests list, with the digest each manifest records."""
    out = {}
    m6 = json.loads((INPUTS / 'manifest_6h.json').read_text(encoding='utf-8'))
    out.update({n: e['sha256'] for n, e in m6['files'].items()})
    md = json.loads((INPUTS / 'manifest_dist_bridge.json').read_text(encoding='utf-8'))
    for section in ('C_distribution', 'D_silent_probe'):
        out.update({n: e['sha256'] for n, e in md[section]['files'].items()})
    mg = json.loads((INPUTS / 'manifest_gaps.json').read_text(encoding='utf-8'))
    out.update({n: e['sha256'] for n, e in mg['files'].items()})
    return out


def session(tries=6):
    from scientific_campaign import connect
    for i in range(tries):
        try:
            return connect()
        except Exception as e:                                   # banner or session errors through the VPN
            print(f'[connect attempt {i + 1} failed: {type(e).__name__}]', flush=True)
            time.sleep(4 + 3 * i)
    raise SystemExit('compute host unreachable')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('action', choices=('upload', 'smoke', 'launch', 'status', 'rerun', 'inspect', 'fetch',
                                       'deviation', 'add', 'stop-gpu', 'a3-arrange', 'a3-launch', 'a3-rerun',
                                       'a3-fetch', 'a3-mirror'))
    ap.add_argument('--gpu', type=int, default=1, help='add / stop-gpu: the GPU the workers use')
    ap.add_argument('--first', type=int, default=4, help='add: index of the first new worker')
    ap.add_argument('--workers', type=int, default=3)
    ap.add_argument('--tasks', default=None)
    ap.add_argument('--n', type=int, default=64)
    ap.add_argument('--smoke', action='store_true')
    ap.add_argument('--folder', default=None, help='inspect or fetch this output folder (default preds)')
    a = ap.parse_args()
    if a.smoke:
        a.folder = 'smoke'
    a.folder = a.folder or 'preds'
    from scientific_campaign import execute
    client, base = session()
    run, py = f'{base}/{RUN}', f'{base}/.venv/bin/python'
    env = ENV.format(base=base)

    def sh(cmd, show=True):
        code, out = execute(client, cmd)
        out = out.replace(base, '<base>')
        if show:
            print(out.rstrip())
        return code, out

    def busy():
        _, out = sh('ps -eo pid,args | grep -E "[v]2_fm_zoo(_checks)?\\.py" || true', show=False)
        return [line for line in out.splitlines() if line.strip()]

    def detach(cmd, log):
        sh(f"cd {run} && setsid -f bash -c 'export {env}; {cmd} > logs/{log} 2>&1' < /dev/null > /dev/null 2>&1")

    def gpu_mem(index):
        _, out = sh('nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv,noheader,nounits', show=False)
        for line in out.splitlines():
            parts = [p.strip() for p in line.split(',')]
            if parts and parts[0] == str(index):
                return (float(parts[2]) - float(parts[1])) / 1024, float(parts[1]) / 1024
        raise SystemExit(f'GPU {index} not found')

    def gpu_apps(index):
        """Compute processes on one GPU: 'ours' if the process belongs to this account, else its user."""
        _, out = sh(f'nvidia-smi -i {index} --query-compute-apps=pid,used_memory --format=csv,noheader,nounits; '
                    'echo "ME $(id -un)"', show=False)
        lines = [x for x in out.splitlines() if x.strip()]
        me = next((x.split()[1] for x in lines if x.startswith('ME ')), None)
        apps = []
        for x in lines:
            if x.startswith('ME ') or 'No running' in x:
                continue
            pid = x.split(',')[0].strip()
            if not pid.isdigit():
                continue
            _, u = sh(f'ps -o user= -p {pid} || echo gone', show=False)
            u = u.strip() or 'unknown'
            apps.append({'pid': pid, 'user': 'ours' if u == me else u})
        return apps

    def gpu_free_gib():
        _, out = sh('nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv,noheader,nounits', show=False)
        for line in out.splitlines():
            parts = [p.strip() for p in line.split(',')]
            if parts and parts[0] == '1':
                return (float(parts[2]) - float(parts[1])) / 1024, float(parts[1]) / 1024
        raise SystemExit('GPU 1 not found')

    try:
        if a.action == 'upload':
            files = model_inputs()
            local = {}
            for name, want in files.items():
                got = sha(INPUTS / name)
                if got != want:
                    raise SystemExit(f'{name}: {got} differs from its manifest ({want})')
                local[f'inputs/{name}'] = INPUTS / name
            for name in MANIFESTS:
                local[f'inputs/{name}'] = INPUTS / name
            for name, path in CODE.items():
                local[name] = path
            for arc, path in local.items():
                if path.name.startswith('.env') or 'deploy.toml' in path.name:
                    raise SystemExit(f'refusing to pack {path}')
            digests = {arc: sha(path) for arc, path in local.items()}
            digests_check = dict(digests, **{'weights.json': sha(HERE / 'weights.json')})
            tmp = Path(tempfile.mkdtemp(prefix='fm_zoo_pack_'))
            try:
                listing = ''.join(f'{d}  {arc}\n' for arc, d in sorted(digests_check.items()))
                (tmp / 'upload_sha256.txt').write_bytes(listing.encode('ascii'))
                pack = tmp / 'pack.tar.gz'
                with tarfile.open(pack, 'w:gz') as tar:
                    for arc, path in sorted(local.items()):
                        tar.add(path, arcname=arc)
                    tar.add(tmp / 'upload_sha256.txt', arcname='upload_sha256.txt')
                sh(f'mkdir -p {run}/inputs {run}/logs {run}/status')
                sftp = client.open_sftp()
                try:
                    sftp.put(str(pack), f'{run}/pack.tar.gz')
                finally:
                    sftp.close()
                print(f'uploaded {len(local)} files ({pack.stat().st_size / 1e6:.1f} MB)')
            finally:
                shutil.rmtree(tmp, ignore_errors=True)
            code, out = sh(f'cd {run} && tar -xzf pack.tar.gz && rm pack.tar.gz && '
                           f'sha256sum -c --quiet upload_sha256.txt && echo ALL_OK', show=False)
            ok = code == 0 and 'ALL_OK' in out
            print(out.strip()[-2000:])
            record = {'declared_in': 'config/bay_reserved_protocol_v2_addendum81.json', 'remote_dir': f'<base>/{RUN}',
                      'uploaded_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
                      'files': dict(sorted(digests_check.items())), 'host_check': 'sha256sum -c',
                      'host_check_passed': ok}
            (HERE / 'upload_manifest.json').write_bytes((json.dumps(record, indent=1) + '\n').encode('utf-8'))
            if not ok:
                raise SystemExit('the host check failed: nothing may run until the upload is repeated')
            print(f'host check passed on {len(digests_check)} files; record in {(HERE / "upload_manifest.json").relative_to(ROOT)}')
        elif a.action in ('smoke', 'launch', 'rerun', 'deviation'):
            running = busy()
            if running:
                print('\n'.join(running))
                raise SystemExit('a model check or runner is still running on the host: nothing launched')
            free, used = gpu_free_gib()
            workers = 1 if a.action != 'launch' else a.workers
            if free < GIB_PER_WORKER * workers or used > 8:
                raise SystemExit(f'GPU 1 has {free:.1f} GiB free and {used:.1f} GiB in use: nothing launched')
            if a.action == 'deviation':
                # NOT declared: TimesFM 2.5 on the gap files longer than its continuous quantile head
                # allows (1440 h), the head switched off, into a folder of its own
                files = ','.join(n for n in model_inputs() if n.startswith('E_') and n.endswith('_L1440.npz'))
                sh(f'cd {run} && rm -rf {DEVIATION}/claims {DEVIATION}/failed')
                detach(f'{py} -u v2_fm_zoo.py run --worker dev1440 --out {DEVIATION} --files {files} '
                       f'--arms timesfm25_uni,timesfm25_cov --tfm25-no-continuous-head', 'zoo_dev1440.out')
            elif a.action == 'smoke':
                sh(f'cd {run} && rm -rf smoke/claims smoke/failed')
                detach(f'{py} -u v2_fm_zoo.py run --worker smoke --out smoke --limit 32 --files smoke', 'zoo_smoke.out')
            elif a.action == 'rerun':
                detach(f'{py} -u v2_fm_zoo.py rerun --out preds --n {a.n}', 'zoo_rerun.out')
            else:
                sh(f'cd {run} && rm -rf preds/claims preds/failed')        # no runner alive: claims are stale
                tasks = f' --tasks {a.tasks}' if a.tasks else ''
                for w in range(1, a.workers + 1):
                    detach(f'{py} -u v2_fm_zoo.py run --worker w{w} --out preds{tasks}', f'zoo_w{w}.out')
                    time.sleep(20)                                      # staggered model loading
            time.sleep(5)
            print('\n'.join(busy()))
        elif a.action == 'add':
            # Vito, 2026-10-11, relayed for this run only: both GPUs and up to about half the cores.
            # New workers join the running queue: each (input, arm) is taken through an exclusive claim
            # file, so a job is never given to two workers and the running ones keep theirs.
            apps = gpu_apps(a.gpu)
            others = [x for x in apps if x['user'] != 'ours']
            if others:
                raise SystemExit(f'GPU {a.gpu} has processes of other users: {others}; nothing added')
            free, used = gpu_mem(a.gpu)
            if used + GIB_PER_WORKER * a.workers > 38 or free < GIB_PER_WORKER * a.workers:
                raise SystemExit(f'GPU {a.gpu}: {used:.1f} GiB used, {free:.1f} GiB free; {a.workers} more '
                                 'workers would pass 38 GiB')
            _, out = sh('cat /proc/loadavg', show=False)
            load = float(out.split()[0])
            if load > 40:
                raise SystemExit(f'load average {load}: nothing added')
            genv = env.replace('CUDA_VISIBLE_DEVICES=1', f'CUDA_VISIBLE_DEVICES={a.gpu}')
            tasks = f' --tasks {a.tasks}' if a.tasks else ''
            for w in range(a.first, a.first + a.workers):
                name = f'g{a.gpu}w{w}'
                sh(f"cd {run} && setsid -f bash -c 'export {genv}; {py} -u v2_fm_zoo.py run --worker {name} "
                   f"--out preds{tasks} > logs/zoo_{name}.out 2>&1' < /dev/null > /dev/null 2>&1")
                time.sleep(15)
            time.sleep(5)
            print('\n'.join(busy()))
            sh('uptime')
        elif a.action == 'stop-gpu':
            # our workers on one GPU, stopped (a killed job is resumed by any later worker: its output is
            # written only when complete, and its claim names a dead process)
            _, out = sh(f'for p in $(pgrep -u $(id -u) -f "v2_fm_zoo.py run --worker g{a.gpu}w"); do echo $p; done',
                        show=False)
            pids = [p for p in out.split() if p.isdigit()]
            if pids:
                sh(f'kill {" ".join(pids)}; sleep 2; ps -eo pid,args | grep -E "[v]2_fm_zoo\\.py run --worker g{a.gpu}w" || true')
            print(f'stopped {len(pids)} workers on GPU {a.gpu}')
        elif a.action == 'a3-arrange':
            if busy():
                raise SystemExit('a runner is still running on the host: nothing moved')
            am = f'--amendment {A3} --amendment-sha256 {sha(ROOT / A3)}'
            steps = [f'supersede --out preds --to {SUP_TFM3} --arms timesfm3_uni,timesfm3_cov {am} --reason timesfm3_batch_one',
                     f'supersede --out preds --to {SUP_TFM25} --tasks AF --arms timesfm25_uni,timesfm25_cov {am} '
                     '--reason positivity_clamps',
                     f'adopt --src {DEVIATION} --out preds {am} --reason timesfm25_long_horizons']
            for s in steps:
                code, _ = sh(f'cd {run} && export {env}; {py} v2_fm_zoo.py {s}')
                if code:
                    raise SystemExit(f'failed: {s}')
            sh(f'cd {run} && for d in preds {SUP_TFM3} {SUP_TFM25} {DEVIATION}; do echo $d $(ls $d/*.npz | wc -l); done')
        elif a.action == 'a3-launch':
            running = busy()
            if running:
                print('\n'.join(running))
                raise SystemExit('a runner is still running on the host: nothing launched')
            others = [x for x in gpu_apps(1) if x['user'] != 'ours']
            if others:
                raise SystemExit(f'GPU 1 has processes of other users: {others}; nothing launched')
            free, used = gpu_free_gib()
            if free < GIB_PER_WORKER * 3 or used > 8:
                raise SystemExit(f'GPU 1 has {free:.1f} GiB free and {used:.1f} GiB in use: nothing launched')
            _, out = sh('cat /proc/loadavg', show=False)
            if float(out.split()[0]) > 40:
                raise SystemExit(f'load average {out.split()[0]}: nothing launched')
            sh(f'cd {run} && rm -rf preds/claims')                     # no runner alive: claims are stale
            am = f'--amendment {A3} --amendment-sha256 {sha(ROOT / A3)}'
            for w, order in enumerate(A3_ORDERS, start=1):
                chain = '; '.join(f'{py} -u v2_fm_zoo.py run --worker a3w{w}-{p} --out preds {A3_PLANS[p]} {am} '
                                  f'> logs/zoo_a3w{w}-{p}.out 2>&1' for p in order)
                sh(f"cd {run} && setsid -f bash -c 'export {env}; {chain}' < /dev/null > /dev/null 2>&1")
                time.sleep(20)                                          # staggered model loading
            time.sleep(5)
            print('\n'.join(busy()))
            sh('uptime; nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv')
        elif a.action == 'a3-rerun':
            if busy():
                raise SystemExit('a runner is still running on the host: nothing launched')
            detach(f'{py} -u v2_fm_zoo.py rerun --out preds --n {a.n} --arms '
                   f'timesfm3_uni,timesfm3_cov,timesfm25_uni,timesfm25_cov --require-amendment {A3} '
                   '--report rerun_check_a3.json', 'zoo_rerun_a3.out')
            time.sleep(5)
            print('\n'.join(busy()))
        elif a.action == 'a3-fetch':
            logs = HERE / 'run_logs'
            logs.mkdir(exist_ok=True)
            code, _ = sh(f'cd {run} && tar -czf fetch_a3.tar.gz preds/*__timesfm3_*.npz preds/A_*__timesfm25_*.npz '
                         f'preds/F_*__timesfm25_*.npz preds/*_L1440__timesfm25_*.npz preds/*.json '
                         f'{SUP_TFM3}/superseded.json {SUP_TFM25}/superseded.json $(ls preds/failed/*.json 2>/dev/null) '
                         'status logs/zoo_* upload_sha256.txt', show=False)
            if code:
                raise SystemExit('could not pack the outputs on the host')
            tmp = Path(tempfile.mkdtemp(prefix='fm_zoo_fetch_'))
            try:
                sftp = client.open_sftp()
                try:
                    sftp.get(f'{run}/fetch_a3.tar.gz', str(tmp / 'f.tar.gz'))
                finally:
                    sftp.close()
                sh(f'rm -f {run}/fetch_a3.tar.gz', show=False)
                n = 0
                with tarfile.open(tmp / 'f.tar.gz') as tar:
                    for m in tar.getmembers():
                        if not m.isfile() or '..' in m.name or m.name.startswith('/'):
                            continue
                        parts = m.name.split('/')
                        if parts[0] == 'preds' and len(parts) == 2:
                            target = PREDS / parts[1]
                        elif parts[0] == 'preds' and parts[1] == 'failed':
                            target = logs / ('failed_' + parts[-1])
                        elif parts[0] in (SUP_TFM3, SUP_TFM25):
                            target = logs / f'host_{parts[0]}_{parts[-1]}'   # the host's record of its move
                        else:
                            target = logs / parts[-1]
                        target.write_bytes(tar.extractfile(m).read())
                        n += m.name.endswith('.npz')
                print(f'fetched {n} prediction files into {PREDS.relative_to(ROOT)}; records into {logs.relative_to(ROOT)}')
            finally:
                shutil.rmtree(tmp, ignore_errors=True)
        elif a.action == 'a3-mirror':
            rec = {'checked_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), 'folders': {}}
            for folder in ('preds', SUP_TFM3, SUP_TFM25, DEVIATION):
                _, out = sh(f'cd {run}/{folder} && sha256sum *.npz', show=False)
                host = {ln.split()[1].lstrip('*'): ln.split()[0] for ln in out.splitlines() if ln.strip().endswith('.npz')}
                here = {p.name: sha(p) for p in sorted((HERE / folder).glob('*.npz'))}
                diff = sorted(k for k in set(host) | set(here) if host.get(k) != here.get(k))
                rec['folders'][folder] = {'host': len(host), 'here': len(here), 'identical': not diff,
                                          'differing_or_missing': diff[:50], 'n_differing_or_missing': len(diff)}
                print(folder, rec['folders'][folder]['host'], rec['folders'][folder]['here'],
                      'identical' if not diff else f'{len(diff)} differ or missing')
            (HERE / 'run_logs' / 'a3_mirror_check.json').write_bytes((json.dumps(rec, indent=1) + '\n').encode('utf-8'))
        elif a.action == 'status':
            sh(f'cd {run} && {py} v2_fm_zoo.py status; '
               f'echo; echo preds: $(ls preds/*.npz 2>/dev/null | wc -l) npz, smoke: $(ls smoke/*.npz 2>/dev/null | wc -l) npz; '
               f'ls preds/failed smoke/failed 2>/dev/null; '
               f'ps -eo pid,etime,nlwp,pcpu,rss,args | grep -E "[v]2_fm_zoo(_checks)?\\.py"; uptime; '
               f'nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv')
        elif a.action == 'inspect':
            sh(f'cd {run} && {env} {py} v2_fm_zoo.py inspect --out {a.folder}')
        elif a.action == 'fetch':
            folder = a.folder
            dest = PREDS if folder == 'preds' else HERE / folder
            dest.mkdir(parents=True, exist_ok=True)
            logs = HERE / 'run_logs'
            logs.mkdir(exist_ok=True)
            code, _ = sh(f'cd {run} && tar -czf fetch_{folder}.tar.gz {folder}/*.npz $(ls {folder}/*.json '
                         f'2>/dev/null) $(ls {folder}/failed/*.json 2>/dev/null) status logs/zoo_* upload_sha256.txt',
                         show=False)
            if code:
                raise SystemExit('could not pack the outputs on the host')
            tmp = Path(tempfile.mkdtemp(prefix='fm_zoo_fetch_'))
            try:
                sftp = client.open_sftp()
                try:
                    sftp.get(f'{run}/fetch_{folder}.tar.gz', str(tmp / 'f.tar.gz'))
                finally:
                    sftp.close()
                sh(f'rm -f {run}/fetch_{folder}.tar.gz', show=False)
                n = 0
                with tarfile.open(tmp / 'f.tar.gz') as tar:
                    for m in tar.getmembers():
                        if not m.isfile() or '..' in m.name or m.name.startswith('/'):
                            continue
                        parts = m.name.split('/')
                        if parts[0] == folder and len(parts) == 2:
                            target = dest / parts[1]
                        elif parts[0] == folder and parts[1] == 'failed':
                            target = logs / ('failed_' + parts[-1])
                        else:
                            target = logs / parts[-1]
                        target.write_bytes(tar.extractfile(m).read())
                        n += m.name.endswith('.npz')
                print(f'fetched {n} prediction files into {dest.relative_to(ROOT)}; logs and status into '
                      f'{logs.relative_to(ROOT)}')
            finally:
                shutil.rmtree(tmp, ignore_errors=True)
    finally:
        client.close()


if __name__ == '__main__':
    main()
