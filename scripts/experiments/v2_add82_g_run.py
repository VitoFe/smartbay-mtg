"""Addendum 82 part G on the compute host: the eight pretrained arms on the Dev-2025 utilisation, direct_raw
and T_raw inputs (18 files), through addendum 81's runner and adapters, unchanged.

The work directory holds byte-identical copies of v2_fm_zoo.py, v2_fm_zoo_models.py, v2_fm_zoo_io.py and
weights.json from the addendum-81 folder (checked against `--zoo`), the inputs and manifest_aou_dev.json under
inputs/, and this driver. Every file is run with task A's call options (the whole file in order, Chronos-Bolt
batch 256, Chronos-2 batch 128), as the Dev panel arms. Positivity clamps and TimesFM 3.0's batch follow
addendum 81 amendment 3: utilisation files, TimesFM 2.5 infer_is_positive False and TimesFM 3.0 one series per
call with make_positive False (the series can be negative); direct_raw and T_raw, the library defaults and
TimesFM 3.0 one series per call.

  python v2_add82_g_run.py check-f --zoo DIR     G-G3: Chronos-Bolt base, Chronos-2 uni and cov rerun on
                                                 addendum 81's task F input, against its declared outputs
  python v2_add82_g_run.py run                   every (input, arm) into preds/ (resumes)
  python v2_add82_g_run.py determinism [--n 256] each arm on the first n rows of one file per kind, again
  python v2_add82_g_run.py status

Declared in config/bay_reserved_protocol_v2_addendum82.json, part G.
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import v2_fm_zoo as zoo  # noqa: E402
import v2_fm_zoo_io as zio  # noqa: E402

AMENDMENT = 'config/bay_reserved_protocol_v2_addendum82.json'
MANIFEST = HERE / 'inputs' / 'manifest_aou_dev.json'
OFFSETS = range(6)
KINDS = ('G_oxygen_aou', 'G_oxygen_raw', 'G_temperature_raw')
CHRONOS = ('chronos_bolt_small_uni', 'chronos_bolt_base_uni', 'chronos2_uni', 'chronos2_cov')
TFM25 = ('timesfm25_uni', 'timesfm25_cov')
TFM3 = ('timesfm3_uni', 'timesfm3_cov')


def log(msg):
    line = f'{zoo.now()} {msg}'
    print(line, flush=True)
    (HERE / 'logs').mkdir(exist_ok=True)
    with open(HERE / 'logs' / 'driver.log', 'a', encoding='utf-8') as f:
        f.write(line + '\n')


def options(kind, arm):
    """(batch, tfm25_config, tfm3_flags) of one arm on one kind of file."""
    aou = kind == 'G_oxygen_aou'
    if arm in TFM25:
        return None, ({'infer_is_positive': False} if aou else {}), {}
    if arm in TFM3:
        return 1, {}, ({'make_positive': False} if aou else {})
    return None, {}, {}


def jobs():
    man = json.loads(MANIFEST.read_text(encoding='utf-8'))
    out = []
    for kind in KINDS:
        for o in OFFSETS:
            name = f'{kind}_offset{o}.npz'
            e = man['files'][name]
            for arm in CHRONOS + TFM25 + TFM3:
                out.append({'file': name, 'stem': name[:-4], 'task': 'A', 'arm': arm, 'n': e['rows'], 'horizon': 6,
                            'sha256': e['sha256'], 'manifest': MANIFEST.name, 'cost': 0.0, 'future_cov': False,
                            'kind': kind})
    return out


def runner(out):
    r = zoo.Runner('add82G', out, amendments=[AMENDMENT])
    return r


def set_opts(r, kind, arm):
    b, c25, f3 = options(kind, arm)
    r.batch, r.tfm25_config, r.tfm3_flags = b, c25, f3


def run(out):
    zoo.INPUTS = HERE / 'inputs'
    r = runner(out)
    js = jobs()
    log(f'{len(js)} jobs')
    # group by option set so the plan of each group is one loop of the runner
    groups = {}
    for j in js:
        groups.setdefault((j['kind'] == 'G_oxygen_aou', options(j['kind'], j['arm'])[0] or 0,
                           json.dumps(options(j['kind'], j['arm'])[1:], sort_keys=True)), []).append(j)
    for key, g in groups.items():
        set_opts(r, g[0]['kind'], g[0]['arm'])
        log(f'group {key}: {len(g)} jobs, overrides {r.overrides(g[0]["arm"])}')
        r.loop(g)
    log('run finished')


def check_f(zoo_dir, out):
    """Chronos-Bolt base, Chronos-2 uni and cov on addendum 81's F input, against its declared outputs."""
    zoo_dir = Path(zoo_dir)
    zoo.INPUTS = zoo_dir / 'inputs'
    r = runner(out)
    name = 'F_oxygen_aou_Reserved-2026.npz'
    m6 = json.loads((zoo.INPUTS / 'manifest_6h.json').read_text(encoding='utf-8'))
    sha = m6['files'][name]['sha256']
    rep = {'what': 'G-G3: the runner reproduces addendum 81 task F declared outputs', 'file': name, 'arms': {}}
    for arm in ('chronos_bolt_base_uni', 'chronos2_uni', 'chronos2_cov'):
        job = {'file': name, 'stem': name[:-4], 'task': 'F', 'arm': arm, 'n': None, 'horizon': 6, 'sha256': sha,
               'manifest': 'manifest_6h.json', 'cost': 0.0, 'future_cov': False}
        r.batch, r.tfm25_config, r.tfm3_flags = None, {}, {}
        data = r.inputs(job)
        t0 = time.time()
        med, q, _ = r.predict(job, data)
        stored = zio.load_predictions(zoo_dir / 'preds' / f'{name[:-4]}__{arm}.npz')
        c = zoo._compare(med, q, stored['median'], stored['q'])
        c['rows'] = int(len(med))
        c['seconds'] = round(time.time() - t0, 1)
        rep['arms'][arm] = c
        log(f'check-f {arm}: {json.dumps(c)}')
    rep['passed'] = all(v['below_1e-6'] for v in rep['arms'].values())
    (HERE / 'check_f.json').write_text(json.dumps(rep, indent=1) + '\n', encoding='utf-8')
    log(f'check-f passed {rep["passed"]}')


def determinism(out, n):
    """Each arm on the first n rows of each kind's offset-0 file, with the options its stored output records."""
    zoo.INPUTS = HERE / 'inputs'
    r = runner(out)
    rep = {'what': 'addendum 81 G3 on each new file kind: the first rows run again against the stored outputs',
           'rows': n, 'files': {}}
    for j in jobs():
        if not j['file'].endswith('offset0.npz'):
            continue
        p = Path(out) / f'{j["stem"]}__{j["arm"]}.npz'
        if not p.exists():
            rep['files'][f'{j["stem"]}__{j["arm"]}'] = {'error': 'no stored output'}
            continue
        stored = zio.load_predictions(p)
        r.set_overrides(stored['meta'].get('overrides'))
        data = r.inputs(j)
        rows = np.arange(min(n, data['n_run']))
        med, q, _ = r.predict(j, data, rows)
        c = zoo._compare(med, q, stored['median'][rows], stored['q'][rows])
        c['overrides'] = r.overrides(j['arm'])
        rep['files'][f'{j["stem"]}__{j["arm"]}'] = c
        log(f'determinism {j["stem"]}__{j["arm"]}: {c["below_1e-6"]} {c["max_abs_diff_median"]:.2e}')
    rep['all_below_1e-6'] = all(v.get('below_1e-6', False) for v in rep['files'].values())
    (HERE / 'determinism.json').write_text(json.dumps(rep, indent=1) + '\n', encoding='utf-8')
    log(f'determinism all below 1e-6: {rep["all_below_1e-6"]}')


def status(out):
    js = jobs()
    done = [j for j in js if (Path(out) / f'{j["stem"]}__{j["arm"]}.npz').exists()]
    failed = sorted(p.name for p in (Path(out) / 'failed').glob('*.json')) if (Path(out) / 'failed').exists() else []
    print(f'{len(done)}/{len(js)} written, {len(failed)} failed {failed[:5]}')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('action', choices=('check-f', 'run', 'determinism', 'status'))
    ap.add_argument('--zoo', default=str(HERE.parent / 'v2_fm_zoo'))
    ap.add_argument('--out', default=str(HERE / 'preds'))
    ap.add_argument('--n', type=int, default=256)
    a = ap.parse_args()
    if a.action != 'status':
        for name in ('v2_fm_zoo.py', 'v2_fm_zoo_models.py', 'v2_fm_zoo_io.py', 'weights.json'):
            mine, theirs = zio.digest(HERE / name), zio.digest(Path(a.zoo) / name)
            if mine != theirs:
                raise SystemExit(f'{name} differs from {a.zoo}: copy it unchanged')
    if a.action == 'check-f':
        check_f(a.zoo, a.out)
    elif a.action == 'run':
        run(a.out)
    elif a.action == 'determinism':
        determinism(a.out, a.n)
    else:
        status(a.out)


if __name__ == '__main__':
    main()
