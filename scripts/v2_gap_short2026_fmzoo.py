"""Addendum 82, part C (C5): the eight pretrained arms of addendum 81 on the 1-3 h Gap-2026 rows.

  inputs    (local) one context per (episode, direction) of the 1-3 h rows, built by addendum 81's
            gap input builder (scripts/v2_fm_zoo_inputs_gaps.build and write, unchanged) with its row
            reader pointed at this part's run, and its checks: E1 rows, E4 chronos_fill with a
            recording stand-in, E5 Chronos-2 on CPU on the probes whose published neighbour order
            differs, E6 round trip (E2 and E3 compare with inputs that exist only for the published
            6-72 h rows). Files E_{variable}_Gap-2026_{fwd|bwd}_L{1,2,3}.npz and manifest_gaps.json
            under runs/v2_addendum82/C/fm_zoo/inputs/
  upload    a run folder of its own on the host, <base>/review_20260919/v2_add82_C/fm_zoo, with the
            addendum-81 runner, model layer, format module and weights.json copied from the host's
            addendum-81 folder after checking them against the repository, the inputs and the
            amendment-3 declaration the TimesFM 3.0 outputs cite
  launch    one process on GPU 1, as addendum 81 ran task E: the six arms other than TimesFM 3.0 with
            the declared options, then TimesFM 3.0 one series per call (amendment 3)
  status    the runner's outputs so far, the GPU and the load
  fetch     the predictions into runs/v2_addendum82/C/fm_zoo/preds/, hash-verified, and a digest
            manifest of every .npz of this part (runs/v2_addendum82/C/npz_manifest.json)

Usage: python -m scripts.v2_gap_short2026_fmzoo <action>
"""
import argparse
import contextlib
import hashlib
import json
import os
import shlex
import sys
import tarfile
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
from scripts.v2_gap_short2026_rows import PART, SHORT, SHORT_RUN  # noqa: E402

HERE = PART / 'fm_zoo'
INPUTS = HERE / 'inputs'
PREDS = HERE / 'preds'
MANIFEST = PART / 'npz_manifest.json'
REMOTE = 'review_20260919/v2_add82_C/fm_zoo'
A81 = 'review_20260919/v2_fm_zoo'
CODE = {'v2_fm_zoo.py': ROOT / 'scripts/experiments/v2_fm_zoo.py',
        'v2_fm_zoo_models.py': ROOT / 'scripts/experiments/v2_fm_zoo_models.py',
        'v2_fm_zoo_io.py': ROOT / 'scripts/v2_fm_zoo_io.py'}
AMENDMENT3 = 'config/bay_reserved_protocol_v2_addendum81_amendment3.json'
ENV = ('CUDA_VISIBLE_DEVICES=1 HF_HOME={base}/.hf HF_HUB_OFFLINE=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 '
       'OPENBLAS_NUM_THREADS=4 V2_THREADS=4 PYTHONDONTWRITEBYTECODE=1')
FIRST = 'chronos_bolt_small_uni,chronos_bolt_base_uni,chronos2_uni,chronos2_cov,timesfm25_uni,timesfm25_cov'
TFM3 = 'timesfm3_uni,timesfm3_cov'
VARIABLES = ('temperature', 'oxygen')
COHORT = 'Gap-2026'


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def short_published_rows(cohort, variable):
    """Addendum 81's row reader, pointed at this part's 1-3 h run (same checks, same columns)."""
    if cohort != COHORT:
        raise SystemExit('only Gap-2026 has 1-3 h rows')
    bolt = pd.read_csv(SHORT / f'{variable}_chronos_bidir' / 'predictions.csv.gz')
    c2 = pd.read_csv(SHORT / f'{variable}_chronos2_cov_bidir' / 'predictions.csv.gz')
    keys = ['episode', 'position']
    identical = bool(bolt[keys].equals(c2[keys]))
    sources = [SHORT / f'{variable}_chronos_bidir' / 'predictions.csv.gz',
               SHORT / f'{variable}_chronos2_cov_bidir' / 'predictions.csv.gz']
    rows = bolt[['episode', 'sensor', 'gap_length', 'position', 'fold', 'time']].copy()
    rows['chronos_bidir'] = bolt['chronos_bidir'].to_numpy()
    rows['chronos2_cov_bidir'] = c2['chronos2_cov_bidir'].to_numpy()
    return rows, identical, sources


def inputs():
    import scripts.v2_fm_zoo_inputs_gaps as IG
    os.chdir(ROOT)
    IG.published_rows = short_published_rows
    IG.OUT = INPUTS
    cache, data, gates = {}, {}, {}
    for variable in VARIABLES:
        built = IG.build(COHORT, variable, cache)
        data[(COHORT, variable)] = built
        cell = {'E1_rows': built['e1'], 'E4_function': IG.gate_function(COHORT, variable, built)}
        gates[f'{COHORT} {variable}'] = cell
        print(variable, json.dumps({k: v.get('passed') for k, v in cell.items()}), flush=True)
    gates['E5_model'] = IG.gate_model(data, True)
    print('E5', json.dumps({k: v for k, v in gates['E5_model'].items() if k != 'cells'}), flush=True)
    failed = [f'{c} {g}' for c, cell in gates.items() if c != 'E5_model'
              for g, v in cell.items() if not v.get('passed')]
    if gates['E5_model'].get('state') != 'run' or not gates['E5_model'].get('passed'):
        failed.append('E5_model')
    if failed:
        raise SystemExit(f'gates failed, nothing written: {failed}')
    INPUTS.mkdir(parents=True, exist_ok=True)
    files = {}
    for (cohort, variable), built in data.items():
        files.update(IG.write(cohort, variable, built))
    bad = [n for n, e in files.items() if not e['roundtrip_passed']]
    gates['E6_roundtrip'] = {'passed': not bad, 'files': len(files), 'failed': bad}
    if bad:
        raise SystemExit(f'round trip failed: {bad}')
    manifest = {'declared_in': 'config/bay_reserved_protocol_v2_addendum82.json', 'part': 'C',
                'task': 'E_gaps', 'builder': 'scripts/v2_gap_short2026_fmzoo.py (inputs), through '
                                             'scripts/v2_fm_zoo_inputs_gaps.build and write',
                'rows_from': f'runs/v2_addendum82/C/remote/{SHORT_RUN} (short)',
                'cohort': COHORT, 'lengths': [1, 2, 3],
                'files': files,
                'totals': {'files': len(files), 'contexts': int(sum(e['n'] for e in files.values()))},
                'gates': gates}
    (INPUTS / 'manifest_gaps.json').write_bytes((json.dumps(manifest, indent=2, default=str) + '\n').encode())
    # the runner reads all three manifests; this folder holds gap inputs only
    (INPUTS / 'manifest_6h.json').write_bytes(b'{"files": {}}\n')
    (INPUTS / 'manifest_dist_bridge.json').write_bytes(
        b'{"C_distribution": {"files": {}}, "D_silent_probe": {"files": {}}}\n')
    print(len(files), 'files,', manifest['totals']['contexts'], 'contexts')


def session(tries=6):
    from scientific_campaign import connect
    for i in range(tries):
        try:
            return connect()
        except Exception as exc:  # the tunnel drops sessions now and then
            print(f'[connect attempt {i + 1} failed: {type(exc).__name__}]', flush=True)
            time.sleep(4 + 3 * i)
    raise SystemExit('compute host unreachable')


def execute(client, command):
    from scientific_campaign import execute as run
    return run(client, command)


def upload():
    client, base = session()
    try:
        remote, a81 = f'{base}/{REMOTE}', f'{base}/{A81}'
        code, out = execute(client, f'mkdir -p {remote}/inputs {remote}/preds {remote}/logs {remote}/config && '
                                    f'cd {a81} && sha256sum ' + ' '.join(CODE))
        if code:
            raise SystemExit(out)
        host = {line.split()[1]: line.split()[0] for line in out.strip().splitlines()}
        mismatch = [n for n, p in CODE.items() if host.get(n) != sha(p)]
        if mismatch:
            raise SystemExit(f'the host runner differs from the repository: {mismatch}')
        code, out = execute(client, f'cp {a81}/v2_fm_zoo.py {a81}/v2_fm_zoo_models.py {a81}/v2_fm_zoo_io.py '
                                    f'{a81}/weights.json {remote}/ && cd {remote} && sha256sum ' + ' '.join(CODE))
        if code:
            raise SystemExit(out)
        sftp = client.open_sftp()
        try:
            names = sorted(p.name for p in INPUTS.iterdir() if p.suffix in ('.npz', '.json'))
            for name in names:
                sftp.put(str(INPUTS / name), f'{remote}/inputs/{name}')
            sftp.put(str(ROOT / AMENDMENT3), f'{remote}/{AMENDMENT3}')
        finally:
            sftp.close()
        local = {f'inputs/{n}': sha(INPUTS / n) for n in names}
        local[AMENDMENT3] = sha(ROOT / AMENDMENT3)
        check = '\n'.join(f'{h}  {n}' for n, h in local.items()) + '\n'
        code, out = execute(client, f'cd {remote} && printf %s {shlex.quote(check)} > upload_sha256.txt && '
                                    f'sha256sum -c upload_sha256.txt | tail -3')
        print(out.replace(base, '<base>').strip())
        if code:
            raise SystemExit('upload check failed')
    finally:
        client.close()


def launch():
    client, base = session()
    try:
        remote = f'{base}/{REMOTE}'
        code, out = execute(client, 'nvidia-smi --query-gpu=index,memory.used,utilization.gpu '
                                    '--format=csv,noheader; nvidia-smi --query-compute-apps=pid,used_memory '
                                    '--format=csv,noheader')
        print(out.strip())
        py = base + '/.venv/bin/python'
        env = ENV.format(base=base)
        a3 = sha(ROOT / AMENDMENT3)
        first = f'env {env} {py} -u v2_fm_zoo.py run --worker c82 --out preds --tasks E --arms {FIRST}'
        second = (f'env {env} {py} -u v2_fm_zoo.py run --worker c82t3 --out preds --tasks E --arms {TFM3} '
                  f'--batch 1 --amendment {AMENDMENT3} --amendment-sha256 {a3}')
        cmd = f'cd {remote} && ({first} > logs/c82.log 2>&1; {second} > logs/c82t3.log 2>&1; echo done > logs/finished)'
        code, out = execute(client, f'setsid -f bash -c {shlex.quote(cmd)} < /dev/null > /dev/null 2>&1')
        if code:
            raise SystemExit(out)
        time.sleep(5)
        code, out = execute(client, "uptime; ps -eo nlwp,pcpu,args | grep '[v]2_fm_zoo.py run' | cut -c1-140")
        print(out.replace(base, '<base>'))
        (HERE / 'transport.json').write_bytes((json.dumps(
            {'remote_dir': f'<base>/{REMOTE}', 'commands': [first.replace(base, '<base>'),
                                                           second.replace(base, '<base>')],
             'launched': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}, indent=2) + '\n').encode())
    finally:
        client.close()


def status():
    client, base = session()
    try:
        remote = f'{base}/{REMOTE}'
        code, out = execute(client, f'cd {remote} && ls preds | wc -l; ls logs; cat logs/finished 2>/dev/null; '
                                    f"grep -c -E 'Traceback|Error' logs/*.log; uptime; "
                                    f'nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader')
        print(out.replace(base, '<base>'))
    finally:
        client.close()


def fetch():
    client, base = session()
    try:
        remote = f'{base}/{REMOTE}'
        code, out = execute(client, f"cd {remote} && tar czf preds_download.tar.gz preds logs && "
                                    f"sha256sum preds_download.tar.gz")
        if code:
            raise SystemExit(out)
        want = out.split()[0]
        HERE.mkdir(parents=True, exist_ok=True)
        local = HERE / 'preds_download.tar.gz'
        sftp = client.open_sftp()
        try:
            sftp.get(f'{remote}/preds_download.tar.gz', str(local))
        finally:
            sftp.close()
        if sha(local) != want:
            raise SystemExit('download checksum mismatch')
        with tarfile.open(local) as a:
            a.extractall(HERE, filter='data')
        local.unlink()
        print('fetched', len(list(PREDS.glob('*.npz'))), 'prediction files')
    finally:
        client.close()
    write_manifest()


def write_manifest():
    files = {p.relative_to(PART).as_posix(): {'bytes': p.stat().st_size, 'sha256': sha(p)}
             for p in sorted(PART.rglob('*.npz'))}
    MANIFEST.write_bytes((json.dumps(
        {'note': 'digests of every .npz of addendum 82 part C, which git does not carry: the contexts '
                 'are observed values and the prediction arrays are model output',
         'files': files}, indent=2) + '\n').encode())
    print(len(files), 'npz files in', MANIFEST.relative_to(ROOT))


def score():
    """C5 with addendum 81's gap scorer (scripts/v2_fm_zoo_score_gaps, unchanged): the 1-3 h cells on
    whole-cohort clusters, the 2026 1-6 h band on band clusters with the published 6 h fills, gates
    G1 (the new fills reproduce this part's Chronos columns to 1e-4), G2 (every file present and
    matching its input) and R (the six-decimal replica agrees with the published function); then
    addendum 81's false-discovery family recomputed with these cells in place of the 6 h-only band."""
    import scripts.v2_fm_zoo_score_gaps as FZ
    import scripts.v2_fm_zoo_summary as SUM
    from scripts.v2_gap_intervals import clusters
    from src.units import factor
    gate = json.loads((PART / 'v2_gap_short2026_gate.json').read_text())
    if not gate['passed']:
        raise SystemExit('the reproduction gates have not passed')
    arms = list(FZ.ARMS)
    fills = [f'{arm}.{mode}' for arm in arms for mode in FZ.MODES]
    c_mg = factor()
    published = {'INP': FZ.INP, 'PRED': FZ.PRED, 'MANIFEST': FZ.MANIFEST}
    mine = {'INP': INPUTS, 'PRED': PREDS, 'MANIFEST': INPUTS / 'manifest_gaps.json'}
    out = {'declared_in': 'config/bay_reserved_protocol_v2_addendum82.json', 'part': 'C', 'contrast': 'C5',
           'scorer': 'scripts/v2_fm_zoo_score_gaps (score_cell, assemble, gate_g1), unchanged',
           'files': {'expected': 96, 'present': len(list(PREDS.glob('*.npz')))},
           'gates': {}, 'cells': {}}
    replica = []
    sg = json.loads(FZ.OUT.read_text())
    for variable in VARIABLES:
        g2 = {'problems': [], 'files_checked': 0, 'files_expected': 0}
        # the published 6-72 h rows and their fills (6 h enters the band), then the 1-3 h rows
        pub = FZ.load(COHORT, variable).reset_index(drop=True)
        fwd = FZ.read_arm(FZ.COHORTS[COHORT][0], FZ.COHORTS[COHORT][1], variable, 'chronos_forward')
        pub['chronos_forward'] = fwd['chronos_forward'].to_numpy()
        six = pub[pub.gap_length == 6].reset_index(drop=True)
        for k, v in published.items():
            setattr(FZ, k, v)
        six = FZ.assemble(six, COHORT, variable, arms, json.loads(FZ.MANIFEST.read_text()),
                          {'problems': [], 'files_checked': 0, 'files_expected': 0}, {})
        cols = ['episode', 'sensor', 'has_twin', 'gap_length', 'position', 'fold', 'time', 'truth', 'linear']
        short = pd.read_csv(SHORT / f'{variable}_chronos_bidir' / 'predictions.csv.gz')[
            cols + ['chronos_bidir']]
        for arm in ('chronos2_cov_bidir', 'chronos_forward'):
            f = pd.read_csv(SHORT / f'{variable}_{arm}' / 'predictions.csv.gz')
            if not f[['episode', 'position']].equals(short[['episode', 'position']]):
                raise SystemExit(f'{variable} {arm}: other rows')
            short[arm] = f[arm].to_numpy()
        for k, v in mine.items():
            setattr(FZ, k, v)
        g2['files_expected'] = 2 * 3 * len(arms)
        short = FZ.assemble(short, COHORT, variable, arms, json.loads(FZ.MANIFEST.read_text()), g2, {})
        for k, v in published.items():
            setattr(FZ, k, v)
        g1, informative = FZ.gate_g1(short, COHORT, variable, fills)
        missing = int(sum(short[f].isna().sum() for f in fills))
        g2['fills_missing'] = missing
        g2['passed'] = bool(not g2['problems'] and g2['files_checked'] == g2['files_expected'] and missing == 0)
        out['gates'][variable] = {'C-G7_G1': g1, 'G1_informative': informative, 'G2': g2,
                                  'passed': bool(g2['passed'] and all(v['passed'] for v in g1.values()))}
        if not out['gates'][variable]['passed']:
            continue
        whole = clusters(pd.concat([pub[['episode', 'sensor', 'time']], short[['episode', 'sensor', 'time']]],
                                   ignore_index=True))
        cells = {}
        for L in (1, 2, 3):
            part = short[short.gap_length == L]
            cells[str(L)] = FZ.score_cell(part, whole, 'overlap cluster (whole cohort)', variable, L,
                                          f'{COHORT}|{variable}|{L}', fills, arms, c_mg, replica)
        band = pd.concat([six[short.columns.intersection(six.columns)], short], ignore_index=True)
        cells['1-6'] = FZ.score_cell(band, clusters(band), 'overlap cluster (band)', variable, '1-6',
                                     f'{COHORT}|{variable}|1-6', fills, arms, c_mg, replica)
        out['cells'][variable] = cells
    out['gates']['R_replica'] = bool(replica and all(r['agree'] for r in replica))
    out['gates']['passed'] = all(out['gates'][v]['passed'] for v in VARIABLES) and out['gates']['R_replica'] \
        and out['files']['present'] == 96
    if out['gates']['passed']:
        # addendum 81's family, recomputed with the 1-3 h cells and the 1-6 h band over 1, 2, 3, 6 h
        # addendum 81's family as published: the summary's family over its three score files alone (the
        # summary itself now holds these cells, so its output is not read here)
        fam0 = SUM.Family(factor())
        SUM.collect(*(SUM.load(SUM.SOURCES[k]) for k in ('6h', 'dist_bridge', 'gaps')), fam0)
        before = SUM.adjust(fam0)
        srcs = {k: SUM.load(p) for k, p in SUM.SOURCES.items()}
        for variable in VARIABLES:
            srcs['gaps']['results'][COHORT][variable]['cells'].update(out['cells'][variable])
        fam = SUM.Family(factor())
        SUM.collect(srcs['6h'], srcs['dist_bridge'], srcs['gaps'], fam)
        fdr = SUM.adjust(fam)
        new = [r for r in fam.items.values() if r['task'] == 'E' and r['cohort'] == COHORT
               and r['cell'] in ('1', '2', '3', '1-6') and r['estimable'] and not r.get('reproduction')]
        out['summary'] = {'addendum81_family_published': before.get('counts'),
                          'addendum81_family_recomputed': fdr['counts'],
                          'cells_of_this_part': {'intervals': len(new),
                                                 'exclude_zero': sum(r['excludes_zero'] for r in new),
                                                 'bh_retained': sum(r['bh'] for r in new),
                                                 'by_retained': sum(r['by'] for r in new)},
                          'note': 'counted in addendum 81\'s family, not in part C\'s'}
        out['family_flags'] = {'|'.join(map(str, r['id'])) if isinstance(r['id'], (list, tuple)) else str(r['id']):
                               {'bh': r['bh'], 'by': r['by'], 'p': r6p(r['p'])} for r in new}
    (PART / 'pretrained.json').write_bytes((json.dumps(out, indent=2, default=str) + '\n').encode('utf-8'))
    print('C5 gates', out['gates'].get('passed'), {v: out['gates'][v]['passed'] for v in VARIABLES},
          'R', out['gates']['R_replica'])
    print('family', out.get('summary'))


def r6p(x):
    return None if x is None else round(float(x), 9)


def main():
    p = argparse.ArgumentParser()
    # without an action it scores, as the pipeline runs it (the other actions build inputs or drive the cluster)
    p.add_argument('action', nargs='?', default='score',
                   choices=['inputs', 'upload', 'launch', 'status', 'fetch', 'manifest', 'score'])
    a = p.parse_args()
    {'inputs': inputs, 'upload': upload, 'launch': launch, 'status': status, 'fetch': fetch,
     'manifest': write_manifest, 'score': score}[a.action]()


if __name__ == '__main__':
    main()
