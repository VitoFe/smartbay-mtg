"""smartbay - one entry point for the whole pipeline.

    python smartbay.py plan                     what would run, in what order, at what cost
    python smartbay.py run --stage metrology    run one stage and its dependencies
    python smartbay.py run --stage all --tier full
    python smartbay.py provenance               trace every table to its artifact
    python smartbay.py verify                   provenance, references and prose in one pass
    python smartbay.py verify --full            also the component replay and every table regenerated
    python smartbay.py numbers                  which printed decimals appear in the cited artifacts
    python smartbay.py tables                   regenerate each generated table and compare
    python smartbay.py component                check and replay the frozen component contract

Determinism. Every producer is invoked as `python -m <module>` in a subprocess with a fixed
PYTHONHASHSEED and a recorded seed, so a run is reproducible from the manifest alone. Each run
writes reports/bay_twin/runs/pipeline_manifest_<timestamp>.json holding the git revision, the
tier, the stages attempted, the exit status of each producer and the digest of every artifact
afterwards.

Tiers. `full` is the published protocol: at least 100 completed Optuna trials per arm, hours of
cluster time for the heavy stages. `smoke` checks that the code executes. A smoke run refuses to
touch a release stage, because an artifact produced under a reduced protocol that overwrites a
published one is indistinguishable from the real thing afterwards.
"""
import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from src.pipeline import component, declarations, doctor, inputs, numbers, provenance, tables
from src.pipeline.registry import (STAGE_BY_NAME, STAGES, artifact_path, frozen_inputs,
                                   producer_command, resolve)

ROOT = Path(__file__).resolve().parents[2]
RUNS = ROOT / 'reports/bay_twin/runs'
DEFAULT_SEED = 20260921


def git_revision():
    try:
        out = subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=ROOT,
                             capture_output=True, text=True, timeout=10)
        rev = out.stdout.strip()
        dirty = subprocess.run(['git', 'status', '--porcelain'], cwd=ROOT,
                               capture_output=True, text=True, timeout=20).stdout.strip()
        return {'revision': rev or None, 'clean': not dirty}
    except Exception:
        return {'revision': None, 'clean': None}


def plan(stages, tier):
    order = resolve(stages)
    total = sum(s.minutes for s in order)
    print(f'{len(order)} stage(s), about {total} minute(s) at tier `full`:\n')
    for i, s in enumerate(order, 1):
        flags = []
        if s.needs_cluster:
            flags.append('cluster')
        if s.release_only:
            flags.append('release-only')
        tag = f"  [{', '.join(flags)}]" if flags else ''
        print(f'  {i}. {s.name:12s} ~{s.minutes:>4d} min{tag}')
        print(f'      {s.summary}')
        if s.depends_on:
            print(f'      after: {", ".join(s.depends_on)}')
        for p in s.producers:
            if not module_present(p):
                print(f'      - {p} (not in this checkout; skipped)')
                continue
            print(f'      $ python {" ".join(producer_command(p))}')
            for path in frozen_inputs(p):
                print(f'          needs {path} (frozen repository only)')
        if s.notes:
            print(f'      note: {s.notes}')
        print()
    blocked = [s.name for s in order if s.release_only] if tier == 'smoke' else []
    if blocked:
        print(f'at tier `smoke` these would be skipped: {", ".join(blocked)}')
    return order


def module_present(module):
    """Whether this checkout holds the producer's module. The public code release leaves out the
    producers that only typeset the manuscript's tables and figures; `run` skips them."""
    path = ROOT.joinpath(*module.split('.'))
    return path.with_suffix('.py').is_file() or (path / '__init__.py').is_file()


def run_producer(module, seed, timeout):
    env = dict(os.environ, PYTHONHASHSEED=str(seed), SMARTBAY_SEED=str(seed),
               PYTHONIOENCODING='utf-8')
    started = time.time()
    try:
        proc = subprocess.run([sys.executable, *producer_command(module)], cwd=ROOT, env=env,
                              timeout=timeout, capture_output=True, text=True)
        code, tail = proc.returncode, (proc.stdout or '') + (proc.stderr or '')
    except subprocess.TimeoutExpired:
        code, tail = 124, f'timed out after {timeout}s'
    return {'module': module, 'command': producer_command(module), 'exit_code': code,
            'seconds': round(time.time() - started, 1),
            'tail': tail.strip().splitlines()[-12:]}


def run(stages, tier, seed, timeout, keep_going):
    order = resolve(stages)
    if tier == 'smoke':
        skipped = [s.name for s in order if s.release_only]
        order = [s for s in order if not s.release_only]
        if skipped:
            print(f'tier `smoke`: skipping release-only stage(s) {", ".join(skipped)}\n')

    started = datetime.now(timezone.utc)
    record = {'started': started.isoformat(), 'tier': tier, 'seed': seed,
              'git': git_revision(), 'stages': []}

    ok = True
    for stage in order:
        print(f'== {stage.name}: {stage.summary}')
        entry = {'stage': stage.name, 'producers': []}
        for module in stage.producers:
            if not module_present(module):
                print(f'   -- {module}: not in this checkout, skipped')
                entry['producers'].append({'module': module, 'skipped': 'not in this checkout'})
                continue
            print(f'   -> {module}', flush=True)
            absent = frozen_inputs(module)
            if absent:
                # An input held only in the frozen repository would end in a FileNotFoundError
                # deep inside the producer; say which, and do not start it.
                result = {'module': module, 'exit_code': 2, 'seconds': 0.0,
                          'tail': ['not run: inputs held only in the frozen repository: '
                                   + ', '.join(absent)]}
            else:
                result = run_producer(module, seed, timeout)
            entry['producers'].append(result)
            mark = 'ok' if result['exit_code'] == 0 else f"FAILED ({result['exit_code']})"
            print(f"      {mark} in {result['seconds']}s")
            if result['exit_code'] != 0:
                for line in result['tail']:
                    print(f'      | {line}')
                ok = False
                if not keep_going:
                    record['stages'].append(entry)
                    break
        entry['artifacts'] = {a: provenance.digest(artifact_path(a)) for a in stage.artifacts}
        record['stages'].append(entry)
        if not ok and not keep_going:
            break

    record['finished'] = datetime.now(timezone.utc).isoformat()
    record['succeeded'] = ok
    RUNS.mkdir(parents=True, exist_ok=True)
    path = RUNS / f"pipeline_manifest_{started.strftime('%Y%m%dT%H%M%SZ')}.json"
    path.write_text(json.dumps(record, indent=2))
    print(f"\n{'all stages succeeded' if ok else 'one or more producers failed'}")
    print('manifest:', path.relative_to(ROOT))
    return 0 if ok else 1


def verify(allow_drift=False, full=False):
    failures = 0
    print('== inputs')
    failures += inputs.report() != 0
    print('\n== provenance')
    failures += provenance.report(allow_drift=allow_drift) != 0
    print('\n== declarations')
    failures += declarations.report() != 0
    print('\n== component contract')
    failures += component.check(replay=full) != 0
    if full:
        print('\n== tables (each generator rerun into a scratch folder)')
        failures += tables.check() != 0
    for label, script in [('references', 'thesis_template/check_references.py'),
                          ('citations', 'thesis_template/check_citations.py'),
                          ('prose', 'thesis_template/check_prose.py')]:
        print(f'\n== {label}')
        proc = subprocess.run([sys.executable, str(ROOT / script)], cwd=ROOT,
                              capture_output=True, text=True)
        print((proc.stdout or proc.stderr).strip()[-2000:])
        failures += proc.returncode != 0
    print('\n' + ('verification passed' if not failures else f'{failures} check(s) failed'))
    return 1 if failures else 0


def build_parser():
    p = argparse.ArgumentParser(prog='smartbay', description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest='command', required=True)

    names = f"{', '.join(STAGE_BY_NAME)}, all"
    for cmd, helptext in [('plan', 'show what would run without running it'),
                          ('run', 'run the stages')]:
        s = sub.add_parser(cmd, help=helptext)
        s.add_argument('--stage', nargs='+', default=['all'], metavar='NAME',
                       help=f'one or more of: {names} (dependencies are pulled in)')
        s.add_argument('--tier', choices=('smoke', 'full'), default='full',
                       help='`full` is the published protocol; `smoke` only checks execution')
        if cmd == 'run':
            s.add_argument('--seed', type=int, default=DEFAULT_SEED)
            s.add_argument('--timeout', type=int, default=36000,
                           help='per-producer wall-clock limit in seconds')
            s.add_argument('--keep-going', action='store_true',
                           help='continue after a producer fails')

    drift = ('report artifacts changed since the lockfile, or not yet in it, without failing '
             '(between regenerating an artifact and re-reading what cites it)')
    pr = sub.add_parser('provenance', help='trace every thesis table and figure to its artifact')
    pr.add_argument('--update', action='store_true', help='rewrite the lockfile')
    pr.add_argument('--verbose', action='store_true', help='list every citation')
    pr.add_argument('--allow-drift', action='store_true', help=drift)

    dr = sub.add_parser('doctor', help='which modules the thesis reaches, and which are left over')
    dr.add_argument('--all', action='store_true', help='list every candidate, not the largest few')
    dr.add_argument('--plan', action='store_true',
                    help='write the retirement analysis to an artifact instead of printing it')

    vr = sub.add_parser('verify', help='inputs, provenance, declarations, the component contract, '
                                       'references, citations and prose in one pass')
    vr.add_argument('--allow-drift', action='store_true', help=drift)
    vr.add_argument('--full', action='store_true',
                    help='also replay the component contract and regenerate every generated table '
                         'into a scratch folder (several minutes)')
    co = sub.add_parser('component', help='check the frozen component contract and replay its '
                                          'verifier without writing anything')
    co.add_argument('--digests-only', action='store_true', help='skip the replay')
    tb = sub.add_parser('tables', help='regenerate every generated table into a scratch folder and '
                                       'compare it with the committed one')
    tb.add_argument('--only', nargs='+', metavar='MODULE', help='only these generators')
    tb.add_argument('--jobs', type=int, default=1, help='generators run at once')
    tb.add_argument('--keep', type=Path, help='keep the regenerated files in this folder')
    nu = sub.add_parser('numbers', help='report (never fail) which printed decimals appear in the '
                                        'artifacts their table or paragraph cites')
    nu.add_argument('--verbose', action='store_true', help='list every unmatched decimal')
    nu.add_argument('--json', type=Path, help='also write the full result to this file')
    sub.add_parser('stages', help='list the declared stages')
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    if args.command == 'stages':
        for s in STAGES:
            dep = f" (after {', '.join(s.depends_on)})" if s.depends_on else ''
            print(f'{s.name:12s} ~{s.minutes:>4d} min{dep}\n    {s.summary}')
        return 0
    if args.command == 'plan':
        plan(args.stage, args.tier)
        return 0
    if args.command == 'run':
        return run(args.stage, args.tier, args.seed, args.timeout, args.keep_going)
    if args.command == 'provenance':
        return 1 if provenance.report(args.update, args.verbose, args.allow_drift) else 0
    if args.command == 'doctor':
        return doctor.write_plan() if args.plan else doctor.report(args.all)
    if args.command == 'verify':
        return verify(args.allow_drift, args.full)
    if args.command == 'component':
        return 1 if component.check(replay=not args.digests_only) else 0
    if args.command == 'tables':
        return 1 if tables.check(args.only, args.jobs, args.keep) else 0
    if args.command == 'numbers':
        numbers.report(args.verbose, args.json)
        return 0
    return 1


if __name__ == '__main__':
    raise SystemExit(main())
