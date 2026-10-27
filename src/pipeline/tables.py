"""Regenerate every generated thesis table into a scratch folder and compare it with the committed one.

The provenance check proves that the artifact under a table has not changed since the table was
read, and src/pipeline/numbers.py reports which printed decimals can be found in it. Neither shows
that the table file still says what its generator would write today: a generator edited after its
last run, or a table touched by hand after it was generated, passes both. This does.

A generator is any first-party module that writes into thesis_template/content/tables/ (found by
that path in its source, so a new one is picked up without registering it here). Each runs in a
child process under src/pipeline/sandbox.py: every file it writes goes to a scratch folder, and any
write the redirection cannot follow is refused, so the committed tables and artifacts are never
touched. Each table it wrote is then compared with the committed file, line endings aside.

Outcomes, per generator:
    identical   every table it wrote matches the committed one
    DIFFERS     a table it wrote does not (the first changed lines are printed)
    skipped     it stopped on an input absent here, typically the data folder, which is not
                distributed, or a run held elsewhere; the missing path is named
    FAILED      it stopped for any other reason, including a refused write
Tables no generator wrote are listed as typed by hand. DIFFERS and FAILED count as failures.

    python -m src.pipeline.tables [--only MODULE ...] [--jobs N] [--keep DIR]
"""
import argparse
import difflib
import os
import re
import shutil
import subprocess
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from src.pipeline.registry import OPERATOR_TOOLS, PRODUCER_ARGS

ROOT = Path(__file__).resolve().parents[2]
TABLES = ROOT / 'thesis_template/content/tables'
TABLES_REL = TABLES.relative_to(ROOT)
#: Arguments a generator needs to write the thesis's copy of its table rather than a draft elsewhere.
GENERATOR_ARGS = {
    'scripts.v2_perfold_neural_table': ('--write-table',),
}
MISSING = re.compile(r"No such file or directory: '([^']+)'|FileNotFoundError: (.+)")


def generators():
    """Modules whose source names the thesis tables folder, in a stable order. Operator tools are
    left out: they name table files to check or bind them, not to write them."""
    found = []
    for folder in ('scripts', 'src'):
        for path in sorted((ROOT / folder).rglob('*.py')):
            if '__pycache__' in path.parts or path.parent == Path(__file__).parent:
                continue
            module = '.'.join(path.relative_to(ROOT).with_suffix('').parts)
            if module in OPERATOR_TOOLS:
                continue
            if 'content/tables' in path.read_text(encoding='utf-8', errors='replace'):
                found.append(module)
    return found


def _norm(path):
    return path.read_bytes().replace(b'\r\n', b'\n').decode('utf-8', 'replace')


def run_one(module, scratch, timeout):
    mirror = scratch / module.replace('.', '_')
    args = [*PRODUCER_ARGS.get(module, ()), *GENERATOR_ARGS.get(module, ())]
    env = dict(os.environ, PYTHONHASHSEED='20260921', PYTHONIOENCODING='utf-8', MPLBACKEND='Agg',
               PYTHONDONTWRITEBYTECODE='1')
    try:
        proc = subprocess.run([sys.executable, '-m', 'src.pipeline.sandbox', str(mirror), module, *args],
                              cwd=ROOT, env=env, capture_output=True, text=True, encoding='utf-8',
                              errors='replace', timeout=timeout)
        code, output = proc.returncode, (proc.stdout or '') + (proc.stderr or '')
    except subprocess.TimeoutExpired:
        code, output = 124, f'timed out after {timeout} s'
    result = {'module': module, 'exit': code, 'tables': {}, 'detail': ''}
    written = sorted((mirror / TABLES_REL).glob('*.tex')) if (mirror / TABLES_REL).is_dir() else []
    if code != 0:
        hit = MISSING.search(output)
        if hit and code != 3:
            result['status'] = 'skipped'
            result['detail'] = 'needs ' + (hit.group(1) or hit.group(2)).strip()
        else:
            result['status'] = 'FAILED'
            result['detail'] = f'exit {code}: ' + (' | '.join(output.strip().splitlines()[-3:])
                                                    or 'no output (the process was stopped)')
        return result
    for new in written:
        old = TABLES / new.name
        if not old.exists():
            result['tables'][new.name] = ('new', [])
            continue
        a, b = _norm(old), _norm(new)
        if a == b:
            result['tables'][new.name] = ('identical', [])
        else:
            diff = [line for line in difflib.unified_diff(a.splitlines(), b.splitlines(), 'committed',
                                                          'regenerated', lineterm='', n=0)
                    if not line.startswith(('---', '+++', '@@'))]
            result['tables'][new.name] = ('DIFFERS', diff)
    if not written:
        result['status'], result['detail'] = 'identical', 'wrote no table'
    elif any(s == 'DIFFERS' for s, _ in result['tables'].values()):
        result['status'] = 'DIFFERS'
    else:
        result['status'] = 'identical'
    return result


def check(only=None, jobs=1, keep=None, timeout=3600, verbose=True):
    """Regenerate and compare; return the number of failures."""
    say = print if verbose else (lambda *a, **k: None)
    modules = [m for m in generators() if not only or m in only]
    scratch = Path(keep) if keep else Path(tempfile.mkdtemp(prefix='tables_'))
    scratch.mkdir(parents=True, exist_ok=True)
    try:
        with ThreadPoolExecutor(max_workers=max(1, jobs)) as pool:
            results = list(pool.map(lambda m: run_one(m, scratch, timeout), modules))
    finally:
        if not keep:
            shutil.rmtree(scratch, ignore_errors=True)
    regenerated = set()
    failures = 0
    for r in results:
        regenerated |= set(r['tables'])
        names = ', '.join(f'{n} {s}' if s != 'identical' else n for n, (s, _) in sorted(r['tables'].items()))
        say(f'  {r["status"]:9s} {r["module"]:44s} {names or r["detail"]}')
        if r['status'] in ('skipped', 'FAILED') and r['detail']:
            say(f'            {r["detail"][:200]}')
        for name, (status, diff) in sorted(r['tables'].items()):
            if status == 'DIFFERS':
                for line in diff[:8]:
                    say(f'            {line[:160]}')
        failures += r['status'] in ('DIFFERS', 'FAILED')
    if not only:
        hand = sorted(t.name for t in TABLES.glob('*.tex') if t.name not in regenerated
                      and not t.stem.endswith('_sources'))
        skipped = [r['module'] for r in results if r['status'] == 'skipped']
        say(f'{len(modules)} generator(s): '
            + ', '.join(f'{sum(r["status"] == s for r in results)} {s.lower()}'
                        for s in ('identical', 'DIFFERS', 'skipped', 'FAILED'))
            + f'; {len(regenerated)} table file(s) regenerated, {len(hand)} written by no generator'
            + (' that ran' if skipped else ''))
    return failures


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--only', nargs='+', metavar='MODULE', help='regenerate only these generators')
    ap.add_argument('--jobs', type=int, default=1,
                    help='generators run at once (default 1; several at once can exhaust memory)')
    ap.add_argument('--keep', type=Path, help='keep the regenerated files in this folder')
    a = ap.parse_args(argv)
    return 1 if check(a.only, a.jobs, a.keep) else 0


if __name__ == '__main__':
    raise SystemExit(main())
