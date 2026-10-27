"""Check the frozen component contract without writing anything.

config/bay_integrated_contract.json binds the retained component recipes: the interface code, the
verification record of each run they were verified on, and the reserved-period protocol, 23 files
by SHA-256. scripts/verify_bay_integrated.py (itself bound, so it cannot be edited) checks those
digests, replays ten interface cases (a field query fresh, with oxygen missing and on a 2026 date;
two six-hour forecasts against the blend ledger and one on a 2026 date; the temperature outage at
24, 72 and 73 hours; an oxygen outage refused) and writes the responses to
reports/bay_twin/integrated/. That last step rewrites a published folder, so nothing ran it.

This runs the same verifier in a child process under src/pipeline/sandbox.py, which sends its two
writes to a scratch folder, and then compares them with the committed record:

    digests   every bound file present and unchanged
    replay    the verifier runs to the end on this repository's runs
    record    its responses equal reports/bay_twin/integrated/examples.json and its case list
              equals the one in verification.json

The record also stores the contract file's own digest; the contract is not among the files it binds,
so a difference there is reported and does not fail the check. The digests take a second; the
replay takes minutes, so `smartbay verify` checks the digests and `smartbay verify --full` (or
`smartbay component`) adds the replay.

    python -m src.pipeline.component [--digests-only]     (exit 1 on a failure, 0 otherwise)
"""
import hashlib
import json
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CONTRACT = ROOT / 'config/bay_integrated_contract.json'
RECORD = ROOT / 'reports/bay_twin/integrated'
RUNS = ROOT / 'reports/bay_twin/runs'
VERIFIER = 'scripts.verify_bay_integrated'


def digests():
    """(missing, changed) bound files."""
    contract = json.loads(CONTRACT.read_text(encoding='utf-8'))
    missing, changed = [], []
    for name, want in contract['files_sha256'].items():
        path = ROOT / name
        if not path.is_file():
            missing.append(name)
        elif hashlib.sha256(path.read_bytes()).hexdigest() != want:
            changed.append(name)
    return missing, changed


def replay_inputs():
    """The run folders the verifier's replay reads, from the contract's components."""
    c = json.loads(CONTRACT.read_text(encoding='utf-8'))['components']
    runs = [c['spatial_field']['field_run'], c['spatial_field']['fallback_run'],
            c['known_sensor_forecast']['point_run'], c['known_sensor_forecast']['blend_run'],
            c['temperature_outage']['run']]
    return [r for r in dict.fromkeys(runs)]


def check(verbose=True, replay=True):
    """Run the check; return the number of failures. With replay=False only the digests."""
    say = print if verbose else (lambda *a, **k: None)
    if not CONTRACT.is_file():
        say('no component contract in this checkout; nothing to check')
        return 0
    missing, changed = digests()
    bound = len(json.loads(CONTRACT.read_text(encoding='utf-8'))['files_sha256'])
    say(f'{bound} bound files: {bound - len(missing) - len(changed)} unchanged, '
        f'{len(changed)} changed, {len(missing)} absent')
    for name in changed:
        say(f'  CHANGED  {name}')
    for name in missing:
        say(f'  absent   {name}')
    absent_runs = [r for r in replay_inputs() if not (RUNS / r).is_dir()]
    for r in absent_runs:
        say(f'  absent   runs/{r} (read by the replay)')
    if changed or missing:
        return len(changed) + len(missing)
    if not replay:
        say('replay not run (smartbay verify --full or smartbay component runs it)')
        return 0
    if absent_runs:
        say('replay not run: the run folders it reads are not all present')
        return 0

    with tempfile.TemporaryDirectory(prefix='component_') as tmp:
        proc = subprocess.run([sys.executable, '-m', 'src.pipeline.sandbox', tmp, VERIFIER], cwd=ROOT,
                              capture_output=True, text=True, encoding='utf-8', errors='replace')
        if proc.returncode != 0:
            say(f'replay FAILED (exit {proc.returncode}):')
            for line in (proc.stdout + proc.stderr).strip().splitlines()[-8:]:
                say(f'  | {line}')
            return 1
        say((proc.stdout.strip().splitlines() or [''])[-1])
        written = Path(tmp) / RECORD.relative_to(ROOT)
        new_examples = json.loads((written / 'examples.json').read_text(encoding='utf-8'))
        new_record = json.loads((written / 'verification.json').read_text(encoding='utf-8'))
    old_examples = json.loads((RECORD / 'examples.json').read_text(encoding='utf-8'))
    old_record = json.loads((RECORD / 'verification.json').read_text(encoding='utf-8'))
    failures = 0
    if new_examples != old_examples:
        differing = sorted(k for k in set(new_examples) | set(old_examples)
                           if new_examples.get(k) != old_examples.get(k))
        say(f'  responses differ from integrated/examples.json: {", ".join(differing)}')
        failures += 1
    for key in ('state', 'cases', 'scope'):
        if new_record.get(key) != old_record.get(key):
            say(f'  integrated/verification.json differs in "{key}"')
            failures += 1
    if not failures:
        say(f'record matches: {len(new_record["cases"])} cases, responses identical to '
            f'integrated/examples.json')
    if new_record.get('contract_sha256') != old_record.get('contract_sha256'):
        say('  note: the record stores the digest of an earlier version of the contract file; the '
            'contract does not bind itself, and every file it binds is unchanged')
    return failures


if __name__ == '__main__':
    raise SystemExit(1 if check(replay='--digests-only' not in sys.argv[1:]) else 0)
