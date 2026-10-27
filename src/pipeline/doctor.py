"""Which modules does the thesis actually depend on, and which are left over?

The repository holds far more code than the manuscript needs. Some of it is pre-v2 work that no
current result touches, some is bound byte-exact by a release contract and must never be removed,
and some is genuinely live. Telling the three apart by reading filenames is guesswork, so it is
done here by following imports.

Reachability starts from the producers declared in the stage registry, plus the CLI and the
manuscript checkers, and follows every import between first-party modules. A module the walk never
reaches is a *candidate* for retirement and nothing stronger: a module can be reached by a path
this walk cannot see, through an importlib call, a campaign uploading a file by name, or a
`python -m` in a shell script. The report says which of those escape hatches mention it.

Contract-bound modules are listed separately and are never candidates. A release contract binds
source hashes, so deleting a bound file invalidates a published digest by design.

A module that names an artifact the thesis cites (in a table, its sources companion, the running
text or a figure) is a seed as well, whether or not a stage declares it. Naming is how a writer
shows itself to a static check, and the walk once called the writers of four cited artifacts dead
because no stage listed them: they were removed and the numbers could no longer be regenerated.
Readers name artifacts too, so this over-protects; that costs a missed candidate, never a deleted
writer.
"""
import ast
import json
import re
from collections import deque
from pathlib import Path, PurePosixPath

from src.pipeline.registry import OPERATOR_TOOLS, STAGES

ROOT = Path(__file__).resolve().parents[2]
CODE_DIRS = ('src', 'scripts')
#: Entry points that are real regardless of what imports them.
EXTRA_SEEDS = (
    'src.pipeline.main', 'src.pipeline.doctor', 'src.pipeline.provenance',
    'src.pipeline.registry', 'src.pipeline.stage_data',
    # launched as `python -m src.pipeline.sandbox` by the component and table checks, never imported
    'src.pipeline.sandbox',
)


def module_name(path):
    return '.'.join(path.relative_to(ROOT).with_suffix('').parts)


def all_modules():
    out = {}
    for d in CODE_DIRS:
        for p in (ROOT / d).rglob('*.py'):
            if '__pycache__' in p.parts or p.name == '__init__.py':
                continue  # packaging, not code
            out[module_name(p)] = p
    return out


def imports_of(path, name, known):
    """First-party modules this file imports, absolute or relative."""
    try:
        tree = ast.parse(path.read_text(encoding='utf8', errors='replace'))
    except SyntaxError:
        return set()
    package = name.rsplit('.', 1)[0]
    found = set()
    by_stem = {}
    for other in known:
        by_stem.setdefault(other.rsplit('.', 1)[-1], []).append(other)

    def record(dotted):
        # `from src.data.bay_observations import read_export` names a module; `from
        # src.data import bay_observations` names a package plus an attribute. Both resolve
        # against the module table, longest first.
        parts = dotted.split('.')
        for cut in range(len(parts), 0, -1):
            candidate = '.'.join(parts[:cut])
            if candidate in known:
                found.add(candidate)
                return
            sibling = f'{package}.{candidate}'
            if sibling in known:
                found.add(sibling)
                return
        # A bare name after a sys.path insert can point anywhere in the tree, not just at a
        # sibling: v2_bottom_aou adds scripts/experiments and then imports exp_bottom. Resolve
        # such a name by file stem, and when the stem is ambiguous take every match, because
        # over-protecting a module costs nothing and under-protecting it deletes a dependency.
        for match in by_stem.get(parts[0], ()):
            found.add(match)

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for a in node.names:
                record(a.name)
        elif isinstance(node, ast.ImportFrom):
            if node.level:  # relative
                base = package.rsplit('.', node.level - 1)[0] if node.level > 1 else package
                stem = f'{base}.{node.module}' if node.module else base
            else:
                stem = node.module or ''
            record(stem)
            for a in node.names:
                record(f'{stem}.{a.name}' if stem else a.name)
    return found


def contract_bound():
    bound = set()
    for cfg in sorted((ROOT / 'config').glob('*contract*.json')):
        try:
            files = json.loads(cfg.read_text()).get('files_sha256', {})
        except json.JSONDecodeError:
            continue
        for rel in files:
            if rel.endswith('.py'):
                bound.add(rel.replace('\\', '/'))
    return bound


#: Files that can launch a module without importing it. reports/ holds 3.9 GB of run artifacts
#: that mention module names only as recorded provenance, so it contributes its prose and nothing
#: else; scanning the artifacts themselves would be slow and would find only echoes of past runs.
def _haystack_paths(all_py):
    seen = []
    for pattern in ('*.md', '*.sh', '*.ps1'):
        seen += [p for p in ROOT.rglob(pattern)
                 if not any(x in ('.git', '__pycache__', 'output', 'scratch', 'node_modules')
                            for x in p.parts)]
    seen += list((ROOT / 'config').glob('*.json'))
    seen += list(all_py)
    return seen


def dynamic_mentions(modules, all_paths):
    """Where a candidate's name appears outside an import: shell launches, campaigns, docs.

    One combined pattern is matched against each file once, rather than each name against each
    file, because the candidate list and the file list are both long.
    """
    if not modules:
        return {}
    by_stem = {}
    for name, path in modules.items():
        by_stem.setdefault(path.stem, []).append(name)
    combined = re.compile(r'(?<![\w.])(' + '|'.join(re.escape(s) for s in by_stem) + r')(?![\w])')

    mentions = {}
    for p in _haystack_paths(all_paths):
        try:
            if p.stat().st_size > 2_000_000:
                continue
            text = p.read_text(encoding='utf8', errors='replace')
        except OSError:
            continue
        for stem in set(combined.findall(text)):
            for name in by_stem[stem]:
                if modules[name] != p:
                    mentions.setdefault(name, []).append(p)
    return mentions


#: File names too common to identify an artifact on their own; such an artifact is named through
#: the nearest folder above it with a distinctive name.
GENERIC_STEMS = {'report', 'summary', 'contrasts', 'status', 'results', 'selection', 'manifest',
                 'verification', 'predictions', 'folds', 'config', 'contract', 'index', 'data',
                 'runs', 'reports', 'bay_twin'}


def artifact_names(artifact):
    """The names by which a module's string literals can point at an artifact: its file or folder
    name, with and without extensions, or, for a generic file name, the nearest distinctive folder."""
    parts = PurePosixPath(artifact.rstrip('/')).parts
    out = set()
    for part in reversed(parts):
        bare = part.split('.')[0]
        if bare.lower() in GENERIC_STEMS or len(bare) < 4:
            continue
        out |= {part, bare, part.rsplit('.', 1)[0]}
        break
    return out


def cited_namers(modules):
    """Modules whose string literals (or f-string patterns) name an artifact the thesis cites."""
    from src.pipeline import declarations, provenance
    names = set()
    for art in provenance.cited_artifacts():
        names |= artifact_names(art)
    found = set()
    for name, path in modules.items():
        plain, patterns = declarations.strings_of(str(path))
        if any(PurePosixPath(text.rstrip('/')).name in names for text in plain if text.strip('/')):
            found.add(name)
            continue
        if any(re.fullmatch(r'(?:.*/)?' + p + r'/?', n) for p in patterns for n in names):
            found.add(name)
    return found


def analyse():
    modules = all_modules()
    graph = {name: imports_of(path, name, modules) for name, path in modules.items()}

    seeds = set(EXTRA_SEEDS) | set(OPERATOR_TOOLS)
    declared = set()
    for stage in STAGES:
        for producer in stage.producers:
            declared.add(producer)
            if producer in modules:
                seeds.add(producer)
    namers = cited_namers(modules)
    seeds |= namers

    missing_producers = sorted(declared - set(modules))

    reachable, queue = set(), deque(seeds & set(modules))
    while queue:
        node = queue.popleft()
        if node in reachable:
            continue
        reachable.add(node)
        queue.extend(graph.get(node, ()) - reachable)

    bound = contract_bound()

    # A contract binds a file's hash, not its imports. Deleting something a bound module imports
    # leaves that module's digest valid and the module unrunnable, which is worse than either a
    # clean deletion or a clean refusal. Everything reachable from a bound module is protected
    # exactly as the bound module itself is.
    bound_closure, queue = set(), deque(
        n for n, p in modules.items() if p.relative_to(ROOT).as_posix() in bound)
    while queue:
        node = queue.popleft()
        if node in bound_closure:
            continue
        bound_closure.add(node)
        queue.extend(graph.get(node, ()) - bound_closure)

    orphans = {}
    for name, path in modules.items():
        if name in reachable:
            continue
        rel = path.relative_to(ROOT).as_posix()
        orphans[name] = {
            'path': rel,
            'lines': len(path.read_text(encoding='utf8', errors='replace').splitlines()),
            'contract_bound': rel in bound or name in bound_closure,
            'needed_by_contract': name in bound_closure and rel not in bound,
        }
    return {'modules': modules, 'reachable': reachable, 'orphans': orphans,
            'missing_producers': missing_producers, 'bound': bound,
            'bound_closure': bound_closure, 'cited_namers': namers}


def dangling_after_removal(result):
    """Would removing the candidates leave a survivor importing something that is gone?

    The reachability walk already guarantees this cannot happen *if* imports_of sees every import
    Python sees. It did not once: a bare name resolved through a sys.path insert was invisible,
    two live modules were called unreachable, and deleting them broke an operator tool. This is
    the independent check, deliberately written against file stems rather than against the same
    resolver, so that a gap in that resolver shows up here instead of in a broken repository.
    """
    modules = result['modules']
    free = {k for k, v in result['orphans'].items() if not v['contract_bound']}
    survivors = {k: v for k, v in modules.items() if k not in free}
    doomed_stems = {}
    for k in free:
        doomed_stems.setdefault(modules[k].stem, []).append(k)

    problems = {}
    for name, path in survivors.items():
        try:
            tree = ast.parse(path.read_text(encoding='utf8', errors='replace'))
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names = [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module:
                names = [node.module]
            else:
                continue
            for dotted in names:
                # An absolute name that resolves to a surviving module is already satisfied; only
                # a name whose only possible target is being deleted is a problem.
                if dotted in modules and dotted not in free:
                    continue
                for token in (dotted.split('.')[0], dotted.split('.')[-1]):
                    targets = [t for t in doomed_stems.get(token, ())]
                    if targets and not any(t in survivors for t in modules if modules[t].stem == token):
                        problems.setdefault(name, set()).add(f'{token} -> {sorted(targets)}')
    return problems


def report(show_all=False):
    result = analyse()
    modules, reachable, orphans = result['modules'], result['reachable'], result['orphans']
    bound_orphans = {k: v for k, v in orphans.items() if v['contract_bound']}
    free = {k: v for k, v in orphans.items() if not v['contract_bound']}

    tools = sorted(set(OPERATOR_TOOLS) & set(modules))
    print(f'{len(modules)} first-party modules')
    print(f'  {len(reachable):3d} reachable from the stage registry, the operator tools and the '
          f'modules naming a cited artifact')
    print(f'      of which {len(tools)} are operator tools, run by hand rather than imported, and '
          f'{len(result["cited_namers"])} name an artifact the thesis cites')
    indirect = sum(1 for v in bound_orphans.values() if v.get('needed_by_contract'))
    print(f'  {len(bound_orphans):3d} unreachable but contract-bound  (never remove)'
          f'{f", {indirect} of them indirectly" if indirect else ""}')
    print(f'  {len(free):3d} unreachable and unbound            (retirement candidates)')

    if result['missing_producers']:
        print('\nproducers declared in the registry with no module on disk:')
        for p in result['missing_producers']:
            print(f'  {p}')

    mentions = (dynamic_mentions({k: modules[k] for k in free}, modules.values())
                if free else {})
    by_dir = {}
    for name, info in free.items():
        by_dir.setdefault(str(Path(info['path']).parent), []).append((name, info))

    print(f'\nretirement candidates, {sum(i["lines"] for i in free.values()):,} lines:')
    for directory in sorted(by_dir):
        rows = sorted(by_dir[directory], key=lambda r: -r[1]['lines'])
        print(f'\n  {directory}/  ({len(rows)} modules, '
              f'{sum(i["lines"] for _, i in rows):,} lines)')
        for name, info in rows if show_all else rows[:8]:
            hits = mentions.get(name, [])
            note = ''
            if hits:
                where = ', '.join(sorted({h.suffix or h.name for h in hits})[:3])
                note = f'  <- named in {len(hits)} file(s) [{where}]'
            print(f'    {Path(info["path"]).name:44s} {info["lines"]:>5} lines{note}')
        if not show_all and len(rows) > 8:
            print(f'    ... and {len(rows) - 8} more (--all to list)')

    problems = dangling_after_removal(result)
    if problems:
        print(f'\nREFUSING: {len(problems)} surviving module(s) would import something removed:')
        for k, v in sorted(problems.items()):
            print(f'  {k}')
            for x in sorted(v):
                print(f'      {x}')
    else:
        print('\npre-flight: no surviving module would import a removed one')

    print('\nNothing is removed by this command. A module named in a campaign manifest or a '
          'shell launch\nis reached by a path the import walk cannot see; check before retiring '
          'one.')
    return 1 if problems else 0


def write_plan(path=None):
    """Emit the retirement analysis as an artifact, so the decision can be reviewed rather than
    taken from a terminal scroll."""
    result = analyse()
    modules, orphans = result['modules'], result['orphans']
    free = {k: v for k, v in orphans.items() if not v['contract_bound']}
    mentions = dynamic_mentions({k: modules[k] for k in free}, modules.values()) if free else {}

    plan = {
        'note': 'candidates for retirement, produced by `smartbay doctor`. Nothing is deleted by '
                'generating this file; it exists so the decision is made against a list rather '
                'than a scroll.',
        'reachable_from': 'the producers declared in src/pipeline/registry.py, the operator tools '
                          'declared there and every module naming an artifact the thesis cites, '
                          'followed through first-party imports',
        'caveats': [
            'a module launched by name from a shell script or a campaign manifest is reached by a '
            'path the import walk cannot see; the mentions field lists where each name appears',
            'contract-bound modules are excluded entirely: a release contract binds their hashes',
            'the v1 work is preserved byte-exact on the v1-contract-snapshot branch, so removing '
            'it from v2 loses nothing that a contract or a branch does not already hold',
        ],
        'totals': {
            'modules': len(modules),
            'reachable': len(result['reachable']),
            'contract_bound_unreachable': len(orphans) - len(free),
            'candidates': len(free),
            'candidate_lines': sum(v['lines'] for v in free.values()),
        },
        'candidates': sorted(
            ({'module': k, **v,
              'mentioned_in': sorted(str(p.relative_to(ROOT).as_posix()) for p in mentions.get(k, []))[:6]}
             for k, v in free.items()),
            key=lambda r: (-r['lines'], r['module'])),
    }
    problems = dangling_after_removal(result)
    plan['preflight'] = ('no surviving module would import a removed one' if not problems
                         else {k: sorted(v) for k, v in problems.items()})
    if problems:
        print(f'REFUSING to write a plan: {len(problems)} survivor(s) would import a removed module')
        for k, v in sorted(problems.items()):
            print(f'  {k} -> {sorted(v)}')
        return 1
    path = path or ROOT / 'reports/bay_twin/runs/pipeline_retirement_plan.json'
    path.write_text(json.dumps(plan, indent=2))
    print(f"{plan['totals']['candidates']} candidates, "
          f"{plan['totals']['candidate_lines']:,} lines -> {path.relative_to(ROOT)}")
    return 0
