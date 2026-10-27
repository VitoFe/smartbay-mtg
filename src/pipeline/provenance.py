"""Trace every thesis table and figure back to the artifact it was computed from, and check it is current.

The rule this enforces: every number in the thesis must be traceable to an artifact under
reports/bay_twin/, and if it cannot be regenerated it must not be written. That rule has until
now been applied by hand. Here it is mechanical.

Five checks:

  1. Every table declares a source. A table with no \\artifact{runs/...} (or \\path{}) cannot be traced at all,
     unless a companion `<table>_sources.tex` beside it lists them: Table 6.1 keeps its source
     list, one row per result, in the summary-sources appendix, and is traced through it.
  2. Every declared source exists on disk and is claimed by a producer in the registry, or is
     listed as a known external artifact with a reason. Sources in the repository's root
     `results/` tree (Optuna study folders) are resolved like any other, and a brace group such
     as `A{6,7}` is expanded into one citation per member, so neither is dropped or left
     unhashed.
  3. Every source's content hash matches the lockfile, so a table is flagged when the artifact
     underneath it has been regenerated since the prose was written. A lockfile entry without a
     digest is an error, not a pass.
  4. Every artifact the running text cites (chapters, abstract, appendices) exists on disk, is
     claimed by a producer or listed as external with a reason, exactly as a table's source must
     be, and is locked with them. Three such citations once pointed at files that had never been
     carried into this repository, which checks 1-3, reading only the tables, could not see; and
     while prose citations needed neither a producer nor a lock entry, the scripts that wrote four
     of them were classified as dead and removed, so those numbers could no longer be regenerated.
  5. Every figure the thesis includes (\\includegraphics) is claimed by the stage that writes it,
     or is a declared byte-identical copy of an artifact that is, and its digest is locked with
     the tables' sources.

What is blocking. Missing, unclaimed and undeclared sources, a figure with no producer, a copy
that no longer matches its source, an unhashable lock entry and a missing or unclaimed prose
citation always block. An artifact that changed since the lockfile was written, or is not in it yet, also blocks,
because it means a table or figure may no longer say what its artifact says; `allow_drift=True`
(`--allow-drift` on the command line) reports those without blocking, for the interval between
regenerating an artifact and re-reading what cites it.

This module checks that an artifact is unchanged. It does not check that a printed number equals
a value inside the artifact; src/pipeline/numbers.py reports that, without blocking.

The lockfile is reports/bay_twin/provenance.lock.json. It is updated deliberately, with
`smartbay provenance --update`, which is the moment to re-read the affected tables and figures.
"""
import hashlib
import json
import re
import subprocess
from pathlib import Path

from src.pipeline.registry import (EXTERNAL_ARTIFACTS, FIGURE_COPIES, STAGES, artifact_path)

ROOT = Path(__file__).resolve().parents[2]
TEX = ROOT / 'thesis_template'
TABLES = TEX / 'content/tables'
BAY = ROOT / 'reports/bay_twin'
LOCK = BAY / 'provenance.lock.json'
#: \artifact{} (an evidence file, listed in the appendix rather than printed), \path{} or \texttt{}, with its
#: argument. The argument may hold escaped braces (\{ \}) and one
#: level of plain brace groups (A{6,7}); a bare [^}]* would cut both at the first closing brace.
CITATION = re.compile(r'\\(?:path|texttt|artifact)\{((?:\\[{}]|[^{}]|\{[^{}]*\})*)\}')
GRAPHICS = re.compile(r'\\includegraphics\s*(?:\[[^\]]*\])?\s*\{([^{}]*)\}')
#: Tokens inside \path{} that are meant as paths rather than as literal keys or identifiers.
#: A JSON key set in \path{} is ordinary markup and is not a broken citation.
PATHLIKE = re.compile(r'^(?:runs/|results/|reports/|src/|scripts/|config/|review_\d|\d{8}T?\d*[A-Z]?_)'
                      r'|\.(?:json|csv|py|gz)$')
FIGURE_SUFFIXES = ('.png', '.pdf', '.jpg', '.jpeg')


def expand(token):
    """`results/x_A{6,7}_y` -> ['results/x_A6_y', 'results/x_A7_y']. A token without a brace
    group is returned alone; several groups expand as a product."""
    match = re.search(r'\{([^{}]*)\}', token)
    if not match:
        return [token]
    head, tail = token[:match.start()], token[match.end():]
    out = []
    for member in match.group(1).split(','):
        out += expand(head + member.strip() + tail)
    return out


def _clean(token):
    # \allowbreak is a line-breaking hint inside \texttt{}, not part of the path
    token = re.sub(r'\\allowbreak\s*', '', token)
    return token.replace('\\_', '_').replace('\\{', '{').replace('\\}', '}').strip().rstrip('/')


def normalise(token):
    r"""Resolve one \path{} token (already brace-expanded) to an artifact key, or None.

    Tables cite artifacts three ways, all of which predate this checker: relative to
    reports/bay_twin/ ('runs/v2_missingness.json'), as a bare run directory
    ('20260918_reserved_evaluation'), and as a full repository path. All three resolve here so
    that existing prose does not have to be rewritten to be checkable. A key is relative to
    reports/bay_twin/, except for the root `results/` tree, which keeps its repository path.
    """
    token = _clean(token)
    if not token or not PATHLIKE.search(token):
        return None
    for base in (BAY, BAY / 'runs', ROOT):
        candidate = base / token
        if candidate.exists():
            resolved = candidate.resolve()
            if resolved in (BAY.resolve(), (BAY / 'runs').resolve()):
                return None  # the evidence tree named as a whole ("under reports/bay_twin/")
            try:
                return resolved.relative_to(BAY).as_posix()
            except ValueError:
                pass
            rel = resolved.relative_to(ROOT).as_posix() if ROOT in resolved.parents else None
            if rel and rel.startswith('results/'):
                return rel
            return None  # real, but outside reports/bay_twin: code or configuration
    # A bare filename that resolves nowhere ("report.json") is a generic mention in prose, not a
    # citation of a particular artifact. Only a token carrying a directory is treated as one.
    return token if '/' in token else None


def citations(text):
    """Every artifact key cited in a piece of LaTeX, brace groups expanded, in order."""
    out = []
    for raw in CITATION.findall(text):
        for token in expand(_clean(raw)):
            art = normalise(token)
            if art:
                out.append(art)
    return out


def producer_of(artifact):
    """Which stage claims this artifact? Directory artifacts match their contained files too."""
    target = artifact.rstrip('/')
    for stage in STAGES:
        for declared in stage.artifacts:
            d = declared.rstrip('/')
            if target == d or target.startswith(d + '/') or d.startswith(target + '/'):
                return stage.name
    return None


#: Extensions whose bytes are the content. Everything else is text, where a line-ending flip is
#: not a change in the evidence and must not be reported as one.
BINARY = {'.gz', '.zip', '.sqlite3', '.db', '.npy', '.npz', '.nc', '.png', '.pdf', '.parquet',
          '.jpg', '.jpeg', '.joblib', '.pkl', '.pt'}


def _content(path):
    """A file's bytes, with line endings normalised for text so that a checkout on a different
    platform does not read as a changed artifact. A JSON summary written with LF on one machine
    and CRLF on another holds the same numbers, and the lockfile exists to flag changed numbers."""
    raw = path.read_bytes()
    if path.suffix.lower() in BINARY:
        return raw
    return raw.replace(b'\r\n', b'\n')


def _published(path):
    """The files under a directory that the repository carries: tracked, or untracked but not
    ignored. Ignored files (logs, tuning databases, rendered PDFs) exist only in the checkout that
    ran the job, so hashing them would make a fresh clone read as changed. Outside a git checkout
    every file counts."""
    try:
        out = subprocess.run(['git', 'ls-files', '--cached', '--others', '--exclude-standard', '-z', '--',
                              str(path)], cwd=ROOT, capture_output=True, check=True)
    except (OSError, subprocess.CalledProcessError):
        return [p for p in path.rglob('*') if p.is_file()]
    files = [ROOT / name for name in out.stdout.decode('utf-8').split('\0') if name]
    return [p for p in files if p.is_file()]


def digest(path):
    """A stable content hash. Directories hash their files' relative names and contents in order."""
    h = hashlib.sha256()
    if path.is_dir():
        for f in sorted(_published(path)):
            h.update(f.relative_to(path).as_posix().encode())
            h.update(_content(f))
    elif path.is_file():
        h.update(_content(path))
    else:
        return None
    return h.hexdigest()


def _status(art, dig):
    if dig is None:
        return 'missing'
    if producer_of(art):
        return 'ok'
    if art.rstrip('/') in EXTERNAL_ARTIFACTS or art in EXTERNAL_ARTIFACTS:
        return 'external'
    return 'unclaimed'


def _evidence(text):
    """The artifacts a piece of LaTeX cites, less citations of source code, which record how a
    number was made rather than what it was made from."""
    return sorted({c for c in citations(text) if not c.startswith(('src/', 'scripts/', 'config/'))})


def table_sources(tex):
    """The artifacts a table file cites, or, when it cites none, those its `<table>_sources.tex`
    companion lists (Table 6.1 keeps its source list in the summary-sources appendix)."""
    cited = _evidence(tex.read_text(encoding='utf8', errors='replace'))
    companion = tex.with_name(tex.stem + '_sources.tex')
    if not cited and companion.exists():
        cited = _evidence(companion.read_text(encoding='utf8', errors='replace'))
    return cited


def scan():
    """Build the table -> artifact -> producer map, with the current digest of each artifact."""
    rows, digests = [], {}
    for tex in sorted(TABLES.glob('*.tex')):
        cited = table_sources(tex)
        if not cited:
            rows.append({'table': tex.name, 'artifact': None, 'producer': None,
                         'status': 'undeclared'})
            continue
        for art in cited:
            if art not in digests:
                digests[art] = digest(artifact_path(art))
            rows.append({'table': tex.name, 'artifact': art,
                         'producer': producer_of(art), 'status': _status(art, digests[art])})
    return rows, digests


def prose_files():
    content = TEX / 'content'
    return sorted(p for p in content.rglob('*.tex') if TABLES not in p.parents)


def figure_key(target):
    """`images/figures/fig_x` -> 'thesis_template/images/figures/fig_x.png' (the file LaTeX
    would pick), or the bare key when no candidate exists."""
    base = TEX / target
    candidates = [base] if base.suffix.lower() in FIGURE_SUFFIXES else \
        [base.with_name(base.name + s) for s in FIGURE_SUFFIXES]
    for c in candidates:
        if c.is_file():
            return c.relative_to(ROOT).as_posix()
    return (TEX / (target if base.suffix else target + '.png')).relative_to(ROOT).as_posix()


def scan_figures():
    """Every included figure: where it is used, who writes it, and its digest."""
    rows, digests = [], {}
    for tex in sorted((TEX / 'content').rglob('*.tex')):
        text = re.sub(r'(?<!\\)%.*', '', tex.read_text(encoding='utf8', errors='replace'))
        for target in GRAPHICS.findall(text):
            key = figure_key(target.strip())
            where = tex.relative_to(TEX / 'content').as_posix()
            if key not in digests:
                digests[key] = digest(ROOT / key)
            if digests[key] is None:
                status, producer = 'missing', None
            elif producer_of(key):
                status, producer = 'ok', producer_of(key)
            elif key in FIGURE_COPIES:
                source = FIGURE_COPIES[key]
                producer = producer_of(source)
                same = digest(artifact_path(source)) == digests[key]
                status = 'copy' if (same and producer) else 'stale copy'
            else:
                status, producer = 'unclaimed', None
            rows.append({'file': where, 'figure': key, 'producer': producer, 'status': status})
    return rows, digests


def scan_prose():
    """Artifacts cited in running text, and those of them that are not on disk."""
    cited, missing = set(), []
    for tex in prose_files():
        for art in citations(tex.read_text(encoding='utf8', errors='replace')):
            if art.startswith(('src/', 'scripts/', 'config/')):
                continue
            cited.add(art)
            if not artifact_path(art).exists():
                missing.append((tex.relative_to(TEX / 'content').as_posix(), art))
    return cited, missing


def prose_rows():
    """Every artifact the running text cites, where, who produces it, its status and its digest.
    The status follows the tables' rule: claimed by a stage ('ok'), listed with a reason
    ('external'), neither ('unclaimed'), or not on disk ('missing')."""
    where = {}
    for tex in prose_files():
        rel = tex.relative_to(TEX / 'content').as_posix()
        for art in _evidence(tex.read_text(encoding='utf8', errors='replace')):
            where.setdefault(art, set()).add(rel)
    rows, digests = [], {}
    for art in sorted(where):
        digests[art] = digest(artifact_path(art))
        rows.append({'artifact': art, 'files': sorted(where[art]), 'producer': producer_of(art),
                     'status': _status(art, digests[art])})
    return rows, digests


def cited_artifacts():
    """Every artifact key the thesis cites, from a table (or its sources companion), the running
    text or an \\includegraphics, without hashing anything."""
    keys = set()
    for tex in TABLES.glob('*.tex'):
        keys.update(table_sources(tex))
    for tex in prose_files():
        keys.update(_evidence(tex.read_text(encoding='utf8', errors='replace')))
    for tex in (TEX / 'content').rglob('*.tex'):
        text = re.sub(r'(?<!\\)%.*', '', tex.read_text(encoding='utf8', errors='replace'))
        keys.update(figure_key(t.strip()) for t in GRAPHICS.findall(text))
    return keys


def compare_lock(digests):
    """Which artifacts changed since the lockfile was written, which are not in it, and which of
    its entries carry no digest (an entry that locks nothing)."""
    if not LOCK.exists():
        return None, list(digests), []
    locked = json.loads(LOCK.read_text())['artifacts']
    changed = [a for a, d in digests.items() if a in locked and locked[a] != d]
    new = [a for a in digests if a not in locked]
    null = sorted(a for a, d in locked.items() if d is None)
    return changed, new, null


def write_lock(digests):
    """Write every digest that exists. A missing artifact has no digest and is left out rather
    than locked as null, which would record nothing and pass silently."""
    LOCK.write_text(json.dumps(
        {'note': 'content digests of every artifact the thesis tables and running text cite and '
                 'every figure it includes; update deliberately',
         'artifacts': dict(sorted((a, d) for a, d in digests.items() if d is not None))},
        indent=2))
    return LOCK


def report(update=False, verbose=False, allow_drift=False):
    rows, digests = scan()
    fig_rows, fig_digests = scan_figures()
    by_status = {}
    for r in rows:
        by_status.setdefault(r['status'], []).append(r)

    print(f'{len(set(r["table"] for r in rows))} tables, {len(digests)} distinct artifacts')
    for status in ('ok', 'external', 'undeclared', 'unclaimed', 'missing'):
        hits = by_status.get(status, [])
        if not hits:
            continue
        tables = sorted(set(r['table'] for r in hits))
        print(f'  {status:11s} {len(hits):3d} citation(s) across {len(tables)} table(s)')
        if verbose or status in ('undeclared', 'unclaimed', 'missing'):
            for r in hits:
                art = r['artifact'] or '(no \\artifact{runs/...} anywhere in the file)'
                print(f'      {r["table"]:42s} {art}')

    fig_status = {}
    for r in fig_rows:
        fig_status.setdefault(r['status'], []).append(r)
    print(f'\n{len(fig_digests)} included figure(s)')
    for status in ('ok', 'copy', 'stale copy', 'unclaimed', 'missing'):
        hits = fig_status.get(status, [])
        if not hits:
            continue
        print(f'  {status:11s} {len(hits):3d}')
        if verbose or status not in ('ok', 'copy'):
            for r in hits:
                print(f'      {Path(r["figure"]).name:42s} {r["producer"] or "(no producer)"}'
                      f'  [{r["file"]}]')

    prose, prose_digests = prose_rows()
    prose_status = {}
    for r in prose:
        prose_status.setdefault(r['status'], []).append(r)
    print(f'\nrunning text cites {len(prose)} artifact(s)')
    for status in ('ok', 'external', 'unclaimed', 'missing'):
        hits = prose_status.get(status, [])
        if not hits:
            continue
        print(f'  {status:11s} {len(hits):3d}')
        if verbose or status in ('unclaimed', 'missing'):
            for r in hits:
                print(f'      {r["artifact"]:52s} [{", ".join(r["files"])}]')

    locked = {**digests, **fig_digests, **prose_digests}
    changed, new, null = compare_lock(locked)
    drift = 0
    if changed is None:
        print('\nno lockfile yet; run with --update to write one')
        drift = len(locked)
    else:
        if changed:
            print(f'\n{len(changed)} artifact(s) changed since the lockfile was written:')
            for a in changed:
                users = sorted(set(r['table'] for r in rows if r['artifact'] == a)
                               | set(r['file'] for r in fig_rows if r['figure'] == a)
                               | {f for r in prose if r['artifact'] == a for f in r['files']})
                print(f'  {a}\n      re-read: {", ".join(users)}')
        if new:
            print(f'\n{len(new)} artifact(s) not yet in the lockfile: {", ".join(sorted(new))}')
        if null:
            print(f'\n{len(null)} lockfile entr(y/ies) with no digest, which lock nothing: '
                  f'{", ".join(null)}')
        if not changed and not new and not null:
            print('\nevery cited artifact and included figure matches the lockfile')
        drift = len(changed) + len(new)
    if drift and allow_drift:
        print(f'  (--allow-drift: {drift} changed or unlocked artifact(s) reported, not blocking)')

    if update:
        print(f'\nwrote {write_lock(locked).relative_to(ROOT)}')
        changed, new, null = compare_lock(locked)
        drift = len(changed) + len(new)

    blocking = (len(by_status.get('missing', [])) + len(by_status.get('unclaimed', []))
                + len(by_status.get('undeclared', []))
                + len(fig_status.get('missing', [])) + len(fig_status.get('unclaimed', []))
                + len(fig_status.get('stale copy', []))
                + len(prose_status.get('missing', [])) + len(prose_status.get('unclaimed', []))
                + len(null or [])
                + (0 if allow_drift else drift))
    return blocking
