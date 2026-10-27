"""Guard: no live code, contract or table may reach a gap-filled dataset.

`data/processed/hdf5/*_filled.h5` are artifacts of the 2024 watermodels codebase. They are not
observations. `OxygenTrainer.get_filled_dataset()` imputes gaps with that model's own predictions,
which the notebooks plot in green against the sensor trace in blue, and small gaps are additionally
closed by pchip interpolation. Training or scoring on them would be circular, and Vito has
instructed that nothing in this repository or the thesis use them.

Nothing does, at the time of writing. This exists so that a later change cannot quietly introduce
one: the check is mechanical and runs inside `smartbay verify`, in the same spirit as the
provenance lockfile. It looks in four places, because a dataset can enter by any of them:

    every Python file the repository carries, outside private/          (an import or an open())
      (the pipeline, the companion app, the deck and thesis checkers, the CLI and the tests)
    the release contracts, the protocol declarations and the data audit  (a bound input hash)
    the thesis sources                                                   (a path in a caption)
    the text of every artifact the thesis cites, in a table, its sources (a number traced to one)
      companion, the running text or a figure

A reference is any HDF5 access: a `.h5` or `.hdf5` path, the `hdf5` folder named as a path
segment, `read_hdf`, `HDFStore`, `h5py`, or `get_filled_dataset`. Only this file and the test that
plants each pattern to see it caught are exempt, because their subject is the patterns. The raw-data
reader is not: while it was exempt wholesale, `read_hdf` on a variable path, `.hdf5` and a
path-joined `'hdf5'` segment went unseen.

The forward fills that *do* exist in the pipeline are a different thing and are deliberately not
flagged. `ffill(limit=6)` on the anchor supplies each model with the last real reading up to six
hours old and flags its age, which is what an operational system has. It never touches a scoring
target: `v2_horizon.build` sets `valid` from the unfilled series at both the issuance and the
target hour, so every scored endpoint rests on two real observations.
"""
import json
import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

#: A path or call is suspect if it matches one of these. `.h5` is included wholesale because every
#: HDF5 file in this repository is a filled product; if a raw one ever appears, name it here.
PATTERNS = (
    re.compile(r'_filled\.h5'),
    re.compile(r'\.h5(?![\w])'),
    re.compile(r'\.hdf5?(?![\w])', re.I),
    re.compile(r'get_filled_dataset'),
    re.compile(r'data/processed/hdf5'),
    re.compile(r'''(?:[/\\]|['"])hdf5(?:[/\\]|['"])'''),   # a path segment, joined or literal
    re.compile(r'\bread_hdf\b'),
    re.compile(r'\bHDFStore\b'),
    re.compile(r'\bh5py\b'),
)

#: The files whose *subject* is this guard: this one documents the patterns, and the guard's test
#: plants each of them to check that it is caught.
EXEMPT = {'src/pipeline/inputs.py', 'tests/test_pipeline_checks.py'}

#: Never scanned: material the repository does not carry (environments, caches, build output) and
#: the private folder, which is not published.
SKIP_PARTS = {'.git', '.venv', 'venv', 'env', '__pycache__', '.cache', 'node_modules', 'scratch',
              'output', 'tmp', 'site-packages', '.pytest_cache', 'private'}

#: One field of a frozen declaration that names the filled datasets in order to say why they are not
#: used. Only that field is blanked before the scan; the rest of the file is scanned as usual, so a
#: real reference anywhere else in it still fails. The declaration is hashed and cannot be reworded.
RATIONALE = {'config/bay_reserved_protocol_v2_addendum32.json':
             ('why', 'why_not_fill_the_small_gaps_and_score_against_the_fill')}

#: Artifact text is scanned only in files small enough to be read whole without cost.
ARTIFACT_SUFFIXES = {'.json', '.txt', '.md', '.csv', '.log'}
ARTIFACT_MAX_BYTES = 5_000_000


def _config_text(path, rel):
    text = path.read_text(encoding='utf-8', errors='replace')
    if rel not in RATIONALE:
        return text
    data = json.loads(text)
    node = data
    *parents, leaf = RATIONALE[rel]
    for key in parents:
        node = node[key]
    node[leaf] = ''
    return json.dumps(data)


def _suspect(text):
    return sorted({m.group(0) for p in PATTERNS for m in p.finditer(text)})


def _carried(root):
    """The .py files git carries in `root` (tracked, or untracked but not ignored), or None when
    `root` is not the top of a git checkout, as in an exported copy."""
    try:
        top = subprocess.run(['git', 'rev-parse', '--show-toplevel'], cwd=root, capture_output=True,
                             text=True, check=True).stdout.strip()
        if Path(top).resolve() != Path(root).resolve():
            return None
        out = subprocess.run(['git', 'ls-files', '--cached', '--others', '--exclude-standard', '-z',
                              '--', '*.py'], cwd=root, capture_output=True, check=True).stdout
    except (OSError, subprocess.CalledProcessError):
        return None
    return [root / name for name in out.decode('utf-8').split('\0') if name]


def code_files(root=None):
    """Every Python file the repository carries, outside private/: the pipeline and experiments,
    the companion app, the deck and thesis checkers, the command line and the tests."""
    root = root or ROOT
    files = _carried(root)
    if files is None:
        files = root.rglob('*.py')
    return sorted(p for p in files if p.is_file()
                  and not SKIP_PARTS.intersection(p.relative_to(root).parts))


def scan_sources(root=None):
    """Every first-party module, whether or not the registry reaches it."""
    root = root or ROOT
    hits = {}
    for path in code_files(root):
        rel = path.relative_to(root).as_posix()
        if rel in EXEMPT:
            continue
        found = _suspect(path.read_text(encoding='utf-8', errors='replace'))
        if found:
            hits[rel] = found
    return hits


def scan_configs(root=None):
    """The release contracts, the data audit and the protocol declarations."""
    root = root or ROOT
    hits = {}
    for path in sorted(root.glob('config/*.json')) + [root / 'reports/bay_twin/data_audit.json']:
        if not path.exists():
            continue
        rel = path.relative_to(root).as_posix()
        found = _suspect(_config_text(path, rel))
        if found:
            hits[rel] = found
    return hits


def scan_thesis(root=None):
    """Table files and chapters, where an artifact path would be cited."""
    root = root or ROOT
    hits = {}
    for path in sorted((root / 'thesis_template/content').glob('**/*.tex')):
        found = _suspect(path.read_text(encoding='utf-8', errors='replace'))
        if found:
            hits[path.relative_to(root).as_posix()] = found
    return hits


def _artifact_files(path):
    files = sorted(p for p in path.rglob('*') if p.is_file()) if path.is_dir() else [path]
    return [f for f in files if f.suffix.lower() in ARTIFACT_SUFFIXES
            and f.stat().st_size <= ARTIFACT_MAX_BYTES]


def scan_artifacts():
    """The text files of every artifact the thesis cites, in a table or its sources companion, the
    running text or a figure: a digest locks what the thesis reads, so a filled dataset recorded as
    an input there would pass every other check."""
    from src.pipeline import provenance
    from src.pipeline.registry import artifact_path
    hits = {}
    for art in sorted(provenance.cited_artifacts()):
        path = artifact_path(art)
        if not path.exists():
            continue
        for f in _artifact_files(path):
            found = _suspect(f.read_text(encoding='utf-8', errors='replace'))
            if found:
                hits[f.relative_to(ROOT).as_posix()] = found
    return hits


def report():
    """Print the audit and return the number of failures, for `smartbay verify`."""
    groups = [('first-party modules', scan_sources()),
              ('contracts and the data audit', scan_configs()),
              ('thesis sources', scan_thesis()),
              ('cited artifacts', scan_artifacts())]
    present = sorted(p.relative_to(ROOT).as_posix()
                     for p in ROOT.glob('data/processed/hdf5/*.h5'))
    total = sum(len(hits) for _, hits in groups)
    if total:
        print(f'{total} reference(s) to gap-filled data FOUND')
    else:
        print(f'{len(present)} filled dataset(s) on disk, reached by none of '
              f'{len(code_files())} Python files (companion app, deck, checkers and tests '
              f'included), no contract, no thesis source and no cited artifact')
    for label, hits in groups:
        if hits:
            print(f'  {label}:')
            for where, what in hits.items():
                print(f'    {where}: {", ".join(what)}')
    if present:
        print('  on disk but unused (safe to keep as provenance of the prior work):')
        for p in present:
            print(f'    {p}')
    return total


if __name__ == '__main__':
    raise SystemExit(1 if report() else 0)
