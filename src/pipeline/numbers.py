"""Which printed decimals can be found in the artifacts their table or paragraph cites?

`smartbay provenance` checks that the artifact under a table has not changed since the table was
read; it never opens the artifact to see whether the table's numbers are in it. This module does,
for every decimal the thesis prints, and only reports: it never fails, because a printed number
can legitimately be absent from an artifact (a difference of two stored values, a unit conversion,
a count divided by another, a number from the literature) and the report cannot tell those apart
from a typing error. It gives the reader of a table or paragraph a list of
the numbers to check by hand.

What is compared. A printed decimal with k digits after the point is found when some number stored
in a cited artifact rounds to it at k digits, sign ignored (contrasts are printed in either
direction), or when a stored fraction does after multiplying by 100 (percentages), or a stored
oxygen value does in mg/l (the artifacts keep the pipeline's micromolar unit). Numbers are
read from every JSON file and every CSV of at most 5 MB in a cited file or folder; compressed row
files, databases and arrays are not opened. One-decimal numbers are reported separately, because at
one digit almost anything matches something.

Where the artifacts come from. A table file's are all the artifacts it cites, or, when it cites
none, those its `<table>_sources.tex` companion lists. A paragraph's are the
artifacts cited in that paragraph; when it cites none, those cited anywhere in its (sub)section are
used and the match is reported as found through the section. A decimal with no cited artifact in
either place is counted as uncited.

Usage: python smartbay.py numbers [--verbose] [--json PATH]
"""
import bisect
import csv
import io
import json
import re
from functools import lru_cache
from pathlib import Path

from src.pipeline import provenance
from src.pipeline.registry import artifact_path

ROOT = Path(__file__).resolve().parents[2]
CONTENT = ROOT / 'thesis_template/content'
TABLES = CONTENT / 'tables'
MAX_CSV = 5_000_000
MAX_JSON = 30_000_000

#: Markup whose arguments hold no printed result: references, citations, paths, lengths, options.
MASK = (
    re.compile(r'(?<!\\)%.*'),
    re.compile(r'\\(?:ref|autoref|eqref|label|cite\w*|path|artifact|texttt|url|href|input|include'
               r'|includegraphics|vspace\*?|hspace\*?|setlength|addtolength|rule|resizebox|scalebox'
               r'|renewcommand|arraystretch|fontsize|linespread|makecell|parbox)\s*'
               r'(?:\[[^\]]*\])?\s*\{(?:[^{}]|\{[^{}]*\})*\}'),
    re.compile(r'\\begin\{(?:tabular\*?|tabularx|adjustbox|minipage|figure|table|center)\}'
               r'(?:\[[^\]]*\])?(?:\{(?:[^{}]|\{[^{}]*\})*\})*'),
    re.compile(r'\\\\\[[^\]]*\]'),
)
#: thousands are separated by a thin space, 1\,071 (or, in older text, 1{,}071)
THOUSANDS = re.compile(r'\{,\}|\\,')
DECIMAL = re.compile(r'(?<![\w.\\])(\d{1,3}(?:(?:\{,\}|\\,)\d{3})+|\d+)\.(\d+)(?!\d)(?!\.\d)'
                     # a TeX length (2.5pt, 0.76\textwidth) is layout, not a printed result
                     r'(?!(?:pt|em|ex|cm|mm)\b|\s*\\(?:textwidth|linewidth|textheight|columnwidth)\b)')
SECTION = re.compile(r'\\(?:chapter|section|subsection)\*?\s*(?:\[[^\]]*\])?\{')


def decimals(text):
    """(value, digits after the point, offset) for every decimal printed in a piece of LaTeX."""
    for pattern in MASK:
        text = pattern.sub(lambda m: ' ' * len(m.group(0)), text)
    out = []
    for m in DECIMAL.finditer(text):
        whole = THOUSANDS.sub('', m.group(1))
        out.append((float(f'{whole}.{m.group(2)}'), len(m.group(2)), m.start()))
    return out


def _numbers_in(obj, out):
    if isinstance(obj, bool) or obj is None:
        return
    if isinstance(obj, (int, float)):
        out.append(abs(float(obj)))
    elif isinstance(obj, str):
        try:
            out.append(abs(float(obj)))
        except ValueError:
            pass
    elif isinstance(obj, dict):
        for k, v in obj.items():
            _numbers_in(k, out)
            _numbers_in(v, out)
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            _numbers_in(v, out)


def _file_numbers(path):
    out = []
    suffix = path.suffix.lower()
    try:
        size = path.stat().st_size
        if suffix == '.json' and size <= MAX_JSON:
            text = path.read_text(encoding='utf8', errors='replace')
            try:
                _numbers_in(json.loads(text), out)
            except json.JSONDecodeError:
                out += [abs(float(t)) for t in re.findall(r'-?\d+\.\d+(?:[eE][-+]?\d+)?', text)]
        elif suffix == '.csv' and size <= MAX_CSV:
            for row in csv.reader(io.StringIO(path.read_text(encoding='utf8', errors='replace'))):
                for cell in row:
                    try:
                        out.append(abs(float(cell)))
                    except ValueError:
                        pass
    except OSError:
        pass
    return out


@lru_cache(maxsize=None)
def artifact_values(key):
    """Sorted absolute values stored in an artifact (file or folder): as stored, times 100, and in
    mg/l from a stored oxygen value (times c for the sondes' per-kilogram unit, times the molar mass
    for a per-litre one; src/units.py)."""
    from src.units import MG_PER_UMOL_L, factor
    path = artifact_path(key)
    files = sorted(p for p in path.rglob('*') if p.is_file()) if path.is_dir() else (
        [path] if path.is_file() else [])
    values = []
    for f in files:
        values += _file_numbers(f)
    values = [v for v in values if v == v and v not in (float('inf'),)]
    c = factor()
    return (tuple(sorted(values)), tuple(sorted(v * 100 for v in values)),
            tuple(sorted(v * c for v in values)), tuple(sorted(v * MG_PER_UMOL_L for v in values)))


def found(value, digits, keys):
    half = 0.5 * 10 ** -digits + 1e-9
    for key in keys:
        for pool in artifact_values(key):
            i = bisect.bisect_left(pool, value - half)
            if i < len(pool) and pool[i] <= value + half:
                return True
    return False


def placebo(value, digits, keys):
    """Would a different number of the same size and precision also be 'found'? The pools of large
    folders are dense enough that a printed four-digit value can round to some stored value by
    chance; the share of placebos found is that chance, and the found share means little above it."""
    shifted = round(value * 1.137 + 3.7 * 10 ** -digits, digits)
    return found(shifted, digits, keys)


def _keys(text):
    return sorted({k for k in provenance.citations(text)
                   if not k.startswith(('src/', 'scripts/', 'config/'))})


@lru_cache(maxsize=None)
def generated_tables():
    """Table stems some first-party script names, i.e. tables with a generator. The rest are
    typed by hand, which is where a printed number most easily parts from its artifact."""
    code = ''.join(p.read_text(encoding='utf8', errors='replace')
                   for d in ('scripts', 'src') for p in (ROOT / d).rglob('*.py')
                   if '__pycache__' not in p.parts)
    return frozenset(t.stem for t in TABLES.glob('*.tex') if t.stem in code)


def check_tables():
    """Every table's decimals against the artifacts it cites, or, for a table citing none, those
    its `<table>_sources.tex` companion lists, as the provenance check traces it (Table 6.1)."""
    rows = []
    for tex in sorted(TABLES.glob('*.tex')):
        text = tex.read_text(encoding='utf8', errors='replace')
        keys = provenance.table_sources(tex)
        for value, digits, offset in decimals(text):
            line = text.count('\n', 0, offset) + 1
            status = ('uncited' if not keys else
                      'found' if found(value, digits, keys) else 'not found')
            rows.append({'file': f'tables/{tex.name}', 'line': line, 'value': value,
                         'digits': digits, 'status': status,
                         'placebo_found': bool(keys) and placebo(value, digits, keys)})
    return rows


def _paragraphs(text):
    """(paragraph text, offset, section text) for every blank-line-separated paragraph."""
    starts = [m.start() for m in SECTION.finditer(text)] + [len(text)]
    bounds = [0] + starts
    for a, b in zip(bounds, bounds[1:]):
        section = text[a:b]
        pos = a
        for para in re.split(r'\n\s*\n', section):
            offset = text.find(para, pos)
            pos = offset + len(para)
            yield para, offset, section


def check_prose():
    rows = []
    for tex in provenance.prose_files():
        text = tex.read_text(encoding='utf8', errors='replace')
        rel = tex.relative_to(CONTENT).as_posix()
        for para, offset, section in _paragraphs(text):
            if not para.strip():
                continue
            own, near = _keys(para), None
            for value, digits, at in decimals(para):
                line = text.count('\n', 0, offset + at) + 1
                if own:
                    keys = own
                    status = 'found' if found(value, digits, own) else 'not found'
                else:
                    near = _keys(section) if near is None else near
                    keys = near
                    status = ('uncited' if not near else
                              'found via section' if found(value, digits, near) else 'not found')
                rows.append({'file': rel, 'line': line, 'value': value, 'digits': digits,
                             'status': status,
                             'placebo_found': bool(keys) and placebo(value, digits, keys)})
    return rows


def _summary(rows, label):
    two = [r for r in rows if r['digits'] >= 2]
    one = [r for r in rows if r['digits'] == 1]
    count = lambda rs, s: sum(1 for r in rs if r['status'] == s)
    print(f'{label}: {len(two)} decimal(s) with two or more digits, {len(one)} with one')
    for status in ('found', 'found via section', 'not found', 'uncited'):
        n2, n1 = count(two, status), count(one, status)
        if n2 or n1:
            print(f'  {status:18s} {n2:5d}   (one digit: {n1})')
    tested = [r for r in two if r['status'] != 'uncited']
    if tested:
        chance = sum(r['placebo_found'] for r in tested) / len(tested)
        print(f'  chance level: {100 * chance:.0f} % of placebo values (x 1.137, shifted) are also '
              f'"found" in the same artifacts, so a match there is weak evidence')
    return two


def report(verbose=False, json_path=None):
    tables, prose = check_tables(), check_prose()
    print('Printed decimals against the artifacts cited beside them (report only; never fails)\n')
    t2 = _summary(tables, 'tables')
    by_file = {}
    for r in t2:
        by_file.setdefault(r['file'], []).append(r)
    worst = sorted(((sum(r['status'] != 'found' for r in rs), len(rs), f) for f, rs in by_file.items()),
                   reverse=True)
    made = generated_tables()
    hand = sorted(t.stem for t in TABLES.glob('*.tex') if t.stem not in made)
    print(f'  {len(hand)} of {len(list(TABLES.glob("*.tex")))} table files have no generator '
          f'(no script names them)')
    print('  tables with the most two-digit decimals not found in what they cite:')
    for miss, n, f in [w for w in worst if w[0]][:12]:
        tag = '' if Path(f).stem in made else '  [typed by hand]'
        print(f'    {f:52s} {miss:4d} of {n}{tag}')
    print()
    p2 = _summary(prose, 'running text (chapters, abstract, appendices)')
    by_prose = {}
    for r in p2:
        if r['status'] in ('not found', 'uncited'):
            by_prose.setdefault(r['file'], {}).setdefault(r['status'], 0)
            by_prose[r['file']][r['status']] += 1
    for f, d in sorted(by_prose.items()):
        print(f'    {f:52s} ' + ', '.join(f'{v} {k}' for k, v in sorted(d.items())))
    if verbose:
        print('\nunmatched two-digit decimals:')
        for r in t2 + p2:
            if r['status'] in ('not found', 'uncited'):
                print(f"  {r['file']}:{r['line']}  {r['value']:.{r['digits']}f}  {r['status']}")
    if json_path:
        Path(json_path).write_text(json.dumps({'tables': tables, 'prose': prose}, indent=1))
        print(f'\nwrote {json_path}')
    return {'tables': tables, 'prose': prose}


if __name__ == '__main__':
    report()
