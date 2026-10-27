"""Does every stage declare only artifacts that one of its own producers writes?

The registry names, per stage, the producers and the artifacts. Nothing checked that the two
lists agreed, and three declarations had drifted: a JSON file declared in the
data stage was written only by a module the stage did not run, two comparison files were written
by no module at all, and a pooled contrast was declared in the forecast stage while its producer
ran in the analysis stage. A stage that declares what it cannot write cannot rebuild its own
output, and `smartbay run` would report success without having made it.

The check is static. An artifact is taken to be named by a producer when a string literal in the
producer, or in a first-party module it imports directly, ends with the artifact's file or folder
name (with or without its extensions), or when an f-string there matches that name with its
interpolated parts read as wildcards (`f'fig_bay_gap_traces_{variable}.png'` names both trace
figures). An f-string whose literal text is shorter than six characters matches nothing, so a
bare `f'{name}.png'` cannot claim every figure. Naming is necessary for writing, not sufficient:
a module that reads an artifact also names it. The check therefore catches a declaration no
producer of the stage could write, which is the failure that occurred, and not a producer that
reads instead of writes.
"""
import ast
import re
from functools import lru_cache
from pathlib import Path, PurePosixPath

from src.pipeline.registry import STAGES

ROOT = Path(__file__).resolve().parents[2]
MIN_LITERAL = 6


def module_path(module):
    path = ROOT / (module.replace('.', '/') + '.py')
    return path if path.exists() else None


@lru_cache(maxsize=None)
def _parse(path):
    try:
        return ast.parse(Path(path).read_text(encoding='utf8', errors='replace'))
    except SyntaxError:
        return None


def _fstring_pattern(node):
    """A JoinedStr as a regex, formatted parts as wildcards; None if too little literal text."""
    parts, literal = [], 0
    for value in node.values:
        if isinstance(value, ast.Constant) and isinstance(value.value, str):
            parts.append(re.escape(value.value))
            literal += len(value.value)
        else:
            parts.append('[^/]*')
    return ''.join(parts) if literal >= MIN_LITERAL else None


@lru_cache(maxsize=None)
def strings_of(path):
    """(plain literals, f-string patterns) in one module."""
    tree = _parse(path)
    if tree is None:
        return frozenset(), frozenset()
    plain, patterns = set(), set()
    for node in ast.walk(tree):
        if isinstance(node, ast.JoinedStr):
            p = _fstring_pattern(node)
            if p:
                patterns.add(p)
        elif isinstance(node, ast.Constant) and isinstance(node.value, str):
            plain.add(node.value.replace('\\', '/'))
    return frozenset(plain), frozenset(patterns)


def _imports(path):
    """First-party modules one hop away, through the doctor's resolver."""
    from src.pipeline import doctor
    known = doctor.all_modules()
    name = doctor.module_name(Path(path))
    return [known[m] for m in doctor.imports_of(Path(path), name, known) if m in known]


def _names(artifact):
    name = PurePosixPath(artifact.rstrip('/')).name
    bare = name.split('.')[0]
    stem = name.rsplit('.', 1)[0] if '.' in name else name
    return {n for n in (name, bare, stem) if n}


def named_by(artifact, paths):
    """Is the artifact named by a literal or an f-string pattern in any of these modules?"""
    names = _names(artifact)
    for path in paths:
        plain, patterns = strings_of(str(path))
        for text in plain:
            text = text.rstrip('/')
            if any(text == n or text.endswith('/' + n) for n in names):
                return True
        for p in patterns:
            if any(re.fullmatch(r'(?:.*/)?' + p + r'/?', n) for n in names):
                return True
    return False


def check():
    """[(stage, artifact)] declared by a stage none of whose producers names it, and
    [(stage, producer)] declared with no module on disk."""
    unnamed, missing = [], []
    for stage in STAGES:
        paths = []
        for module in stage.producers:
            path = module_path(module)
            if path is None:
                missing.append((stage.name, module))
                continue
            paths.append(path)
            paths += _imports(path)
        for artifact in stage.artifacts:
            if not named_by(artifact, paths):
                unnamed.append((stage.name, artifact))
    return unnamed, missing


def report():
    unnamed, missing = check()
    total = sum(len(s.artifacts) for s in STAGES)
    print(f'{len(STAGES)} stages declare {total} artifacts; '
          f'{total - len(unnamed)} are named by a producer of their own stage')
    for stage, artifact in unnamed:
        print(f'  NOT NAMED   {stage:12s} {artifact}')
    for stage, module in missing:
        print(f'  NO MODULE   {stage:12s} {module}')
    return len(unnamed) + len(missing)


if __name__ == '__main__':
    raise SystemExit(1 if report() else 0)
