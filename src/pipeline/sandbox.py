"""Run repository code so that it cannot change a file of the repository.

Two checks re-run code whose normal job is to write published files: the component replay
(src/pipeline/component.py), whose verifier rewrites reports/bay_twin/integrated/, and the table
regeneration (src/pipeline/tables.py), whose generators write the thesis tables and often an
artifact beside them. Both must compare what the code would write with what is committed, and
neither may touch the committed copy.

    python -m src.pipeline.sandbox <mirror-folder> <module> [args ...]

runs `python -m <module> [args ...]` with every write under the repository redirected into the
mirror folder at the same relative path (reads prefer the mirror copy, so a generator that writes an
intermediate file and reads it back sees its own output), and with an audit hook that refuses any
other write, rename, removal or new folder under the repository. Code that writes by a route the
redirection does not cover therefore fails loudly instead of changing a file. The process exits with
the module's own status; a refused write exits 3.

Inside one process, `guard()` installs the audit hook alone; audit hooks cannot be removed, which
is why the checks run each sandboxed job in a child process.
"""
import builtins
import io
import os
import runpy
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
REFUSED = 3
_WRITE_FLAGS = os.O_WRONLY | os.O_RDWR | os.O_APPEND | os.O_CREAT | os.O_TRUNC


class RefusedWrite(PermissionError):
    pass


def _inside(path, root):
    try:
        Path(os.path.abspath(os.fsdecode(path))).relative_to(root)
        return True
    except (TypeError, ValueError):
        return False


def guard(root=ROOT, allowed=()):
    """Refuse, for the rest of this process, any write under `root` outside the `allowed` folders."""
    root, allowed = Path(root).resolve(), [Path(a).resolve() for a in allowed]

    def blocked(path):
        return (isinstance(path, (str, bytes, os.PathLike)) and _inside(path, root)
                and not any(_inside(path, a) for a in allowed))

    def hook(event, args):
        if event == 'open':
            path, mode, flags = args
            writing = (mode is not None and any(c in str(mode) for c in 'wax+')) or \
                (flags is not None and flags & _WRITE_FLAGS)
            if writing and blocked(path):
                raise RefusedWrite(f'refused: a write to {path}')
        elif event in ('os.rename', 'os.replace') and (blocked(args[0]) or blocked(args[1])):
            raise RefusedWrite(f'refused: renaming {args[0]}')
        elif event in ('os.remove', 'os.rmdir', 'shutil.rmtree') and blocked(args[0]):
            raise RefusedWrite(f'refused: removing {args[0]}')
        elif event == 'os.mkdir' and blocked(args[0]) and not os.path.isdir(args[0]):
            raise RefusedWrite(f'refused: a new folder {args[0]}')

    sys.addaudithook(hook)


def redirect(mirror, root=ROOT):
    """Send writes under `root` to the same relative path under `mirror`; reads of a path the
    mirror holds come from the mirror."""
    root, mirror = Path(root).resolve(), Path(mirror).resolve()
    real_open = io.open
    real_mkdir = Path.mkdir

    def mapped(path, writing):
        if isinstance(path, int) or not isinstance(path, (str, bytes, os.PathLike)):
            return path
        absolute = Path(os.path.abspath(os.fsdecode(path)))
        try:
            rel = absolute.relative_to(root)
        except ValueError:
            return path
        if _inside(absolute, mirror):
            return path
        target = mirror / rel
        if writing:
            target.parent.mkdir(parents=True, exist_ok=True)
            return str(target)
        return str(target) if target.exists() else path

    def sandbox_open(file, mode='r', *args, **kwargs):
        writing = any(c in mode for c in 'wax+')
        return real_open(mapped(file, writing), mode, *args, **kwargs)

    def sandbox_mkdir(self, mode=0o777, parents=False, exist_ok=False):
        target = Path(mapped(self, True)) if _inside(self, root) and not _inside(self, mirror) else self
        if target != self:
            # mapped() made the parent; make the folder itself in the mirror
            return real_mkdir(target, mode, parents=True, exist_ok=True)
        return real_mkdir(self, mode, parents, exist_ok)

    builtins.open = io.open = sandbox_open
    Path.mkdir = sandbox_mkdir


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if len(argv) < 2:
        raise SystemExit('usage: python -m src.pipeline.sandbox <mirror-folder> <module> [args ...]')
    mirror, module, rest = Path(argv[0]).resolve(), argv[1], argv[2:]
    mirror.mkdir(parents=True, exist_ok=True)
    redirect(mirror)
    guard(allowed=[mirror])
    sys.argv = [module, *rest]
    sys.path.insert(0, str(ROOT))
    try:
        runpy.run_module(module, run_name='__main__', alter_sys=True)
    except RefusedWrite as e:
        print(e, file=sys.stderr)
        return REFUSED
    except SystemExit as e:
        return e.code if isinstance(e.code, int) else (0 if e.code is None else 1)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
