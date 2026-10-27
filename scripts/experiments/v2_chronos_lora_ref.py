"""Gate G2 of addendum 40: the windows the unmodified v2_chronos.py draws, with nothing trained.

Loads v2_chronos.py from the given path, replaces its run_study by a function that reads the
training and validation arrays out of the objective's closure and hashes them, and runs its main().
The hashes must equal the ones v2_chronos_lora.py --gates writes for the same tasks.

  python v2_chronos_lora_ref.py <path to the unmodified v2_chronos.py> <output json>
"""
import hashlib, importlib.util, json, sys
from pathlib import Path
import numpy as np

ref, target = Path(sys.argv[1]).resolve(), Path(sys.argv[2])
sys.argv = [str(ref)]                                   # the worker path over both tasks: no flags
spec = importlib.util.spec_from_file_location('v2_chronos_reference', ref)
module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
found = {}


def closure(fn):
    return dict(zip(fn.__code__.co_freevars, (c.cell_contents for c in fn.__closure__)))


def intercept(name, objective, folder, quick=False, n_trials=None):
    cells = closure(closure(objective)['fit_config'])
    h = hashlib.sha256()
    for key in ('ctxs', 'tgts', 'ti', 'vi'): h.update(np.ascontiguousarray(cells[key]).tobytes())
    found[name] = {'n_windows': int(len(cells['ctxs'])), 'n_train': int(len(cells['ti'])),
                   'n_val': int(len(cells['vi'])), 'sha256': h.hexdigest()}


module.run_study = intercept
module.main()
target.write_text(json.dumps({'reference': str(ref), 'windows': found}, indent=2))
print(json.dumps(found, indent=2))
