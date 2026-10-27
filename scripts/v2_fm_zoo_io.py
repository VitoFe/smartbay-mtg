"""Common input and output format for addendum 81 (every pretrained forecaster on every task).

Each task's builder (scripts/v2_fm_zoo_inputs_<task>.py, run locally from the published code paths)
writes one input file per (task, variable, cohort) with `save_inputs`; the runner on the compute
host (scripts/experiments/v2_fm_zoo.py) reads it with `load_inputs`, runs every arm and writes one
prediction file per (input file, arm) with `save_predictions`; the task's scorer reads the
predictions back with `load_predictions` and joins them to the published rows through the keys.

Input file (npz, no pickled objects):
  meta            JSON string: task, variable, cohort, horizon, units, standardisation, covariate
                  names, note, and the digest of every source the builder read
  key_<name>      one array per key column (str as unicode, times as int64 ns UTC)
  ctx_values      float32, every context concatenated (target series, oldest first, NaN kept)
  ctx_offsets     int64 [N + 1], context i is ctx_values[ctx_offsets[i]:ctx_offsets[i + 1]]
  pcov_values     float32 [C, total], past covariates aligned to ctx_values (absent if C = 0)
  fcov_values     float32 [N, C, H], covariates over the horizon known to a live system at the
                  target hour (absent if the task gives none)
  reverse         bool [N], True where the context is a reversed (backward) window

Prediction file (npz):
  meta            JSON string: input digest, arm, model revision, NaN treatment, horizon method
  median          float32 [N, H]
  q               float32 [N, 9, H] at LEVELS
Keys are not repeated: rows follow the input file's order.
"""
import hashlib
import json

import numpy as np
import pandas as pd

LEVELS = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)
ARMS = ('chronos_bolt_small_uni', 'chronos_bolt_base_uni', 'chronos2_uni', 'chronos2_cov',
        'timesfm25_uni', 'timesfm25_cov', 'timesfm3_uni', 'timesfm3_cov')


def digest(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def _key_array(s):
    if pd.api.types.is_datetime64_any_dtype(s):
        t = pd.to_datetime(s, utc=True)
        return t.dt.tz_convert('UTC').dt.tz_localize(None).astype('datetime64[ns]').to_numpy().astype('int64')
    if s.dtype == object or pd.api.types.is_string_dtype(s):
        return s.astype(str).to_numpy().astype('U')
    return s.to_numpy()


def save_inputs(path, *, task, variable, cohort, keys, contexts, horizon, units,
                past_cov=None, future_cov=None, cov_names=(), reverse=None,
                standardisation=None, sources=(), note=''):
    """Write one input file. `keys` is a DataFrame with one row per context; `contexts` a list
    of 1-D arrays; `past_cov` a list of [C, len(context)] arrays or None; `future_cov` an
    [N, C, horizon] array or None."""
    n = len(contexts)
    if len(keys) != n:
        raise ValueError(f'{len(keys)} keys for {n} contexts')
    lengths = np.array([len(c) for c in contexts], dtype=np.int64)
    if (lengths < 1).any():
        raise ValueError('empty context')
    offsets = np.concatenate([[0], np.cumsum(lengths)])
    out = {'ctx_values': np.concatenate([np.asarray(c, dtype=np.float32) for c in contexts]),
           'ctx_offsets': offsets,
           'reverse': np.zeros(n, bool) if reverse is None else np.asarray(reverse, bool)}
    c = len(cov_names)
    if past_cov is not None:
        for i, p in enumerate(past_cov):
            if np.shape(p) != (c, lengths[i]):
                raise ValueError(f'past covariates of context {i}: {np.shape(p)} != {(c, lengths[i])}')
        out['pcov_values'] = np.concatenate([np.asarray(p, dtype=np.float32) for p in past_cov], axis=1)
    if future_cov is not None:
        f = np.asarray(future_cov, dtype=np.float32)
        if f.shape != (n, c, horizon):
            raise ValueError(f'future covariates {f.shape} != {(n, c, horizon)}')
        out['fcov_values'] = f
    for col in keys.columns:
        out[f'key_{col}'] = _key_array(keys[col])
    meta = {'task': task, 'variable': variable, 'cohort': cohort, 'horizon': int(horizon),
            'units': units, 'standardisation': standardisation, 'cov_names': list(cov_names),
            'n': n, 'key_columns': list(keys.columns), 'sources': {str(s): digest(s) for s in sources},
            'note': note}
    out['meta'] = np.array(json.dumps(meta))
    np.savez_compressed(path, **out)
    return meta


def load_inputs(path):
    z = np.load(path, allow_pickle=False)
    meta = json.loads(str(z['meta']))
    off = z['ctx_offsets']
    vals = z['ctx_values']
    contexts = [vals[off[i]:off[i + 1]] for i in range(len(off) - 1)]
    past = None
    if 'pcov_values' in z.files:
        p = z['pcov_values']
        past = [p[:, off[i]:off[i + 1]] for i in range(len(off) - 1)]
    future = z['fcov_values'] if 'fcov_values' in z.files else None
    keys = pd.DataFrame({k[4:]: z[k] for k in z.files if k.startswith('key_')})
    return {'meta': meta, 'keys': keys, 'contexts': contexts, 'past_cov': past,
            'future_cov': future, 'reverse': z['reverse']}


def save_predictions(path, *, median, q, meta):
    median = np.asarray(median, dtype=np.float32)
    q = np.asarray(q, dtype=np.float32)
    if q.shape[:2] != (median.shape[0], len(LEVELS)) or q.shape[2] != median.shape[1]:
        raise ValueError(f'quantiles {q.shape} do not match median {median.shape}')
    np.savez_compressed(path, median=median, q=q, meta=np.array(json.dumps(meta)))


def load_predictions(path):
    z = np.load(path, allow_pickle=False)
    return {'median': z['median'], 'q': z['q'], 'meta': json.loads(str(z['meta']))}
