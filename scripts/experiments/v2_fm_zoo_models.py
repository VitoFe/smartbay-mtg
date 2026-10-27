"""Addendum 81 on the compute host: one interface over every pretrained forecaster of the addendum.

    from v2_fm_zoo_models import Zoo, LEVELS
    zoo = Zoo()                                    # loads each checkpoint on first use, offline
    out = zoo.predict('timesfm3_cov', contexts, horizon=72, past_cov=..., future_cov=...)
    out['median']   float32 [N, H]
    out['q']        float32 [N, 9, H] at LEVELS (0.1 ... 0.9)
    out['meta']     model id, cached revision, levels the library returned, NaN treatment,
                    horizon method, batching, library flags, covariates used

Arms (config/bay_reserved_protocol_v2_addendum81.json): chronos_bolt_small_uni,
chronos_bolt_base_uni, chronos2_uni, chronos2_cov, timesfm25_uni, timesfm25_cov, timesfm3_uni,
timesfm3_cov. No model is fitted or tuned here; every call is zero-shot.

Inputs
  contexts    list of 1-D float arrays, oldest first, ending at the last hour the forecast may read;
              lengths may differ; NaN marks a missing hour
  past_cov    None, a list of {name: 1-D array as long as its context}, or a list of [C, len]
              arrays (or one [N, C, L] array) with `cov_names`
  future_cov  None, a list of {name: 1-D array of length H}, or an [N, C, H] array with `cov_names`:
              covariate values over the horizon a live system has at the target hour. A name with
              a future part must also have a past part.

Missing values (the declaration's context_rule)
  Chronos-Bolt, Chronos-2   the context exactly as given, NaN included: both libraries mask it.
                            Covariates as given (Chronos-2 masks NaN in them too).
  TimesFM 2.5, 3.0          `timesfm_fill` (default on): leading NaN trimmed, interior NaN filled by
                            linear interpolation in time (hour index), trailing NaN held at the last
                            observed value; covariates trimmed by the same leading hours and filled
                            the same way over past and future together.
  A covariate with no finite value over its past, or over a future it was given, carries nothing and
  is dropped for that series (`drop_dark`, v2_gap_recovery.chronos_fill's rule), for every model.

Horizons (horizon_rule): every model is called once for the full horizon.
  Chronos-Bolt   64 steps natively; beyond, the pipeline's own autoregression on its nine quantile
                 paths, reduced back to nine levels (chronos_bolt.py, predict).
  Chronos-2      model_prediction_length natively; beyond, the pipeline's long-horizon unrolling.
  TimesFM 2.5    decodes the compiled max_horizon (rounded up to 128) and truncates; beyond 128,
                 autoregressive in 128-step output patches (timesfm_2p5_torch.py, decode).
  TimesFM 3.0    one non-autoregressive pass, the horizon as masked patches, stitched from
                 overlapping 64-step patch outputs (timesfm3/torch/model.py, decode).

Covariates
  chronos2_cov   native past_covariates / future_covariates, items batched by covariate names
                 (the pipeline needs one schema per call), as v2_probabilistic.predict.
  timesfm25_cov  the library's XReg (TimesFM_2p5.forecast_with_covariates, mode 'xreg + timesfm'):
                 an ordinary least-squares regression of the context on the covariates, then TimesFM
                 on its residual, the regression added back over the horizon. XReg needs every
                 covariate over context AND horizon. It is called one series at a time, because the
                 library fits one regression over everything in a call: in a batch, an early
                 issuance would read a coefficient fitted on later contexts. The library's xreg_lib
                 imports jax, which the compute host does not have and may not install; its source
                 is executed unchanged with numpy standing in for jax.numpy (float64), see
                 `_xreg_lib`.
  timesfm3_cov   native: names with a future part go in as past_future_covariates, the others as
                 past_only_covariates (TimesFM3Forecaster.predict_batch).

Usage on the compute host (HF_HOME set, HF_HUB_OFFLINE=1 after v2_fm_zoo_fetch.py):
  python v2_fm_zoo_models.py --info     print every model's revision, levels and horizon limits
"""
import contextlib
import importlib.util
import json
import math
import os
import sys
import time
import types
import warnings
from pathlib import Path

import numpy as np

os.environ.setdefault('HF_HOME', '/home/vito/smartbay_do/.hf')

LEVELS = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)
ARMS = ('chronos_bolt_small_uni', 'chronos_bolt_base_uni', 'chronos2_uni', 'chronos2_cov',
        'timesfm25_uni', 'timesfm25_cov', 'timesfm3_uni', 'timesfm3_cov')
REPOS = {'chronos_bolt_small': 'amazon/chronos-bolt-small',
         'chronos_bolt_base': 'amazon/chronos-bolt-base',
         'chronos2': 'amazon/chronos-2',
         'timesfm25': 'google/timesfm-2.5-200m-pytorch',
         'timesfm3': 'google/timesfm-3.0-pytorch'}
#: the revision each name must resolve to in the cache; Chronos-Bolt base and Chronos-2 are
#: companion/fetch_models.py's, the others the revisions cached or downloaded for this addendum
PINNED = {'amazon/chronos-bolt-small': '772f3d25d38aec6d914c8949dab4462e2d46f5d8',
          'amazon/chronos-bolt-base': '5d9f166d69f47aef3401367a7b842e78fe97b121',
          'amazon/chronos-2': '29ec3766d36d6f73f0696f85560a422f50e8498c',
          'google/timesfm-2.5-200m-pytorch': '1d952420fba87f3c6dee4f240de0f1a0fbc790e3',
          'google/timesfm-3.0-pytorch': '43046b85ec22d584a13f8098c2ed39c889e129c2'}
#: per-call batch sizes of the published code paths: Chronos-Bolt 64 (chronos_fill, v2_chronos_adapt;
#: v2_sequence_arms used 256), Chronos-2 128 (v2_chronos2, v2_probabilistic; chronos_fill 64);
#: TimesFM 2.5 decodes per_core_batch_size = 1 series at a time (ForecastConfig default, as
#: v2_sequence_arms); TimesFM 3.0 32 (the library README)
BATCH = {'chronos_bolt_small': 64, 'chronos_bolt_base': 64, 'chronos2': 128, 'timesfm25': 1, 'timesfm3': 32}
#: TimesFM 2.5 ForecastConfig: normalize_inputs=True as v2_sequence_arms.timesfm_predict, plus the
#: continuous quantile head and the quantile-crossing fix the library README recommends for quantile
#: forecasts (addendum 81 amendment 2; the published arm had both off and used only the median);
#: every other flag the library default (force_flip_invariance=True, infer_is_positive=True)
TFM25_CONFIG = {'normalize_inputs': True, 'use_continuous_quantile_head': True,
                'fix_quantile_crossing': True}
#: the published arm's configuration, kept for the G1 check of its median
TFM25_PUBLISHED_CONFIG = {'normalize_inputs': True}
#: TimesFM 3.0 predict_batch flags: the library's zero-shot benchmark defaults (TimesFM3Evaluator)
TFM3_FLAGS = {'use_symmetric_averaging': True, 'make_positive': True, 'sort_quantiles': True,
              'use_znorm': False, 'padding_mode': 'none'}
MIN_CONTEXT_TFM = 512                      # the published max_context (v2_sequence_arms.CTX)


def arm_parts(arm):
    if arm not in ARMS:
        raise ValueError(f'unknown arm {arm}; one of {ARMS}')
    family, mode = arm.rsplit('_', 1)
    return family, mode


def cached_revision(repo):
    """(revision, snapshot directory) the name resolves to in the local cache (refs/main)."""
    from huggingface_hub import constants
    d = Path(constants.HF_HUB_CACHE) / f'models--{repo.replace("/", "--")}'
    rev = (d / 'refs' / 'main').read_text().strip()
    snap = d / 'snapshots' / rev
    if not snap.is_dir():
        raise FileNotFoundError(f'{repo}: snapshot {rev} missing from {d}')
    return rev, snap


# --------------------------------------------------------------------------------------------------
# inputs
# --------------------------------------------------------------------------------------------------
def _covariates(n, lengths, horizon, past_cov, future_cov, cov_names):
    """Per series: (past {name: float32 1-D}, future {name: float32 1-D})."""
    def per_series(obj, length_of, what):
        if obj is None:
            return [{} for _ in range(n)]
        if isinstance(obj, np.ndarray) and obj.ndim == 3:
            obj = list(obj)
        if len(obj) != n:
            raise ValueError(f'{what}: {len(obj)} entries for {n} contexts')
        out = []
        for i, item in enumerate(obj):
            if item is None:
                out.append({})
                continue
            if isinstance(item, dict):
                d = {str(k): np.asarray(v, np.float32) for k, v in item.items()}
            else:
                a = np.atleast_2d(np.asarray(item, np.float32))
                names = list(cov_names) if cov_names is not None else [f'c{j}' for j in range(a.shape[0])]
                if len(names) != a.shape[0]:
                    raise ValueError(f'{what} of series {i}: {a.shape[0]} rows for {len(names)} names')
                d = {nm: a[j] for j, nm in enumerate(names)}
            for k, v in d.items():
                if v.ndim != 1 or len(v) != length_of(i):
                    raise ValueError(f'{what} {k} of series {i}: shape {v.shape}, expected ({length_of(i)},)')
            out.append(d)
        return out
    past = per_series(past_cov, lambda i: lengths[i], 'past_cov')
    future = per_series(future_cov, lambda i: horizon, 'future_cov')
    for i in range(n):
        if set(future[i]) - set(past[i]):
            raise ValueError(f'series {i}: future covariates {sorted(set(future[i]) - set(past[i]))} have no past')
    return past, future


def _drop_dark(past, future):
    """chronos_fill's rule: a covariate with no finite past, or no finite future where one is given,
    carries nothing and is dropped for that series."""
    dropped = 0
    for p, f in zip(past, future):
        for k in list(p):
            if not np.isfinite(p[k]).any() or (k in f and not np.isfinite(f[k]).any()):
                p.pop(k)
                f.pop(k, None)
                dropped += 1
    return dropped


def _fill(x):
    """Interior NaN by linear interpolation in the hour index; both ends held at the nearest
    observed value (np.interp). None if nothing is observed."""
    x = np.asarray(x, np.float64)
    ok = np.isfinite(x)
    if not ok.any():
        return None
    if ok.all():
        return x
    idx = np.arange(len(x))
    y = x.copy()
    y[~ok] = np.interp(idx[~ok], idx[ok], x[ok])
    return y


def timesfm_inputs(contexts, past, future):
    """The declared TimesFM treatment: leading NaN trimmed, interior interpolated, trailing held;
    covariates trimmed by the same leading hours and filled over past and future together.
    Returns contexts, past, future, and a record of what was done."""
    ctx_out, past_out, fut_out = [], [], []
    rec = {'trimmed_leading_hours': 0, 'series_trimmed': 0, 'series_interpolated': 0,
           'series_trailing_held': 0, 'series_without_data': 0, 'covariates_dropped_after_trim': 0}
    for c, p, f in zip(contexts, past, future):
        c = np.asarray(c, np.float64)
        ok = np.isfinite(c)
        if not ok.any():
            rec['series_without_data'] += 1
            ctx_out.append(None); past_out.append({}); fut_out.append({})
            continue
        first = int(np.argmax(ok))
        last = len(c) - 1 - int(np.argmax(ok[::-1]))
        if first:
            rec['series_trimmed'] += 1
            rec['trimmed_leading_hours'] += first
        if last < len(c) - 1:
            rec['series_trailing_held'] += 1
        if not ok[first:last + 1].all():
            rec['series_interpolated'] += 1
        ctx_out.append(_fill(c[first:]).astype(np.float32))
        pp, ff = {}, {}
        for k, v in p.items():
            seg = np.asarray(v, np.float64)[first:]
            whole = np.concatenate([seg, np.asarray(f[k], np.float64)]) if k in f else seg
            filled = _fill(whole)
            if filled is None:
                rec['covariates_dropped_after_trim'] += 1
                continue
            pp[k] = filled[:len(seg)].astype(np.float32)
            if k in f:
                ff[k] = filled[len(seg):].astype(np.float32)
        past_out.append(pp); fut_out.append(ff)
    return ctx_out, past_out, fut_out, rec


def select_levels(returned, Q, levels=LEVELS):
    """Q [..., len(returned)] -> [..., len(levels)]. The declared rule: the returned levels are used
    where the library returns them; interpolation only if it returns none of them."""
    returned = [float(a) for a in returned]
    idx = [next((j for j, b in enumerate(returned) if abs(a - b) < 1e-9), None) for a in levels]
    if all(j is not None for j in idx):
        return Q[..., idx], 'returned by the library'
    if all(j is None for j in idx):
        order = np.argsort(returned)
        r = np.asarray(returned)[order]
        Qs = Q[..., order]
        flat = Qs.reshape(-1, len(r))
        out = np.stack([np.array([np.interp(a, r, row) for a in levels]) for row in flat])
        return out.reshape(Q.shape[:-1] + (len(levels),)), 'interpolated linearly between returned levels'
    raise ValueError(f'the library returns some but not all of {levels}: {returned}')


def _crossing_share(q):
    """Share of (series, step) pairs whose nine quantiles are not non-decreasing."""
    if q.size == 0:
        return 0.0
    return float((np.diff(q, axis=1) < 0).any(axis=1).mean())


# --------------------------------------------------------------------------------------------------
# TimesFM 2.5 XReg without jax
# --------------------------------------------------------------------------------------------------
def _xreg_lib():
    """timesfm.utils.xreg_lib, importable. With jax installed it is the library module as is.
    Without jax (the compute host), the module's own source is executed unchanged with numpy
    standing in for jax.numpy: its fit is np.linalg.pinv(X'X + ridge I) X'y on the same padded
    matrices, in the dtype of the inputs (float64 here; jax would default to float32). The
    stand-in implements only what xreg_lib reaches with max_rows_per_col = 0."""
    name = 'timesfm.utils.xreg_lib'
    if name in sys.modules:
        return sys.modules[name], getattr(sys.modules[name], '_backend', 'jax')
    try:
        import jax  # noqa: F401
        mod = importlib.import_module(name)     # the module-level `import importlib.util` binds it
        mod._backend = 'jax'
        return mod, 'jax'
    except ImportError:
        pass
    import timesfm
    path = Path(timesfm.__file__).resolve().parent / 'utils' / 'xreg_lib.py'
    fake = types.ModuleType('jax')
    fake.numpy = np
    fake.Array = np.ndarray
    fake.devices = lambda kind=None: [None]
    fake.default_device = lambda device: contextlib.nullcontext()

    def _no_random(*a, **k):
        raise RuntimeError('max_rows_per_col subsampling needs jax.random; it is not used here')
    fake.random = types.SimpleNamespace(PRNGKey=_no_random, choice=_no_random)
    saved = {k: sys.modules.get(k) for k in ('jax', 'jax.numpy')}
    sys.modules['jax'], sys.modules['jax.numpy'] = fake, np
    try:
        spec = importlib.util.spec_from_file_location(name, path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
    finally:
        for k, v in saved.items():
            if v is None:
                sys.modules.pop(k, None)
            else:
                sys.modules[k] = v
    mod._backend = 'numpy float64 in place of jax (library source executed unchanged)'
    mod._source = str(path)
    sys.modules[name] = mod
    import timesfm.utils as tu
    tu.xreg_lib = mod
    return mod, mod._backend


# --------------------------------------------------------------------------------------------------
# the zoo
# --------------------------------------------------------------------------------------------------
class Zoo:
    """Loads each checkpoint once, offline, and checks it resolves to its pinned revision."""

    def __init__(self, device=None, threads=None, offline=True, check_pins=True):
        if offline:
            os.environ['HF_HUB_OFFLINE'] = '1'
        import torch
        self.torch = torch
        torch.set_num_threads(int(threads or os.environ.get('V2_THREADS', '4')))
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.check_pins = check_pins
        self.models = {}
        self.revisions = {}

    # ----------------------------------------------------------------------------------- loading
    def _revision(self, family):
        repo = REPOS[family]
        rev, _ = cached_revision(repo)
        if self.check_pins and rev != PINNED[repo]:
            raise RuntimeError(f'{repo} resolves to {rev}, pinned {PINNED[repo]}')
        self.revisions[family] = rev
        return rev

    def model(self, family):
        if family in self.models:
            return self.models[family]
        torch = self.torch
        self._revision(family)
        t0 = time.time()
        if family.startswith('chronos'):
            from chronos import BaseChronosPipeline
            m = BaseChronosPipeline.from_pretrained(REPOS[family], device_map=self.device,
                                                    torch_dtype=torch.float32)
            m.model.eval()
        elif family == 'timesfm25':
            import timesfm
            m = timesfm.TimesFM_2p5_200M_torch.from_pretrained(REPOS[family])   # torch_compile default
        elif family == 'timesfm3':
            from timesfm3 import TimesFM3Forecaster
            m = TimesFM3Forecaster.from_pretrained(REPOS[family], device=self.device,
                                                   per_core_batch_size=BATCH['timesfm3'])
        else:
            raise ValueError(family)
        self.models[family] = m
        self.load_seconds = getattr(self, 'load_seconds', {})
        self.load_seconds[family] = round(time.time() - t0, 1)
        return m

    def release(self, family=None):
        for f in ([family] if family else list(self.models)):
            self.models.pop(f, None)
        if self.torch.cuda.is_available():
            self.torch.cuda.empty_cache()

    # ----------------------------------------------------------------------------------- predict
    def predict(self, arm, contexts, horizon, past_cov=None, future_cov=None, cov_names=None,
                levels=LEVELS, batch_size=None, timesfm_fill=True, drop_dark=True,
                xreg_future=None, xreg_mode='xreg + timesfm', tfm25_config=None, tfm3_flags=None):
        """See the module docstring. `xreg_future='hold_last'` lets timesfm25_cov run on a task that
        gives past-only covariates by holding each covariate's last observed value over the horizon
        (off by default: the declaration does not provide for it)."""
        family, mode = arm_parts(arm)
        levels = tuple(float(a) for a in levels)
        if 0.5 not in levels:
            raise ValueError('levels must include the median 0.5')
        horizon = int(horizon)
        contexts = [np.asarray(c, np.float32).reshape(-1) for c in contexts]
        n = len(contexts)
        lengths = [len(c) for c in contexts]
        past, future = _covariates(n, lengths, horizon, past_cov, future_cov, cov_names)
        if mode == 'uni':
            given = sum(len(p) for p in past)
            past, future = [{} for _ in range(n)], [{} for _ in range(n)]
        else:
            given = None
            if not any(past):
                raise ValueError(f'{arm}: no covariates given')
        dropped = _drop_dark(past, future) if (drop_dark and mode == 'cov') else 0
        t0 = time.time()
        meta = {'arm': arm, 'model_id': REPOS[family], 'n': n, 'horizon': horizon,
                'context_lengths': [int(min(lengths)), int(max(lengths))] if n else [0, 0],
                'device': self.device, 'levels': list(levels)}
        if mode == 'uni' and given:
            meta['covariates_ignored'] = 'univariate arm: covariates given were not used'
        if mode == 'cov':
            names = sorted({k for p in past for k in p})
            meta['covariates'] = {'names': names, 'with_future': sorted({k for f in future for k in f}),
                                  'dropped_dark': dropped, 'drop_dark': bool(drop_dark),
                                  'series_without_any': int(sum(1 for p in past if not p))}
        if family.startswith('chronos_bolt'):
            med, q, m = self._bolt(family, contexts, horizon, levels, batch_size or BATCH[family])
        elif family == 'chronos2':
            med, q, m = self._chronos2(contexts, past, future, horizon, levels, batch_size or BATCH[family])
        elif family == 'timesfm25':
            med, q, m = self._timesfm25(mode, contexts, past, future, horizon, levels, timesfm_fill,
                                        batch_size or BATCH[family], xreg_future, xreg_mode,
                                        tfm25_config or {})
        else:
            med, q, m = self._timesfm3(mode, contexts, past, future, horizon, levels, timesfm_fill,
                                       batch_size or BATCH[family], tfm3_flags or {})
        meta.update(m)
        meta['revision'] = self.revisions[family]
        meta['seconds'] = round(time.time() - t0, 3)
        meta['quantile_crossing_share'] = _crossing_share(q)
        meta['versions'] = _versions()
        return {'median': med.astype(np.float32), 'q': q.astype(np.float32), 'meta': meta}

    # -------------------------------------------------------------------------------- per model
    def _bolt(self, family, contexts, horizon, levels, batch):
        """exp_chronos.predict_at / v2_chronos_adapt.forecast / chronos_fill: predict_quantiles on
        a list of float32 tensors with NaN left for the pipeline to mask."""
        torch = self.torch
        pipe = self.model(family)
        trained = [float(a) for a in pipe.quantiles]
        if not set(levels) <= set(trained):
            raise ValueError(f'{family} returns {trained}; asked {levels}')
        out = []
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')          # the pipeline warns past 64 steps
            for k in range(0, len(contexts), batch):
                chunk = [torch.tensor(c, dtype=torch.float32) for c in contexts[k:k + batch]]
                q, _ = pipe.predict_quantiles(inputs=chunk, prediction_length=horizon,
                                              quantile_levels=list(levels))
                out.append(q.float().cpu().numpy())                         # [B, H, L]
        Q = np.concatenate(out).transpose(0, 2, 1) if out else np.zeros((0, len(levels), horizon), np.float32)
        native = int(pipe.model_prediction_length)
        meta = {'library_levels': trained, 'level_method': 'returned by the library',
                'nan_treatment': 'NaN passed through; the pipeline masks it (left-padded with NaN to the batch length)',
                'context_limit': int(pipe.model_context_length),
                'horizon_method': (f'one call; native {native} steps; beyond, the pipeline autoregresses on its '
                                   f'nine quantile paths and reduces them to nine levels (chronos_bolt.py predict)'),
                'native_horizon': native, 'autoregressive_blocks': max(0, math.ceil(horizon / native) - 1),
                'batch': batch, 'call': 'BaseChronosPipeline.predict_quantiles'}
        return Q[:, levels.index(0.5), :], Q, meta

    def _chronos2(self, contexts, past, future, horizon, levels, batch):
        """v2_chronos2.forecast / v2_probabilistic.predict / chronos_fill: dict items, NaN in the
        target kept, items grouped by covariate names, `batch` items per pipe.predict call."""
        torch = self.torch
        pipe = self.model('chronos2')
        qs = [float(a) for a in pipe.quantiles]
        items = []
        for c, p, f in zip(contexts, past, future):
            it = {'target': c.astype(np.float32)}
            if p:
                it['past_covariates'] = {k: v.astype(np.float32) for k, v in p.items()}
            if f:
                it['future_covariates'] = {k: v.astype(np.float32) for k, v in f.items()}
            items.append(it)
        result = [None] * len(items)
        groups = {}
        for n_, it in enumerate(items):
            groups.setdefault((tuple(sorted(it.get('past_covariates', {}))),
                               tuple(sorted(it.get('future_covariates', {})))), []).append(n_)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            for members in groups.values():
                for k in range(0, len(members), batch):
                    chunk = members[k:k + batch]
                    got = pipe.predict([items[n_] for n_ in chunk], prediction_length=horizon)
                    for n_, o in zip(chunk, got):
                        a = o.detach().float().cpu().numpy() if torch.is_tensor(o) else np.asarray(o)
                        result[n_] = a.reshape(-1, len(qs), a.shape[-1])[0]      # [21, H]
        R = np.stack(result).transpose(0, 2, 1) if result else np.zeros((0, horizon, len(qs)), np.float32)
        Q, how = select_levels(qs, R, levels)
        Q = Q.transpose(0, 2, 1)
        native = int(pipe.model_prediction_length)
        meta = {'library_levels': qs, 'level_method': how,
                'nan_treatment': 'NaN passed through in target and covariates; the pipeline masks it',
                'context_limit': int(pipe.model_context_length),
                'horizon_method': (f'one call; native up to {native} steps '
                                   f'({pipe.max_output_patches} output patches of {pipe.model_output_patch_size}); '
                                   'beyond, the pipeline unrolls its quantiles autoregressively'),
                'native_horizon': native, 'batch': batch, 'schemas': len(groups),
                'call': 'Chronos2Pipeline.predict (cross_learning=False)'}
        return Q[:, levels.index(0.5), :], Q, meta

    def _timesfm25(self, mode, contexts, past, future, horizon, levels, fill, per_core, xreg_future,
                   xreg_mode, overrides):
        """v2_sequence_arms.timesfm_predict: TimesFM_2p5_200M_torch.from_pretrained, compile with
        ForecastConfig(max_context=512, max_horizon=H, normalize_inputs=True), forecast(); the
        quantile channels 1..9 are the levels 0.1..0.9, channel 5 (the median) is the point."""
        import timesfm
        model = self.model('timesfm25')
        rec = {}
        if fill:
            contexts, past, future, rec = timesfm_inputs(contexts, past, future)
        maxlen = max([len(c) for c in contexts if c is not None] or [1])
        max_context = max(MIN_CONTEXT_TFM, int(math.ceil(maxlen / 32) * 32))
        cfg = dict(TFM25_CONFIG, **overrides)
        cfg.update(max_context=max_context, max_horizon=horizon, per_core_batch_size=per_core)
        library_levels = list(model.model.config.quantiles)
        N = len(contexts)
        Q10 = np.full((N, horizon, 10), np.nan, np.float32)
        live = [i for i, c in enumerate(contexts) if c is not None]
        meta = {'library_levels': library_levels, 'level_method': 'returned by the library',
                'nan_treatment': ('declared rule applied here (leading trimmed, interior linear in time, '
                                  'trailing held)' if fill else 'none here; the library trims leading NaN and '
                                  'interpolates the rest itself (timesfm_2p5_base.forecast)'),
                'nan_record': rec, 'per_core_batch_size': per_core,
                'call': 'TimesFM_2p5_200M_torch.forecast' if mode == 'uni' else
                        'TimesFM_2p5_200M_torch.forecast_with_covariates'}
        if mode == 'uni':
            model.compile(timesfm.ForecastConfig(**cfg))
            fc = model.forecast_config
            for k in range(0, len(live), 256):
                rows = live[k:k + 256]
                _, quant = model.forecast(horizon=horizon, inputs=[contexts[i] for i in rows])
                Q10[rows] = np.asarray(quant)[:, :horizon, :]
        else:
            xl, backend = _xreg_lib()
            cfg['return_backcast'] = True
            model.compile(timesfm.ForecastConfig(**cfg))
            fc = model.forecast_config
            held = 0
            no_cov = []
            for i in live:
                p, f = past[i], future[i]
                if not p:
                    no_cov.append(i)
                    continue
                missing = [k for k in p if k not in f]
                if missing:
                    if xreg_future != 'hold_last':
                        raise ValueError('timesfm25_cov: XReg needs every covariate over the horizon; '
                                         f'{missing} have no future part (xreg_future="hold_last" holds the last value)')
                    f = dict(f)
                    for k in missing:
                        f[k] = np.full(horizon, p[k][-1], np.float32)
                    held += 1
                dyn = {k: [np.concatenate([p[k], f[k]]).astype(np.float64)] for k in sorted(p)}
                pts, qts = model.forecast_with_covariates(
                    inputs=[contexts[i].astype(np.float64)], dynamic_numerical_covariates=dyn,
                    xreg_mode=xreg_mode, normalize_xreg_target_per_input=True, ridge=0.0)
                Q10[i] = np.asarray(qts[0])[:horizon]
            if no_cov:                      # every covariate dark: the univariate forecast stands
                model.compile(timesfm.ForecastConfig(**dict(cfg, return_backcast=False)))
                for k in range(0, len(no_cov), 256):
                    rows = no_cov[k:k + 256]
                    _, quant = model.forecast(horizon=horizon, inputs=[contexts[i] for i in rows])
                    Q10[rows] = np.asarray(quant)[:, :horizon, :]
            meta.update(xreg={'mode': xreg_mode, 'scope': 'one series per call (the library fits one regression per call)',
                              'ridge': 0.0, 'normalize_xreg_target_per_input': True, 'backend': backend,
                              'future': xreg_future or 'given', 'series_with_held_future': held,
                              'series_without_covariates_forecast_univariate': len(no_cov)})
        median = Q10[:, :, 5]
        Q = Q10[:, :, 1:10].transpose(0, 2, 1)
        nonneg = [i for i in live if np.nanmin(contexts[i]) >= 0]
        meta.update({
            'forecast_config': {k: getattr(fc, k) for k in ('max_context', 'max_horizon', 'normalize_inputs',
                                                             'per_core_batch_size', 'use_continuous_quantile_head',
                                                             'force_flip_invariance', 'infer_is_positive',
                                                             'fix_quantile_crossing', 'return_backcast')},
            'horizon_method': ('one call; decodes the compiled max_horizon (H rounded up to 128) and truncates; '
                               'beyond 128, autoregressive in 128-step output patches'),
            'native_horizon': 128, 'autoregressive_steps': (fc.max_horizon - 1) // 128,
            'context_limit': 16384,
            'positivity_clamp': {'series_with_nonnegative_context': len(nonneg),
                                 'of_them_with_a_zero_output': int(sum(1 for i in nonneg if (Q10[i, :, 1:] == 0).any()))},
            'series_without_data': int(N - len(live))})
        return median, Q, meta

    def _timesfm3(self, mode, contexts, past, future, horizon, levels, fill, per_core, flags):
        """TimesFM3Forecaster.predict_batch with the library's zero-shot benchmark flags; covariates
        with a future part as past_future_covariates, the others as past_only_covariates."""
        fc = self.model('timesfm3')
        if fc.config.per_core_batch_size != per_core:
            import dataclasses
            fc.config = dataclasses.replace(fc.config, per_core_batch_size=per_core)
        flags = dict(TFM3_FLAGS, **flags)
        rec = {}
        if fill:
            contexts, past, future, rec = timesfm_inputs(contexts, past, future)
        library_levels = [float(a) for a in fc.config.quantiles]
        N = len(contexts)
        R = np.full((N, horizon, len(library_levels)), np.nan, np.float32)
        groups = {}
        for i, c in enumerate(contexts):
            if c is None:
                continue
            po = tuple(sorted(k for k in past[i] if k not in future[i]))
            pf = tuple(sorted(k for k in past[i] if k in future[i]))
            groups.setdefault((po, pf), []).append(i)
        for (po, pf), rows in groups.items():
            po_list = [np.stack([past[i][k] for k in po]) for i in rows] if po else None
            pf_list = [np.stack([np.concatenate([past[i][k], future[i][k]]) for k in pf]) for i in rows] if pf else None
            outs = list(fc.predict_batch(contexts=[contexts[i] for i in rows], horizon=horizon,
                                         past_only_covariates=po_list, past_future_covariates=pf_list,
                                         return_quantiles=True, **flags))
            for i, o in zip(rows, outs):
                R[i] = np.asarray(o.quantiles)[:horizon]
        Q, how = select_levels(library_levels, R, levels)
        Q = Q.transpose(0, 2, 1)
        live = [i for i in range(N) if contexts[i] is not None]
        nonneg = [i for i in live if np.nanmin(contexts[i]) >= 0]
        cfg = fc.config
        meta = {'library_levels': library_levels, 'level_method': how,
                'nan_treatment': ('declared rule applied here (leading trimmed, interior linear in time, '
                                  'trailing held)' if fill else 'none here; the library trims leading NaN and '
                                  'interpolates the rest itself (predict_batch)'),
                'nan_record': rec, 'flags': flags, 'per_core_batch_size': cfg.per_core_batch_size,
                'model_config': {k: getattr(cfg, k) for k in ('input_patch_length', 'output_patch_length',
                                                               'use_stitching', 'use_linear_detrending',
                                                               'linear_detrending_threshold', 'use_iterative_cpm_revin',
                                                               'use_variate_attention', 'median_quantile_index')},
                'schemas': len(groups),
                'horizon_method': ('one non-autoregressive pass: the horizon enters as masked patches and is '
                                   'stitched from overlapping 64-step patch outputs (model.decode)'),
                'context_limit': int(fc.global_context),
                'positivity_clamp': {'series_with_nonnegative_context': len(nonneg),
                                     'of_them_with_a_zero_output': int(sum(1 for i in nonneg if (R[i] == 0).any()))},
                'series_without_data': int(N - len(live)),
                'call': 'TimesFM3Forecaster.predict_batch'}
        if mode == 'cov':
            meta['covariate_roles'] = {'past_only': sorted({k for (po, _) in groups for k in po}),
                                       'past_future': sorted({k for (_, pf) in groups for k in pf})}
        return Q[:, levels.index(0.5), :], Q, meta


_VERSIONS = None


def _versions():
    global _VERSIONS
    if _VERSIONS is None:
        import importlib.metadata as md
        _VERSIONS = {}
        for p in ('chronos-forecasting', 'timesfm', 'torch', 'transformers', 'huggingface_hub', 'numpy'):
            try:
                _VERSIONS[p] = md.version(p)
            except md.PackageNotFoundError:
                _VERSIONS[p] = None
    return _VERSIONS


_ZOO = None


def predict(arm, contexts, horizon, past_cov=None, future_cov=None, levels=LEVELS, **kw):
    """Module-level shortcut on one shared Zoo."""
    global _ZOO
    if _ZOO is None:
        _ZOO = Zoo()
    return _ZOO.predict(arm, contexts, horizon, past_cov=past_cov, future_cov=future_cov, levels=levels, **kw)


def info():
    zoo = Zoo()
    out = {}
    for family in REPOS:
        m = zoo.model(family)
        d = {'repo': REPOS[family], 'revision': zoo.revisions[family], 'load_seconds': zoo.load_seconds[family]}
        if family.startswith('chronos'):
            d.update(levels=[float(a) for a in m.quantiles], native_horizon=int(m.model_prediction_length),
                     context_limit=int(m.model_context_length), pipeline=type(m).__name__)
            if family == 'chronos2':
                d.update(max_output_patches=int(m.max_output_patches), output_patch=int(m.model_output_patch_size))
        elif family == 'timesfm25':
            c = m.model.config
            d.update(levels=list(c.quantiles), output_patch=int(c.output_patch_len),
                     quantile_head_horizon=int(c.output_quantile_len), context_limit=int(c.context_limit))
        else:
            c = m.config
            d.update(levels=list(c.quantiles), output_patch=int(c.output_patch_length),
                     input_patch=int(c.input_patch_length), use_stitching=bool(c.use_stitching),
                     context_limit=int(m.global_context))
        out[family] = d
        zoo.release(family)
    print(json.dumps(out, indent=2))
    return out


if __name__ == '__main__':
    if '--info' in sys.argv:
        info()
