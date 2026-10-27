"""Addendum 81 on the compute host: every applicable arm on every input file of the six tasks.

  python v2_fm_zoo.py plan    [--tasks ABFCDE] [--files a.npz,b.npz] [--arms ...]
  python v2_fm_zoo.py run     --worker NAME [--out preds] [--limit N] [--tasks ...] [--files ...] [--arms ...]
  python v2_fm_zoo.py rerun   [--out preds] [--n 64]          determinism check, preds/rerun_check.json
  python v2_fm_zoo.py inspect [--out preds]                   shapes, NaN counts and meta of the outputs

Amendment 3 options (run, recorded in every output's meta under `overrides`, `amendments` and `command`):
  --files A_*,F_*            names or shell patterns of input files
  --batch 1                  the batch size of every arm in the plan (TimesFM 3.0: per_core_batch_size)
  --tfm3-flags make_positive=false        merged over v2_fm_zoo_models.TFM3_FLAGS
  --tfm25-config infer_is_positive=false  merged over v2_fm_zoo_models.TFM25_CONFIG
  --amendment PATH [--amendment-sha256 HEX]   cited in the meta beside amendments 1 and 2
  python v2_fm_zoo.py supersede --out preds --to DIR --arms ... [--tasks ...] --amendment PATH --reason KEY
                             move the outputs an amendment supersedes out of the declared folder (kept)
  python v2_fm_zoo.py adopt --src DIR --out preds --amendment PATH --reason KEY
                             copy outputs an amendment declares into the declared folder, meta updated
  python v2_fm_zoo.py rerun --files ... --arms ... --require-amendment PATH --report rerun_check_a3.json
                             determinism of the amended outputs, each with the options its meta records
  python v2_fm_zoo.py clamp-effect --out preds --superseded DIR   task A, TimesFM 2.5 clamp on vs off
  python v2_fm_zoo.py verify-declared --out preds   every (input, arm) present once, input digests match

Declared in config/bay_reserved_protocol_v2_addendum81.json with amendments 1, 2 and 3. Nothing is fitted,
tuned or selected here and nothing is scored: the task scorers read the predictions back.

Inputs (inputs/ beside this script) are the files the three manifests list, each checked against the
manifest's sha256 before it is run; nothing else is a model input (the F reconstruction file and any
alternative AOU-unit build are not). For every input file and every applicable arm one prediction file
<out>/<input stem>__<arm>.npz is written with v2_fm_zoo_io.save_predictions: median [N, H], q [N, 9, H] at
the levels 0.1 ... 0.9, rows in the input file's order, meta with the input's digest, the arm, the model
revision (weights.json and the cache), the adapter's own meta and the wall time. A file that exists with
the same input digest and arm is skipped, so a run resumes where it stopped.

Applicable arms: chronos_bolt_small_uni and chronos_bolt_base_uni (univariate only); chronos2, timesfm25
and timesfm3 univariate always, and with covariates wherever the file carries covariates.

How each rule of the declaration is applied (`OPTIONS`, `groups`, `run_d_cov`):
  horizon        the file's meta: 6 for A, B, F; 72 for C and D; the gap length for E.
  missing values Chronos-Bolt and Chronos-2 get the contexts and covariates as stored (NaN included);
                 TimesFM 2.5 and 3.0 get the adapter's declared fill (timesfm_fill=True; the adapter
                 applies it to TimesFM only). A covariate with no finite past, or no finite future where
                 one is given, is dropped per item (drop_dark=True, amendment 1).
  TimesFM 2.5    covariates through XReg; xreg_future='hold_last' where the file gives past-only
                 covariates (A, B, C, F), future covariates as stored in D and E.
  task D, cov    amendment 1 no_lookahead_in_task_D: one call per lead h = 6, 12, ..., 72, prediction
                 length 72, the future covariates after onset + h replaced by the last value available at
                 onset + h (the stored value at onset + h, or the last finite one before it; none, the
                 covariate is dark and dropped for that item), and only step h kept. The stored [N, 72]
                 arrays are NaN at every hour that is not a lead. Univariate arms: one call, horizon 72.
  task E         forward and backward files are separate inputs; a backward context is already reversed
                 in time, covariates too, and its prediction is stored in the model's own (reversed)
                 order: the scorer flips and blends.
  task A         standardised contexts and physical covariates, passed as stored.
  batches        each task's published code path: Chronos-Bolt 256 for A (v2_sequence_arms) and for B and
                 F (exp_chronos.predict_at, 256 per sensor, issuances in time order), 64 for C (no
                 published Chronos-Bolt arm: the adapter's default), D (v2_chronos_adapt.forecast) and E
                 (chronos_fill); Chronos-2 128 for A (v2_sequence_sweep), B and F (v2_chronos2.forecast,
                 per sensor), C (v2_weather_oracle.predict, per sensor), D (adapter default), 64 for E
                 (chronos_fill); TimesFM 2.5 one series per decode (the published arm), TimesFM 3.0 32.
                 B, C and F are called per sensor, issuances in time order, as their published paths.
"""
import argparse
import datetime as dt
import errno
import fnmatch
import json
import os
import socket
import sys
import time
import traceback
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
for p in (HERE, HERE.parent):                    # flat on the host; scripts/ and scripts/experiments/ locally
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))
import v2_fm_zoo_io as zio  # noqa: E402

INPUTS = Path(os.environ.get('V2_FM_ZOO_INPUTS', HERE / 'inputs'))      # the variable: a local plan only
MANIFESTS = ('manifest_6h.json', 'manifest_dist_bridge.json', 'manifest_gaps.json')
TASK_ORDER = {'A': 0, 'B': 0, 'F': 0, 'C': 1, 'D': 2, 'E': 3}
LEVELS = zio.LEVELS
D_LEADS = tuple(range(6, 73, 6))
#: per task: Chronos-Bolt batch, Chronos-2 batch, whether calls are made per sensor in time order
OPTIONS = {'A': {'bolt': 256, 'chronos2': 128, 'per_sensor': False},
           'B': {'bolt': 256, 'chronos2': 128, 'per_sensor': True},
           'F': {'bolt': 256, 'chronos2': 128, 'per_sensor': True},
           'C': {'bolt': 64, 'chronos2': 128, 'per_sensor': True},
           'D': {'bolt': 64, 'chronos2': 128, 'per_sensor': False},
           'E': {'bolt': 64, 'chronos2': 64, 'per_sensor': False}}
#: relative cost per context-hour, only to order the work (long jobs first within a task group)
COST = {'chronos_bolt_small': 0.02, 'chronos_bolt_base': 0.03, 'chronos2': 0.02, 'timesfm25': 1.0,
        'timesfm3': 0.15}
SMOKE_FILES = ('A_oxygen_offset0.npz', 'B_oxygen_Reserved-2026.npz', 'F_oxygen_aou_Reserved-2026.npz',
               'C_temperature_Reserved-2026.npz', 'D_temperature_Bridge-2026.npz',
               'E_temperature_Gap-2026_fwd_L24.npz', 'E_temperature_Gap-2026_bwd_L24.npz',
               'E_oxygen_Gap-Dev_fwd_L1.npz', 'E_temperature_Gap-Long-Dev_bwd_L1440.npz')
RERUN_FILES = ('A_temperature_offset0.npz', 'B_temperature_Reserved-2026.npz', 'F_oxygen_aou_Reserved-2026.npz',
               'C_oxygen_Reserved-2026.npz', 'D_oxygen_Bridge-Dev.npz', 'E_temperature_Gap-Dev_bwd_L72.npz')


def now():
    return dt.datetime.now(dt.timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')


# --------------------------------------------------------------------------------------------------
# the plan
# --------------------------------------------------------------------------------------------------
def catalogue():
    """[(name, manifest sha256, manifest)] for every model input the three manifests list."""
    out = {}
    m6 = json.loads((INPUTS / 'manifest_6h.json').read_text(encoding='utf-8'))
    for name, e in m6['files'].items():
        out[name] = (e['sha256'], 'manifest_6h.json')
    md = json.loads((INPUTS / 'manifest_dist_bridge.json').read_text(encoding='utf-8'))
    for section in ('C_distribution', 'D_silent_probe'):
        for name, e in md[section]['files'].items():
            out[name] = (e['sha256'], 'manifest_dist_bridge.json')
    mg = json.loads((INPUTS / 'manifest_gaps.json').read_text(encoding='utf-8'))
    for name, e in mg['files'].items():
        out[name] = (e['sha256'], 'manifest_gaps.json')
    bad = [n for n in out if n.endswith('_reconstruction.npz') or not n.endswith('.npz')]
    if bad:
        raise SystemExit(f'not model inputs, listed in a manifest: {bad}')
    return out


def header(path):
    """meta, number of contexts and whether the file carries covariates, without loading the arrays."""
    with np.load(path, allow_pickle=False) as z:
        meta = json.loads(str(z['meta']))
        has_cov = 'pcov_values' in z.files or 'fcov_values' in z.files
        has_future = 'fcov_values' in z.files
    return meta, has_cov, has_future


def applicable(has_cov):
    arms = ['chronos_bolt_small_uni', 'chronos_bolt_base_uni', 'chronos2_uni', 'timesfm25_uni', 'timesfm3_uni']
    if has_cov:
        arms += ['chronos2_cov', 'timesfm25_cov', 'timesfm3_cov']
    return arms


def matches(name, files):
    """True if `files` (a set of names or shell patterns such as A_*) selects the input `name`."""
    return name in files or any(fnmatch.fnmatchcase(name, p) for p in files)


def plan(tasks=None, files=None, arms=None):
    cat = catalogue()
    jobs = []
    for name, (sha, manifest) in cat.items():
        task = name[0]
        if tasks and task not in tasks:
            continue
        if files and not matches(name, files):
            continue
        path = INPUTS / name
        if not path.exists():
            raise SystemExit(f'{name}: listed in {manifest} but missing from {INPUTS}')
        meta, has_cov, has_future = header(path)
        if meta['task'][0] != task:
            raise SystemExit(f'{name}: task {meta["task"]} does not match its name')
        for arm in applicable(has_cov and bool(meta['cov_names'])):
            if arms and arm not in arms:
                continue
            family = arm.rsplit('_', 1)[0]
            calls = len(D_LEADS) if (task == 'D' and arm.endswith('_cov')) else 1
            cost = COST[family] * meta['n'] * meta['horizon'] ** 0.5 * calls * (1.3 if arm.endswith('_cov') else 1)
            jobs.append({'file': name, 'stem': name[:-4], 'task': task, 'arm': arm, 'n': meta['n'],
                         'horizon': meta['horizon'], 'sha256': sha, 'manifest': manifest, 'cost': cost,
                         'future_cov': has_future})
    jobs.sort(key=lambda j: (TASK_ORDER[j['task']], -j['cost'], j['file'], j['arm']))
    return jobs


# --------------------------------------------------------------------------------------------------
# one job
# --------------------------------------------------------------------------------------------------
def groups(task, keys, n):
    """Index arrays, one per call: the whole file in order, or (B, C, F) per sensor with issuances in
    time order, as exp_chronos.predict_at, v2_chronos2.forecast and v2_weather_oracle.run_chronos call."""
    if not OPTIONS[task]['per_sensor']:
        return [np.arange(n)]
    sensors = keys['sensor'].to_numpy().astype(str)[:n]
    times = keys['time'].to_numpy()[:n]
    out = []
    for s in dict.fromkeys(sensors):
        idx = np.flatnonzero(sensors == s)
        out.append(idx[np.argsort(times[idx], kind='stable')])
    return out


#: NOT the declared configuration. TimesFM 2.5 refuses the continuous quantile head that amendment 2
#: declares for horizons above 1024 ("Continuous quantile head is not supported for horizons > 1024"),
#: so the 1440 h gap files have no declared TimesFM 2.5 run. With --tfm25-no-continuous-head the head is
#: switched off (everything else as declared; amendment 2 records that the head does not move the median)
#: and the output must go to a folder of its own, for a decision by amendment.
TFM25_NO_CQH = {'use_continuous_quantile_head': False}


#: the library options an override may name (amendment 3: the positivity clamps)
TFM25_KEYS = ('normalize_inputs', 'use_continuous_quantile_head', 'fix_quantile_crossing', 'infer_is_positive',
              'force_flip_invariance')
TFM3_KEYS = ('use_symmetric_averaging', 'make_positive', 'sort_quantiles', 'use_znorm', 'padding_mode')


def arm_overrides(arm, batch=None, tfm25_config=None, tfm3_flags=None):
    """The command-line overrides that apply to `arm`: the batch size to every arm of the plan, the
    TimesFM 2.5 configuration to timesfm25 arms, the TimesFM 3.0 flags to timesfm3 arms."""
    family = arm.rsplit('_', 1)[0]
    o = {}
    if batch:
        o['batch_size'] = int(batch)
    if family == 'timesfm25' and tfm25_config:
        o['tfm25_config'] = dict(tfm25_config)
    if family == 'timesfm3' and tfm3_flags:
        o['tfm3_flags'] = dict(tfm3_flags)
    return o


def arm_kwargs(task, arm, future_cov, tfm25_no_cqh=False, overrides=None):
    family = arm.rsplit('_', 1)[0]
    kw = {'timesfm_fill': True, 'drop_dark': True}
    if family == 'timesfm25' and tfm25_no_cqh:
        kw['tfm25_config'] = dict(TFM25_NO_CQH)
    if family.startswith('chronos_bolt'):
        kw['batch_size'] = OPTIONS[task]['bolt']
    elif family == 'chronos2':
        kw['batch_size'] = OPTIONS[task]['chronos2']
    if family == 'timesfm25' and arm.endswith('_cov') and not future_cov:
        kw['xreg_future'] = 'hold_last'
    o = overrides or {}
    if o.get('batch_size'):
        kw['batch_size'] = int(o['batch_size'])
    if o.get('tfm25_config'):
        kw['tfm25_config'] = dict(kw.get('tfm25_config', {}), **o['tfm25_config'])
    if o.get('tfm3_flags'):
        kw['tfm3_flags'] = dict(o['tfm3_flags'])
    return kw


def parse_options(text, allowed, what):
    """'a=false,b=1' -> {'a': False, 'b': 1}; each value read as JSON, else kept as a string."""
    out = {}
    for part in (text or '').split(','):
        if not part.strip():
            continue
        k, _, v = part.partition('=')
        k = k.strip()
        if k not in allowed:
            raise SystemExit(f'{what}: unknown option {k!r}; one of {allowed}')
        try:
            out[k] = json.loads(v.strip())
        except json.JSONDecodeError:
            out[k] = v.strip()
    return out


def carry_forward(F, h):
    """Amendment 1: the future covariates after onset + h replaced by the last value available at
    onset + h. F [N, C, H]; positions 0 .. h - 1 (onset + 1 .. onset + h) are kept as stored."""
    G = F.copy()
    head = F[:, :, :h]
    ok = np.isfinite(head)
    last_idx = np.where(ok, np.arange(h), -1).max(axis=-1)                     # [N, C]
    last = np.take_along_axis(head, np.clip(last_idx, 0, None)[..., None], axis=-1)[..., 0]
    last = np.where(last_idx >= 0, last, np.nan)
    G[:, :, h:] = last[..., None]
    rec = {'covariate_items_value_at_lead_missing_carried_from_earlier': int(((last_idx >= 0) & (last_idx < h - 1)).sum()),
           'covariate_items_without_any_value_up_to_lead': int((last_idx < 0).sum())}
    return G, rec


def slim(meta):
    drop = ('versions',)
    return {k: v for k, v in meta.items() if k not in drop}


class Runner:
    def __init__(self, worker, out, limit=None, tfm25_no_cqh=False, batch=None, tfm25_config=None,
                 tfm3_flags=None, amendments=(), amendment_sha256=None):
        self.worker, self.out, self.limit = worker, Path(out), limit
        self.tfm25_no_cqh = tfm25_no_cqh
        self.batch, self.tfm25_config, self.tfm3_flags = batch, tfm25_config or {}, tfm3_flags or {}
        self.amendments = list(amendments)
        self.amendment_sha256 = dict(amendment_sha256 or {})
        self.command = ' '.join(sys.argv[1:])
        if tfm25_no_cqh and self.out.name == 'preds':
            raise SystemExit('--tfm25-no-continuous-head is not the declared configuration: write it elsewhere')
        self.out.mkdir(parents=True, exist_ok=True)
        (self.out / 'tmp').mkdir(exist_ok=True)
        (HERE / 'logs').mkdir(exist_ok=True)
        (HERE / 'status').mkdir(exist_ok=True)
        self.claims = self.out / 'claims'
        self.claims.mkdir(exist_ok=True)
        self.failed_dir = self.out / 'failed'
        self.failed_dir.mkdir(exist_ok=True)
        self.log_path = HERE / 'logs' / f'zoo_{worker}.log'
        self.status_path = HERE / 'status' / f'{worker}.json'
        self.status = {'worker': worker, 'pid': os.getpid(), 'host': socket.gethostname(), 'out': str(self.out),
                       'limit': limit, 'started': now(), 'updated': now(), 'current': None, 'done': [],
                       'skipped_existing': 0, 'failed': [], 'finished': False}
        self.weights = json.loads((HERE / 'weights.json').read_text(encoding='utf-8'))['models']
        self.code = {n: zio.digest(HERE / n) for n in ('v2_fm_zoo.py', 'v2_fm_zoo_models.py', 'v2_fm_zoo_io.py')
                     if (HERE / n).exists()}
        self._zoo = None
        self._inputs = {}

    # ------------------------------------------------------------------------------------ bookkeeping
    def log(self, msg):
        line = f'{now()} [{self.worker}] {msg}'
        print(line, flush=True)
        with open(self.log_path, 'a', encoding='utf-8') as f:
            f.write(line + '\n')

    def save_status(self):
        self.status['updated'] = now()
        tmp = self.status_path.with_suffix('.tmp')
        tmp.write_bytes((json.dumps(self.status, indent=1, default=str) + '\n').encode('utf-8'))
        os.replace(tmp, self.status_path)

    @property
    def zoo(self):
        if self._zoo is None:
            import v2_fm_zoo_models as zm
            self.zm = zm
            self._zoo = zm.Zoo()
            self.log(f'zoo on {self._zoo.device}, torch threads {self._zoo.torch.get_num_threads()}')
        return self._zoo

    def output(self, job):
        return self.out / f'{job["stem"]}__{job["arm"]}.npz'

    def overrides(self, arm):
        return arm_overrides(arm, self.batch, self.tfm25_config, self.tfm3_flags)

    def set_overrides(self, o):
        """Take the overrides an output's meta records (rerun)."""
        o = o or {}
        self.batch, self.tfm25_config, self.tfm3_flags = (o.get('batch_size'), o.get('tfm25_config') or {},
                                                          o.get('tfm3_flags') or {})

    def is_done(self, job):
        p = self.output(job)
        if not p.exists():
            return False
        try:
            meta = zio.load_predictions(p)['meta']
        except Exception:
            return False
        return (meta.get('input_sha256') == job['sha256'] and meta.get('arm') == job['arm']
                and meta.get('limit') == self.limit
                and (meta.get('overrides') or {}) == self.overrides(job['arm']))

    def other_options(self, job):
        """An output for this (input, arm) written with other options: never overwritten (supersede it
        first, so that the record keeps it)."""
        p = self.output(job)
        if not p.exists():
            return None
        try:
            meta = zio.load_predictions(p)['meta']
        except Exception:
            return None
        if meta.get('input_sha256') == job['sha256'] and meta.get('arm') == job['arm'] and not self.is_done(job):
            return {'overrides': meta.get('overrides') or {}, 'amendments': meta.get('amendments')}
        return None

    def claim(self, job):
        path = self.claims / f'{job["stem"]}__{job["arm"]}.claim'
        for _ in range(2):
            try:
                fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            except OSError as e:
                if e.errno != errno.EEXIST:
                    raise
                try:
                    pid = int(json.loads(path.read_text())['pid'])
                    os.kill(pid, 0)
                    return None                                   # held by a live worker
                except (ProcessLookupError, ValueError, KeyError, json.JSONDecodeError, FileNotFoundError):
                    path.unlink(missing_ok=True)                  # a dead worker's claim
                    continue
                except PermissionError:
                    return None
            with os.fdopen(fd, 'w') as f:
                f.write(json.dumps({'worker': self.worker, 'pid': os.getpid(), 'since': now()}))
            return path
        return None

    def inputs(self, job):
        name = job['file']
        if name not in self._inputs:
            path = INPUTS / name
            sha = zio.digest(path)
            if sha != job['sha256']:
                raise RuntimeError(f'{name}: sha256 {sha} differs from {job["manifest"]} ({job["sha256"]})')
            data = zio.load_inputs(path)
            n = len(data['contexts']) if self.limit is None else min(self.limit, len(data['contexts']))
            data['n_run'] = n
            self._inputs = {name: data}                           # one file in memory at a time
        return self._inputs[name]

    # ------------------------------------------------------------------------------------ prediction
    def predict(self, job, data, rows=None):
        """median [n, H], q [n, 9, H], meta for the rows `rows` (default: the first n_run)."""
        task, arm = job['task'], job['arm']
        n_all = data['n_run']
        rows = np.arange(n_all) if rows is None else np.asarray(rows)
        H = int(data['meta']['horizon'])
        ctx = [data['contexts'][i] for i in rows]
        uni = arm.endswith('_uni')                                # the target's own history only
        past = [data['past_cov'][i] for i in rows] if (data['past_cov'] is not None and not uni) else None
        future = data['future_cov'][rows] if (data['future_cov'] is not None and not uni) else None
        names = data['meta']['cov_names']
        kw = arm_kwargs(task, arm, data['future_cov'] is not None, self.tfm25_no_cqh, self.overrides(arm))
        if getattr(self, 'batch_override', None):
            kw['batch_size'] = self.batch_override             # diagnostics only (batch1check)
        keys = data['keys'].iloc[rows].reset_index(drop=True)
        plan_ = groups(task, keys, len(rows))
        median = np.full((len(rows), H), np.nan, np.float32)
        q = np.full((len(rows), len(LEVELS), H), np.nan, np.float32)
        metas = []
        d_cov = task == 'D' and arm.endswith('_cov')
        for g in plan_:
            sub_ctx = [ctx[i] for i in g]
            sub_past = [past[i] for i in g] if past is not None else None
            sub_future = future[g] if future is not None else None
            if not d_cov:
                r = self._call(arm, sub_ctx, H, sub_past, sub_future, names, kw)
                median[g], q[g] = r['median'], r['q']
                metas.append(r['meta'])
                continue
            per_lead = {}
            for h in D_LEADS:
                fut_h, rec = carry_forward(sub_future, h)
                r = self._call(arm, sub_ctx, H, sub_past, fut_h, names, kw)
                median[g, h - 1] = r['median'][:, h - 1]
                q[g, :, h - 1] = r['q'][:, :, h - 1]
                m = r['meta']
                per_lead[h] = {'seconds': m.get('seconds'), 'covariates': m.get('covariates'),
                               'nan_record': m.get('nan_record'), 'xreg': m.get('xreg'),
                               'covariate_roles': m.get('covariate_roles'), 'carry_forward': rec,
                               'positivity_clamp': m.get('positivity_clamp'),
                               'quantile_crossing_share': m.get('quantile_crossing_share')}
                if h == D_LEADS[0]:
                    first = m
            first = dict(first)
            first['per_lead'] = per_lead
            metas.append(first)
        meta = self._merge(metas, plan_)
        meta['call_options'] = {k: v for k, v in kw.items()}
        if d_cov:
            meta['task_d_no_lookahead'] = {
                'rule': ('config/bay_reserved_protocol_v2_addendum81_amendment1.json no_lookahead_in_task_D: one '
                         'call per lead h with prediction length 72; future covariates after onset + h replaced by '
                         'the last value available at onset + h (the stored value at onset + h, else the last '
                         'finite one before it; none: dark, dropped for the item); only step h kept'),
                'leads': list(D_LEADS),
                'stored': 'median and q hold step h - 1 for h in leads; NaN at every other hour'}
        return median, q, meta

    def _call(self, arm, ctx, H, past, future, names, kw):
        zoo = self.zoo
        batch = kw.get('batch_size')
        while True:
            try:
                r = zoo.predict(arm, ctx, H, past_cov=past, future_cov=future, cov_names=names, **dict(kw, batch_size=batch))
                if batch != kw.get('batch_size'):
                    r['meta']['batch_fallback'] = {'declared': kw.get('batch_size'), 'used': batch,
                                                   'reason': 'CUDA out of memory at the declared batch'}
                return r
            except zoo.torch.cuda.OutOfMemoryError:
                zoo.torch.cuda.empty_cache()
                cur = batch or self.zm.BATCH[arm.rsplit('_', 1)[0]]
                if cur <= 1:
                    raise
                batch = max(1, cur // 2)
                self.log(f'  out of memory; retrying {arm} with batch {batch}')

    @staticmethod
    def _merge(metas, plan_):
        if not metas:
            return {}
        meta = dict(metas[0])
        meta['calls'] = len(metas)
        meta['rows_per_call'] = [int(len(g)) for g in plan_] if len(plan_) <= 64 else \
            {'calls': len(plan_), 'min': int(min(len(g) for g in plan_)), 'max': int(max(len(g) for g in plan_))}
        meta['seconds'] = round(sum(m.get('seconds', 0) or 0 for m in metas), 3)
        if len(metas) > 1:
            for key in ('nan_record', 'positivity_clamp', 'covariates', 'xreg'):
                vals = [m.get(key) for m in metas if isinstance(m.get(key), dict)]
                if vals:
                    agg = {}
                    for v in vals:
                        for k, x in v.items():
                            if isinstance(x, (int, float)) and not isinstance(x, bool):
                                agg[k] = agg.get(k, 0) + x
                            elif isinstance(x, list):
                                agg[k] = sorted(set(agg.get(k, [])) | set(map(str, x)))
                            else:
                                agg.setdefault(k, x)
                    meta[key] = agg
            meta['quantile_crossing_share_per_call'] = [m.get('quantile_crossing_share') for m in metas]
            meta['context_lengths'] = [min(m['context_lengths'][0] for m in metas), max(m['context_lengths'][1] for m in metas)]
            meta['n'] = int(sum(m.get('n', 0) for m in metas))
        return meta

    def run_job(self, job):
        torch = self.zoo.torch
        data = self.inputs(job)
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        t0 = time.time()
        median, q, ameta = self.predict(job, data)
        wall = time.time() - t0
        family = job['arm'].rsplit('_', 1)[0]
        repo = self.zm.REPOS[family]
        w = self.weights.get(repo, {})
        if ameta.get('revision') != w.get('revision'):
            raise RuntimeError(f'{repo}: ran {ameta.get("revision")}, weights.json records {w.get("revision")}')
        im = data['meta']
        base = ['config/bay_reserved_protocol_v2_addendum81_amendment1.json',
                'config/bay_reserved_protocol_v2_addendum81_amendment2.json']
        meta = {'declared_in': 'config/bay_reserved_protocol_v2_addendum81.json',
                'amendments': base + [x for x in self.amendments if x not in base],
                'input': job['file'], 'input_sha256': job['sha256'], 'input_manifest': job['manifest'],
                'task': im['task'], 'variable': im['variable'], 'cohort': im['cohort'], 'units': im.get('units'),
                'standardisation': im.get('standardisation'), 'horizon': int(im['horizon']),
                'rows': int(data['n_run']), 'rows_in_input': int(im['n']), 'limit': self.limit,
                'key_columns': im.get('key_columns'), 'reverse_rows': int(np.asarray(data['reverse'][:data['n_run']]).sum()),
                'time_order': ('the model\'s own: backward contexts are reversed in time, so are these predictions'
                               if np.asarray(data['reverse'][:data['n_run']]).any() else 'forward in time'),
                'arm': job['arm'], 'model_id': repo, 'model_revision': ameta.get('revision'),
                'weights_json_revision': w.get('revision'), 'levels': list(LEVELS),
                'adapter': slim(ameta), 'versions': ameta.get('versions'),
                'wall_seconds': round(wall, 3), 'peak_gib': (round(torch.cuda.max_memory_allocated() / 2 ** 30, 3)
                                                             if torch.cuda.is_available() else None),
                'nan_median': int(np.isnan(median).sum()), 'nan_q': int(np.isnan(q).sum()),
                'worker': self.worker, 'host': socket.gethostname(),
                'device': (torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu'),
                'cuda_visible_devices': os.environ.get('CUDA_VISIBLE_DEVICES'),
                'threads': {k: os.environ.get(k) for k in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS',
                                                          'OPENBLAS_NUM_THREADS', 'V2_THREADS')},
                'torch_threads': torch.get_num_threads(), 'code_sha256': self.code, 'created_utc': now()}
        if self.amendment_sha256:
            meta['amendment_sha256'] = self.amendment_sha256
        meta['overrides'] = self.overrides(job['arm'])
        meta['command'] = self.command
        meta['call_batch_size'] = int(ameta.get('batch', ameta.get('per_core_batch_size')) or 0) or None
        if family == 'timesfm3':
            meta['library_options'] = ameta.get('flags')
        elif family == 'timesfm25':
            meta['library_options'] = ameta.get('forecast_config')
        if self.tfm25_no_cqh and family == 'timesfm25':
            meta['deviation_from_declaration'] = {
                'what': 'TimesFM 2.5 with use_continuous_quantile_head=False (amendment 2 declares True)',
                'why': 'the library refuses the continuous quantile head for horizons > 1024',
                'status': 'not declared; for a decision by amendment'}
        tmp = self.out / 'tmp' / f'{job["stem"]}__{job["arm"]}.{os.getpid()}.npz'
        zio.save_predictions(tmp, median=median, q=q, meta=meta)
        os.replace(tmp, self.output(job))
        return meta

    # ------------------------------------------------------------------------------------ loop
    def loop(self, jobs):
        self.save_status()
        self.log(f'{len(jobs)} jobs in the plan; out {self.out}; limit {self.limit}')
        for job in jobs:
            tag = f'{job["stem"]}__{job["arm"]}'
            if (self.failed_dir / f'{tag}.json').exists():
                continue
            if self.is_done(job):
                self.status['skipped_existing'] += 1
                continue
            other = self.other_options(job)
            if other is not None:
                self.log(f'skip  {tag}: an output with other options exists ({json.dumps(other)}); '
                         'supersede it first')
                self.status.setdefault('kept_other_options', []).append(tag)
                continue
            claim = self.claim(job)
            if claim is None:
                continue
            try:
                if self.is_done(job):
                    self.status['skipped_existing'] += 1
                    continue
                self.status['current'] = {'job': tag, 'rows': job['n'] if self.limit is None else min(job['n'], self.limit),
                                          'horizon': job['horizon'], 'since': now()}
                self.save_status()
                self.log(f'start {tag} rows {self.status["current"]["rows"]} horizon {job["horizon"]}')
                try:
                    meta = self.run_job(job)
                except Exception as e:
                    tb = traceback.format_exc()
                    self.log(f'FAILED {tag}: {type(e).__name__}: {e}\n{tb}')
                    rec = {'job': tag, 'error': f'{type(e).__name__}: {e}', 'traceback': tb, 'at': now(),
                           'worker': self.worker}
                    (self.failed_dir / f'{tag}.json').write_text(json.dumps(rec, indent=1))
                    self.status['failed'].append({'job': tag, 'error': rec['error']})
                    if self._zoo is not None and self._zoo.torch.cuda.is_available():
                        self._zoo.torch.cuda.empty_cache()
                    continue
                done = {'job': tag, 'rows': meta['rows'], 'seconds': meta['wall_seconds'],
                        'peak_gib': meta['peak_gib'], 'nan_median': meta['nan_median'], 'at': now()}
                self.status['done'].append(done)
                self.log(f'done  {tag} rows {meta["rows"]} seconds {meta["wall_seconds"]} '
                         f'({meta["rows"] / max(meta["wall_seconds"], 1e-9):.1f}/s) peak {meta["peak_gib"]} GiB '
                         f'nan median {meta["nan_median"]}')
            finally:
                claim.unlink(missing_ok=True)
                self.status['current'] = None
                self.save_status()
        self.status['finished'] = True
        self.save_status()
        self.log(f'finished: {len(self.status["done"])} written, {self.status["skipped_existing"]} existing, '
                 f'{len(self.status["failed"])} failed')


# --------------------------------------------------------------------------------------------------
# determinism and inspection
# --------------------------------------------------------------------------------------------------
def _compare(med, q, sm, sq):
    both = np.isfinite(sq) & np.isfinite(q)
    rec = {'max_abs_diff_median': float(np.nanmax(np.abs(med - sm))) if np.isfinite(med - sm).any() else 0.0,
           'max_abs_diff_q': float(np.abs(q - sq)[both].max()) if both.any() else 0.0,
           'identical': bool(np.array_equal(q, sq, equal_nan=True) and np.array_equal(med, sm, equal_nan=True)),
           'nan_pattern_differs': int((np.isnan(q) != np.isnan(sq)).sum() + (np.isnan(med) != np.isnan(sm)).sum())}
    rec['below_1e-6'] = bool(max(rec['max_abs_diff_median'], rec['max_abs_diff_q']) < 1e-6
                             and rec['nan_pattern_differs'] == 0)
    return rec


def rerun(out, n_rows, files=None, arms=None, report_name='rerun_check.json', require_amendment=None):
    """Each arm of one file per task run again and compared with the stored prediction (the spirit of
    gate G3), with the call options the stored output's meta records (its `overrides`), on two row sets:
      first_rows     the first n_rows contexts of the file, called alone;
      first_batch    the full run's first call group cut to its first max(n_rows, batch) rows, which
                     reproduces the composition of the full run's first batch (a Chronos-Bolt batch of 64
                     against one of 256 moves the A medians by up to 1.2e-6 in the model checks), except
                     where a covariate arm splits a batch by the covariates left after dropping dark ones.
    With `require_amendment`, only stored outputs whose meta cites that amendment are rerun."""
    r = Runner('rerun', out)
    import v2_fm_zoo_models as zm
    report = {'declared_in': 'config/bay_reserved_protocol_v2_addendum81.json (G3 spirit)', 'rows': n_rows,
              'files': {}, 'started': now(), 'command': ' '.join(sys.argv[1:])}
    if require_amendment:
        report['outputs_of'] = require_amendment
    path = Path(out) / report_name
    jobs = plan(files=set(files or RERUN_FILES), arms=arms)
    for job in jobs:
        stored_path = r.output(job)
        tag = f'{job["stem"]}__{job["arm"]}'
        if not stored_path.exists():
            report['files'][tag] = {'error': 'no stored prediction'}
            continue
        stored = zio.load_predictions(stored_path)
        if require_amendment and require_amendment not in stored['meta'].get('amendments', []):
            continue
        r.set_overrides(stored['meta'].get('overrides'))
        data = r.inputs(job)
        family = job['arm'].rsplit('_', 1)[0]
        batch = (r.batch or OPTIONS[job['task']]['bolt'] if family.startswith('chronos_bolt') else
                 r.batch or OPTIONS[job['task']]['chronos2'] if family == 'chronos2' else r.batch or zm.BATCH[family])
        g0 = groups(job['task'], data['keys'], data['n_run'])[0]
        sets = {'first_rows': np.arange(min(n_rows, data['n_run'])),
                'first_batch': g0[:max(n_rows, batch)]}
        rec = {}
        for name, rows in sets.items():
            t0 = time.time()
            med, q, _ = r.predict(job, data, rows)
            c = _compare(med, q, stored['median'][rows], stored['q'][rows])
            c.update(rows=int(len(rows)), seconds=round(time.time() - t0, 2))
            rec[name] = c
        rec['batch'] = batch
        rec['overrides'] = r.overrides(job['arm'])
        rec['stored_created_utc'] = stored['meta'].get('created_utc')
        report['files'][tag] = rec
        r.log(f'rerun {tag} {json.dumps(rec)}')
        path.write_bytes((json.dumps(report, indent=1) + '\n').encode('utf-8'))
    report['all_below_1e-6'] = {s: all(v[s]['below_1e-6'] for v in report['files'].values() if s in v)
                                for s in ('first_rows', 'first_batch')}
    report['all_identical'] = {s: all(v[s]['identical'] for v in report['files'].values() if s in v)
                               for s in ('first_rows', 'first_batch')}
    report['max_abs_diff'] = {s: max([max(v[s]['max_abs_diff_median'], v[s]['max_abs_diff_q'])
                                      for v in report['files'].values() if s in v] or [None])
                              for s in ('first_rows', 'first_batch')}
    report['outputs_checked'] = sum(1 for v in report['files'].values() if 'first_rows' in v)
    report['finished'] = now()
    path.write_bytes((json.dumps(report, indent=1) + '\n').encode('utf-8'))
    r.log(f'rerun check written: all below 1e-6 = {report["all_below_1e-6"]}')


def batch1check(out, n_rows):
    """Diagnostic, not a gate: how far each batched arm's stored prediction (the declared batches) moves
    when the same first n_rows contexts are forecast one series per call. TimesFM 3.0 pads every batch to
    its longest context and Chronos pads to the batch length, so a forecast can depend on the lengths of
    the other series in its batch. Writes <out>/batch1_check.json."""
    r = Runner('batch1', out)
    r.batch_override = 1
    report = {'what': 'stored prediction (declared batches) minus a one-series-per-call run, first rows',
              'rows': n_rows, 'files': {}, 'started': now()}
    path = Path(out) / 'batch1_check.json'
    for job in plan(files=set(RERUN_FILES)):
        if job['arm'].startswith('timesfm25'):
            continue                                          # already one series per decode
        stored_path = r.output(job)
        tag = f'{job["stem"]}__{job["arm"]}'
        if not stored_path.exists():
            continue
        stored = zio.load_predictions(stored_path)
        data = r.inputs(job)
        rows = np.arange(min(n_rows, data['n_run']))
        t0 = time.time()
        med, q, _ = r.predict(job, data, rows)
        c = _compare(med, q, stored['median'][rows], stored['q'][rows])
        sm = stored['median'][rows]
        c.update(rows=int(len(rows)), seconds=round(time.time() - t0, 2),
                 mean_abs_diff_median=float(np.nanmean(np.abs(med - sm))),
                 rows_moved_over_1e_3=int((np.nanmax(np.abs(med - sm), axis=1) > 1e-3).sum()),
                 units=data['meta'].get('units'), standardisation=data['meta'].get('standardisation'))
        report['files'][tag] = c
        r.log(f'batch1 {tag} {json.dumps(c, default=str)}')
        path.write_bytes((json.dumps(report, indent=1, default=str) + '\n').encode('utf-8'))
    report['finished'] = now()
    path.write_bytes((json.dumps(report, indent=1, default=str) + '\n').encode('utf-8'))


def inspect(out):
    rows = []
    for p in sorted(Path(out).glob('*.npz')):
        z = zio.load_predictions(p)
        m = z['meta']
        a = m.get('adapter', {})
        rows.append({'file': p.name, 'median': list(z['median'].shape), 'q': list(z['q'].shape),
                     'nan_median': int(np.isnan(z['median']).sum()), 'nan_q': int(np.isnan(z['q']).sum()),
                     'non_monotone_q_share': float((np.diff(z['q'], axis=1) < -1e-6).any(axis=1).mean()) if z['q'].size else 0.0,
                     'median_minus_q50_max': float(np.nanmax(np.abs(z['median'] - z['q'][:, 4]))) if z['q'].size else 0.0,
                     'horizon': m.get('horizon'), 'rows': m.get('rows'), 'seconds': m.get('wall_seconds'),
                     'peak_gib': m.get('peak_gib'), 'revision': (m.get('model_revision') or '')[:8],
                     'level_method': a.get('level_method'), 'covariates': a.get('covariates'),
                     'xreg_future': (a.get('xreg') or {}).get('future'), 'calls': a.get('calls'),
                     'nan_record': a.get('nan_record'), 'clamp': a.get('positivity_clamp'),
                     'batch': a.get('batch', a.get('per_core_batch_size')), 'fallback': a.get('batch_fallback'),
                     'd_leads': (m.get('adapter', {}).get('task_d_no_lookahead') or {}).get('leads')})
    print(json.dumps(rows, indent=1, default=str))


# --------------------------------------------------------------------------------------------------
# amendments: superseded outputs kept, declared outputs adopted, effects recorded without observations
# --------------------------------------------------------------------------------------------------
def _write_json(path, obj):
    path.write_bytes((json.dumps(obj, indent=1, default=str) + '\n').encode('utf-8'))


def supersede(out, to, tasks, files, arms, amendment, reason):
    """Move the outputs an amendment supersedes out of the declared folder `out` into `to`, where they are
    kept. All or nothing: every (input, arm) of the plan must have its output in `out`, written before the
    amendment (its meta does not cite it), and none may exist in `to`. Records <to>/superseded.json."""
    out, to = Path(out), Path(to)
    if not arms or not amendment or not reason:
        raise SystemExit('supersede needs --arms, --amendment and --reason')
    moves = []
    for job in plan(tasks, files, arms):
        name = f'{job["stem"]}__{job["arm"]}.npz'
        src, dst = out / name, to / name
        if not src.exists():
            raise SystemExit(f'{name}: not in {out}; nothing moved')
        meta = zio.load_predictions(src)['meta']
        if amendment in meta.get('amendments', []):
            raise SystemExit(f'{name}: already an output of {amendment}; nothing moved')
        if meta.get('input_sha256') != job['sha256'] or meta.get('arm') != job['arm']:
            raise SystemExit(f'{name}: meta does not match its input or arm; nothing moved')
        if dst.exists():
            raise SystemExit(f'{name}: already in {to}; nothing moved')
        moves.append((name, src, dst, meta))
    to.mkdir(parents=True, exist_ok=True)
    rec_path = to / 'superseded.json'
    rec = json.loads(rec_path.read_text(encoding='utf-8')) if rec_path.exists() else {'files': {}}
    rec.update({'superseded_by': amendment, 'reason': f'{amendment}, changes.{reason}', 'from_folder': out.name,
                'moved_utc': now(), 'host': socket.gethostname(), 'command': ' '.join(sys.argv[1:]),
                'code_sha256': zio.digest(Path(__file__))})
    for name, src, dst, meta in moves:
        a = meta.get('adapter') or {}
        rec['files'][name] = {'sha256': zio.digest(src), 'input': meta.get('input'),
                              'input_sha256': meta.get('input_sha256'), 'arm': meta.get('arm'),
                              'created_utc': meta.get('created_utc'), 'code_sha256': meta.get('code_sha256'),
                              'per_core_batch_size': a.get('per_core_batch_size'),
                              'library_options': a.get('flags', a.get('forecast_config'))}
        os.rename(src, dst)
    rec['count'] = len(rec['files'])
    _write_json(rec_path, rec)
    print(f'moved {len(moves)} outputs from {out} to {to} ({rec["count"]} recorded there)')


def adopt(src, out, amendment, reason, amendment_sha256=None):
    """Copy the outputs of `src` that an amendment declares into the declared folder `out`, arrays
    unchanged, meta citing the amendment in place of the deviation it records. The files in `src` are not
    touched. Records <out>/adopted_<src>.json."""
    src, out = Path(src), Path(out)
    if not amendment or not reason:
        raise SystemExit('adopt needs --amendment and --reason')
    cat = catalogue()
    code = zio.digest(Path(__file__))
    todo = []
    for p in sorted(src.glob('*.npz')):
        z = zio.load_predictions(p)
        m = z['meta']
        if m.get('input') not in cat:
            raise SystemExit(f'{p.name}: input {m.get("input")} is in no manifest; nothing adopted')
        want = cat[m['input']][0]
        if m.get('input_sha256') != want or zio.digest(INPUTS / m['input']) != want:
            raise SystemExit(f'{p.name}: input digest does not match {m["input"]}; nothing adopted')
        if p.name != f'{m["input"][:-4]}__{m["arm"]}.npz':
            raise SystemExit(f'{p.name}: name does not match its meta; nothing adopted')
        if (out / p.name).exists():
            raise SystemExit(f'{p.name}: already in {out}; nothing adopted')
        todo.append((p, z))
    rec = {'adopted_by': amendment, 'reason': f'{amendment}, changes.{reason}', 'source_folder': src.name,
           'into': out.name, 'adopted_utc': now(), 'host': socket.gethostname(), 'code_sha256': code,
           'command': ' '.join(sys.argv[1:]), 'files': {}}
    (out / 'tmp').mkdir(parents=True, exist_ok=True)
    for p, z in todo:
        m = dict(z['meta'])
        was = m.pop('deviation_from_declaration', None)
        m['amendments'] = list(m.get('amendments', [])) + ([amendment] if amendment not in m.get('amendments', []) else [])
        if amendment_sha256:
            m['amendment_sha256'] = dict(m.get('amendment_sha256') or {}, **amendment_sha256)
        a = m.get('adapter') or {}
        cfg = (a.get('call_options') or {}).get('tfm25_config')
        m['overrides'] = {'tfm25_config': dict(cfg)} if cfg else {}
        m['call_batch_size'] = a.get('per_core_batch_size', a.get('batch'))
        m['library_options'] = a.get('forecast_config', a.get('flags'))
        sha = zio.digest(p)
        m['adopted'] = {'by': amendment, 'rule': f'changes.{reason}', 'source': f'{src.name}/{p.name}',
                        'source_sha256': sha, 'arrays': 'median and q as computed, unchanged',
                        'adopted_utc': rec['adopted_utc'], 'adopted_by_code_sha256': code}
        tmp = out / 'tmp' / f'{p.stem}.adopt.{os.getpid()}.npz'
        zio.save_predictions(tmp, median=z['median'], q=z['q'], meta=m)
        back = zio.load_predictions(tmp)
        if not (np.array_equal(back['median'], z['median'], equal_nan=True)
                and np.array_equal(back['q'], z['q'], equal_nan=True)):
            raise SystemExit(f'{p.name}: arrays changed on rewrite')
        os.replace(tmp, out / p.name)
        rec['files'][p.name] = {'source_sha256': sha, 'sha256': zio.digest(out / p.name),
                                'deviation_record_replaced': bool(was)}
    rec['count'] = len(rec['files'])
    _write_json(out / f'adopted_{src.name}.json', rec)
    print(f'adopted {len(todo)} outputs from {src} into {out}')


def clamp_effect(out, superseded):
    """Task A, TimesFM 2.5: the declared outputs (positivity clamp off) against the superseded ones (clamp
    on, the published configuration), output against output only: per file the rows whose median at the
    six-hour lead (and at any lead, and any quantile) differs and the largest difference in physical units
    (standardised difference x the input's scale). No observation is read. Writes <out>/clamp_effect_A.json."""
    out, sup = Path(out), Path(superseded)
    rep = {'what': ('task A, TimesFM 2.5: declared outputs (infer_is_positive False, amendment 3) minus the '
                    'superseded ones (infer_is_positive True); output against output, no observation read'),
           'declared_folder': out.name, 'clamped_folder': sup.name, 'created_utc': now(),
           'code_sha256': zio.digest(Path(__file__)), 'files': {}, 'by_variable': {}}
    for job in plan(tasks={'A'}, arms={'timesfm25_uni', 'timesfm25_cov'}):
        name = f'{job["stem"]}__{job["arm"]}.npz'
        d, c = zio.load_predictions(out / name), zio.load_predictions(sup / name)
        md, mc = d['meta'], c['meta']
        if not (md['input_sha256'] == mc['input_sha256'] == job['sha256']):
            raise SystemExit(f'{name}: the two outputs are not of the same input')
        fd = (md['adapter'].get('forecast_config') or {}).get('infer_is_positive')
        fc = (mc['adapter'].get('forecast_config') or {}).get('infer_is_positive')
        if fd is not False or fc is not True:
            raise SystemExit(f'{name}: infer_is_positive is {fd} (declared) and {fc} (superseded)')
        st = md['standardisation']
        scale, centre = float(st['scale']), float(st['center'])
        H = int(md['horizon'])
        dm, cm, dq, cq = d['median'], c['median'], d['q'], c['q']
        diff = dm != cm
        lead = diff[:, H - 1]
        qdiff = (dq != cq).any(axis=(1, 2))
        rep['files'][name] = {
            'arm': job['arm'], 'variable': md['variable'], 'offset': int(job['stem'].rsplit('offset', 1)[1]),
            'rows': int(len(dm)), 'units_physical': ('degC' if md['variable'] == 'temperature' else 'umol/kg'),
            'scale': scale, 'centre': centre, 'lead_h': H,
            'rows_median_differs_at_lead': int(lead.sum()),
            'max_abs_diff_median_at_lead_physical': float(np.abs(dm[:, H - 1] - cm[:, H - 1]).max() * scale),
            'rows_median_differs_any_lead': int(diff.any(axis=1).sum()),
            'max_abs_diff_median_any_lead_physical': float(np.abs(dm - cm).max() * scale),
            'rows_any_quantile_differs': int(qdiff.sum()),
            'max_abs_diff_q_physical': float(np.abs(dq - cq).max() * scale),
            'clamped_median_is_zero_wherever_they_differ': bool((cm[diff] == 0).all()) if diff.any() else None,
            'clamped_value_physical': centre,
            'superseded_positivity_clamp_record': mc['adapter'].get('positivity_clamp'),
            'declared_positivity_clamp_record': md['adapter'].get('positivity_clamp')}
    for (var, arm) in sorted({(v['variable'], v['arm']) for v in rep['files'].values()}):
        fs = [v for v in rep['files'].values() if v['variable'] == var and v['arm'] == arm]
        rep['by_variable'][f'{var}__{arm}'] = {
            'rows': sum(v['rows'] for v in fs),
            'rows_median_differs_at_lead': sum(v['rows_median_differs_at_lead'] for v in fs),
            'max_abs_diff_median_at_lead_physical': max(v['max_abs_diff_median_at_lead_physical'] for v in fs),
            'rows_median_differs_any_lead': sum(v['rows_median_differs_any_lead'] for v in fs),
            'max_abs_diff_median_any_lead_physical': max(v['max_abs_diff_median_any_lead_physical'] for v in fs),
            'rows_any_quantile_differs': sum(v['rows_any_quantile_differs'] for v in fs),
            'max_abs_diff_q_physical': max(v['max_abs_diff_q_physical'] for v in fs),
            'by_offset': {v['offset']: [v['rows_median_differs_at_lead'], round(v['max_abs_diff_median_at_lead_physical'], 6)]
                          for v in fs}}
    _write_json(out / 'clamp_effect_A.json', rep)
    print(json.dumps(rep['by_variable'], indent=1))


def amendment3_applies(job):
    """The outputs amendment 3 declares anew: every TimesFM 3.0 output (one series per call); TimesFM 2.5
    in tasks A and F (clamp off) and at the 1440 h gap files (continuous quantile head off)."""
    family = job['arm'].rsplit('_', 1)[0]
    return family == 'timesfm3' or (family == 'timesfm25' and (job['task'] in 'AF' or job['horizon'] > 1024))


def verify_declared(out, amendment):
    """Every (input, arm) of the plan has exactly one output in `out`, nothing else is there, each output's
    meta names its input with the input file's digest (computed here) and the manifest's, its arm, all
    rows, and cites `amendment` exactly where the amendment declares it. Writes <out>/declared_check_a3.json."""
    out = Path(out)
    jobs = plan()
    expected = {f'{j["stem"]}__{j["arm"]}.npz': j for j in jobs}
    have = {p.name for p in out.glob('*.npz')}
    digests = {}
    problems = []
    cites = 0
    for name, j in sorted(expected.items()):
        if name not in have:
            continue
        if j['file'] not in digests:
            digests[j['file']] = zio.digest(INPUTS / j['file'])
        with np.load(out / name, allow_pickle=False) as z:
            m = json.loads(str(z['meta']))
            shape = z['median'].shape
        why = []
        if m.get('input') != j['file']:
            why.append('input name')
        if not (m.get('input_sha256') == j['sha256'] == digests[j['file']]):
            why.append('input digest')
        if m.get('arm') != j['arm']:
            why.append('arm')
        if m.get('limit') is not None or m.get('rows') != j['n'] or shape != (j['n'], j['horizon']):
            why.append(f'rows or shape {shape}')
        c = amendment in m.get('amendments', [])
        cites += c
        if c != amendment3_applies(j):
            why.append(f'cites amendment: {c}, expected {amendment3_applies(j)}')
        if 'deviation_from_declaration' in m:
            why.append('records a deviation')
        if why:
            problems.append({'file': name, 'problems': why})
    rep = {'folder': out.name, 'checked_utc': now(), 'code_sha256': zio.digest(Path(__file__)),
           'expected': len(expected), 'present': len(have & set(expected)),
           'missing': sorted(set(expected) - have), 'extra': sorted(have - set(expected)),
           'inputs_hashed': len(digests), 'citing_amendment': cites, 'amendment': amendment,
           'expected_citing': sum(amendment3_applies(j) for j in jobs), 'problems': problems}
    rep['ok'] = not (rep['missing'] or rep['extra'] or problems) and rep['present'] == rep['expected']
    _write_json(out / 'declared_check_a3.json', rep)
    print(json.dumps({k: v for k, v in rep.items() if k != 'problems'}, indent=1), 'problems:', len(problems))
    return rep


def status():
    """One line per worker from status/<worker>.json, and the plan's progress."""
    for p in sorted((HERE / 'status').glob('*.json')):
        s = json.loads(p.read_text(encoding='utf-8'))
        secs = sum(d['seconds'] for d in s['done'])
        print(f'{s["worker"]:6s} pid {s["pid"]} written {len(s["done"])} ({secs / 60:.1f} min) existing '
              f'{s["skipped_existing"]} failed {len(s["failed"])} finished {s["finished"]} updated {s["updated"]} '
              f'current {s["current"]}')
        for f in s['failed']:
            print('   FAILED', f['job'], f['error'][:300])
    jobs = plan()
    out = HERE / 'preds'
    have = {p.name for p in out.glob('*.npz')} if out.exists() else set()
    by = {}
    for j in jobs:
        t = by.setdefault(j['task'], [0, 0])
        t[0] += 1
        t[1] += f'{j["stem"]}__{j["arm"]}.npz' in have
    print('preds by task (written / planned):', {k: f'{v[1]}/{v[0]}' for k, v in sorted(by.items())})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('action', choices=('plan', 'run', 'rerun', 'inspect', 'status', 'batch1check', 'supersede',
                                       'adopt', 'clamp-effect', 'verify-declared'))
    ap.add_argument('--worker', default='w0')
    ap.add_argument('--out', default=str(HERE / 'preds'))
    ap.add_argument('--limit', type=int, default=None)
    ap.add_argument('--tasks', default=None, help='letters, e.g. ABF')
    ap.add_argument('--files', default=None, help='comma-separated input names or patterns (A_*), or "smoke"')
    ap.add_argument('--arms', default=None)
    ap.add_argument('--n', type=int, default=64)
    ap.add_argument('--tfm25-no-continuous-head', action='store_true',
                    help='NOT declared: TimesFM 2.5 without the continuous quantile head (horizons > 1024)')
    ap.add_argument('--batch', type=int, default=None, help='batch size of every arm in the plan')
    ap.add_argument('--tfm25-config', default=None, help='key=value,... over TFM25_CONFIG')
    ap.add_argument('--tfm3-flags', default=None, help='key=value,... over TFM3_FLAGS')
    ap.add_argument('--amendment', action='append', default=[], help='an amendment the outputs cite')
    ap.add_argument('--amendment-sha256', default=None, help='its sha256 (one --amendment)')
    ap.add_argument('--to', default=None, help='supersede: the folder the superseded outputs move to')
    ap.add_argument('--src', default=None, help='adopt: the folder the adopted outputs come from')
    ap.add_argument('--reason', default=None, help='supersede, adopt: the key of the amendment\'s change')
    ap.add_argument('--superseded', default=None, help='clamp-effect: the folder of the clamped outputs')
    ap.add_argument('--report', default='rerun_check.json', help='rerun: the report file name in --out')
    ap.add_argument('--require-amendment', default=None, help='rerun: only outputs citing this amendment')
    a = ap.parse_args()

    def where(p):
        return Path(p) if os.path.isabs(p) else HERE / p
    out = where(a.out)
    files = None
    if a.files:
        files = set(SMOKE_FILES) if a.files == 'smoke' else set(a.files.split(','))
    arms = set(a.arms.split(',')) if a.arms else None
    tasks = set(a.tasks) if a.tasks else None
    tfm25_config = parse_options(a.tfm25_config, TFM25_KEYS, '--tfm25-config')
    tfm3_flags = parse_options(a.tfm3_flags, TFM3_KEYS, '--tfm3-flags')
    if a.amendment_sha256 and len(a.amendment) != 1:
        raise SystemExit('--amendment-sha256 goes with exactly one --amendment')
    asha = {a.amendment[0]: a.amendment_sha256} if a.amendment_sha256 else None
    if a.action == 'plan':
        jobs = plan(tasks, files, arms)
        for j in jobs:
            print(f'{j["task"]} {j["file"]:45s} {j["arm"]:24s} n={j["n"]:5d} H={j["horizon"]:4d} cost={j["cost"]:9.1f}')
        print(len(jobs), 'jobs')
    elif a.action == 'run':
        jobs = plan(tasks, files, arms)
        if files:
            unmatched = sorted(p for p in files if not any(matches(j['file'], {p}) for j in jobs))
            if unmatched:
                raise SystemExit(f'files not in the manifests (or not in the plan): {unmatched}')
        Runner(a.worker, out, a.limit, a.tfm25_no_continuous_head, batch=a.batch, tfm25_config=tfm25_config,
               tfm3_flags=tfm3_flags, amendments=a.amendment, amendment_sha256=asha).loop(jobs)
    elif a.action == 'rerun':
        rerun(out, a.n, files, arms, a.report, a.require_amendment)
    elif a.action == 'supersede':
        supersede(out, where(a.to), tasks, files, arms, (a.amendment or [None])[0], a.reason)
    elif a.action == 'adopt':
        adopt(where(a.src), out, (a.amendment or [None])[0], a.reason, asha)
    elif a.action == 'clamp-effect':
        clamp_effect(out, where(a.superseded))
    elif a.action == 'verify-declared':
        verify_declared(out, (a.amendment or [None])[0])
    elif a.action == 'status':
        status()
    elif a.action == 'batch1check':
        batch1check(out, a.n)
    else:
        inspect(out)


if __name__ == '__main__':
    main()
