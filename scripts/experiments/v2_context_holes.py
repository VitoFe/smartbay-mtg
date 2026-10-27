"""Addendum 53 on the compute host: Chronos-Bolt forward forecasts under three treatments of the
holes in its context, the holes bridged by Chronos-2 run from both edges.

Reads inputs/*.npz built locally by scripts/v2_context_holes_inputs.py (numbers only) and writes
results/<task>.npz with one prediction per context and treatment:

  linear  the published context (the gates reproduce the published columns from it)
  mask    the raw context, holes as NaN, which Chronos-Bolt masks
  bridge  every hole the inputs name filled by zero-shot Chronos-2, forward from the 512 h before
          the hole and backward from the record after it, blended with chronos_fill's weights
          w = (L - k) / (L + 1); what is left missing is then treated as the linear treatment treats
          it (interpolated, and for A1 left-padded with the first value)

A tasks forecast 6 h and keep the median at the sixth hour; B tasks forecast the gap's own length
and keep the median trace. Chronos-Bolt runs as the published code ran it (predict_quantiles,
quantiles 0.1/0.5/0.9, batches of 256 for A and 64 for B); Chronos-2 as chronos_fill runs it
(pipe.predict, the median of pipe.quantiles).

  python v2_context_holes.py [--quick]     # --quick: the first 40 contexts of each task, results_smoke/

Declared in config/bay_reserved_protocol_v2_addendum53.json.
"""
import json
import os
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')
HERE = Path(__file__).resolve().parent
os.environ.setdefault('HF_HOME', '/home/vito/smartbay_do/.hf')
import torch  # noqa: E402

torch.set_num_threads(int(os.environ.get('V2_THREADS', '3')))
QUICK = '--quick' in sys.argv
INPUTS = HERE / 'inputs'
OUT = HERE / ('results_smoke' if QUICK else 'results')
OUT.mkdir(exist_ok=True)
BOLT, CHRONOS2 = 'amazon/chronos-bolt-base', 'amazon/chronos-2'
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
QUICK_N = 40
MAX_GAP = 72


def log(msg):
    print(f'{time.strftime("%H:%M:%S")} {msg}', flush=True)


def tail(row, length):
    return row[len(row) - length:]


def chronos2_traces(pipe, contexts, lengths, needed):
    """Median traces of Chronos-2 for the needed rows of a context array, grouped by length."""
    median = int(np.argmin(np.abs(np.asarray(pipe.quantiles) - 0.5)))
    out = {}
    horizon = {}
    for i in needed:
        horizon.setdefault(int(lengths[i][1]), []).append(i)
    for L, rows in sorted(horizon.items()):
        size = 256 if L <= 64 else 64 if L <= 256 else 16
        for k in range(0, len(rows), size):
            chunk = rows[k:k + size]
            items = [{'target': tail(contexts[i], int(lengths[i][0])).astype(np.float32)} for i in chunk]
            got = pipe.predict(items, prediction_length=L)
            for i, o in zip(chunk, got):
                a = o.detach().float().cpu().numpy() if torch.is_tensor(o) else np.asarray(o)
                out[i] = a.reshape(-1, len(pipe.quantiles), a.shape[-1])[0, median]
    return out


def bridged_contexts(d, rows, pipe):
    """The bridge treatment's context for each row in `rows`."""
    keep = np.isin(d['place_n'], rows)
    pn, pb, pk, pp = (d['place_n'][keep], d['place_bridge'][keep], d['place_k'][keep],
                      d['place_pos'][keep])
    bridges = np.unique(pb)
    L = d['bridge_len']
    fwd_len, bwd_len = {}, {}
    for br in bridges:                  # an item always belongs to holes of one length
        fwd_len[int(d['bridge_f'][br])] = int(L[br])
        bwd_len[int(d['bridge_b'][br])] = int(L[br])
    log(f'  chronos-2: {len(fwd_len)} forward, {len(bwd_len)} backward items')
    f_tr = chronos2_traces(pipe, d['fwd_ctx'], {i: (d['fwd_len'][i], fwd_len[i]) for i in fwd_len},
                           list(fwd_len))
    b_tr = chronos2_traces(pipe, d['bwd_ctx'], {i: (d['bwd_len'][i], bwd_len[i]) for i in bwd_len},
                           list(bwd_len))
    contexts = {int(n): d['raw'][n].astype(np.float64).copy() for n in rows}
    for n, br, k, p in zip(pn, pb, pk, pp):
        length = int(L[br])
        f = f_tr[int(d['bridge_f'][br])]
        b = b_tr[int(d['bridge_b'][br])][::-1]
        w = (length - k) / (length + 1)
        contexts[int(n)][p] = w * f[k] + (1 - w) * b[k]
    out = {}
    for n in rows:
        row = tail(contexts[int(n)], int(d['length'][n]))
        row = pd.Series(row).interpolate(limit_direction='both').to_numpy(np.float32)
        if bool(d['pad_edge']) and len(row) < len(d['raw'][n]):
            row = np.concatenate([np.full(len(d['raw'][n]) - len(row), row[0], np.float32), row])
        out[int(n)] = row
    return out


def bolt_six_hours(pipe, contexts, horizon=6, batch=256):
    out = []
    for k in range(0, len(contexts), batch):
        q, _ = pipe.predict_quantiles(inputs=[torch.tensor(c) for c in contexts[k:k + batch]],
                                      prediction_length=horizon, quantile_levels=[0.1, 0.5, 0.9])
        out.append(q[:, horizon - 1, 1].float().cpu().numpy())
    return np.concatenate(out) if out else np.zeros(0, np.float32)


def bolt_traces(pipe, contexts, gaps, batch=64):
    out = np.full((len(contexts), MAX_GAP), np.nan, np.float32)
    by = {}
    for i, L in enumerate(gaps):
        by.setdefault(int(L), []).append(i)
    for L, rows in sorted(by.items()):
        for k in range(0, len(rows), batch):
            chunk = rows[k:k + batch]
            q, _ = pipe.predict_quantiles(inputs=[torch.tensor(contexts[i]) for i in chunk],
                                          prediction_length=L, quantile_levels=[0.1, 0.5, 0.9])
            out[chunk, :L] = q[:, :, 1].float().cpu().numpy()
    return out


def main():
    from chronos import BaseChronosPipeline
    started = time.time()
    tasks = sorted(p.stem for p in INPUTS.glob('*.npz'))
    status = {'declared_in': 'config/bay_reserved_protocol_v2_addendum53.json', 'device': DEVICE,
              'quick': QUICK, 'torch': torch.__version__, 'tasks': {}}
    c2 = BaseChronosPipeline.from_pretrained(CHRONOS2, device_map=DEVICE, torch_dtype=torch.float32)
    bridged = {}
    for task in tasks:
        d = dict(np.load(INPUTS / f'{task}.npz'))
        rows = np.arange(min(QUICK_N, len(d['raw'])) if QUICK else len(d['raw']))
        log(f'{task}: {len(rows)} contexts, bridging')
        t0 = time.time()
        bridged[task] = bridged_contexts(d, rows, c2)
        status['tasks'][task] = {'contexts': int(len(rows)), 'bridge_seconds': round(time.time() - t0, 1)}
    del c2
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

    bolt = BaseChronosPipeline.from_pretrained(BOLT, device_map=DEVICE, torch_dtype=torch.float32)
    for task in tasks:
        d = dict(np.load(INPUTS / f'{task}.npz'))
        rows = sorted(bridged[task])
        lengths = d['length'][rows]
        a_task = task.startswith('A')
        linear = [d['linear'][n] if bool(d['pad_edge']) else tail(d['linear'][n], int(d['length'][n]))
                  for n in rows]
        mask = [tail(d['raw'][n], int(d['length'][n])) for n in rows]
        bridge = [bridged[task][n] for n in rows]
        assert all(np.isfinite(c).all() for c in bridge), f'{task}: a bridged context kept a hole'
        t0 = time.time()
        if a_task:
            preds = {name: bolt_six_hours(bolt, ctx, int(d['horizon']))
                     for name, ctx in (('linear', linear), ('mask', mask), ('bridge', bridge))}
        else:
            gaps = d['gap_len'][rows]
            preds = {name: bolt_traces(bolt, ctx, gaps)
                     for name, ctx in (('linear', linear), ('mask', mask), ('bridge', bridge))}
        np.savez_compressed(OUT / f'{task}.npz', rows=np.array(rows),
                            **{f'pred_{k}': v for k, v in preds.items()},
                            bridged_changed=np.array([not np.allclose(
                                np.nan_to_num(b[-int(L):]), np.nan_to_num(tail(lin, int(L))))
                                for b, lin, L in zip(bridge, linear, lengths)]))
        status['tasks'][task]['bolt_seconds'] = round(time.time() - t0, 1)
        log(f'{task}: done')
    status['seconds'] = round(time.time() - started, 1)
    (OUT / 'status.json').write_text(json.dumps(status, indent=2))
    log('all done')


if __name__ == '__main__':
    main()
