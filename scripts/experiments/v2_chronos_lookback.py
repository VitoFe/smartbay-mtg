"""Addendum 46 on Trinity: zero-shot Chronos-Bolt as a continuation, at five context lengths.

Reads only the arrays scripts/v2_lookback_inputs.py built locally (inputs/), one file per variable
and context length, and runs the pretrained model on every continuation context of both cohorts
exactly as addendum 41's zero-shot job did: each context passed whole (missing hours as missing),
in batches of 64, median of the 48 h forecast. Nothing is trained. The 512 h traces are gate G2.

  python v2_chronos_lookback.py            every length, both variables, into results/
  python v2_chronos_lookback.py --quick    a few contexts per file, into results_smoke/
Declared in config/bay_reserved_protocol_v2_addendum46.json.
"""
import json, os, sys, time, warnings
from pathlib import Path
import numpy as np

warnings.filterwarnings('ignore')
HERE = Path(__file__).resolve().parent
os.environ.setdefault('HF_HOME', '/home/vito/smartbay_do/.hf')
import torch
torch.set_num_threads(int(os.environ.get('V2_THREADS', '3')))
from chronos import BaseChronosPipeline

REPO = 'amazon/chronos-bolt-base'
HORIZON = 48
LENGTHS = (48, 128, 256, 512, 1024)
SEED = 20260924                          # addendum 41's zero-shot job seeded torch the same way
QUICK = '--quick' in sys.argv
INP = HERE / 'inputs'
OUT = HERE / ('results_smoke' if QUICK else 'results'); OUT.mkdir(exist_ok=True)


def forecast(pipe, contexts, length):
    """Median trajectories, in batches of 64, as addendum 41's zero-shot job ran them."""
    out = []
    for k in range(0, len(contexts), 64):
        q, _ = pipe.predict_quantiles(inputs=contexts[k:k + 64], prediction_length=length, quantile_levels=[0.1, 0.5, 0.9])
        out.append(q[:, :, 1].float().cpu().numpy())
    return np.concatenate(out) if out else np.zeros((0, length), np.float32)


def main():
    t0 = time.time()
    torch.manual_seed(SEED)
    pipe = BaseChronosPipeline.from_pretrained(REPO, device_map='cuda', torch_dtype=torch.float32)
    pipe.model.eval()
    status = {}
    for variable in ('temperature', 'oxygen'):
        for length in LENGTHS:
            data = np.load(INP / f'continuation_{variable}_{length}.npz', allow_pickle=True)
            rows = np.arange(len(data['contexts']))
            if QUICK:
                rows = rows[:8]
            contexts = [torch.tensor(data['contexts'][i]) for i in rows]
            traces = forecast(pipe, contexts, HORIZON)
            np.savez_compressed(OUT / f'continuation_{variable}_{length}.npz', rows=rows, traces=traces)
            status[f'{variable}_{length}'] = int(len(rows))
            print('wrote', variable, length, len(rows), flush=True)
    status['seconds'] = round(time.time() - t0)
    (OUT / 'lookback.status.json').write_text(json.dumps(status))


if __name__ == '__main__':
    main()
