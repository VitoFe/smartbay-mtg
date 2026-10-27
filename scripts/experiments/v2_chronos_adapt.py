"""Addendum 41 on Trinity: fine-tuned Chronos-Bolt as a continuation and as a bidirectional gap filler.

Reads only the arrays scripts/v2_chronos_adapt_inputs.py built locally (inputs/), so every scored
context is the one the published comparands saw. Two kinds of job:

  --zero-shot                   the pretrained model on every continuation context (both cohorts)
                                and every Gap-2026 context; the gap traces are gate G2
  --variable V --method M       fine-tune (M = full or lora) with the configuration the six-hour
                                study selected, once per task: forward windows with 48 h targets
                                for the continuation, forward and reversed windows with 64 h
                                targets for gap filling. Predicts Bridge-2026 and Gap-2026 only.

No hyperparameter search is run (Vito's instruction, recorded in the declaration). The adapter
implementation is addendum 40's (v2_chronos_lora.py), repeated here so this snapshot stands alone,
and adapters are merged into the weights before prediction.

  python v2_chronos_adapt.py --zero-shot
  python v2_chronos_adapt.py --variable temperature --method lora
  add --quick for a smoke run (20 steps, a few contexts) into results_smoke/
"""
import json, math, os, sys, time, warnings, zlib
from pathlib import Path
import numpy as np

warnings.filterwarnings('ignore')
HERE = Path(__file__).resolve().parent
os.environ.setdefault('HF_HOME', '/home/vito/smartbay_do/.hf')
import torch
torch.set_num_threads(int(os.environ.get('V2_THREADS', '3')))
from chronos import BaseChronosPipeline

REPO = 'amazon/chronos-bolt-base'
CTX = 512
HORIZON = {'continuation': 48, 'gap': 64}
N_WINDOWS = 24000
SEED = 20260924
QUICK = '--quick' in sys.argv
INP = HERE / 'inputs'
OUT = HERE / ('results_smoke' if QUICK else 'results'); OUT.mkdir(exist_ok=True)
TARGETS = {'qv': ('q', 'v'), 'qkvo': ('q', 'k', 'v', 'o'), 'qkvo_ffn': ('q', 'k', 'v', 'o', 'wi', 'wo')}


class LoRALinear(torch.nn.Module):
    """W x + (alpha / r) B A dropout(x), with W frozen and B = 0 at the start (addendum 40)."""

    def __init__(self, base, rank, alpha, dropout):
        super().__init__()
        self.base, self.scale, self.drop = base, alpha / rank, torch.nn.Dropout(dropout)
        kw = dict(device=base.weight.device, dtype=base.weight.dtype)
        self.lora_a = torch.nn.Parameter(torch.empty(rank, base.in_features, **kw))
        self.lora_b = torch.nn.Parameter(torch.zeros(base.out_features, rank, **kw))
        torch.nn.init.kaiming_uniform_(self.lora_a, a=math.sqrt(5))

    @property
    def weight(self):                    # T5's feed-forward reads wo.weight.dtype to cast its input
        return self.base.weight

    def forward(self, x):
        return self.base(x) + (self.drop(x) @ self.lora_a.T @ self.lora_b.T) * self.scale


def attach(model, targets, rank, alpha, dropout):
    for p in model.parameters():
        p.requires_grad_(False)
    names, n = TARGETS[targets], 0
    for stack in (model.encoder, model.decoder):
        for parent in list(stack.modules()):
            for child_name, child in list(parent.named_children()):
                if child_name in names and isinstance(child, torch.nn.Linear):
                    setattr(parent, child_name, LoRALinear(child, rank, alpha, dropout)); n += 1
    return n


def merge(model):
    """Fold every adapter into its base weight, W + scale B A, and put the plain Linear back."""
    with torch.no_grad():
        for stack in (model.encoder, model.decoder):
            for parent in list(stack.modules()):
                for child_name, child in list(parent.named_children()):
                    if isinstance(child, LoRALinear):
                        child.base.weight += child.scale * (child.lora_b @ child.lora_a)
                        setattr(parent, child_name, child.base)


def windows(variable, task, rng):
    """Training windows from the development panel: forward, and for gap filling also reversed."""
    panel = np.load(INP / f'panel_{variable}.npz', allow_pickle=True)
    values, H = panel['values'].astype(np.float64), HORIZON[task]
    assert int(panel['times'].max()) < 1767225600 * 10 ** 9, 'a training hour reaches 2026'   # gate G3
    per_segment = 2 if task == 'gap' else 1
    ctxs, tgts = [], []
    for j in range(values.shape[1]):
        v = values[:, j]
        ends = np.arange(CTX, len(v) - H)
        if not len(ends):
            continue
        take = min(len(ends), max(1, N_WINDOWS // per_segment // values.shape[1]))
        for e in rng.choice(ends, take, replace=False):
            segment = v[e - CTX:e + H]
            for s in ((segment,) if per_segment == 1 else (segment, segment[::-1])):
                c, t = s[:CTX], s[CTX:]
                if np.isfinite(c).sum() >= 96 and np.isfinite(t).all():
                    ctxs.append(c); tgts.append(t)
    ctxs, tgts = np.asarray(ctxs, np.float32), np.asarray(tgts, np.float32)
    n_val = max(128, int(.1 * len(ctxs))); idx = rng.permutation(len(ctxs))
    return ctxs, tgts, idx[n_val:], idx[:n_val]


def train(method, params, ctxs, tgts, ti, vi, rng):
    pipe = BaseChronosPipeline.from_pretrained(REPO, device_map='cuda', torch_dtype=torch.float32)
    model = pipe.model
    info = {}
    if method == 'lora':
        info['wrapped_layers'] = attach(model, params['targets'], params['rank'],
                                        params['alpha_over_rank'] * params['rank'], params['dropout'])
    trainable = [p for p in model.parameters() if p.requires_grad]
    info['trainable_params'] = int(sum(p.numel() for p in trainable))
    model.train(); opt = torch.optim.AdamW(trainable, lr=params['lr'], weight_decay=params['weight_decay'])

    def loss_on(sel):
        c = torch.tensor(ctxs[sel], device='cuda'); t = torch.tensor(tgts[sel], device='cuda')
        return model(context=torch.nan_to_num(c), mask=torch.isfinite(c), target=t, target_mask=torch.isfinite(t)).loss
    best, best_state, patience, curve = float('inf'), None, 0, []
    steps = 20 if QUICK else params['max_steps']
    for step in range(steps):
        l = loss_on(rng.choice(ti, params['batch'], replace=False)); opt.zero_grad(); l.backward()
        torch.nn.utils.clip_grad_norm_(trainable, 1.0); opt.step()
        if (step + 1) % (10 if QUICK else 100) == 0:
            model.eval()
            with torch.no_grad(): vl = float(np.mean([loss_on(vi[i:i + 64]).item() for i in range(0, len(vi), 64)]))
            model.train(); curve.append([step + 1, vl])
            if not np.isfinite(vl): raise ValueError('non-finite validation loss')
            if vl < best - 1e-4:
                best, patience = vl, 0
                best_state = {n: p.detach().clone() for n, p in model.named_parameters() if p.requires_grad}
            else:
                patience += 1
                if patience >= 4: break
    if best_state is None: raise ValueError('no validation improvement recorded')
    with torch.no_grad():
        for n, p in model.named_parameters():
            if n in best_state: p.copy_(best_state[n])
    if method == 'lora':
        merge(model)
    model.eval()
    info.update(best_val=best, stopped_at=curve[-1][0], curve=curve)
    return pipe, info


def forecast(pipe, contexts, length):
    """Median trajectories, in batches of 64 as chronos_fill runs them."""
    out = []
    for k in range(0, len(contexts), 64):
        q, _ = pipe.predict_quantiles(inputs=contexts[k:k + 64], prediction_length=length, quantile_levels=[0.1, 0.5, 0.9])
        out.append(q[:, :, 1].float().cpu().numpy())
    return np.concatenate(out) if out else np.zeros((0, length), np.float32)


def continuation(pipe, variable, cohorts):
    data = np.load(INP / f'continuation_{variable}.npz', allow_pickle=True)
    keep = np.flatnonzero(np.isin(data['cohort'], cohorts))
    if QUICK: keep = keep[:8]
    contexts = [torch.tensor(data['contexts'][i]) for i in keep]
    return {'rows': keep, 'traces': forecast(pipe, contexts, HORIZON['continuation'])}


def gap(pipe, variable):
    """chronos_fill's loop: grouped by length, forward then backward, in episode order."""
    data = np.load(INP / f'gap_{variable}.npz', allow_pickle=True)
    lengths = data['lengths']; n = len(lengths)
    fwd = [None] * n; bwd = [None] * n
    for length in sorted(set(lengths.tolist())):
        members = np.flatnonzero(lengths == length)
        if QUICK: members = members[:4]
        for direction, store in (('fwd', fwd), ('bwd', bwd)):
            arr, lens = data[direction], data[f'{direction}_len']
            contexts = [torch.tensor(arr[i, CTX - lens[i]:]) for i in members]
            traces = forecast(pipe, contexts, int(length))
            for i, tr in zip(members, traces):
                store[i] = tr if direction == 'fwd' else tr[::-1].copy()
    width = int(lengths.max())
    pad = lambda xs: np.stack([np.concatenate([x, np.full(width - len(x), np.nan, np.float32)]) if x is not None
                               else np.full(width, np.nan, np.float32) for x in xs])
    return {'keys': data['keys'], 'lengths': lengths, 'fwd': pad(fwd), 'bwd': pad(bwd)}


def save(name, payload):
    np.savez_compressed(OUT / f'{name}.npz', **payload)
    print('wrote', name, flush=True)


def main():
    t0 = time.time()
    if '--zero-shot' in sys.argv:
        torch.manual_seed(SEED)
        pipe = BaseChronosPipeline.from_pretrained(REPO, device_map='cuda', torch_dtype=torch.float32)
        pipe.model.eval()
        for variable in ('temperature', 'oxygen'):
            save(f'continuation_{variable}_zs', continuation(pipe, variable, ['Bridge-Dev', 'Bridge-2026']))
            save(f'gap_{variable}_zs', gap(pipe, variable))
        (OUT / 'zero_shot.status.json').write_text(json.dumps({'seconds': round(time.time() - t0)}))
        return
    variable = sys.argv[sys.argv.index('--variable') + 1]
    method = sys.argv[sys.argv.index('--method') + 1]
    params = json.loads((INP / 'configurations.json').read_text())[method][variable]['params']
    record = {'variable': variable, 'method': method, 'params': params, 'tasks': {}}
    for task in ('continuation', 'gap'):
        label = f'{variable}|{method}|{task}'
        rng = np.random.default_rng([SEED, zlib.crc32(label.encode())])
        torch.manual_seed(SEED + zlib.crc32(label.encode()) % 100000)
        ctxs, tgts, ti, vi = windows(variable, task, rng)
        pipe, info = train(method, params, ctxs, tgts, ti, vi, rng)
        info.update(n_windows=int(len(ctxs)), n_train=int(len(ti)), n_val=int(len(vi)), horizon=HORIZON[task])
        if task == 'continuation':
            save(f'continuation_{variable}_{method}', continuation(pipe, variable, ['Bridge-2026']))
        else:
            save(f'gap_{variable}_{method}', gap(pipe, variable))
        record['tasks'][task] = info
        print(label, {k: v for k, v in info.items() if k != 'curve'}, flush=True)
        del pipe; torch.cuda.empty_cache()
    record['seconds'] = round(time.time() - t0)
    (OUT / f'train_{variable}_{method}.json').write_text(json.dumps(record, indent=2))


if __name__ == '__main__':
    main()
