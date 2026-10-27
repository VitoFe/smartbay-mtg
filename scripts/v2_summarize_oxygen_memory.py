"""Table 5.7 Panel B under tuning protocol v2: v1 scoring, sparse-cell sensitivity and week bootstrap, unchanged."""
from pathlib import Path
import json, sys
import numpy as np
import pandas as pd
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'scripts'))
from verify_bay_oxygen_memory import macro, sparse_sensitivity

root = ROOT / 'results/v2_oxygen_memory' if len(sys.argv) < 2 else Path(sys.argv[1])
p = pd.read_csv(root / 'predictions.csv.gz'); key = ['fold', 'horizon', 'sensor', 'episode']
reference = p[p.family == 'persistence']; pairs = []
for (h, family), g in p.groupby(['horizon', 'family']):
    v = g.dropna(subset=['target_oxygen', 'prediction'])
    paired = v.merge(reference[key + ['prediction']].rename(columns={'prediction': 'baseline'}), on=key, validate='one_to_one').dropna(subset=['baseline'])
    pairs.append(dict(horizon=h, family=family, endpoints=len(paired), mae=float(macro(paired)), persistence_mae=float(macro(paired, 'baseline'))))
pd.DataFrame(pairs).to_csv(root / 'paired_summary.csv', index=False)
sparse_sensitivity(root, p)
rng = np.random.default_rng(20260917); boot = []
for h, family in [(h, f) for h in (24, 72) for f in ('coarse100', 'solubility', 'ridge', 'selected')]:
    a = p[(p.family == family) & (p.horizon == h)].dropna(subset=['target_oxygen', 'prediction']).copy(); draws = []
    for _ in range(500):
        pieces = []
        for fold, g in a.groupby('fold'):
            weeks = g.episode.unique(); pieces.extend([g[g.episode == w] for w in rng.choice(weeks, len(weeks), replace=True)])
        g = pd.concat(pieces, ignore_index=True); draws.append(macro(g) - macro(g, 'initial_oxygen'))
    boot.append(dict(horizon=h, family=family, delta_mae=float(macro(a) - macro(a, 'initial_oxygen')),
                     lower=float(np.quantile(draws, .025)), upper=float(np.quantile(draws, .975)), draws=500))
pd.DataFrame(boot).to_csv(root / 'paired_bootstrap.csv', index=False)
print(pd.DataFrame(pairs).to_string(index=False)); print(pd.DataFrame(boot).to_string(index=False))
print(pd.read_csv(root / 'ridge_support_sensitivity.csv').drop(columns='policy').to_string(index=False))
print(json.dumps([{k: c[k] for k in ('fold', 'horizon', 'selected')} for c in json.loads((root / 'contract.json').read_text())['folds']]))
