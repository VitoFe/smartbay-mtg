"""Stage `data`: load the raw record, apply quality control, and write the cohort inventory.

Everything downstream reads through the same two loaders, so this stage exists to make what they
return explicit and checkable rather than implicit in each experiment. It fits no model.

Two loader restrictions are deliberate and are recorded here rather than silently applied:
load_observations excludes 2026, and load_profiles restricts to Gen2 casts before 2026. The
reserved-period work reads the 2026 record through its own frozen path, which is why the reserved
cohort does not appear in these counts.

It also makes the stage's hourly file reproducible to the byte. src/data/bay_observations.audit
writes reports/bay_twin/hourly_observations.csv.gz through pandas, whose gzip header carries the
time of writing and the file name, so two runs over the same record gave files with the same content
and different bytes, and a byte digest of the file could never be met again. The audit module is
bound by the component contract (config/bay_integrated_contract.json), so it is left as it is and
the file is rewritten here: the same CSV, compressed at the same level, with a zero time stamp and
no name.
"""
import gzip
import json

import numpy as np
import pandas as pd

from src.data.bay_observations import read_export
from src.data.bay_profile_context import ROOT

OUT = ROOT / 'reports/bay_twin/runs/pipeline_data_summary.json'
GEOMETRY = ROOT / 'reports/bay_twin/sensor_geometry.csv'
HOURLY = ROOT / 'reports/bay_twin/hourly_observations.csv.gz'
QUANT_STEP = 0.1


def deterministic_gzip(path):
    """Rewrite a gzip file with a zero time stamp and no file name in its header, keeping its
    content; leave it untouched when it already has that form. Returns True when it was rewritten."""
    raw = path.read_bytes()
    fixed = gzip.compress(gzip.decompress(raw), compresslevel=9, mtime=0)
    if fixed == raw:
        return False
    path.write_bytes(fixed)
    return True


def probe_inventory():
    meta = pd.read_csv(GEOMETRY).set_index('sensor')
    rows = []
    for name in meta.index:
        _, frame, _ = read_export(ROOT / 'data/raw' / f'{name}.txt')
        t = frame['temperature'].dropna()
        o = frame['oxygen'].dropna() if 'oxygen' in frame else pd.Series(dtype=float)
        rows.append({
            'sensor': name,
            'site': str(meta.loc[name].get('site', '')),
            'nominal_depth_m': float(meta.loc[name].nominal_depth_m),
            'temperature_hours': int(len(t)),
            'oxygen_hours': int(len(o)),
            'first': str(t.index.min()) if len(t) else None,
            'last': str(t.index.max()) if len(t) else None,
        })
    return meta, rows


def quantisation_check(meta):
    """Confirm the 0.1 degC export resolution the metrology chapter rests on, probe by probe."""
    out = []
    for name in meta.index:
        _, frame, _ = read_export(ROOT / 'data/raw' / f'{name}.txt')
        t = frame['temperature'].dropna()
        if t.empty:
            continue
        # The raw exports are one-decimal throughout. Sub-step values can only come from our own
        # within-hour averaging, so they are counted rather than assumed away.
        steps = np.abs(np.diff(np.unique(np.round(t.to_numpy(), 6))))
        steps = steps[steps > 0]
        out.append({'sensor': name,
                    'min_step': round(float(steps.min()), 6) if steps.size else None,
                    'modal_step': round(float(np.median(steps)), 6) if steps.size else None})
    return out


def main():
    if HOURLY.exists() and deterministic_gzip(HOURLY):
        print('rewrote', HOURLY.name, 'with a zero time stamp and no file name in its gzip header')
    meta, probes = probe_inventory()
    quant = quantisation_check(meta)
    at_tenth = sum(1 for q in quant if q['min_step'] is not None
                   and abs(q['modal_step'] - QUANT_STEP) < 1e-9)

    summary = {
        'note': 'cohort inventory for the stage graph; fits nothing and decides nothing',
        'loader_restrictions': {
            'load_observations': 'excludes 2026 by design',
            'load_profiles': 'Gen2 casts before 2026 by design',
        },
        'probes': len(probes),
        'sites': int(meta['site'].nunique()) if 'site' in meta else None,
        'temperature_hours_total': int(sum(p['temperature_hours'] for p in probes)),
        'oxygen_hours_total': int(sum(p['oxygen_hours'] for p in probes)),
        'probes_with_oxygen': sum(1 for p in probes if p['oxygen_hours'] > 0),
        'quantisation': {
            'declared_step_degC': QUANT_STEP,
            'probes_at_declared_step': at_tenth,
            'standard_uncertainty_degC': round(QUANT_STEP / float(np.sqrt(12)), 4),
            'per_probe': quant,
        },
        'inventory': probes,
    }
    OUT.write_text(json.dumps(summary, indent=2))
    print(f"{summary['probes']} probes at {summary['sites']} sites; "
          f"{summary['temperature_hours_total']:,} temperature hours, "
          f"{summary['oxygen_hours_total']:,} oxygen hours "
          f"on {summary['probes_with_oxygen']} probes")
    print(f"quantisation: {at_tenth}/{len(quant)} probes at the declared "
          f"{QUANT_STEP} degC step, u = {summary['quantisation']['standard_uncertainty_degC']} degC")
    print('wrote', OUT)


if __name__ == '__main__':
    main()
