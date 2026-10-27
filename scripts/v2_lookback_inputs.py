"""Addendum 46: build, locally, the continuation contexts of every look-back length for the cluster.

For each context length L in 48, 128, 256, 512 and 1024 h, the probe's own hourly series over the
L hours ending at each addendum-39 onset, built by the published code path
(scripts.v2_chronos_adapt_inputs.continuation) with its context length set to L and nothing else
changed. Gate G1: the 512 h contexts equal addendum 41's uploaded ones bit for bit, every context
ends on its episode's persistence value (the builder checks it), and each shorter context equals
the last L hours of the 1024 h one.

Writes reports/bay_twin/runs/v2_lookback/inputs/continuation_<variable>_<L>.npz, gate_G1.json and a
manifest of their digests.

Usage: python -m scripts.v2_lookback_inputs
Declared in config/bay_reserved_protocol_v2_addendum46.json.
"""
import hashlib
import json

import numpy as np

from scripts import v2_chronos_adapt_inputs as published
from src.data.bay_profile_context import ROOT

RUNS = ROOT / 'reports/bay_twin/runs'
OUT = RUNS / 'v2_lookback' / 'inputs'
LENGTHS = (48, 128, 256, 512, 1024)
VARIABLES = ('temperature', 'oxygen')


def build(variable):
    """{L: (episodes, contexts)} for every declared length, and the G1 checks for this variable."""
    out = {}
    for length in LENGTHS:
        published.CTX = length                 # the one setting that differs from addendum 41
        episodes, contexts, g1_last = published.continuation(variable)
        out[length] = (episodes, contexts, g1_last)
    published.CTX = 512
    ref = np.load(RUNS / 'v2_chronos_adapt' / 'inputs' / f'continuation_{variable}.npz', allow_pickle=True)
    ep512, ctx512, _ = out[512]
    same_keys = (list(ref['sensor']) == list(ep512.sensor) and list(ref['cohort']) == list(ep512.cohort)
                 and list(ref['fold']) == list(ep512.fold))
    long = out[max(LENGTHS)][1]
    nested = {str(L): bool(np.array_equal(out[L][1], long[:, -L:], equal_nan=True)) for L in LENGTHS}
    checks = {'contexts_512_equal_addendum41': bool(same_keys and np.array_equal(ctx512, ref['contexts'], equal_nan=True)),
              'every_context_ends_on_persistence': all(v[2] for v in out.values()),
              'shorter_contexts_are_tails_of_1024': nested}
    checks['passed'] = checks['contexts_512_equal_addendum41'] and checks['every_context_ends_on_persistence'] \
        and all(nested.values())
    return out, checks


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    gate, manifest = {}, {}
    for variable in VARIABLES:
        built, checks = build(variable)
        gate[variable] = checks
        if not checks['passed']:
            (OUT / 'gate_G1.json').write_text(json.dumps(gate, indent=2))
            raise SystemExit(f'G1 failed for {variable}: {checks}')
        for length, (episodes, contexts, _) in built.items():
            path = OUT / f'continuation_{variable}_{length}.npz'
            np.savez_compressed(path, contexts=contexts, cohort=episodes.cohort.to_numpy(str),
                                fold=episodes.fold.to_numpy(), sensor=episodes.sensor.to_numpy(str),
                                onset=episodes.onset.astype(str).to_numpy(str),
                                persistence=episodes.persistence.to_numpy(float))
            manifest[path.name] = digest(path)
        print(variable, len(built[512][0]), 'episodes at', LENGTHS, 'h; G1', checks['passed'], flush=True)
    (OUT / 'gate_G1.json').write_text(json.dumps(gate, indent=2))
    (OUT / 'manifest.json').write_text(json.dumps(manifest, indent=2))


if __name__ == '__main__':
    main()
