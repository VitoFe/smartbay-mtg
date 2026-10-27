"""A published horizon arm re-tuned under a different TPE sampler seed, everything else identical.

A magnitude should not be quoted until the published grid has been re-scored under a second
sampler seed. It had never been done: every study in this project uses the protocol's sampler
seed, 42, and the pair of offset-0 runs once read as a seed difference had differed in the
selection objective instead. This runs `v2_horizon.run` unchanged, after setting
the sampler seed that `run_study` reads at call time, so neither contract-bound module is edited.

Development folds only; outputs go to the run's own `results/` under the published tag.
Usage: python -m src.experiments.v2_seed_replicate --variable oxygen --arm learned_source
       --horizon 6 --seed 7
"""
import argparse

from src.tuning import protocol_v2


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--variable', required=True)
    p.add_argument('--arm', required=True)
    p.add_argument('--horizon', type=int, required=True)
    p.add_argument('--seed', type=int, required=True)
    p.add_argument('--quick', action='store_true')
    a = p.parse_args()
    if a.seed == protocol_v2.PROTOCOL['sampler_seed']:
        raise SystemExit('that is the published seed; nothing to replicate')
    protocol_v2.PROTOCOL['sampler_seed'] = a.seed
    from src.experiments.v2_horizon import run
    run(a.variable, a.arm, a.horizon, a.quick)


if __name__ == '__main__':
    main()
