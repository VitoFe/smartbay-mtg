"""Tuning-protocol-v2 campaigns on Trinity: hash-verified snapshot, detached jobs, status and fetch.

Same transport as scripts/bay_twin_campaign.py (which is left unchanged); credentials are read by
scientific_campaign.connect() and never printed.
"""
from pathlib import Path
import argparse, csv, hashlib, io, json, os, re, shlex, sys, tarfile
from datetime import datetime, timezone
sys.path.insert(0, str(Path(__file__).resolve().parent))
from scientific_campaign import connect
from bay_twin_campaign import execute

ROOT = Path(__file__).resolve().parents[1]
V2 = ['config/tuning_protocol_v2.json', 'src/tuning/protocol_v2.py']


def base_files():
    files = [ROOT / p for p in ['src/experiments/bay_twin_sweep.py', 'src/data/bay_observations.py', 'config/bay_twin_campaign.json']]
    files += list((ROOT / 'reports/bay_twin').glob('*.json'))
    files += [ROOT / 'reports/bay_twin' / p for p in ['sensor_inventory.csv', 'sensor_geometry.csv', 'hourly_observations.csv.gz', 'geometry.npz']]
    audit = json.loads((ROOT / 'reports/bay_twin/data_audit.json').read_text())
    files += [ROOT / p for p in audit['sources_sha256']]
    return files + [ROOT / p for p in V2]


def profile_files():
    files = [ROOT / p for p in ['src/data/bay_profile_context.py', 'src/models/bay_query_field.py', 'src/experiments/bay_profile_field.py',
                                'config/bay_profile_field.json', 'src/data/thermodynamics.py', 'reports/bay_twin/gen2_profile_inventory.csv',
                                'data/processed/ctd_all_casts.csv.gz', 'reports/bay_twin/profile_boundary_context.csv.gz',
                                'src/experiments/bay_kernel_field.py', 'src/experiments/bay_field_export.py']]
    with (ROOT / 'reports/bay_twin/gen2_profile_inventory.csv').open() as stream:
        files += [ROOT / 'data/raw' / (row['cast_id'] + '.txt') for row in csv.DictReader(stream)]
    provenance = json.loads((ROOT / 'reports/bay_twin/profile_boundary_provenance.json').read_text())
    return files + [ROOT / p for p in provenance['input_sha256']]


FORECAST_FILES = ['src/data/bay_profile_context.py', 'src/models/bay_graph_dynamics.py', 'src/experiments/bay_graph_dynamics.py',
                  'src/models/bay_query_field.py', 'src/experiments/bay_dynamics_tree.py', 'src/data/bay_forcing.py',
                  'data/raw/T2-Meteo.txt', 'src/experiments/v2_forecast.py']


def plan(campaign):
    """Return (files, jobs); a job is (name, module args, gpu, threads)."""
    if campaign == 'forecast':
        files = base_files() + [ROOT / p for p in FORECAST_FILES]
        jobs = []
        for v in ['temperature', 'oxygen']:
            for arm in ['tree', 'weather_tree']:
                jobs.append((f'{v}_{arm}_nested', ['src.experiments.v2_forecast', '--variable', v, '--arm', arm], '', 4))
            jobs.append((f'{v}_weather_tree_release', ['src.experiments.v2_forecast', '--variable', v, '--arm', 'weather_tree', '--mode', 'release'], '', 4))
            for arm in ['learned_source', 'conservative_exchange', 'coast_exchange', 'coast_breakwater_exchange', 'coast_depth_exchange'] + \
                       (['saturation_source', 'saturation_exchange'] if v == 'oxygen' else []):
                jobs.append((f'{v}_{arm}_nested', ['src.experiments.v2_forecast', '--variable', v, '--arm', arm], '1', 2))
        return files, jobs
    if campaign == 'horizon':  # addendum-2: does physical guidance earn its place at longer lead times?
        files = base_files() + [ROOT / p for p in FORECAST_FILES] + \
                [ROOT / p for p in ['src/experiments/v2_horizon.py', 'config/bay_reserved_protocol_v2_addendum.json']]
        jobs = []
        for v in ['temperature', 'oxygen']:
            for horizon in [int(h) for h in os.environ.get('V2_HORIZONS', '6,24,48').split(',')]:
                for arm in ['tree', 'weather_tree']:
                    jobs.append((f'{v}_{arm}_h{horizon}', ['src.experiments.v2_horizon', '--variable', v, '--arm', arm,
                                                           '--horizon', str(horizon)], '', 3))
                for arm in ['learned_source', 'conservative_exchange'] + (['saturation_exchange'] if v == 'oxygen' else []):
                    jobs.append((f'{v}_{arm}_h{horizon}', ['src.experiments.v2_horizon', '--variable', v, '--arm', arm,
                                                           '--horizon', str(horizon)], '1', 2))
        return files, jobs
    if campaign == 'sequence':  # addendum-23: is the six-hour task saturated, or does pretraining carry something?
        files = base_files() + [ROOT / p for p in FORECAST_FILES] + \
                [ROOT / p for p in ['src/experiments/v2_horizon.py', 'src/experiments/v2_sequence_arms.py',
                                    'config/bay_reserved_protocol_v2_addendum23.json']] + \
                [ROOT / f'reports/bay_twin/runs/20260919T144822Z_v2_forecast/results/'
                        f'v2_forecast_{v}_tree_nested/{v}_tree_nested_selection.json'
                 for v in ['temperature', 'oxygen']]
        jobs = [(f'{v}_sequence_arms', ['src.experiments.v2_sequence_arms', '--variable', v], '1', 4)
                for v in ['temperature', 'oxygen']]
        return files, jobs
    if campaign == 'river':  # addendum-6: does Magra river level improve known-sensor forecasting?
        files = base_files() + [ROOT / p for p in FORECAST_FILES] +                 [ROOT / p for p in ['src/experiments/v2_horizon.py', 'src/experiments/v2_river.py',
                                    'src/data/bay_river.py', 'config/bay_reserved_protocol_v2_addendum6.json']] +                 sorted((ROOT / 'data/external/Rivers').glob('arpa_magra_ameglia_portata_*.csv'))
        jobs = [(f'{v}_{arm}_h{h}', ['src.experiments.v2_river', '--variable', v, '--arm', arm,
                                     '--horizon', str(h)], '', 3)
                for v in ['temperature', 'oxygen'] for arm in ['river_tree', 'weather_river_tree']
                for h in [6, 24, 48]]
        return files, jobs
    if campaign == 'gaprecovery':  # addendum-30: recovering a masked gap by regression, rolling
        #                            forecast, hybrid, and an in-context tabular model
        files = base_files() + [ROOT / p for p in FORECAST_FILES] + \
                [ROOT / p for p in ['src/experiments/v2_gap_recovery.py',
                                    'config/bay_reserved_protocol_v2_addendum30.json',
                                    'config/bay_reserved_protocol_v2_addendum31.json',
                                    'config/bay_reserved_protocol_v2_addendum32.json',
                                    'config/bay_reserved_protocol_v2_addendum33.json',
                                    # addendum 34 reads the Magra level beside the meteorology
                                    'config/bay_reserved_protocol_v2_addendum34.json',
                                    'config/bay_reserved_protocol_v2_addendum34_amendment1.json',
                                    'config/bay_reserved_protocol_v2_addendum35.json',
                                    'config/bay_reserved_protocol_v2_addendum35_amendment1.json',
                                    # addendum 59: the Gaussian process's mean per pattern of
                                    # reporting neighbours (V2_GAP_GP_PATTERNS, on by default)
                                    'config/bay_reserved_protocol_v2_addendum59.json',
                                    'src/data/bay_river.py']] + \
                sorted((ROOT / 'data/external/Rivers').glob('arpa_magra_ameglia_portata_*.csv'))
        # V2_GAP_ARMS narrows the launch, so a single arm can be added later without re-running
        # eleven that already finished. The default is every arm that needs no licence token.
        arms = os.environ.get('V2_GAP_ARMS',
                              'regression_causal,regression_retro,rolling_forward,'
                              'rolling_backward,hybrid,tabicl,tabpfn,chronos_forward,'
                              'chronos_bidir,polynomial,gpr,compressed_sensing,lstm').split(',')
        GPU_ARMS = ('tabicl', 'tabpfn', 'chronos_forward', 'chronos_bidir', 'lstm',
                    'chronos2_bidir', 'chronos2_cov_bidir',
                    # addendum 35
                    'chronos2_exo_bidir', 'chronos2_all_bidir', 'bolt_resid_bidir',
                    'chronos2_cov_resid_bidir', 'bolt_resid_lin_bidir',
                    'chronos2_cov_resid_lin_bidir')
        # Both in-context models need the GPU. TabPFN refuses more than 5000 training rows on CPU,
        # and the prompt here is larger than that, so denying it a device fails the job outright.
        # Addendum 34: the state-space arms are a ridge solve per probe and need one thread; the
        # anchored trees get two, as every tuned tree here has.
        # GPU 1 by default, as the project's shared-machine rule asks; V2_GAP_GPU moves a launch to
        # the other card when this campaign has already filled GPU 1.
        gpu = os.environ.get('V2_GAP_GPU', '1')
        # V2_GAP_VARIABLES narrows a relaunch to the jobs that failed, as V2_GAP_ARMS does.
        variables = os.environ.get('V2_GAP_VARIABLES', 'temperature,oxygen').split(',')
        jobs = [(f'{v}_{arm}', ['src.experiments.v2_gap_recovery', '--variable', v, '--arm', arm],
                 gpu if arm in GPU_ARMS else '', 1 if arm.startswith('arx_') else 2)
                for v in variables for arm in arms]
        return files, jobs
    if campaign == 'gapradius':  # addendum 75: the Gaussian process with its neighbours beyond a radius
        files, _ = plan('gaprecovery')
        radius = __import__('importlib').import_module('scripts.v2_gap_neighbour_radius')
        files += [ROOT / 'scripts/v2_gap_neighbour_radius.py', ROOT / 'scripts/v2_gap_climatology.py',
                  ROOT / 'config/bay_reserved_protocol_v2_addendum75.json']
        files += [radius.INPUTS / f'{c}.csv.gz' for c in radius.COHORTS]
        job = ['scripts.v2_gap_neighbour_radius', '--cohort']
        jobs = [(f'gate_{c}', job + [c, '--gate'], '', 2) for c in radius.COHORTS]
        jobs += [(f'fill_{c}_r{r}', job + [c, '--fill', '--radius', str(r)], '', 2)
                 for c in radius.COHORTS for r in radius.RADII]
        return files, jobs
    if campaign == 'gpgate':  # addendum 59 part A: with the fix switched off, the Gaussian process
        #                       reproduces every published GP column it will replace, to 1e-9
        files, _ = plan('gaprecovery')
        gate = __import__('importlib').import_module('scripts.v2_gap_gp_gate')
        files += [ROOT / 'scripts/v2_gap_gp_gate.py']
        files += [ROOT / 'reports/bay_twin/runs' / run / results / f'{v}_{arm}' / 'predictions.csv.gz'
                  for run, results, arm, _ in gate.PUBLISHED_GP for v in ('temperature', 'oxygen')]
        jobs = [(f'gate_{run[:16]}_{v}_{arm}',
                 ['scripts.v2_gap_gp_gate', '--only', run, '--arm', arm, '--variable', v,
                  '--out', f'results/gate_{run}__{v}_{arm}.json'], '',
                 int(os.environ.get('V2_GATE_THREADS', '2')))
                for run, results, arm, _ in gate.PUBLISHED_GP for v in ('temperature', 'oxygen')]
        return files, jobs
    if campaign == 'phasefix':  # every phase tuned on the published objective, the
        #                         operator's matched tree on the operator's objective, and a real
        #                         second and third sampler seed on the published grid
        operator_files = [p for o in range(1, 6) for p in (
            f'reports/bay_twin/runs/v2_phase_shift/phase_operator_oxygen_offset{o}.csv.gz',
            f'reports/bay_twin/runs/v2_phase_shift/phase_operator_oxygen_offset{o}_summary.json')]
        files = base_files() + [ROOT / p for p in FORECAST_FILES] + \
                [ROOT / p for p in ['src/experiments/v2_horizon.py', 'src/experiments/v2_phase_shift.py',
                                    'src/experiments/v2_phase_operator.py',
                                    'config/bay_reserved_protocol_v2_addendum16.json',
                                    'config/bay_reserved_protocol_v2_addendum17.json',
                                    'config/bay_reserved_protocol_v2_addendum18.json',
                                    'config/bay_reserved_protocol_v2_addendum29.json'] + operator_files]
        shift = ['src.experiments.v2_phase_shift', '--objective', 'ratio']
        jobs = ([(f'h{h}_ratio_offset{o}', shift + ['--offset', str(o), '--horizon', str(h)], '', 2)
                 for h in (6, 24) for o in range(6)] +
                [(f'operator_treeratio_offset{o}', ['src.experiments.v2_phase_operator', '--offset',
                                                    str(o), '--tree-objective', 'ratio'], '', 2)
                 for o in range(1, 6)] +
                [(f'h{h}_ratio_offset0_seed{s}', shift + ['--offset', '0', '--horizon', str(h),
                                                          '--seed', str(s)], '', 2)
                 for h in (6, 24) for s in (7, 123)])
        return files, jobs
    if campaign == 'phaseriver':  # addendum 44: the 24 h oxygen river contrast over the six phases
        #                           and two further sampler seeds, paired with the phasefix trees
        trees = 'reports/bay_twin/runs/20260923T103430Z_v2_phasefix/reports/bay_twin/runs/v2_phase_shift_h24'
        cells = [(o, '') for o in range(6)] + [(0, f'_seed{s}') for s in (7, 123)]
        files = base_files() + [ROOT / p for p in FORECAST_FILES] + \
                [ROOT / p for p in ['src/experiments/v2_horizon.py', 'src/experiments/v2_phase_shift.py',
                                    'src/experiments/v2_phase_river.py', 'src/data/bay_river.py',
                                    'config/bay_reserved_protocol_v2_addendum6.json',
                                    'config/bay_reserved_protocol_v2_addendum44.json',
                                    # the G1 reference, read at run time (added after run 20260924T210004Z)
                                    'reports/bay_twin/runs/v2_augmented_vs_plain.json']] + \
                sorted((ROOT / 'data/external/Rivers').glob('arpa_magra_ameglia_portata_*.csv')) + \
                [ROOT / trees / f'phase_shift_{name}' for o, s in cells
                 for name in (f'oxygen_offset{o}_ratio{s}.csv.gz', f'offset{o}_ratio{s}_summary.json')]
        jobs = [(f'river_h24_offset{o}{s}', ['src.experiments.v2_phase_river', '--offset', str(o)] +
                 (['--seed', s.removeprefix('_seed')] if s else []), '', 2) for o, s in cells]
        return files, jobs
    if campaign == 'phasewriver':  # addendum 45 part A: weather plus river against the weather tree,
        #                            the same cells and reused trees as phaseriver
        files, _ = plan('phaseriver')
        files += [ROOT / p for p in ['src/experiments/v2_phase_weather_river.py',
                                     'config/bay_reserved_protocol_v2_addendum45.json',
                                     'reports/bay_twin/runs/v2_river_summary.json']]
        cells = [(o, '') for o in range(6)] + [(0, f'_seed{s}') for s in (7, 123)]
        jobs = [(f'weather_river_h24_offset{o}{s}', ['src.experiments.v2_phase_weather_river', '--offset', str(o)] +
                 (['--seed', s.removeprefix('_seed')] if s else []), '', 2) for o, s in cells]
        return files, jobs
    if campaign == 'phaseriver6':  # addendum 78: the six-hour river contrast on both variables over the
        #                            six phases and two further seeds, paired with the phasefix trees
        trees = 'reports/bay_twin/runs/20260923T103430Z_v2_phasefix/reports/bay_twin/runs/v2_phase_shift'
        cells = [(o, '') for o in range(6)] + [(0, f'_seed{s}') for s in (7, 123)]
        variables = ('temperature', 'oxygen')
        files = base_files() + [ROOT / p for p in FORECAST_FILES] + \
                [ROOT / p for p in ['src/experiments/v2_horizon.py', 'src/experiments/v2_phase_shift.py',
                                    'src/experiments/v2_phase_river.py', 'src/experiments/v2_phase_river6.py',
                                    'src/data/bay_river.py', 'config/bay_reserved_protocol_v2_addendum6.json',
                                    'config/bay_reserved_protocol_v2_addendum78.json',
                                    'reports/bay_twin/runs/v2_augmented_vs_plain.json']] + \
                sorted((ROOT / 'data/external/Rivers').glob('arpa_magra_ameglia_portata_*.csv')) + \
                [ROOT / trees / f'phase_shift_{v}_offset{o}_ratio{s}.csv.gz' for o, s in cells for v in variables] + \
                [ROOT / trees / f'phase_shift_offset{o}_ratio{s}_summary.json' for o, s in cells]
        jobs = [(f'river_h6_{v}_offset{o}{s}', ['src.experiments.v2_phase_river6', '--variable', v, '--offset', str(o)] +
                 (['--seed', s.removeprefix('_seed')] if s else []), '', 2) for v in variables for o, s in cells]
        return files, jobs
    if campaign == 'heldnested':  # the held-site temperature forecast with its own
        #                           selection per (fold, site), which never sees the held site
        sites = ['Baia_Pontone', 'Baia_S.Teresa', 'Mitili_Centro', 'Mitili_Centro_Est',
                 'Mitili_Centro_Ovest', 'Mitili_Est', 'Portovenere_Benzinaio', 'Portovenere_Secco']
        files = base_files() + [ROOT / p for p in FORECAST_FILES] + \
                [ROOT / p for p in ['src/experiments/v2_horizon.py', 'src/experiments/v2_held_station.py',
                                    'src/experiments/v2_closed_loop.py',
                                    'config/bay_reserved_protocol_v2_addendum10.json']]
        jobs = [(f'temperature_nested_{s}', ['src.experiments.v2_held_station', '--variable', 'temperature',
                                              '--nested-site', s], '', 2) for s in sites]
        return files, jobs
    if campaign == 'seedrep':  # the graph arms under two more sampler seeds, so the
        #                        physics contrasts at 6 h (and the 24 h oxygen operator) can be read
        #                        against the spread the sampler alone produces
        files = base_files() + [ROOT / p for p in FORECAST_FILES] + \
                [ROOT / p for p in ['src/experiments/v2_horizon.py', 'src/experiments/v2_seed_replicate.py',
                                    'config/bay_reserved_protocol_v2_addendum.json']]
        cases = [(v, arm, 6) for v in ('oxygen', 'temperature')
                 for arm in ('learned_source', 'conservative_exchange')] + [('oxygen', 'learned_source', 24)]
        # ONE seed per launch. v2_horizon names its study without the seed, so two seeds in one run
        # directory share one SQLite study; run 20260923T105150Z did exactly that and is not used.
        seeds = [int(s) for s in os.environ.get('V2_SEEDS', '7').split(',')]
        if len(seeds) != 1:
            raise ValueError('seedrep takes one seed per launch')
        jobs = [(f'{v}_{arm}_h{h}_seed{s}', ['src.experiments.v2_seed_replicate', '--variable', v, '--arm', arm,
                                              '--horizon', str(h), '--seed', str(s)], '1', 2)
                for v, arm, h in cases for s in seeds]
        return files, jobs
    if campaign == 'phase24':  # addendum-29: the six issuance grids at a 24 h lead, where the
        #                        target falls on the SAME clock hour as its issuance
        files = base_files() + [ROOT / p for p in FORECAST_FILES] + \
                [ROOT / p for p in ['src/experiments/v2_horizon.py', 'src/experiments/v2_phase_shift.py',
                                    'config/bay_reserved_protocol_v2_addendum29.json']]
        jobs = [(f'phase24_offset{o}', ['src.experiments.v2_phase_shift', '--offset', str(o),
                                        '--horizon', '24'], '', 4) for o in range(6)]
        return files, jobs
    if campaign == 'riverlag':  # addendum-26: a tuned lag, and a second catchment gated on the local one
        files = base_files() + [ROOT / p for p in FORECAST_FILES] + \
                [ROOT / p for p in ['src/experiments/v2_horizon.py', 'src/experiments/v2_river_lag.py',
                                    'src/data/bay_river.py', 'src/data/bay_bocca.py',
                                    'data/external/Rivers/liv_TOS_Arno_Bocca.csv',
                                    'config/bay_reserved_protocol_v2_addendum26.json']] + \
                sorted((ROOT / 'data/external/Rivers').glob('arpa_magra_ameglia_portata_*.csv'))
        jobs = [(f'{v}_{arm}_h{h}', ['src.experiments.v2_river_lag', '--variable', v, '--arm', arm,
                                     '--horizon', str(h)], '', 3)
                for v in ['temperature', 'oxygen']
                for arm in ['ameglia_lag', 'bocca_lag', 'bocca_gated'] for h in [6, 24]]
        return files, jobs
    if campaign == 'ispra':  # addendum-7: external reference station as forecasting context
        files = base_files() + [ROOT / p for p in FORECAST_FILES] +                 [ROOT / p for p in ['src/experiments/v2_horizon.py', 'src/experiments/v2_ispra.py',
                                    'src/data/bay_ispra.py', 'data/external/Ispra/ISPRA_La_Spezia_2026_06_10.csv',
                                    'config/bay_reserved_protocol_v2_addendum7.json']]
        jobs = [(f'{v}_{arm}_h{h}', ['src.experiments.v2_ispra', '--variable', v, '--arm', arm,
                                     '--horizon', str(h)], '', 3)
                for v in ['temperature', 'oxygen'] for arm in ['sealevel_tree', 'extboundary_tree']
                for h in [6, 24, 48]]
        return files, jobs
    if campaign == 'bathy':  # addendum-8: surveyed seabed depth as a static feature
        files = base_files() + [ROOT / p for p in FORECAST_FILES] +                 [ROOT / p for p in ['src/experiments/v2_horizon.py', 'src/experiments/v2_bathy.py',
                                    'config/bay_reserved_protocol_v2_addendum8.json']]
        jobs = [(f'{v}_bathy_h{h}', ['src.experiments.v2_bathy', '--variable', v, '--horizon', str(h)], '', 3)
                for v in ['temperature', 'oxygen'] for h in [6, 24]]
        return files, jobs
    if campaign == 'xgb':  # addendum-1: tuned XGBoost counterparts of the two LightGBM tree arms
        files = base_files() + [ROOT / p for p in FORECAST_FILES] + [ROOT / 'config/bay_reserved_protocol_v2_addendum.json']
        jobs = [(f'{v}_{arm}_nested', ['src.experiments.v2_forecast', '--variable', v, '--arm', arm], '', 4)
                for v in ['temperature', 'oxygen'] for arm in ['xgb_tree', 'xgb_weather_tree']]
        return files, jobs
    if campaign == 'bayes':  # addendum-1 PyMC experiment on Field-57
        files = base_files() + profile_files() + [ROOT / p for p in ['src/experiments/v2_field.py', 'src/experiments/v2_bayes_field.py',
                                                                     'config/bay_reserved_protocol_v2_addendum.json']]
        return files, [('bayes_field', ['src.experiments.v2_bayes_field'], '', 4)]
    if campaign == 'bridge72':  # addendum 63: the silent-probe continuation from 54 to 72 h. The
        #                         files are every one a quick run opened or imported (traced with an
        #                         audit hook), so the snapshot carries exactly what the experiment reads
        code = ['src/experiments/' + m + '.py' for m in (
            'v2_bridge72', 'v2_bridge', 'v2_closed_loop', 'v2_horizon', 'v2_horizon_reserved', 'v2_forecast',
            'v2_gap_recovery', 'bay_reserved_evaluation', 'bay_twin_sweep', 'bay_dynamics_tree',
            'bay_graph_dynamics')]
        code += ['src/data/bay_forcing.py', 'src/data/bay_observations.py', 'src/data/bay_profile_context.py',
                 'src/data/thermodynamics.py', 'src/models/bay_graph_dynamics.py',
                 'src/models/bay_sensor_inference.py', 'src/protocols/bay_calibration.py']
        inputs = ['reports/bay_twin/sensor_geometry.csv',
                  'config/bay_reserved_protocol_v2_addendum39.json', 'config/bay_reserved_protocol_v2_addendum63.json']
        inputs += [f'reports/bay_twin/runs/{d}/{p}_{v}.csv.gz' for v in ('temperature', 'oxygen')
                   for d, p in (('v2_bridge', 'bridge'), ('v2_closed_loop', 'closed_loop'))]
        raw = ([f'data/raw/Nodo_P0{i}.txt' for i in (1, 2, 3)] + [f'data/raw/Nodo_S{i:02d}.txt' for i in range(1, 12)]
               + ['data/raw/T2-Meteo.txt']
               + [f'data/raw/T{s}-CTD(calibrato).txt' for s in ('BaiaP_01m', 'MitiliE_05m', 'MitiliO_05m',
                                                                 'PortovenereP_02m', 'PortovenereS_02m')])
        files = [ROOT / p for p in code + inputs + raw + V2]
        job = ['src.experiments.v2_bridge72', '--variable']
        jobs = [('temperature_seed42', job + ['temperature'], '', 8), ('oxygen_seed42', job + ['oxygen'], '', 8),
                ('temperature_seed7', job + ['temperature', '--seed', '7'], '', 8),
                ('temperature_seed123', job + ['temperature', '--seed', '123'], '', 8)]
        return files, jobs
    if campaign == 'probabilistic':  # addendum 64: quantile forecasts, anticipated change, a forecast-based
        #                                decline alert and the tide; files traced with an audit hook from quick
        #                                runs, plus the published rows gates G1 and G2 read
        code = ['src/experiments/' + m + '.py' for m in (
            'v2_probabilistic', 'v2_horizon', 'v2_horizon_reserved', 'v2_forecast', 'bay_reserved_evaluation',
            'bay_twin_sweep', 'bay_dynamics_tree', 'bay_graph_dynamics')]
        code += ['src/data/bay_forcing.py', 'src/data/bay_ispra.py', 'src/data/bay_observations.py',
                 'src/data/bay_profile_context.py', 'src/models/bay_graph_dynamics.py',
                 'src/models/bay_sensor_inference.py', 'src/protocols/bay_calibration.py']
        inputs = ['reports/bay_twin/sensor_geometry.csv', 'reports/bay_twin/hourly_observations.csv.gz',
                  'data/external/Ispra/ISPRA_La_Spezia_2026_06_10.csv', 'config/bay_reserved_protocol_v2_addendum64.json']
        inputs += [f'reports/bay_twin/runs/v2_horizon_reserved_h24/reserved_h24_{v}.csv.gz' for v in ('temperature', 'oxygen')]
        inputs += [p.relative_to(ROOT).as_posix() for v in ('temperature', 'oxygen') for h in (1, 3, 6, 12, 24, 36, 48)
                   for p in ROOT.glob(f'reports/bay_twin/runs/20260920T*_v2_horizon/results/v2_horizon_{v}_tree_h{h}/'
                                      'selected_predictions.csv.gz')]
        raw = ([f'data/raw/Nodo_P0{i}.txt' for i in (1, 2, 3)] + [f'data/raw/Nodo_S{i:02d}.txt' for i in range(1, 12)]
               + [f'data/raw/T{s}-CTD(calibrato).txt' for s in ('BaiaP_01m', 'MitiliE_05m', 'MitiliO_05m',
                                                                 'PortovenereP_02m', 'PortovenereS_02m')])
        files = [ROOT / p for p in code + inputs + raw + V2]
        job = ['src.experiments.v2_probabilistic', '--part', 'tree', '--variable']
        jobs = [(f'{v}_{arm}_seed{seed}', job + [v, '--arm', arm, '--seed', str(seed)], '', threads)
                for v, threads in (('temperature', 6), ('oxygen', 2)) for seed in (42, 7, 123)
                for arm in ('quantile_tree', 'quantile_tree_tide')]
        jobs.append(('chronos', ['src.experiments.v2_probabilistic', '--part', 'chronos'], '1', 4))
        return files, jobs
    if campaign == 'oracle':  # addendum 65: a perfect weather forecast to 72 h; files traced with an audit
        #                         hook from quick runs
        code = ['src/experiments/' + m + '.py' for m in (
            'v2_weather_oracle', 'v2_probabilistic', 'v2_horizon', 'v2_horizon_reserved', 'v2_forecast',
            'bay_reserved_evaluation', 'bay_twin_sweep', 'bay_dynamics_tree', 'bay_graph_dynamics')]
        code += ['src/data/bay_forcing.py', 'src/data/bay_ispra.py', 'src/data/bay_observations.py',
                 'src/data/bay_profile_context.py', 'src/models/bay_graph_dynamics.py',
                 'src/models/bay_sensor_inference.py', 'src/protocols/bay_calibration.py']
        inputs = ['reports/bay_twin/sensor_geometry.csv', 'reports/bay_twin/hourly_observations.csv.gz',
                  'config/bay_reserved_protocol_v2_addendum65.json']
        raw = ([f'data/raw/Nodo_P0{i}.txt' for i in (1, 2, 3)] + [f'data/raw/Nodo_S{i:02d}.txt' for i in range(1, 12)]
               + ['data/raw/T2-Meteo.txt']
               + [f'data/raw/T{s}-CTD(calibrato).txt' for s in ('BaiaP_01m', 'MitiliE_05m', 'MitiliO_05m',
                                                                 'PortovenereP_02m', 'PortovenereS_02m')])
        files = [ROOT / p for p in code + inputs + raw + V2]
        job = ['src.experiments.v2_weather_oracle', '--part', 'tree', '--variable']
        jobs = [(f'{v}_{arm}_seed{seed}', job + [v, '--arm', arm, '--seed', str(seed)], '', threads)
                for v, threads in (('temperature', 6), ('oxygen', 2)) for seed in (42, 7, 123)
                for arm in ('quantile_tree_weather', 'quantile_tree_weather_future')]
        jobs.append(('chronos', ['src.experiments.v2_weather_oracle', '--part', 'chronos'], '1', 4))
        return files, jobs
    if campaign == 'currents':  # addendum 80: the Mitili Est currents as a perfect input, on addendum 65's trees
        files, _ = plan('oracle')
        files += [ROOT / p for p in ['src/experiments/v2_currents_oracle.py',
                                     'config/bay_reserved_protocol_v2_addendum80.json',
                                     'data/raw/TMitiliE_02m-Corrente.txt', 'data/raw/TMitiliE_08m-Corrente.txt']]
        job = ['src.experiments.v2_currents_oracle', '--variable']
        jobs = [(f'{v}_{arm}_seed{seed}', job + [v, '--arm', arm, '--seed', str(seed)], '', threads)
                for v, threads in (('temperature', 6), ('oxygen', 2)) for seed in (42, 7, 123)
                for arm in ('quantile_tree_weather_currents', 'quantile_tree_weather_future_currents_future')]
        return files, jobs
    if campaign == 'arima':  # addendum 66: the ARIMA family to 72 h; files traced with an audit hook from
        #                        quick runs
        code = ['src/experiments/' + m + '.py' for m in (
            'v2_arima', 'v2_probabilistic', 'v2_horizon', 'v2_horizon_reserved', 'v2_forecast',
            'bay_reserved_evaluation', 'bay_twin_sweep', 'bay_dynamics_tree', 'bay_graph_dynamics')]
        code += ['src/data/bay_forcing.py', 'src/data/bay_ispra.py', 'src/data/bay_observations.py',
                 'src/data/bay_profile_context.py', 'src/models/bay_graph_dynamics.py',
                 'src/models/bay_sensor_inference.py', 'src/protocols/bay_calibration.py']
        inputs = ['reports/bay_twin/sensor_geometry.csv', 'config/bay_reserved_protocol_v2_addendum66.json']
        raw = ([f'data/raw/Nodo_P0{i}.txt' for i in (1, 2, 3)] + [f'data/raw/Nodo_S{i:02d}.txt' for i in range(1, 12)]
               + [f'data/raw/T{s}-CTD(calibrato).txt' for s in ('BaiaP_01m', 'MitiliE_05m', 'MitiliO_05m',
                                                                 'PortovenereP_02m', 'PortovenereS_02m')])
        files = [ROOT / p for p in code + inputs + raw + V2]
        job = ['src.experiments.v2_arima', '--variable']
        jobs = [(f'{v}_{arm}_seed{seed}', job + [v, '--arm', arm, '--seed', str(seed)], '', threads)
                for seed in (42, 7, 123)
                for v, arm, threads in (('temperature', 'linear_ar', 2), ('oxygen', 'linear_ar', 2),
                                        ('oxygen', 'linear_ar_joint', 2), ('temperature', 'linear_ar_tree', 6),
                                        ('oxygen', 'linear_ar_tree', 2))]
        return files, jobs
    if campaign == 'bayests':  # addendum 72: Bayesian DLM and BSTS to 72 h, on addendum 67's rows
        code = ['src/experiments/' + m + '.py' for m in (
            'v2_bayes_ts', 'v2_statespace', 'v2_probabilistic', 'v2_horizon', 'v2_horizon_reserved', 'v2_forecast',
            'bay_reserved_evaluation', 'bay_twin_sweep', 'bay_dynamics_tree', 'bay_graph_dynamics')]
        code += ['src/data/bay_forcing.py', 'src/data/bay_ispra.py', 'src/data/bay_observations.py',
                 'src/data/bay_profile_context.py', 'src/models/bay_graph_dynamics.py',
                 'src/models/bay_sensor_inference.py', 'src/protocols/bay_calibration.py',
                 'tests/test_bayes_ts.py']
        inputs = ['reports/bay_twin/sensor_geometry.csv', 'config/bay_reserved_protocol_v2_addendum72.json',
                  'config/bay_reserved_protocol_v2_addendum72_amendment1.json']
        raw = ([f'data/raw/Nodo_P0{i}.txt' for i in (1, 2, 3)] + [f'data/raw/Nodo_S{i:02d}.txt' for i in range(1, 12)]
               + [f'data/raw/T{s}-CTD(calibrato).txt' for s in ('BaiaP_01m', 'MitiliE_05m', 'MitiliO_05m',
                                                                 'PortovenereP_02m', 'PortovenereS_02m')])
        files = [ROOT / p for p in code + inputs + raw + V2]
        job = ['src.experiments.v2_bayes_ts', '--arm']
        procs = {'temperature': 7, 'oxygen': 5}
        jobs = [(f'{v}_dlm_seed{seed}', job + ['dlm', '--variable', v, '--seed', str(seed), '--procs', str(procs[v])],
                 '', procs[v]) for v in procs for seed in (42, 7, 123)]
        # BSTS takes the seed-42 DLM's structure, and waits for that job to write it
        jobs += [(f'{v}_bsts', job + ['bsts', '--variable', v, '--procs', str(procs[v] - 1)], '', procs[v] - 1)
                 for v in procs]
        return files, jobs
    if campaign == 'bayesbsts':  # addendum 72 amendment 2: the BSTS alone, Metropolis, on run 1's DLM selection
        files, _ = plan('bayests')
        files += [ROOT / p for p in ['config/bay_reserved_protocol_v2_addendum72_amendment2.json',
                                     'config/bay_reserved_protocol_v2_addendum72_amendment3.json',
                                     'config/bay_reserved_protocol_v2_addendum72_amendment4.json']
                  + [f'reports/bay_twin/runs/v2_bayes_ts/seed42/bayes_{v}_dlm.json' for v in ('temperature', 'oxygen')]]
        job = ['src.experiments.v2_bayes_ts', '--arm', 'bsts', '--variable']
        jobs = [(f'{v}_bsts', job + [v, '--procs', str(n)], '', n) for v, n in (('temperature', 7), ('oxygen', 4))]
        return files, jobs
    if campaign == 'statespace':  # addendum 67: state-space ARIMA and N-BEATS to 72 h
        code = ['src/experiments/' + m + '.py' for m in (
            'v2_statespace', 'v2_probabilistic', 'v2_horizon', 'v2_horizon_reserved', 'v2_forecast',
            'bay_reserved_evaluation', 'bay_twin_sweep', 'bay_dynamics_tree', 'bay_graph_dynamics')]
        code += ['src/data/bay_forcing.py', 'src/data/bay_ispra.py', 'src/data/bay_observations.py',
                 'src/data/bay_profile_context.py', 'src/models/bay_graph_dynamics.py',
                 'src/models/bay_sensor_inference.py', 'src/protocols/bay_calibration.py']
        inputs = ['reports/bay_twin/sensor_geometry.csv', 'config/bay_reserved_protocol_v2_addendum67.json']
        raw = ([f'data/raw/Nodo_P0{i}.txt' for i in (1, 2, 3)] + [f'data/raw/Nodo_S{i:02d}.txt' for i in range(1, 12)]
               + [f'data/raw/T{s}-CTD(calibrato).txt' for s in ('BaiaP_01m', 'MitiliE_05m', 'MitiliO_05m',
                                                                 'PortovenereP_02m', 'PortovenereS_02m')])
        files = [ROOT / p for p in code + inputs + raw + V2]
        job = ['src.experiments.v2_statespace', '--arm']
        procs = {'temperature': 7, 'oxygen': 5}
        jobs = [(f'{v}_arima_ss_seed{seed}', job + ['arima_ss', '--variable', v, '--seed', str(seed), '--procs', str(procs[v])],
                 '', procs[v]) for v in procs for seed in (42, 7, 123)]
        jobs += [(f'{v}_auto_arima', job + ['auto_arima', '--variable', v, '--procs', str(procs[v])], '', procs[v])
                 for v in procs]
        jobs += [(f'{v}_nbeats_seed{seed}', job + ['nbeats', '--variable', v, '--seed', str(seed)], '1', 2)
                 for v in procs for seed in (42, 7, 123)]
        return files, jobs
    if campaign == 'field':
        files = base_files() + profile_files() + [ROOT / 'src/experiments/v2_field.py']
        jobs = [(f'{task}_{arm}', ['src.experiments.v2_field', '--task', task, '--arm', arm], '', 4)
                for task in ['held_site'] for arm in __import__('importlib').import_module('src.experiments.v2_field').ARMS]
        return files, jobs
    raise ValueError(campaign)


#: What a campaign does with 2026, where it does not exclude it.
EVALUATION = {'bridge72': 'Development tuning under protocol v2; Bridge-2026 at 54 to 72 h is read once, '
                          'with every model selected before 2026 (addendum 63).',
              'probabilistic': 'Development tuning under protocol v2; the Reserved-2026 grid at seven leads and '
                               'Warning-2026 are read once, with every model selected before 2026 or zero-shot '
                               '(addendum 64).',
              'oracle': 'Development tuning under protocol v2; the Reserved-2026 grid at eight leads and Warning-2026 '
                        'are read once, with every model selected before 2026 or zero-shot; the weather after '
                        'issuance stands in for a perfect forecast (addendum 65).',
              'currents': 'Development tuning under protocol v2; the Reserved-2026 grid at eight leads is read once, '
                          'with every model selected before 2026; the Mitili Est currents after issuance stand in for '
                          'a perfect current forecast (addendum 80).',
              'arima': 'Development tuning under protocol v2; the Reserved-2026 grid at eight leads is read once, '
                       'with every model selected before 2026 (addendum 66).',
              'statespace': 'Development tuning under protocol v2; the Reserved-2026 grid at eight leads is read once, '
                            'with every model selected before 2026 (addendum 67).',
              'bayests': 'Development tuning under protocol v2; the Reserved-2026 grid at eight leads is read once, '
                         'with every model selected before 2026 (addendum 72).',
              'bayesbsts': 'No tuning: the BSTS takes the seed-42 DLM selection of run 20261006T022926Z_v2_bayests; '
                           'the Reserved-2026 grid at eight leads is read once (addendum 72, amendment 2).',
              'gapradius': 'No tuning: the published Gaussian process refitted per episode with its neighbours beyond '
                           'a radius, on the published gap cohorts; radius 0 must reproduce it first (addendum 75).'}


def launch(client, base, campaign, quick):
    files, jobs = plan(campaign)
    files = list(dict.fromkeys(files))
    missing = [p for p in files if not p.exists()]
    if missing: raise FileNotFoundError(missing[:5])
    run = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ') + f'_v2_{campaign}' + ('_smoke' if quick else '')
    local = ROOT / 'reports/bay_twin/runs' / run; local.mkdir(parents=True)
    manifest = {'run_id': run, 'protocol': json.loads((ROOT / V2[0]).read_text())['version'],
                'files_sha256': {p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for p in files},
                'created_utc': datetime.now(timezone.utc).isoformat(), 'smoke_only': quick,
                'evaluation': EVALUATION.get(campaign, 'Development-only tuning under protocol v2. '
                                                       'All 2026 values excluded by experiment code.')}
    (local / 'snapshot_manifest.json').write_text(json.dumps(manifest, indent=2))
    archive = local / 'snapshot.tar.gz'
    with tarfile.open(archive, 'w:gz') as tar:
        for p in files: tar.add(p, arcname=p.relative_to(ROOT).as_posix())
        content = json.dumps(manifest).encode(); info = tarfile.TarInfo('snapshot_manifest.json'); info.size = len(content)
        tar.addfile(info, io.BytesIO(content))
    remote = base + '/bay_twin/' + run
    code, out = execute(client, 'mkdir -p ' + shlex.quote(remote))
    if code: raise RuntimeError(out)
    sftp = client.open_sftp()
    try: sftp.put(str(archive), remote + '/snapshot.tar.gz')
    finally: sftp.close()
    probe = """import tarfile,pathlib,hashlib,json,importlib.metadata,subprocess
p=pathlib.Path.cwd()
with tarfile.open('snapshot.tar.gz') as t: t.extractall(p,filter='data')
m=json.loads((p/'snapshot_manifest.json').read_text())
for name,want in m['files_sha256'].items():
 assert hashlib.sha256((p/name).read_bytes()).hexdigest()==want,name
runtime={'packages':{n:importlib.metadata.version(n) for n in ['numpy','pandas','torch','optuna','lightgbm','scikit-learn']},'gpu':subprocess.check_output(['nvidia-smi','--query-gpu=name,memory.free','--format=csv,noheader'],text=True)}
(p/'runtime.json').write_text(json.dumps(runtime,indent=2)); print(json.dumps(runtime))
"""
    py = base + '/.venv/bin/python'
    code, out = execute(client, f'cd {shlex.quote(remote)} && {shlex.quote(py)} -c {shlex.quote(probe)}')
    if code: raise RuntimeError(out)
    print(out.strip())
    record = []
    # Experiment switches set locally (period, seed, lengths) must reach the remote process, or a
    # run meant for 2026 silently scores 2025 under the default. Only V2_GAP_* is forwarded, so no
    # other part of the local environment leaks into a job.
    forwarded = {k: v for k, v in os.environ.items() if k.startswith('V2_GAP_')}
    for name, args, gpu, threads in jobs:
        args = [py, '-u', '-m'] + args + (['--quick'] if quick else [])
        if ROOT / (args[3].replace('.', '/') + '.py') not in files: raise ValueError('module absent from snapshot: ' + args[3])
        script = ("import os,subprocess; env=os.environ.copy(); "
                  f"env.update({forwarded!r}); "
                  f"env.update(OMP_NUM_THREADS='{threads}',OPENBLAS_NUM_THREADS='{threads}',MKL_NUM_THREADS='{threads}',V2_THREADS='{threads}',CUDA_VISIBLE_DEVICES={gpu!r}); "
                  f"log=open({name + '.log'!r},'ab'); p=subprocess.Popen({args!r},env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True); print(p.pid)")
        code, out = execute(client, f'cd {shlex.quote(remote)} && {shlex.quote(py)} -c {shlex.quote(script)}')
        if code: raise RuntimeError(out)
        record.append({'job': name, 'pid': int(out.strip()), 'gpu': gpu, 'threads': threads})
    (local / 'transport.json').write_text(json.dumps({'run_id': run, 'remote_dir': remote, 'jobs': record, 'smoke_only': quick,
                                                     'forwarded_env': forwarded}, indent=2))
    print(run, len(record), 'jobs launched')


def status(client, base, run, fetch):
    if not re.fullmatch(r'\d{8}T\d{6}Z_v2_[a-z0-9_]+', run): raise ValueError('bad run id')
    remote = base + '/bay_twin/' + run; py = base + '/.venv/bin/python'
    probe = """import pathlib,json,sqlite3
p=pathlib.Path('results'); out=[]
for f in sorted(p.glob('status_*.json')):
    s=json.loads(f.read_text()); s.pop('result_sha256',None)
    name=f.stem.removeprefix('status_'); db=list(p.glob(name+'/*.sqlite3'))
    if db:
        c=sqlite3.connect(db[0]); s['complete']=c.execute("select count(*) from trials where state='COMPLETE'").fetchone()[0]
        s['failed']=c.execute("select count(*) from trials where state='FAIL'").fetchone()[0]; c.close()
    out.append(s)
print(json.dumps(out))
"""
    code, out = execute(client, f'cd {shlex.quote(remote)} && {py} -c {shlex.quote(probe)}')
    if code: raise RuntimeError(out)
    rows = json.loads(out)
    for s in rows:
        print(f"{s.get('variable', s.get('task', '')):12s} {s.get('arm', ''):28s} "
              f"{('h%d' % s['horizon']) if 'horizon' in s else s.get('mode', ''):8s} {s['state']:10s} "
              f"trials {s.get('complete', 0):3d} failed {s.get('failed', 0):2d} {s.get('error', '')[:80]}")
    if fetch:
        local = ROOT / 'reports/bay_twin/runs' / run
        bundle = """import pathlib,tarfile,hashlib,json
p=pathlib.Path.cwd()
m=json.loads((p/'snapshot_manifest.json').read_text())['files_sha256']
made=[f for f in (p/'reports').rglob('*') if f.is_file() and f.relative_to(p).as_posix() not in m] if (p/'reports').exists() else []
paths=[p/'runtime.json',p/'snapshot_manifest.json']+[f for f in (p/'results').rglob('*') if f.is_file() and f.name.endswith(('.json','.csv.gz','.sqlite3'))]+[f for f in made if f.name.endswith(('.json','.csv.gz','.sqlite3'))]+list(p.glob('*.log'))
with tarfile.open('results_download.tar.gz','w:gz',compresslevel=1) as a:
 for f in paths: a.add(f,arcname=f.relative_to(p).as_posix())
f=p/'results_download.tar.gz'; print(json.dumps({'sha256':hashlib.sha256(f.read_bytes()).hexdigest(),'bytes':f.stat().st_size,'files':len(paths)}))
"""
        code, out = execute(client, f'cd {shlex.quote(remote)} && {py} -c {shlex.quote(bundle)}')
        if code: raise RuntimeError(out)
        expected = json.loads(out); download = local / 'results_download.tar.gz'
        sftp = client.open_sftp()
        try: sftp.get(remote + '/results_download.tar.gz', str(download), max_concurrent_prefetch_requests=64)
        finally: sftp.close()
        if hashlib.sha256(download.read_bytes()).hexdigest() != expected['sha256']: raise ValueError('download checksum mismatch')
        with tarfile.open(download) as a: a.extractall(local, filter='data')
        (local / 'download_manifest.json').write_text(json.dumps(expected, indent=2))
        print('fetched and hash-verified', expected['files'], 'files')


def main():
    p = argparse.ArgumentParser(); p.add_argument('action', choices=['launch', 'status', 'fetch'])
    p.add_argument('--campaign', choices=['forecast', 'field', 'xgb', 'bayes', 'horizon', 'river', 'ispra', 'bathy', 'sequence', 'riverlag', 'phase24', 'gaprecovery', 'phasefix', 'seedrep', 'heldnested', 'phaseriver', 'phasewriver', 'gpgate', 'bridge72', 'probabilistic', 'oracle', 'arima', 'statespace', 'bayests', 'bayesbsts', 'gapradius', 'phaseriver6', 'currents']); p.add_argument('--run-id'); p.add_argument('--quick', action='store_true')
    a = p.parse_args(); c, b = connect()
    try:
        if a.action == 'launch': launch(c, b, a.campaign, a.quick)
        else: status(c, b, a.run_id, a.action == 'fetch')
    finally: c.close()


if __name__ == '__main__':
    main()
