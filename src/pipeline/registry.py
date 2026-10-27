"""The declared pipeline: which stage produces which artifact, and what each one costs.

This is the single place where the dependency structure of the thesis is written down. Every
number in the manuscript traces to an artifact under reports/bay_twin/runs/; every artifact traces
to exactly one producer here; every producer belongs to exactly one stage. The provenance checker
reads this registry, so a table citing an artifact no producer claims is reported as an error
rather than going unnoticed. Every figure the thesis includes is declared here too, as an artifact
of the stage whose producer writes it (src/pipeline/declarations.py checks that the producer's
source names it).

Artifact paths are relative to reports/bay_twin/, as the thesis cites them, except for two
repository trees that live elsewhere and keep their repository path: `results/` (the Optuna study
folders a campaign writes at the repository root) and `thesis_template/` (the figures).

Tiers. `full` runs the published protocol, which means at least 100 completed Optuna trials per
arm and several hours on a cluster for the heavy stages. `smoke` runs a reduced protocol for
checking that the code executes. A smoke run is never allowed to overwrite a release artifact,
because a figure tuned at ten trials is not the figure the thesis reports.
"""
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
BAY = ROOT / 'reports/bay_twin'
#: Artifact keys with these prefixes are repository paths; every other key is under reports/bay_twin.
ROOT_PREFIXES = ('results/', 'thesis_template/')
FIGURES = 'thesis_template/images/figures/'


def artifact_path(artifact):
    """The file or folder an artifact key names."""
    key = artifact.rstrip('/')
    return ROOT / key if key.startswith(ROOT_PREFIXES) else BAY / key


@dataclass(frozen=True)
class Stage:
    name: str
    summary: str
    #: dotted module paths run with `python -m`, in order
    producers: tuple
    #: artifact keys (see artifact_path): relative to reports/bay_twin/, or results/ and
    #: thesis_template/ paths from the repository root
    artifacts: tuple
    #: stages that must have run first
    depends_on: tuple = ()
    #: rough wall-clock at tier `full`, for the plan printout
    minutes: int = 1
    #: True when the published protocol needs the cluster
    needs_cluster: bool = False
    #: True when a smoke-tier run would produce a misleading artifact
    release_only: bool = False
    notes: str = ''


STAGES = (
    Stage(
        name='data',
        summary='Load the raw exports and casts, apply quality control, derive the sensor geometry '
                'and the Copernicus boundary context, and emit the cohort inventory.',
        producers=('src.data.bay_observations',
                   'scripts.build_bay_geometry',
                   'scripts.build_bay_boundary_context',
                   'src.pipeline.stage_data',
                   'scripts.bay_twin_diagnostics',
                   # the bathymetry and cast-archive audits Chapter 3 cites: the MaGIC grids'
                   # coverage of the network, the EMODnet DTM against the surveyed depths, and the
                   # Gen. 2 cast inventory (each reads sensor_geometry.csv or the cast archive)
                   'scripts.audit_magic_coverage',
                   'scripts.v2_emodnet_check',
                   'scripts.audit_gen2_profiles',
                   # contiguous missing-hour runs in the delivered record (descriptive)
                   'scripts.v2_gap_durations'),
        artifacts=('data_audit.json',
                   'sensor_inventory.csv',
                   'hourly_observations.csv.gz',
                   'bathymetry_bands.geojson',
                   'sensor_geometry.csv',
                   'geometry.npz',
                   'geometry_provenance.json',
                   FIGURES + 'fig_bay_sensor_geometry.png',
                   'profile_boundary_context.csv.gz',
                   'profile_boundary_provenance.json',
                   'runs/pipeline_data_summary.json',
                   'quality_diagnostics.json',
                   'colocated_probe_diagnostics.json',
                   'copernicus_local_inventory.json',
                   'temperature_availability.png',
                   'magic_coverage.json',
                   'magic_sensor_coverage.csv',
                   'runs/v2_emodnet_check.json',
                   'gen2_profile_audit.json',
                   'gen2_profile_inventory.csv',
                   'runs/v2_gap_durations.json'),
        minutes=14,
        notes='Deterministic, to the byte: src.pipeline.stage_data rewrites the hourly file\'s gzip '
              'header without a write time or a file name, which pandas would otherwise stamp into '
              'it. Everything downstream assumes this has run. The boundary context reads '
              'data/external/Copernicus/T2023_2026 and takes a few minutes; it leaves the '
              'published file untouched when the table it computes is identical.',
    ),
    Stage(
        name='metrology',
        summary='The measurement floors: oxygen reproducibility, co-located agreement by depth, '
                'and the variance-component budget.',
        producers=('src.experiments.v2_oxygen_floor',
                   'src.experiments.v2_colocated_depth',
                   'src.experiments.v2_variance_components',
                   'src.experiments.v2_biofouling',
                   'src.experiments.v2_logbook',
                   'src.experiments.v2_fouling_response',
                   'src.experiments.v2_fouling_learned',
                   # addendum 56: a clean-state reference model for fouling (2026-09-29, Trinity run)
                   'src.experiments.v2_fouling_reference',
                   # addendum 57: the jump at a cleaning by interval length, and the casts (Trinity run)
                   'src.experiments.v2_fouling_jump',
                   # addendum 58: the probe's response to sunlight and temperature by days since cleaning
                   'src.experiments.v2_fouling_diel',
                   # the CTD-side floor numbers of Chapter 3 (2026-09-26)
                   'scripts.v2_ctd_floor',
                   # the Garcia-Gordon coefficient set the code uses, checked against the
                   # article's table (Chapter 2; 2026-09-26)
                   'scripts.solubility_coefficient_check',
                   # the factor that states oxygen in mg/l: 0.0319988 x rho/1000 per hour
                   'scripts.v2_oxygen_units'),
        artifacts=('runs/v2_oxygen_floor.json',
                   'runs/v2_colocated_depth.json',
                   'runs/v2_variance_components.json',
                   'runs/v2_biofouling.json',
                   'runs/v2_logbook.json',
                   'runs/v2_fouling_response.json',
                   'runs/v2_fouling_learned.json',
                   'runs/v2_fouling_reference.json',
                   'runs/v2_fouling_reference',
                   'runs/v2_fouling_jump.json',
                   'runs/v2_fouling_jump',
                   'runs/v2_fouling_diel.json',
                   'runs/v2_fouling_diel',
                   'runs/v2_ctd_floor.json',
                   'runs/solubility_coefficients.json',
                   'runs/v2_oxygen_units.json',
                   'runs/v2_oxygen_units'),
        depends_on=('data',),
        minutes=6,
        notes='No model is fitted. These set the interpretive floor for every later contrast, so '
              'they are run before anything that would be read against them.',
    ),
    Stage(
        name='spatial',
        summary='The held-station field arms, their naive comparators, the CTD vertical prior and '
                'its matched non-physical placebo.',
        producers=('src.experiments.v2_field',
                   'src.experiments.v2_field_naive',
                   'src.experiments.v2_field_ctdprior',
                   'src.experiments.v2_field_intercept',
                   'src.experiments.v2_field_placebo',
                   'scripts.v2_background_bias',
                   'scripts.v2_field_contrasts',
                   'scripts.v2_field_tables'),
        artifacts=('runs/v2_field_naive/field_naive_summary.json',
                   'runs/v2_field_contrasts.json',
                   # addendum 69 (review R3): the comparators of the per-fold model
                   # (v2_field_naive --perfold) and every contrast with them and with the cast prior
                   # rebuilt per inner split (v2_field_ctdprior --inner-prior); v2_field_contrasts
                   # writes the second by default (scripts.v2_oxygen_mgl_field_contrasts --a69 too)
                   'runs/v2_field_naive/perfold',
                   'runs/v2_field_contrasts_a69.json',
                   'results/v2_field_held_site_A6_ctdprior_direct_innerprior',
                   'results/v2_field_held_site_A7_ctdprior_direct_innerprior',
                   'runs/v2_background_bias.json',
                   # the Optuna study folders the campaign writes at the repository root; the CTD
                   # prior and placebo tables cite them beside the contrasts
                   'results/v2_field_held_site_A6_ctdprior_direct',
                   'results/v2_field_held_site_A7_ctdprior_direct',
                   'results/v2_field_held_site_P1_placebo_direct',
                   # addendum 71: the fitted depth control with its basis fitted inside each inner split
                   'results/v2_field_held_site_P1_placebo_direct_innerprior',
                   'results/v2_field_held_site_P2_placebo_direct'),
        depends_on=('data', 'metrology'),
        minutes=240,
        needs_cluster=True,
        release_only=True,
        notes='One Optuna study per arm, never shared. The triple-nested macro rule is applied '
              'inside bay_profile_field.aggregate; a pooled mean does not reproduce these figures.',
    ),
    Stage(
        name='forecast',
        summary='Six-hour forecasting: the horizon sweep, site-level holdout, issuance-phase '
                'sweep, AOU reparameterisation and the exogenous ablations.',
        producers=('src.experiments.v2_forecast',
                   'src.experiments.v2_horizon',
                   'src.experiments.v2_held_station',
                   'src.experiments.v2_phase_shift',
                   'scripts.v2_phase24_table',
                   'src.experiments.v2_phase_operator',
                   'src.experiments.v2_aou_forecast',
                   'src.experiments.v2_bathy',
                   'scripts.v2_bathy_compare',
                   'src.experiments.v2_sequence_arms',
                   # the sequence-arm table's baselines on its own endpoints (2026-09-26);
                   # writes runs/v2_sequence_arms/sequence_baselines.json
                   'scripts.v2_sequence_baselines',
                   'scripts.v2_sequence_table',
                   'src.experiments.v2_river',
                   'src.experiments.v2_river_lag',
                   'scripts.v2_riverlag_table',
                   'scripts.v2_transfer_intervals',
                   'scripts.v2_phase_tables',
                   'scripts.v2_seed_contrasts',
                   # addendum 44: the river contrast over six phases and three seeds
                   'src.experiments.v2_phase_river',
                   'scripts.v2_phase_river_table',
                   # addendum 45 part A: weather plus river against the weather tree, swept
                   'src.experiments.v2_phase_weather_river',
                   # addendum 78: the six-hour river contrast on both variables over six phases
                   # and three seeds (2026-10-07, cluster; the table runs locally)
                   'src.experiments.v2_phase_river6',
                   'scripts.v2_phase_river6_table',
                   # addenda 49 and 51: the recurrent arm on a calendar split, and the sequence
                   # comparison over six issuance phases and three seeds (2026-09-27, cluster)
                   'src.experiments.v2_sequence_sweep',
                   'scripts.v2_sequence_sweep_contrasts',
                   # addendum 64: quantile forecasts, anticipated change, a forecast-based decline
                   # alert and the astronomical tide (2026-10-04, cluster; the report runs locally)
                   'src.experiments.v2_probabilistic',
                   'src.experiments.v2_probabilistic_report',
                   # addendum 65: the station's record after issuance as a perfect weather forecast, to
                   # 72 h (2026-10-05, cluster; the report runs locally)
                   'src.experiments.v2_weather_oracle',
                   'src.experiments.v2_weather_oracle_report',
                   # addendum 80: the Mitili Est currents as a perfect input on the same trees
                   # (2026-10-07, cluster; the report runs locally)
                   'src.experiments.v2_currents_oracle',
                   'src.experiments.v2_currents_oracle_report',
                   # addendum 66: the ARIMA family to 72 h, its hybrid with a tree on the residuals,
                   # a joint oxygen and temperature variant and GARCH-type bands (2026-10-05,
                   # cluster; the report runs locally)
                   'src.experiments.v2_arima',
                   'src.experiments.v2_arima_report',
                   # addendum 67: state-space ARIMA (orders by a nested study and by auto_arima) and
                   # N-BEATS, to 72 h (2026-10-05, cluster; the report runs locally)
                   'src.experiments.v2_statespace',
                   'src.experiments.v2_statespace_report',
                   # the ARIMA family's tables, and the state-space table once its report exists
                   'scripts.v2_arima_tables',
                   # addendum 72: a Bayesian dynamic linear model and a Bayesian structural time series
                   # against the state-space ARIMA, to 72 h (2026-10-06, cluster; the report and its
                   # table run locally)
                   'src.experiments.v2_bayes_ts',
                   'src.experiments.v2_bayes_ts_report',
                   'scripts.v2_bayes_ts_table',
                   # addendum 70: the equal-weight combination of three forecasters of the distribution
                   # and conformal recalibration of the 80 % intervals (2026-10-05, local), and its table
                   'src.experiments.v2_ensemble',
                   'scripts.v2_ensemble_table',
                   'scripts.v2_probabilistic_tables'),
        artifacts=('runs/v2_horizon_summary.json',
                   'runs/v2_held_station',
                   'runs/v2_phase_shift',
                   'runs/v2_phase24_summary.json',
                   'runs/v2_aou_forecast',
                   'runs/v2_bathy_compare.json',
                   'runs/v2_sequence_arms',
                   'runs/v2_river_summary.json',
                   'runs/v2_riverlag_summary.json',
                   'runs/v2_transfer_intervals.json',
                   'runs/v2_phase_matched.json',
                   'runs/v2_seed_contrasts.json',
                   'runs/v2_phase_river.json',
                   'runs/v2_phase_weather_river.json',
                   'runs/v2_phase_river6.json',
                   'runs/v2_sequence_sweep',
                   'runs/v2_sequence_sweep.json',
                   # addendum 69: the sweep with the tree selected per outer fold
                   # (v2_sequence_sweep panel-perfold, then scripts.v2_sequence_sweep_contrasts)
                   'runs/v2_sequence_sweep_a69.json',
                   'runs/v2_probabilistic',
                   'runs/v2_weather_oracle',
                   'runs/v2_currents_oracle.json',
                   'runs/v2_arima',
                   'runs/v2_statespace',
                   'runs/v2_bayes_ts',
                   'runs/v2_ensemble'),
        depends_on=('data', 'metrology'),
        minutes=420,
        needs_cluster=True,
        release_only=True,
        notes='Hold out sites, not probes, and mask 250 m: six probe pairs sit at 0.00 m and a '
              'probe-level split understates the transfer penalty threefold. A 6 h effect must '
              'also be checked at another issuance offset before it is called a horizon property.',
    ),
    Stage(
        name='perfold',
        summary='Addenda 47 to 50: every tree, ridge and graph arm re-selected with one study per '
                'outer fold on that fold\'s inner data only, against the shared published design '
                're-run in the same software; XGBoost against LightGBM over issuance phase and seed.',
        producers=('src.experiments.v2_perfold',
                   'scripts.v2_perfold_contrasts',
                   'scripts.v2_perfold_table',
                   # addendum 48: the remaining arms (2026-09-27)
                   'src.experiments.v2_perfold_rest',
                   'scripts.v2_perfold_rest_contrasts',
                   'scripts.v2_perfold_rest_table',
                   # addendum 61: the two Arno arms with the lag searched up to seven days, shared
                   # and per-fold designs, against the same arms up to four days (2026-10-02)
                   'src.experiments.v2_arno_lag7',
                   'scripts.v2_arno_lag7_contrasts',
                   # addendum 50: the library comparison swept (2026-09-27)
                   'src.experiments.v2_xgb_sweep',
                   'scripts.v2_xgb_sweep_contrasts',
                   'scripts.v2_xgb_sweep_table',
                   # addendum 49: the graph arms and the tree at every lead (2026-09-27, cluster)
                   'src.experiments.v2_perfold_neural',
                   'scripts.v2_perfold_neural_contrasts',
                   'scripts.v2_perfold_neural_table',
                   # addendum 60 part B: seven published oxygen configurations refitted in mg/l
                   # (B1, B4, B7 here; B2, B3, B5, B6 on the cluster, through
                   # scripts.experiments.v2_oxygen_equivalence_remote), against c x published
                   'scripts.v2_oxygen_equivalence',
                   # addendum 69 (review R3): the declared gates G1-G4 of the per-fold and inner-split
                   # runs (it also reads the operational and reserved stages' per-fold outputs)
                   'scripts.v2_a69_gates',
                   # the cast prior's gain by half-year, from the stored per-fold predictions (review R3)
                   'scripts.v2_prior_halfyear'),
        artifacts=('runs/v2_perfold',
                   'runs/v2_perfold.json',
                   'runs/v2_perfold_rest',
                   'runs/v2_perfold_rest.json',
                   # addendum 69: the Field-57 sections with the cast prior rebuilt per inner split,
                   # written by scripts.v2_oxygen_mgl_field_contrasts --a69-perfold (with their mg/l
                   # counterparts in runs/v2_oxygen_mgl/C1/); the recurrent block re-read on the rows
                   # with the tree selected per outer fold (v2_perfold_neural_contrasts --recurrent-only)
                   'runs/v2_perfold_a69.json',
                   'runs/v2_perfold_rest_a69.json',
                   'runs/v2_perfold_neural_a69.json',
                   'runs/v2_a69_gates.json',
                   'runs/v2_prior_halfyear.json',
                   'runs/v2_arno_lag7',
                   'runs/v2_arno_lag7.json',
                   'runs/v2_xgb_sweep',
                   'runs/v2_xgb_sweep.json',
                   'runs/v2_perfold_neural',
                   'runs/v2_perfold_neural.json',
                   'runs/v2_oxygen_mgl/equivalence.json',
                   'runs/v2_oxygen_mgl/equivalence'),
        depends_on=('spatial', 'forecast'),
        minutes=1710,
        needs_cluster=True,
        release_only=True,
        notes='Gate G2 reads eight addendum-2 selection records copied byte-identical from the '
              'frozen repository into runs/20260920T083322Z_v2_horizon/results/; addendum 48 reads '
              'the published horizon (1, 3, 12, 36, 48 h), river, river-lag, ISPRA and seabed '
              'records copied the same way. Every XGBoost fit must run with V2_THREADS=2 '
              '(addendum 48 amendment 1): its row subsampling depends on the thread count. '
              'Addenda 47, 48 and 50 run locally; addendum 49 needs GPU 1 of the cluster (about 22 h '
              'on two slots) and reads the published neural horizon records copied from the frozen '
              'repository; its contrasts read runs/v2_seed_contrasts.json for the seed ranges.',
    ),
    Stage(
        name='reserved',
        summary='The reserved-period reading on 2026, scored once against the frozen protocol.',
        producers=('src.experiments.v2_reserved',
                   # prints the reserved-evaluation table's quarter rows from the reading's
                   # endpoints, under the table's one aggregation rule; writes nothing
                   'scripts.v2_reserved_table',
                   'src.experiments.v2_field_reserved2026',
                   'src.experiments.v2_field_reserved2026_copernicus',
                   # addendum 71 / second review F05: the released Copernicus configuration against the
                   # cast prior selected inside each inner split, into runs/v2_field_reserved2026/innerprior/
                   'scripts.v2_field2026_innerprior_copernicus',
                   'scripts.v2_field2026_table',
                   # addendum 68 (review R4): the background with each remote probe at its measured
                   # or nominal depth, on Field-57 and Field-2026 (2026-10-05, local)
                   'scripts.v2_probe_depth',
                   'src.experiments.v2_field_depth'),
        artifacts=('runs/v2_reserved_reading_v2',
                   'runs/v2_field_reserved2026',
                   'runs/v2_field_reserved2026_copernicus',
                   'runs/v2_probe_depth',
                   'runs/v2_field_depth'),
        depends_on=('spatial', 'forecast'),
        minutes=45,
        release_only=True,
        notes='Scored against config/bay_reserved_protocol_v2.json. Re-running this against a '
              'changed model is a new reading and needs its own protocol increment.',
    ),
    Stage(
        name='operational',
        summary='Closed-loop rollout through a dropout, warning-episode detection, outage '
                'recovery, gap recovery and the missingness analysis of the scored subset.',
        producers=('src.experiments.v2_closed_loop',
                   # addendum 5: output gates on the rollout, single dropout and blackout
                   'src.experiments.v2_physics_gate',
                   'scripts.experiments.exp_warning_curve',
                   # addendum 87: the rate rule's grid extended past its edges (rates to 15 umol/kg/h,
                   # cooldowns from 1 h), selected on development by the same rule; the operating point
                   # that the episode scoring, the decline table and the pH gate read
                   'scripts.v2_warning_grid_ext',
                   'src.experiments.v2_warning_episodes',
                   'src.experiments.v2_missingness',
                   # the deployed forecaster's availability on the reserved grid, then the
                   # missingness table that reads both artifacts (2026-09-26)
                   'scripts.v2_availability_deployed',
                   'scripts.v2_missingness_table',
                   # review R11: the released bands' width, interval score, coverage by probe and
                   # quarter and over weeks, and the all-history fallback (2026-10-05)
                   'scripts.v2_interval_calibration',
                   # the 2025 oxygen outage study (the outage-recovery table, panel B)
                   'src.experiments.v2_oxygen_memory',
                   'scripts.v2_summarize_oxygen_memory',
                   'scripts.v2_gap_feasibility',
                   'src.experiments.v2_gap_recovery',
                   'scripts.v2_gap_climatology',
                   'scripts.v2_gap_neighbours',
                   'scripts.v2_gap_table',
                   'scripts.v2_gap_splines',
                   'scripts.v2_gap_intervals',
                   'scripts.v2_gap_twin_rule',
                   # addendum 74 (2026-10-07): the twin's offset re-anchored at both edges of the gap
                   'scripts.v2_gap_twin_drift',
                   # addendum 75 (2026-10-07): the Gaussian process with its neighbours beyond a radius
                   # (fills on the cluster, campaign gapradius; inputs and reading local)
                   'scripts.v2_gap_neighbour_radius',
                   'scripts.v2_gap_long',
                   'scripts.v2_gap_variants',
                   'scripts.v2_gap_exogenous',
                   'scripts.v2_gap_chronos_covariates',
                   'scripts.v2_gap_exo2',
                   'scripts.v2_closed_loop_leads',
                   # addendum 76 (2026-10-07): the conservation constraint isolated under feedback
                   # (v2_closed_loop --perfold --graph-arm learned_source, then this reading)
                   'scripts.v2_closed_loop_constraint',
                   'scripts.v2_neighbour_fallback',
                   # (the closed-loop, continuation and 72 h tables read addendum 82's rollout, so they
                   # are written by the consolidation stage)
                   'scripts.v2_decline_types',
                   # addendum 62: the water-mass declines against the south-easterly wind component
                   # (2026-10-03)
                   'scripts.v2_scirocco_declines',
                   # descriptive reads of records the forecasting pipeline does not use, cited where
                   # the thesis discusses low oxygen and the declines: the SeapHOx at 16 m, the 2023
                   # low-oxygen hours, the 2 and 8 m thermistors, and pH as an alert gate
                   'scripts.eda_opportunities.eda_b_seaphox',
                   'scripts.eda_opportunities.eda_b2_lows',
                   'scripts.eda_opportunities.eda_c2_strat',
                   'scripts.eda_opportunities.eda_g_ph_gate',
                   'src.experiments.v2_bridge',
                   # addendum 45 part B: the learned continuation under two further sampler seeds
                   'src.experiments.v2_bridge_seed',
                   # addendum 63: the silent-probe continuation from 54 to 72 h (2026-10-03)
                   'src.experiments.v2_bridge72'),
        artifacts=('runs/v2_closed_loop',
                   'runs/v2_physics_gate',
                   'runs/v2_bridge',
                   'runs/v2_bridge_seeds.json',
                   'runs/v2_bridge72',
                   'runs/v2_closed_loop_leads.json',
                   # addendum 69: the rollout with every arm selected per outer fold
                   # (v2_closed_loop --perfold, into runs/v2_closed_loop/perfold/), re-read by lead
                   'runs/v2_closed_loop_leads_a69.json',
                   'runs/v2_neighbour_fallback.json',
                   'runs/v2_gap_neighbours',
                   'runs/v2_gap_neighbours.json',
                   'runs/v2_warning_curve',
                   'runs/v2_warning_grid_ext.json',
                   'runs/v2_warning_grid_ext',
                   'runs/v2_warning_episodes.json',
                   'runs/v2_decline_types.json',
                   'runs/v2_scirocco_declines.json',
                   'runs/eda_opportunities/eda_b_seaphox.json',
                   'runs/eda_opportunities/eda_b2_lows.json',
                   'runs/eda_opportunities/eda_c2_strat.json',
                   'runs/eda_opportunities/eda_g_ph_gate.json',
                   'runs/v2_missingness.json',
                   'runs/v2_interval_calibration.json',
                   'runs/v2_availability_deployed.json',
                   'results/v2_oxygen_memory',
                   'runs/v2_gap_feasibility.json',
                   'runs/v2_gap_climatology',
                   'runs/v2_gap_climatology.json',
                   'runs/v2_gap_recovery_summary.json',
                   'runs/v2_gap_splines.json',
                   'runs/v2_gap_intervals.json',
                   'runs/v2_gap_intervals_reserved2026.json',
                   'runs/v2_gap_intervals_seed20260923.json',
                   'runs/v2_gap_twin_rule.json',
                   'runs/v2_gap_twin_drift.json',
                   'runs/v2_gap_neighbour_radius.json',
                   'runs/v2_gap_neighbour_radius',
                   'runs/v2_closed_loop_constraint.json',
                   'runs/v2_closed_loop_learned_source',
                   'runs/v2_gap_long.json',
                   'runs/v2_gap_variants.json',
                   'runs/v2_gap_variants',
                   'runs/v2_gap_exogenous.json',
                   'runs/v2_gap_exogenous',
                   'runs/v2_gap_chronos_covariates.json',
                   'runs/v2_gap_chronos_covariates',
                   'runs/v2_gap_exo2.json',
                   FIGURES + 'fig_bay_gap_exo2.png'),
        depends_on=('reserved',),
        minutes=25,
        notes='Warning detection differences a uniform hourly grid. Differencing a gap-collapsed '
              'series makes shift(6) span six observations and manufactures declines.',
    ),
    Stage(
        name='analysis',
        summary='Re-analyses of frozen predictions: attribution, the sensor-axis bootstrap and the '
                'augmented-against-plain contrasts.',
        producers=('scripts.v2_sensor_axis',
                   'scripts.v2_shap_horizons',
                   # the sea-level and offshore-boundary arms against the weather tree, then every
                   # augmented arm against the plain tree (the physical-feature results)
                   'scripts.v2_ispra_table',
                   'scripts.v2_augmented_vs_plain',
                   'scripts.v2_bathy_pooled',
                   # the oracle tree/graph blend bound by horizon, and what the constant salinity of
                   # the utilisation arms costs
                   'scripts.v2_horizon_blend',
                   'scripts.v2_salinity_check',
                   'scripts.v2_aggregation_rule',
                   'scripts.v2_horizon_reserved_cells',
                   'scripts.experiments.exp_placement',
                   # the reserved-period ledger behind the reserved-evaluation table
                   'scripts.v2_reserved_ledger',
                   # the AOU and near-bottom summaries, the AOU re-read in both unit
                   # conventions, and the Section 5.4-5.5.1 numbers that had no producer
                   'scripts.v2_summarize_aou',
                   'scripts.v2_summarize_bottom',
                   'scripts.v2_aou_units',
                   'scripts.v2_section5_numbers'),
        artifacts=('runs/v2_sensor_axis.json',
                   'runs/v2_horizon_reserved_cells.json',
                   'runs/v2_shap_horizons.json',
                   'runs/v2_ispra_summary.json',
                   'runs/v2_augmented_vs_plain.json',
                   'runs/v2_bathy_pooled.json',
                   'runs/v2_horizon_blend.json',
                   'runs/v2_salinity_check.json',
                   'runs/v2_aggregation_rule.json',
                   'runs/v2_placement',
                   'runs/v2_reserved_ledger.json',
                   'runs/v2_bottom_aou/aou_summary.json',
                   'runs/v2_bottom_aou/bottom/contrasts.json',
                   # the near-bottom arms mapped back at what their predictors can read (2026-09-27)
                   'runs/v2_bottom_aou/bottom_ref',
                   'runs/v2_bottom_aou/aou_units',
                   'runs/v2_section5_numbers.json'),
        depends_on=('forecast', 'reserved'),
        minutes=15,
        notes='Report every augmented arm against the plain tree as well as the arm it augments; '
              'a weak base arm turns a null feature into a large false gain.',
    ),
    Stage(
        name='pretrained',
        summary='Zero-shot, fine-tuned and low-rank-adapted pretrained forecasters, with and '
                'without the AOU target reparameterisation.',
        producers=('scripts.experiments.v2_chronos',
                   'scripts.experiments.v2_chronos2',
                   'scripts.experiments.v2_chronos_lora',
                   'scripts.v2_chronos_lora_contrasts',
                   'scripts.v2_chronos_adapt_inputs',
                   'scripts.experiments.v2_chronos_adapt',
                   'scripts.v2_chronos_adapt_contrasts',
                   'scripts.v2_chronos_adapt_table',
                   'scripts.experiments.v2_chronos_timeval',
                   'scripts.v2_chronos_timeval_contrasts',
                   'scripts.experiments.v2_chronos_refit',
                   'scripts.v2_chronos_refit_contrasts',
                   # addendum 77 (2026-10-07): weight interpolation between zero-shot and each addendum-42
                   # selection (cluster, GPU 1, via scripts.v2_t1_remote), read locally
                   'scripts.experiments.v2_chronos_wiseft',
                   'scripts.v2_chronos_wiseft_contrasts',
                   # addendum 46: how much of its own past a continuation needs (Chronos context
                   # length on the cluster, the anchored continuation's own-history window locally)
                   'scripts.v2_lookback_inputs',
                   'scripts.experiments.v2_chronos_lookback',
                   'src.experiments.v2_bridge_lookback',
                   'scripts.v2_lookback_contrasts',
                   'scripts.v2_lookback_table',
                   # addendum 52: the utilisation arms with saturation in umol/L (2026-09-27, cluster)
                   'scripts.experiments.v2_aou_units_chronos',
                   'scripts.v2_aou_units_chronos_contrasts',
                   # addendum 53: the holes in Chronos-Bolt's context bridged by Chronos-2 (2026-09-28,
                   # contexts built locally, Chronos on the cluster, scored locally)
                   'scripts.v2_context_holes_inputs',
                   'scripts.experiments.v2_context_holes',
                   'scripts.v2_context_holes',
                   # addendum 54: the sonde's own temperature, observed or filled, as a Chronos-2
                   # covariate for oxygen gaps and forecasts (2026-09-29, inputs local, cluster run)
                   'scripts.v2_o2_temperature_inputs',
                   'scripts.experiments.v2_o2_temperature',
                   'scripts.v2_o2_temperature',
                   # addendum 55: the same temperature as features of the tuned LightGBM gap filler
                   # and forecaster (2026-09-29, cluster run, temperature fills from addendum 54)
                   'scripts.experiments.v2_o2_temperature_lgbm',
                   'scripts.v2_o2_temperature_lgbm'),
        artifacts=('runs/battery/v2_chronos',
                   'runs/v2_chronos_lora',
                   'runs/v2_chronos_adapt',
                   'runs/v2_chronos_timeval',
                   'runs/v2_chronos_refit',
                   'runs/v2_chronos_wiseft',
                   'runs/v2_lookback',
                   'runs/v2_lookback.json',
                   'runs/v2_aou_units_chronos',
                   'runs/v2_aou_units_chronos.json',
                   'runs/v2_context_holes',
                   'runs/v2_context_holes.json',
                   'runs/v2_o2_temperature',
                   'runs/v2_o2_temperature.json',
                   'runs/v2_o2_temperature_lgbm',
                   'runs/v2_o2_temperature_lgbm.json'),
        # the continuation arms of addenda 41 and 46 are scored on the addendum-39 rows
        depends_on=('data', 'forecast', 'operational'),
        minutes=180,
        needs_cluster=True,
        release_only=True,
        notes='The only part of this project that needs a GPU. Zero-shot Chronos-Bolt, on the raw '
              'series or as AOU, beats the climatological-increment baseline on oxygen, and it '
              'sits outside the supported pipeline.',
    ),
    Stage(
        name='fm_zoo',
        summary='Addendum 81: eight pretrained forecasters (Chronos-Bolt small and base, Chronos-2, '
                'TimesFM 2.5 and 3.0, on the probe\'s own record and, where a model accepts them, with '
                'covariates) on every task: six hours on Dev-2025 over six issuance grids and on '
                'Reserved-2026, oxygen as utilisation, the forecast distribution to 72 h at nine levels, '
                'the silent probe to 72 h and the four gap cohorts, each against that task\'s reference.',
        producers=(# inputs, built locally from the published code paths and gated against the
                   # published rows and contexts
                   'scripts.v2_fm_zoo_inputs_6h',
                   'scripts.v2_fm_zoo_inputs_dist',
                   'scripts.v2_fm_zoo_inputs_bridge',
                   'scripts.v2_fm_zoo_inputs_gaps',
                   # every arm on every input file (2026-10-11, cluster, GPU; uploaded, launched and
                   # fetched by scripts.v2_fm_zoo_remote); writes runs/v2_fm_zoo/preds
                   'scripts.experiments.v2_fm_zoo',
                   # gated and scored locally, task by task (the cross-task summary with
                   # false-discovery control and the tables are written by the consolidation stage,
                   # since addendum 82 adds Gap-2026's gaps of 1 to 3 h to that family)
                   'scripts.v2_fm_zoo_score_6h',
                   'scripts.v2_fm_zoo_score_dist_bridge',
                   'scripts.v2_fm_zoo_score_gaps',
                   # the contexts (observed values) and prediction arrays are git-ignored .npz; this
                   # binds every one by path, size and sha256
                   'scripts.v2_fm_zoo_npz_manifest'),
        artifacts=('runs/v2_fm_zoo/npz_manifest.json',
                   'runs/v2_fm_zoo/inputs/manifest_6h.json',
                   'runs/v2_fm_zoo/inputs/manifest_dist_bridge.json',
                   'runs/v2_fm_zoo/inputs/manifest_gaps.json',
                   'runs/v2_fm_zoo/score_6h.json',
                   'runs/v2_fm_zoo/score_dist_bridge.json',
                   'runs/v2_fm_zoo/score_gaps.json'),
        # the six-hour rows of the sweep and the battery, the distribution reports, the silent-probe
        # and gap rows, and the mg/l utilisation columns
        depends_on=('forecast', 'reserved', 'operational', 'pretrained', 'oxygen_mgl'),
        minutes=240,
        needs_cluster=True,
        release_only=True,
        notes='The pretrained models are applied without local training, so no study is run. The '
              'distribution reports of the forecast stage score nine levels (0.1 to 0.9) and keep the '
              'eleven-level score as a secondary block, which the distribution scorer reads for its '
              'reproduction gate. The arrays are git-ignored; runs/v2_fm_zoo/npz_manifest.json binds '
              'them by digest (--check verifies the files on disk).',
    ),
    Stage(
        name='consolidation',
        summary='Addendum 82: the remaining asymmetries closed. A, the silent probe\'s rollout and '
                'rolled graphs on Bridge-Dev and Bridge-2026 to 72 h, the hour-of-day increment as the '
                'printed reference; B, the exogenous feature families over six issuance grids and three '
                'sampler seeds; C, Gap-2026 at 1-3 h (every gap method and the eight pretrained arms) '
                'and PCHIP with an interval; D, forecasting at a held-out site per fold with site '
                'intervals, oxygen included; E, the graph operator, constraint and saturation-prior '
                'arms over six grids; F, one causal recalibration rule for every forecaster of the '
                'distribution; G, the pretrained arms on Dev-2025 utilisation over six grids and the '
                'utilisation tree applied frozen on 2026; H, the seabed features at a held-out site.',
        producers=(# A: the 2026 studies of the rollout's step tree and rolled graphs, the rollouts on
                   # both cohorts (cluster), then the scoring with every gate first, and the
                   # closed-loop, continuation and 72 h tables that read it
                   'src.experiments.v2_rollout_a82_studies',
                   'src.experiments.v2_rollout_a82',
                   'scripts.v2_rollout_a82_report',
                   'scripts.v2_closed_loop_table',
                   'scripts.v2_bridge_table',
                   'scripts.v2_bridge72_table',
                   # B: the feature-family arms on shifted grids and further seeds (cluster), the
                   # gates and contrasts, and the two tables
                   'src.experiments.v2_phase_family',
                   'scripts.v2_phase_family_contrasts',
                   'scripts.v2_phase_family_table',
                   'scripts.v2_feature_families_table',
                   # C: the gap arms on Gap-2026 at 1-3 h applied frozen (cluster), the reproduction
                   # gates, the published gap scorers rerun on the extended rows, the eight
                   # pretrained arms on the same rows, and PCHIP with its paired interval
                   'src.experiments.v2_gap_short2026',
                   'scripts.v2_gap_short2026_gate',
                   'scripts.v2_gap_pchip',
                   'scripts.v2_gap_short2026_fmzoo',
                   'scripts.v2_gap_short2026_score',
                   # D and H: the local reproduction gates, the held-site studies (cluster), the
                   # transfer contrasts per fold with site intervals, and the seabed at a held site
                   'scripts.v2_add82_DH_gates',
                   'src.experiments.v2_held_site_nested',
                   'scripts.v2_transfer_perfold',
                   'scripts.v2_seabed_site',
                   # E: the graph arms on shifted grids (cluster, GPU), contrasts and the table
                   'src.experiments.v2_phase_graph',
                   'scripts.v2_phase_graph_contrasts',
                   'scripts.v2_phase_graph_table',
                   # F: the causal recalibration of every forecaster of the distribution (local)
                   'src.experiments.v2_recalibration',
                   # G: the utilisation inputs, the pretrained arms on them (cluster, GPU), the 2026
                   # utilisation tree applied frozen, and the reading in both cohorts
                   'scripts.v2_fm_zoo_inputs_aou_dev',
                   'scripts.experiments.v2_add82_g_run',
                   'scripts.v2_aou_frozen2026',
                   'scripts.v2_aou_symmetric_score',
                   # the pretrained arms' cross-task summary, its false-discovery family now holding
                   # part C's gaps of 1 to 3 h, and the thesis and supplement tables
                   'scripts.v2_fm_zoo_summary',
                   'scripts.v2_fm_zoo_table',
                   # the git-ignored .npz of parts C and G bound by digest
                   'scripts.v2_add82_npz_manifest'),
        artifacts=('runs/v2_addendum82/A', 'runs/v2_rollout_a82.json', 'runs/v2_addendum82_A.json',
                   'runs/v2_addendum82/B', 'runs/v2_addendum82_B.json',
                   'runs/v2_addendum82/C', 'runs/v2_addendum82_C.json',
                   'runs/v2_addendum82/D', 'runs/v2_addendum82_D.json',
                   'runs/v2_addendum82/E', 'runs/v2_addendum82_E.json',
                   'runs/v2_addendum82/F', 'runs/v2_addendum82_F.json',
                   'runs/v2_addendum82/G', 'runs/v2_addendum82_G.json',
                   'runs/v2_addendum82/H', 'runs/v2_addendum82_H.json',
                   'runs/v2_addendum82/remote',
                   'runs/v2_addendum82/gates_local.json',
                   'runs/v2_addendum82/npz_manifest.json',
                   'runs/v2_fm_zoo.json'),
        # the published rows, studies and predictions every part reruns or reads: the horizon
        # study and its sweeps, the reserved grid, the continuation and gap rows, the per-fold
        # selections, the pretrained arms and the mg/l reruns
        depends_on=('forecast', 'reserved', 'operational', 'perfold', 'analysis', 'pretrained',
                    'oxygen_mgl', 'fm_zoo'),
        minutes=600,
        needs_cluster=True,
        release_only=True,
        notes='One declaration (config/bay_reserved_protocol_v2_addendum82.json) and its amendment, '
              'before any part was scored. Every part reproduces the published numbers it extends '
              'before reading anything new, and nothing published is overwritten: part A writes its '
              'rollouts under runs/v2_addendum82/A, part C rescores the published gap scorers into '
              'runs/v2_addendum82/C/rescored. scripts.v2_fdr reads the tables this stage writes, so '
              'the figures stage runs after it. The .npz of parts C and G are git-ignored and bound '
              'by runs/v2_addendum82/npz_manifest.json.',
    ),
    Stage(
        name='feature_leads',
        summary='Addendum 83: every physical-feature family in the forecasting tree (the weather '
                'channels, the Magra level alone and beside the weather, sea level and water '
                'temperature from the ISPRA station, the seabed) at every lead of 1, 3, 6, 12, 24, '
                '36, 48 and 72 h, on both variables, over six issuance grids and three sampler seeds '
                'and with every arm selected per outer fold, against the plain and the weather tree.',
        producers=(# the plain and weather trees and the five feature arms at every (variable, lead,
                   # issuance phase, sampler seed) cell on the cluster, and the per-fold studies (the
                   # ISPRA arms on the cluster, the rest locally; scripts.launchers.add83_*)
                   'src.experiments.v2_lead_family',
                   # gates L-G0 to L-G9 before any contrast, the paired contrasts, both verdicts, the
                   # floor flags and the false-discovery control over the addendum's family
                   'scripts.v2_lead_family_contrasts',
                   # the supplement table; Chapter 5 draws the figure (figures stage)
                   'scripts.v2_lead_family_table'),
        artifacts=('runs/v2_addendum83', 'runs/v2_addendum83.json'),
        # the horizon study's builds and folds, the reused phase-sweep trees, the per-fold
        # selections and reports, the river, ISPRA and seabed designs, and addendum 82 part B's
        # trees and verdicts at the leads it swept
        depends_on=('forecast', 'perfold', 'analysis', 'consolidation'),
        minutes=120,
        needs_cluster=True,
        release_only=True,
        notes='One declaration (config/bay_reserved_protocol_v2_addendum83.json) and its amendment, '
              'before any study ran. Development folds only: no 2026 value is loaded. The trees '
              'and arms already published at a lead are reused, each hash-checked, and the plain '
              'and weather trees at 72 h are new. scripts.v2_fdr reads the table this stage '
              'writes, and the figure checks itself against it, so the figures stage runs after it.',
    ),
    Stage(
        name='gap_climanchor',
        summary='Addendum 84: the month-by-hour climatology anchored to both edges of a gap (linear '
                'interpolation plus the climatology\'s departure from its own chord), on the four gap '
                'cohorts and both variables, against linear interpolation, the plain climatology, the '
                'Kalman bridge, the incumbents of each band or length and the no-neighbour reading.',
        producers=(# gates A84-G1 to G4 (the published episodes rebuilt, no masked reading in a fill, the
                   # published climatology and Kalman-bridge contrasts reproduced), then the paired
                   # contrasts and the arm's per-row column
                   'scripts.v2_gap_climanchor',),
        artifacts=('runs/v2_gap_climanchor.json', 'runs/v2_gap_climanchor'),
        # the published gap runs, their per-row comparators and intervals (operational), and the
        # rescored variants, exogenous and no-neighbour summaries of addendum 82 part C (consolidation)
        depends_on=('operational', 'consolidation'),
        minutes=2,
        notes='One declaration (config/bay_reserved_protocol_v2_addendum84.json) before any code. '
              'Parameter-free and run locally; nothing published is read differently or overwritten. '
              'No table, figure or sentence of the thesis or the deck reads this artifact: the arm '
              'replaces no incumbent and changes no no-neighbour answer.',
    ),
    Stage(
        name='common_rows',
        summary='Addendum 85: every forecaster, the silent-probe arms and every gap method on common rows. '
                'A, every forecast issued at onset against the network-aided forecast and the other arms '
                'that read the network after onset on Bridge-Dev and Bridge-2026 to 72 h, with recursive '
                'minus direct at matched lead; B, every forecaster on identical Dev-2025 and Reserved-2026 '
                'rows at 1-72 h, with the plain and weather point trees on 2026 selected on the 2026 fold\'s '
                'own inner window; C, one six-hour Dev-2025 tree value (shared and per fold, reference grid '
                'and six grids); D, every gap method and pretrained gap arm per cohort and length against '
                'linear interpolation and the incumbent.',
        producers=(# gates A85-G1 to G4 (objective and predictions of the stored per-fold Dev-2025 trees,
                   # the frozen 24 h 2026 trees, addendum 82's six-hour study), then the 32 studies of
                   # the 2026 point trees and their 2026 predictions (`launch`)
                   'src.experiments.v2_common_rows',
                   # gates A85-G5 to G7, then parts A-D on stored predictions, and the family adjustment
                   'scripts.v2_common_rows_report',
                   # the method each task names: for every task, variable and band, the arm with the lowest
                   # mean ratio of its development error to the reference's among the arms scored in both
                   # cohorts, with its 2026 result read (never used to choose) against the reference; reads
                   # stored scores and contrasts only and also the station task's Field-57 and Field-2026
                   # files (2026-10-20). The figures below read it, so it runs first
                   'scripts.v2_best_by_task',
                   # Appendix A's table of both winners per task band, from the file above only
                   'scripts.v2_best_by_task_table',
                   # three Chapter 5 figures from parts B, A and D, each arm on its cell's common rows and
                   # every choice made on the development cohorts, with the facts a paragraph cites
                   # (figures.json) and the digests of its sources (figures_manifest.json)
                   'scripts.v2_common_rows_figures',),
        artifacts=('runs/v2_addendum85.json', 'runs/v2_addendum85',
                   'runs/v2_best_by_task.json',
                   'runs/v2_addendum85/figures.json',
                   FIGURES + 'fig_bay_forecast_leads.png',
                   FIGURES + 'fig_bay_silent_probe.png',
                   FIGURES + 'fig_bay_gap_common.png'),
        # the horizon, distribution and pretrained forecasts (forecast, fm_zoo), the per-fold trees and
        # graphs (perfold, feature_leads), the 24 h 2026 reading and the battery (reserved), the Bridge
        # reports and gap runs (operational), the rollout and the rescored gap rows (consolidation)
        # (v2_best_by_task also reads the cast floors and units of metrology and the mg/l copies of oxygen_mgl)
        depends_on=('metrology', 'forecast', 'perfold', 'reserved', 'operational', 'fm_zoo', 'consolidation',
                    'feature_leads', 'oxygen_mgl'),
        minutes=60,
        release_only=True,
        notes='One declaration (config/bay_reserved_protocol_v2_addendum85.json) before any code. The 2026 '
              'studies ran locally, four at a time at two threads; the reading fits nothing. The deck\'s '
              'slides 6 and 7 draw parts B and A (presentation/make_figures_v2.py forecast_leads, '
              'silent_probe), and slide 8 part D through the gap part of figures.json (gap_recovery). scripts.v2_common_rows_figures draws the thesis figures of parts B, A and D '
              '(fig_bay_forecast_leads, fig_bay_silent_probe, fig_bay_gap_common) and refuses unless each '
              'part\'s gates passed and every drawn arm is on its cell\'s common rows; it reads the rescored '
              'no-neighbour grouping of stage consolidation.',
    ),
    Stage(
        name='figures',
        summary='Regenerate every figure in the manuscript from the artifacts above.',
        producers=('scripts.v2_figures',
                   # the issuance-grid, sampler-seed and per-fold sweeps of Dev-2025, and the
                   # per-fold shift of every quoted contrast (2026-10-01)
                   'scripts.v2_robustness_figures',
                   'scripts.v2_timeseries_figures',
                   'scripts.v2_forecast_figures',
                   'scripts.v2_gap_figures',
                   'scripts.v2_gap_traces',
                   'scripts.v2_gap_long_traces',
                   'scripts.v2_gap_tables',
                   # a probe with no neighbour: the gap methods grouped by what they read, the best
                   # of each paired on the published rows (2026-09-28)
                   'scripts.v2_gap_isolated',
                   # the supervisors' time-depth sections from the casts, and the average day by
                   # season at the moorings that the casts cannot see (descriptive, 2026-09-25)
                   'scripts.v2_ctd_sections',
                   'scripts.v2_daily_cycle',
                   # the inner bay's surface against the offshore column at PBoaT, from the casts
                   # (descriptive, raised by the supervisors on 2026-10-02)
                   'scripts.v2_bay_offshore',
                   # addendum 59: neighbour availability over masked and real gaps, and the
                   # neighbour-reading arms stratified by it (2026-10-01)
                   'scripts.v2_gap_neighbour_availability',
                   # addendum 59 part G: measured probe depths, the sonde oxygen adjustment and the
                   # temperature offsets against casts; writes table_bay_probe_key (2026-10-01)
                   'scripts.v2_metrology_review',
                   # addendum 59 part F: near-bottom coverage, paired placement, episode-level
                   # warning leads (2026-10-01)
                   'scripts.v2_observability_reanalysis',
                   # addendum 64: a forecast of the distribution around the largest well-observed
                   # 2026 decline (2026-10-04)
                   'scripts.v2_probabilistic_figure',
                   # addendum 65: forecasts of the distribution by lead, with a perfect weather
                   # forecast and the anticipated share of the largest falls (2026-10-05)
                   'scripts.v2_weather_oracle_figure',
                   # addendum 83: every physical-feature family at every lead from 1 to 72 h, drawn
                   # after checking each verdict, base ratio and value against the summary and the
                   # generated table (2026-10-13)
                   'scripts.v2_lead_family_figure',
                   # addendum 79: false-discovery control over Table 6.1's estimates and over every
                   # interval in the tables (2026-10-07); it reads the summary table's row builders,
                   # and the table then marks what it does not retain, so it runs first
                   'scripts.v2_fdr',
                   # the Chapter 6 summary table, read from the artifacts of every stage (2026-09-27)
                   'scripts.v2_summary_table'),
        artifacts=('runs/v2_gap_traces.json', 'runs/v2_gap_long_traces.json',
                   'runs/v2_gap_isolated.json',
                   'runs/v2_ctd_sections.json', 'runs/v2_daily_cycle.json', 'runs/v2_bay_offshore.json',
                   'runs/v2_gap_neighbour_availability.json', 'runs/v2_metrology_review.json',
                   'runs/v2_observability_reanalysis.json', 'runs/v2_fdr.json',
                   FIGURES + 'fig_bay_horizon_contrasts.png',
                   FIGURES + 'fig_bay_sweep_robustness.png',
                   FIGURES + 'fig_bay_perfold_shift.png',
                   FIGURES + 'fig_bay_copernicus_geometry.png',
                   FIGURES + 'fig_bay_forecast_trace.png',
                   FIGURES + 'fig_bay_saturation_trace.png',
                   FIGURES + 'fig_bay_gap_bridging.png',
                   FIGURES + 'fig_bay_gap_recovery.png',
                   FIGURES + 'fig_bay_gap_traces_oxygen.png',
                   FIGURES + 'fig_bay_gap_traces_temperature.png',
                   FIGURES + 'fig_bay_gap_long_traces_oxygen.png',
                   FIGURES + 'fig_bay_gap_long_traces_temperature.png',
                   FIGURES + 'fig_bay_ctd_sections.png',
                   FIGURES + 'fig_bay_daily_cycle.png',
                   FIGURES + 'fig_bay_quantile_fan.png',
                   FIGURES + 'fig_bay_quantile_lead.png',
                   FIGURES + 'fig_bay_feature_leads.png'),
        # (common_rows: Table 6.1's per-task rows read runs/v2_best_by_task.json)
        depends_on=('metrology', 'operational', 'analysis', 'perfold', 'fm_zoo', 'consolidation',
                    'feature_leads', 'common_rows', 'guidance_2026'),
        minutes=4,
        notes='Palettes are validated for colour-vision deficiency; series also carry line styles '
              'so identity is never colour alone.',
    ),
    Stage(
        name='oxygen_mgl',
        summary='Addendum 60 part C: the analyses that mixed per-kilogram and per-litre oxygen, rerun '
                'at their published configurations with every oxygen input in mg/l by its own source '
                'unit, each first reproduced in the published unit, and every printed oxygen number '
                'set against c x published.',
        producers=(  # the conversion itself lives in scripts.v2_oxygen_mgl_common
                   'scripts.v2_oxygen_mgl_field',            # C1/C2: the Field-57 arms
                   'scripts.v2_oxygen_mgl_field_contrasts',  # C1/C2: comparators and contrasts
                   'scripts.v2_oxygen_mgl_field2026',        # C1/C2: Field-2026
                   'scripts.v2_oxygen_mgl_aou_forecast',     # C3: the utilisation tree, thermodynamic baseline
                   'scripts.v2_oxygen_mgl_bottom',           # C2/C3: near-bottom, the utilisation ridge
                   'scripts.v2_oxygen_mgl_casts',            # C2: floors, sections, placement, Section 5
                   'scripts.v2_oxygen_mgl_seaphox',          # C4: Copernicus against the Tinetto SeapHOx
                   # C5 (amendment 3): the moorings' daily cycle, saturation as one unit over another
                   'scripts.v2_oxygen_mgl_daily_cycle',
                   # C3: the pretrained arms, from the columns the cluster wrote
                   # (scripts.v2_oxygen_mgl_chronos_remote fetches them)
                   'scripts.v2_oxygen_mgl_chronos_contrasts',
                   'scripts.v2_oxygen_mgl_summary',
                   # addendum 71 part B: the same refits with the consistent Benson and Krause solubility
                   # (the pretrained arms on GPU 1 of the cluster, kind zeroshot-bk of the remote launcher)
                   'scripts.v2_solubility_sensitivity',
                   # MAPE beside the deck's headline errors, from the same rows and averaging rule as each
                   # MAE, which it recomputes first (reads the mg/l reruns above, so it sits in this stage)
                   'scripts.v2_mape',
                   # addendum 73 (2026-10-07, local): the cast archive since 2021 without moored context on
                   # Field-57, and spatial conformal intervals on Field-57 and Field-2026; both read the
                   # per-fold model, its mg/l rerun and the 2026 reading's held-station estimates
                   'src.experiments.v2_field_archive',
                   'src.experiments.v2_spatial_conformal'),
        artifacts=('runs/v2_oxygen_mgl/C1',
                   'runs/v2_oxygen_mgl/C2',
                   'runs/v2_oxygen_mgl/C3',
                   'runs/v2_oxygen_mgl/C4',
                   'runs/v2_oxygen_mgl/C5',
                   'runs/v2_oxygen_mgl/harmonised_summary.json',
                   'runs/v2_solubility_bk',
                   'runs/v2_solubility_sensitivity.json',
                   'runs/v2_mape.json',
                   'runs/v2_field_archive',
                   'runs/v2_spatial_conformal'),
        depends_on=('metrology', 'spatial', 'forecast', 'perfold', 'reserved', 'operational', 'analysis',
                    'pretrained'),
        minutes=150,
        needs_cluster=True,
        release_only=True,
        notes='No study is run: every refit uses the configuration its published study selected. '
              'Normalisation constants written in micromoles (the forecasting z-score, the Field-57 '
              'objective scale) are carried into mg/l by c. XGBoost reproduces only with four '
              'threads (the near-bottom year-boundary arm). The pretrained utilisation arms run on '
              'GPU 1 of the cluster (scripts.v2_oxygen_mgl_chronos_remote). Amendment 4: the sonde-only '
              'utilisation arms (forecasting tree, thermodynamic baseline, pretrained arms) read the sonde '
              'oxygen x c and their references are the published predictions x c; the first run, by the '
              'hourly factor, is kept in runs/v2_oxygen_mgl/C3/_hourly_factor (--hourly-factor).',
    ),
    Stage(
        name='station_figure',
        summary='The depth-offset contrasts of Section 5.1 drawn as one figure: each arm minus the arm without '
                'a depth offset, with its interval over cast dates, on Field-57 per fold and on Field-2026, '
                'temperature and oxygen.',
        producers=(# reads only stored contrasts and floors, and refuses unless each drawn value rounds to the
                   # number the Appendix A table (or Chapter 4's margins table) prints
                   'scripts.v2_depth_offset_figure',),
        artifacts=('runs/v2_depth_offset_figure.json',
                   FIGURES + 'fig_bay_depth_offset.png'),
        # the cast floors (metrology), the per-fold Field-57 arms (perfold), the Field-2026 reading
        # (reserved) and their mg/l reruns (oxygen_mgl)
        depends_on=('metrology', 'perfold', 'reserved', 'oxygen_mgl'),
        minutes=1,
        notes='Nothing is recomputed: the values file records every drawn value with its source and key, '
              'the printed form it matches and the digests of its sources.',
    ),
    Stage(
        name='overview_figure',
        summary='The opening figure of Chapter 1: the inner gulf with the probe sites, the cast stations and the '
                'Copernicus cells, beside each probe\'s daily share of hours with a reading.',
        producers=('scripts.v2_overview_figure',),
        artifacts=('runs/v2_overview_figure.json',
                   FIGURES + 'fig_bay_overview.png'),
        # the hourly record and the sensor geometry (data)
        depends_on=('data',),
        minutes=1,
        notes='Draws the deck\'s map (presentation/make_figures_v2.py: _draw_bay_map) at the thesis\'s text width '
              'with no label below 7 pt, and counts the hours with a reading in the delivered hourly record; '
              'nothing else is computed.',
    ),
    Stage(
        name='guidance_2026',
        summary='Addendum 86: each physical-guidance contrast the development cohorts report, scored on the 2026 '
                'test period with the arms frozen before 2026 (feature trees selected on the 2026 fold\'s inner '
                'window, graph arms refitted at their fold-2026 selections, the range bound on the 2026 '
                'blackout), and the development readings of the two contrasts the thesis scored on 2026 only.',
        producers=('src.experiments.v2_guidance2026',
                   'scripts.v2_guidance2026_report',
                   # the Appendix A table of every contrast on both periods
                   'scripts.v2_guidance2026_table'),
        artifacts=('runs/v2_addendum86.json',
                   'runs/v2_addendum86'),
        # B's trees and inner window (common_rows), the fold-2026 graph selections (consolidation), the
        # feature designs (feature_leads), the range gate (operational), the depth-curve placebo (spatial)
        depends_on=('metrology', 'spatial', 'perfold', 'reserved', 'operational', 'consolidation',
                    'feature_leads', 'common_rows', 'oxygen_mgl'),
        minutes=210,
        release_only=True,
        notes='One declaration (config/bay_reserved_protocol_v2_addendum86.json) before anything was scored; '
              'nine 100-trial studies and one graph study, run locally; the graph refits reproduce the '
              'published oxygen predictions only to 1e-5 umol/kg (temperature to 6e-7), as the declaration '
              'allows, so part C is host-dependent.',
    ),
)

STAGE_BY_NAME = {s.name: s for s in STAGES}

#: A thesis figure that is a copy of an artifact written elsewhere. The provenance check accepts
#: it only while it is byte-identical to its source and the source is claimed by a stage. The
#: availability heatmap is written by scripts/bay_twin_diagnostics.py to reports/bay_twin/ and
#: copied into the thesis; writing it straight to the figures folder would retire this entry.
FIGURE_COPIES = {
    FIGURES + 'fig_bay_temperature_availability.png': 'temperature_availability.png',
}

#: Artifacts a table may cite that no stage produces, with the reason. Anything cited and not
#: found here or in a stage is a provenance error.
EXTERNAL_ARTIFACTS = {
    'runs/20260919T144822Z_v2_forecast':
        'the Dev-2025 six-hour forecast campaign on the cluster (one 100-trial nested study per '
        'arm), fetched once; the controlled physics comparisons are summarised from it',
    'runs/20260919T145045Z_v2_field':
        'the Field-57 spatial campaign on the cluster (one study per arm), fetched once; the '
        'boundary ablation reads its per-arm reports',
    'runs/20260919T183751Z_v2_xgb':
        'gradient-boosting comparator, run once under the addendum-1 protocol',
    'runs/v2_release_v2':
        'the packaged release-v2 bundle (refit forecaster, seasonal stress tests, robust '
        'forecast), assembled once when release v2 was frozen; its builder is in the history of '
        'inrim2026/SmartBayDO',
    'runs/20260918_reserved_evaluation':
        'first reserved-period evaluation, retained for the spatial-CTD and warning tables and for '
        'the 2026 rows of the outage table (read through scripts/v2_neighbour_fallback.py)',
    'runs/20260917T184229Z_thermal_memory_pilot':
        'thermal-memory pilot from the pre-v2 campaign, cited by the outage-recovery table',
    'runs/v2_horizon_reserved_h24/reserved_h24_oxygen_summary.json':
        'the oxygen summary of the third reading of the reserved period at 24 h (addendum 3), run once on the cluster by '
        'src/experiments/v2_horizon_reserved.py through scripts/v2_campaign.py; '
        'scripts/v2_horizon_reserved_cells.py and scripts/v2_common_rows_report.py read it, and the '
        'physical-guidance section cites its oxygen summary',
    'runs/v2_liguria_bathymetry.json':
        'the check of the second Regione Liguria bathymetry export (shapefile, 23 polygons) against '
        'the probes, made once; its writer was not kept and this environment has no shapefile '
        'reader, so it is cited as a record and not regenerated',
}

#: Inputs a producer reads that this repository does not hold. The repository was trimmed for
#: publication on 2026-09-23, and these run folders stayed in the earlier working repository only;
#: they were found on 2026-09-26 by comparing each producer's named run folders with that copy.
#: `smartbay run` refuses these producers with this list instead of failing on a FileNotFoundError.
#: Copy a folder byte-identical to the same relative path and its entry stops applying. Many other
#: folders are partial here (logs, transport records and Optuna databases were not carried over);
#: the producers that read them were rerun in this tree (the gap producers, v2_field2026_table,
#: v2_transfer_intervals), so partial is not listed as a failure.
FROZEN_INPUTS = {
    'src.experiments.v2_reserved': ('runs/20260918_reserved_refit',),
    'src.experiments.v2_closed_loop': ('runs/20260920T133747Z_v2_horizon',),
    'scripts.v2_gap_neighbours': tuple(f'runs/20260922T{t}Z_v2_gaprecovery' for t in (
        '215308', '215328', '215413', '215431', '223901', '223918', '224158', '224215')),
    'scripts.v2_gap_exogenous': tuple(f'runs/{d}T{t}Z_v2_gaprecovery' for d, t in (
        ('20260922', '215308'), ('20260922', '215328'), ('20260922', '215413'),
        ('20260922', '215431'), ('20260923', '101654'), ('20260923', '101716'),
        ('20260923', '101736'), ('20260923', '101758'))),
    'scripts.v2_gap_chronos_covariates': tuple(f'runs/20260922T{t}Z_v2_gaprecovery' for t in (
        '223901', '223918', '224158', '224215', '225335', '225350', '225405', '225420')),
}


def frozen_inputs(producer):
    """The input folders of a producer that exist only in the frozen repository, less any that
    have since been copied in."""
    return tuple(p for p in FROZEN_INPUTS.get(producer, ()) if not artifact_path(p).exists())


#: Live entry points that are not pipeline producers. They are run by hand, so nothing imports
#: them and a reachability walk would call them dead.
OPERATOR_TOOLS = {
    'src.experiments.v2_seed_replicate': 'a horizon arm re-tuned under another sampler seed; '
                                         'launched by the seedrep campaign',
    'scripts.v2_fetch_reports': 'fetches what a campaign wrote under reports/bay_twin/runs/',
    'scripts.verify_contract': 'checks every file of the release against config/release_contract.json',
    'scripts.scientific_campaign': 'the Trinity connector every remote run goes through',
    'scripts.review_trinity_link': 'the same, with an explicit host and port for a tunnel',
    'scripts.v2_campaign': 'uploads a per-run source snapshot and launches it on the cluster',
    'scripts.v2_t1_remote': 'addenda 49, 51 and 52: uploads, launches, polls and fetches the rerun '
                            'queue on the cluster',
    'scripts.experiments.v2_t1_queue': 'the cluster-side queue of addenda 49, 51 and 52 (at most '
                                              'two GPU jobs, load and memory checks)',
    'scripts.trinity_cleanup': 'run on Trinity: frees disk by removing only files with a verified copy '
                               '(unpacked snapshot inputs, downloaded result bundles, per-trial columns)',
    'scripts.experiments.v2_chronos_timeval_repair': 'addendum 42: marks orphaned trials failed and '
                                                           'quarantines unreadable columns after an interrupted run',
    'scripts.experiments.v2_chronos_lora_ref': 'gate G2 of addendum 40: the windows the unmodified '
                                                      'v2_chronos.py draws, hashed without training',
    'scripts.data_setup': 'links <repo>/data to SMARTBAY_DATA_DIR and checks the expected inputs',
    'scripts.download_copernicus': 'downloads the Copernicus Marine extractions the recipe reads',
    'src.tuning.protocol_v2': 'run_study, which refuses to return fewer than 100 completed trials',
    'src.pipeline.inputs': 'guards against any live path reaching a gap-filled dataset (smartbay verify)',
    'scripts.v2_gap_status': 'progress and fetch for the addendum-30 gap campaign',
    'scripts.v2_lookback_remote': 'addendum 46: uploads, launches, polls and fetches the Chronos look-back job',
    'scripts.v2_context_holes_remote': 'addendum 53: uploads, launches, polls and fetches the context-hole job',
    'scripts.v2_o2_temperature_remote': 'addendum 54: uploads, launches, polls and fetches the oxygen-temperature job',
    'scripts.v2_fouling_reference_remote': 'addendum 56: packs, uploads, launches, polls and fetches the '
                                           'clean-state fouling reference run',
    'scripts.v2_fouling_jump_remote': 'addendum 57: packs, uploads, launches, polls and fetches the '
                                      'cleaning-jump and cast run',
    'scripts.v2_fouling_diel_remote': 'addendum 58: packs, uploads, launches, polls and fetches the '
                                      'sunlight and temperature coupling run',
    'scripts.v2_gap_gp_gate': 'addendum 59 part A: the gate that the Gaussian process with the fix '
                              'switched off reproduces the published predictions, and its record',
    'scripts.v2_ispra_rebin_remote': 'addendum 59 part B: the binning gate, and pack, upload, launch, '
                                     'poll and fetch of the per-fold ISPRA studies',
    'scripts.v2_o2_temperature_lgbm_remote':'addendum 55: copies the published selections, uploads, launches, '
                                             'polls and fetches the LightGBM temperature-feature jobs',
    'scripts.v2_gap_watch': 'emits one line per finished gap arm, for a background watcher',
    'scripts.v2_fm_zoo_remote': 'addendum 81: uploads, smoke-tests, launches, polls, checks and fetches the '
                                'pretrained forecasters on the cluster',
    'scripts.experiments.v2_fm_zoo_checks': 'addendum 81, run on the cluster before any task: each arm\'s '
                                            'determinism and its six-hour medians against the published '
                                            'columns (model_checks.json)',
    'scripts.experiments.v2_fm_zoo_fetch': 'addendum 81, run on the cluster: caches every checkpoint and '
                                           'records its revision (weights.json)',
    'scripts.v2_gap_short2026_remote': 'addendum 82 part C: the published selections, then pack, upload, '
                                       'cross-check, launch, poll and fetch of the 1-3 h Gap-2026 runs',
    'scripts.v2_add82_DH_remote': 'addendum 82 parts D and H: packs, uploads, queues (bounded slots), polls '
                                  'and fetches the held-site studies on the cluster',
    # Kept byte for byte as they ran (scripts/launchers/README.md gives each file's runs and digest).
    'scripts.launchers.add82_BE_launch': 'addendum 82 parts B and E: packs, uploads, queues (bounded '
                                         'xargs), polls and fetches the phase and seed studies',
    'scripts.launchers.add82_E_requeue_gpu0': 'addendum 82 part E, run on the cluster: shares the jobs not '
                                              'yet started with a second runner on GPU 0 (claim guards)',
    'scripts.launchers.host_command': 'run, get and put on the cluster; carried the inline launches of '
                                      'addendum 82 parts A and G and addendum 81\'s GPU-0 workers',
    # Addendum 83's launchers, written in the repository and run from it (each run's record is under
    # runs/v2_addendum83/launch/ and runs/v2_addendum83/logs/).
    'scripts.launchers.add83_launch': 'addendum 83: hash-verified snapshot, upload, the adaptive job queue '
                                      '(width and load limits), status and fetch of the shared studies',
    'scripts.launchers.add83_queue': 'addendum 83, run on the cluster: starts each queued study at most '
                                     'once while the job count and the load average stay under their limits',
    'scripts.launchers.add83_perfold_host': 'addendum 83, amendment 1: the per-fold studies of the ISPRA '
                                            'arms on the cluster, through a second copy of the queue',
    'scripts.launchers.add83_perfold_local': 'addendum 83: the other per-fold studies, run locally after gate '
                                             'L-G7, a fixed number at a time',
    'scripts.experiments.v2_oxygen_equivalence_remote': 'addendum 60 part B: packs, uploads, launches, polls '
                                                        'and fetches the cluster arms B2, B3, B5 and B6',
    'scripts.v2_oxygen_mgl_chronos_remote': 'addendum 60 part C3: packs, uploads, launches, polls and fetches '
                                            'the pretrained utilisation arms in mg/l on the cluster',
    'scripts.experiments.v2_oxygen_mgl_chronos': 'addendum 60 part C3, run on the cluster: the zero-shot '
                                                 'utilisation arms, published unit then mg/l',
    'scripts.experiments.v2_oxygen_mgl_chronos_adapted': 'addendum 60 part C3, run on the cluster: the adapted '
                                                         'utilisation arms retrained, published unit then mg/l',
    'scripts.v2_seasonal_table': 'the seasonal stress table\'s rows from the release-v2 bundle '
                                 '(python -m scripts.v2_seasonal_table v2_release_v2); it writes '
                                 'seasonal_table.csv inside that external, locked bundle, so no stage '
                                 'runs it',
    # Exploratory analysis of data the forecasting pipeline does not read (2026-09-23). Not declared
    # experiments, and the thesis draws no result from these; the four EDA scripts whose outputs the
    # thesis does cite are producers of the operational stage instead.
    **{f'scripts.eda_opportunities.{name}': 'exploratory: ' + what for name, what in (
        ('eda_a_exo2', 'EXO2 chlorophyll, turbidity and pH, and the 2026 decline episodes'),
        ('eda_c_currents', 'the ZPulse current meters and the 5 m salinity-rise events'),
        ('eda_d_gen1', 'decline timing and levels on the Gen-1 oxygen record'),
        ('eda_d2_nocturnal', 'the overnight-onset test of v2_warning_episodes outside 2026'),
        ('eda_e_rain', 'the station rain gauge against the Magra level and salinity'),
        ('eda_f_bio_features', 'chlorophyll, pH and turbidity as oxygen forecast features'))},
}

#: Command-line arguments a producer needs, passed by `smartbay run` after `python -m <module>`: the
#: run folders under reports/bay_twin/runs/ its published artifact was computed from, each checked by
#: regenerating that artifact byte-identical. Without them these scripts would stop on a missing
#: argument or, worse, write an empty summary over the published one.
PRODUCER_ARGS = {
    'scripts.v2_horizon_blend': ('20260920T083322Z_v2_horizon', '20260920T133747Z_v2_horizon'),
    'scripts.v2_bathy_pooled': ('20260920T194734Z_v2_bathy', '20260920T083322Z_v2_horizon'),
    'scripts.v2_shap_horizons': ('20260920T083322Z_v2_horizon', '20260920T133747Z_v2_horizon'),
    # prints the reserved-evaluation table's rows; the printout matches the typed table
    'scripts.v2_reserved_table': ('v2_reserved_reading_v2',),
}


def producer_command(module):
    """The argument list `smartbay run` executes for a producer, after the interpreter."""
    return ['-m', module, *PRODUCER_ARGS.get(module, ())]

# A working checkout may keep further operator tools outside the published tree; merge them if present.
try:
    from private.registry_extra import OPERATOR_TOOLS as _LOCAL_TOOLS
except ImportError:
    _LOCAL_TOOLS = {}
OPERATOR_TOOLS.update(_LOCAL_TOOLS)


def resolve(names):
    """Expand stage names into a run order, pulling in dependencies. 'all' means every stage."""
    wanted = list(STAGE_BY_NAME) if 'all' in names else list(names)
    unknown = [n for n in wanted if n not in STAGE_BY_NAME]
    if unknown:
        raise SystemExit(f"unknown stage(s): {', '.join(unknown)}\n"
                         f"known: {', '.join(STAGE_BY_NAME)}, all")
    order, seen = [], set()

    def visit(name, trail=()):
        if name in seen:
            return
        if name in trail:
            raise SystemExit(f"dependency cycle: {' -> '.join(trail + (name,))}")
        for dep in STAGE_BY_NAME[name].depends_on:
            visit(dep, trail + (name,))
        seen.add(name)
        order.append(STAGE_BY_NAME[name])

    for n in wanted:
        visit(n)
    return order
