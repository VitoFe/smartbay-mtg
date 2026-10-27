# SmartBayDO: code for *Mind the Gaps*

This repository holds the code of the MSc thesis *Mind the Gaps: Physical Guidance, State
Estimation and Telemetry Recovery in a Coastal Observing Network. Application to the Gulf of La
Spezia* (Vito Ferri, Politecnico di Torino, 2026; supervisor Prof. Tatiana Tommasi, co-supervisor
Dott. Gianfranco Durin, INRiM).

The code turns the records of the SmartBay observing network in the Gulf of La Spezia (moored
temperature probes, oxygen sondes, CTD casts, a weather station, river gauges and the Copernicus
Marine Mediterranean product) into the thesis's results. It covers four estimation tasks for water
temperature and dissolved oxygen: a station without a probe, a forecast at a probe from one hour to
three days ahead, a probe that has fallen silent, and a gap filled once both of its edges are known.
On each it tests four kinds of physical guidance: physical features, reference states and
transformations, hard constraints and dynamical operators.

The observing data, the result files, the manuscript and the presentation are not part of this
repository. With access to the data, the pipeline regenerates the results stage by stage.

## Repository layout

| Path | What it holds |
|---|---|
| `smartbay.py`, `src/pipeline/` | the pipeline's command line and its stage registry, which records which program produces which result |
| `src/data/` | loading, quality control, forcing and seawater thermodynamics |
| `src/models/`, `src/experiments/` | the estimators and the experiments |
| `src/tuning/` | the tuning protocol: one Optuna study of at least one hundred trials per method and outer fold |
| `src/units.py`, `src/settings.py` | the oxygen unit and the machine-specific settings |
| `scripts/` | further experiments and their scoring, data set-up and download, remote campaign launchers |

## Setting up

Python 3.12; the versions in `requirements.txt` are those the thesis was produced with.

```bash
python -m venv .venv
.venv/Scripts/activate            # Windows; on Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env              # then fill in what you need (see below)
```

`requirements-remote.txt` lists the extra packages of the GPU host (PyTorch and the pretrained
time-series models); they are needed only for the stages that run those models.

### Where each source comes from

| Source | Availability |
|---|---|
| WNode loggers, EXO2 sondes, current meters, SeapHOx, meteorology, CTD casts, logbook | the monitoring programme's own records; available from the project on request |
| Copernicus Marine, MEDSEA analysis-forecast (`cmems_mod_med_*_anfc_4.2km_P1D-m`) | public, free account: downloadable with the script below |
| Magra stage height | ARPAL Liguria, hydrometric archive (station Ameglia Foce Magra) |
| Arno stage height | Servizio Idrologico Regionale della Toscana (Bocca d'Arno) |
| La Spezia tide gauge | ISPRA, Rete Mareografica Nazionale |
| Regione Liguria depth bands | Geoportale Regione Liguria |
| Coastal Zones 2018 | Copernicus Land Monitoring Service |
| EMODnet DTM 2024 | EMODnet Bathymetry portal |
| MaGIC grids | MaGIC project (Marine Geohazards along the Italian Coasts) |

### Downloading the Copernicus extractions

```bash
python scripts/download_copernicus.py --period all --dry-run   # show the requests
python scripts/download_copernicus.py --period all             # thetao, so and o2 for both periods
python scripts/download_copernicus.py --period prior --products all   # the first extraction's full set
```

The script calls the official `copernicusmarine` toolbox with the extent the thesis used
(9.77–10.01 °E, 44.00–44.04 °N, 1–16.3 m, daily means) and writes into
`<data>/external/Copernicus/<T2023_2026|T2026>/`. The toolbox reads
`COPERNICUSMARINE_SERVICE_USERNAME` and `..._PASSWORD` from the environment, so no credential
appears on a command line. Analysis-forecast products are re-issued as new dataset versions, so a
fresh download can differ in bytes from the thesis's files even where the values agree;
`--dataset-version` requests a specific one.

## Running

```bash
python smartbay.py stages                 # the pipeline stages and their dependencies
python smartbay.py plan --stage all       # what each stage runs, in order
python smartbay.py run --stage data       # runs a stage and the ones it depends on
python smartbay.py run --stage all --tier smoke   # a reduced run that skips the release-only stages
```

`run` invokes each producer as `python -m <module>` with a fixed `PYTHONHASHSEED` and writes a
manifest of exit codes and output digests under `reports/bay_twin/runs/`. The producers that
typeset the thesis's tables and figures are not in this repository; `plan` lists them and `run`
skips them. The stages that run the pretrained models (`pretrained` and the gap-recovery and
forecasting experiments that use them) were run on a GPU host through
`scripts/scientific_campaign.py`, which uploads a snapshot of the code and inputs and reads the
`SMARTBAY_SSH_*` settings.

Every learned model is tuned by its own Optuna study of at least one hundred completed trials,
nested inside the training data of each outer fold (`src/tuning/protocol_v2.py` refuses to return
fewer). Two things change results in the last digits: XGBoost's row subsampling depends on the
thread count (set `V2_THREADS=2` and leave `OMP_NUM_THREADS` unset), and Optuna's sampler can draw
different trials in another major version.

## Tests

```bash
python -m pytest -q tests
```

The tests run on the code alone. They cover the rules the results depend on: nested averaging and
the bootstrap units, the guard that keeps a probe from being its own neighbour, the tuning
protocol's refusals, the isolation of fitted controls from the labels they are scored on, disjoint
folds, the absence of hard-coded metrics, seawater thermodynamics, the spatial estimator, the
Gaussian-process gap filler, the causality of the forecasters' inputs, and the scoring rules of the
forecast distributions. Three need the oxygen unit factor, which `python -m scripts.v2_oxygen_units`
computes from the data, and one needs the data folder; without them they are skipped.

## Citing

Vito Ferri, *Mind the Gaps: Physical Guidance, State Estimation and Telemetry Recovery in a Coastal
Observing Network. Application to the Gulf of La Spezia*, MSc thesis in Computer Engineering,
Politecnico di Torino, 2026.
