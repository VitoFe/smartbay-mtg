"""Download the Copernicus Marine extractions the thesis reads, with the official toolbox.

The recipe reads the Mediterranean analysis-forecast system at 4.2 km (MEDSEA_ANALYSISFORECAST):
temperature, salinity and dissolved oxygen, daily means, on a single row of six cells south of the
Gulf of La Spezia, at seven levels between 1.02 and 16.27 m. Two extractions exist:

    prior   2023-11-01 .. 2026-01-16   data/external/Copernicus/T2023_2026/  (the delivered system)
    2026    2025-12-28 .. 2026-06-30   data/external/Copernicus/T2026/       (held-out reading only)

    pip install copernicusmarine
    python scripts/download_copernicus.py --period all            # thetao, so, o2
    python scripts/download_copernicus.py --period prior --products all
    python scripts/download_copernicus.py --period 2026 --dry-run # print the requests only

Credentials are never passed on the command line: the toolbox reads
COPERNICUSMARINE_SERVICE_USERNAME and COPERNICUSMARINE_SERVICE_PASSWORD from the environment, which
src/settings.py fills from the uncommitted .env. A free account is at https://marine.copernicus.eu.

Analysis-forecast products are re-issued as new dataset versions, and a later download can differ
in bytes from the files the thesis used (the release contract binds the 2026 files' digests) even
where the values agree; pass --dataset-version to request a specific version.
"""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.settings import data_dir  # noqa: E402

EXTENT = dict(minimum_longitude=9.77, maximum_longitude=10.01,
              minimum_latitude=44.00, maximum_latitude=44.04,
              minimum_depth=1.0, maximum_depth=16.3)

PERIODS = {
    'prior': ('2023-11-01', '2026-01-16', 'T2023_2026'),
    '2026': ('2025-12-28', '2026-06-30', 'T2026'),
}

# (dataset id, variables): what the recipe reads, then what the first extraction also carried.
USED = [
    ('cmems_mod_med_phy-tem_anfc_4.2km_P1D-m', ['thetao']),
    ('cmems_mod_med_phy-sal_anfc_4.2km_P1D-m', ['so']),
    ('cmems_mod_med_bgc-bio_anfc_4.2km_P1D-m', ['o2']),
]
EXTRA = [
    ('cmems_mod_med_bgc-bio_anfc_4.2km_P1D-m', ['nppv', 'o2']),
    ('cmems_mod_med_bgc-car_anfc_4.2km_P1D-m', ['dissic', 'ph', 'talk']),
    ('cmems_mod_med_bgc-co2_anfc_4.2km_P1D-m', ['fgco2', 'spco2']),
    ('cmems_mod_med_bgc-nut_anfc_4.2km_P1D-m', ['nh4', 'no3', 'po4', 'si']),
    ('cmems_mod_med_bgc-pft_anfc_4.2km_P1D-m', ['chl', 'phyc', 'zooc']),
    ('cmems_mod_med_phy-cur_anfc_4.2km_P1D-m', ['uo', 'vo']),
]


def requests(period, products):
    table = USED if products == 'used' else USED[:2] + EXTRA
    start, end, folder = PERIODS[period]
    for dataset, variables in table:
        yield folder, dict(dataset_id=dataset, variables=variables, start_datetime=start,
                           end_datetime=end, **EXTENT)


def main():
    p = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    p.add_argument('--period', choices=['prior', '2026', 'all'], default='all')
    p.add_argument('--products', choices=['used', 'all'], default='used',
                   help='used: thetao, so, o2 (what the code reads); all: the first extraction\'s full set')
    p.add_argument('--dataset-version', default=None)
    p.add_argument('--output-root', type=Path, default=None,
                   help='default: <data dir>/external/Copernicus')
    p.add_argument('--dry-run', action='store_true')
    a = p.parse_args()

    root = a.output_root or data_dir() / 'external' / 'Copernicus'
    periods = ['prior', '2026'] if a.period == 'all' else [a.period]
    plan = [(root / folder, req) for per in periods for folder, req in requests(per, a.products)]
    for out, req in plan:
        print(f"{req['dataset_id']:42s} {','.join(req['variables']):16s} "
              f"{req['start_datetime']}..{req['end_datetime']} -> {out}")
    if a.dry_run:
        return 0

    import os
    if not (os.environ.get('COPERNICUSMARINE_SERVICE_USERNAME')
            and os.environ.get('COPERNICUSMARINE_SERVICE_PASSWORD')):
        print('COPERNICUSMARINE_SERVICE_USERNAME / _PASSWORD are not set (environment or .env); '
              'alternatively run `copernicusmarine login` once.')
    try:
        import copernicusmarine
    except ImportError:
        print('the Copernicus Marine toolbox is not installed: pip install copernicusmarine')
        return 1
    for out, req in plan:
        out.mkdir(parents=True, exist_ok=True)
        if a.dataset_version:
            req = dict(req, dataset_version=a.dataset_version)
        copernicusmarine.subset(output_directory=str(out), **req)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
