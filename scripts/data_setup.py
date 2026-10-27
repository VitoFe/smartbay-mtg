"""Point the repository at its data, and say what is missing.

The code reads every input under `<repo>/data`. The data are not in the repository; they live
wherever `SMARTBAY_DATA_DIR` says (environment or `.env`), and `link` makes `<repo>/data` a
directory junction (Windows) or a symbolic link (elsewhere) to that folder, so no module has to
know where the data really are and the contract-bound code stays byte-exact.

    python scripts/data_setup.py link     # create <repo>/data -> $SMARTBAY_DATA_DIR
    python scripts/data_setup.py check    # report every expected input, present or missing

`check` exits non-zero when a required input is missing. Optional inputs feed one analysis each
(named in the report) and are listed but never fail the check.
"""
import argparse
import glob
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.settings import data_dir  # noqa: E402

# (pattern under data/, minimum matches, required, what reads it)
LAYOUT = [
    ('raw/Nodo_*.txt', 14, True, 'WNode temperature loggers, hourly telemetry exports'),
    ('raw/*-CTD(calibrato).txt', 5, True, 'YSI EXO2 sondes (temperature, oxygen, salinity)'),
    ('raw/T2-Meteo.txt', 1, True, 'Santa Teresa meteorological station, 10-min'),
    ('processed/ctd_all_casts.csv.gz', 1, True, 'the quality-controlled CTD archive (722 casts)'),
    ('external/Copernicus/T2023_2026/*phy-tem*.nc', 1, True, 'CMEMS MED temperature, vertical prior'),
    ('external/Copernicus/T2023_2026/*phy-sal*.nc', 1, True, 'CMEMS MED salinity, vertical prior'),
    ('external/Copernicus/T2023_2026/*bgc-bio*.nc', 1, True, 'CMEMS MED oxygen, vertical prior'),
    ('external/Batimetrie_RegioneLiguria/Batimetrie.kml', 1, True, 'depth bands for the support gates'),
    ('external/CoastalZones2018/bay_landcover.geojson', 1, True, 'land mask for the support gates'),
    ('external/Copernicus/T2026/*.nc', 3, False, 'CMEMS MED for 2026, held-out spatial reading (addendum 38)'),
    ('external/Rivers/arpa_magra_ameglia_portata_*.csv', 1, False, 'Magra stage height (river features)'),
    ('external/Rivers/liv_TOS_Arno_Bocca.csv', 1, False, 'Arno stage height at Bocca d\'Arno'),
    ('external/Ispra/ISPRA_La_Spezia_*.csv', 1, False, 'ISPRA La Spezia tide gauge'),
    ('external/Bathymetry/Batimetrie.shp', 1, False, 'second export of the Regione Liguria layer'),
    ('external/EMODnet2024/bay_dtm.nc', 1, False, 'EMODnet DTM check (Section 3.2)'),
    ('external/MaGIC_LaSpezia/*.txt', 1, False, 'MaGIC coverage audit'),
    ('raw/*-Ossigeno(calibrato).txt', 1, False, 'first-generation optodes (fouling limitation, EDA)'),
    ('raw/*-Corrente.txt', 1, False, 'current-meter thermistors at 2 and 8 m (EDA)'),
    ('raw/*-CTD+pH.txt', 1, False, 'pH records (decline typing, EDA)'),
    ('raw/TBoaT_16m-SeapHOx.txt', 1, False, 'Tinetto SeapHOx at 16 m (EDA, in-situ product check)'),
    ('logbook.txt', 1, False, 'station cleaning log (fouling analyses)'),
    ('notebooks_from_previous_projects/*.ipynb', 1, False, 'prior-project notebooks (data audit)'),
]


def link(args):
    target = data_dir()
    here = ROOT / 'data'
    if target == here:
        print('SMARTBAY_DATA_DIR is not set; the data are expected in', here)
        return 0
    if not target.is_dir():
        print('SMARTBAY_DATA_DIR does not exist:', target)
        return 1
    if here.exists() or here.is_symlink():
        if here.resolve() == target:
            print('already linked:', here, '->', target)
            return 0
        print(here, 'already exists and points elsewhere; remove it first')
        return 1
    if os.name == 'nt':
        # A junction needs no administrator rights, unlike a symbolic link.
        subprocess.run(['cmd', '/c', 'mklink', '/J', str(here), str(target)], check=True,
                       stdout=subprocess.DEVNULL)
    else:
        here.symlink_to(target, target_is_directory=True)
    print('linked', here, '->', target)
    return 0


def check(args):
    base = ROOT / 'data'
    if not base.exists():
        print('no data folder at', base, '- set SMARTBAY_DATA_DIR and run: python scripts/data_setup.py link')
        return 1
    failed = 0
    for pattern, minimum, required, what in LAYOUT:
        n = len(glob.glob(str(base / pattern)))
        ok = n >= minimum
        mark = 'ok  ' if ok else ('MISS' if required else 'opt ')
        failed += (not ok) and required
        print(f'{mark} {n:4d}  data/{pattern:52s} {what}')
    print('required inputs missing:', failed)
    return 1 if failed else 0


def main():
    p = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    sub = p.add_subparsers(dest='cmd', required=True)
    sub.add_parser('link', help='make <repo>/data point at SMARTBAY_DATA_DIR').set_defaults(fn=link)
    sub.add_parser('check', help='report the expected inputs').set_defaults(fn=check)
    a = p.parse_args()
    return a.fn(a)


if __name__ == '__main__':
    raise SystemExit(main())
