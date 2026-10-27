"""Shared loaders for the opportunity EDA: raw exports with their own column names, hourly."""
import re
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'reports/bay_twin/runs/eda_opportunities'
OUT.mkdir(parents=True, exist_ok=True)
RAW = ROOT / 'data/raw'
MG_TO_UMOL = 31.2512  # umol/l per mg/l of O2 (1 mg = 1/31.9988 mmol)


def header(path):
    with open(path, encoding='cp1252', errors='replace') as stream:
        text = stream.read(40000)
    cols = None
    for line in text.splitlines()[:80]:
        if line.startswith('#') and 'Data' in line and 'Ora' in line:
            cols = line.lstrip('#').split()
            break
    meta = {k.strip(): v.strip() for k, v in re.findall(r'^#\s*([^=\n]+)=(.*)$', text[:20000], re.M)}
    return cols, meta


def load(name, hourly=True, how='mean'):
    """A raw export as a frame indexed by UTC time, columns named from its header, sentinels as NaN.
    Hourly means are labelled at the interval end, as read_export does."""
    path = RAW / name
    cols, meta = header(path)
    f = pd.read_csv(path, comment='#', sep=r'\s+', header=None, encoding='cp1252', low_memory=False)
    f.columns = cols[:f.shape[1]] if cols and len(cols) >= f.shape[1] else list(range(f.shape[1]))
    t = pd.to_datetime(f.iloc[:, 0].astype(str) + ' ' + f.iloc[:, 1].astype(str),
                       format='%d/%m/%Y %H:%M:%S', errors='coerce', utc=True)
    f = f.iloc[:, 2:].apply(pd.to_numeric, errors='coerce')
    f = f.where(f > -999)
    f.index = t
    f = f[f.index.notna()].sort_index()
    f = f.groupby(level=0).mean()
    if hourly:
        f = f.resample('1h', closed='left', label='right').agg(how)
    return f, meta


def meteo():
    m, _ = load('T2-Meteo.txt')
    return m


def windows_drop(series, hours=6):
    """Change over the previous `hours` hours at each hour (negative = decline)."""
    return series - series.shift(hours)
