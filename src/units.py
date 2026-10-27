"""Dissolved oxygen in mg/l, the unit the manuscript states.

The pipeline carries oxygen per kilogram of seawater (the sondes' micromolar column). Every table
generator and figure producer that prints or draws oxygen converts through this module, so that one
factor, computed from the record's own density by scripts/v2_oxygen_units.py, governs every number:

    from src.units import oxygen_mg, OXYGEN_UNIT_TEX
    oxygen_mg(5.849)        # umol/kg -> mg/l
"""
import json
import math
from functools import lru_cache
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
UNITS = ROOT / 'reports/bay_twin/runs/v2_oxygen_units.json'

#: the unit as the manuscript typesets it, and as plain text for figures
OXYGEN_UNIT_TEX = r'mg/$\ell$'
OXYGEN_UNIT_TEXT = 'mg/ℓ'


@lru_cache(maxsize=1)
def factor():
    """mg/l per umol/kg: 0.0319988 x the record's mean density / 1000."""
    return float(json.loads(UNITS.read_text())['c_mg_per_l_per_umol_kg'])


def oxygen_mg(value):
    """A pipeline oxygen value, error or contrast (umol/kg) in mg/l. None and NaN pass through."""
    if value is None:
        return None
    try:
        return value * factor()
    except TypeError:
        return [oxygen_mg(v) for v in value]


#: mg of O2 per micromole: a per-litre source (the casts' mg/l column, Copernicus, the SeapHOx, the
#: per-litre Garcia-Gordon saturation) converts by this alone, without the density in c
MG_PER_UMOL_L = 0.0319988


def sig_decimals(value, sig):
    """The number of decimals that prints `value` with `sig` significant figures (at least 0)."""
    if value == 0 or not math.isfinite(value):
        return sig
    return max(0, sig - 1 - math.floor(math.log10(abs(value))))


def mg_decimals(value, decimals, scale=None):
    """Decimals for an oxygen value printed at `decimals` in umol so that, in mg/l, it keeps the
    significant figures it had (at least two). A zero keeps two more decimals than it had."""
    if value is None or not math.isfinite(value) or round(abs(value), decimals) == 0:
        return decimals + 2
    printed = f'{abs(value):.{decimals}f}'.replace('.', '').lstrip('0')
    sig = max(2, len(printed))
    return sig_decimals(value * (factor() if scale is None else scale), sig)


def column_decimals(values, decimals, scale=None):
    """One number of decimals for a column of oxygen values: the most any of them needs, so no value
    prints with fewer significant figures than it had, and never more than two beyond `decimals`."""
    found = [mg_decimals(v, decimals, scale) for v in values
             if v is not None and math.isfinite(v) and round(abs(v), decimals) != 0]
    return min(decimals + 2, max(found)) if found else decimals + 2
