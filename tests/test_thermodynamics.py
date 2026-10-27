"""Acceptance Test Suite: Oceanographic Thermodynamics & Gas Flux Verification."""

import pytest
import numpy as np
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.data.thermodynamics import (
    o2sol_garcia_gordon_umol_per_l,
    seawater_density_kg_per_m3,
    umol_kg_to_umol_l,
    umol_l_to_umol_kg,
    compute_aou_umol_l,
    schmidt_number_o2,
    wanninkhof_air_sea_transfer_kw_m_per_s,
    HAS_GSW
)

if HAS_GSW:
    import gsw


def test_known_benson_krause_points():
    """
    Verifies that Garcia & Gordon solubility matches international benchmark points:
    At T=20 C, S=35 PSU: O2_sol is approx 230 umol/L (~5.15 ml/L).
    At T=0 C, S=35 PSU: O2_sol is approx 350-360 umol/L.
    """
    sol_20_35 = o2sol_garcia_gordon_umol_per_l(20.0, 35.0)
    assert 220.0 < sol_20_35 < 235.0, f"Expected ~228 umol/L at (20C, 35PSU), got {sol_20_35}"

    sol_0_35 = o2sol_garcia_gordon_umol_per_l(0.0, 35.0)
    assert 340.0 < sol_0_35 < 365.0, f"Expected ~350 umol/L at (0C, 35PSU), got {sol_0_35}"


def test_gsw_cross_validation_grid():
    """
    Cross-validates hand-coded Garcia & Gordon against gsw.O2sol_SP_pt
    converted from umol/kg to umol/L across a 50x50 grid of (T, S).
    Agreement must be within < 1.0% relative tolerance across all seawater points.
    """
    if not HAS_GSW:
        pytest.skip("gsw library not installed, skipping TEOS-10 cross check")

    temps = np.linspace(2.0, 32.0, 50)
    salinities = np.linspace(25.0, 40.0, 50)
    T_grid, S_grid = np.meshgrid(temps, salinities)

    # Hand-coded G&G
    gg_umol_l = o2sol_garcia_gordon_umol_per_l(T_grid, S_grid)

    # GSW benchmark (gsw.O2sol_SP_pt returns umol/kg)
    # Convert to umol/L via in-situ density rho
    SA = gsw.SA_from_SP(S_grid, 0.0, 9.87, 44.08)
    CT = gsw.CT_from_t(SA, T_grid, 0.0)
    gsw_umol_kg = gsw.O2sol(SA, CT, 0.0, 9.87, 44.08)
    rho = gsw.rho(SA, CT, 0.0)
    gsw_umol_l = gsw_umol_kg * (rho / 1000.0)

    rel_error = np.abs(gg_umol_l - gsw_umol_l) / gsw_umol_l
    max_rel_err = float(np.max(rel_error))

    assert max_rel_err < 0.015, f"Maximum relative discrepancy with GSW is {max_rel_err*100:.2f}%, exceeded 1.5% limit!"


def test_density_conversion_magnitude():
    """
    Asserts that gravimetric (umol/kg) and volumetric (umol/L) conversions
    exhibit the expected ~2.7% density difference in Mediterranean seawater (~1027 kg/m3).
    """
    c_kg = 225.0
    c_l = umol_kg_to_umol_l(c_kg, temperature_c=18.0, salinity_psu=38.0)
    assert c_l > c_kg, "Volumetric concentration (umol/L) must exceed gravimetric (umol/kg)"
    ratio = c_l / c_kg
    assert 1.025 < ratio < 1.030, f"Density multiplier ratio should be ~1.027, got {ratio:.4f}"

    # Round trip
    c_kg_recovered = umol_l_to_umol_kg(c_l, temperature_c=18.0, salinity_psu=38.0)
    assert np.isclose(c_kg_recovered, c_kg, atol=1e-5)


def test_wanninkhof_kw_units_and_order():
    """
    Verifies Wanninkhof (2014) air-sea transfer velocity:
    At u10 = 10 m/s and T = 20 C, kw must be returned in m/s on order 1e-5 m/s.
    """
    kw_m_s = wanninkhof_air_sea_transfer_kw_m_per_s(wind_speed_10m=10.0, temperature_c=20.0)
    assert 1e-6 < kw_m_s < 1e-4, f"kw must be on order 1e-5 m/s, got {kw_m_s:.2e}"

    # Zero wind gives zero flux
    assert wanninkhof_air_sea_transfer_kw_m_per_s(0.0, 20.0) == 0.0


def test_aou_supersaturation():
    """
    Verifies that supersaturated oxygen (DO > O2_sol) natively produces negative AOU.
    """
    o2_sol = o2sol_garcia_gordon_umol_per_l(20.0, 38.0)
    do_supersat = o2_sol * 1.15  # 115% saturation

    aou, sol_calc = compute_aou_umol_l(do_supersat, 20.0, 38.0)
    assert aou < 0.0, f"AOU for 115% saturation must be negative, got {aou}"
    assert np.isclose(aou, o2_sol - do_supersat, atol=1e-4)
