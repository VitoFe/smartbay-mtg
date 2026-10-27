"""High-Precision Oceanographic Thermodynamics & Gas Flux Engine.

Implements:
1. Garcia & Gordon (1992) equilibrium oxygen solubility in seawater.
2. TEOS-10 in-situ density conversion: C[umol/L] = C[umol/kg] * rho / 1000.
3. Apparent Oxygen Utilization (AOU): AOU = O2_sol - DO.
4. Wanninkhof (2014) air-sea gas transfer velocity kw in m/s:
   kw = (0.251 * u10^2) * (Sc / 660)^(-1/2) * (1 / 360000) [m/s]
5. Temperature-dependent Schmidt number Sc(T) polynomial.
"""

import numpy as np
from typing import Union, Tuple

try:
    import gsw
    HAS_GSW = True
except ImportError:
    HAS_GSW = False


# Conversion constants for Dissolved Oxygen:
MG_L_TO_UMOL_L = 31.2512
UMOL_L_TO_MG_L = 1.0 / MG_L_TO_UMOL_L
ML_L_TO_UMOL_L = 44.661  # Project convention: real-gas O2 volume at STP.
UMOL_L_TO_ML_L = 1.0 / ML_L_TO_UMOL_L


def o2sol_garcia_gordon_umol_per_l(
    temperature_c: Union[float, np.ndarray],
    salinity_psu: Union[float, np.ndarray]
) -> Union[float, np.ndarray]:
    """
    Computes equilibrium oxygen solubility C* (umol/L) with the Garcia & Gordon (1992) equation
    in its mL/L form, converted at 44.661 umol per mL (the project's convention; SCOR WG 142
    gives 44.6596).

    The coefficients are not one published set. A0-A5 are Garcia & Gordon's fit to the
    Benson & Krause (1984) data; B0-B3 and C0 are the salinity terms of their combined fit.
    The Benson & Krause salinity terms are B = (-6.24523e-3, -7.37614e-3, -1.03410e-2,
    -8.17083e-3) and C0 = -4.88682e-7. Over 10 to 28 deg C and practical salinity 24 to 38 this
    function exceeds the consistent Benson & Krause set (at 44.6596) by -0.03 to +0.22 umol/L;
    the code is unchanged pending a contract increment.

    Valid for T in [0, 40] deg C and practical salinity S_P in [0, 42] (dimensionless; the
    argument name keeps "psu" as an identifier).
    """
    T = np.asarray(temperature_c, dtype=np.float64)
    S = np.asarray(salinity_psu, dtype=np.float64)

    # Scaled temperature variable Ts
    T_k = T + 273.15
    Ts = np.log((298.15 - T) / T_k)

    # Garcia & Gordon (1992) Table 1 coefficients (mL/L form, Benson & Krause)
    A0 = 2.00907
    A1 = 3.22014
    A2 = 4.05010
    A3 = 4.94457
    A4 = -0.256847
    A5 = 3.88767

    B0 = -0.00624097
    B1 = -0.00693498
    B2 = -0.00690358
    B3 = -0.00429155

    C0 = -0.00000031168

    ln_Cstar = (
        A0 + A1 * Ts + A2 * (Ts**2) + A3 * (Ts**3) + A4 * (Ts**4) + A5 * (Ts**5)
        + S * (B0 + B1 * Ts + B2 * (Ts**2) + B3 * (Ts**3))
        + C0 * (S**2)
    )

    Cstar_umol_l = np.exp(ln_Cstar) * ML_L_TO_UMOL_L

    if np.isscalar(temperature_c) and np.isscalar(salinity_psu):
        return float(Cstar_umol_l)
    return Cstar_umol_l


#: Garcia & Gordon (1992) Table 1, the Benson & Krause cm3/dm3 column, every term
BK_A = (2.00907, 3.22014, 4.05010, 4.94457, -2.56847e-1, 3.88767)
BK_B = (-6.24523e-3, -7.37614e-3, -1.03410e-2, -8.17083e-3)
BK_C0 = -4.88682e-7
GG_UMOL_PER_CM3 = 1000.0 / 22.3916     # 44.6596, Garcia & Gordon's molar volume


def o2sol_benson_krause_umol_per_l(
    temperature_c: Union[float, np.ndarray],
    salinity_psu: Union[float, np.ndarray]
) -> Union[float, np.ndarray]:
    """
    Equilibrium oxygen solubility C* (umol/L) with the consistent Benson & Krause set of Garcia &
    Gordon (1992) for every term, converted at their 44.6596 umol per cm3. It reproduces the table's
    check value of 6.315 cm3/dm3 at 10 deg C and S = 35 (scripts/solubility_coefficient_check.py).

    Not used by any published arm: addendum 71 refits the utilisation arms with it in place of
    o2sol_garcia_gordon_umol_per_l to measure what the project's coefficient mixture moves.
    """
    T = np.asarray(temperature_c, dtype=np.float64)
    S = np.asarray(salinity_psu, dtype=np.float64)
    Ts = np.log((298.15 - T) / (T + 273.15))
    ln_Cstar = (sum(a * Ts**i for i, a in enumerate(BK_A))
                + S * sum(b * Ts**j for j, b in enumerate(BK_B)) + BK_C0 * S**2)
    Cstar_umol_l = np.exp(ln_Cstar) * GG_UMOL_PER_CM3
    if np.isscalar(temperature_c) and np.isscalar(salinity_psu):
        return float(Cstar_umol_l)
    return Cstar_umol_l


def seawater_density_kg_per_m3(
    temperature_c: Union[float, np.ndarray],
    salinity_psu: Union[float, np.ndarray],
    pressure_dbar: Union[float, np.ndarray] = 0.0,
    lat: float = 44.08,
    lon: float = 9.87
) -> Union[float, np.ndarray]:
    """
    Computes in-situ seawater density rho (kg/m3) via TEOS-10 (gsw) or UNESCO equation of state.
    """
    T = np.asarray(temperature_c, dtype=np.float64)
    S = np.asarray(salinity_psu, dtype=np.float64)
    p = np.asarray(pressure_dbar, dtype=np.float64)

    if HAS_GSW:
        SA = gsw.SA_from_SP(S, p, lon, lat)
        CT = gsw.CT_from_t(SA, T, p)
        rho = gsw.rho(SA, CT, p)
        if np.isscalar(temperature_c):
            return float(rho)
        return rho
    else:
        # Standard UNESCO (1980) 1-atm approximation
        rho0 = (
            999.842594 + 6.793952e-2 * T - 9.095290e-3 * (T**2)
            + 1.001685e-4 * (T**3) - 1.120083e-6 * (T**4) + 6.536332e-9 * (T**5)
        )
        A = 8.24493e-1 - 4.0899e-3 * T + 7.6438e-5 * (T**2) - 8.2467e-7 * (T**3) + 5.3875e-9 * (T**4)
        B = -5.72466e-3 + 1.0227e-4 * T - 1.6546e-6 * (T**2)
        C = 4.8314e-4
        rho = rho0 + A * S + B * (S**1.5) + C * (S**2)
        if np.isscalar(temperature_c):
            return float(rho)
        return rho


def umol_kg_to_umol_l(
    concentration_umol_kg: Union[float, np.ndarray],
    temperature_c: Union[float, np.ndarray],
    salinity_psu: Union[float, np.ndarray],
    pressure_dbar: Union[float, np.ndarray] = 0.0
) -> Union[float, np.ndarray]:
    """
    Converts oxygen concentration from gravimetric (umol/kg) to volumetric (umol/L).
    C[umol/L] = C[umol/kg] * rho(S, T, p) / 1000.
    """
    rho = seawater_density_kg_per_m3(temperature_c, salinity_psu, pressure_dbar)
    return concentration_umol_kg * (rho / 1000.0)


def umol_l_to_umol_kg(
    concentration_umol_l: Union[float, np.ndarray],
    temperature_c: Union[float, np.ndarray],
    salinity_psu: Union[float, np.ndarray],
    pressure_dbar: Union[float, np.ndarray] = 0.0
) -> Union[float, np.ndarray]:
    """
    Converts oxygen concentration from volumetric (umol/L) to gravimetric (umol/kg).
    """
    rho = seawater_density_kg_per_m3(temperature_c, salinity_psu, pressure_dbar)
    return concentration_umol_l / (rho / 1000.0)


def compute_aou_umol_l(
    do_measured_umol_l: Union[float, np.ndarray],
    temperature_c: Union[float, np.ndarray],
    salinity_psu: Union[float, np.ndarray]
) -> Tuple[Union[float, np.ndarray], Union[float, np.ndarray]]:
    """
    Computes Apparent Oxygen Utilization (AOU) and equilibrium solubility in umol/L.
    AOU = O2_sol - DO.
    Supersaturation (DO > O2_sol) natively gives negative AOU (AOU < 0).
    """
    o2_sol = o2sol_garcia_gordon_umol_per_l(temperature_c, salinity_psu)
    aou = o2_sol - do_measured_umol_l
    return aou, o2_sol


def schmidt_number_o2(temperature_c: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """
    Computes the dimensionless Schmidt number Sc for O2 in seawater
    using Wanninkhof (2014) polynomial for T in [0, 40] deg C.
    """
    T = np.clip(np.asarray(temperature_c, dtype=np.float64), -2.0, 40.0)
    # Wanninkhof (2014) Table 1 coefficients for O2 in seawater
    sc = 1920.4 - 135.6 * T + 5.2122 * (T**2) - 0.10939 * (T**3) + 0.00093777 * (T**4)
    if np.isscalar(temperature_c):
        return float(sc)
    return sc


def wanninkhof_air_sea_transfer_kw_m_per_s(
    wind_speed_10m: Union[float, np.ndarray],
    temperature_c: Union[float, np.ndarray]
) -> Union[float, np.ndarray]:
    """
    Computes Wanninkhof (2014) air-sea gas transfer velocity kw in SI units (m/s).
    k660 = 0.251 * u10^2 (in cm/h).
    kw [m/s] = (0.251 * u10^2) * (Sc / 660)^(-0.5) * (0.01 / 3600).
    """
    u10 = np.clip(np.asarray(wind_speed_10m, dtype=np.float64), 0.0, 50.0)
    sc = schmidt_number_o2(temperature_c)

    # 0.251 cm/h -> m/s: multiply by 1e-2 / 3600 = 1 / 360000
    cm_h_to_m_s = 1.0 / 360000.0
    kw_cm_h = 0.251 * (u10**2) * np.sqrt(660.0 / sc)
    kw_m_s = kw_cm_h * cm_h_to_m_s

    if np.isscalar(wind_speed_10m) and np.isscalar(temperature_c):
        return float(kw_m_s)
    return kw_m_s
