"""Which Garcia-Gordon coefficients src/data/thermodynamics.py uses, and what that costs.

Garcia and Gordon (1992, Limnol. Oceanogr. 37:1307, Table 1) give three coefficient sets for their
Eq. 8, each in cm3 dm-3 and in umol kg-1: a fit to the data of Benson and Krause (1984), a fit to
the data of Murray and Riley (1969) and Carpenter (1966), and a combined fit giving the two equal
weight. They recommend the Benson and Krause set. The cm3 dm-3 values below were transcribed from
the table of the published article; the conversion at a molar volume of 22 391.6 cm3 mol-1
(the paper's own value) is 1000/22.3916 = 44.6596 umol per cm3.

The project function takes A0-A5 from the Benson and Krause cm3 dm-3 column and B0-B3, C0 from
the combined-fit cm3 dm-3 column, and multiplies by 44.661. This script identifies each term with
its column, evaluates the difference against the consistent Benson and Krause set over
10-28 degC and S 24-38, and writes reports/bay_twin/runs/solubility_coefficients.json.

Usage (from the repository root): python -m scripts.solubility_coefficient_check
"""
import json
from pathlib import Path

import numpy as np

from src.data import thermodynamics as th

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'reports/bay_twin/runs/solubility_coefficients.json'

# Garcia and Gordon (1992), Table 1, cm3 dm-3 columns
TABLE1 = {
    'benson_krause': {'A': [2.00907, 3.22014, 4.05010, 4.94457, -2.56847e-1, 3.88767],
                      'B': [-6.24523e-3, -7.37614e-3, -1.03410e-2, -8.17083e-3], 'C0': -4.88682e-7,
                      'check_10C_S35': 6.315},
    'murray_riley_carpenter': {'A': [2.00805, 3.22773, 3.93008, 4.68335, 2.51836, 4.60916e-1],
                               'B': [-6.23669e-3, -6.49387e-3, -3.47040e-3, -4.27025e-4], 'C0': -6.40583e-8,
                               'check_10C_S35': 6.318},
    'combined': {'A': [2.00856, 3.22400, 3.99063, 4.80299, 9.78188e-1, 1.71069],
                 'B': [-6.24097e-3, -6.93498e-3, -6.90358e-3, -4.29155e-3], 'C0': -3.11680e-7,
                 'check_10C_S35': 6.316},
}
UMOL_PER_CM3 = 1000.0 / 22.3916   # 44.6596, Garcia and Gordon's molar volume


def ln_c(t, s, A, B, C0):
    ts = np.log((298.15 - t) / (273.15 + t))
    return (sum(a * ts ** i for i, a in enumerate(A)) + s * sum(b * ts ** j for j, b in enumerate(B))
            + C0 * s * s)


def main():
    # which column each group of the project's coefficients belongs to (read from its source)
    code_A = [2.00907, 3.22014, 4.05010, 4.94457, -0.256847, 3.88767]
    code_B = [-0.00624097, -0.00693498, -0.00690358, -0.00429155]
    code_C0 = -0.00000031168
    ident = {group: [name for name, col in TABLE1.items() if np.allclose(col[key] if key != 'C0' else [col['C0']],
                                                                       vals, rtol=1e-6, atol=0)]
             for group, key, vals in (('A', 'A', code_A), ('B', 'B', code_B), ('C0', 'C0', [code_C0]))}
    # the table's own check values reproduce, so the transcription is right
    checks = {name: {'table': col['check_10C_S35'],
                     'computed': round(float(np.exp(ln_c(10.0, 35.0, col['A'], col['B'], col['C0']))), 4)}
              for name, col in TABLE1.items()}
    t, s = np.meshgrid(np.arange(10.0, 28.0001, 0.1), np.arange(24.0, 38.0001, 0.1))
    code = th.o2sol_garcia_gordon_umol_per_l(t, s)
    bk = TABLE1['benson_krause']
    bk_gg = np.exp(ln_c(t, s, bk['A'], bk['B'], bk['C0'])) * UMOL_PER_CM3
    bk_code_factor = np.exp(ln_c(t, s, bk['A'], bk['B'], bk['C0'])) * th.ML_L_TO_UMOL_L
    d = code - bk_gg
    d_coef = code - bk_code_factor
    out = {
        'source': 'Garcia and Gordon (1992), Limnol. Oceanogr. 37(6):1307-1312, Table 1, cm3 dm-3 columns',
        'project_function': 'src/data/thermodynamics.py:o2sol_garcia_gordon_umol_per_l',
        'project_terms_match_column': ident,
        'project_factor_umol_per_cm3': th.ML_L_TO_UMOL_L,
        'garcia_gordon_factor_umol_per_cm3': round(UMOL_PER_CM3, 5),
        'table_check_values_cm3_dm3': checks,
        'project_at_10C_S35_cm3_dm3': round(float(th.o2sol_garcia_gordon_umol_per_l(10.0, 35.0) / th.ML_L_TO_UMOL_L), 4),
        'grid': {'temperature_c': [10.0, 28.0], 'salinity': [24.0, 38.0], 'step': 0.1},
        'difference_umol_per_l_project_minus_benson_krause': {
            'min': round(float(d.min()), 3), 'max': round(float(d.max()), 3),
            'at_min': [round(float(t.flat[d.argmin()]), 1), round(float(s.flat[d.argmin()]), 1)],
            'at_max': [round(float(t.flat[d.argmax()]), 1), round(float(s.flat[d.argmax()]), 1)]},
        'of_which_coefficients_only_umol_per_l': {'min': round(float(d_coef.min()), 3),
                                                   'max': round(float(d_coef.max()), 3)},
        'solubility_range_umol_per_l': [round(float(code.min()), 1), round(float(code.max()), 1)],
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(out, indent=2))


if __name__ == '__main__':
    main()
