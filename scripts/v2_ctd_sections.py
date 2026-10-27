"""Time-depth sections from the CTD archive: the vertical structure of the bay through the year.

Asked by the supervisors on 2026-09-25: do the casts show changes at the scale of seasons or years,
periods when the column is mixed and periods when surface and bottom differ, and does temperature
drive them? The answer is descriptive. No model is fitted and nothing is scored, so nothing here
was declared in advance, and the one link it suggests to a result (the water-mass declines of
addendum 37) is reported as a candidate that the record cannot test.

The archive is every quality-controlled cast from 2021 to June 2026 (the CTD-722 cohort), filtered
as the delivered loader filters (qc_flag 0, finite targets, the same range gates) but not
restricted to the Gen. 2 years, because the question is about the whole record. Casts taken on the
same day at a station are averaged bin by bin. A map column is drawn at most 10.5 days either side
of its cast date, so a gap of three weeks or more stays blank rather than interpolated.

Surface is the mean over 0-1.25 m, bottom the mean over 6-7.5 m (every station reaches it; the
8 to 11 m beds of the moored sites, so at PBoaS and PMitiliE the layer is mid-column rather than
at the bed) or 19-21 m at PBoaT, the one station deep enough for a thermocline. Density is TEOS-10
potential density (gsw). The density difference is split into temperature and salinity parts with
the expansion and contraction coefficients at the mean of the two layers. The salinity share is
salinity's part of the summed stabilising contributions of a season's casts (negative, that is
destabilising, parts are set to zero), so a season whose temperature difference destabilises the
column reads 1.00. Thresholds are descriptive choices, stated with the table: stratified when
bottom is denser by more than 0.5 kg/m3, mixed when the difference is under 0.1 kg/m3. Salinity is
practical salinity S_P, which is dimensionless; the column names keep "psu" as identifiers.

It also counts the casts per station from 24 July 2024 (every bin, before the loader's filters),
which reproduces the by-station block of reports/bay_twin/gen2_profile_audit.json, and writes that
inventory as Chapter 3's station table.

Writes reports/bay_twin/runs/v2_ctd_sections.json, thesis_template/images/figures/
fig_bay_ctd_sections.png, thesis_template/content/tables/table_bay_ctd_stratification.tex and
thesis_template/content/tables/table_bay_ctd_stations.tex.
Usage: python -m scripts.v2_ctd_sections
"""
import json

import gsw
import matplotlib
matplotlib.use('Agg')
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from scripts.v2_timeseries_figures import BLUE, GRID, INK, MUTED, ORANGE, ROOT, THESIS
from src.data.thermodynamics import o2sol_garcia_gordon_umol_per_l
from src.units import OXYGEN_UNIT_TEX, OXYGEN_UNIT_TEXT, column_decimals, factor

CASTS = ROOT / 'data/processed/ctd_all_casts.csv.gz'
PUBLISHED = ROOT / 'reports/bay_twin/runs/v2_ctd_sections.json'
OUT = PUBLISHED
#: the same report with every cast's oxygen in mg/l by its own source unit (addendum 60 part C2,
#: scripts.v2_oxygen_mgl_casts, which runs this module with OUT redirected and its inputs already in
#: mg/l). A run that writes the published report prints and draws its oxygen in mg/l from that rerun.
OXYGEN_MGL = ROOT / 'reports/bay_twin/runs/v2_oxygen_mgl/C2/v2_ctd_sections.json'
#: colour limits of the oxygen section, mg/l
OXYGEN_LIMITS = (5.5, 9.5)
DECLINES = ROOT / 'reports/bay_twin/runs/v2_decline_types.json'
TABLE = ROOT / 'thesis_template/content/tables/table_bay_ctd_stratification.tex'
STATION_TABLE = ROOT / 'thesis_template/content/tables/table_bay_ctd_stations.tex'
PROFILE_AUDIT = ROOT / 'reports/bay_twin/gen2_profile_audit.json'
FIGURE = THESIS / 'fig_bay_ctd_sections.png'
DECK = ROOT / 'presentation/figures_v2'
DECK_FIGURE = DECK / 'fig_bay_ctd_sections.pdf'
TARGETS = ['temp_c', 'salinity_psu', 'do_umol_l']
SURFACE = (0.0, 1.25)
BOTTOM = {'shallow': (6.0, 7.5), 'deep': (19.0, 21.0)}
STRATIFIED, MIXED = 0.5, 0.1            # kg/m3, bottom minus surface potential density
HALF_WIDTH = pd.Timedelta(days=10.5)
GEN2_START = pd.Timestamp('2024-07-24', tz='UTC')
SEASON = {12: 'DJF', 1: 'DJF', 2: 'DJF', 3: 'MAM', 4: 'MAM', 5: 'MAM',
          6: 'JJA', 7: 'JJA', 8: 'JJA', 9: 'SON', 10: 'SON', 11: 'SON'}
SEASONS = ['DJF', 'MAM', 'JJA', 'SON']
SEASON_NAME = {'DJF': 'December to February', 'MAM': 'March to May', 'JJA': 'June to August',
               'SON': 'September to November'}
#: (key, station, bottom layer) summarised by season; the 8 to 11 m stations share one bottom layer
SUMMARIES = [('PBaia', 'PBaia', 'shallow'), ('PBoaS', 'PBoaS', 'shallow'), ('PMitiliE', 'PMitiliE', 'shallow'),
             ('PBoaT', 'PBoaT', 'shallow'), ('PBoaT_deep', 'PBoaT', 'deep')]
SHALLOW_STATIONS = ('PBaia', 'PBoaS', 'PMitiliE')
#: sonde name prefix -> CTD station at the same site
SONDE_STATION = {'TBaiaP': 'PBaia', 'TMitiliE': 'PMitiliE', 'TMitiliO': 'PMitiliO',
                 'TPortovenereS': 'PPortovenereS', 'TPortovenereP': 'PPortovenereP'}

#: drawn near its printed size (the page caps it at about 13 x 17 cm), so the lettering prints
#: at 7 pt or more rather than being scaled down from a large canvas
def rho_eos80(salinity_psu, temp_c):
    """EOS-80 density at one atmosphere (UNESCO 1981), from practical salinity and ITS-90 temperature.

    The check on TEOS-10 (2026-09-25): colleagues at INRiM dispute TEOS-10 over how salinity is
    measured. Its Absolute Salinity S_A (g/kg) is the Reference Salinity S_R = (35.16504/35) g/kg
    times the measured practical salinity S_P, plus an anomaly delta S_A estimated from a global
    relation with silicate (here 0.002 to 0.003 g/kg: gsw.SA_from_SP minus gsw.SR_from_SP). EOS-80
    uses the practical salinity alone. At 25 m the pressure effect on potential temperature is
    under a thousandth of a degree, so sigma_t stands in for potential density.
    """
    t = 1.00024 * np.asarray(temp_c, dtype=float)            # IPTS-68, as EOS-80 is defined
    s = np.asarray(salinity_psu, dtype=float)
    rw = (999.842594 + 6.793952e-2 * t - 9.095290e-3 * t ** 2 + 1.001685e-4 * t ** 3
          - 1.120083e-6 * t ** 4 + 6.536332e-9 * t ** 5)
    a = 8.24493e-1 - 4.0899e-3 * t + 7.6438e-5 * t ** 2 - 8.2467e-7 * t ** 3 + 5.3875e-9 * t ** 4
    b = -5.72466e-3 + 1.0227e-4 * t - 1.6546e-6 * t ** 2
    return rw + a * s + b * s ** 1.5 + 4.8314e-4 * s ** 2


# UNESCO (1983) check values, at IPTS-68 temperatures: a transcription error fails here, not in a table
assert np.allclose([rho_eos80(35, 5 / 1.00024), rho_eos80(35, 25 / 1.00024)], [1027.67547, 1023.34306],
                   atol=1e-5)

plt.rcParams.update({'font.size': 10, 'axes.titlesize': 10, 'legend.fontsize': 9,
                     'xtick.labelsize': 9, 'ytick.labelsize': 9, 'axes.labelsize': 10,
                     'axes.spines.top': False, 'axes.spines.right': False})


def load():
    d = pd.read_csv(CASTS)
    d['t_utc'] = pd.to_datetime(d.t_utc, utc=True)
    starts = d.groupby('cast_id').t_utc.first()
    keep = ((d.qc_flag == 0) & np.isfinite(d[TARGETS]).all(axis=1) & d.temp_c.between(-2, 40)
            & d.salinity_psu.between(0, 42) & d.do_umol_l.between(0, 600))
    d = d[keep].copy()
    p = gsw.p_from_z(-d.depth_m.to_numpy(), d.lat.to_numpy())
    d['SA'] = gsw.SA_from_SP(d.salinity_psu.to_numpy(), p, d.lon.to_numpy(), d.lat.to_numpy())
    d['CT'] = gsw.CT_from_t(d.SA.to_numpy(), d.temp_c.to_numpy(), p)
    d['sigma0'] = gsw.sigma0(d.SA.to_numpy(), d.CT.to_numpy())
    d['SR'] = gsw.SR_from_SP(d.salinity_psu.to_numpy())    # absolute salinity without the atlas anomaly
    d['sigma_t80'] = rho_eos80(d.salinity_psu, d.temp_c) - 1000
    # percent saturation against the thesis's own solubility (Garcia & Gordon 1992, practical salinity)
    d['o2_sol'] = o2sol_garcia_gordon_umol_per_l(d.temp_c.to_numpy(), d.salinity_psu.to_numpy())
    d['o2_sat'] = 100 * d.do_umol_l / d.o2_sol
    d['date'] = d.t_utc.dt.tz_localize(None).dt.normalize()
    return d, starts


def pairs(d, station, bottom):
    """One row per cast: surface and bottom layer means and their differences."""
    x = d[d.station == station]
    cols = TARGETS + ['SA', 'CT', 'sigma0', 'SR', 'sigma_t80', 'o2_sol', 'o2_sat']
    lo, hi = BOTTOM[bottom]
    top = x[x.depth_m.between(*SURFACE)].groupby('cast_id')[cols].mean()
    bot = x[x.depth_m.between(lo, hi)].groupby('cast_id')[cols].mean()
    j = top.join(bot, lsuffix='_s', rsuffix='_b', how='inner')
    j = j.join(x.groupby('cast_id').agg(t=('t_utc', 'first'), date=('date', 'first')))
    sa, ct = (j.SA_s + j.SA_b) / 2, (j.CT_s + j.CT_b) / 2
    rho = gsw.rho(sa.to_numpy(), ct.to_numpy(), 0)
    alpha, beta = gsw.alpha(sa.to_numpy(), ct.to_numpy(), 0), gsw.beta(sa.to_numpy(), ct.to_numpy(), 0)
    j['dT'] = j.temp_c_s - j.temp_c_b                     # surface minus bottom
    j['dS'] = j.salinity_psu_s - j.salinity_psu_b
    j['dO'] = j.do_umol_l_s - j.do_umol_l_b
    j['dsig'] = j.sigma0_b - j.sigma0_s                    # bottom minus surface: positive is stable
    j['dsig_T'] = -rho * alpha * (j.CT_b - j.CT_s)
    j['dsig_S'] = rho * beta * (j.SA_b - j.SA_s)
    # the same difference and split under EOS-80, from practical salinity and in-situ temperature
    sm, tm = (j.salinity_psu_s + j.salinity_psu_b) / 2, (j.temp_c_s + j.temp_c_b) / 2
    j['dsig80'] = j.sigma_t80_b - j.sigma_t80_s
    j['dsig80_T'] = rho_eos80(sm, j.temp_c_b) - rho_eos80(sm, j.temp_c_s)
    j['dsig80_S'] = rho_eos80(j.salinity_psu_b, tm) - rho_eos80(j.salinity_psu_s, tm)
    j['dSat'] = j.o2_sat_s - j.o2_sat_b                    # percentage points, surface minus bottom
    j['dSol'] = j.o2_sol_s - j.o2_sol_b                    # how much of dO solubility alone would give
    j['season'] = j.t.dt.month.map(SEASON)
    return j


def eos_sensitivity(tables, d):
    """Does anything reported here depend on TEOS-10 rather than EOS-80?"""
    j = pd.concat(tables.values())
    classes = lambda v: np.select([v > STRATIFIED, v.abs() < MIXED], ['stratified', 'mixed'], 'between')
    changed = int((classes(j.dsig) != classes(j.dsig80)).sum())
    share_diff = []
    for key, t in tables.items():
        for season in SEASONS:
            g = t[t.season == season]
            if not len(g):
                continue
            a = salinity_share(g)
            s, th = g.dsig80_S.clip(lower=0).sum(), g.dsig80_T.clip(lower=0).sum()
            b = float(s / (s + th)) if s + th > 0 else None
            if a is not None and b is not None:
                share_diff.append(abs(a - b))
    diff = (j.dsig80 - j.dsig).abs()
    return {'pairs_compared': int(len(j)),
            'abs_difference_kg_m3': {'median': round(float(diff.median()), 4), 'max': round(float(diff.max()), 4)},
            'casts_changing_class': changed,
            'salinity_share_max_abs_difference': round(float(max(share_diff)), 3),
            'absolute_salinity_anomaly_g_kg': {'min': round(float((d.SA - d.SR).min()), 4),
                                               'max': round(float((d.SA - d.SR).max()), 4)}}


def salinity_share(g):
    """Share of the stabilising contributions that comes from salinity (positive parts only)."""
    s, t = g.dsig_S.clip(lower=0).sum(), g.dsig_T.clip(lower=0).sum()
    return float(s / (s + t)) if s + t > 0 else None


def by_season(j):
    out = {}
    for season in SEASONS:
        g = j[j.season == season]
        if not len(g):
            continue
        out[season] = {
            'casts': int(len(g)),
            'dT_median': round(float(g.dT.median()), 2), 'dS_median': round(float(g.dS.median()), 2),
            'dsig_median': round(float(g.dsig.median()), 2), 'dO_median': round(float(g.dO.median()), 1),
            'dSat_median': round(float(g.dSat.median()), 1), 'dSol_median': round(float(g.dSol.median()), 1),
            'surface_saturation_median': round(float(g.o2_sat_s.median()), 1),
            'dT_p10_p90': [round(float(g.dT.quantile(.1)), 2), round(float(g.dT.quantile(.9)), 2)],
            'share_stratified': round(float((g.dsig > STRATIFIED).mean()), 2),
            'share_mixed': round(float((g.dsig.abs() < MIXED).mean()), 2),
            'salinity_share': round(salinity_share(g), 2) if salinity_share(g) is not None else None,
        }
    return out


#: the squared buoyancy frequency of the idealised stratified runs in Porciello et al. (2015), s^-2
MODEL_N2 = 1e-3


def buoyancy_frequency(tables):
    """N^2 = g * dsigma / (rho * dz) between the surface and bottom layers of each cast, by season:
    the layering expressed as the idealised circulation model of the gulf states its stratified runs."""
    g = 9.81
    out = {}
    for key, _, bottom in SUMMARIES:
        j = tables.get(key)
        if j is None or not len(j):
            continue
        lo, hi = BOTTOM[bottom]
        dz = (lo + hi) / 2 - sum(SURFACE) / 2
        rho = 1000 + (j.sigma0_s + j.sigma0_b) / 2
        n2 = g * j.dsig / (rho * dz)
        block = {'layer_separation_m': round(dz, 3), 'seasons': {}}
        for season in SEASONS:
            v = n2[j.season == season]
            if len(v):
                block['seasons'][season] = {'casts': int(len(v)), 'median': float(f'{v.median():.3g}'),
                                            'p90': float(f'{v.quantile(.9):.3g}'),
                                            'share_at_least_model': round(float((v >= MODEL_N2).mean()), 2)}
        top = n2.idxmax()
        block['max'] = {'value': float(f'{n2.max():.3g}'), 'date': str(j.loc[top, 'date'].date()),
                        'season': j.loc[top, 'season']}
        out[key] = block
    return {'model_value_s2': MODEL_N2, 'by_summary': out}


def density_extremes(tables):
    """The largest bottom-minus-surface density difference in each summary, with its cast date."""
    out = {}
    for key, j in tables.items():
        if not len(j):
            continue
        row = j.loc[j.dsig.idxmax()]
        out[key] = {'dsig_max': round(float(row.dsig), 2), 'date': str(row.date.date()),
                    'season': row.season}
    top = max(out, key=lambda k: out[k]['dsig_max'])
    return {'by_summary': out, 'largest': {'summary': top, **out[top]}}


def station_inventory():
    """Casts per station from the first moored observation, every bin, before the loader's filters.

    This is the date-era subset of gen2_profile_audit.json (2026 included); it is recomputed here so
    that the table which prints it cites an artifact with a producer, and checked against the audit."""
    d = pd.read_csv(CASTS)
    d['t_utc'] = pd.to_datetime(d.t_utc, utc=True)
    g = d[d.t_utc >= GEN2_START]
    rows = []
    for station, s in g.groupby('station'):
        beds = sorted(s.groupby('cast_id').seabed_m.first().dropna().unique())
        rows.append({'station': station, 'casts': int(s.cast_id.nunique()),
                     'casts_before_2026': int(s[s.t_utc < pd.Timestamp('2026-01-01', tz='UTC')].cast_id.nunique()),
                     'min_depth_m': float(s.depth_m.min()), 'max_depth_m': float(s.depth_m.max()),
                     'seabed_m': [float(b) for b in beds]})
    rows.sort(key=lambda r: (-r['casts'], r['station']))
    audit = json.loads(PROFILE_AUDIT.read_text())
    ref = {r['station']: r for r in audit['by_station']}
    same = all(ref[r['station']]['casts'] == r['casts']
               and ref[r['station']]['min_depth_m'] == r['min_depth_m']
               and ref[r['station']]['max_depth_m'] == r['max_depth_m'] for r in rows)
    return {'from': str(GEN2_START.date()), 'casts': int(g.cast_id.nunique()), 'bins': int(len(g)),
            'stations': rows, 'matches_gen2_profile_audit': bool(same and audit['casts'] == g.cast_id.nunique()
                                                                 and audit['bins'] == len(g))}


def station_table(inv):
    rows = []
    varied = [r['station'] for r in inv['stations'] if len(r['seabed_m']) > 1]
    note = (f"{', '.join(varied)} {'has' if len(varied) == 1 else 'have'} more than one seabed depth "
            'in the cast metadata. ') if varied else ''
    for r in inv['stations']:
        bed = r['seabed_m']
        bed_txt = f'{bed[0]:g}' if len(bed) == 1 else f'{bed[0]:g} to {bed[-1]:g}'
        rows.append(f"{r['station']:<14s} & {r['casts']} & {r['casts_before_2026']} & "
                    f"{r['min_depth_m']:.2f} to {r['max_depth_m']:.2f} & {bed_txt} \\\\")
    return (
        '\\begin{table}[htbp]\n\\centering\n'
        '\\caption{CTD casts by station, 24 July 2024 to June 2026.}\n'
        '\\label{tab:bay_ctd_stations}\n'
        '\\begin{tabular}{@{}lrrrr@{}}\n\\toprule\n'
        '\\textbf{Station} & \\textbf{Casts} & \\textbf{Before 2026} & \\textbf{Binned depths (m)} & '
        '\\textbf{Seabed (m)} \\\\\n\\midrule\n'
        + '\n'.join(rows) + '\n\\midrule\n'
        f"All & {inv['casts']} & {sum(r['casts_before_2026'] for r in inv['stations'])} & & \\\\\n"
        '\\bottomrule\n\\end{tabular}\n\n'
        '\\vspace{2pt}\n\\footnotesize\\justifying\\noindent Every cast and $0.5$\\,m depth bin, before the\n'
        f'loader\'s filters; seabed depths as recorded. {note.rstrip()}'
        '\\artifact{runs/v2_ctd_sections.json}\\artifact{gen2_profile_audit.json}\n'
        '\\end{table}\n')


def oxygen_layering(tables):
    j = pd.concat([tables[k] for k in SHALLOW_STATIONS])
    salty = j[(j.dsig > STRATIFIED) & (j.dsig_S > j.dsig_T)]
    mixed = j[j.dsig.abs() < MIXED]

    def describe(g):
        return {'casts': int(len(g)), 'dO_median': round(float(g.dO.median()), 1),
                'dO_iqr': [round(float(g.dO.quantile(.25)), 1), round(float(g.dO.quantile(.75)), 1)],
                'share_bottom_lower': round(float((g.dO > 0).mean()), 2),
                'dS_median': round(float(g.dS.median()), 2),
                # the same contrast in saturation, and the part of dO that solubility alone explains
                'dSat_median': round(float(g.dSat.median()), 1),
                'dSat_iqr': [round(float(g.dSat.quantile(.25)), 1), round(float(g.dSat.quantile(.75)), 1)],
                'share_bottom_less_saturated': round(float((g.dSat > 0).mean()), 2),
                'dSol_median': round(float(g.dSol.median()), 1)}

    return {'stations': list(SHALLOW_STATIONS), 'salinity_stratified': describe(salty), 'mixed': describe(mixed),
            'salinity_stratified_months': {int(k): int(v) for k, v in
                                           salty.t.dt.month.value_counts().sort_index().items()}}


def declines(d):
    """Season of the typed declines (addendum 37) and the casts that precede them within a week."""
    typed = json.loads(DECLINES.read_text())['development']['types']
    months = {k: {int(m): n for m, n in v['months'].items()} for k, v in typed.items()}
    # the same month windows as Chapter 5's account of the two kinds
    out = {'water_mass': {'n': typed['water_mass']['n'],
                          'february_april': sum(n for m, n in months['water_mass'].items() if m in (2, 3, 4))},
           'in_place': {'n': typed['in_place']['n'],
                        'june_october': sum(n for m, n in months['in_place'].items() if m in range(6, 11))}}
    import scripts.v2_decline_types as types        # reads the sonde exports; imported only here
    episodes = types.episodes(*types.DEV)
    tables = {st: pairs(d, st, 'shallow') for st in set(SONDE_STATION.values())}
    near = []
    for _, e in episodes.iterrows():
        station = SONDE_STATION.get(e.sensor.split('_')[0])
        t = tables.get(station)
        if t is None or not len(t):
            continue
        w = t[(t.t < e.onset) & (t.t >= e.onset - pd.Timedelta(days=7))]
        if len(w):
            c = w.sort_values('t').iloc[-1]
            near.append({'sensor': e.sensor, 'onset': str(e.onset), 'type': e.type, 'station': station,
                         'cast_date': str(c.date.date()),
                         'cast_days_before': round((e.onset - c.t).total_seconds() / 86400, 1),
                         'cast_dsig': round(float(c.dsig), 2), 'cast_dS': round(float(c.dS), 2),
                         'cast_dO': round(float(c.dO), 1),
                         'cast_salinity_dominated': bool(c.dsig_S > c.dsig_T)})
    out['casts_within_7_days_before'] = near
    water_mass = [r for r in near if r['type'] == 'water_mass']
    out['water_mass_with_a_cast'] = len(water_mass)
    out['water_mass_casts_distinct'] = len({(r['station'], r['cast_date']) for r in water_mass})
    return out


def column_edges(dates):
    dates = pd.DatetimeIndex(dates)
    mids = dates[:-1] + (dates[1:] - dates[:-1]) / 2
    left = [dates[0] - HALF_WIDTH] + [max(m, t - HALF_WIDTH) for m, t in zip(mids, dates[1:])]
    right = [min(m, t + HALF_WIDTH) for m, t in zip(mids, dates[:-1])] + [dates[-1] + HALF_WIDTH]
    return left, right


def section(ax, d, station, var, cmap, vmin, vmax, zmax, label, title):
    tab = (d[(d.station == station) & (d.depth_m <= zmax)]
           .groupby(['date', 'depth_m'])[var].mean().unstack('depth_m').sort_index())
    z = tab.columns.to_numpy()
    zedge = np.concatenate([[z[0] - 0.25], z + 0.25])
    norm = plt.Normalize(vmin, vmax)
    for left, right, (_, row) in zip(*column_edges(tab.index), tab.iterrows()):
        ax.pcolormesh(mdates.date2num([left, right]), zedge, row.to_numpy()[:, None], cmap=cmap, norm=norm,
                      shading='flat', rasterized=True)
    ax.plot(tab.index, np.zeros(len(tab)), '|', color=INK, ms=4, mew=0.6, clip_on=False)
    ax.set_ylim(zedge[-1], 0)
    ax.grid(False)
    ax.set_ylabel('Depth (m)')
    ax.set_title(title, loc='left')
    bar = plt.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax, pad=0.012, fraction=0.035,
                       extend='both')
    bar.set_label(label)
    return len(tab)


def stratification_panel(ax, j):
    g = j.groupby('date')[['dsig', 'dsig_T', 'dsig_S']].mean()
    # the mixed range as a fill a reader sees: above the grid, below the zero line and the data
    ax.axhspan(-MIXED, MIXED, color=MIXED_FILL, lw=0, zorder=1)
    ax.axhline(0, color=MUTED, lw=0.5, zorder=1.5)
    ax.axhline(STRATIFIED, color=MUTED, lw=0.6, ls=':', zorder=1.5)
    ax.plot(g.index, g.dsig_S, color=BLUE, lw=0.9, marker='o', ms=2.0, label='from salinity', zorder=3)
    ax.plot(g.index, g.dsig_T, color=ORANGE, lw=0.9, ls='--', marker='s', ms=2.0, label='from temperature', zorder=3)
    ax.plot(g.index, g.dsig, color=INK, lw=0.0, marker='.', ms=3.0, label='total', zorder=4)
    ax.set_ylabel(r'$\Delta\sigma_0$ (kg/m$^3$)')
    ax.set_title('(e) PBaia: density difference, bottom (6–7.5 m) minus surface (0–1.25 m)', loc='left')
    ax.legend(loc='upper left', ncol=3, frameon=False)


def oxygen_mgl(d):
    """The casts' oxygen in mg/l, each bin by its own source unit as the part-C rerun converts it
    (scripts.v2_oxygen_mgl_common.convert_casts). A run that the rerun redirects reads mg/l already."""
    if OUT != PUBLISHED:
        return d.do_umol_l
    from scripts.v2_oxygen_mgl_common import convert_casts
    return convert_casts(d).do_umol_l


#: the section panels: station, variable, colour map, limits, depth, colour-bar label, title
SPEC = [('PBaia', 'temp_c', 'magma', 11, 28, 8, 'Temperature (°C)', '(a) PBaia: temperature'),
        ('PBaia', 'salinity_psu', 'viridis', 35, 38.4, 8, 'Practical salinity', '(b) PBaia: salinity'),
        ('PBaia', 'do_umol_l', 'cividis', *OXYGEN_LIMITS, 8, f'Oxygen ({OXYGEN_UNIT_TEXT})', '(c) PBaia: oxygen'),
        ('PBoaT', 'temp_c', 'magma', 11, 28, 25, 'Temperature (°C)', '(d) PBoaT: temperature')]
TIME_LIMITS = (pd.Timestamp('2021-06-01'), pd.Timestamp('2026-07-15'))
#: The thesis figure is drawn at the size it prints at (0.76 \textheight high, which sets its width through its
#: aspect) and saved without cropping, so that no text prints below 7 pt and the page does not move.
SIZE_IN = (4.7136, 6.2211)
PRINT_STYLE = {'font.size': 7.5, 'axes.titlesize': 7.5, 'axes.labelsize': 7.5, 'xtick.labelsize': 7,
               'ytick.labelsize': 7, 'legend.fontsize': 7}
#: the mixed range of panel (e): a grey dark enough to show as a band at the printed height
MIXED_FILL = '#c4c2bb'


def figure(d, tables):
    d = d.assign(do_umol_l=oxygen_mgl(d))             # the column keeps its name; its values are mg/l
    with plt.rc_context(PRINT_STYLE):
        fig, axes = plt.subplots(5, 1, figsize=SIZE_IN, sharex=True,
                                 gridspec_kw={'height_ratios': [1, 1, 1, 1.2, 1.35]})
        spec = SPEC
        dates = {}
        for ax, (station, var, cmap, vmin, vmax, zmax, label, title) in zip(axes, spec):
            dates[(station, var)] = section(ax, d, station, var, cmap, vmin, vmax, zmax, label, title)
        stratification_panel(axes[-1], tables['PBaia'])
        # an invisible colourbar keeps the last panel's axis as wide as the four maps above it
        spacer = plt.colorbar(plt.cm.ScalarMappable(), ax=axes[-1], pad=0.012, fraction=0.035)
        spacer.ax.set_visible(False)
        axes[-1].xaxis.set_major_locator(mdates.YearLocator())
        axes[-1].xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
        axes[-1].xaxis.set_minor_locator(mdates.MonthLocator(bymonth=[7]))
        axes[-1].set_xlim(*mdates.date2num(list(TIME_LIMITS)))
        fig.align_ylabels(axes)
        fig.tight_layout(h_pad=0.6)
        fig.savefig(FIGURE, dpi=300)
        plt.close(fig)
    return {f'{s}/{v}': n for (s, v), n in dates.items()}


def deck_figure(d):
    """The three PBaia panels alone, for the slide on the water column, with the time axis drawn under
    them: the year at each January, July marked, every month ticked. In the full figure only the
    density panel carries the dates, so a slide that shows the PBaia panels needs its own axis."""
    d = d.assign(do_umol_l=oxygen_mgl(d))
    # drawn at the shape and near the size of its box on the slide (about 215 x 150 pt), so the text\n    # is not shrunk below legibility
    with plt.rc_context({'font.size': 7, 'axes.titlesize': 7.5, 'axes.labelsize': 7,
                         'xtick.labelsize': 6.5, 'ytick.labelsize': 6.5}):
        fig, axes = plt.subplots(3, 1, figsize=(3.75, 3.1), sharex=True)
        # each panel's title names the variable, so its colour bar carries only the unit
        units = ('°C', 'salinity', OXYGEN_UNIT_TEXT)
        for ax, unit, (station, var, cmap, vmin, vmax, zmax, _, title) in zip(axes, units, SPEC[:3]):
            section(ax, d, station, var, cmap, vmin, vmax, zmax, unit, title)
        _deck_axis(fig, axes)
    # the deck verifier checks that every figure it includes records the artifacts it was drawn from
    (DECK / 'manifest_ctd.json').write_bytes((json.dumps(
        {'fig_bay_ctd_sections': ['data/processed/ctd_all_casts.csv.gz',
                                  'reports/bay_twin/runs/v2_ctd_sections.json']}, indent=2) + '\n').encode('utf-8'))


def _deck_axis(fig, axes):
    """The year at each January, July marked, every month ticked; then save the slide figure."""
    axis = axes[-1].xaxis
    axis.set_major_locator(mdates.MonthLocator(bymonth=[1, 7]))
    axis.set_major_formatter(plt.FuncFormatter(
        lambda x, _: mdates.num2date(x).strftime('%Y' if mdates.num2date(x).month == 1 else '%b')))
    axis.set_minor_locator(mdates.MonthLocator())
    axes[-1].tick_params(axis='x', which='major', length=4)
    axes[-1].tick_params(axis='x', which='minor', length=1.5)
    axes[-1].set_xlim(*mdates.date2num(list(TIME_LIMITS)))
    fig.canvas.draw()
    for label in axes[-1].get_xticklabels():
        if not label.get_text().isdigit():
            label.set_color(MUTED)
    fig.align_ylabels(axes)
    fig.tight_layout(h_pad=0.6)
    DECK.mkdir(parents=True, exist_ok=True)
    fig.savefig(DECK_FIGURE, bbox_inches='tight')
    plt.close(fig)


def fmt(v, digits, sign=True):
    """Signed values in math mode, so a negative prints a minus sign rather than a hyphen."""
    if v is None:
        return '---'
    return f'${v:+.{digits}f}$' if sign else f'{v:.{digits}f}'


def mg_digits(values_mg, decimals):
    """Decimals for oxygen values already in mg/l whose report printed them at `decimals` in its own
    unit: the significant figures they would have had there (src.units.column_decimals)."""
    return column_decimals([v / factor() for v in values_mg if v is not None], decimals)


def table(report):
    # oxygen in mg/l: from the part-C rerun when this run writes the published report, else (the rerun
    # itself) from the report just computed
    oxygen = json.loads(OXYGEN_MGL.read_text()) if OUT == PUBLISHED else report
    keys = (('PBaia', 'PBaia, 0--1.25 against 6--7.5\\,m'), ('PBoaT_deep', 'PBoaT, 0--1.25 against 19--21\\,m'))
    d_season = mg_digits([oxygen['seasons'][key][season]['dO_median'] for key, _ in keys for season in SEASONS
                          if season in report['seasons'][key]], 1)
    rows = []
    for key, name in keys:
        rows.append(f'\\multicolumn{{9}}{{@{{}}l}}{{\\textit{{{name}}}}} \\\\')
        for season in SEASONS:
            s = report['seasons'][key].get(season)
            if s is None:
                continue
            rows.append(' & '.join([SEASON_NAME[season], str(s['casts']), fmt(s['dT_median'], 2),
                                    fmt(s['dS_median'], 2), fmt(s['dsig_median'], 2),
                                    fmt(s['salinity_share'], 2, sign=False), fmt(s['share_stratified'], 2, sign=False),
                                    fmt(s['share_mixed'], 2, sign=False),
                                    fmt(oxygen['seasons'][key][season]['dO_median'], d_season)]) + ' \\\\')
    sources = ('\\artifact{runs/v2_ctd_sections.json}\\artifact{runs/v2_oxygen_mgl/C2/v2_ctd_sections.json}'
               if OUT == PUBLISHED else '\\artifact{runs/v2_ctd_sections.json}')
    # The layers, the thresholds and the salinity share are defined in Chapter 3's text, which the
    # note points to; the oxygen contrasts under salinity layering and the EOS-80 check are stated there.
    return (
        '\\begin{table}[htbp]\n\\centering\n'
        '\\caption{Water-column structure by season, CTD archive, 2021 to June 2026.}\n'
        '\\label{tab:bay_ctd_stratification}\n'
        '\\footnotesize\\setlength{\\tabcolsep}{4pt}\n'
        '\\begin{tabular}{@{}lrrrrrrrr@{}}\n\\toprule\n'
        '\\textbf{Season} & \\textbf{Casts} & \\makecell[r]{$\\Delta T$\\\\($^\\circ$C)} & '
        '\\makecell[r]{$\\Delta S_{\\mathrm{P}}$\\\\\\phantom{(}} & \\makecell[r]{$\\Delta\\sigma_0$\\\\(kg/m$^3$)} & '
        '\\makecell[r]{Salinity\\\\share} & \\makecell[r]{Share\\\\stratified} & \\makecell[r]{Share\\\\mixed} & '
        f'\\makecell[r]{{$\\Delta$O$_2$\\\\({OXYGEN_UNIT_TEX})}} \\\\\n\\midrule\n'
        + '\n'.join(rows) + '\n\\bottomrule\n\\end{tabular}\n\n'
        '\\vspace{2pt}\n\\scriptsize\\justifying\\noindent Medians over casts of surface minus bottom\n'
        '($\\Delta\\sigma_0$: bottom minus surface). Layers, the stratified and mixed thresholds and\n'
        'the salinity share as defined in Section~\\ref{subsec:data_ctd_structure}; practical salinity is\n'
        f'dimensionless.{sources}\n\\end{{table}}\n')


def main():
    d, starts = load()
    hours = starts.dt.hour + starts.dt.minute / 60
    tables = {key: pairs(d, station, bottom) for key, station, bottom in SUMMARIES}
    report = {
        'question': 'Supervisors, 2026-09-25: large-scale changes in the vertical structure; mixed '
                    'periods against periods when surface and bottom differ.',
        'descriptive': 'No model fitted, nothing scored, nothing declared in advance.',
        'source': 'data/processed/ctd_all_casts.csv.gz',
        'casts_in_archive': int(len(starts)), 'casts_used': int(d.cast_id.nunique()), 'bins_used': int(len(d)),
        'cast_dates_used': int(d.date.nunique()),
        'cast_start_share': {'07:00-11:00': round(float(((hours >= 7) & (hours <= 11)).mean()), 3),
                             '07:00-11:59': round(float(((hours >= 7) & (hours < 12)).mean()), 3)},
        # the 164 casts from the first moored observation on, which Chapter 3's inventory counts
        'cast_start_share_from_2024_07_24': {
            'casts': int((starts >= GEN2_START).sum()),
            '07:00-11:59': round(float(((hours >= 7) & (hours < 12))[starts >= GEN2_START].mean()), 3)},
        'layers_m': {'surface': list(SURFACE), **{k: list(v) for k, v in BOTTOM.items()}},
        'thresholds_kg_m3': {'stratified': STRATIFIED, 'mixed': MIXED},
        'seasons': {key: by_season(tables[key]) for key, _, _ in SUMMARIES},
        'oxygen_under_salinity_stratification': oxygen_layering(tables),
        'eos_sensitivity': eos_sensitivity(tables, d),
        'density_difference_extremes': density_extremes(tables),
        'buoyancy_frequency_squared': buoyancy_frequency(tables),
        'gen2_era_stations': station_inventory(),
        'declines': declines(d),
    }
    report['figure_columns'] = figure(d, tables)
    deck_figure(d)
    OUT.write_text(json.dumps(report, indent=2))
    TABLE.write_text(table(report), encoding='utf-8')
    STATION_TABLE.write_text(station_table(report['gen2_era_stations']), encoding='utf-8')
    print(json.dumps(report, indent=1))
    print('wrote', OUT.relative_to(ROOT), FIGURE.relative_to(ROOT), TABLE.relative_to(ROOT))


if __name__ == '__main__':
    main()
