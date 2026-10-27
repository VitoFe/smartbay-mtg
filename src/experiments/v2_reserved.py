"""Second, labelled reading of the 2026 reserved period under reserved protocol v2.

bay_reserved_evaluation.py is contract-bound and hard-codes the v1 refit path inside
evaluate_forecast(); it is not edited. evaluate_forecast() below is that function verbatim with the
refit directory as an argument. `--refit v1` must reproduce the published v1 summary exactly before
the v2 reading is trusted. The model-free parts (temperature outage persistence, spatial CTD
abstention, warning rule) are called from the bound module unchanged.
"""
import argparse, hashlib, json
from pathlib import Path
import numpy as np
import pandas as pd
import lightgbm as lgb
from src.experiments import bay_reserved_evaluation as v1
from src.protocols.bay_calibration import finite_radius
from src.models.bay_sensor_inference import causal_features

ROOT = v1.ROOT
REFITS = {'v1': ROOT / 'reports/bay_twin/runs/20260918_reserved_refit/results/forecast',
          'v2': ROOT / 'reports/bay_twin/runs/v2_release_v2/refit_forecast'}


def evaluate_forecast(refit_root):
    meta, frames = v1.load_all_observations()
    eval_times = pd.date_range('2026-01-01T00:00:00', '2026-06-19T18:00:00', freq='6h', tz='UTC')
    weather_df = v1.load_weather_2026(eval_times)
    forecast_results = []; summary_by_var = {}
    for variable in ['temperature', 'oxygen']:
        cp = json.loads((refit_root / f'refit_{variable}_model.json').read_text()); cp['variable'] = variable
        model = lgb.Booster(model_str=cp['model_string']); alpha = cp['global_blend_weight']
        cal_df = pd.read_csv(refit_root / f'calibration_{variable}.csv.gz'); cal_df['time'] = pd.to_datetime(cal_df['time'], utc=True)
        cal_residuals = cal_df[['time', 'sensor', 'truth', 'prediction']].copy(); cal_residuals['error'] = cal_residuals['truth'] - cal_residuals['prediction']
        all_residuals = [cal_residuals]; rows = []
        for q in eval_times:
            current_frames = {n: f.loc[f.index <= q] for n, f in frames.items()}
            w_row = weather_df.loc[q].to_numpy()
            x, anchors, freshness = causal_features(current_frames, meta, cp, q, w_row)
            cur_history = pd.concat(all_residuals, ignore_index=True)
            avail_mask = cur_history['time'] + pd.Timedelta(hours=7) <= q
            recent_mask = avail_mask & (cur_history['time'] + pd.Timedelta(hours=7) >= q - pd.Timedelta(days=30))
            quarter = 'Q1' if q.month <= 3 else 'Q2'
            for s, sensor in enumerate(cp['names']):
                if sensor not in cp['contract']['supported_sensors']: continue
                anchor = anchors[s]; has_anchor = np.isfinite(anchor)
                target_frame = frames[sensor]; target_time = q + pd.Timedelta(hours=6)
                has_target = (target_time in target_frame.index) and (variable in target_frame) and pd.notna(target_frame.loc[target_time, variable])
                truth_val = float(target_frame.loc[target_time, variable]) if has_target else np.nan
                base = {'variable': variable, 'sensor': sensor, 'time': str(q), 'target_time': str(target_time), 'quarter': quarter,
                        'week': int(q.isocalendar().week), 'truth': truth_val}
                if has_anchor and has_target:
                    pred_inc = float(model.predict(x[s:s + 1])[0] * cp['scale']); raw_pred = float(anchor + pred_inc)
                    pred_val = float(anchor + alpha * (raw_pred - anchor)); persist_val = float(anchor)
                    sensor_recent = cur_history[recent_mask & (cur_history['sensor'] == sensor)]
                    sensor_cal = cur_history[avail_mask & (cur_history['sensor'] == sensor)]
                    pool = sensor_recent if len(sensor_recent) >= 30 else sensor_cal
                    has_interval = len(pool) >= 30; rad = finite_radius(pool['error'].to_numpy()) if has_interval else np.nan
                    rows.append({**base, 'has_anchor': True, 'has_target': True, 'prediction': pred_val, 'persistence': persist_val,
                                 'error': abs(pred_val - truth_val), 'persistence_error': abs(persist_val - truth_val), 'has_interval': has_interval,
                                 'radius': rad, 'covered': bool(abs(truth_val - pred_val) <= rad) if has_interval else None})
                    all_residuals.append(pd.DataFrame([{'time': target_time, 'sensor': sensor, 'truth': truth_val, 'prediction': pred_val, 'error': truth_val - pred_val}]))
                else:
                    rows.append({**base, 'has_anchor': has_anchor, 'has_target': has_target, 'prediction': np.nan,
                                 'persistence': float(anchor) if has_anchor else np.nan, 'error': np.nan, 'persistence_error': np.nan,
                                 'has_interval': False, 'radius': np.nan, 'covered': None})
        df_var = pd.DataFrame(rows); forecast_results.append(df_var); scored = df_var[df_var['error'].notna()]
        macro = float(scored.groupby(['quarter', 'sensor'])['error'].mean().groupby('quarter').mean().mean())
        p_macro = float(scored.groupby(['quarter', 'sensor'])['persistence_error'].mean().groupby('quarter').mean().mean())
        with_int = scored[scored['has_interval']]
        summary_by_var[variable] = {'scheduled_endpoints': len(df_var), 'scored_endpoints': len(scored), 'macro_mae': macro, 'persistence_macro_mae': p_macro,
                                    'mae_reduction': float(p_macro - macro), 'mae_reduction_pct': float(100 * (p_macro - macro) / p_macro) if p_macro > 0 else 0.0,
                                    'interval_coverage': float(with_int['covered'].mean()) if len(with_int) else np.nan,
                                    'interval_availability': float(scored['has_interval'].mean()) if len(scored) else 0.0,
                                    'interval_mean_width': float(2 * with_int['radius'].mean()) if len(with_int) else np.nan}
        print(f'[{variable}] 2026 MAE {macro:.4f} vs persistence {p_macro:.4f} ({len(scored)}/{len(df_var)})', flush=True)
    return pd.concat(forecast_results, ignore_index=True), summary_by_var


def main(which, full):
    out = ROOT / 'reports/bay_twin/runs' / f'v2_reserved_reading_{which}'; out.mkdir(parents=True, exist_ok=True)
    forecast_df, forecast_summary = evaluate_forecast(REFITS[which])
    forecast_df.to_csv(out / 'forecast_endpoints_2026.csv.gz', index=False, compression='gzip')
    summary = {'refit': which, 'refit_root': str(REFITS[which].relative_to(ROOT)), 'forecast': forecast_summary}
    if full:
        o, summary['temperature_outage'] = v1.evaluate_temperature_outages(); o.to_csv(out / 'outage_endpoints_2026.csv.gz', index=False)
        w, summary['oxygen_warnings'] = v1.evaluate_oxygen_warnings(); w.to_csv(out / 'oxygen_warnings_2026.csv', index=False)
    (out / 'summary.json').write_text(json.dumps(summary, indent=2))
    (out / 'verification.json').write_text(json.dumps({'results_sha256': {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(out.iterdir()) if p.name != 'verification.json'}}, indent=2))


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--refit', choices=['v1', 'v2'], required=True); p.add_argument('--full', action='store_true')
    a = p.parse_args(); main(a.refit, a.full)
