"""Explicit output contract for verified archived sensor forecasts."""
import math


def forecast_payload(row,metadata,provenance):
    """Whitelist outputs: future observed truth is never returned as an input/state."""
    supported=all(math.isfinite(float(row[k])) for k in ['lower','upper'])
    interval=None
    if supported:
        interval={'lower':float(row['lower']),'upper':float(row['upper']),'nominal_coverage':.9,
                  'policy':row['policy'],'status':row['interval_status'],'residual_count':int(row['residual_count']),
                  'latest_residual_available':str(row['latest_residual_available']),
                  'guarantee':'No guaranteed conditional coverage under temporal drift.'}
    value=float(row['prediction']);anchor=float(row['persistence'])
    return {'status':'EXPERIMENTAL_ARCHIVE_REPLAY','task':'known_sensor_six_hour_forecast',
        'variable':provenance['variable'],'units':'degC' if provenance['variable']=='temperature' else 'umol/L',
        'sensor':row['sensor'],'issuance':str(row['time']),'horizon_hours':6,
        'location':{'longitude':float(metadata['longitude']),'latitude':float(metadata['latitude']),'depth_m':float(metadata['nominal_depth_m'])},
        'observed_anchor':anchor,'prediction':value,'predicted_change':value-anchor,'predicted_change_per_hour':(value-anchor)/6,
        'interval':interval,'interval_status':row['interval_status'],'provenance':provenance,
        'support':'Recorded sensor location and depth only. Not a continuous spatial field or live deployment.',
        'warning':{'status':'not_issued','reason':'Point change is not a validated episode-alert or biological-stress decision.'}}
