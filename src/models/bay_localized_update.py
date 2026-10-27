"""Portable observation update from a saved training calibration (no target labels)."""
import numpy as np


def localized_update(raw_context,boundary,calibration,filtered=False):
    """Return state and per-variable observation support; no uncertainty guarantee.

    Boundary columns: T,S,O absolute, then vertical differences relative to3m.
    When filtered=True, callers must supply causally filtered observation context.
    """
    boundary=np.asarray(boundary,dtype=float)
    if boundary.ndim!=2 or boundary.shape[1]<6 or not np.isfinite(boundary[:,:6]).all():
        raise ValueError('Finite absolute and vertical boundary values are required')
    fallback=calibration['contract']['training_fallback']
    background=np.empty((len(boundary),3))
    for j,record in enumerate(calibration['background']):
        if record['background']=='bias_corrected_coarse':background[:,j]=boundary[:,j]+record['coarse_bias']
        elif record['background']=='training_median_vertical':background[:,j]=fallback[j]+boundary[:,j+3]
        else:raise ValueError('Unknown background recipe')
    context=raw_context[['temperature_context','salinity_context','oxygen_context']].to_numpy(dtype=float)
    supported=np.isfinite(context)
    observation=np.where(supported,context+boundary[:,3:6],background)
    gain=np.asarray(calibration['gain']['filtered' if filtered else 'raw'])
    if gain.shape!=(3,) or not np.isfinite(gain).all() or ((gain<0)|(gain>1)).any():raise ValueError('Invalid update gains')
    state=background+gain*(observation-background)
    state[:,0]=np.clip(state[:,0],-2,40);state[:,1]=np.clip(state[:,1],0,42)
    return state,supported


def supported_field_state(learned,raw_context,boundary,calibration):
    """Retain learned estimates with context; use calibrated background otherwise.

    The boolean mask identifies observed-context support, not calibrated confidence.
    """
    background_calibration=dict(calibration,gain={'raw':[0.,0.,0.]})
    background,supported=localized_update(raw_context,boundary,background_calibration)
    learned=np.asarray(learned,dtype=float)
    if learned.shape!=background.shape or not np.isfinite(learned).all():
        raise ValueError('Finite learned predictions matching background shape required')
    return np.where(supported,learned,background),np.where(supported,'observation_supported','degraded_background_only')
