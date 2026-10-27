"""Label-free inference features from observations available at issuance."""
import numpy as np
import pandas as pd


def causal_features(frames,metadata,checkpoint,stamp,weather_values):
    q=pd.Timestamp(stamp)
    if q.tzinfo is None:raise ValueError('Timezone required')
    q=q.tz_convert('UTC')
    if q.minute or q.second or q.microsecond or q.hour%6:raise ValueError('Current recipe supports six-hour issuance grid only')
    names=checkpoint['names'];variable=checkpoint['variable'];center=checkpoint['center'];scale=checkpoint['scale']
    meta=metadata.loc[names];index=pd.date_range(q-pd.Timedelta(hours=6),q,freq='h')
    def values(col):
        return pd.DataFrame({n:frames[n][col].reindex(index) if n in frames and col in frames[n] else np.nan for n in names},index=index)
    raw=values(variable);latest=(raw.ffill(limit=6)-center)/scale
    x=latest.iloc[-1].to_numpy();available=np.isfinite(x)
    z=(raw-center)/scale;trend=(z.iloc[-1]-z.iloc[0]).to_numpy()
    xy=meta[['x_m','y_m']].to_numpy();distance=np.linalg.norm(xy[:,None]-xy[None,:],axis=-1)
    adjacency=np.exp(-distance/1500);np.fill_diagonal(adjacency,0)
    static=np.column_stack([(xy[:,0]-568000)/5000,(xy[:,1]-4880000)/5000,meta.nominal_depth_m/10])
    calendar=np.array([np.sin(q.dayofyear*2*np.pi/365.25),np.cos(q.dayofyear*2*np.pi/365.25),np.sin(q.hour*2*np.pi/24),np.cos(q.hour*2*np.pi/24)])
    inputs=[np.nan_to_num(trend)[:,None],np.isfinite(trend)[:,None],available[:,None],static,np.broadcast_to(calendar,(len(names),4))]
    for col,offset,divisor in [('temperature',20,10),('salinity',35,5)]:
        a=values(col).ffill(limit=6).iloc[-1].to_numpy()
        inputs += [np.nan_to_num((a-offset)/divisor)[:,None],np.isfinite(a)[:,None]]
    forcing=np.concatenate(inputs,axis=-1).astype('float32')
    filled=np.where(available,x,np.asarray(checkpoint['fallback']))
    k=adjacency*available[:,None]*available[None,:]
    neighbor=np.einsum('ij,j->i',k,filled)/np.maximum(k.sum(-1),1e-6)
    neighbor=np.where(k.sum(-1)>0,neighbor,filled)
    result=np.concatenate([filled[:,None],neighbor[:,None],forcing,np.broadcast_to(np.asarray(weather_values),(len(names),5))],axis=-1)
    if result.shape[-1]!=checkpoint['feature_count']:raise ValueError('Checkpoint feature schema mismatch')
    ages=[]
    for n in names:
        seen=raw[n].dropna();ages.append(float((q-seen.index[-1])/pd.Timedelta(hours=1)) if len(seen) else None)
    return result,raw.iloc[-1].to_numpy(),{'source_names':names,'ages_hours':ages,'available_count':int(available.sum()),'latest_input_time':str(q)}
