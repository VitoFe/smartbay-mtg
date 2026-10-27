import numpy as np
import pandas as pd
import torch
from src.models.bay_query_field import BayQueryField,solubility_torch,stability_loss
from src.data.thermodynamics import o2sol_garcia_gordon_umol_per_l
from src.data.bay_profile_context import allowed_sources,latest_context


def test_differentiable_solubility_matches_reference_and_has_gradient():
    t=torch.tensor([5.,20.,30.],dtype=torch.float64,requires_grad=True); s=torch.tensor([30.,37.,40.],dtype=torch.float64)
    c=solubility_torch(t,s)
    np.testing.assert_allclose(c.detach().numpy(),o2sol_garcia_gordon_umol_per_l(t.detach().numpy(),s.numpy()),rtol=1e-12)
    c.sum().backward(); assert torch.isfinite(t.grad).all() and (t.grad<0).all()


def test_zero_correction_reconstructs_base_including_supersaturation():
    base=torch.tensor([[20.,37.,300.],[15.,36.,180.]])
    for thermo in [False,True]:
        model=BayQueryField(4,thermodynamic=thermo)
        torch.testing.assert_close(model(torch.ones(2,4),base),base)


def test_stability_prior_penalizes_dense_water_over_light_water():
    upper=torch.tensor([0]); lower=torch.tensor([1]); dz=torch.tensor([1.])
    stable=torch.tensor([[25.,37.,200.],[15.,37.,210.]])
    assert stability_loss(stable,upper,lower,dz)==0
    assert stability_loss(stable.flip(0),upper,lower,dz)>0


def test_query_and_held_station_colocation_masks_apply_together():
    xy=np.array([[0,0],[50,0],[1000,0],[1050,0],[2000,0]])
    np.testing.assert_array_equal(allowed_sources(xy,[0,0],[1000,0]),[False,False,False,False,True])


def test_context_never_reads_future_or_stale_observations():
    t=pd.date_range('2025-01-01',periods=10,freq='h',tz='UTC')
    f=pd.DataFrame({'temperature':[10.]+[np.nan]*8+[99.], 'oxygen':[np.nan]*8+[200.,999.]},index=t)
    z=latest_context(f,t[8]+pd.Timedelta(minutes=30))
    assert np.isnan(z[0]).all()
    np.testing.assert_allclose(z[1],[200.,.5])
