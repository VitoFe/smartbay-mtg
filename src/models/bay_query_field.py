"""Joint observation-conditioned field with explicit thermodynamic/stability priors."""
import torch
from torch import nn


def solubility_torch(t, s):
    u=torch.log((298.15-t)/(273.15+t))
    a=2.00907+3.22014*u+4.05010*u**2+4.94457*u**3-.256847*u**4+3.88767*u**5
    b=-.00624097-.00693498*u-.00690358*u**2-.00429155*u**3
    return torch.exp(a+s*b-.00000031168*s*s)*44.661


def density_zero_pressure(t,s):
    """UNESCO atmospheric-pressure EOS approximation, not TEOS-10 N squared."""
    fresh=999.842594+.06793952*t-.00909529*t**2+.0001001685*t**3-.000001120083*t**4+.000000006536332*t**5
    a=.824493-.0040899*t+.000076438*t**2-.00000082467*t**3+.0000000053875*t**4
    b=-.00572466+.00010227*t-.0000016546*t**2
    return fresh+a*s+b*s.clamp_min(0)**1.5+.00048314*s*s


def stability_loss(pred, upper, lower, dz):
    rho=density_zero_pressure(pred[:,0],pred[:,1])
    return torch.relu((rho[upper]-rho[lower])/dz).square().mean()


class BayQueryField(nn.Module):
    def __init__(self, n_features, hidden=64, thermodynamic=True):
        super().__init__(); self.thermodynamic=thermodynamic
        self.net=nn.Sequential(nn.Linear(n_features,hidden),nn.Tanh(),nn.Linear(hidden,hidden),nn.Tanh(),nn.Linear(hidden,3))
        nn.init.zeros_(self.net[-1].weight); nn.init.zeros_(self.net[-1].bias)

    def forward(self,x,base):
        correction=torch.tanh(self.net(x))
        t=(base[:,0]+10*correction[:,0]).clamp(-2,40)
        s=(base[:,1]+5*correction[:,1]).clamp(0,42)
        if self.thermodynamic:
            aou=solubility_torch(base[:,0],base[:,1])-base[:,2]+100*correction[:,2]
            oxygen=solubility_torch(t,s)-aou
        else:
            oxygen=base[:,2]+100*correction[:,2]
        return torch.stack([t,s,oxygen],dim=1)
