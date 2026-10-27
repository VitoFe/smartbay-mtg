"""Learned source dynamics with optional symmetric internal sensor-graph exchange.

Unit capacities are an architectural audit device, not surveyed bay cell volumes.
"""
import torch
from torch import nn


def observation_update(state,observations,available,denied):
    use=available & ~denied
    innovation=torch.where(use,observations-state,torch.zeros_like(state))
    return state+innovation,innovation


class BayGraphDynamics(nn.Module):
    def __init__(self,adjacency,n_forcing,hidden=32,exchange=True,relaxation=False):
        super().__init__();self.exchange=exchange;self.relaxation=relaxation
        a=torch.as_tensor(adjacency,dtype=torch.float32)
        if not torch.allclose(a,a.T) or (a<0).any():raise ValueError('Require symmetric nonnegative adjacency')
        a=a/a.sum(1).max().clamp_min(1.)
        self.register_buffer('adjacency',a);self.log_rate=nn.Parameter(torch.tensor(-3.))
        self.source=nn.Sequential(nn.Linear(n_forcing+3,hidden),nn.Tanh(),nn.Linear(hidden,1))
        nn.init.zeros_(self.source[-1].weight);nn.init.zeros_(self.source[-1].bias)
        if relaxation:self.log_relaxation=nn.Parameter(torch.tensor(-3.))

    def step(self,state,available,forcing,step):
        k=self.adjacency[None]*available[:,:,None]*available[:,None,:]
        neighbor=torch.einsum('bij,bj->bi',k,state)/k.sum(-1).clamp_min(1e-6)
        neighbor=torch.where(k.sum(-1)>0,neighbor,state)
        exchange=(k*(state[:,None,:]-state[:,:,None])).sum(-1)
        exchange=exchange*(.25*torch.sigmoid(self.log_rate) if self.exchange else 0.)
        features=torch.cat([state[...,None],neighbor[...,None],forcing,torch.full_like(state[...,None],step/6)],dim=-1)
        source=.2*torch.tanh(self.source(features).squeeze(-1))
        if self.relaxation:
            from src.models.bay_query_field import solubility_torch
            # Frozen issuance T/S; only apply where both are observed within six hours.
            temperature=20+10*forcing[...,10];salinity=35+5*forcing[...,12]
            equilibrium=(solubility_torch(temperature,salinity)-230)/50
            observed=(forcing[...,11]>0)&(forcing[...,13]>0)
            relaxation=.1*torch.sigmoid(self.log_relaxation)*(equilibrium-state)
            source=source+torch.where(observed,relaxation,torch.zeros_like(state))
        return state+exchange+source,exchange,source

    def rollout(self,state,available,forcing,future=None,future_mask=None,denied=None,hours=6):
        original=state
        source_sum=torch.zeros_like(state);innovation_sum=torch.zeros_like(state);exchange_sum=torch.zeros_like(state)
        max_exchange=torch.tensor(0.,device=state.device)
        for h in range(hours):
            state,exchange,source=self.step(state,available,forcing,h)
            exchange_sum=exchange_sum+exchange;source_sum=source_sum+source
            max_exchange=torch.maximum(max_exchange,exchange.sum(-1).abs().max())
            if future is not None:
                state,innovation=observation_update(state,future[:,h],future_mask[:,h],denied)
                innovation_sum=innovation_sum+innovation
                available=available | (future_mask[:,h] & ~denied)
        closure=(state-original-exchange_sum-source_sum-innovation_sum).abs().max()
        return state,{'max_internal_exchange_sum':max_exchange,'max_state_budget_residual':closure,
                      'mean_absolute_observation_increment':innovation_sum.abs().mean()}
