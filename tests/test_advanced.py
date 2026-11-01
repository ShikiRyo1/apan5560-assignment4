import math
import torch
import pytest
from helper_lib.advanced_models import EnergyModel, UNet, TimeEmbedding, DiffusionSchedule, langevin, ddim_sample


def test_time_embedding_numeric():
    embedding = TimeEmbedding(8)(torch.tensor([1]))[0]
    frequencies = [10000**(-i/3) for i in range(4)]
    expected = [math.sin(f) for f in frequencies]+[math.cos(f) for f in frequencies]
    torch.testing.assert_close(embedding, torch.tensor(expected))


def test_unet_shapes_and_gradients():
    torch.set_num_threads(2)
    model = UNet()
    x = torch.randn(2,3,32,32)
    result = model(x,torch.tensor([0,999]))
    assert result.shape == x.shape
    result.square().mean().backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())


def test_input_gradients_not_parameter_accumulation():
    model = EnergyModel().eval()
    initial = torch.randn(2,3,32,32).clamp(-1,1)
    state = {k:v.clone() for k,v in model.state_dict().items()}
    result = langevin(model,initial,steps=3,noise_std=0)
    assert not result.requires_grad
    assert all(p.grad is None for p in model.parameters())
    assert all(torch.equal(v,state[k]) for k,v in model.state_dict().items())
    assert result.min()>=-1 and result.max()<=1
    assert not torch.equal(result,initial)


def test_langevin_quadratic_energy_lowers():
    class Quadratic(torch.nn.Module):
        def forward(self,x): return x.square().flatten(1).sum(1)/2
    x=torch.full((2,3,4,4),0.5)
    result=langevin(Quadratic(),x,steps=5,step_size=1,noise_std=0)
    assert torch.all(Quadratic()(result)<Quadratic()(x))
    torch.testing.assert_close(result,torch.full_like(x,0.35))


def test_schedule_reconstruction_and_oracle_ddim():
    schedule=DiffusionSchedule()
    clean=torch.full((2,3,32,32),0.2)
    times=torch.tensor([0,999])
    noise=torch.ones_like(clean)
    xt=schedule.add_noise(clean,times,noise)
    a=schedule.alpha_bars[times][:,None,None,None]
    torch.testing.assert_close((xt-(1-a).sqrt()*noise)/a.sqrt(),clean,atol=1e-5,rtol=1e-5)
    class Oracle(torch.nn.Module):
        def forward(self,x,t):
            a=schedule.alpha_bars[t][:,None,None,None]
            return (x-a.sqrt()*clean)/(1-a).sqrt()
    output=ddim_sample(Oracle(),schedule,torch.randn_like(clean),steps=10)
    torch.testing.assert_close(output,clean,atol=2e-5,rtol=1e-5)


def test_q6_q9_actual_autograd():
    x=torch.tensor([2.],requires_grad=True)
    (x**2+3*x).backward()
    assert x.grad.item()==7
    w=torch.tensor([1.,3.],requires_grad=True)
    x=torch.tensor([2.],requires_grad=True)
    (w[0]*x**2+w[1]*x).backward()
    assert w.grad.tolist()==[4.,2.]
    assert x.grad.item()==7
    with pytest.raises(RuntimeError):
        (torch.tensor([2.])**2).backward()
    x=torch.tensor([1.],requires_grad=True)
    y=x*3
    with pytest.raises(RuntimeError):
        (y.detach()*2).backward()
    z=y.clone()
    (z*2).backward()
    assert x.grad.item()==6
    x.grad=None
    (x*2).backward()
    (x*3).backward()
    assert x.grad.item()==5
    x.grad=None
    (x*3).backward()
    assert x.grad.item()==3
