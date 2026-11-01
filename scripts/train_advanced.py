"""Actual full CIFAR-10 training; final predeclared epochs, no test-set selection."""
import argparse
import copy
import hashlib
import json
import platform
import sys
import time
from pathlib import Path

import torch
from torch.nn import functional as F
from torchvision.datasets import CIFAR10
from torchvision.utils import save_image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from helper_lib.advanced_models import EnergyModel, UNet, DiffusionSchedule, langevin, ddim_sample


def save_json(path, data):
    path.write_text(json.dumps(data, indent=2), encoding='utf-8')


def train(args):
    torch.set_num_threads(4)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.benchmark = False
    device = torch.device(args.device)
    dataset = CIFAR10(args.data, train=True, download=False)
    # torchvision validates the official batch-file MD5 checksums on construction.
    data = torch.from_numpy(dataset.data).permute(0, 3, 1, 2).contiguous()
    assert len(data) == 50000
    config = vars(args).copy()
    config.update(dataset='CIFAR-10', training_samples=len(data), image_shape=[3,32,32],
                  checkpoint_selection='final predeclared epoch; test set not used for selection',
                  torch_version=str(torch.__version__), python_version=platform.python_version(),
                  device_name=torch.cuda.get_device_name(device) if device.type=='cuda' else 'CPU')
    out = ROOT/'results'/args.kind
    out.mkdir(parents=True, exist_ok=True)
    save_json(out/'config.json', config)
    model = (EnergyModel() if args.kind=='ebm' else UNet()).to(device)
    config['parameter_count'] = sum(p.numel() for p in model.parameters())
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9,0.999))
    schedule = DiffusionSchedule().to(device)
    ema = copy.deepcopy(model).eval() if args.kind=='diffusion' else None
    if ema:
        ema.requires_grad_(False)
    replay = torch.rand(8192, 3, 32, 32, device=device)*2-1 if args.kind=='ebm' else None
    shuffle_rng = torch.Generator().manual_seed(args.seed)
    global_step = 0
    history = []
    started = time.perf_counter()
    for epoch in range(1, args.epochs+1):
        tick = time.perf_counter()
        permutation = torch.randperm(len(data), generator=shuffle_rng)
        totals = {'loss':0., 'positive_energy':0., 'negative_energy':0., 'contrastive':0., 'regularizer':0.}
        seen = 0
        for indices in permutation.split(args.batch_size):
            clean = data[indices].to(device).float()/127.5-1
            n = len(clean)
            optimizer.zero_grad(set_to_none=True)
            if args.kind=='ebm':
                positions = torch.randperm(len(replay), device=device)[:n]
                initial = replay[positions].clone()
                fresh = torch.rand(n, device=device) < 0.05
                initial[fresh] = torch.rand(int(fresh.sum()),3,32,32,device=device)*2-1
                model.eval()
                model.requires_grad_(False)
                negative = langevin(model, initial, args.langevin_steps)
                model.requires_grad_(True)
                model.train()
                replay[positions] = negative
                real = (clean + 0.005*torch.randn_like(clean)).clamp(-1,1)
                positive_energy, negative_energy = model(real), model(negative)
                contrastive = positive_energy.mean()-negative_energy.mean()
                regularizer = 0.1*(positive_energy.square().mean()+negative_energy.square().mean())
                loss = contrastive+regularizer
                for key, value in [('positive_energy',positive_energy.mean()),('negative_energy',negative_energy.mean()),
                                   ('contrastive',contrastive),('regularizer',regularizer)]:
                    totals[key] += value.item()*n
            else:
                model.train()
                times = torch.randint(0, schedule.steps, (n,), device=device)
                noise = torch.randn_like(clean)
                noisy = schedule.add_noise(clean, times, noise)
                loss = F.mse_loss(model(noisy, times), noise)
            if not torch.isfinite(loss):
                raise FloatingPointError(f'nonfinite loss at epoch {epoch}, step {global_step}')
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
            optimizer.step()
            global_step += 1
            if ema:
                decay = min(0.999, (1+global_step)/(10+global_step))
                with torch.no_grad():
                    for target, source in zip(ema.parameters(), model.parameters()):
                        target.lerp_(source, 1-decay)
            totals['loss'] += loss.item()*n
            seen += n
        record = dict(epoch=epoch, samples=seen, optimizer_steps=global_step,
                      seconds=time.perf_counter()-tick, **{k:v/seen for k,v in totals.items()})
        if args.kind=='diffusion':
            record = {k:v for k,v in record.items() if k not in ('positive_energy','negative_energy','contrastive','regularizer')}
        history.append(record)
        save_json(out/'history.json', history)
        config.update(completed_epochs=epoch, optimizer_steps=global_step, elapsed_seconds=time.perf_counter()-started)
        chosen = ema if ema else model
        checkpoint = {'format_version':1,'kind':args.kind,'state_dict':{k:v.cpu().clone() for k,v in chosen.state_dict().items()},'metadata':config}
        torch.save(checkpoint, ROOT/'models'/f'cifar10_{args.kind}.pt')
        # Training-state archive stays outside the minimal submission.
        torch.save({'model':model.state_dict(),'ema':ema.state_dict() if ema else None,
                    'optimizer':optimizer.state_dict(),'epoch':epoch,'config':config},out/'training_state.pt')
        print(json.dumps(record), flush=True)
    model = ema if ema else model
    model.eval().requires_grad_(False)
    rng = torch.Generator(device=device).manual_seed(2026)
    if args.kind=='ebm':
        initial = torch.rand(64,3,32,32,device=device,generator=rng)*2-1
        samples = langevin(model, initial, 60, generator=rng)
    else:
        initial = torch.randn(64,3,32,32,device=device,generator=rng)
        samples = ddim_sample(model,schedule,initial,50)
    save_image(samples.cpu(),out/'samples_seed2026.png',nrow=8,normalize=True,value_range=(-1,1))
    save_json(out/'summary.json',config)
    print('COMPLETE '+args.kind, flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('kind',choices=['ebm','diffusion'])
    p.add_argument('--data',required=True)
    p.add_argument('--epochs',type=int,required=True)
    p.add_argument('--batch-size',type=int,default=128)
    p.add_argument('--langevin-steps',type=int,default=40)
    p.add_argument('--lr',type=float,default=0.0002)
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--device',default='cuda')
    args=p.parse_args()
    if args.epochs<1 or args.batch_size<1 or args.langevin_steps<1 or args.lr<=0:
        p.error('positive epochs, batch size, steps and learning rate required')
    train(args)
