"""Frozen, offline CPU inference for the two trained CIFAR-10 generators."""
import base64
import hashlib
import math
import secrets
import threading
from pathlib import Path

import numpy as np
import torch
from helper_lib.advanced_models import EnergyModel, UNet, DiffusionSchedule, langevin, ddim_sample
from app.gan_service import png_bytes

MODELS = Path(__file__).resolve().parents[1]/'models'


class AdvancedService:
    def __init__(self, kind, model_dir=MODELS):
        self.kind = kind
        path = model_dir/f'cifar10_{kind}.pt'
        checkpoint = torch.load(path, map_location='cpu', weights_only=True)
        if checkpoint['format_version'] != 1 or checkpoint['kind'] != kind:
            raise ValueError('Invalid advanced generator checkpoint')
        self.model = EnergyModel() if kind == 'ebm' else UNet()
        self.model.load_state_dict(checkpoint['state_dict'], strict=True)
        self.model.eval().requires_grad_(False)
        self.schedule = DiffusionSchedule()
        self.lock = threading.Lock()
        self.metadata = dict(checkpoint['metadata'])
        self.metadata.pop('data', None)
        self.metadata.update(device='cpu', checkpoint_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                             model=kind, image_shape=[3,32,32], class_conditioning=False,
                             sampler='clipped short-run Langevin' if kind=='ebm' else 'DDIM eta=0',
                             limitations='Small unconditional teaching model; output quality varies. No FID or convergence claim.')

    def generate(self, count, seed, steps):
        seed = secrets.randbelow(2**31) if seed is None else seed
        rng = torch.Generator(device='cpu').manual_seed(seed)
        # Requests serialize within each model, with their own RNG and frozen weights.
        with self.lock:
            if self.kind == 'ebm':
                initial = torch.rand(count,3,32,32,generator=rng)*2-1
                samples = langevin(self.model, initial, steps=steps, generator=rng)
            else:
                initial = torch.randn(count,3,32,32,generator=rng)
                samples = ddim_sample(self.model, self.schedule, initial, steps=steps)
        if not torch.isfinite(samples).all():
            raise RuntimeError('Generator returned nonfinite pixels')
        pixels = samples.add(1).mul(127.5).round().clamp(0,255).byte().permute(0,2,3,1).numpy()
        columns = min(4, math.ceil(math.sqrt(count)))
        rows = math.ceil(count/columns)
        grid = np.full((rows*32, columns*32, 3),255,dtype=np.uint8)
        images = []
        for index, pixel in enumerate(pixels):
            row, col = divmod(index,columns)
            grid[row*32:(row+1)*32,col*32:(col+1)*32] = pixel
            images.append(base64.b64encode(png_bytes(pixel)).decode('ascii'))
        return dict(model=self.kind,count=count,seed=seed,steps=steps,image_shape=[3,32,32],
                    mime_type='image/png',images_base64=images,grid_shape=list(grid.shape),
                    grid_png_base64=base64.b64encode(png_bytes(grid)).decode('ascii'))
