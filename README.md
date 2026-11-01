# Assignment 4: Advanced Image Generation

Yuhui Wu · APAN PS5560 Applied Generative AI

Repository: https://github.com/ShikiRyo1/apan5560-assignment4

This extends the existing course FastAPI application with two trained,
unconditional CIFAR-10 generators: a convolutional energy model with short-run
Langevin sampling, and a time-conditioned U-Net trained to predict diffusion
noise. Both inference checkpoints are included. Image endpoints execute these
networks; they do not return stored example images.

## Start on the instructor's computer

Install and start Docker Desktop with Linux containers. Extract the ZIP, open
a terminal in the folder containing this README and `compose.yaml`, then run:

```sh
docker compose up --build -d
docker compose ps
```

Open **http://127.0.0.1:8010/docs** for Swagger or
**http://127.0.0.1:8010/advanced/demo** to generate and download images.
The initial build requires internet for locked dependencies, including the
existing spaCy word-vector model. Inference then runs offline on CPU, without
a GPU, API key, host virtual environment, or CIFAR-10 dataset. Allow several
minutes for the first build and model loading. The container health check
queries both new models.

If port 8010 is occupied, change only the left host port in `compose.yaml`
(for example `127.0.0.1:8011:8000`) and use that port in the URLs.
Inspect failures with `docker compose logs api`; stop with `docker compose down`.
No bind mounts or absolute host paths are required.

Alternatively:

```sh
docker build -t apan5560-assignment4 .
docker run --rm -e OMP_NUM_THREADS=2 -e MKL_NUM_THREADS=2 -p 127.0.0.1:8010:8000 apan5560-assignment4
```

## Query both trained models

In Swagger, expand the POST endpoint, select **Try it out**, enter JSON,
and select **Execute**. For PowerShell:

```powershell
$body = '{"count":4,"seed":42,"steps":60}'
$r = Invoke-RestMethod 'http://127.0.0.1:8010/ebm/generate' -Method Post -ContentType 'application/json' -Body $body
[IO.File]::WriteAllBytes((Join-Path (Get-Location) 'energy.png'), [Convert]::FromBase64String($r.grid_png_base64))
$body = '{"count":4,"seed":42,"steps":50}'
$r = Invoke-RestMethod 'http://127.0.0.1:8010/diffusion/generate' -Method Post -ContentType 'application/json' -Body $body
[IO.File]::WriteAllBytes((Join-Path (Get-Location) 'diffusion.png'), [Convert]::FromBase64String($r.grid_png_base64))
```

Each response includes one PNG per sample in `images_base64`, a contact sheet
in `grid_png_base64`, `image_shape: [3,32,32]`, and the actual seed/step count.
Use `/ebm/model` and `/diffusion/model` to inspect training metadata and exact
checkpoint SHA-256. Count accepts integers 1–16, seed accepts null or integers
0–2147483647, EBM steps accepts 1–120, and diffusion steps accepts 2–100.
Booleans, strings in numeric fields, unknown fields and nonfinite numbers are
rejected. `seed: null` selects and returns a fresh seed. Fixed seeds reproduce
results in the same tested environment; cross-platform bit identity is not
promised. Each request uses its own random-number generator and frozen weights.

Existing course endpoints remain: `/health`, `/embedding?word=apple`,
`/generate` (bigram text), `/model`, `/classify` (CIFAR-10 classifier),
`/gan/model`, `/gan/generate`, and `/gan/demo` (MNIST GAN).

## Training and architecture

All 50,000 official CIFAR-10 training images are used each epoch, normalized
to [-1,1]. Class labels are intentionally unused. Seed is 42; optimizer is
Adam at 0.0002. Final epochs were selected in advance; test images were not
used for checkpoint selection. Actual run summaries, epoch losses and sample
grids are retained in the GitHub repository's `results/` and
[training report](https://github.com/ShikiRyo1/apan5560-assignment4/blob/main/TRAINING_REPORT.md).
The Canvas ZIP contains only runtime/training source, required weights and
deployment instructions; test artifacts and training plots stay in the repository.

| Model | Architecture / objective | Actual training |
|---|---|---|
| Energy | Stride-2 convolutions 3→32→64→128→128, scalar head; mean real energy minus mean negative energy plus 0.1 squared-energy regularizer | 10 epochs, batch 128, 3,910 optimizer steps; 8,192 replay samples, 5% random restart, 40 Langevin steps per update |
| Diffusion | Residual U-Net, GroupNorm, time-MLP, three down/up stages 32→16→8→4; epsilon MSE | 30 epochs, batch 64, 23,460 optimizer steps; 1,000 linear beta steps 0.0001→0.02; EMA checkpoint |

Diffusion inference uses deterministic DDIM (eta=0), with clipped reconstructed
x0 and a recomputed consistent epsilon. The time embedding uses a sin block
followed by a cos block, frequencies `10000**(-i/(half-1))`, matching the
new Module 8 Practical 3 convention. The written answers also show the
interleaved Transformer convention explicitly to avoid conflating them.

The EBM sampler updates pixels with clipped energy gradients and small noise;
it is an approximate short-run generator, not an exact equilibrium MCMC
sampler. A low energy-training loss is not a likelihood or quality score.
These are small teaching models; EBM images remain mostly coarse textures,
and diffusion outputs show limited low-resolution structure. No FID, convergence,
photorealism, or instructor-side execution claim is made.

To retrain, use Python 3.12 with PyTorch 2.11.0 and torchvision 0.26.0. The
locked local dependency environment defaults to CPU; a separately configured
CUDA build was used for the recorded GPU training. From this project folder:

```sh
uv sync --group train
uv run python -c "from torchvision.datasets import CIFAR10; CIFAR10('data', train=True, download=True)"
uv run python scripts/train_advanced.py ebm --data data --epochs 10 --batch-size 128 --device cpu
uv run python scripts/train_advanced.py diffusion --data data --epochs 30 --batch-size 64 --device cpu
```

Training on CPU is substantially slower; use `--device cuda` only in a working
CUDA environment. These commands overwrite the two inference checkpoints in
`models/`; copy them first if retaining the submitted run. Training archives
include optimizer/model states but do not provide exact RNG/replay-buffer
resume. Dataset and environment folders are excluded from the submission.

## Organization and checks

- `helper_lib/advanced_models.py`: architectures, forward noising, Langevin and DDIM.
- `scripts/train_advanced.py`: actual training and checkpoint creation.
- `app/advanced_service.py`: strict checkpoint loading, frozen CPU inference and PNGs.
- `app/main.py`: typed FastAPI requests/responses and existing routes.
- `app/static/advanced_demo.html`: interactive client for the same API.
- `models/`: four real course checkpoints, including both new generators.
- Repository `tests/test_advanced.py`: tensor, sampler, schedule and autograd checks.
- `pyproject.toml`, `uv.lock`, `Dockerfile`, `compose.yaml`: reproducible runtime.

After cloning the full repository, run core tests with `uv run pytest -q`.
Independent HTTP and fresh-ZIP Docker checks are summarized in the repository's
[verification record](https://github.com/ShikiRyo1/apan5560-assignment4/blob/main/VERIFICATION.md). The separate answers
PDF covers all nine theoretical/calculation questions with working.

## References and AI assistance

Course Module 8 energy/diffusion practicals and Module 9 positional encoding;
[DDPM](https://arxiv.org/abs/2006.11239),
[DDIM](https://arxiv.org/abs/2010.02502),
[Attention Is All You Need](https://arxiv.org/abs/1706.03762), and
[PyTorch documentation](https://docs.pytorch.org/docs/2.11/).
See `AI_USAGE.md` for the required acknowledgment of Codex assistance.
Original restricted course materials and datasets are not republished here.
