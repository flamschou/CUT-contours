# CUT – Contrastive Unpaired Translation

PyTorch implementation of unpaired image-to-image translation via patchwise contrastive learning (ECCV 2020). Paper: [arxiv 2007.15651](https://arxiv.org/abs/2007.15651).

## Architecture

Three model variants share the same codebase:

| Model | Class | Notes |
|---|---|---|
| CUT | `CUTModel` | Full model, NCE identity loss on |
| FastCUT | `CUTModel` | Half the compute, no NCE identity |
| CycleGAN | `CycleGANModel` | Baseline for comparison |
| SinCUT | `SinCUTModel` | Single-image variant |

**Core components:**
- `models/cut_model.py` — main training logic (forward, losses, optimizer steps)
- `models/networks.py` — Generator (ResNet 9-block), Discriminator (PatchGAN), MLP projector (netF)
- `models/patchnce.py` — PatchNCE contrastive loss
- `data/` — dataset loaders (`unaligned`, `single`, `singleimage`)
- `options/` — all CLI arguments via argparse (base → train/test)

## Key Commands

### Environment
```bash
conda env create -f environment.yml
conda activate contrastive-unpaired-translation
# or
pip install -r requirements.txt
```

### Training
```bash
# CUT (horse → zebra)
python train.py --dataroot ./datasets/horse2zebra --name horse2zebra_cut --CUT_mode CUT

# FastCUT
python train.py --dataroot ./datasets/horse2zebra --name horse2zebra_fastcut --CUT_mode FastCUT

# Single-image translation
python train.py --dataroot ./datasets/grumpifycat --name cat2grumpy --model sincut

# Multi-GPU (e.g. 2 GPUs)
python train.py --dataroot ./datasets/horse2zebra --name horse2zebra_cut --CUT_mode CUT --gpu_ids 0,1
```

### Testing / Inference
```bash
python test.py --dataroot ./datasets/horse2zebra --name horse2zebra_cut --CUT_mode CUT --phase test
# Results saved to ./results/<name>/test_latest/
```

### Download datasets
```bash
bash datasets/download_cut_dataset.sh horse2zebra
bash datasets/download_cut_dataset.sh grumpifycat
```

### Linting
```bash
flake8 .   # config in tox.ini (max-line-length=120)
```

## Important Options

| Flag | Default | Meaning |
|---|---|---|
| `--dataroot` | — | Path with `trainA/`, `trainB/` subdirs |
| `--name` | `experiment_name` | Run name; controls checkpoint dir |
| `--checkpoints_dir` | `./checkpoints` | Where models are saved |
| `--n_epochs` | 200 | Epochs at initial LR |
| `--n_epochs_decay` | 200 | Epochs for LR linear decay |
| `--gpu_ids` | `0` | Comma-separated IDs; `-1` for CPU |
| `--batch_size` | `1` | Typically 1 for 256×256 images |
| `--lambda_NCE` | `1.0` | Weight of PatchNCE loss |
| `--nce_layers` | `0,4,8,12,16` | Encoder layers used for NCE |

## Directory Layout

```
.
├── train.py / test.py      # Entry points
├── models/
│   ├── cut_model.py        # CUT/FastCUT
│   ├── cycle_gan_model.py  # CycleGAN baseline
│   ├── sincut_model.py     # Single-image CUT
│   ├── networks.py         # All network definitions
│   └── patchnce.py         # NCE loss
├── data/                   # Dataset classes
├── options/                # CLI argument definitions
├── util/                   # Visualizer, HTML report, image utils
├── experiments/            # Launcher scripts (tmux multi-run)
├── datasets/               # Download scripts
└── checkpoints/            # Saved model weights (gitignored)
```

## Coding Conventions

- Network architecture changes → `models/networks.py`
- New loss terms → add to `loss_names` list in the model and compute in `calculate_NCE_loss` or `forward`
- New CLI flags → `modify_commandline_options` of the relevant model class
- Checkpoints are named `<epoch>_net_G.pth`, `<epoch>_net_D.pth`, `latest_net_*.pth`
- `opt` object is the single source of truth for all hyperparameters; never hardcode values that should be tunable
