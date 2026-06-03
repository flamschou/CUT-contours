# CUT – Contrastive Unpaired Translation

PyTorch implementation of unpaired image-to-image translation via patchwise contrastive learning (ECCV 2020). Paper: [arxiv 2007.15651](https://arxiv.org/abs/2007.15651).

## Architecture

Two active model variants:

| Model | Flag | Notes |
|---|---|---|
| CUT | `--CUT_mode CUT` | Full model, NCE identity loss on |
| FastCUT | `--CUT_mode FastCUT` | Lighter, no identity loss, 2x less memory |

**Core components:**
- `models/cut_model.py` — main training logic (forward, losses, optimizer steps)
- `models/networks.py` — Generator (ResNet 9-block), Discriminator (PatchGAN), MLP projector (netF)
- `models/patchnce.py` — PatchNCE contrastive loss
- `data/unaligned_dataset.py` — dataset loader (supports split.json and dataroot)
- `options/` — all CLI arguments via argparse (base → train/test)

## Dataset Setup

Instead of copying heavy image datasets into the repo, use a `split.json` file listing absolute paths:

```json
{
  "trainA": ["/data/domainA/img001.jpg", "/data/domainA/img002.jpg"],
  "trainB": ["/data/domainB/img001.jpg", "/data/domainB/img002.jpg"],
  "testA":  ["/data/domainA/test001.jpg"],
  "testB":  ["/data/domainB/test001.jpg"]
}
```

Pass it with `--split_file /path/to/split.json` — no `--dataroot` required.

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
# With split.json (recommended for large datasets)
python train.py --split_file ./split.json --name my_experiment --CUT_mode CUT

# FastCUT (lighter, no identity loss)
python train.py --split_file ./split.json --name my_experiment_fast --CUT_mode FastCUT

# Multi-GPU (e.g. 2 GPUs)
python train.py --split_file ./split.json --name my_experiment --CUT_mode CUT --gpu_ids 0,1

# Legacy: with dataroot directories
python train.py --dataroot ./datasets/horse2zebra --name horse2zebra_cut --CUT_mode CUT
```

### Inference
```bash
# With split.json
python test.py --split_file ./split.json --name my_experiment --CUT_mode CUT --phase test

# Results saved to ./results/<name>/test_latest/
```

### Linting
```bash
flake8 .   # config in tox.ini (max-line-length=120)
```

## Important Options

| Flag | Default | Meaning |
|---|---|---|
| `--split_file` | `None` | JSON file with trainA/trainB/testA/testB path lists |
| `--dataroot` | — | Alternative to split_file: dir with trainA/, trainB/ subdirs |
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
├── split.json              # Your dataset split (not committed)
├── models/
│   ├── cut_model.py        # CUT/FastCUT training logic
│   ├── networks.py         # Generator, Discriminator, netF definitions
│   ├── patchnce.py         # PatchNCE contrastive loss
│   ├── stylegan_networks.py # StyleGAN2 variants (available but not default)
│   └── base_model.py       # Abstract base class
├── data/
│   ├── unaligned_dataset.py # Main dataset class (split.json or dataroot)
│   └── image_folder.py     # Image path utilities
├── options/                # CLI argument definitions
├── util/                   # Visualizer, HTML report, image utils
└── checkpoints/            # Saved model weights (gitignored)
```

## Coding Conventions

- Network architecture changes → `models/networks.py`
- New loss terms → add to `loss_names` in the model, compute in `calculate_NCE_loss` or `forward`
- New CLI flags → `modify_commandline_options` of the relevant model class
- Checkpoints: `<epoch>_net_G.pth`, `<epoch>_net_D.pth`, `latest_net_*.pth`
- `opt` object is the single source of truth for all hyperparameters
