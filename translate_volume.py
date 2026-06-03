#!/usr/bin/env python3
"""
Translate a 3D CT contour volume to MRI-style using a trained CUT generator.

Loads a .mat volume, processes each coronal slice through the generator,
reassembles the full 3D volume, and saves it as .mat float32 in [0, 1].

Usage:
    python translate_volume.py \
        --input   /data/ct/patient001.mat \
        --output  /results/patient001_mri.mat \
        --name    mon_experience            \
        --epoch   latest                   \
        --mat_key data                     \
        --coronal_axis 1

Or batch mode (text file of input paths):
    python translate_volume.py \
        --input_list ct_test_paths.txt \
        --output_dir /results/         \
        --name mon_experience
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch

from data.mat_io import load_mat_volume


def save_mat_volume(path, vol, mat_key):
    import scipy.io
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    scipy.io.savemat(str(path), {mat_key: vol.astype(np.float32)})


# ---------------------------------------------------------------------------
# Generator loading
# ---------------------------------------------------------------------------

def load_generator(name, checkpoints_dir, epoch, gpu_ids):
    """Load the CUT model (generator only used at test time)."""
    from options.test_options import TestOptions
    from models import create_model

    cmd = (f"--name {name} --checkpoints_dir {checkpoints_dir} "
           f"--epoch {epoch} --model cut --CUT_mode CUT "
           f"--dataroot placeholder --no_dropout --gpu_ids {gpu_ids}")
    opt = TestOptions(cmd_line=cmd).parse()
    model = create_model(opt)
    model.setup(opt)
    model.eval()
    return model


# ---------------------------------------------------------------------------
# Slice-level translation
# ---------------------------------------------------------------------------

def translate_slice(generator, sl_np, device):
    """
    Translate one 2D float32 [0,1] slice.
    Returns float32 numpy array in [0,1].
    """
    # [0,1] → [-1,1], add batch and channel dims → (1, 1, H, W)
    t = torch.from_numpy(sl_np.astype(np.float32))
    t = (t * 2 - 1).unsqueeze(0).unsqueeze(0).to(device)

    with torch.no_grad():
        fake = generator.netG(t)          # (1, 1, H, W) in [-1, 1]

    out = (fake.squeeze().cpu().numpy() + 1) / 2  # → [0, 1]
    return np.clip(out, 0.0, 1.0).astype(np.float32)


# ---------------------------------------------------------------------------
# Volume translation
# ---------------------------------------------------------------------------

def translate_volume(input_path, output_path, generator, coronal_axis, mat_key, device):
    vol, detected_key = load_mat_volume(input_path, mat_key)
    out_key = mat_key or detected_key

    vol = vol.astype(np.float32)
    output_vol = np.zeros_like(vol)

    n_slices = vol.shape[coronal_axis]
    print(f"  shape {vol.shape}  |  {n_slices} coronal slices along axis {coronal_axis}")

    for i in range(n_slices):
        sl = np.take(vol, i, axis=coronal_axis)
        translated = translate_slice(generator, sl, device)

        # Write back into the output volume
        idx = [slice(None)] * 3
        idx[coronal_axis] = i
        output_vol[tuple(idx)] = translated

        if (i + 1) % 20 == 0 or i == n_slices - 1:
            print(f"    {i+1}/{n_slices} slices", end='\r')

    print()
    save_mat_volume(output_path, output_vol, out_key)
    print(f"  saved → {output_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument('--input',      help='Single input .mat path')
    src.add_argument('--input_list', help='Text file of input .mat paths (one per line)')

    dst = p.add_mutually_exclusive_group()
    dst.add_argument('--output',     help='Output .mat path (single mode)')
    dst.add_argument('--output_dir', help='Output directory (batch mode)')

    p.add_argument('--name',            required=True, help='Experiment name (same as training)')
    p.add_argument('--checkpoints_dir', default='./checkpoints')
    p.add_argument('--epoch',           default='latest')
    p.add_argument('--mat_key',         default=None,  help='MATLAB variable name (auto-detected if omitted)')
    p.add_argument('--coronal_axis',    type=int, default=1, choices=[0, 1, 2])
    p.add_argument('--gpu_ids',         type=str, default='0', help='GPU ids, -1 for CPU')
    return p.parse_args()


def main():
    args = parse_args()

    # Device
    if args.gpu_ids == '-1' or not torch.cuda.is_available():
        device = torch.device('cpu')
    else:
        device = torch.device(f'cuda:{args.gpu_ids.split(",")[0]}')
    print(f"Device: {device}")

    # Load generator
    print(f"Loading generator '{args.name}' epoch={args.epoch}...")
    generator = load_generator(args.name, args.checkpoints_dir, args.epoch, args.gpu_ids)

    # Build list of (input, output) pairs
    if args.input:
        inputs = [args.input]
        if args.output:
            outputs = [args.output]
        else:
            out_dir = args.output_dir or '.'
            outputs = [os.path.join(out_dir, Path(args.input).name)]
    else:
        with open(args.input_list) as f:
            inputs = [l.strip() for l in f if l.strip() and not l.startswith('#')]
        out_dir = args.output_dir or '.'
        outputs = [os.path.join(out_dir, Path(p).name) for p in inputs]

    # Translate
    for i, (inp, out) in enumerate(zip(inputs, outputs)):
        print(f"\n[{i+1}/{len(inputs)}] {Path(inp).name}")
        try:
            translate_volume(inp, out, generator, args.coronal_axis, args.mat_key, device)
        except Exception as e:
            print(f"  ERROR: {e}")

    print("\nDone.")


if __name__ == '__main__':
    main()
