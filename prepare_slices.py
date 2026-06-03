#!/usr/bin/env python3
"""
Extract 2D coronal slices from 3D .mat volumes and build split.json for CUT.

Input: two text files listing .mat paths (one per line), one for CT, one for MRI.
Output: PNG slices in --output_dir and a split.json ready for train.py / test.py.

Usage:
    python prepare_slices.py \
        --ct_files  ct_paths.txt  \
        --mri_files mri_paths.txt \
        --output_dir ./slices     \
        --split_file ./split.json

ct_paths.txt format (one absolute path per line, # for comments):
    /data/ct/patient001.mat
    /data/ct/patient002.mat
    ...
"""

import argparse
import json
import os
import random
from pathlib import Path

import numpy as np

from data.mat_io import load_mat_volume


# ---------------------------------------------------------------------------
# Slice extraction and normalisation
# ---------------------------------------------------------------------------

def extract_coronal_slices(vol, coronal_axis, min_content):
    """Yield (index, 2D array) for coronal slices with enough non-zero content."""
    n = vol.shape[coronal_axis]
    for i in range(n):
        sl = np.take(vol, i, axis=coronal_axis)
        if np.count_nonzero(sl) / sl.size >= min_content:
            yield i, sl


def to_float32(sl, raw_intensities=False):
    """
    Ensure slice is float32 in [0, 1].
    If raw_intensities=True, apply percentile clipping for non-normalized data.
    """
    if raw_intensities:
        lo, hi = np.percentile(sl, 2), np.percentile(sl, 98)
        if hi <= lo:
            return np.zeros(sl.shape, dtype=np.float32)
        return np.clip((sl - lo) / (hi - lo), 0.0, 1.0).astype(np.float32)
    return np.clip(sl, 0.0, 1.0).astype(np.float32)


def process_volume(mat_path, out_dir, coronal_axis, mat_key, min_content, raw_intensities):
    vol, _ = load_mat_volume(mat_path, mat_key)
    vol_id = Path(mat_path).stem
    saved = []
    for idx, sl in extract_coronal_slices(vol, coronal_axis, min_content):
        arr = to_float32(sl, raw_intensities=raw_intensities)
        fpath = os.path.join(out_dir, f"{vol_id}_cor{idx:04d}.npy")
        np.save(fpath, arr)
        saved.append(fpath)
    return saved


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def read_file_list(path):
    with open(path) as f:
        return [l.strip() for l in f if l.strip() and not l.startswith('#')]


def vol_train_test_split(paths, train_ratio, seed):
    shuffled = paths[:]
    random.seed(seed)
    random.shuffle(shuffled)
    n = int(len(shuffled) * train_ratio)
    return shuffled[:n], shuffled[n:]


def process_all(vol_paths, out_dir, coronal_axis, mat_key, min_content, already_normalized):
    all_slices = []
    for i, vpath in enumerate(vol_paths):
        print(f"  [{i+1}/{len(vol_paths)}] {Path(vpath).name}")
        try:
            slices = process_volume(vpath, out_dir, coronal_axis, mat_key, min_content, already_normalized)
            print(f"    -> {len(slices)} slices saved")
            all_slices.extend(slices)
        except Exception as e:
            print(f"    WARNING: skipped ({e})")
    return all_slices


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--ct_files',     required=True, help='Text file listing CT .mat paths')
    p.add_argument('--mri_files',    required=True, help='Text file listing MRI .mat paths')
    p.add_argument('--output_dir',   default='./slices',     help='Root dir for saved PNGs')
    p.add_argument('--split_file',   default='./split.json', help='Output split.json path')
    p.add_argument('--mat_key',      default=None,  help='Variable name inside .mat (auto-detected if omitted)')
    p.add_argument('--coronal_axis', type=int, default=1, choices=[0, 1, 2],
                   help='Volume axis corresponding to coronal direction (default: 1)')
    p.add_argument('--train_ratio',  type=float, default=0.85, help='Fraction of volumes for training')
    p.add_argument('--min_content',  type=float, default=0.02,
                   help='Min fraction of non-zero pixels to keep a slice (default: 0.02)')
    p.add_argument('--raw_intensities', action='store_true',
                   help='Use percentile clipping instead of direct *255 scaling (for raw CT/MRI intensities not in [0,1])')
    p.add_argument('--seed',         type=int,   default=42)
    args = p.parse_args()

    ct_paths  = read_file_list(args.ct_files)
    mri_paths = read_file_list(args.mri_files)

    ct_train_vols,  ct_test_vols  = vol_train_test_split(ct_paths,  args.train_ratio, args.seed)
    mri_train_vols, mri_test_vols = vol_train_test_split(mri_paths, args.train_ratio, args.seed)

    ct_out  = os.path.join(args.output_dir, 'ct')
    mri_out = os.path.join(args.output_dir, 'mri')
    os.makedirs(ct_out,  exist_ok=True)
    os.makedirs(mri_out, exist_ok=True)

    kw = dict(coronal_axis=args.coronal_axis, mat_key=args.mat_key,
              min_content=args.min_content, raw_intensities=args.raw_intensities)

    print(f"\nCT train ({len(ct_train_vols)} volumes):")
    ct_train_slices  = process_all(ct_train_vols,  ct_out,  **kw)
    print(f"\nCT test ({len(ct_test_vols)} volumes):")
    ct_test_slices   = process_all(ct_test_vols,   ct_out,  **kw)
    print(f"\nMRI train ({len(mri_train_vols)} volumes):")
    mri_train_slices = process_all(mri_train_vols, mri_out, **kw)
    print(f"\nMRI test ({len(mri_test_vols)} volumes):")
    mri_test_slices  = process_all(mri_test_vols,  mri_out, **kw)

    split = {
        'trainA': ct_train_slices,
        'trainB': mri_train_slices,
        'testA':  ct_test_slices,
        'testB':  mri_test_slices,
    }
    with open(args.split_file, 'w') as f:
        json.dump(split, f, indent=2)

    print(f"\n{'─'*50}")
    print(f"  trainA (CT) : {len(ct_train_slices):>6} slices  ({len(ct_train_vols)} vols)")
    print(f"  trainB (MRI): {len(mri_train_slices):>6} slices  ({len(mri_train_vols)} vols)")
    print(f"  testA  (CT) : {len(ct_test_slices):>6} slices  ({len(ct_test_vols)} vols)")
    print(f"  testB  (MRI): {len(mri_test_slices):>6} slices  ({len(mri_test_vols)} vols)")
    print(f"  split.json  -> {args.split_file}")


if __name__ == '__main__':
    main()
