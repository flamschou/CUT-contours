#!/usr/bin/env python3
"""
Build split.json from two lists of .mat volume paths.
No preprocessing — volumes are loaded directly during training.

Usage:
    python build_split.py \
        --ct_files  ct_paths.txt  \
        --mri_files mri_paths.txt \
        --split_file split.json
"""

import argparse
import json
import random


def read_file_list(path):
    with open(path) as f:
        return [l.strip() for l in f if l.strip() and not l.startswith('#')]


def vol_split(paths, train_ratio, seed):
    shuffled = paths[:]
    random.seed(seed)
    random.shuffle(shuffled)
    n = int(len(shuffled) * train_ratio)
    return shuffled[:n], shuffled[n:]


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--ct_files',    required=True, help='Text file of CT .mat paths')
    p.add_argument('--mri_files',   required=True, help='Text file of MRI .mat paths')
    p.add_argument('--split_file',  default='./split.json')
    p.add_argument('--train_ratio', type=float, default=0.85)
    p.add_argument('--seed',        type=int,   default=42)
    args = p.parse_args()

    ct_paths  = read_file_list(args.ct_files)
    mri_paths = read_file_list(args.mri_files)

    ct_train,  ct_test  = vol_split(ct_paths,  args.train_ratio, args.seed)
    mri_train, mri_test = vol_split(mri_paths, args.train_ratio, args.seed)

    split = {'trainA': ct_train, 'trainB': mri_train,
             'testA':  ct_test,  'testB':  mri_test}

    with open(args.split_file, 'w') as f:
        json.dump(split, f, indent=2)

    print(f"trainA (CT) : {len(ct_train)} volumes")
    print(f"trainB (MRI): {len(mri_train)} volumes")
    print(f"testA  (CT) : {len(ct_test)} volumes")
    print(f"testB  (MRI): {len(mri_test)} volumes")
    print(f"→ {args.split_file}")


if __name__ == '__main__':
    main()
