import json
import random
from collections import OrderedDict

import numpy as np

from data.base_dataset import BaseDataset, npy_to_tensor
from data.mat_io import load_mat_volume


_VOL_CACHE_SIZE = 4  # volumes kept in memory per DataLoader worker


class MatUnalignedDataset(BaseDataset):
    """
    Loads 2D coronal slices directly from .mat volumes — no preprocessing step.

    split.json format (volume paths, not slice paths):
        {
          "trainA": ["/data/ct/p001.mat", ...],
          "trainB": ["/data/mri/p001.mat", ...],
          "testA":  [...],
          "testB":  [...]
        }

    At init   : each volume is loaded once to record non-empty slice indices, then freed.
    At getitem: slices are loaded on demand with a per-worker LRU volume cache.

    Required flags: --split_file, --coronal_axis (default 1), --mat_key (auto-detected)
    Usage        : --dataset_mode mat_unaligned
    """

    @staticmethod
    def modify_commandline_options(parser, is_train):
        return parser

    def __init__(self, opt):
        BaseDataset.__init__(self, opt)

        with open(opt.split_file) as f:
            split = json.load(f)

        self._mat_key = getattr(opt, 'mat_key', None) or None
        self._coronal_axis = getattr(opt, 'coronal_axis', 1)
        min_content = getattr(opt, 'min_content', 0.02)

        key_A, key_B = opt.phase + 'A', opt.phase + 'B'
        if key_A not in split or key_B not in split:
            raise KeyError(f"split_file must have keys '{key_A}' and '{key_B}' for phase '{opt.phase}'")

        print(f"[MatUnaligned] Scanning {len(split[key_A])} domain-A volumes...")
        self.A_entries = _scan_volumes(split[key_A], self._coronal_axis, self._mat_key, min_content)
        print(f"  → {len(self.A_entries)} slices retained")

        print(f"[MatUnaligned] Scanning {len(split[key_B])} domain-B volumes...")
        self.B_entries = _scan_volumes(split[key_B], self._coronal_axis, self._mat_key, min_content)
        print(f"  → {len(self.B_entries)} slices retained")

        self.A_size = len(self.A_entries)
        self.B_size = len(self.B_entries)

        # Populated lazily per DataLoader worker process (not shared across workers)
        self._cache: OrderedDict = OrderedDict()

    def _load_slice(self, path, idx):
        if path not in self._cache:
            if len(self._cache) >= _VOL_CACHE_SIZE:
                self._cache.popitem(last=False)  # evict LRU
            vol, _ = load_mat_volume(path, self._mat_key)
            self._cache[path] = vol
        return np.take(self._cache[path], idx, axis=self._coronal_axis)

    def __getitem__(self, index):
        A_path, A_idx = self.A_entries[index % self.A_size]
        B_path, B_idx = self.B_entries[random.randint(0, self.B_size - 1)]

        A_sl = self._load_slice(A_path, A_idx)
        B_sl = self._load_slice(B_path, B_idx)

        is_finetuning = self.opt.isTrain and self.current_epoch > self.opt.n_epochs
        A = npy_to_tensor(A_sl, self.opt, is_finetuning)
        B = npy_to_tensor(B_sl, self.opt, is_finetuning)

        return {'A': A, 'B': B, 'A_paths': A_path, 'B_paths': B_path}

    def __len__(self):
        return max(self.A_size, self.B_size)


def _scan_volumes(vol_paths, coronal_axis, mat_key, min_content):
    """Load each volume once to record non-empty slice indices, then free it."""
    entries = []
    for vpath in vol_paths:
        try:
            vol, _ = load_mat_volume(vpath, mat_key)
            n = vol.shape[coronal_axis]
            for i in range(n):
                sl = np.take(vol, i, axis=coronal_axis)
                if np.count_nonzero(sl) / sl.size >= min_content:
                    entries.append((vpath, i))
            del vol
        except Exception as e:
            print(f"  WARNING: skipping {vpath} — {e}")
    return entries
