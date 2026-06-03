"""Shared .mat volume loading — handles legacy scipy and HDF5 (MATLAB v7.3) formats."""
import numpy as np


def load_mat_volume(path, mat_key=None):
    """Return (float32 ndarray, key_used) for the first 3D array found in a .mat file."""
    path = str(path)
    try:
        import scipy.io
        data = scipy.io.loadmat(path)
        key = _find_key(data, mat_key, path)
        return data[key].astype(np.float32), key
    except NotImplementedError:
        return _load_hdf5(path, mat_key)


def _find_key(data, mat_key, path):
    if mat_key:
        if mat_key not in data:
            raise KeyError(f"Key '{mat_key}' not found in {path}")
        return mat_key
    candidates = [k for k, v in data.items()
                  if not k.startswith('_') and isinstance(v, np.ndarray) and v.ndim == 3]
    if not candidates:
        raise ValueError(f"No 3D array in {path}. Keys: {[k for k in data if not k.startswith('_')]}")
    if len(candidates) > 1:
        print(f"  [warn] multiple 3D arrays in {path}: {candidates}. Using '{candidates[0]}'.")
    return candidates[0]


def _load_hdf5(path, mat_key):
    try:
        import h5py
    except ImportError:
        raise ImportError("pip install h5py  (required for MATLAB v7.3 .mat files)")
    with h5py.File(path, 'r') as f:
        if mat_key:
            return np.array(f[mat_key]).T.astype(np.float32), mat_key
        candidates = [k for k in f.keys()
                      if hasattr(f[k], 'shape') and len(f[k].shape) == 3]
        if not candidates:
            raise ValueError(f"No 3D dataset in {path}. Keys: {list(f.keys())}")
        key = candidates[0]
        return np.array(f[key]).T.astype(np.float32), key
