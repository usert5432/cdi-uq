"""
Dataset implementation for CBED images repacked into HDF5 chunk files.

This dataset keeps the same label CSV format as the regular ``cbed`` dataset,
but reads image arrays from HDF5 files instead of loading one ``.npz`` file per
sample.  The expected layout is:

    PATH_TO_LABELS/
        +-- f11a/
        |     +-- labels_train.csv
        |     +-- labels_val.csv
        |     +-- labels.csv
        |     +-- ...
        +-- f11b/
        |     +-- labels_train.csv
        |     +-- ...
        +-- ...

Each label CSV must contain a ``file`` column plus the configured target
columns.  For repacked datasets with a single ``labels.csv`` file per subdir,
pass ``labels_file = 'labels.csv'`` in the dataset config.

Each HDF5 file must contain:

    images
        CBED image array, typically shaped ``(N, H, W)``.
    filenames
        Original per-sample filenames for rows in ``images``, for example
        ``sample_00000.npz``.

The CSV ``file`` entries are matched to HDF5 rows through the HDF5 chunk name
and the original filename stored in ``filenames``.  For example, row
``f11a_cbed0000000.ity/sample_00000.npz`` in subdir ``f11a`` maps to row
``sample_00000.npz`` in ``f11a/f11a_cbed0000000.ity.h5``.
"""

# pylint: disable=no-member
import glob
import os

import numpy as np
import torch

from torch.utils.data import Dataset
from torchvision.transforms.v2.functional import to_image, to_dtype

from .cbed_dataset import collect_datasets

KEY_FILENAMES = 'filenames'
KEY_IMAGES    = 'images'

def decode_filename(value):
    if isinstance(value, bytes):
        return value.decode('utf-8')

    return str(value)

def fname_to_csv_key(h5_path, fname, root, subdir):
    h5_stem = os.path.splitext(os.path.basename(h5_path))[0]
    fname   = os.path.basename(fname)

    if subdir is None:
        chunk_dir = os.path.relpath(os.path.dirname(h5_path), root)
        if chunk_dir == '.':
            return os.path.join(h5_stem, fname)

        return os.path.join(chunk_dir, h5_stem, fname)

    return os.path.join(subdir, h5_stem, fname)

def list_h5_files(root):
    paths = glob.glob(os.path.join(root, '*.h5'))
    paths.extend(glob.glob(os.path.join(root, '*.hdf5')))
    paths.sort()

    return paths

def collect_h5_index(root, subdir):
    # pylint: disable=import-outside-toplevel
    import h5py

    if subdir is None:
        path = root
    else:
        path = os.path.join(root, subdir)

    result = {}

    for h5_path in list_h5_files(path):
        with h5py.File(h5_path, 'r') as f:
            filenames = f[KEY_FILENAMES][:]

        for (idx, fname) in enumerate(filenames):
            csv_key = fname_to_csv_key(
                h5_path, decode_filename(fname), root, subdir
            )

            if csv_key in result:
                raise RuntimeError(f"Duplicate H5 filename entry: '{csv_key}'")

            result[csv_key] = (h5_path, idx)

    return result

def collect_h5_indices(root, subdirs):
    if subdirs is None:
        return collect_h5_index(root, None)

    result = {}

    for subdir in subdirs:
        for (fname, spec) in collect_h5_index(root, subdir).items():
            if fname in result:
                raise RuntimeError(f"Duplicate H5 filename entry: '{fname}'")

            result[fname] = spec

    return result

def load_h5_image(h5_file, idx, dtype):
    torch_dtype = getattr(torch, dtype)
    np_dtype    = getattr(np,    dtype)

    image = h5_file[KEY_IMAGES][idx]
    image = image.astype(np_dtype)

    if image.ndim == 2:
        image = torch.tensor(image[None], dtype = torch_dtype)
        return image

    return to_dtype(
        to_image(image), dtype = torch_dtype, scale = False
    )

class CBEDH5Dataset(Dataset):
    # pylint: disable=too-many-instance-attributes

    def __init__(
        self, path, split, target_columns,
        subdirs          = None,
        dtype            = 'float32',
        transform_image  = None,
        transform_labels = None,
        csv_suffix       = None,
        labels_file      = None,
    ):
        # pylint: disable=too-many-arguments

        if labels_file is not None:
            fname_labels = labels_file
        elif csv_suffix is None:
            fname_labels = f'labels_{split}.csv'
        else:
            fname_labels = f'labels_{split}_{csv_suffix}.csv'

        self._labels_root = path
        self._samples     = collect_datasets(path, subdirs, fname_labels)
        self._h5_index    = collect_h5_indices(path, subdirs)
        self._h5_files    = {}

        self._dtype = dtype

        assert transform_labels is None

        self._transform_image = transform_image
        self._target_columns  = target_columns

    def __len__(self):
        return len(self._samples)

    def get_h5_file(self, path):
        # pylint: disable=import-outside-toplevel
        import h5py
        # pylint: disable=unused-import
        import hdf5plugin

        if path not in self._h5_files:
            self._h5_files[path] = h5py.File(path, mode = 'r')

        return self._h5_files[path]

    def path_to_h5_sample(self, path):
        fname = os.path.relpath(path, self._labels_root)

        if fname not in self._h5_index:
            raise KeyError(f"Failed to find '{fname}' in H5 index.")

        return self._h5_index[fname]

    def __getitem__(self, idx):
        path, row       = self._samples[idx]
        h5_path, h5_idx = self.path_to_h5_sample(path)

        image = load_h5_image(self.get_h5_file(h5_path), h5_idx, self._dtype)

        targets = row[self._target_columns]
        targets = targets.astype(np.float32).values
        targets = torch.tensor(targets)

        if self._transform_image is not None:
            image = self._transform_image(image)

        return (image, targets)
