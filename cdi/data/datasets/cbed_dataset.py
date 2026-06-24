import os

import torch

import numpy as np
import pandas as pd

from torch.utils.data import Dataset
from torchvision.transforms.v2.functional import to_image, to_dtype

def load_image(path, dtype):
    # TODO: Fix this hack
    torch_dtype = getattr(torch, dtype)
    np_dtype    = getattr(np,    dtype)

    f     = np.load(path)

    image = f[f.files[0]]
    image = image.astype(np_dtype)

    return to_dtype(
        to_image(image), dtype = torch_dtype, scale = False
    )

def collect_single_dataset(root, subdir, fname_labels):
    if subdir is None:
        path   = root
    else:
        path = os.path.join(root, subdir)

    path_labels = os.path.join(path, fname_labels)
    df = pd.read_csv(path_labels)

    return [
        (os.path.join(path, row.file), row) for (_idx, row) in df.iterrows()
    ]

def collect_datasets(root, subdirs, fname_labels):
    if subdirs is None:
        return collect_single_dataset(root, None, fname_labels)

    result = []

    for subdir in subdirs:
        result.extend(collect_single_dataset(root, subdir, fname_labels))

    return result

class CBEDDataset(Dataset):
    # pylint: disable=too-many-instance-attributes

    def __init__(
        self, path, split, target_columns,
        subdirs          = None,
        dtype            = 'float32',
        transform_image  = None,
        transform_labels = None,
        csv_suffix       = None,
    ):
        # pylint: disable=too-many-arguments
        # pylint: disable=too-many-locals

        if csv_suffix is None:
            fname_labels = f'labels_{split}.csv'
        else:
            fname_labels = f'labels_{split}_{csv_suffix}.csv'

        self._root  = path

        self._samples = collect_datasets(path, subdirs, fname_labels)
        self._dtype = dtype

        assert transform_labels is None

        self._transform_image = transform_image
        self._target_columns  = target_columns

    def __len__(self):
        return len(self._samples)

    def __getitem__(self, idx):
        path, row = self._samples[idx]

        image   = load_image(path, self._dtype)

        targets = row[self._target_columns]
        targets = targets.astype(np.float32).values
        targets = torch.tensor(targets)

        if self._transform_image is not None:
            image = self._transform_image(image)

        return (image, targets)

if __name__ == '__main__':
    import sys

    dset = CBEDDataset(
        sys.argv[1],
        subdirs = [ 'f11a', 'f11b', ],
        split   = 'train',
        target_columns = [ 'label', ],
    )

    print(len(dset))
