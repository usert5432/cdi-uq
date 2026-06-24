import os

from cdi.bundled.leanbase.base.funcs import extract_name_kwargs

from cdi.consts import ROOT_DATA
from .cbed_dataset import CBEDDataset
from .cbed_h5_dataset import CBEDH5Dataset
from .gaussian_mixture_2d import GaussianMixture2DDataset


DSET_DICT = {
    'cbed'                : CBEDDataset,
    'cbed-h5'             : CBEDH5Dataset,
    'gaussian-mixture-2d' : GaussianMixture2DDataset,
}


def select_dataset(
    dataset, split, transform_image, transform_labels
):
    name, kwargs = extract_name_kwargs(dataset)
    if 'path' in kwargs:
        kwargs['path'] = os.path.join(ROOT_DATA, kwargs['path'])

    if name not in DSET_DICT:
        raise ValueError(
            f"Unknown dataset: '{name}'. Supported: {list(DSET_DICT.keys())}."
        )

    return DSET_DICT[name](
        split = split,
        transform_image  = transform_image,
        transform_labels = transform_labels,
        **kwargs
    )
