from torch.utils.data import DataLoader

from cdi.bundled.leanbase.base.data_loader_zipper import (
    DataLoaderListZipper, DataLoaderDictZipper
)

from .datasets  import select_dataset
from .transform import (
    select_image_transform, select_labels_transform
)

def construct_single_data_loader(
    data_config, split, pin_memory = False, persistent_workers = False
):
    transform_image  = select_image_transform(data_config.transform_image)
    transform_labels = select_labels_transform(data_config.transform_labels)

    dataset = select_dataset(
        data_config.dataset, split,
        transform_image, transform_labels
    )

    dl = DataLoader(
        dataset,
        batch_size  = data_config.batch_size,
        shuffle     = data_config.shuffle,
        drop_last   = data_config.drop_last,
        num_workers = data_config.workers,
        pin_memory  = pin_memory,
        persistent_workers = persistent_workers,
    )

    return dl

def construct_data_loader(data_config, split, pin_memory = False):
    if isinstance(data_config, list):
        loaders = [
            construct_single_data_loader(conf, split, pin_memory)
                for conf in data_config
        ]

        return DataLoaderListZipper(loaders)

    if isinstance(data_config, dict):
        loaders = {
            k : construct_single_data_loader(conf, split, pin_memory)
                for (k ,conf) in data_config.items()
        }

        return DataLoaderDictZipper(loaders)

    return construct_single_data_loader(data_config, split, pin_memory)
