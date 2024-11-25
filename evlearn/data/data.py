from torch.utils.data import DataLoader

from evlearn.bundled.leanbase.base.data_loader_zipper import (
    DataLoaderListZipper, DataLoaderDictZipper
)

from .datasets  import select_dataset
from .samplers  import select_sampler
from .collate   import select_collate_fn
from .transform import (
    select_frame_transform, select_labels_transform, select_video_transform
)

# NOTE: pin_memory does not play well with tv_tensors
def construct_single_data_loader(
    data_config, split, pin_memory = False, persistent_workers = False
):
    transform_video  = select_video_transform(data_config.transform_video)
    transform_frame  = select_frame_transform(data_config.transform_frame)
    transform_labels = select_labels_transform(data_config.transform_labels)

    dataset = select_dataset(
        data_config.dataset, split,
        transform_video, transform_frame, transform_labels
    )

    sampler = select_sampler(
        data_config.sampler, dataset, data_config.batch_size
    )

    collate_fn = select_collate_fn(data_config.collate)

    dl = DataLoader(
        dataset,
        batch_sampler = sampler,
        collate_fn    = collate_fn,
        num_workers   = data_config.workers,
        pin_memory    = pin_memory,
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

