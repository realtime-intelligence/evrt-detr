from evlearn.bundled.leanbase.base.config_base import ConfigBase

class DatasetConfig(ConfigBase):
    # pylint: disable=too-many-instance-attributes

    __slots__ = [
        'batch_size',
        'dataset',
        'sampler',
        'collate',
        'shapes',
        'transform_video',
        'transform_frame',
        'transform_labels',
        'workers',
    ]

    def __init__(
        self, batch_size, dataset, sampler, collate, shapes,
        transform_video  = None,
        transform_frame  = None,
        transform_labels = None,
        workers          = 1,
    ):
        # pylint: disable=too-many-arguments
        self.batch_size       = batch_size
        self.dataset          = dataset
        self.sampler          = sampler
        self.collate          = collate
        self.shapes           = shapes
        self.transform_video  = transform_video
        self.transform_frame  = transform_frame
        self.transform_labels = transform_labels
        self.workers          = workers

def unpack_dataset_config(config):
    if isinstance(config, (list, tuple)):
        return [ DatasetConfig(**x) for x in config ]

    if isinstance(config, dict):
        if (
                ('batch_size' in config)
            and ('dataset' in config)
            and ('sampler' in config)
        ):
            return DatasetConfig(**config)
        else:
            return { k : DatasetConfig(**x) for (k, x) in config.items() }

    raise ValueError(f'Unknown data config type: {type(config)}')

class DataConfig(ConfigBase):
    # pylint: disable=too-many-instance-attributes

    __slots__ = [
        'train',
        'eval',
    ]

    def __init__(self, train, eval):
        # pylint: disable=too-many-arguments
        # pylint: disable=redefined-builtin

        self.train = unpack_dataset_config(train)
        self.eval  = unpack_dataset_config(eval)

