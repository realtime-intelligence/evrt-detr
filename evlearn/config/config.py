import os

from evlearn.bundled.leanbase.base.config_base import ConfigBase
from .data_config import DataConfig
from .net_config import NetConfig
from .transfer_config import TransferConfig


class Config(ConfigBase):

    # pylint: disable=too-many-instance-attributes

    __slots__ = [
        'data',
        'nets',
        'epochs',
        'model',
        'losses',
        'optimizers',
        'schedulers',
        'seed',
        'val_interval',
        'steps_per_train_epoch',
        'steps_per_val_epoch',
        'transfer',
    ]

    def __init__(
        self,
        data,
        epochs,
        model,
        nets,
        losses,
        optimizers,
        schedulers            = None,
        seed                  = 0,
        val_interval          = 1,
        steps_per_train_epoch = None,
        steps_per_val_epoch   = None,
        transfer              = None,
    ):
        # pylint: disable=too-many-arguments
        data      = data or {}
        self.data = DataConfig(**data)

        self.epochs = epochs
        self.seed   = seed
        self.model  = model

        nets      = nets or {}
        self.nets = { k : NetConfig(**v) for (k, v) in nets.items() }

        self.losses     = losses     or {}
        self.optimizers = optimizers or {}
        self.schedulers = schedulers or {}

        self.val_interval          = val_interval
        self.steps_per_train_epoch = steps_per_train_epoch
        self.steps_per_val_epoch   = steps_per_val_epoch

        if transfer is not None:
            transfer = TransferConfig(**transfer)

        self.transfer = transfer

    def get_savedir(self, outdir, label = None):
        if label is None:
            label = self.get_hash()

        model = self.model
        if isinstance(model, dict):
            model = model.get('name', 'unkn')

        savedir = f'model_m({model})_{label}'

        savedir = savedir.replace('/', ':')
        path    = os.path.join(outdir, savedir)

        os.makedirs(path, exist_ok = True)
        return path

    def to_dict(self):
        result = {
            x : getattr(self, x) for x in self.__slots__ if x != 'misc'
        }

        return result

