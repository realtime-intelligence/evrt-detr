from evlearn.bundled.leanbase.base.config_base import ConfigBase

class TransferConfig(ConfigBase):

    __slots__ = [
        'base_model',
        'transfer_map',
        'strict',
        'load_train_state',
        'use_last_checkpoint',
        'fuzzy',
    ]

    def __init__(
        self,
        base_model,
        transfer_map        = None,
        strict              = True,
        load_train_state    = False,
        use_last_checkpoint = False,
        fuzzy               = None,
    ):
        # pylint: disable=too-many-arguments
        self.base_model          = base_model
        self.transfer_map        = transfer_map  or {}
        self.strict              = strict
        self.load_train_state    = load_train_state
        self.use_last_checkpoint = use_last_checkpoint
        self.fuzzy               = fuzzy

