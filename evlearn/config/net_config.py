from evlearn.bundled.leanbase.base.config_base import ConfigBase

class NetConfig(ConfigBase):
    # pylint: disable=too-many-instance-attributes

    __slots__ = [
        'model',
    ]

    def __init__(self, model):
        # pylint: disable=too-many-arguments
        self.model = model

