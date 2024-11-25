import logging

from torch import nn
from evlearn.bundled.rtdetr_pytorch.nn.backbone.presnet import PResNet

LOGGER = logging.getLogger('evlearn.nn.backbone.presnet_rtdetr')

class PResNetRTDETR(nn.Module):

    def __init__(
        self, input_shape, depth,
        variant     = 'd',
        num_stages  = 4,
        return_idx  = (0, 1, 2, 3),
        act         = 'relu',
        freeze_at   = -1,
        freeze_norm = False,
        pretrained  = False,
    ):
        # pylint: disable=too-many-arguments
        super().__init__()

        if pretrained:
            LOGGER.warning(
                'Trying to use PResNet bacbkone pre-trained on natural images'
                '. Make sure you know what you are doing.'
            )

        if freeze_norm:
            LOGGER.warning(
                'Trying to use PResNet with frozen BN'
                '. Make sure you know what you are doing.'
            )

        if freeze_at >= 0:
            LOGGER.warning(
                'Trying to use PResNet with frozen parameters'
                '. Make sure you know what you are doing.'
            )

        self._net = PResNet(
            features_input = input_shape[0],
            depth          = depth,
            variant        = variant,
            num_stages     = num_stages,
            return_idx     = return_idx,
            act            = act,
            freeze_at      = freeze_at,
            freeze_norm    = freeze_norm,
            pretrained     = pretrained,
        )

    @property
    def fpn_shapes(self):
        raise NotImplementedError

    def forward(self, x):
        return self._net(x)

