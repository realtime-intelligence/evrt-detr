from torch import nn
from evlearn.bundled.yolox.models.yolo_pafpn import YOLOPAFPN
from evlearn.bundled.yolox.models.yolo_fpn   import YOLOFPN

class PAFPNYoloX(nn.Module):

    def __init__(
        self, input_shape,
        depth       = 1.0,
        width       = 1.0,
        in_features = ("dark3", "dark4", "dark5"),
        in_channels = (256, 512, 1024),
        depthwise   = False,
        act         = "silu",
    ):
        # pylint: disable=too-many-arguments
        super().__init__()

        self._net = YOLOPAFPN(
            depth       = depth,
            width       = width,
            in_features = in_features,
            in_channels = in_channels,
            depthwise   = depthwise,
            act         = act,
            input_in_channels = input_shape[0]
        )

    @property
    def fpn_shapes(self):
        raise NotImplementedError

    def forward(self, x):
        return self._net(x)

class FPNYoloX(nn.Module):

    def __init__(
        self, input_shape,
        depth       = 53,
        in_features = ("dark3", "dark4", "dark5"),
    ):
        # pylint: disable=too-many-arguments
        super().__init__()

        self._net = YOLOFPN(
            depth       = depth,
            in_features = in_features,
            input_in_channels = input_shape[0],
        )

    @property
    def fpn_shapes(self):
        raise NotImplementedError

    def forward(self, x):
        return self._net(x)

