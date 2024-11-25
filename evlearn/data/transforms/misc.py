import torch

from torchvision import tv_tensors
from torchvision.transforms.v2 import Transform

from .funcs import torchvision_isinstance

def channel_dropout(inpt, p = 0.01):
    if not torchvision_isinstance(inpt, (torch.Tensor, tv_tensors.Image)):
        return inpt

    n_channels = inpt.shape[-3]
    drop_mask  = (torch.rand(n_channels) < p)

    inpt[..., drop_mask, :, :] = 0

    return inpt

class ChannelDropout(Transform):

    def __init__(self, p = 0.01):
        super().__init__()
        self._p = p

    def _transform(self, inpt, params):
        return channel_dropout(inpt, self._p)

