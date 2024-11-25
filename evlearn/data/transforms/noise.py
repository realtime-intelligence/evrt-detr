import torch

from torchvision import tv_tensors
from torchvision.transforms.v2 import Transform

from .funcs import torchvision_isinstance

def poisson_noise(inpt, preserve_empty = True):
    if not torchvision_isinstance(inpt, (torch.Tensor, tv_tensors.Image)):
        return inpt

    result = torch.poisson(inpt)

    if preserve_empty:
        result[(inpt == 0)] = 0

    return result

def gauss_noise(
    inpt, sigma = 1, multiplicative = False, preserve_empty = True
):
    if not torchvision_isinstance(inpt, (torch.Tensor, tv_tensors.Image)):
        return inpt

    noise  = sigma * torch.randn_like(inpt)

    if multiplicative:
        result = inpt * (1 + noise)
    else:
        result = inpt + noise

    if preserve_empty:
        result[(inpt == 0)] = 0

    return (inpt + noise)

class PoissonNoise(Transform):

    def __init__(self, preserve_empty = True):
        super().__init__()
        self._preserve_empty = preserve_empty

    def _transform(self, inpt, params):
        return poisson_noise(inpt, preserve_empty = self._preserve_empty)

class GaussNoise(Transform):

    def __init__(
        self, sigma = 1, multiplicative = False, preserve_empty = True
    ):
        super().__init__()
        self._sigma = sigma
        self._preserve_empty = preserve_empty
        self._multiplicative = multiplicative

    def _transform(self, inpt, params):
        return gauss_noise(
            inpt, self._sigma, self._multiplicative, self._preserve_empty
        )

