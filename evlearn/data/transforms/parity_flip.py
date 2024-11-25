import torch

from torchvision import tv_tensors
from torchvision.transforms.v2 import Transform

from .funcs import torchvision_isinstance

class ParityFlip(Transform):

    def __init__(self, p = 0.5):
        super().__init__()
        self._p = p

    def _transform(self, inpt, params):
        if not torchvision_isinstance(inpt, (torch.Tensor, tv_tensors.Image)):
            return inpt

        q = torch.rand(1)

        if q < self._p:
            return inpt

        result = inpt
        need_squeezing = False

        if result.ndim == 3:
            # result : (P * T, H, W)
            (PT, H, W) = result.shape

            assert PT % 2 == 0
            T = PT // 2

            result = result.reshape(2, T, H, W)
            need_squeezing = True

        # result : (P, T, H, W)
        # NOTE: need to flip polarity (P) + motion (T)
        result = torch.flip(result, [ 0, 1 ])

        if need_squeezing:
            result = result.reshape((-1, *result.shape[-2:]))

        return result

