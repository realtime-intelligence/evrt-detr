import logging
from torch import nn

from ..base.funcs import extract_name_kwargs

LOGGER = logging.getLogger('leanbase.torch')

WINIT_DICT = {
    'normal'     : nn.init.normal_,
    'xavier'     : nn.init.xavier_normal_,
    'kaiming'    : nn.init.kaiming_normal_,
    'orthogonal' : nn.init.orthogonal_,
}

WINIT_CLASSES = (
    nn.Conv1d,
    nn.Conv2d,
    nn.Conv3d,
    nn.ConvTranspose1d,
    nn.ConvTranspose2d,
    nn.ConvTranspose3d,
    nn.Linear,
)

def winit_callback(module, winit_fn, **kwargs):
    if not hasattr(module, 'weight'):
        return

    if not isinstance(module, WINIT_CLASSES):
        return

    winit_fn(module.weight.data, **kwargs)

def select_winit_fn(winit):
    name, kwargs = extract_name_kwargs(winit)

    if name not in WINIT_DICT:
        raise ValueError(f"Unknown init method: {name}")

    return WINIT_DICT[name], kwargs

def init_weights(module, winit):
    if winit is None:
        return

    winit_fn, winit_kwargs = select_winit_fn(winit)

    LOGGER.debug("Initializnig network with '%s'", winit)

    module.apply(
        lambda module, winit_fn=winit_fn, kwargs=winit_kwargs : \
            winit_callback(module, winit_fn, **kwargs)
    )

