import torch
from torch import nn

from ..base.funcs import extract_name_kwargs

def select_norm_layer(norm, features):
    name, kwargs = extract_name_kwargs(norm)

    if name is None:
        return nn.Identity(**kwargs)

    if name == 'layer':
        return nn.LayerNorm((features,), **kwargs)

    if name == 'batch-2d':
        return nn.BatchNorm2d(features, **kwargs)

    if name == 'group':
        return nn.GroupNorm(num_channels = features, **kwargs)

    if name == 'instance-2d':
        return nn.InstanceNorm2d(features, **kwargs)

    raise ValueError(f"Unknown Normalization Layer: '{name}'")

def select_norm_layer_fn(norm):
    return lambda features : select_norm_layer(norm, features)

def select_activation(activ, inplace = True):
    # pylint: disable=too-many-return-statements
    name, kwargs = extract_name_kwargs(activ)

    if (name is None) or (name == 'linear'):
        return nn.Identity()

    if name == 'gelu':
        return nn.GELU(**kwargs)

    if name == 'selu':
        return nn.SELU(**kwargs)

    if name == 'relu':
        return nn.ReLU(inplace = inplace, **kwargs)

    if name == 'leakyrelu':
        return nn.LeakyReLU(inplace = inplace, **kwargs)

    if name == 'tanh':
        return nn.Tanh()

    if name == 'silu':
        return nn.SiLU(inplace = inplace, **kwargs)

    if name == 'sigmoid':
        return nn.Sigmoid()

    raise ValueError(f"Unknown activation: '{name}'")

def select_optimizer(parameters, optimizer):
    name, kwargs = extract_name_kwargs(optimizer)

    if name == 'AdamW':
        return torch.optim.AdamW(parameters, **kwargs)

    if name == 'Adam':
        return torch.optim.Adam(parameters, **kwargs)

    raise ValueError(f"Unknown optimizer: '{name}'")

def select_loss(loss, reduction = None):
    name, kwargs = extract_name_kwargs(loss)

    if reduction is not None:
        kwargs['reduction'] = reduction

    if name.lower() in [ 'l1', 'mae' ]:
        return nn.L1Loss(**kwargs)

    if name.lower() in [ 'l2', 'mse' ]:
        return nn.MSELoss(**kwargs)

    if name.lower() in [ 'bce', 'binary-cross-entropy' ]:
        return nn.BCELoss(**kwargs)

    if name.lower() in [ 'bce-logits', ]:
        return nn.BCEWithLogitsLoss(**kwargs)

    raise ValueError(f"Unknown loss: '{name}'")

