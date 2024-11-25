import torch

def clip_gradients(optimizer, norm = None, value = None):
    if (norm is None) and (value is None):
        return

    params = [
        param
            for param_group in optimizer.param_groups
                for param in param_group['params']
    ]

    if norm is not None:
        torch.nn.utils.clip_grad_norm_(params, max_norm = norm)

    if value is not None:
        torch.nn.utils.clip_grad_value_(params, clip_value = value)


# The following code is from uvcgan2 project.
# Copyright (c) 2021-2023, The LS4GAN Project Developers.
# Source: https://github.com/LS4GAN/uvcgan2
# Licensed under BSD-2-Clause License
@torch.no_grad()
def update_ema_model(ema_model, model, momentum):
    online_params = dict(model.named_parameters())
    online_bufs   = dict(model.named_buffers())

    for (k, v) in ema_model.named_parameters():
        if v.ndim == 0:
            v.copy_(momentum * v + (1 - momentum) * online_params[k])
        else:
            v.lerp_(online_params[k], (1 - momentum))

    for (k, v) in ema_model.named_buffers():
        if v.ndim == 0:
            v.copy_(momentum * v + (1 - momentum) * online_bufs[k])
        else:
            v.lerp_(online_bufs[k], (1 - momentum))

