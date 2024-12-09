from torch.optim  import lr_scheduler
from ..base.funcs import extract_name_kwargs

SCHED_DICT = {
    'step'            : lr_scheduler.StepLR,
    'cosine'          : lr_scheduler.CosineAnnealingLR,
    'cosine-restarts' : lr_scheduler.CosineAnnealingWarmRestarts,
    'one-cycle'       : lr_scheduler.OneCycleLR,
    'constant'        : lr_scheduler.ConstantLR,
    'linear'          : lr_scheduler.LinearLR,
    'reduce-lr-on-plateau' : lr_scheduler.ReduceLROnPlateau,
}

def select_single_scheduler(optimizer, scheduler):
    if scheduler is None:
        return None

    name, kwargs = extract_name_kwargs(scheduler)

    if name not in SCHED_DICT:
        raise ValueError(
            f"Unknown scheduler: '{name}'. Supported: {SCHED_DICT.keys()}"
        )

    return SCHED_DICT[name](optimizer, **kwargs)

def select_scheduler(optimizer, scheduler, compose = False):
    if scheduler is None:
        return None

    if not isinstance(scheduler, (list, tuple)):
        scheduler = [ scheduler, ]

    result = [ select_single_scheduler(optimizer, x) for x in scheduler ]

    if compose:
        if len(result) == 1:
            return result[0]
        else:
            return lr_scheduler.ChainedScheduler(result)
    else:
        return result

