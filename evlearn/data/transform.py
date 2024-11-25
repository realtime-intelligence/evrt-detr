from evlearn.bundled.leanbase.torch.transforms import select_transform
from .transforms.noise import PoissonNoise, GaussNoise
from .transforms.misc  import ChannelDropout
from .transforms.parity_flip import ParityFlip

def select_custom_transform(name, **kwargs):
    if name == 'gauss-noise':
        return GaussNoise(**kwargs)

    if name == 'poisson-noise':
        return PoissonNoise(**kwargs)

    if name == 'channel-dropout':
        return ChannelDropout(**kwargs)

    if name == 'parity-flip':
        return ParityFlip(**kwargs)

    raise ValueError(f"Unknown transform: '{name}'")

def select_frame_transform(transform):
    return select_transform(
        transform, custom_select_fn = select_custom_transform
    )

def select_labels_transform(transform):
    return select_transform(transform)

def select_video_transform(transform):
    return select_transform(
        transform, custom_select_fn = select_custom_transform
    )

