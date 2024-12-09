import torch

try:
    from torchvision.transforms import v2 as transforms
    HAS_V2 = True
except ImportError:
    print("torchvision v2 transforms not found")
    from torchvision import transforms
    HAS_V2 = False

from ..base.funcs import extract_name_kwargs

# pylint: disable=unnecessary-lambda-assignment
FromNumpy = lambda : torch.from_numpy

TRANSFORM_DICT = {
    'center-crop'            : transforms.CenterCrop,
    'color-jitter'           : transforms.ColorJitter,
    'random-affine'          : transforms.RandomAffine,
    'random-perspective'     : transforms.RandomPerspective,
    'random-erasing'         : transforms.RandomErasing,
    'random-crop'            : transforms.RandomCrop,
    'random-flip-vertical'   : transforms.RandomVerticalFlip,
    'random-flip-horizontal' : transforms.RandomHorizontalFlip,
    'random-rotation'        : transforms.RandomRotation,
    'random-resize-crop'     : transforms.RandomResizedCrop,
    'random-solarize'        : transforms.RandomSolarize,
    'random-invert'          : transforms.RandomInvert,
    'gaussian-blur'          : transforms.GaussianBlur,
    'resize'                 : transforms.Resize,
    'normalize'              : transforms.Normalize,
    'pad'                    : transforms.Pad,
    'grayscale'              : transforms.Grayscale,
    'to-tensor'              : transforms.ToTensor,
    'from-numpy'             : FromNumpy,
    'CenterCrop'             : transforms.CenterCrop,
    'ColorJitter'            : transforms.ColorJitter,
    'RandomCrop'             : transforms.RandomCrop,
    'RandomVerticalFlip'     : transforms.RandomVerticalFlip,
    'RandomHorizontalFlip'   : transforms.RandomHorizontalFlip,
    'RandomRotation'         : transforms.RandomRotation,
    'Resize'                 : transforms.Resize,
}

if HAS_V2:
    TRANSFORM_DICT.update({
        'random-zoom-out' : transforms.RandomZoomOut,
        'sanitize-bboxes' : transforms.SanitizeBoundingBoxes,
        'clamp-bboxes'    : transforms.ClampBoundingBoxes,
        'elastic'         : transforms.ElasticTransform,
        'scale-jitter'    : transforms.ScaleJitter,
        'random-iou-crop' : transforms.RandomIoUCrop,
        'random-channel-permutation' : transforms.RandomChannelPermutation,
    })

INTERPOLATION_DICT = {
    'nearest'  : transforms.InterpolationMode.NEAREST,
    'bilinear' : transforms.InterpolationMode.BILINEAR,
    'bicubic'  : transforms.InterpolationMode.BICUBIC,
    'lanczos'  : transforms.InterpolationMode.LANCZOS,
    'box'      : transforms.InterpolationMode.BOX,
}

def parse_interpolation(kwargs):
    if 'interpolation' in kwargs:
        kwargs['interpolation'] = INTERPOLATION_DICT[kwargs['interpolation']]

def select_single_transform(transform, custom_select_fn = None):
    name, kwargs = extract_name_kwargs(transform)

    if name == 'random-apply':
        transform = select_transform_basic(
            kwargs.pop('transforms'), custom_select_fn
        )
        return transforms.RandomApply(transform, **kwargs)

    if name == 'random-order':
        transform = select_transform_basic(
            kwargs.pop('transforms'), custom_select_fn
        )
        return transforms.RandomOrder(transform, **kwargs)

    if name == 'random-choice':
        transform = select_transform_basic(
            kwargs.pop('transforms'), custom_select_fn
        )
        return transforms.RandomChoice(transform, **kwargs)

    if name == 'compose':
        transform = select_transform_basic(
            kwargs.pop('transforms'), custom_select_fn
        )
        return transforms.Compose(transform, **kwargs)

    if name not in TRANSFORM_DICT:
        if custom_select_fn is None:
            raise ValueError(f"Unknown transform: '{name}'")

        return custom_select_fn(name, **kwargs)

    parse_interpolation(kwargs)

    return TRANSFORM_DICT[name](**kwargs)

def select_transform_basic(
    transform, custom_select_fn = None, compose = False
):
    result = []

    if transform is not None:
        if not isinstance(transform, (list, tuple)):
            transform = [ transform, ]

        result = [
            select_single_transform(x, custom_select_fn) for x in transform
        ]

    if compose:
        if len(result) == 1:
            return result[0]
        else:
            return transforms.Compose(result)
    else:
        return result

def select_transform(transform, custom_select_fn = None):
    result = select_transform_basic(transform, custom_select_fn)

    if not result:
        return None

    return transforms.Compose(result)

