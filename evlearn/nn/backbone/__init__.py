from evlearn.bundled.leanbase.base.funcs  import extract_name_kwargs

from .presnet_rtdetr import PResNetRTDETR
from .yolox_fpn      import FPNYoloX, PAFPNYoloX

BACKBONES = {
    'presnet-rtdetr' : PResNetRTDETR,
    'fpn-yolox'      : FPNYoloX,
    'pafpn-yolox'    : PAFPNYoloX,
}

def select_backbone(backbone, input_shape):
    name, kwargs = extract_name_kwargs(backbone)

    if name not in BACKBONES:
        raise ValueError(f'Unknown backbone: {name}')

    return BACKBONES[name](input_shape = input_shape, **kwargs)

def construct_backbone(backbone, input_shape, device):
    model = select_backbone(backbone.model, input_shape)
    model = model.to(device)

    return model

