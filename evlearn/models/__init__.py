from evlearn.bundled.leanbase.base.funcs  import extract_name_kwargs

from .frame_detection_rtdetr       import FrameDetectionRTDETR
from .frame_detection_yolox        import FrameDetectionYoloX
from .vcf_detection_evrtdetr       import VCFDetectionEvRTDETR

MODELS_DICT = {
    'frame-detection-rtdetr'       : FrameDetectionRTDETR,
    'frame-detection-yolox'        : FrameDetectionYoloX,
    'vcf-detection-evrtdetr'       : VCFDetectionEvRTDETR,
}

def select_model(model, **kwargs):
    name, model_kwargs = extract_name_kwargs(model)

    if name not in MODELS_DICT:
        raise ValueError(f"Unknown model: '{name}'")

    return MODELS_DICT[name](**kwargs, **model_kwargs)

def construct_model(config, device, dtype, init_train, savedir):
    model = select_model(
        config.model,
        config     = config,
        device     = device,
        dtype      = dtype,
        init_train = init_train,
        savedir    = savedir,
    )

    return model

