from evlearn.bundled.leanbase.base.funcs  import extract_name_kwargs

from .conv_lstm import TemporalConvLSTM

TEMPORAL_DICT = {
    'conv-lstm' : TemporalConvLSTM,
}

def select_temporal(temporal):
    name, kwargs = extract_name_kwargs(temporal)

    if name not in TEMPORAL_DICT:
        raise ValueError(f"Unknown Temporal Encoder: '{name}'")

    return TEMPORAL_DICT[name](**kwargs)

def construct_temporal(temporal, device):
    model = select_temporal(temporal.model)
    model = model.to(device)

    return model

