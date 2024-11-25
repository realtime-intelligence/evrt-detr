from evlearn.bundled.leanbase.base.funcs import extract_name_kwargs
from .jarr_sampler       import JArrSampler, VideoSampler
from .jarr_subseq_sampler  import JArrSubseqSampler, VideoClipSampler
from .jarr_random        import JArrRandomSampler

SAMPLER_DICT = {
    'jagged-array'        : JArrSampler,
    'jagged-array-subseq' : JArrSubseqSampler,
    'video-element'       : VideoSampler,
    'video-clip'          : VideoClipSampler,
    'random-jarr-element' : JArrRandomSampler,
}

def select_sampler(sampler, dataset, batch_size):
    # pylint: disable=too-many-arguments

    name, kwargs = extract_name_kwargs(sampler)

    if name not in SAMPLER_DICT:
        raise ValueError(
            f"Unknown sampler: '{name}'."
            f"  Supported: {list(SAMPLER_DICT.keys())}."
        )

    return SAMPLER_DICT[name](
        dataset = dataset, batch_size = batch_size, **kwargs
    )

