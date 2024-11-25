import os

from evlearn.bundled.leanbase.base.funcs import extract_name_kwargs

from evlearn.consts import ROOT_DATA
from .ebc_video_frame_dataset    import EBCVideoFrameDataset
from .ebc_video_h5frame_dataset  import EBCVideoH5FrameDataset
from .rvt_video_h5frame_dataset  import RVTVideoFrameDataset

DSET_DICT = {
    'ebc-video-frame'   : EBCVideoFrameDataset,
    'ebc-video-h5frame' : EBCVideoH5FrameDataset,
    'rvt-video-frame'   : RVTVideoFrameDataset,
}

def select_dataset(
    dataset, split, transform_video, transform_frame, transform_labels
):
    # pylint: disable=too-many-arguments

    name, kwargs = extract_name_kwargs(dataset)
    if 'path' in kwargs:
        kwargs['path'] = os.path.join(ROOT_DATA, kwargs['path'])

    if name not in DSET_DICT:
        raise ValueError(
            f"Unknown dataset: '{name}'. Supported: {list(DSET_DICT.keys())}."
        )

    return DSET_DICT[name](
        split = split,
        transform_video  = transform_video,
        transform_frame  = transform_frame,
        transform_labels = transform_labels,
        **kwargs
    )

