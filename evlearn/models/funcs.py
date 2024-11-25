from functools import partial

import torch
from evlearn.bundled.leanbase.base.funcs import extract_name_kwargs

def linear_schedule(
    epoch, epochs_warmup, epochs_anneal, value_start, value_end
):
    if epoch is None:
        return value_end

    if epoch <= epochs_warmup:
        return value_start

    epochs_since_warmup = (epoch - epochs_warmup)

    if epochs_since_warmup >= epochs_anneal:
        return value_end

    return (
        value_start
        + (value_end - value_start) * epochs_since_warmup / epochs_anneal
    )

def constant_schedule(_epoch, value):
    return value

def select_value_schedule_fn(schedule_fn):
    name, kwargs = extract_name_kwargs(schedule_fn)

    if name == 'linear':
        return partial(linear_schedule, **kwargs)

    if name == 'constant':
        return partial(constant_schedule, **kwargs)

    raise ValueError(f"Unknown value schedule fn: {schedule_fn}")

def find_new_video_mask(clip_index, prev_clip_index = None):
    # clip_index      : (T, N, [ arr_idx, elem_idx ])
    # prev_clip_index : (N, [ arr_idx, elem_idx ])

    # clip_arr_index       : (T, N)
    clip_arr_index = clip_index[..., 0]
    (T, N)         = clip_arr_index.shape

    if T == 0:
        return torch.empty(
            (N, ), dtype = torch.bool, device = clip_index.device
        )

    if prev_clip_index is None:
        # first_frame_new : (N, )
        first_frame_new = torch.tensor(
            [ True, ] * N, dtype = torch.bool, device = clip_index.device
        )
    else:
        # prev_clip_arr_index  : (N)
        prev_clip_arr_index = prev_clip_index[..., 0]
        # first_frame_new : (N, )
        first_frame_new = (prev_clip_arr_index != clip_arr_index[0])

    # first_frame_new : (1, N)
    first_frame_new = first_frame_new.unsqueeze(0)

    if T == 1:
        # result : (1, N)
        return first_frame_new

    # subseq_frame_new : (T-1, N)
    subseq_frame_new = (clip_arr_index[1:] != clip_arr_index[:-1])

    # result : (T, N)
    result = torch.cat([ first_frame_new, subseq_frame_new ], dim = 0)

    return result

