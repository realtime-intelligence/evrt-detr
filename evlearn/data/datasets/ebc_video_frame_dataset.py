"""
Dataset implementation for event-based camera (EBC) videos stored in numpy
frame format.

This module provides a concrete implementation of JaggedArrayDataset for
handling event camera video data. Each video is represented as a directory
containing frame-wise data and optional label in .npz format, organized as:

    PATH_TO_DATASET/split/
        +-- video1/
        |     +-- data_00001.npz
        |     +-- labels_0001.npz
        |     +-- data_0002.npz
        |     +-- labels_0002.npz
        |     +-- ...
        +-- video2/
        |     +-- data_00001.npz
        |     +-- ...
        +-- ...

For detailed format of data and label .npz files, refer to funcs_npz.py
"""

import os
import torch

from .jagged_array_dataset import JaggedArrayDataset
from .jagged_array_specs   import SimpleJaggedArraySpecs
from .funcs       import cantor_pairing, nan_sanitize
from .funcs_npz   import collect_videos, load_frame
from .funcs_frame import apply_transforms_to_data

class EBCVideoFrameDataset(JaggedArrayDataset):
    # pylint: disable=too-many-instance-attributes

    def __init__(
        self, path, split,
        skip_unlabeled       = False,
        data_dtype_list      = [ ('frame', 'float32'), ],
        label_dtype_list     = [ ('boxes', 'float32'), ('labels', 'int32'), ],
        transform_video      = None,
        transform_frame      = None,
        transform_labels     = None,
        bbox_fmt             = 'XYXY',
        canvas_size          = None,
        squash_time_polarity = True,
        return_vdir_fname    = False,
        return_index         = False,
    ):
        # pylint: disable=too-many-arguments
        # pylint: disable=dangerous-default-value
        self._videos_root = os.path.join(path, split)

        video_specs = collect_videos(self._videos_root, skip_unlabeled)
        video_specs = SimpleJaggedArraySpecs(video_specs)

        super().__init__(video_specs)

        self._data_dtype_list  = data_dtype_list
        self._label_dtype_list = label_dtype_list

        self._transform_video  = transform_video
        self._transform_frame  = transform_frame
        self._transform_labels = transform_labels

        self._squash_pt   = squash_time_polarity
        self._bbox_fmt    = bbox_fmt
        self._canvas_size = canvas_size

        self._return_vdir_fname = return_vdir_fname
        self._return_index      = return_index

    def get_null_elem(self):
        result = tuple(None for _ in range(len(self._data_dtype_list)+1))

        if self._return_index:
            result += (None, )

        return result

    def get_video_seed(self, arr_idx):
        return cantor_pairing(self._seed, arr_idx)

    def get_elem(self, arr_idx, elem_idx):
        vdir = self._specs.get_array_name(arr_idx)

        (_, fname_data, fname_labels) \
            = self._specs.get_elem_spec(arr_idx, elem_idx)

        data, labels = load_frame(
            self._videos_root, vdir, fname_data, fname_labels,
            self._data_dtype_list, self._label_dtype_list,
            self._bbox_fmt, self._canvas_size
        )

        nan_sanitize(data[0], f'Raw data has NaNs: File: {vdir}/{fname_data}')

        data, labels = apply_transforms_to_data(
            data, labels, self._transform_video, self._transform_frame,
            self._transform_labels, self._squash_pt,
            self.get_video_seed(arr_idx)
        )

        nan_sanitize(
            data[0],
            f'Data after transforms has NaN: File: {vdir}/{fname_data}'
        )

        if self._return_vdir_fname:
            if labels is None:
                labels = {}

            labels['vdir'] = vdir
            labels['fname_data']   = fname_data
            labels['fname_labels'] = fname_labels

        if self._return_index:
            data.append(
                torch.Tensor([ arr_idx, elem_idx]).to(dtype = torch.int32)
            )

        # unpack for bkw compat
        return (*data, labels)

if __name__ == '__main__':
    import sys

    dset = EBCVideoFrameDataset(
        sys.argv[1], split = 'test', skip_unlabeled = False,
        return_vdir_fname = True
    )

    import IPython
    IPython.embed()

