"""
Dataset implementation for event-based camera (EBC) videos stored in HDF5
frame format.

This module provides a concrete implementation of JaggedArrayDataset for
handling event camera video data. Data frames are stored in a single HDF5 file
with optional labels in separate directories, organized as:

    PATH_TO_DATASET/split/
        +-- frames.h5          # All frames in a single H5 file
        +-- video1/            # Directories with labels
        |     +-- labels_0001.npz
        |     +-- labels_0002.npz
        |     +-- ...
        +-- video2/
        |     +-- labels_0001.npz
        |     +-- ...
        +-- ...

For detailed format of data and label files, refer to funcs_hdf.py
"""

import os
import h5py
import torch

from .jagged_array_dataset import JaggedArrayDataset

from .funcs       import cantor_pairing
from .funcs_hdf   import HDF5JaggedArraySpecs, load_h5frame, FNAME_FRAMES
from .funcs_frame import apply_transforms_to_data

class EBCVideoH5FrameDataset(JaggedArrayDataset):
    # pylint: disable=too-many-instance-attributes

    def __init__(
        self, path, split,
        data_dtype_list      = [ ('frame', 'float32'), ],
        label_dtype_list     = [ ('boxes', 'float32'), ('labels', 'int32'), ],
        transform_video      = None,
        transform_frame      = None,
        transform_labels     = None,
        bbox_fmt             = 'XYXY',
        canvas_size          = None,
        squash_time_polarity = True,
        return_index         = False,
    ):
        # pylint: disable=too-many-arguments
        # pylint: disable=dangerous-default-value
        self._videos_root = os.path.join(path, split)
        self._frames_path = os.path.join(self._videos_root, FNAME_FRAMES)

        self._file = h5py.File(self._frames_path, mode = 'r')

        video_specs = HDF5JaggedArraySpecs(self._file)
        super().__init__(video_specs)

        self._data_dtype_list  = data_dtype_list
        self._label_dtype_list = label_dtype_list

        self._transform_video  = transform_video
        self._transform_frame  = transform_frame
        self._transform_labels = transform_labels

        self._squash_pt      = squash_time_polarity
        self._bbox_fmt       = bbox_fmt
        self._canvas_size    = canvas_size
        self._return_index = return_index

    def get_null_elem(self):
        result = (None,) * len(self._data_dtype_list)

        if self._return_index:
            result = result + (None, )

        return result + (None, )

    def get_video_seed(self, arr_idx):
        return cantor_pairing(self._seed, arr_idx)

    def get_elem(self, arr_idx, elem_idx):
        vdir = self._specs.get_array_name(arr_idx)
        (_, _, fname_labels) = self._specs.get_elem_spec(arr_idx, elem_idx)

        data, labels = load_h5frame(
            self._file, self._videos_root, vdir, elem_idx, fname_labels,
            self._data_dtype_list, self._label_dtype_list,
            self._bbox_fmt, self._canvas_size
        )

        data, labels = apply_transforms_to_data(
            data, labels, self._transform_video, self._transform_frame,
            self._transform_labels, self._squash_pt,
            self.get_video_seed(arr_idx)
        )

        if self._return_index:
            data.append(
                torch.Tensor([ arr_idx, elem_idx ]).to(dtype = torch.int32)
            )

        # unpack for bkw compat
        return (*data, labels)

if __name__ == '__main__':
    import sys

    dset = EBCVideoH5FrameDataset(sys.argv[1], split = 'test')

    import IPython
    IPython.embed()

