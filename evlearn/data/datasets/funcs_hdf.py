"""
Utility functions for handling event camera data stored in HDF5 format.

This module provides functions for loading and managing event camera data with
frame data stored in HDF5 files and labels stored in .npz files. The data must
be organized as:

    +-- frames.h5          # All frames data in a single HDF5 file
    +-- video1/            # Labels stored per video
          +-- labels_0001.npz
          +-- labels_0002.npz
          +-- ...

where
- frames.h5: Contains all frame data in HDF5 format with the following
  structure:

  Root level:
   * `index`: Global video index table [(index, vdir)]
     - `vdir` values serve the role of video names. For each `vdir` value a
       separate group will be created in the HDF5 file, containing the frame
       data for the corresponding video.

  Per video group (<vdir>/):
   * `index` : Frame index table [(index, fname_data, fname_labels), ...]
     - `frame_data` is an unused dummy parameter.
     - `frame_labels` specifies file name of the npz file with labels for the
        current frame.
   * `data_<name>` : Frame data arrays.
      Multiple arrays can be stored.
      These arrays are loaded by 'load_h5frame_data' based on the
      `data_dtype_list` argument.
      Only arrays whose `<name>` matches the `data` field in `data_dtype_list`
      entries are loaded.

- labels_XXXXX.npz: Contains annotation data. For the label format
  specification refer to funcs_npz.py
"""
import os
import torch

from .jagged_array_specs import IJaggedArraySpecs, ElemSpec
from .funcs_npz          import (load_frame_labels)

FNAME_FRAMES = 'frames.h5'
STR_ENC      = 'utf-8'

# VIDEO_INDEX  : (index, vdir)
KEY_VIDEOS_INDEX_TABLE = 'index'
# FRAMES_INDEX : (index, fname_data, fname_labels)
KEY_FRAMES_INDEX_TABLE = 'index'
KEY_FRAMES_DATA_TABLE  = 'data'

class HDF5JaggedArraySpecs(IJaggedArraySpecs):

    def __init__(self, f):
        self._f = f
        self._arr_index = f[KEY_VIDEOS_INDEX_TABLE][:]

        self._elem_index_dict = {
            name : self._f[name][KEY_FRAMES_INDEX_TABLE][:]
                for name in (
                    self.get_array_name(idx)
                        for idx in range(len(self._arr_index))
                )
        }

    def __len__(self):
        return sum(
            len(self._f[group.decode(STR_ENC)][KEY_FRAMES_INDEX_TABLE])
                for group in self._arr_index['vdir']
        )

    def get_n_arrays(self):
        return len(self._arr_index)

    def get_array_name(self, arr_idx):
        return self._arr_index['vdir'][arr_idx].decode(STR_ENC)

    def get_array_length(self, arr_idx):
        elem_index = self._elem_index_dict[self.get_array_name(arr_idx)]
        return len(elem_index)

    def get_elem_spec(self, arr_idx, elem_idx):
        elem_index = self._elem_index_dict[self.get_array_name(arr_idx)]
        elem       = elem_index[elem_idx]

        return ElemSpec(
            elem[0],
            elem[1].decode(STR_ENC),
            elem[2].decode(STR_ENC) or None,
        )

    def has_labels(self, arr_idx, elem_idx):
        elem_index   = self._elem_index_dict[self.get_array_name(arr_idx)]
        fname_labels = elem_index['fname_labels'][elem_idx]

        return (len(fname_labels) > 0)

def extract_and_format(container, idx, dtype):
    result = container[idx]

    if dtype is None:
        return result

    return result.astype(dtype)

def load_h5frame_data(f, vdir, elem_idx, data_dtype_list):
    result = []

    for (name, dtype) in data_dtype_list:
        table_data = f[vdir][KEY_FRAMES_DATA_TABLE + '_' + name]
        data       = extract_and_format(table_data, elem_idx, dtype)

        result.append(data)

    return [ torch.from_numpy(d) for d in result ]

def load_h5frame(
    f, root, vdir, elem_idx, fname_labels, data_dtype_list, label_dtype_list,
    bbox_fmt, canvas_size
):
    # pylint: disable=too-many-arguments
    video_root = os.path.join(root, vdir)

    data   = load_h5frame_data(f, vdir, elem_idx, data_dtype_list)
    labels = load_frame_labels(
        video_root, fname_labels, label_dtype_list, bbox_fmt, canvas_size
    )

    return (data, labels)

