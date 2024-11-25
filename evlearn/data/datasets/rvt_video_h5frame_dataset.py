"""
Dataset implementation for the RVT (Recurrent Vision Transformer) dataset
format with HDF5 frame storage.

This module provides a concrete implementation of JaggedArrayDataset for
handling RVT-style event camera data. Data frames are stored in HDF5 files with
labels in a separate directory structure, organized as:

   PATH_TO_DATASET/split/
       +-- video1/
       |   +-- event_representations_v2/        # Directory with frame data
       |   |     +-- {ev_repr}/                 # Event representation type
       |   |           +-- event_representations.h5
       |   |           +-- timestamps_us.npy
       |   |           +-- objframe_idx_2_repr_idx.npy
       |   +-- labels_v2/                      # Directory with labels
       |         +-- labels.npz
       |         +-- timestamps_us.npy
       +-- ...

For detailed format specification refer to https://github.com/uzh-rpg/RVT
"""

import os

import torch
from torchvision import tv_tensors

import h5py
import numpy as np

from .jagged_array_dataset import JaggedArrayDataset
from .jagged_array_specs   import SimpleJaggedArraySpecs, ElemSpec, JArrSpec
from .funcs       import cantor_pairing
from .funcs_frame import apply_transforms_to_frame

DIR_LABELS = 'labels_v2'
DIR_FRAMES = 'event_representations_v2'

FNAME_FRAMES     = 'event_representations.h5'
FNAME_FRAMES_ALT = 'event_representations_ds2_nearest.h5'

FNAME_TIMESTAMPS = 'timestamps_us.npy'
FNAME_FRAME2OBJ  = 'objframe_idx_2_repr_idx.npy'
FNAME_LABELS     = 'labels.npz'

TABLE_DATA       = 'data'
KEY_OBJ_TO_LABEL = 'objframe_idx_2_label_idx'
KEY_LABELS       = 'labels'

def get_n_frames(video_root, ev_repr):
    path = os.path.join(video_root, DIR_FRAMES, ev_repr, FNAME_TIMESTAMPS)
    f = np.load(path)
    return len(f)

def load_frame_to_obj_map(video_root, ev_repr, n):
    path = os.path.join(video_root, DIR_FRAMES, ev_repr, FNAME_FRAME2OBJ)
    obj_to_frame = np.load(path)

    result = [ None for idx in range(n) ]

    for (obj_idx, frame_idx) in enumerate(obj_to_frame):
        result[frame_idx] = obj_idx

    return result

def load_obj_to_label_map(video_root):
    path = os.path.join(video_root, DIR_LABELS, FNAME_LABELS)
    f = np.load(path)

    label_boundaries = f[KEY_OBJ_TO_LABEL]

    n_labels  = len(f[KEY_LABELS])

    obj_to_label_number = list(label_boundaries[1:] - label_boundaries[:-1])
    obj_to_label_number.append(n_labels - label_boundaries[-1])

    return list(zip(label_boundaries, obj_to_label_number))

def collect_frames(video_root, ev_repr, skip_unlabeled):
    n = get_n_frames(video_root, ev_repr)
    # List of Optional[obj_indices] corresponding to frames
    frame_to_obj_map = load_frame_to_obj_map(video_root, ev_repr, n)
    # List of (label_start_idx, n_labels) of labels per object
    obj_to_label_map = load_obj_to_label_map(video_root)

    result = []

    for frame_idx, obj_idx in enumerate(frame_to_obj_map):
        labels = None

        if obj_idx is None:
            if skip_unlabeled:
                continue
        else:
            labels = obj_to_label_map[obj_idx]

        result.append(ElemSpec(frame_idx, frame_idx, labels))

    return result

def collect_videos(root, ev_repr, skip_unlabeled):
    subdirs = os.listdir(root)
    subdirs.sort()

    result = []

    for subdir in subdirs:
        video_root = os.path.join(root, subdir)
        frames     = collect_frames(video_root, ev_repr, skip_unlabeled)
        result.append(JArrSpec(subdir, frames))

    return result

def load_rvt_frame_data(
    video_root, ev_repr, frame_idx, load_extra_data, fname_frames
):
    path_frames = os.path.join(video_root, DIR_FRAMES, ev_repr, fname_frames)
    assert not load_extra_data, "Not implemented"

    # pylint: disable=unused-import
    # pylint: disable=import-outside-toplevel
    import hdf5plugin
    f = h5py.File(path_frames, 'r')
    return (f[TABLE_DATA][frame_idx], [])

def clamp_labels(labels, canvas_size, downsampling_factor):
    height, width = canvas_size

    if downsampling_factor is not None:
        height = height * downsampling_factor
        width  = width  * downsampling_factor

    x0 = labels['x']
    y0 = labels['y']

    x1 = x0 + labels['w']
    y1 = y0 + labels['h']

    x0 = np.clip(x0, 0, width-1)
    x1 = np.clip(x1, 0, width-1)
    y0 = np.clip(y0, 0, height-1)
    y1 = np.clip(y1, 0, height-1)

    labels['x'] = x0
    labels['y'] = y0
    labels['w'] = (x1 - x0)
    labels['h'] = (y1 - y0)

    non_degenerate = (labels['w'] > 0) & (labels['h'] > 0)

    return labels[non_degenerate]

def load_rvt_frame_labels(
    video_root, obj_label_spec, load_extra_labels, canvas_size,
    downsampling_factor
):
    if obj_label_spec is None:
        return None

    n_obj_labels = obj_label_spec[1]
    idx_start    = obj_label_spec[0]
    idx_end      = idx_start + n_obj_labels

    path_labels = os.path.join(video_root, DIR_LABELS, FNAME_LABELS)
    f = np.load(path_labels)

    labels     = f[KEY_LABELS]
    obj_labels = labels[idx_start:idx_end]
    obj_labels = clamp_labels(obj_labels, canvas_size, downsampling_factor)

    boxes = np.stack(
        (obj_labels['x'], obj_labels['y'], obj_labels['w'], obj_labels['h']),
        axis = 1,
    )

    if downsampling_factor is not None:
        boxes = boxes / downsampling_factor

    result = {
        'labels' : torch.from_numpy(obj_labels['class_id'].astype(np.int32)),
        'boxes'  : tv_tensors.BoundingBoxes(
            boxes, format = 'XYWH', canvas_size = canvas_size
        ),
        'time'   : torch.from_numpy(obj_labels['t'].astype(np.int64)),
    }

    if load_extra_labels:
        result['psee_labels'] = obj_labels

    return result

def load_rvt_frame(
    root, vdir, frame_idx, frame_label_spec, ev_repr,
    load_extra_data, load_extra_labels, fname_frames, downsampling_factor
):
    # pylint: disable=too-many-arguments
    video_root = os.path.join(root, vdir)

    frame, data = load_rvt_frame_data(
        video_root, ev_repr, frame_idx, load_extra_data, fname_frames
    )

    labels = load_rvt_frame_labels(
        video_root, frame_label_spec, load_extra_labels,
        canvas_size         = frame.shape[-2:],
        downsampling_factor = downsampling_factor
    )

    return (frame, data, labels)

class RVTVideoFrameDataset(JaggedArrayDataset):
    # pylint: disable=too-many-instance-attributes

    def __init__(
        self, path, split,
        ev_repr           = 'stacked_histogram_dt=50_nbins=10',
        skip_unlabeled    = False,
        load_extra_data   = False,
        load_extra_labels = False,
        transform_video   = None,
        transform_frame   = None,
        transform_labels  = None,
        dtype             = 'float32',
        fname_frames      = FNAME_FRAMES,
        return_index      = False,
        downsampling_factor = None,
    ):
        # pylint: disable=too-many-arguments
        self._videos_root = os.path.join(path, split)

        video_specs = collect_videos(
            self._videos_root, ev_repr, skip_unlabeled
        )
        video_specs = SimpleJaggedArraySpecs(video_specs)

        super().__init__(video_specs)

        self._dtype   = dtype
        self._ev_repr = ev_repr
        self._load_extra_data   = load_extra_data
        self._load_extra_labels = load_extra_labels

        self._transform_video  = transform_video
        self._transform_frame  = transform_frame
        self._transform_labels = transform_labels

        self._fname_frames = fname_frames
        self._down_factor  = downsampling_factor
        self._return_index = return_index

    def get_null_elem(self):
        if self._return_index:
            return (None, None, None)
        else:
            return (None, None)

    def get_video_seed(self, arr_idx):
        return cantor_pairing(self._seed, arr_idx)

    def get_elem(self, arr_idx, elem_idx):
        vdir = self._specs.get_array_name(arr_idx)

        (_, frame_idx, frame_label_spec) \
            = self._specs.get_elem_spec(arr_idx, elem_idx)

        image, data, labels = load_rvt_frame(
            self._videos_root, vdir, frame_idx, frame_label_spec,
            self._ev_repr, self._load_extra_data, self._load_extra_labels,
            self._fname_frames, self._down_factor
        )

        image, labels = apply_transforms_to_frame(
            image, labels, self._transform_video, self._transform_frame,
            self._transform_labels, self._dtype,
            squash_time_polarity = False,
            video_seed           = self.get_video_seed(arr_idx)
        )

        if self._return_index:
            data.append(torch.LongTensor([ arr_idx, elem_idx ]))

        return (image, *data, labels)

if __name__ == '__main__':
    import sys

    dset = RVTVideoFrameDataset(
        sys.argv[1], split = 'test',
        skip_unlabeled = False, load_extra_labels = False
    )

    import IPython
    IPython.embed()


