"""
Utility functions for handling event camera data stored in NumPy (.npz) format.

This module provides functions for loading and managing event camera data with
frame data and labels stored in separate .npz files. The data must be organized
as:
    +-- video1/
          +-- data_00001.npz
          +-- labels_0001.npz
          +-- data_0002.npz
          +-- labels_0002.npz
          +-- ...

where
- data_XXXXX.npz: Contains event data as named numpy arrays.

  These arrays can be loaded by `load_frame_data` function.
  This function loads only arrays specified in the `data_dtype_list` argument,
  and optionally converts their dtype.

- labels_XXXXX.npz: Contains annotation data as numpy arrays
  Annotations can be arbitrary data.
  The label loading is performed by `load_frame_labels`, which loads only
  arrays specified in `label_dtype_list` list.
  This function recognizes special array names and converts them into the
  corresponding PyTorch/TorchVision objects:
   * 'masks'  - segmentation masks (converted to tv_tensors.Mask)
   * 'boxes'  - bounding boxes     (converted to tv_tensors.BoundingBoxes)
   * 'labels' - class labels       (converted to torch.Tensor)
   * 'time'   - event time         (converted to torch.Tensor)

"""
import os
import re

import numpy as np
import torch
from torchvision import tv_tensors

from .jagged_array_specs import ElemSpec, JArrSpec

RE_DATA   = re.compile(r'^data_(\d+)\.npz$')
RE_LABELS = re.compile(r'^labels_(\d+)\.npz$')

def select_files(files, regexp):
    result = {}

    for fname in files:
        m = regexp.match(fname)
        if not m:
            continue

        index = int(m.group(1))
        if index in result:
            raise RuntimeError(f"Duplicated index '{index}' found.")

        result[index] = fname

    return result

def extract_and_format(container, key, dtype):
    if dtype is None:
        return container[key]

    return container[key].astype(dtype)

def collect_frames(root, skip_unlabeled):
    files = os.listdir(root)

    image_dict = select_files(files, RE_DATA)
    label_dict = select_files(files, RE_LABELS)

    if skip_unlabeled:
        result = sorted(
            ElemSpec(idx, fname, label_dict[idx])
                for (idx, fname) in image_dict.items()
                    if idx in label_dict
        )
        if not result:
            result = None

    else:
        result = sorted(
            ElemSpec(idx, fname, label_dict.get(idx, None))
                for (idx, fname) in image_dict.items()
        )

    return result

def collect_videos(root, skip_unlabeled):
    subdirs = os.listdir(root)
    subdirs.sort()

    result = [
        JArrSpec(
            vdir, collect_frames(os.path.join(root, vdir), skip_unlabeled)
        ) for vdir in subdirs
    ]

    if skip_unlabeled:
        result = [
            JArrSpec(vdir, frames) for (vdir, frames) in result
                if frames is not None
        ]

    return result

def load_frame_labels(
    video_root, fname_labels, label_dtype_list, bbox_fmt, canvas_size
):
    if fname_labels is None:
        return None

    with np.load(os.path.join(video_root, fname_labels)) as f:
        labels = {
            l : extract_and_format(f, l, dtype)
                for (l, dtype) in label_dtype_list
        }

    if 'boxes' in labels:
        labels['boxes'] = tv_tensors.BoundingBoxes(
            labels['boxes'],
            format      = bbox_fmt,
            canvas_size = canvas_size,
        )

    if 'masks' in labels:
        labels['masks'] = tv_tensors.Mask(labels['masks'])

    if 'labels' in labels:
        labels['labels'] = torch.from_numpy(labels['labels'])

    if 'time' in labels:
        labels['time'] = torch.from_numpy(labels['time'])

    return labels

def load_frame_data(video_root, fname_image, data_dtype_list):
    with np.load(os.path.join(video_root, fname_image)) as f:
        data = [
            extract_and_format(f, d, dtype) for (d, dtype) in data_dtype_list
        ]

    return [ torch.from_numpy(d) for d in data ]

def load_frame(
    root, vdir, fname_image, fname_labels, data_dtype_list, label_dtype_list,
    bbox_fmt, canvas_size
):
    # pylint: disable=too-many-arguments
    video_root = os.path.join(root, vdir)

    data   = load_frame_data(video_root, fname_image, data_dtype_list)
    labels = load_frame_labels(
        video_root, fname_labels, label_dtype_list, bbox_fmt, canvas_size
    )

    return (data, labels)

