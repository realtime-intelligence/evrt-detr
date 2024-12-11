"""Convert PSEE event camera recordings into a standardized NPZ frame format.

Input format:
    Source dataset should be organized as:
        root/
            split1/                 # e.g. train/val/test
                video1_td.dat       # Raw event data
                video1_bbox.npy     # Optional bounding box annotations
                video2_td.dat
                video2_bbox.npy
            split2/
                ...

Output format:
    Destination dataset will be organized as:
        outdir/
            config.json            # Processing parameters
            split1/
                video1/
                    data_00000.npz    # Event frame data
                        frame: (2, n_bins, height, width) uint16
                        time: int
                    data_00001.npz
                    labels_00000.npz   # Optional annotation data
                    labels_00001.npz
                video2/
                    ...
            split2/
                ...

Frame generation:
    - Event stream is partitioned into fixed-time windows of specified duration
    - Each frame accumulates events over the time window
    - Each frame is subdivided into n_bins temporal bins
    - Events are split by polarity (positive/negative)
    - Frames are stored as (2, n_bins, height, width) uint16
    - Optional annotation data is preserved

Mangling options (dataset-specific preprocessing):
    rvt-gen1 : 240x304 frame size
        - Crop boxes to FOV
        - Filter small boxes (diag >= 30, side >= 10)
        - Filter large boxes in train split (width <= 0.9*frame_width)
    rvt-gen4 : 720x1280 frame size
        - Crop boxes to FOV
        - Filter small boxes (diag >= 5, side >= 5)
        - Filter large boxes in train split (width <= 0.9*frame_width)
        - Keep only classes [0,1,2] (pedestrian, two wheeler, car)
"""

import argparse
from collections import namedtuple
import json
import os
import logging
import multiprocessing
import warnings

import numpy as np
import tqdm

from psee_adt.io.psee_loader import PSEELoader
from psee_adt.io.box_loading import reformat_boxes

warnings.simplefilter('once', UserWarning)

FrameConfig = namedtuple(
    'FrameConfig', [ 'duration', 'n_bins', 'mangling' ]
)

SUFFIX_VIDEO = '_td.dat'
SUFFIX_LABEL = '_bbox.npy'
DTYPE_HIST   = np.uint16
DTYPE_COORD  = np.float32

R_MICROSECOND = 1
R_MILISECOND  = R_MICROSECOND * 1000
R_SECOND      = R_MILISECOND * 1000

MANGLE_FOV_CROP       = 'fov-crop'
MANGLE_MINBOX_FILTER  = 'minbox-filter'
MANGLE_MAXBOX_FILTER  = 'maxbox-filter'
MANGLE_CLASSID_FILTER = 'classid-filter'

MANGLING = {
    'rvt-gen1' : {
        MANGLE_FOV_CROP       : 'rvt-gen1',
        MANGLE_MINBOX_FILTER  : 'rvt-gen1',
        MANGLE_MAXBOX_FILTER  : 'rvt-gen1',
        MANGLE_CLASSID_FILTER : 'rvt-gen1',
    },
    'rvt-gen4' : {
        MANGLE_FOV_CROP       : 'rvt-gen4',
        MANGLE_MINBOX_FILTER  : 'rvt-gen4',
        MANGLE_MAXBOX_FILTER  : 'rvt-gen4',
        MANGLE_CLASSID_FILTER : 'rvt-gen4',
    },
}

FRAME_SIZES = {
    # source: rvt/scripts/genx/preprocess_dataset.py
    'rvt-gen1' : {
        'height' : 240,
        'width'  : 304,
    },
    'rvt-gen4' : {
        'height' : 720,
        'width'  : 1280,
    },
}

FILTERS_MINBOX = {
    # source: psee_evaluator.py
    'gen1' : {
        'min_diag' : 30,
        'min_side' : 20,
    },
    # source: rvt/scripts/genx/preprocess_dataset.py
    'rvt-gen1' : {
        'min_diag' : 30,
        'min_side' : 10,
    },
    # source: rvt/scripts/genx/preprocess_dataset.py
    'rvt-gen4' : {
        'min_diag' : 5,
        'min_side' : 5,
    },
    'gen4' : {
        'min_diag' : 60,
        'min_side' : 20,
    },
}

FILTERS_MAXBOX = {
    # source: rvt/scripts/genx/preprocess_dataset.py
    'rvt-gen1' : {
        'width' : (9 * FRAME_SIZES['rvt-gen1']['width']) // 10,
    },
    'rvt-gen4' : {
        'width' : (9 * FRAME_SIZES['rvt-gen4']['width']) // 10,
    },
}

FILTERS_LABEL = {
    'rvt-gen1' : None,
    'rvt-gen4' : [ 0, 1, 2 ], # keep { pedestrian, two wheeler, car }
}

class FrameProcessor:

    def __init__(self, root_src, root_dst, split, frame_config):
        self._root_src = root_src
        self._root_dst = root_dst
        self._split    = split

        self._frame_config = frame_config

    def __call__(self, index_path):
        (video_index, fname) = index_path
        path = os.path.join(self._root_src, fname)

        process_video(
            path, self._root_dst, video_index, self._split,
            self._frame_config
        )

def parse_cmdargs():
    parser = argparse.ArgumentParser(
        description = 'Convert psee dataset into the standard npz frame format'
    )

    parser.add_argument(
        'source',
        metavar  = 'SOURCE',
        help     = 'path to psee dataset',
        type     = str,
    )

    parser.add_argument(
        'target',
        metavar  = 'DESTINATION',
        help     = 'directory to save converted dataset',
        type     = str,
    )

    parser.add_argument(
        '-d', '--duration',
        dest     = 'duration',
        default  = None,
        help     = 'frame duration',
        type     = str,
        required = True,
    )

    parser.add_argument(
        '-n', '--nbins',
        dest     = 'n_bins',
        default  = 10,
        help     = 'number of time bins per frame',
        type     = int,
    )

    parser.add_argument(
        '--workers',
        dest     = 'workers',
        default  = 1,
        help     = 'number of parallel workers to use',
        type     = int,
    )

    parser.add_argument(
        '--mangling',
        choices  = list(MANGLING),
        dest     = 'mangling',
        default  = None,
        help     = 'add data mangling',
        type     = str,
    )

    return parser.parse_args()

def parse_duration(duration):
    if duration.endswith('ms'):
        return int(duration[:-2]) * R_MILISECOND

    if duration.endswith('us'):
        return int(duration[:-2]) * R_MICROSECOND

    if duration.endswith('s'):
        return int(duration[:-1]) * R_SECOND

    raise ValueError(f"Failed to parse duration: '{duration}'")

def parse_frame_config(duration, n_bins, mangling):
    duration = parse_duration(duration)

    return FrameConfig(
        duration = duration, n_bins = n_bins, mangling = mangling,
    )

def load_video_label(path_video):
    assert path_video.endswith(SUFFIX_VIDEO)

    path_label = path_video.rstrip(SUFFIX_VIDEO) + SUFFIX_LABEL

    video    = PSEELoader(path_video)
    labels   = np.load(path_label)
    basename = os.path.basename(path_video).rstrip(SUFFIX_VIDEO)

    return (video, labels, basename)

def apply_psee_min_box_filter(labels, filter_name):
    min_diag = FILTERS_MINBOX[filter_name]['min_diag']
    min_side = FILTERS_MINBOX[filter_name]['min_side']

    # pylint: disable=import-outside-toplevel
    from psee_adt.io.box_filtering import filter_boxes

    return filter_boxes(
        labels, skip_ts = -1, min_box_diag = min_diag, min_box_side = min_side
    )

def apply_rvt_max_box_filter(labels, split, filter_name):
    if split != 'train':
        return labels

    width    = FILTERS_MAXBOX[filter_name]['width']
    ok_width = (labels['w'] <= width)

    return labels[ok_width]

def crop_to_fov(labels, filter_name):
    height = FRAME_SIZES[filter_name]['height']
    width  = FRAME_SIZES[filter_name]['width']

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

def filter_class_id(labels, filter_name):
    acceptable_classes = FILTERS_LABEL[filter_name]

    if acceptable_classes is None:
        return labels

    class_id = labels['class_id']
    mask     = np.isin(class_id, list(acceptable_classes))

    return labels[mask]

def mangle_labels(labels, split, mangling):
    if mangling is None:
        return labels

    mangle_dict = MANGLING[mangling]

    for (name, spec_name) in mangle_dict.items():
        if name == MANGLE_FOV_CROP:
            labels = crop_to_fov(labels, spec_name)
        elif name == MANGLE_MINBOX_FILTER:
            labels = apply_psee_min_box_filter(labels, spec_name)
        elif name == MANGLE_MAXBOX_FILTER:
            labels = apply_rvt_max_box_filter(labels, split, spec_name)
        elif name == MANGLE_CLASSID_FILTER:
            labels = filter_class_id(labels, spec_name)
        else:
            raise ValueError(f"Unknown mangling: {name}")

    return labels

def has_multiple_annotations_per_frame(frame_labels):
    # NOTE:
    #   This script constructs frames by accumulating data over a fixed-time
    #   window. Likewise, it merges all annotations from a fixed-time window.
    #
    #   If several independent annotations fall into a single time-window,
    #   that will result in duplicated annotations for a single frame.
    #
    #   This function checks that there are no duplicates.
    #
    if len(frame_labels) < 2:
        return False

    times = np.unique(frame_labels['t'])

    if len(times) > 1:
        warnings.warn(
            f'Found multiple annotated samples in a single frame: {len(times)}'
        )
        return True

    return False

def cherry_pick_label_timestamps(frame_labels):
    # Cherry pick annotations from the middle of the frame
    times, counts = np.unique(frame_labels['t'], return_counts = True)
    max_counts    = np.max(counts)
    median_time   = np.median(times)

    times_with_max_counts = times[counts == max_counts]

    nearest_to_median_idx = np.argmin(
        np.abs(times_with_max_counts - median_time)
    )
    cherry_picked_time = times[nearest_to_median_idx]

    time_mask = (frame_labels['t'] == cherry_picked_time)

    return frame_labels[time_mask]

# NOTE: Frame generation algorithm from a stream of events:
#  - find lowest/highest timestamps
#  - sample frames at `duration` frequency
#    * if labels is None, save events
#    * if events is None, save labels and an empty frame

def get_frames(video, labels, duration):
    events = video.load_n_events(ev_count = video.event_count())

    time_start = min(events['t'].min(), labels['t'].min())
    time_end   = max(events['t'].max(), labels['t'].max()) + 1

    frame_start_time = (time_start // duration) * duration

    while frame_start_time < time_end:
        frame_end_time = frame_start_time + duration

        mask_events = (
              (events['t'] >= frame_start_time)
            & (events['t']  < frame_end_time)
        )

        mask_labels = (
              (labels['t'] >= frame_start_time)
            & (labels['t']  < frame_end_time)
        )

        frame_events = events[mask_events]
        frame_labels = labels[mask_labels]

        frame_time       = frame_start_time
        frame_start_time = frame_end_time

        yield (frame_events, frame_labels, frame_time)

def scatter_events_into_frame(frame, events):
    # pylint: disable=too-many-locals
    _, nbins, H, W = frame.shape

    # +1 so that bins can be counted as low_edge <= x < high_edge
    time_min = np.min(events['t'])
    time_max = np.max(events['t']) + 1

    mask_pos = (events['p'] > 0)
    mask_neg = ~mask_pos

    curr_time = time_min
    time_step = (time_max - time_min) / nbins

    y_valid = (events['y'] >= 0) & (events['y'] < H)
    x_valid = (events['x'] >= 0) & (events['x'] < W)
    valid   = x_valid & y_valid

    if np.any(~valid):
        warnings.warn(
            f'Found values outsize of image range: {H}x{W}: {events[~valid]}'
        )

    for bin_idx in range(nbins):
        next_time = curr_time + time_step

        if bin_idx == nbins-1:
            next_time = time_max

        mask_bin = (events['t'] >= curr_time) & (events['t'] < next_time)

        x_pos = events[mask_bin & mask_pos & valid]['x']
        y_pos = events[mask_bin & mask_pos & valid]['y']

        x_neg = events[mask_bin & mask_neg & valid]['x']
        y_neg = events[mask_bin & mask_neg & valid]['y']

        np.add.at(frame, (0, bin_idx, y_pos, x_pos), 1)
        np.add.at(frame, (1, bin_idx, y_neg, x_neg), 1)

        curr_time = next_time

def events_to_hist(events, image_size, nbins):
    # pylint: disable=too-many-locals
    (H, W) = image_size

    # result : (polarity, bin, height, width)
    result = np.zeros((2, nbins, H, W), dtype = DTYPE_HIST)

    if len(events) == 0:
        return result

    scatter_events_into_frame(result, events)

    return result

def labels_to_dict(labels, image_index):
    # convert labels into torchvision format
    # https://pytorch.org/tutorials/intermediate/torchvision_tutorial.html
    result = { }

    n = len(labels)

    if n == 0:
        return None

    x0 = labels['x']
    y0 = labels['y']
    x1 = x0 + labels['w']
    y1 = y0 + labels['h']

    boxes    = np.stack((x0, y0, x1, y1), axis = 1, dtype = DTYPE_COORD)
    areas    = (labels['w'] * labels['h']).astype(DTYPE_COORD)
    image_id = np.full((n,), fill_value = image_index, dtype = np.uint64)
    class_id = labels['class_id']
    iscrowd  = np.zeros((n, ), dtype = np.uint8)

    result = {
        'boxes'       : boxes,
        'labels'      : class_id,
        'image_id'    : image_id,
        'area'        : areas,
        'iscrowd'     : iscrowd,
        'time'        : labels['t'],
        'psee_labels' : labels
    }

    return result

def save_frame(root, event_hist, labels_dict, frame_time, frame_idx):
    path_data  = os.path.join(root, f'data_{frame_idx:05d}.npz')
    np.savez_compressed(path_data, frame = event_hist, time = frame_time)

    if labels_dict:
        path_labels = os.path.join(root, f'labels_{frame_idx:05d}.npz')
        np.savez_compressed(path_labels, **labels_dict)

def process_single_frame(
    frame_data, video_index, frame_idx, frame_size, frame_config
):
    events, labels, frame_time = frame_data
    # Pack video_index and frame_idx into a single 64-bit ID
    # where high 32 bits = video_index, low 32 bits = frame_idx
    image_index = (video_index << 32) + frame_idx
    event_hist  = events_to_hist(events, frame_size, frame_config.n_bins)

    if has_multiple_annotations_per_frame(labels):
        orig_labels = labels
        labels      = cherry_pick_label_timestamps(labels)
    else:
        orig_labels = None

    labels_dict = labels_to_dict(labels, image_index)

    if orig_labels is not None:
        labels_dict['psee_labels_orig'] = orig_labels

    return event_hist, labels_dict, frame_time

def process_video(path, outdir_root, video_index, split, frame_config):
    # pylint: disable=too-many-locals
    video, labels, basename = load_video_label(path)

    # NOTE: psee_adt is full of backward incompatibility.
    #       if reformat_boxes is not done -- evaluation will break later
    labels = reformat_boxes(labels)
    if len(labels) == 0:
        logging.warning("No labels in: '%s'", path)
        return

    labels = mangle_labels(labels, split, frame_config.mangling)
    if len(labels) == 0:
        logging.warning("No labels left after mangling in: '%s'", path)
        return

    frame_size = video.get_size()

    outdir = os.path.join(outdir_root, basename)
    os.makedirs(outdir, exist_ok = True)

    frame_it = get_frames(video, labels, frame_config.duration)

    for (frame_idx, frame_data) in enumerate(frame_it):
        event_hist, labels_dict, frame_time = process_single_frame(
            frame_data, video_index, frame_idx, frame_size, frame_config
        )

        save_frame(outdir, event_hist, labels_dict, frame_time, frame_idx)

def collect_videos(root):
    result = [ x for x in os.listdir(root) if x.endswith(SUFFIX_VIDEO) ]
    result.sort()

    return result

def process_dataset_split(root, outdir_root, split, frame_config, workers):
    root_src = os.path.join(root,        split)
    root_dst = os.path.join(outdir_root, split)

    videos = collect_videos(root_src)
    worker = FrameProcessor(root_src, root_dst, split, frame_config)

    pbar = tqdm.tqdm(
        desc  = f"Processing '{split}'",
        total = len(videos),
        dynamic_ncols = True
    )

    with multiprocessing.Pool(processes = workers) as pool:
        for _ in pool.imap_unordered(worker, enumerate(videos)):
            pbar.update()

    pbar.close()

def process_dataset(root, outdir_root, frame_config, workers):
    splits = sorted(os.listdir(root))
    logging.info("Found datset splits: '%s'", splits)

    for split in splits:
        process_dataset_split(root, outdir_root, split, frame_config, workers)

def save_metadata(outdir_root, frame_config):
    os.makedirs(outdir_root, exist_ok = True)

    path = os.path.join(outdir_root, 'config.json')
    conf = {
        'duration'     : frame_config.duration,
        'n_bins'       : frame_config.n_bins,
        'mangling'     : frame_config.mangling,
    }

    with open(path, 'wt', encoding = 'utf-8') as f:
        json.dump(conf, f, indent = 4)

def main():
    cmdargs = parse_cmdargs()
    logging.basicConfig(
        level  = logging.INFO,
        format = '[%(asctime)s] %(levelname)s %(message)s'
    )

    if os.path.exists(cmdargs.target):
        raise RuntimeError(
            f"Target directory exists: '{cmdargs.target}'"
            "\nRefusing to overwrite."
        )

    frame_config = parse_frame_config(
        cmdargs.duration, cmdargs.n_bins, cmdargs.mangling,
    )

    logging.info("Processing dataset: '%s'", cmdargs.source)

    save_metadata(cmdargs.target, frame_config)

    process_dataset(
        cmdargs.source, cmdargs.target, frame_config, cmdargs.workers
    )

if __name__ == '__main__':
    main()

