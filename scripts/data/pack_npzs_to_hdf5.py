"""Convert a dataset of NPZ files into an HDF5 format for efficient access.

Input format (dataset constructed with `psee_to_frames.py`):
    Source dataset should be organized as:
        root/
            video1/
                data_00000.npz    # Event frame data
                data_00001.npz
                labels_00000.npz  # Optional annotation data
                labels_00001.npz
            video2/
                ...

Output HDF5 structure:
    outdir/
        frames.h5:
            video1/                       # Group per video
                index                     # Table with frame info
                    columns: (index, fname_data, fname_labels)
                data_{datakey}            # Table per each key in source NPZ
                    shape: (n_frames, *data_shape)
            video2/
                ...
            index                         # Table mapping video IDs to names
                columns: (index, vdir)
        video1/
            labels_00000.npz  # Optional annotation data
            labels_00001.npz
        video2/
            ...

Data transformation options:
    --dtype name:dtype[,name:dtype,...]
        Convert specific data arrays to different dtypes.
        Example: --dtype "frame:uint8,time:float32"
        Supported: uint8/16/32, int8/16/32, float16/32

    --clamp-min/--clamp-max name:value[,name:value,...]
        Clamp array values to specified min/max before type conversion.
        Example: --clamp-min "frame:0" --clamp-max "frame:255"
        Useful when converting to smaller dtypes to avoid overflow.
"""
import argparse
from collections import namedtuple
import json
import shutil
import os
import multiprocessing

import numpy as np
import tqdm
import h5py

from evlearn.data.datasets.funcs_npz import RE_DATA, RE_LABELS
from evlearn.data.datasets.funcs_hdf import (
    FNAME_FRAMES,
    # HDF5 tables names
    KEY_VIDEOS_INDEX_TABLE,
    KEY_FRAMES_INDEX_TABLE,
    KEY_FRAMES_DATA_TABLE,
    STR_ENC
)

HDF5_PLUGIN_COMPRESSION = { 'blosc', 'blosc2', }

Config = namedtuple('Config', [
    'clamp_map', 'dtype_map', 'compression',
    'filters', 'chunking', 'require_plugin'
])

Frame = namedtuple('Frame', [ 'index', 'fname_data', 'fname_labels' ])
Video = namedtuple('Video', [ 'vdir',  'frames' ])

NP_DTYPES = {
    'uint8'   : np.uint8,
    'uint16'  : np.uint16,
    'uint32'  : np.uint32,
    'int8'    : np.int8,
    'int16'   : np.int16,
    'int32'   : np.int32,
    'float16' : np.float16,
    'float32' : np.float32,
}

class VideoPacker:

    def __init__(self, srcdir, outdir, videos, config):
        self._video_root = srcdir
        self._outdir     = outdir
        self._videos     = videos
        self._config     = config
        self._filters    = construct_hdf_filters(config)

    def __call__(self, vidx):
        video  = self._videos[vidx]
        frames = video.frames

        output     = os.path.join(self._outdir,     video.vdir + '.h5')
        video_root = os.path.join(self._video_root, video.vdir)

        pbar = tqdm.tqdm(
            total         = len(video.frames),
            desc          = f'Packing video {vidx} / {len(self._videos)}',
            # Using process _identity to get unique positions for tqdm progress
            # bars in multiprocessing.
            # Looks hacky but not sure if there is a better way
            position      = multiprocessing.current_process()._identity[0],
            leave         = False,
            dynamic_ncols = True,
        )

        with h5py.File(output, 'w') as f:
            grp = f.create_group(video.vdir)
            save_frame_index_table(grp, frames)

            tables = init_frame_data_tables(
                grp, video_root, frames, self._config, self._filters
            )

            save_frame_data(
                video_root, frames, tables, pbar,
                self._config.clamp_map, self._config.dtype_map
            )

        return output

def parse_cmdargs():
    parser = argparse.ArgumentParser("Pack npz data into an h5 file")

    parser.add_argument(
        'source',
        help    = 'path to the source npz dataset',
        metavar = 'SOURCE',
        type    = str,
    )

    parser.add_argument(
        'dest',
        help    = 'path to save h5 dataset to',
        metavar = 'DESTINATION',
        type    = str,
    )

    parser.add_argument(
        '--compression',
        dest    = 'compression',
        default = None,
        help    = 'compressor',
        type    = str,
    )

    parser.add_argument(
        '--filters',
        dest    = 'filters',
        default = None,
        help    = (
            'a json string of hdf5 filters in form "key:value".'
            ' Example: "{compression_level=9}"'
        ),
        type    = str,
    )

    parser.add_argument(
        '--chunk-size',
        dest    = 'chunking',
        default = 10,
        help    = 'compression chunk size. Use 0 to disable, -1 for automatic',
        type    = int,
    )

    parser.add_argument(
        '-n', '--workers',
        default = None,
        dest    = 'workers',
        help    = 'number of parallel workers to use',
        type    = int,
    )

    parser.add_argument(
        '--dtype',
        default = None,
        dest    = 'dtype_map',
        help    = 'comma separated map name:dtype specifying data dtypes',
        type    = str,
    )

    parser.add_argument(
        '--clamp-max',
        default = None,
        dest    = 'clamp_max',
        help    = 'comma separated map name:clamp_max specifying max clamping',
        type    = str,
    )

    parser.add_argument(
        '--clamp-min',
        default = None,
        dest    = 'clamp_min',
        help    = 'comma separated map name:clamp_min specifying min clamping',
        type    = str,
    )

    return parser.parse_args()

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

def load_frame_data(path):
    with np.load(path) as f:
        data = [ (k, f[k]) for k in f.files ]

    return data

def collect_frames(root):
    files = os.listdir(root)

    image_dict = select_files(files, RE_DATA)
    label_dict = select_files(files, RE_LABELS)

    # NOTE: some frames have no annotations
    result = sorted(
        Frame(idx, fname, label_dict.get(idx, None))
            for (idx, fname) in image_dict.items()
    )

    return result

def collect_videos(root):
    subdirs = os.listdir(root)
    subdirs.sort()

    result = [
        Video(vdir, collect_frames(os.path.join(root, vdir)))
            for vdir in subdirs
    ]

    return result

def save_video_index_table(f, videos):
    dtype = np.dtype([
        ('index', np.int32),
        ('vdir',  h5py.string_dtype(encoding = STR_ENC))
    ])
    video_index = [
        (index, video.vdir) for (index, video) in enumerate(videos)
    ]
    video_index = np.array(video_index, dtype=dtype)
    f.create_dataset(KEY_VIDEOS_INDEX_TABLE, data = video_index)

def save_frame_index_table(f, frames):
    dtype = np.dtype([
        ('index',        np.int32),
        ('fname_data',   h5py.string_dtype(encoding = STR_ENC)),
        ('fname_labels', h5py.string_dtype(encoding = STR_ENC)),
    ])

    frame_index = [
        (index, frame.fname_data, frame.fname_labels or '')
            for (index, frame) in enumerate(frames)
    ]
    frame_index = np.array(frame_index, dtype=dtype)
    f.create_dataset(KEY_FRAMES_INDEX_TABLE, data = frame_index)

def construct_hdf_filters(config):
    result = {}

    if config.require_plugin:
        # pylint: disable=import-outside-toplevel
        import hdf5plugin

        filters = config.filters or {}

        if config.compression == 'blosc2':
            result.update(**hdf5plugin.Blosc2(**filters))
        elif config.compression == 'blosc':
            result.update(**hdf5plugin.Blosc(**filters))
        else:
            raise ValueError(
                f'Compression {config.compression} not implemented'
            )

    else:
        if config.compression is not None:
            result['compression'] = config.compression

        if config.filters is not None:
            result.update(**config.filters)

    return result

def create_table_for_data(f, name, n, sample, config, filters):
    # pylint: disable=too-many-arguments
    chunks = False

    if isinstance(config.chunking, int):
        # -1 == automatic chunking;
        #  0 == disable chunking;
        #  N == chunk size equal to N frames
        if config.chunking > 0:
            chunks = (config.chunking, *sample.shape)
        elif config.chunking == -1:
            chunks = True

    dtype = sample.dtype
    if name in config.dtype_map:
        dtype = NP_DTYPES[config.dtype_map[name]]

    return f.create_dataset(
        name, shape = (n, *sample.shape), dtype = dtype, chunks = chunks,
        **filters
    )

def init_frame_data_tables(f, video_root, frames, config, filters):
    # pylint: disable=too-many-locals
    n      = len(frames)
    result = {}

    if n == 0:
        return {}

    path_frame = os.path.join(video_root, frames[0].fname_data)
    data       = load_frame_data(path_frame)

    for (name, sample) in data:
        table_name = KEY_FRAMES_DATA_TABLE + '_' + name
        table_data = create_table_for_data(
            f, table_name, n, sample, config, filters
        )

        result[name] = table_data

    return result

def save_frame_data(video_root, frames, tables, pbar, clamp_map, dtype_map):
    # pylint: disable=too-many-arguments
    n = len(frames)

    if n == 0:
        return

    for idx, frame in enumerate(frames):
        path_frame = os.path.join(video_root, frame.fname_data)
        data       = load_frame_data(path_frame)

        for (name, sample) in data:
            if name in clamp_map:
                sample = np.clip(sample, *clamp_map[name])

            if name in dtype_map:
                sample = sample.astype(NP_DTYPES[dtype_map[name]])

            tables[name][idx] = sample

        pbar.update()

def pack_videos(root, outdir, videos, config, workers):
    pbar = tqdm.tqdm(
        desc = 'Packing videos', total = len(videos), dynamic_ncols = True
    )

    worker = VideoPacker(root, outdir, videos, config)
    packed_files = []

    with multiprocessing.Pool(processes = workers) as pool:
        for fname in pool.imap_unordered(worker, range(len(videos))):
            packed_files.append(fname)
            pbar.update()

    pbar.close()

    return packed_files

def merge_packed_videos(path_dest, videos, packed_files):
    pbar = tqdm.tqdm(
        desc  = 'Merging packed videos',
        total = len(packed_files),
        dynamic_ncols = True
    )

    with h5py.File(path_dest, 'w') as f:
        save_video_index_table(f, videos)

        for path in packed_files:
            with h5py.File(path, 'r') as fsrc:
                for (name, dataset) in fsrc.items():
                    fsrc.copy(dataset, dest=f, name=name)

            os.remove(path)
            pbar.update()

def copy_labels(root, outdir, videos):
    n = sum(
        sum(1 for frame in video.frames if frame.fname_labels is not None)
        for video in videos
    )

    pbar = tqdm.tqdm(
        desc = 'Copying labels', total = n, dynamic_ncols = True
    )

    for video in videos:
        video_root_src = os.path.join(root,   video.vdir)
        video_root_dst = os.path.join(outdir, video.vdir)
        os.mkdir(video_root_dst)

        labels = (
            frame.fname_labels
                for frame in video.frames if frame.fname_labels is not None
        )

        for label in labels:
            src = os.path.join(video_root_src, label)
            dst = os.path.join(video_root_dst, label)

            shutil.copy(src, dst)
            pbar.update()

def parse_dtype_map(dtype_map):
    result = {}

    if dtype_map is not None:
        for token in dtype_map.split(','):
            k, v = token.split(':', maxsplit = 1)
            result[k] = v

    return result

def parse_clamping(clamp_min, clamp_max):
    result = {}

    result_min = {}
    result_max = {}

    if clamp_min is not None:
        for token in clamp_min.split(','):
            k, v = token.split(':', maxsplit = 1)
            result_min[k] = float(v)

    if clamp_max is not None:
        for token in clamp_max.split(','):
            k, v = token.split(':', maxsplit = 1)
            result_max[k] = float(v)

    names = set(result_min.keys()) | set(result_max.keys())

    for name in names:
        result[name] = (result_min.get(name, None), result_max.get(name, None))

    return result

def create_config(cmdargs):
    filters = None
    if cmdargs.filters is not None:
        filters = json.loads(cmdargs.filters)

    clamp_map = parse_clamping(cmdargs.clamp_min, cmdargs.clamp_max)
    dtype_map = parse_dtype_map(cmdargs.dtype_map)

    require_plugin = cmdargs.compression in HDF5_PLUGIN_COMPRESSION
    if require_plugin:
        # pylint: disable=import-outside-toplevel
        # pylint: disable=unused-import
        import hdf5plugin

    return Config(
        clamp_map      = clamp_map,
        dtype_map      = dtype_map,
        compression    = cmdargs.compression,
        filters        = filters,
        chunking       = cmdargs.chunking,
        require_plugin = require_plugin,
    )

def save_config(dest, config):
    path = os.path.join(dest, 'config.json')
    with open(path, 'wt', encoding = 'utf-8') as f:
        json.dump(config._asdict(), f)

def main():
    cmdargs = parse_cmdargs()
    config  = create_config(cmdargs)

    if os.path.exists(cmdargs.dest):
        raise RuntimeError(
            f"Destination '{cmdargs.dest}' exists. Refusing to overwrite."
        )

    print(f"Collecting all files in a directory: {cmdargs.source}")
    videos = collect_videos(cmdargs.source)

    tmpdir      = cmdargs.dest + '.tmp'
    path_frames = os.path.join(cmdargs.dest, FNAME_FRAMES)

    os.makedirs(cmdargs.dest, exist_ok = False)
    os.mkdir(tmpdir)

    save_config(cmdargs.dest, config)
    copy_labels(cmdargs.source, cmdargs.dest, videos)

    packed_files = pack_videos(
        cmdargs.source, tmpdir, videos, config, workers = cmdargs.workers
    )

    merge_packed_videos(path_frames, videos, packed_files)

    os.rmdir(tmpdir)

if __name__ == '__main__':
    main()

