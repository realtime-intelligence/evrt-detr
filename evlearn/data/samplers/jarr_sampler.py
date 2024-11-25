"""
PyTorch sampler for sampling jagged arrays (e.g. videos) in parallel while
maintaining temporal order.

For a dataset of N sequences and batch size B:
- Creates B parallel sampling streams
- Each batch position delivers frames sequentially from its assigned sequence
- Each new batch continues exactly where previous batch left off
- When a sequence ends, next sequence can begin in that batch position

Example with 3 sequences [A(5 frames), B(3), C(4)] and batch_size=2:
Batch 1: A[0], B[0]
Batch 2: A[1], B[1]
Batch 3: A[2], B[2]
Batch 4: A[3], C[0]  # B ended, C starts
Batch 5: A[4], C[1]
Batch 6: None, C[2]  # A ended
Batch 7: None, C[3]

Options:
- Shuffle sequence and/or frame order
- Handle uneven sequence lengths via padding or dropping
- Split sequences across batch slots evenly or by jagged array starts
"""

import random

from torch.utils.data.sampler import Sampler
from .funcs import (
    split_indices_by_array_start,
    split_indices_equally,
    drop_last_batches,
    calculate_sampler_length
)

def generate_samples(
    array_specs, shuffle_arrays, shuffle_elements, skip_unlabeled, prg
):
    # Flatten and shuffle a jagged array spec `array_specs`
    result = []

    array_indices = list(range(array_specs.get_n_arrays()))

    if shuffle_arrays:
        prg.shuffle(array_indices)

    array_start_indices = []

    for arr_idx in array_indices:
        n_curr_elems = len(result)
        array_start_indices.append(n_curr_elems)

        element_indices = list(range(array_specs.get_array_length(arr_idx)))

        if shuffle_elements:
            prg.shuffle(element_indices)

        if skip_unlabeled:
            element_indices = [
                elem_idx for elem_idx in element_indices
                    if array_specs.has_labels(arr_idx, elem_idx)
            ]

        result += [
            (arr_idx, elem_idx) for elem_idx in element_indices
        ]

    return result, array_start_indices

class JArrSamplerIt:

    def __init__(
        self,
        array_specs, batch_size,
        shuffle_arrays        = False,
        shuffle_elems         = False,
        skip_unlabeled        = False,
        split_by_array_starts = True,
        drop_last             = False,
        pad_empty             = True,
        seed                  = 0
    ):
        # pylint: disable=too-many-arguments
        self._prg        = random.Random(seed)
        self._batch_size = batch_size
        self._pad_empty  = pad_empty

        # samples             : [ (array_idx, elem_idx) ]
        # array_start_indices : [ flat_idx, ]
        self._flat_to_nested_index, array_start_indices = generate_samples(
            array_specs, shuffle_arrays, shuffle_elems, skip_unlabeled,
            self._prg
        )

        n_samples = len(self._flat_to_nested_index)

        # _batch_positions and _batch_end_positions control sampling for each
        # batch slot
        # Example for batch_size == 3
        #   _batch_positions     = [0, 3, 6]  # starting indices for each slot
        #   _batch_end_positions = [3, 6, 8]  # where each slot should stop
        #
        # The following indices will be sampled for each batch
        # batch #1: [ 0, 3, 6 ]
        # batch #2: [ 1, 4, 7 ]
        # batch #3: [ 2, 5, ]
        # batch #4: None

        if split_by_array_starts:
            self._batch_positions, self._batch_end_positions \
                = split_indices_by_array_start(
                    n_samples, array_start_indices, batch_size
                )
        else:
            self._batch_positions, self._batch_end_positions \
                = split_indices_equally(n_samples, batch_size)

        if drop_last:
            self._batch_end_positions = drop_last_batches(
                self._batch_positions, self._batch_end_positions
            )

        self._length = calculate_sampler_length(
            self._batch_positions, self._batch_end_positions
        )

    def __iter__(self):
        return self

    def __len__(self):
        return self._length

    def __next__(self):
        result = []

        for bi in range(self._batch_size):
            sample_index = self._batch_positions[bi]

            if sample_index >= self._batch_end_positions[bi]:
                if self._pad_empty:
                    result.append(None)
                continue

            arr_idx, elem_idx = self._flat_to_nested_index[sample_index]
            result.append((arr_idx, elem_idx))

            self._batch_positions[bi] += 1

        if (len(result) == 0) or all(x is None for x in result):
            raise StopIteration

        return result

class JArrSampler(Sampler):
    # pylint: disable=too-many-instance-attributes

    def __init__(
        self, dataset, batch_size,
        shuffle_arrays        = False,
        shuffle_elems         = False,
        skip_unlabeled        = False,
        split_by_array_starts = True,
        drop_last             = False,
        pad_empty             = True,
        seed                  = 0,
    ):
        # pylint: disable=too-many-arguments
        super().__init__(dataset)

        self._dset        = dataset
        self._array_specs = dataset.array_specs
        self._batch_size  = batch_size
        self._drop_last   = drop_last
        self._pad_empty   = pad_empty
        self._seed        = seed

        self._shuffle_arrays = shuffle_arrays
        self._shuffle_elems  = shuffle_elems
        self._skip_unlabeled = skip_unlabeled
        self._split_by_array_starts = split_by_array_starts

        self._cycles  = 0
        self._reinit_iter()

    def _reinit_iter(self):
        self._dset.set_seed(self._seed + self._cycles)

        self._curr_it = JArrSamplerIt(
            self._array_specs, self._batch_size,
            self._shuffle_arrays,
            self._shuffle_elems,
            self._skip_unlabeled,
            self._split_by_array_starts,
            self._drop_last,
            self._pad_empty,
            self._seed + self._cycles,
        )

    def __len__(self):
        return len(self._curr_it)

    def __iter__(self):
        result = self._curr_it

        self._cycles += 1
        self._reinit_iter()

        return result

class VideoSampler(JArrSampler):
    # Simple facade over JArrSampler

    def __init__(
        self, dataset, batch_size,
        shuffle_videos        = False,
        shuffle_frames        = False,
        skip_unlabeled        = False,
        split_by_video_starts = True,
        drop_last             = False,
        pad_empty             = True,
        seed                  = 0,
    ):
        # pylint: disable=too-many-arguments
        super().__init__(
            dataset, batch_size,
            shuffle_videos, shuffle_frames, skip_unlabeled,
            split_by_video_starts, drop_last, pad_empty, seed
        )

