"""
PyTorch sampler for sampling subsequences from jagged arrays (e.g. video clips) 
in parallel while maintaining temporal order.

For a dataset of N sequences and batch size B:
- Creates B parallel sampling streams
- Each batch position delivers fixed-length subsequences sequentially from its
  assigned sequence
- Each new batch continues exactly where previous batch left off
- When a sequence ends, next sequence can begin in that batch position

Example with 3 sequences [A(7 frames), B(5), C(6)], subsequence_length=3, and
batch_size=2:
Batch 1: A[0:3], B[0:3]           # First subsequence from A and B
Batch 2: A[3:6], [*B[3:5],None]   # B has only 2 frames left, padded with None
Batch 3: [A[6],None,None], C[0:3] # A has 1 frame left, padded; C starts
Batch 4: None, C[3:6]             # Only C frames remain

Options:
- Shuffle sequences and/or frame order
- Shuffle generated subsequences
- Handle uneven sequence lengths via padding or dropping
- Split sequences across batch slots evenly or by sequence starts
- Skip subsequences without labels
- Control subsequence length
- Reproducible sampling with seed

Use VideoClipSampler facade for more intuitive parameter names when working
with video data.
"""

import random

from torch.utils.data.sampler import Sampler

from .funcs import (
    drop_last_batches, split_indices_by_array_start, split_indices_equally,
    calculate_sampler_length
)
from .jarr_sampler import generate_samples

def append_sample(array_specs, samples, subsequence, skip_unlabeled):
    if skip_unlabeled:
        has_labels = any(
            array_specs.has_labels(*elem) for elem in subsequence
        )

        if has_labels:
            samples.append(subsequence)

    else:
        samples.append(subsequence)

def generate_subseq_samples(
    array_specs, shuffle_arrays, shuffle_elements, subseq_length,
    skip_unlabeled, prg, split_by_array_starts
):
    # pylint: disable=too-many-arguments
    # pylint: disable=too-many-locals

    # Create a sample list of [ [(arr_idx, elem_idx),]*subseq_length ]
    #
    # 1. Generate base samples (full sequences)
    # 2. Split into subsequences of fixed length
    # 3. Handle array starts and unlabeled data
    # 4. Return subsequences and array start indices

    result = [ ]

    element_samples, array_start_indices = generate_samples(
        array_specs, shuffle_arrays, shuffle_elements, skip_unlabeled = False,
        prg = prg
    )
    array_start_indices = set(array_start_indices)
    subseq_array_start_indices = [ ]

    curr_subseq = []

    for idx, elem in enumerate(element_samples):
        if split_by_array_starts and (idx in array_start_indices):
            if curr_subseq:
                append_sample(array_specs, result, curr_subseq, skip_unlabeled)
                curr_subseq = []

            subseq_array_start_indices.append(len(result))

        curr_subseq.append(elem)

        if len(curr_subseq) >= subseq_length:
            append_sample(array_specs, result, curr_subseq, skip_unlabeled)
            curr_subseq = []

    if curr_subseq:
        append_sample(array_specs, result, curr_subseq, skip_unlabeled)
        curr_subseq = []

    return result, subseq_array_start_indices

class JArrSubseqSamplerIt:
    # pylint: disable=too-many-instance-attributes

    def __init__(
        self,
        array_specs, batch_size,
        shuffle_arrays        = False,
        shuffle_elems         = False,
        shuffle_subseqs       = False,
        skip_unlabeled        = False,
        split_by_array_starts = True,
        drop_last             = False,
        pad_empty             = True,
        subseq_length         = 21,
        seed                  = 0
    ):
        # pylint: disable=too-many-arguments
        self._prg        = random.Random(seed)
        self._batch_size = batch_size
        self._pad_empty  = pad_empty
        self._sub_length = subseq_length

        # samples: list of [ [(arr_idx, elem_idx), ]*subseq_length ]
        # subseq_array_start_indices: [ sample_idx, ]
        (self._flat_to_nested_index, subseq_array_start_indices) \
            = generate_subseq_samples(
                array_specs, shuffle_arrays, shuffle_elems, subseq_length,
                skip_unlabeled, self._prg, split_by_array_starts
            )

        n_samples = len(self._flat_to_nested_index)

        # _batch_positions and _batch_end_positions control subsequence
        # sampling for each # batch slot. Each index refers to a complete
        # subsequence of length L.
        #
        # Example for batch_size == 3, with 8 subsequences total:
        #   _batch_positions     = [0, 3, 6]
        #       # starting subsequence indices for each slot
        #   _batch_end_positions = [3, 6, 8]
        #       # where each slot should stop
        #
        # The following subsequences will be sampled for each batch:
        # batch #1: [ subseq[0], subseq[3], subseq[6] ]
        # batch #2: [ subseq[1], subseq[4], subseq[7] ]
        # batch #3: [ subseq[2], subseq[5], None ]        # last slot exhausted
        # batch #4: None                                  # sampling complete
        #
        # Each subseq[i] contains L consecutive elements from the original
        # sequences, with padding if needed.

        if shuffle_subseqs:
            self._prg.shuffle(self._flat_to_nested_index)
            # NOTE: If subsequences are shuffled, array start positions have no
            # useful meaning anymore. So, we reset them.
            subseq_array_start_indices = list(
                range(len(self._flat_to_nested_index))
            )

        if split_by_array_starts:
            (self._batch_positions, self._batch_end_positions) \
                = split_indices_by_array_start(
                    n_samples, subseq_array_start_indices, batch_size
                )
        else:
            (self._batch_positions, self._batch_end_positions) \
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
        result   = []
        end_iter = True

        for bi in range(self._batch_size):
            sample_index = self._batch_positions[bi]

            if sample_index >= self._batch_end_positions[bi]:
                if self._pad_empty:
                    result.append([ None, ] * self._sub_length)
                continue

            curr_samples = []
            elements     = self._flat_to_nested_index[sample_index]

            for elem in elements:
                curr_samples.append(elem)
                end_iter = False

            if len(curr_samples) < self._sub_length:
                curr_samples += [
                    None for _ in range(self._sub_length - len(curr_samples))
                ]

            self._batch_positions[bi] += 1
            result.append(curr_samples)

        if (len(result) == 0) or end_iter:
            raise StopIteration

        return result

class JArrSubseqSampler(Sampler):
    # pylint: disable=too-many-instance-attributes

    def __init__(
        self, dataset, batch_size,
        shuffle_arrays        = False,
        shuffle_elems         = False,
        shuffle_subseqs       = False,
        skip_unlabeled        = False,
        split_by_array_starts = True,
        drop_last             = False,
        pad_empty             = True,
        subseq_length         = 21,
        seed                  = 0,
    ):
        # pylint: disable=too-many-arguments
        super().__init__(dataset)

        self._dset        = dataset
        self._array_specs = dataset.array_specs
        self._batch_size  = batch_size
        self._drop_last   = drop_last
        self._pad_empty   = pad_empty
        self._sub_length  = subseq_length
        self._seed        = seed

        self._shuffle_arrays  = shuffle_arrays
        self._shuffle_elems   = shuffle_elems
        self._shuffle_subseqs = shuffle_subseqs
        self._skip_unlabeled  = skip_unlabeled
        self._split_by_array_starts = split_by_array_starts

        self._cycles  = 0
        self._reinit_iter()

    def _reinit_iter(self):
        self._dset.set_seed(self._seed + self._cycles)

        self._curr_it = JArrSubseqSamplerIt(
            self._array_specs, self._batch_size,
            self._shuffle_arrays,
            self._shuffle_elems,
            self._shuffle_subseqs,
            self._skip_unlabeled,
            self._split_by_array_starts,
            self._drop_last,
            self._pad_empty,
            self._sub_length,
            self._seed + self._cycles,
        )

    def __len__(self):
        return len(self._curr_it)

    def __iter__(self):
        result = self._curr_it

        self._cycles += 1
        self._reinit_iter()

        return result

class VideoClipSampler(JArrSubseqSampler):
    # Simple facade over JArrSubseqSampler

    def __init__(
        self, dataset, batch_size,
        shuffle_videos        = False,
        shuffle_frames        = False,
        shuffle_clips         = False,
        skip_unlabeled        = False,
        split_by_video_starts = True,
        drop_last             = False,
        pad_empty             = True,
        clip_length           = 21,
        seed                  = 0,
    ):
        # pylint: disable=too-many-arguments
        super().__init__(
            dataset, batch_size,
            shuffle_videos, shuffle_frames, shuffle_clips, skip_unlabeled,
            split_by_video_starts, drop_last, pad_empty, clip_length, seed
        )

