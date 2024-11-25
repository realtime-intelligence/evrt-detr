""" Uniform random sampler without replacement for JaggedArrayDataset """
import random
from torch.utils.data.sampler import Sampler

from .funcs import (
    split_indices_equally, drop_last_batches, calculate_sampler_length
)

def generate_sampling_index(array_specs, skip_unlabeled, prg):
    # Flatten and shuffle jagged array specs `array_specs`
    elements = []

    for arr_idx in range(array_specs.get_n_arrays()):
        n_elements = array_specs.get_array_length(arr_idx)

        if not skip_unlabeled:
            elements += [
                (arr_idx, elem_idx) for elem_idx in range(n_elements)
            ]
        else:
            elements += [
                (arr_idx, elem_idx)
                    for elem_idx in range(n_elements)
                        if array_specs.has_labels(arr_idx, elem_idx)
            ]

    prg.shuffle(elements)

    return dict(enumerate(elements))

class JArrRandomSamplerIt:

    def __init__(
        self,
        array_specs, batch_size,
        skip_unlabeled        = False,
        drop_last             = False,
        pad_empty             = True,
        seed                  = 0
    ):
        # pylint: disable=too-many-arguments
        self._prg        = random.Random(seed)
        self._batch_size = batch_size
        self._pad_empty  = pad_empty

        # map flat_idx to (array_idx, elem_idx) in dataset
        self._flat_to_nested_idx = generate_sampling_index(
            array_specs, skip_unlabeled, self._prg
        )

        n_samples = len(self._flat_to_nested_idx)

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
            flat_index = self._batch_positions[bi]

            if flat_index >= self._batch_end_positions[bi]:
                if self._pad_empty:
                    result.append(None)
                continue

            arr_idx, elem_idx = self._flat_to_nested_idx[flat_index]
            result.append((arr_idx, elem_idx))

            self._batch_positions[bi] += 1

        if (len(result) == 0) or all(x is None for x in result):
            raise StopIteration

        return result

class JArrRandomSampler(Sampler):
    # pylint: disable=too-many-instance-attributes

    def __init__(
        self, dataset, batch_size,
        skip_unlabeled        = False,
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

        self._skip_unlabeled = skip_unlabeled

        self._cycles  = 0
        self._reinit_iter()

    def _reinit_iter(self):
        self._dset.set_seed(self._seed + self._cycles)

        self._curr_it = JArrRandomSamplerIt(
            self._array_specs, self._batch_size,
            self._skip_unlabeled,
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

