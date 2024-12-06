from itertools import product
import unittest
import random

from evlearn.data.samplers.jarr_sampler import JArrSamplerIt

class MockArraySpecs:
    def __init__(self, lengths, labeled_indices=None):
        self.lengths = lengths
        self.labeled_indices = labeled_indices or set()

    def __len__(self):
        return sum(self.lengths)

    def get_array_length(self, idx):
        return self.lengths[idx]

    def get_n_arrays(self):
        return len(self.lengths)

    def has_labels(self, arr_idx, elem_idx):
        return (arr_idx, elem_idx) in self.labeled_indices

class TestJArrSubseqSamplerIt(unittest.TestCase):

    def assertBatchSize(self, actual_batches, batch_size):
        # Invariant 1: Each batch has correct size
        for batch in actual_batches:
            self.assertEqual(
                len(batch), batch_size,
                f"Each batch should have size {batch_size}"
            )

    def assertBatchCoverage(self, actual_batches, array_specs):
        # Invariant 2: All elements should be sampled and only once
        sampled_items = set()

        for batch in actual_batches:
            for item in batch:
                if item is not None:
                    self.assertNotIn(
                        item, sampled_items,
                        f"Frame {item} was sampled multiple times"
                    )
                    sampled_items.add(item)

        expected_items = set(
            (arr_idx, elem_idx)
            for arr_idx in range(array_specs.get_n_arrays())
            for elem_idx in range(array_specs.get_array_length(arr_idx))
        )

        self.assertEqual(
            sampled_items, expected_items, "Some frames were not sampled"
        )

    def assertElementOrder(self, sampled_elements):
        for idx, elements in enumerate(sampled_elements):
            sorted_elements = sorted(elements)

            self.assertEqual(
                elements, sorted_elements,
                (
                    f"Sequence {idx} elemets are not ordered"
                    f" {elements} vs {sorted_elements}"
                )
            )

    def assertBatchContinuitySplitByArrayStarts(
        self, actual_batches, shuffle_elems
    ):
        # When not shuffling, each sequence stays in its assigned slot
        sequence_slots    = { }  # arr_idx -> slot_idx
        sequence_elements = { }  # arr_idx -> [ elem_idx ]

        for batch in actual_batches:
            for slot_idx, item in enumerate(batch):
                if item is not None:
                    arr_idx, elem_idx = item
                    if arr_idx not in sequence_slots:
                        sequence_slots[arr_idx]    = slot_idx
                        sequence_elements[arr_idx] = [ elem_idx, ]
                        continue

                    sequence_elements[arr_idx].append(elem_idx)
                    self.assertEqual(
                        slot_idx, sequence_slots[arr_idx],
                        (
                            f"Sequence {arr_idx} appeared in slot {slot_idx}"
                            " but was previously in slot"
                            f" {sequence_slots[arr_idx]}"
                        )
                    )

        if not shuffle_elems:
            self.assertElementOrder(sequence_elements.values())

    def assertBatchContinuitySplitUniformly(
        self, actual_batches, shuffle_elems
    ):
        # When not shuffling, each sequence stays in its assigned slot
        slot_arrays   = { }  # slot_idx -> arr_idx
        slot_elements = { }  # slot_idx -> [ elem_idx, ]

        sequences = []

        for batch in actual_batches:
            for slot_idx, item in enumerate(batch):
                if item is not None:
                    arr_idx, elem_idx = item

                    if slot_idx not in slot_arrays:
                        slot_arrays  [slot_idx] = arr_idx
                        slot_elements[slot_idx] = [ elem_idx, ]

                    elif slot_arrays[slot_idx] != arr_idx:
                        sequences.append(slot_elements[slot_idx])
                        slot_arrays  [slot_idx] = arr_idx
                        slot_elements[slot_idx] = [ elem_idx, ]

                    else:
                        slot_elements[slot_idx].append(elem_idx)

        sequences += list(slot_elements.values())

        if not shuffle_elems:
            self.assertElementOrder(sequences)

    def assertBatchInvariants(
        self, actual_batches, batch_size, array_specs,
        shuffle_arrays, shuffle_elems, split_by_array_starts
    ):
        # Invariant 1: Each batch has correct size
        self.assertBatchSize(actual_batches, batch_size)

        # Invariant 2: All elements should be sampled and only once
        self.assertBatchCoverage(actual_batches, array_specs)

        if split_by_array_starts:
            # Invariant 3: Each array corresponds to a fixed batch slot
            self.assertBatchContinuitySplitByArrayStarts(
                actual_batches, shuffle_elems
            )
        else:
            self.assertBatchContinuitySplitUniformly(
                actual_batches, shuffle_elems
            )

    def test_basic_sampling_no_shuffle(self):
        array_specs           = MockArraySpecs([5, 3, 4])
        batch_size            = 2
        shuffle_arrays        = False
        shuffle_elems         = False
        split_by_array_starts = True

        sampler     = JArrSamplerIt(
            array_specs           = array_specs,
            batch_size            = batch_size,
            shuffle_arrays        = shuffle_arrays,
            shuffle_elems         = shuffle_elems,
            skip_unlabeled        = False,
            split_by_array_starts = split_by_array_starts,
            pad_empty             = True,
            seed                  = 0
        )

        actual_batches = list(sampler)
        self.assertBatchInvariants(
            actual_batches, batch_size, array_specs,
            shuffle_arrays, shuffle_elems, split_by_array_starts
        )

    def test_basic_sampling_shuffle_arrays(self):
        array_specs           = MockArraySpecs([5, 3, 4])
        batch_size            = 2
        shuffle_arrays        = True
        shuffle_elems         = False
        split_by_array_starts = True

        sampler     = JArrSamplerIt(
            array_specs           = array_specs,
            batch_size            = batch_size,
            shuffle_arrays        = shuffle_arrays,
            shuffle_elems         = shuffle_elems,
            skip_unlabeled        = False,
            split_by_array_starts = split_by_array_starts,
            pad_empty             = True,
            seed                  = 0
        )

        actual_batches = list(sampler)
        self.assertBatchInvariants(
            actual_batches, batch_size, array_specs,
            shuffle_arrays, shuffle_elems, split_by_array_starts
        )

    def test_basic_sampling_shuffle_elements(self):
        array_specs           = MockArraySpecs([5, 3, 4])
        batch_size            = 2
        shuffle_arrays        = False
        shuffle_elems         = True
        split_by_array_starts = True

        sampler = JArrSamplerIt(
            array_specs           = array_specs,
            batch_size            = batch_size,
            shuffle_arrays        = shuffle_arrays,
            shuffle_elems         = shuffle_elems,
            skip_unlabeled        = False,
            split_by_array_starts = split_by_array_starts,
            pad_empty             = True,
            seed                  = 0
        )

        actual_batches = list(sampler)
        self.assertBatchInvariants(
            actual_batches, batch_size, array_specs,
            shuffle_arrays, shuffle_elems, split_by_array_starts
        )

    def test_basic_sampling_shuffle_elements_and_arrays(self):
        array_specs           = MockArraySpecs([5, 3, 4])
        batch_size            = 2
        shuffle_arrays        = True
        shuffle_elems         = True
        split_by_array_starts = True

        sampler = JArrSamplerIt(
            array_specs           = array_specs,
            batch_size            = batch_size,
            shuffle_arrays        = shuffle_arrays,
            shuffle_elems         = shuffle_elems,
            skip_unlabeled        = False,
            split_by_array_starts = split_by_array_starts,
            pad_empty             = True,
            seed                  = 0
        )

        actual_batches = list(sampler)
        self.assertBatchInvariants(
            actual_batches, batch_size, array_specs,
            shuffle_arrays, shuffle_elems, split_by_array_starts
        )

    def test_basic_sampling_shuffle_elements_and_arrays_uniform_split(self):
        array_specs           = MockArraySpecs([5, 3, 4])
        batch_size            = 2
        shuffle_arrays        = True
        shuffle_elems         = True
        split_by_array_starts = False

        sampler     = JArrSamplerIt(
            array_specs           = array_specs,
            batch_size            = batch_size,
            shuffle_arrays        = shuffle_arrays,
            shuffle_elems         = shuffle_elems,
            skip_unlabeled        = False,
            split_by_array_starts = split_by_array_starts,
            pad_empty             = True,
            seed                  = 0
        )

        actual_batches = list(sampler)
        self.assertBatchInvariants(
            actual_batches, batch_size, array_specs,
            shuffle_arrays, shuffle_elems, split_by_array_starts
        )

    def test_basic_sampling_noshuffle_elements_and_arrays_uniform_split(self):
        array_specs           = MockArraySpecs([5, 3, 4])
        batch_size            = 2
        shuffle_arrays        = False
        shuffle_elems         = False
        split_by_array_starts = False

        sampler     = JArrSamplerIt(
            array_specs           = array_specs,
            batch_size            = batch_size,
            shuffle_arrays        = shuffle_arrays,
            shuffle_elems         = shuffle_elems,
            skip_unlabeled        = False,
            split_by_array_starts = split_by_array_starts,
            pad_empty             = True,
            seed                  = 0
        )

        actual_batches = list(sampler)
        self.assertBatchInvariants(
            actual_batches, batch_size, array_specs,
            shuffle_arrays, shuffle_elems, split_by_array_starts
        )

    def test_large_sampling_noshuffle_elements_and_arrays_uniform_split(self):
        array_specs           = MockArraySpecs([5, 3, 4, 64, 128, 1024])
        batch_size            = 32
        shuffle_arrays        = False
        shuffle_elems         = False
        split_by_array_starts = False

        sampler = JArrSamplerIt(
            array_specs           = array_specs,
            batch_size            = batch_size,
            shuffle_arrays        = shuffle_arrays,
            shuffle_elems         = shuffle_elems,
            skip_unlabeled        = False,
            split_by_array_starts = split_by_array_starts,
            pad_empty             = True,
            seed                  = 0
        )

        actual_batches = list(sampler)
        self.assertBatchInvariants(
            actual_batches, batch_size, array_specs,
            shuffle_arrays, shuffle_elems, split_by_array_starts
        )

    def test_skip_unlabeled_uniform_split(self):
        # Array with alternating labeled frames
        # [L, U, L, U, L] - should sample only [0, 2, 4]
        labeled_indices = {(0,0), (0,2), (0,4)}
        array_specs     = MockArraySpecs([5, ], labeled_indices)

        sampler = JArrSamplerIt(
            array_specs           = array_specs,
            batch_size            = 2,
            shuffle_arrays        = False,
            shuffle_elems         = False,
            skip_unlabeled        = True,
            split_by_array_starts = False,  # Use uniform split
            pad_empty             = True,
            seed                  = 0
        )

        actual_batches = list(sampler)

        # With uniform split and no shuffling, we should get:
        # [(0,0), (0,2)]
        # [(0,4), None]
        expected_batches = [
            [(0,0), (0,2)],
            [None,  (0,4)]
        ]

        # Extra verification that batches match exactly what we expect
        self.assertEqual(actual_batches, expected_batches)

if __name__ == '__main__':
    unittest.main()

