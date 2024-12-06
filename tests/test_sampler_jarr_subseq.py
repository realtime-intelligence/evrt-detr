import unittest
from evlearn.data.samplers.jarr_subseq_sampler import JArrSubseqSamplerIt

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

    def assertSubsequenceSize(self, actual_batches, subseq_length):
        # Invariant 2: Each subsequence has correct length
        for batch in actual_batches:
            for subseq in batch:
                self.assertEqual(
                    len(subseq), subseq_length,
                    f"Each subsequence should have length {subseq_length}"
                )

    def assertBatchCoverage(self, actual_batches, array_specs):
        # Invariant 3: All elements should be sampled and only once
        sampled_items = set()

        for batch in actual_batches:
            for subseq in batch:
                for item in subseq:
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

    def assertBatchContinuitySplitByArrayStarts(self, actual_batches, shuffle_elems):
        # When splitting by array starts, each array stays in its assigned
        # batch slot AND elements within each slot must be contiguous
        sequence_slots    = {}    # arr_idx -> slot_idx
        sequence_elements = {}    # arr_idx -> [elem_idx]

        for batch in actual_batches:
            for slot_idx, subseq in enumerate(batch):
                for item in subseq:
                    if item is not None:
                        arr_idx, elem_idx = item

                        if arr_idx not in sequence_slots:
                            sequence_slots[arr_idx]    = slot_idx
                            sequence_elements[arr_idx] = [elem_idx]
                            continue

                        sequence_elements[arr_idx].append(elem_idx)
                        self.assertEqual(
                            slot_idx, sequence_slots[arr_idx],
                            f"Sequence {arr_idx} appeared in slot {slot_idx} "
                            "but was previously in slot"
                            f" {sequence_slots[arr_idx]}:\n"
                            f" > {actual_batches}"
                        )

        if not shuffle_elems:
            self.assertElementOrder(sequence_elements.values())

    def assertBatchContinuitySplitUniformly(self, actual_batches, shuffle_elems):
        # When not shuffling, each sequence stays in its assigned slot
        slot_arrays   = {}    # slot_idx -> arr_idx
        slot_elements = {}    # slot_idx -> [elem_idx]
        
        sequences = []
        
        for batch in actual_batches:
            for slot_idx, subseq in enumerate(batch):
                for item in subseq:
                    if item is not None:
                        arr_idx, elem_idx = item
                        
                        if slot_idx not in slot_arrays:
                            slot_arrays[slot_idx]   = arr_idx
                            slot_elements[slot_idx] = [elem_idx]
                            
                        elif slot_arrays[slot_idx] != arr_idx:
                            sequences.append(slot_elements[slot_idx])
                            slot_arrays[slot_idx]   = arr_idx
                            slot_elements[slot_idx] = [elem_idx]
                            
                        else:
                            slot_elements[slot_idx].append(elem_idx)
                            
        sequences += list(slot_elements.values())

        if not shuffle_elems:
            self.assertElementOrder(sequences)

    def assertBatchInvariants(
        self, actual_batches, batch_size, subseq_length, array_specs,
        shuffle_arrays, shuffle_elems, split_by_array_starts
    ):
        self.assertBatchSize(actual_batches, batch_size)
        self.assertSubsequenceSize(actual_batches, subseq_length)
        self.assertBatchCoverage(actual_batches, array_specs)

        if split_by_array_starts:
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
        subseq_length         = 2
        shuffle_arrays        = False
        shuffle_elems         = False
        shuffle_subseqs       = False
        split_by_array_starts = True
        
        sampler = JArrSubseqSamplerIt(
            array_specs           = array_specs,
            batch_size            = batch_size,
            shuffle_arrays        = shuffle_arrays,
            shuffle_elems         = shuffle_elems,
            shuffle_subseqs       = shuffle_subseqs,
            skip_unlabeled        = False,
            split_by_array_starts = split_by_array_starts,
            pad_empty             = True,
            subseq_length         = subseq_length,
            seed                  = 0
        )

        actual_batches = list(sampler)

        self.assertBatchInvariants(
            actual_batches, batch_size, subseq_length, array_specs,
            shuffle_arrays, shuffle_elems, split_by_array_starts
        )

    def test_skip_unlabeled(self):
        labeled_indices = {(0,0), (0,1), (0,4), (1,0), (1,2)}
        array_specs     = MockArraySpecs([5, 3], labeled_indices)
        batch_size      = 2
        subseq_length   = 2
        
        sampler = JArrSubseqSamplerIt(
            array_specs           = array_specs,
            batch_size            = batch_size,
            shuffle_arrays        = False,
            shuffle_elems         = False,
            skip_unlabeled        = True,
            split_by_array_starts = True,
            pad_empty             = True,
            subseq_length         = subseq_length,
            seed                  = 0
        )

        actual_batches = list(sampler)
        
        for batch in actual_batches:
            for subseq in batch:
                has_label = False
                for item in subseq:
                    if item is not None:
                        self.assertTrue(item in labeled_indices)
                        has_label = True
                        break

                if not all(x is None for x in subseq):
                    self.assertTrue(
                        has_label, "Unlabeled subsequence was included"
                    )

    def test_basic_sampling_shuffle_arrays(self):
        array_specs           = MockArraySpecs([5, 3, 4])
        batch_size            = 2
        subseq_length         = 2
        shuffle_arrays        = True
        shuffle_elems         = False
        shuffle_subseqs       = False
        split_by_array_starts = True

        sampler = JArrSubseqSamplerIt(
            array_specs           = array_specs,
            batch_size            = batch_size,
            shuffle_arrays        = shuffle_arrays,
            shuffle_elems         = shuffle_elems,
            shuffle_subseqs       = shuffle_subseqs,
            skip_unlabeled        = False,
            split_by_array_starts = split_by_array_starts,
            pad_empty             = True,
            subseq_length         = subseq_length,
            seed                  = 0
        )

        actual_batches = list(sampler)

        self.assertBatchInvariants(
            actual_batches, batch_size, subseq_length, array_specs,
            shuffle_arrays, shuffle_elems, split_by_array_starts
        )

    def test_basic_sampling_shuffle_elements(self):
        array_specs           = MockArraySpecs([5, 3, 4])
        batch_size            = 2
        subseq_length         = 2
        shuffle_arrays        = False
        shuffle_elems         = True
        shuffle_subseqs       = False
        split_by_array_starts = True

        sampler = JArrSubseqSamplerIt(
            array_specs           = array_specs,
            batch_size            = batch_size,
            shuffle_arrays        = shuffle_arrays,
            shuffle_elems         = shuffle_elems,
            shuffle_subseqs       = shuffle_subseqs,
            skip_unlabeled        = False,
            split_by_array_starts = split_by_array_starts,
            pad_empty             = True,
            subseq_length         = subseq_length,
            seed                  = 0
        )

        actual_batches = list(sampler)

        self.assertBatchInvariants(
            actual_batches, batch_size, subseq_length, array_specs,
            shuffle_arrays, shuffle_elems, split_by_array_starts
        )

    def test_basic_sampling_shuffle_elements_and_arrays(self):
        array_specs           = MockArraySpecs([5, 3, 4])
        batch_size            = 2
        subseq_length         = 2
        shuffle_arrays        = True
        shuffle_elems         = True
        shuffle_subseqs       = False
        split_by_array_starts = True

        sampler = JArrSubseqSamplerIt(
            array_specs           = array_specs,
            batch_size            = batch_size,
            shuffle_arrays        = shuffle_arrays,
            shuffle_elems         = shuffle_elems,
            shuffle_subseqs       = shuffle_subseqs,
            skip_unlabeled        = False,
            split_by_array_starts = split_by_array_starts,
            pad_empty             = True,
            subseq_length         = subseq_length,
            seed                  = 0
        )

        actual_batches = list(sampler)

        self.assertBatchInvariants(
            actual_batches, batch_size, subseq_length, array_specs,
            shuffle_arrays, shuffle_elems, split_by_array_starts
        )

    def test_basic_sampling_uniform_split(self):
        array_specs           = MockArraySpecs([5, 3, 4])
        batch_size            = 2
        subseq_length         = 2
        shuffle_arrays        = False
        shuffle_elems         = False
        shuffle_subseqs       = False
        split_by_array_starts = False

        sampler = JArrSubseqSamplerIt(
            array_specs           = array_specs,
            batch_size            = batch_size,
            shuffle_arrays        = shuffle_arrays,
            shuffle_elems         = shuffle_elems,
            shuffle_subseqs       = shuffle_subseqs,
            skip_unlabeled        = False,
            split_by_array_starts = split_by_array_starts,
            pad_empty             = True,
            subseq_length         = subseq_length,
            seed                  = 0
        )

        actual_batches = list(sampler)

        self.assertBatchInvariants(
            actual_batches, batch_size, subseq_length, array_specs,
            shuffle_arrays, shuffle_elems, split_by_array_starts
        )

    def test_large_sampling(self):
        array_specs           = MockArraySpecs([64, 32, 128, 96, 256, 48])
        batch_size            = 16
        subseq_length         = 8
        shuffle_arrays        = False
        shuffle_elems         = False
        shuffle_subseqs       = False
        split_by_array_starts = False

        sampler = JArrSubseqSamplerIt(
            array_specs           = array_specs,
            batch_size            = batch_size,
            shuffle_arrays        = shuffle_arrays,
            shuffle_elems         = shuffle_elems,
            shuffle_subseqs       = shuffle_subseqs,
            skip_unlabeled        = False,
            split_by_array_starts = split_by_array_starts,
            pad_empty             = True,
            subseq_length         = subseq_length,
            seed                  = 0
        )

        actual_batches = list(sampler)

        self.assertBatchInvariants(
            actual_batches, batch_size, subseq_length, array_specs,
            shuffle_arrays, shuffle_elems, split_by_array_starts
        )


if __name__ == '__main__':
    unittest.main()
