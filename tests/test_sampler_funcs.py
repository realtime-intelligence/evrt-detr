import unittest
import numpy as np

from evlearn.data.samplers.funcs import (
    split_indices_equally,
    split_indices_by_array_start,
    calc_n_batches,
    drop_last_batches,
    calculate_sampler_length
)

class TestBatchProcessing(unittest.TestCase):
    def test_split_indices_equally_basic(self):
        samples    = 100
        batch_size = 4

        start, end = split_indices_equally(samples, batch_size)

        self.assertEqual(start, [0,  25, 50, 75])
        self.assertEqual(end,   [25, 50, 75, 100])

    def test_split_indices_equally_uneven(self):
        samples    = 10
        batch_size = 3

        start, end = split_indices_equally(samples, batch_size)

        self.assertEqual(start, [0, 3, 6])
        self.assertEqual(end,   [3, 6, 10])

    def test_split_indices_by_array_start_basic(self):
        samples      = 100
        batch_size   = 4
        array_starts = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90]

        start, end \
            = split_indices_by_array_start(samples, array_starts, batch_size)

        self.assertEqual(start, [0,  30, 50, 80])
        self.assertEqual(end,   [30, 50, 80, 100])

    def test_split_indices_by_array_start_exact_matches(self):
        samples      = 100
        batch_size   = 4
        array_starts = [0, 25, 50, 75, 100]

        start, end \
            = split_indices_by_array_start(samples, array_starts, batch_size)

        self.assertEqual(start, [0,  25, 50, 75])
        self.assertEqual(end,   [25, 50, 75, 100])

    def test_split_indices_by_array_start_minimum_indices(self):
        samples      = 100
        batch_size   = 4
        array_starts = [0, 30, 60, 90]

        start, end \
            = split_indices_by_array_start(samples, array_starts, batch_size)

        self.assertEqual(start, [0,  30, 60, 90])
        self.assertEqual(end,   [30, 60, 90, 100])

    def test_split_indices_by_array_start_early_start(self):
        samples      = 100
        batch_size   = 4
        array_starts = [0, 1, 2, 3, 4]

        start, end \
            = split_indices_by_array_start(samples, array_starts, batch_size)

        self.assertEqual(start, [0, 2, 3, 4])
        self.assertEqual(end,   [2, 3, 4, 100])

    def test_split_indices_by_array_start_late_start(self):
        samples      = 100
        batch_size   = 4
        array_starts = [0, 97, 98, 99]

        start, end \
            = split_indices_by_array_start(samples, array_starts, batch_size)

        self.assertEqual(start, [0,  97, 98, 99])
        self.assertEqual(end,   [97, 98, 99, 100])

    def test_split_indices_by_array_start_very_uneven_start(self):
        samples      = 100
        batch_size   = 4
        array_starts = [0, 1, 2, 98, 99]

        start, end \
            = split_indices_by_array_start(samples, array_starts, batch_size)

        self.assertEqual(start, [0,  2, 98, 99])
        self.assertEqual(end,   [2, 98, 99, 100])

    def test_calc_n_batches_basic(self):
        starts = [0,  30, 60]
        ends   = [30, 60, 100]

        batches = calc_n_batches(starts, ends)

        self.assertEqual(batches, [30, 30, 40])

    def test_drop_last_batches_basic(self):
        starts = [0,  30, 60]
        ends   = [30, 60, 100]

        new_ends = drop_last_batches(starts, ends)

        self.assertEqual(new_ends, [30, 60, 90])

    def test_calculate_sampler_length_basic(self):
        starts = [0, 30, 60]
        ends   = [30, 60, 100]

        length = calculate_sampler_length(starts, ends)

        self.assertEqual(length, 40)  # Largest batch size

if __name__ == '__main__':
    unittest.main()
