import numpy as np

def split_indices_equally(n_samples, batch_size):
    n_samples_per_batch = n_samples // batch_size

    start_indices = [ n_samples_per_batch * i for i in range(batch_size) ]
    end_indices   = start_indices[1:] + [ n_samples, ]

    return (start_indices, end_indices)

def split_indices_by_array_start(n_samples, array_start_indices, batch_size):
    # NOTE: this alorithm is fragile. Should rewrite as a solid DP.
    assert len(array_start_indices) >= batch_size, \
        "Need to implement an edge case handling"

    array_start_indices = sorted(array_start_indices)

    optimal_split_start, _optimal_split_end \
        = split_indices_equally(n_samples, batch_size)

    start_indices = []

    for n, optimal_index in enumerate(optimal_split_start):
        #    left :   ``a[i-1] < v <= a[i]``
        #   right :   ``a[i-1] <= v < a[i]``

        # Ensure that there are enough start positions left for the following
        # batch slots
        remaining_slots = batch_size - n - 1
        valid_range_end = len(array_start_indices) - remaining_slots

        i = np.searchsorted(
            array_start_indices[:valid_range_end], optimal_index, side = 'left'
        )
        i = max(i, 0)
        i = min(i, valid_range_end-1)

        start_indices.append(array_start_indices[i])

        # prevent past indices from being picked up again
        array_start_indices = array_start_indices[i+1:]

    assert len(set(start_indices)) == len(start_indices), \
        "Got duplicate indices"

    end_indices = start_indices[1:] + [ n_samples, ]

    return (start_indices, end_indices)

def calc_n_batches(batch_start_indices, batch_end_indices):
    return [
        (batch_end_indices[i] - batch_start_indices[i])
            for i in range(len(batch_start_indices))
    ]

def drop_last_batches(batch_start_indices, batch_end_indices):
    n_batches     = calc_n_batches(batch_start_indices, batch_end_indices)
    min_n_batches = min(n_batches)

    return [
        batch_start_idx + min_n_batches
            for batch_start_idx in batch_start_indices
    ]

def calculate_sampler_length(batch_start_indices, batch_end_indices):
    n_batches = calc_n_batches(batch_start_indices, batch_end_indices)
    return max(n_batches)


