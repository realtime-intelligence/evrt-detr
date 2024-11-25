"""
Functions for collating batches of jagged arrays into fixed-size tensors with
padding.

This module provides tools for collating batches with the following structure:
    input : List[(x, y, ..., labels)]
        # batch of tuples with variable length sequences

    output: (                          # tuple of fixed size tensors/lists
        Tensor[x]: Shape[N, L, ...],   # padded sequence tensors
        Tensor[y]: Shape[N, L, ...],
        ...,
        List[labels]                   # nested list of labels
    )

where:
- (x, y, ...) are jagged arrays (sequences of varying lengths)
- N is the batch size
- L is the maximum sequence length in the batch
- `labels` can be arbitrary nested data structures

Note: The module supports both batch-first (N, L, ...) and
time-first (L, N, ...) tensor formats via the batch_first parameter.
"""

import copy
import logging

import numpy as np
import torch

from .simple import simple_collate

LOGGER = logging.getLogger('evlearn.data.collabe')

def determine_seq_elem_shape_dtype_device(batch):
    for seq in batch:
        if seq is None:
            continue

        for elem in seq:
            if elem is None:
                continue

            if isinstance(elem, torch.Tensor):
                return (elem.shape, elem.dtype, elem.device)
            elif isinstance(elem, np.ndarray):
                dtype = elem.dtype.name
                # NOTE: need to think how to fix this dtype conversion hack
                dtype = getattr(torch, dtype)
                return elem.shape, dtype, 'cpu'
            else:
                raise ValueError(f"Unknown how to collate type '{type(elem)}'")

    return (None, None, None)

def determine_embedding_for_batch_of_jagged_arrays(batch):
    n = len(batch)

    if (n == 0) or all(seq is None for seq in batch):
        return n, 0, None, None, None

    t = max(len(seq) for seq in batch if seq is not None)
    elem_shape, elem_dtype, elem_device \
        = determine_seq_elem_shape_dtype_device(batch)

    return n, t, elem_shape, elem_dtype, elem_device

def collate_batch_of_jagged_arrays(
    batch, fill_value = 0, batch_first = True, device = None, dtype = None
):
    # pylint: disable=too-many-locals
    n, t, elem_shape, embed_dtype, embed_device \
        = determine_embedding_for_batch_of_jagged_arrays(batch)

    if device is None:
        device = embed_device

    if dtype is None:
        dtype = embed_dtype

    if elem_shape is None:
        return None

    if batch_first:
        embed_shape = (n, t, *elem_shape)
    else:
        embed_shape = (t, n, *elem_shape)

    result = torch.full(
        embed_shape, fill_value, dtype = dtype, device = device
    )

    for bi, seq in enumerate(batch):
        if seq is None:
            continue

        for ti, elem in enumerate(seq):
            if elem is None:
                continue

            if isinstance(elem, np.ndarray):
                elem = torch.from_numpy(elem)

            if batch_first:
                result[bi, ti, ...] = elem
            else:
                result[ti, bi, ...] = elem

    return result

def verify_shape_of_batch_of_jagged_labels_vs_shape_of_data(
    batch, n_data, t_data
):
    n_labels = len(batch)

    if (n_labels == 0) or all(seq is None for seq in batch):
        return

    t_labels = max(len(seq) for seq in batch if seq is not None)

    if (t_labels > t_data) or (n_labels > n_data):
        logging.warning(
            "Mismatch detected between the shape of a batch of jagged arrays"
            " (%d, %d) vs the shape of their labels (%d, %d).",
            n_data, t_data, n_labels, t_labels
        )

def collate_batch_of_jagged_labels(
    batch, collated_arrays, batch_first = True, make_copy = False
):
    if collated_arrays is None:
        return None

    shape  = collated_arrays.shape[:2]
    result = [ [ None for _ in range(shape[1]) ] for _ in range(shape[0]) ]

    if batch is None:
        return result

    if batch_first:
        (n, t) = shape
    else:
        (t, n) = shape

    verify_shape_of_batch_of_jagged_labels_vs_shape_of_data(batch, n, t)

    for bi, seq in enumerate(batch):
        if bi >= n:
            break

        if seq is None:
            continue

        for ti, elem in enumerate(seq):
            if make_copy:
                elem = copy.deepcopy(elem)

            if ti >= t:
                continue

            if batch_first:
                result[bi][ti] = elem
            else:
                result[ti][bi] = elem

    return result

def collate_batch_of_jagged_arrays_and_labels(
    batch, fill_value = 0, batch_first = True, make_copy = False
):
    if len(batch) == 0:
        return batch

    simply_collated_batch = simple_collate(batch)

    if len(simply_collated_batch) == 0:
        return simply_collated_batch

    data   = simply_collated_batch[:-1]
    labels = simply_collated_batch[-1]

    if len(data) == 0:
        raise RuntimeError('Got jagged array made of labels only.')

    data = [
        collate_batch_of_jagged_arrays(x, fill_value, batch_first)
            for x in data
    ]

    labels = collate_batch_of_jagged_labels(
        labels, data[0], batch_first, make_copy
    )

    data.append(labels)

    return data

