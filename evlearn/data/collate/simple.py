"""
Functions for simple collation of batch tuples into lists or tensors.

This module provides tools for collating batches with the following structure:

    input  : List[(x, y, ..., labels)]  # batch of tuples

    output : Tuple containing either:
      - (List[x], List[y], ..., List[labels])
            # via simple_collate

      - (Tensor[x], Tensor[y], ..., List[labels])
            # via default_with_simple_labels

Note: Labels are always kept as lists to support arbitrary data structures.
"""
from torch.utils.data import default_collate

def simple_collate(batch):
    n = len(batch)

    if n == 0:
        return batch

    elem = batch[0]

    if isinstance(elem, (list, tuple)):
        cls = type(elem)
        return cls(
            [ batch[bidx][eidx] for bidx in range(n) ]
                for eidx in range(len(elem))
        )

    if isinstance(elem, dict):
        return {
            k : [ batch[bidx][k] for bidx in range(n) ]
                for (k, v) in elem.items()
        }

    return batch

def default_with_simple_labels(batch):
    # batch : List of length batch_size of [ (x, y, ..., labels) ]
    if len(batch) == 0:
        return batch

    # simply_collated_batch : [List[x], List[y], ..., List[labels]]
    simply_collated_batch = simple_collate(batch)

    if len(simply_collated_batch) == 0:
        return simply_collated_batch

    data   = simply_collated_batch[:-1]
    labels = simply_collated_batch[-1]

    data = [ default_collate(x) for x in data ]

    data.append(labels)

    return data

