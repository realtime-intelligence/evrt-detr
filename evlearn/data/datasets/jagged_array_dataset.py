"""
A base dataset class for working with jagged arrays in PyTorch.

This module provides an abstract based dataset class for handling jagged
arrays, where each array can have a different length. It supports both single
element access via (array_idx, elem_idx) pairs and batched loading with
automatic tensor stacking. Likewise, it integrates with IJaggedArraySpecs for
efficient array metadata caching

The dataset requires implementation of:
- get_elem(arr_idx, elem_idx): Fetch specific element from a jagged array
- get_null_elem(): Provide a null/padding element for special cases

Important:
The seed property is expected to be updated at each epoch. It is used by some
downstream implementations for uniquely initializing RNG for jagged-array wise
data augmentations.
"""

import torch
from torch.utils.data import Dataset

class JaggedArrayDataset(Dataset):

    def __init__(self, array_specs):
        self._specs = array_specs
        self._seed  = 0

    @property
    def array_specs(self):
        return self._specs

    def set_seed(self, seed):
        self._seed = seed

    def __len__(self):
        return len(self._specs)

    def get_elem(self, arr_idx, elem_idx):
        raise NotImplementedError

    def get_null_elem(self):
        raise NotImplementedError

    def __getitem__(self, arr_elem_index_pair):
        if arr_elem_index_pair is None:
            return self.get_null_elem()

        if isinstance(arr_elem_index_pair, list):
            result = [ self[p] for p in arr_elem_index_pair ]
            result = list(zip(*result))
            result = [
                torch.stack(x, dim = 0) if isinstance(x, torch.Tensor) else x
                    for x in result
            ]
            return result

        arr_idx, elem_idx = arr_elem_index_pair

        if (arr_idx is None) and (elem_idx is None):
            return self.get_null_elem()

        return self.get_elem(arr_idx, elem_idx)

