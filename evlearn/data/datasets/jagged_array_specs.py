"""
Provides metadata handling for jagged arrays in datasets.

This module defines interfaces and implementations for managing jagged array
specifications. It enables efficient metadata caching to avoid repeated disk
access when working with jagged array datasets.

Key components:
- ElemSpec: Named tuple for element-level specifications
- JArrSpec: Named tuple for array-level specifications
- IJaggedArraySpecs: Base interface for jagged array specifications
- SimpleJaggedArraySpecs: Basic implementation of the specification interface

The metadata provided by IJaggedArraySpecs includes:
- number of jagged arrays in a dataset
- size of each jagged array
- custom data associated with each jagged array (`name` field)
- custom data associated with each element of a jagged array
  (`name`, `data`, `labels`)
- `labels` field of a jagged array element has special meaning -- indicating
  whether the element has annotation or not (labels is None)
"""

from collections import namedtuple

ElemSpec = namedtuple('ElemSpec', [ 'name', 'data', 'labels' ])
JArrSpec = namedtuple('JArrSpec', [ 'name', 'elements' ])

class IJaggedArraySpecs:

    def __len__(self):
        raise NotImplementedError

    def get_n_arrays(self):
        raise NotImplementedError

    def get_array_name(self, arr_idx):
        raise NotImplementedError

    def get_array_length(self, arr_idx):
        raise NotImplementedError

    def get_elem_spec(self, arr_idx, elem_idx):
        raise NotImplementedError

    def has_labels(self, arr_idx, elem_idx):
        raise NotImplementedError

class SimpleJaggedArraySpecs(IJaggedArraySpecs):

    def __init__(self, jagged_arrays_specs):
        # jagged_arrays_specs : List[JArrSpec]
        self._specs = jagged_arrays_specs

    def __len__(self):
        return sum(len(arr.elements) for arr in self._specs)

    def get_n_arrays(self):
        return len(self._specs)

    def get_array_name(self, arr_idx):
        return self._specs[arr_idx].name

    def get_array_length(self, arr_idx):
        return len(self._specs[arr_idx].elements)

    def get_elem_spec(self, arr_idx, elem_idx):
        return self._specs[arr_idx].elements[elem_idx]

    def has_labels(self, arr_idx, elem_idx):
        return self._specs[arr_idx].elements[elem_idx].labels is not None

