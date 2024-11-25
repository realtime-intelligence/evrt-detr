from functools import partial
from torch.utils.data import default_collate

from evlearn.bundled.leanbase.base.funcs import extract_name_kwargs

from .simple import simple_collate, default_with_simple_labels
from .jarr   import collate_batch_of_jagged_arrays_and_labels

def select_collate_fn(collate):
    name, kwargs = extract_name_kwargs(collate)

    if name in ('default', 'torch-default'):
        return partial(default_collate, **kwargs)

    if name == 'simple':
        return partial(simple_collate, **kwargs)

    if name == 'default-with-labels':
        return partial(default_with_simple_labels, **kwargs)

    if name in ('jarr', 'video'):
        return partial(collate_batch_of_jagged_arrays_and_labels, **kwargs)

    raise ValueError(f'Unknown collation strategy: {collate}')

