# Data Processing Tools

Event camera data consists of many video sequences of varying lengths. For
training neural networks, we need to sample elements from these sequences and
form them into batches. This presents a unique challenge.

The standard PyTorch DataLoader samples elements independently and randomly.
However, this approach breaks temporal relationships within video sequences -
we need to preserve these relationships while still enabling random access for
training.

To address this, we first introduced custom samplers (`samplers/`) that
understand video sequence structure and maintain temporal order during
sampling. However, to support this sampling pattern, we need datasets that can
load elements using nested indices `(array_idx, elem_idx)` - where `array_idx`
selects the video sequence and `elem_idx` selects the frame within that
sequence. This led to the creation of `JaggedArrayDataset` base class.

At the same time, efficient sampling requires quick access to sequence
metadata like dataset shapes and array lengths. Loading this information from
disk for each sampling operation would be prohibitively expensive.

This led us to implement metadata caching via `IJaggedArraySpecs`. The specs
provide fast, in-memory access to sequence information without disk IO,
allowing samplers to quickly determine valid sample ranges.

Finally, since our sequences have different lengths, we can't directly stack
them into tensors like regular PyTorch batches. The `collate/` module provides
the final piece - tools to properly pad and combine variable-length sequences
into unified batches while preserving their temporal information.

NOTE: It is likely possible to perform a more efficient collation using
NestedTensors.

## Submodule Structure

- `collate/` 	- Batch collation utilities
- `datasets/` 	- Dataset implementations and IO functions
- `samplers/` 	- Batch sampling utilities
- `transforms/` - Data augmentation transforms

