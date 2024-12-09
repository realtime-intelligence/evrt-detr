
class DataLoaderListZipper:

    def __init__(self, loaders):
        self._loaders = loaders

    def __len__(self):
        return min(len(d) for d in self._loaders)

    def __iter__(self):
        return zip(*self._loaders)

class IterDict:

    def __init__(self, container):
        self._iters = { k : iter(v) for (k, v) in container.items() }

    def __iter__(self):
        return self

    def __next__(self):
        return { k : next(v) for (k, v) in self._iters.items() }

class DataLoaderDictZipper:

    def __init__(self, loaders):
        self._loaders = loaders

    def __len__(self):
        return min(len(d) for d in self._loaders.values())

    def __iter__(self):
        return IterDict(self._loaders)

