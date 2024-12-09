from copy import deepcopy
from collections import defaultdict

class Metrics:

    def __init__(self, values = None, counts = None, prefix = None):
        if values is None:
            values = defaultdict(int)

        if counts is None:
            counts = defaultdict(int)

        self._values = values
        self._counts = counts
        self._prefix = prefix or ''

    def get(self):
        if self._values is None:
            return {}

        return { k : v / self._counts[k] for (k, v) in self._values.items() }

    def update(self, values):
        if values is None:
            return

        for (k,v) in values.items():
            self._values[self._prefix + k] += v
            self._counts[self._prefix + k] += 1

    def join(self, other):
        # pylint: disable=protected-access
        result = deepcopy(self)

        if other is None:
            return result

        for (k, v) in other._values.items():
            result._values[k] = v
            result._counts[k] = other._counts[k]

        return result

