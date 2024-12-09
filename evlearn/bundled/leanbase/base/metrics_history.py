import copy
import os
import re

import pandas as pd

ENCODING = 'utf-8'

LOG_FNAME_BASE = 'history'
LOG_FNAME_RE   = r'^' + LOG_FNAME_BASE + r'_(.*)\.csv$'

# TODO: drop pandas dep and use simple python primitives

class MetricsHistory:

    def __init__(self, name, savedir):
        self._history = None
        self._name    = name
        self._savedir = savedir

    def log_metrics(self, epoch, metrics, step = None, save = True):
        metrics = metrics or {}
        values  = copy.deepcopy(metrics)

        values['epoch'] = epoch
        values['time']  = pd.Timestamp.utcnow()

        if step is not None:
            values['step'] = step

        if self._history is None:
            self._history = pd.DataFrame([ values, ])
        else:
            df_values     = pd.DataFrame([ values, ])
            self._history = pd.concat(
                [ self._history, df_values ], ignore_index = True
            )

        if save:
            self.save(only_last_row = True)

    def get_save_path(self):
        return os.path.join(self._savedir, f'{LOG_FNAME_BASE}_{self._name}.csv')

    def save(self, only_last_row = False):
        path = self.get_save_path()

        if not only_last_row:
            self._history.to_csv(path, index = False, encoding = ENCODING)

        else:
            with open(path, 'ab') as f:
                exists = (f.tell() > 0)

                if not exists:
                    self._history.to_csv(f, index = False, encoding = ENCODING)
                else:
                    self._history.iloc[-1:].to_csv(
                        f, index = False, header = False, encoding = ENCODING
                    )

    def load(self):
        path = self.get_save_path()

        if os.path.exists(path):
            self._history = pd.read_csv(path, parse_dates = [ 'time', ])

    @property
    def values(self):
        return self._history

class MetricsHistoryDict:

    def __init__(self, savedir):
        self._history_dict = {}
        self._savedir      = savedir

    def log_metrics(self, name, epoch, metrics, step = None, save = True):
        # pylint: disable=too-many-arguments
        if name not in self._history_dict:
            self._history_dict[name] = MetricsHistory(name, self._savedir)

        self._history_dict[name].log_metrics(epoch, metrics, step, save)

    def save(self):
        for history in self._history_dict.values():
            history.save()

    def load(self):
        reg = re.compile(LOG_FNAME_RE)

        for fname in os.listdir(self._savedir):
            m = reg.match(fname)
            if not m:
                continue

            name = m.group(1)

            history = MetricsHistory(name, self._savedir)
            history.load()

            self._history_dict[name] = history

    def get_values(self, name):
        return self._history_dict[name].values

    def get_values_dict(self):
        return { k : v.values for (k, v) in self._history_dict.items() }

