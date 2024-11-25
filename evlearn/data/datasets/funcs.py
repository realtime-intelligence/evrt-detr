import logging

import numpy as np
import torch

LOGGER = logging.getLogger('evlearn.data.datasets.funcs')

def cantor_pairing(x, y, mod = (1 << 32)):
    # https://stackoverflow.com/questions/919612/mapping-two-integers-to-one-in-a-unique-and-deterministic-way
    # https://en.wikipedia.org/wiki/Pairing_function#Cantor_pairing_function
    result = (x + y) * (x + y + 1) // 2 + y

    if mod is not None:
        result = result % mod

    return result

def nan_guard(data, msg = ''):
    # pylint: disable=forgotten-debug-statement
    # pylint: disable=import-outside-toplevel

    if data is None:
        print(f"NONE (empty) data detected: {msg}")
        return

    if isinstance(data, (list, tuple)):
        for idx,d in enumerate(data):
            nan_guard(d, msg + f'. List[{idx}]')
        return

    if isinstance(data, dict):
        for k, d in data.items():
            nan_guard(d, msg + f'. Dict[{k}]')
        return

    if isinstance(data, np.ndarray) and np.all(np.isfinite(data)):
        return

    if data.isfinite().all():
        return

    print(f"Nan detected: {msg}")
    breakpoint()

    import IPython
    IPython.embed()

def has_nans(data, msg = ''):
    if (data is None) or isinstance(data, (int, float, bool)):
        return False

    if isinstance(data, (list, tuple)):
        for idx,d in enumerate(data):
            if has_nans(d, f'List[{idx}]. ' + msg):
                return True

    elif isinstance(data, dict):
        for k, d in data.items():
            if has_nans(d, f'Dict[{k}]. ' + msg):
                return True

    elif isinstance(data, np.ndarray):
        nanmask = ~np.isfinite(data)
        if np.any(nanmask):
            n_nan = np.count_nonzero(nanmask)
            n_tot = nanmask.size

            LOGGER.warning(
                'NaN values in an np.array (%d out of %d). %s',
                n_nan, n_tot, msg
            )

            return True

    elif isinstance(data, torch.Tensor):
        nanmask = ~data.isfinite()
        if nanmask.any():
            n_nan = torch.count_nonzero(nanmask)
            n_tot = torch.numel(nanmask)

            LOGGER.warning(
                'NaN values in a torch.Tensor (%d out of %d). %s',
                n_nan, n_tot, msg
            )

            return True

    else:
        LOGGER.warning(
            'Unknown how to check for NaN values in %s', type(data)
        )

    return False

def nan_sanitize(data, msg = '', fill = 0):
    if (data is None) or isinstance(data, (int, float, bool)):
        return

    if isinstance(data, (list, tuple)):
        for idx,d in enumerate(data):
            nan_sanitize(d, f'List[{idx}]. ' + msg)

    elif isinstance(data, dict):
        for k, d in data.items():
            nan_sanitize(d, f'Dict[{k}]. ' + msg)

    elif isinstance(data, np.ndarray):
        nanmask = ~np.isfinite(data)
        if np.any(nanmask):
            n_nan = np.count_nonzero(nanmask)
            n_tot = nanmask.size

            LOGGER.warning(
                'Sanitizing NaN values in an np.array (%d out of %d). %s',
                n_nan, n_tot, msg
            )

            data[nanmask] = fill

    elif isinstance(data, torch.Tensor):
        nanmask = ~data.isfinite()
        if nanmask.any():
            n_nan = torch.count_nonzero(nanmask)
            n_tot = torch.numel(nanmask)

            LOGGER.warning(
                'Sanitizing NaN values in a torch.Tensor (%d out of %d). %s',
                n_nan, n_tot, msg
            )

            data[nanmask] = fill

    else:
        LOGGER.warning(
            'Unknown how to sanitize NaN values in %s', type(data)
        )

