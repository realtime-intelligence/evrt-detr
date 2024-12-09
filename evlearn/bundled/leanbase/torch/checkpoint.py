import os
import re
import torch

CHECKPOINTS_DIR = 'checkpoints'

def find_last_checkpoint_epoch(savedir):
    root = os.path.join(savedir, CHECKPOINTS_DIR)
    if not os.path.exists(root):
        return -1

    r = re.compile(r'^epoch_(\d+)$')
    last_epoch = -1

    for fname in os.listdir(root):
        m = r.match(fname)
        if m:
            epoch      = int(m.group(1))
            last_epoch = max(last_epoch, epoch)

    return last_epoch

def get_save_path(savedir, name, epoch, mkdir = False):
    if epoch is None:
        root = savedir
    else:
        root = os.path.join(savedir, CHECKPOINTS_DIR, f'epoch_{epoch:04d}')

    fname  = f'{name}.pth'
    result = os.path.join(root, fname)

    if mkdir:
        os.makedirs(root, exist_ok = True)

    return result

def save(named_dict, savedir, prefix, epoch = None):
    for (k,v) in named_dict.items():
        if v is None:
            continue

        save_path = get_save_path(
            savedir, prefix + '_' + k, epoch, mkdir = True
        )

        if isinstance(v, torch.nn.DataParallel):
            torch.save(v.module.state_dict(), save_path)
        else:
            torch.save(v.state_dict(), save_path)

def load(named_dict, savedir, prefix, epoch, device):
    for (k,v) in named_dict.items():
        if v is None:
            continue

        load_path = get_save_path(
            savedir, prefix + '_' + k, epoch, mkdir = False
        )

        if isinstance(v, torch.nn.DataParallel):
            v.module.load_state_dict(
                torch.load(load_path, map_location = device)
            )
        else:
            try:
                v.load_state_dict(torch.load(load_path, map_location = device))
            except RuntimeError as e:
                print(f"Failed to load: {e}. Trying non-strict load")
                v.load_state_dict(
                    torch.load(load_path, map_location = device),
                    strict = False
                )

