import argparse
import os

import torch
import pandas as pd

from evlearn.train.train import eval_epoch
from evlearn.eval.eval   import load_model, load_eval_dset

def parse_cmdargs():
    parser = argparse.ArgumentParser(description = 'Evaluate model metrics')

    parser.add_argument(
        'model',
        metavar  = 'MODEL',
        help     = 'model directory',
        type     = str,
    )

    parser.add_argument(
        '-e', '--epoch',
        default  = None,
        dest     = 'epoch',
        help     = 'epoch',
        type     = int,
    )

    parser.add_argument(
        '--device',
        choices  = [ 'cuda', 'cpu' ],
        default  = 'cuda',
        dest     = 'device',
        help     = 'device to use for evaluation (cuda/cpu)',
        type     = str,
    )

    parser.add_argument(
        '--data-path',
        default  = None,
        dest     = 'data_path',
        help     = 'path to the new dataset to evaluate',
        type     = str,
    )

    parser.add_argument(
        '--split',
        default  = 'test',
        dest     = 'split',
        help     = 'dataset split',
        type     = str,
    )

    parser.add_argument(
        '--steps',
        default  = None,
        dest     = 'steps',
        help     = 'steps for evaluation',
        type     = int,
    )

    parser.add_argument(
        '--batch-size',
        default  = None,
        dest     = 'batch_size',
        help     = 'batch size for evaluation',
        type     = int,
    )

    parser.add_argument(
        '--workers',
        default  = None,
        dest     = 'workers',
        help     = 'number of workers to use for evaluation',
        type     = int,
    )


    return parser.parse_args()

def make_eval_directory(model, savedir, mkdir = True):
    result = os.path.join(savedir, 'evals')

    if model.current_epoch is None:
        result = os.path.join(result, 'final')
    else:
        result = os.path.join(result, f'epoch_{model.current_epoch}')

    if mkdir:
        os.makedirs(result, exist_ok = True)

    return result

def save_metrics(evaldir, data_path, split, steps, metrics):
    fname = 'metrics'

    if data_path is not None:
        fname += f'_path({data_path})'

    fname += f'_split({split})'

    if steps is not None:
        fname += f'_nb({steps})'

    fname += '.csv'
    path   = os.path.join(evaldir, fname)

    df = pd.Series(metrics).to_frame().T
    df.to_csv(path, index = False)

def main():
    cmdargs = parse_cmdargs()

    args, model = load_model(
        cmdargs.model, epoch = cmdargs.epoch, device = cmdargs.device
    )

    evaldir     = make_eval_directory(model, cmdargs.model)
    data_config = args.data.eval

    if cmdargs.batch_size is not None:
        data_config.batch_size = cmdargs.batch_size

    if cmdargs.workers is not None:
        data_config.workers = cmdargs.workers

    if cmdargs.data_path is not None:
        data_config.dataset['path'] = cmdargs.data_path

    dl = load_eval_dset(args, split = cmdargs.split)

    with torch.inference_mode():
        metrics = eval_epoch(
            dl, model, title = 'Evaluation', steps_per_epoch = cmdargs.steps
        )

    save_metrics(
        evaldir, cmdargs.data_path, cmdargs.split, cmdargs.steps, metrics.get()
    )

if __name__ == '__main__':
    main()

