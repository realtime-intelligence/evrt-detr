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
        '--data-name',
        default  = None,
        dest     = 'data_name',
        help     = 'name of the dataset to use',
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

def save_metrics(evaldir, data_name, data_path, split, steps, metrics):
    # pylint: disable=too-many-arguments
    fname = f'metrics_data({data_name})'

    if data_path is not None:
        fname += f'_path({data_path})'

    fname += f'_split({split})'

    if steps is not None:
        fname += f'_nb({steps})'

    fname += '.csv'
    path   = os.path.join(evaldir, fname)

    df = pd.Series(metrics).to_frame().T
    df.to_csv(path, index = False)

def eval_single_dataset(
    model, args, data_name, data_config, split, steps, evaldir, batch_size,
    workers, data_path
):
    # pylint: disable=too-many-arguments
    if batch_size is not None:
        data_config.batch_size = batch_size

    if workers is not None:
        data_config.workers = workers

    if data_path is not None:
        data_config.dataset['path'] = data_path

    args.config.data.eval = { data_name : data_config }
    dl = load_eval_dset(args, split = split)

    metrics = eval_epoch(
        dl, model,
        title           = f'Evaluation: {data_name}',
        steps_per_epoch = steps
    )

    save_metrics(evaldir, data_name, data_path, split, steps, metrics.get())

def main():
    cmdargs = parse_cmdargs()

    args, model = load_model(
        cmdargs.model, epoch = cmdargs.epoch, device = cmdargs.device
    )
    data_config_dict = args.config.data.eval
    assert isinstance(data_config_dict, dict)

    evaldir = make_eval_directory(model, cmdargs.model)

    if cmdargs.data_name is not None:
        datasets = [ cmdargs.data_name ]
    else:
        datasets = list(sorted(data_config_dict.keys()))

    with torch.inference_mode():
        for name in datasets:
            eval_single_dataset(
                model, args, name, data_config_dict[name],
                cmdargs.split, cmdargs.steps, evaldir, cmdargs.batch_size,
                cmdargs.workers, cmdargs.data_path
            )

if __name__ == '__main__':
    main()

