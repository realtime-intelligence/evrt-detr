import argparse
from itertools import islice
import os

import torch
import tqdm
import pandas as pd

from evlearn.bundled.leanbase.base.metrics import Metrics

from evlearn.train.train import infer_steps
from evlearn.eval.eval   import load_model, load_eval_dset

DTYPE_MAP = {
    'float32'  : torch.float32,
    'float16'  : torch.float16,
    'bfloat16' : torch.bfloat16,
}

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

    parser.add_argument(
        '--dtype',
        choices  = [ 'float32', 'float16', 'bfloat16' ],
        default  = 'float32',
        dest     = 'dtype',
        help     = 'data type to use for evaluation',
    )

    parser.add_argument(
        '--float32-precision',
        choices  = [ 'highest', 'high', 'medium' ],
        default  = None,
        dest     = 'float32_precision',
        help     = 'set float32 precision',
    )

    parser.add_argument(
        '--compile',
        action   = 'store_true',
        dest     = 'compile',
        help     = 'compile model',
    )

    parser.add_argument(
        '--compile-fullgraph',
        action   = 'store_true',
        dest     = 'compile_fullgraph',
        help     = 'discover and compile all possible graph paths',
    )

    parser.add_argument(
        '--compile-dynamic',
        action   = 'store_true',
        dest     = 'compile_dynamic',
        help     = 'use dynamic shape tracing',
    )

    # pylint: disable=protected-access
    parser.add_argument(
        '--compile-backend',
        choices  = torch._dynamo.list_backends(),
        dest     = 'compile_backend',
        help     = 'compilation backend',
        default  = 'inductor',
    )

    parser.add_argument(
        '--compile-mode',
        choices  = [
            'default', 'reduce-overhead', 'max-autotune',
            'max-autotune-no-cudagraphs'
        ],
        dest     = 'compile_mode',
        help     = 'compilation mode',
        default  = 'default',
    )

    parser.add_argument(
        '--fuse-postproc',
        action   = 'store_true',
        dest     = 'fuse_postproc',
        help     = 'fuse postprocessor with models',
    )

    parser.add_argument(
        '--eval-flops',
        action   = 'store_true',
        dest     = 'eval_flops',
        help     = 'evaluate flops and exit',
    )

    return parser.parse_args()

@torch.no_grad()
def eval_flops(dl, inference_engine, torch_model):
    print("Evaluating FLOPS")
    # pylint: disable=import-outside-toplevel
    from fvcore.nn import FlopCountAnalysis

    temporal_batch = next(iter(dl))
    inference_engine.set_inputs(temporal_batch)

    for batch in inference_engine.data_it():
        # batch: [ data0, data1, ..., labels ]
        flops = FlopCountAnalysis(torch_model, *batch[:-1])
        total = flops.total()

        print(f"FLOPS: {total}")
        print(f"GFLOPS: {total / 10**9}")

        break

def benchmark_eval_epoch(
    dl_test, inference_engine, torch_model, steps_per_epoch
):
    inference_engine.eval_epoch_start()

    steps   = infer_steps(dl_test, steps_per_epoch)
    progbar = tqdm.tqdm(desc = 'Bench', total = steps, dynamic_ncols = True)
    metrics = Metrics(prefix = 'bench_')
    times   = []

    for temporal_batch in islice(dl_test, steps):
        inference_engine.set_inputs(temporal_batch)

        for batch in inference_engine.data_it():
            # batch: [ data0, data1, ..., labels ]
            start = torch.cuda.Event(enable_timing = True)
            end   = torch.cuda.Event(enable_timing = True)

            torch.cuda.synchronize()

            start.record()
            outputs = torch_model(*batch[:-1])
            end.record()

            torch.cuda.synchronize()

            curr_metrics = inference_engine.eval_step_standanlone(
                outputs, batch[-1]
            )

            metrics.update(curr_metrics)
            times.append((len(batch[0]), start.elapsed_time(end)))

        progbar.set_postfix(metrics.get(), refresh = False)
        progbar.update()

    metrics.update(inference_engine.eval_epoch_end())

    progbar.set_postfix(metrics.get(), refresh = True)
    progbar.close()

    return metrics, times

def make_eval_directory(model, savedir, mkdir = True):
    result = os.path.join(savedir, 'evals')

    if model.current_epoch is None:
        result = os.path.join(result, 'final')
    else:
        result = os.path.join(result, f'epoch_{model.current_epoch}')

    if mkdir:
        os.makedirs(result, exist_ok = True)

    return result

def construct_file_suffix(cmdargs, compile_kwargs):
    fname = 'standalone'

    fname += f'_dtype({cmdargs.dtype})'
    fname += f'_compile({compile_kwargs})'

    if cmdargs.float32_precision is not None:
        fname += f'_fp32({cmdargs.float32_precision})'

    if cmdargs.batch_size is not None:
        fname += f'_bs({cmdargs.batch_size})'

    if cmdargs.data_path is not None:
        fname += f'_path({cmdargs.data_path})'

    fname += f'_split({cmdargs.split})'

    if cmdargs.steps is not None:
        fname += f'_nb({cmdargs.steps})'

    return fname

def save_metrics(metrics, evaldir, fname_suffix):
    fname = f'metrics_{fname_suffix}.csv'
    path  = os.path.join(evaldir, fname)

    df = pd.Series(metrics).to_frame().T
    df.to_csv(path, index = False)

def save_times(times, evaldir, fname_suffix):
    fname = f'times_{fname_suffix}.csv'
    path  = os.path.join(evaldir, fname)

    df = pd.DataFrame(times, columns = [ 'batch_size', 'time' ])
    df.to_csv(path, index = False)

def print_df_times(df):
    time_per_event = df['time'] / df['batch_size']

    total_n_events = df['batch_size'].values.sum()
    total_time     = df['time'].values.sum()

    time_per_event = total_time / total_n_events

    print(f'  - total time {total_time} [ms]')
    print(f'  - n events   {total_n_events}')
    print(f'  - average time per event {time_per_event} [ms]')
    print(f'  - average batch size     {df["batch_size"].mean()}')
    print(f'  - average time per batch {df["time"].mean()} [ms]')
    print(f'  - median  time per batch {df["time"].median()} [ms]')
    print(f'  - stdev   time per batch {df["time"].std()} [ms]')

def print_times(times, n_warmup = 5):
    df      = pd.DataFrame(times, columns = [ 'batch_size', 'time' ])
    df_warm = df[n_warmup:]

    print('All iterations')
    print_df_times(df)

    print(f'Warm (>{n_warmup}) iterations')
    print_df_times(df_warm)

def parse_compile_kwargs(cmdargs):
    if not cmdargs.compile:
        return None

    return {
        'fullgraph' : cmdargs.compile_fullgraph,
        'dynamic'   : cmdargs.compile_dynamic,
        'backend'   : cmdargs.compile_backend,
        'mode'      : cmdargs.compile_mode,
    }

def main():
    cmdargs = parse_cmdargs()

    args, model = load_model(
        cmdargs.model, epoch = cmdargs.epoch, device = cmdargs.device
    )

    model.to_dtype(DTYPE_MAP[cmdargs.dtype])
    model.eval()

    inference_engine = model.construct_inference_engine(cmdargs.fuse_postproc)
    compile_kwargs   = parse_compile_kwargs(cmdargs)
    evaldir          = make_eval_directory(model, cmdargs.model)
    data_config      = args.data.eval

    if cmdargs.float32_precision is not None:
        torch.set_float32_matmul_precision(cmdargs.float32_precision)

    if cmdargs.batch_size is not None:
        data_config.batch_size = cmdargs.batch_size

    if cmdargs.eval_flops is not None:
        data_config.batch_size = 1

    if cmdargs.workers is not None:
        data_config.workers = cmdargs.workers

    if cmdargs.data_path is not None:
        data_config.dataset['path'] = cmdargs.data_path

    torch_model = inference_engine.construct_torch_model()
    dl          = load_eval_dset(args, split = cmdargs.split)

    if cmdargs.eval_flops:
        eval_flops(dl, inference_engine, torch_model)
        return

    if compile_kwargs is not None:
        torch_model.compile(**compile_kwargs)

    metrics, times = benchmark_eval_epoch(
        dl, inference_engine, torch_model, cmdargs.steps
    )

    fname_suffix = construct_file_suffix(cmdargs, compile_kwargs)

    save_metrics(metrics.get(), evaldir, fname_suffix)
    save_times  (times,         evaldir, fname_suffix)

    print_times(times)

if __name__ == '__main__':
    main()

