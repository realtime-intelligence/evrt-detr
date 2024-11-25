from itertools import islice
import logging

import torch
import tqdm

from evlearn.bundled.leanbase.base.metrics import Metrics
from evlearn.bundled.leanbase.base.metrics_history import MetricsHistoryDict

from evlearn.bundled.leanbase.torch.funcs import seed_everything

from evlearn.config.args import Args
from evlearn.data.data   import construct_data_loader

from evlearn.models import construct_model

from .transfer import transfer

LOGGER = logging.getLogger('evlearn.train')

def infer_steps(dl, steps_per_epoch):
    try:
        steps = len(dl)
    except TypeError:
        return steps_per_epoch

    if steps_per_epoch is not None:
        steps = min(steps, steps_per_epoch)

    return steps

def train_epoch(dl_train, model, title, steps_per_epoch):
    model.train()

    steps   = infer_steps(dl_train, steps_per_epoch)
    progbar = tqdm.tqdm(desc = title, total = steps, dynamic_ncols = True)
    metrics = Metrics()

    for batch in islice(dl_train, steps):
        model.set_inputs(batch)
        model.train_step()

        metrics.update(model.get_current_losses())

        progbar.set_postfix(metrics.get(), refresh = False)
        progbar.update()

    progbar.close()
    return metrics

def eval_epoch(dl_test, model, title, steps_per_epoch):
    model.eval()
    model.eval_epoch_start()

    steps   = infer_steps(dl_test, steps_per_epoch)
    progbar = tqdm.tqdm(
        desc = title, total = steps, dynamic_ncols = True
    )
    metrics = Metrics(prefix = 'val_')

    for batch in islice(dl_test, steps):
        model.set_inputs(batch)
        metrics.update(model.eval_step())

        progbar.set_postfix(metrics.get(), refresh = False)
        progbar.update()

    metrics.update(model.eval_epoch_end())

    progbar.set_postfix(metrics.get(), refresh = True)
    progbar.close()

    return metrics

def train(
    dl_train, dl_val, model, history_dict, epochs, checkpoint,
    val_interval, steps_per_train_epoch, steps_per_val_epoch, first_epoch = 1
):
    # pylint: disable=too-many-arguments
    print("Training...")

    for epoch in range(first_epoch, epochs + 1):
        title = f'Epoch {epoch} / {epochs}'

        model.epoch_start(epoch)

        metrics = train_epoch(dl_train, model, title, steps_per_train_epoch)
        history_dict.log_metrics('train', epoch, metrics.get())

        if epoch % val_interval == 0:
            metrics_val = eval_epoch(
                dl_val, model, 'Eval ' + title, steps_per_val_epoch
            )
            history_dict.log_metrics('val',   epoch, metrics_val.get())
            metrics = metrics.join(metrics_val)

        model.epoch_end(metrics)

        if epoch % checkpoint == 0:
            model.save(epoch)

    model.save(None)

def try_continue_train(model, history_dict):
    try:
        model.load(None)
        history_dict.load()

        LOGGER.info('Found fully trained model')

        return None
    except IOError:
        pass

    last_epoch = model.find_last_checkpoint_epoch()
    if last_epoch == -1:
        LOGGER.info("No checkpoints found.")
        return 1

    LOGGER.info("Found checkpoint '%d' . Continuing training", last_epoch)

    model.load(last_epoch)
    history_dict.load()

    return last_epoch + 1

def setup_and_train(args_dict, device = 'cuda', dtype = torch.float32):
    args = Args.from_args_dict(**args_dict)

    LOGGER.info('Starting training: ')
    LOGGER.info(args.config.to_json(indent = 4))

    seed_everything(args.config.seed)

    dl_train = construct_data_loader(args.config.data.train, split = 'train')
    dl_val   = construct_data_loader(args.config.data.eval,  split = 'val')

    model = construct_model(
        args.config, device, dtype, init_train = True, savedir = args.savedir,
    )

    history_dict = MetricsHistoryDict(args.savedir)
    first_epoch  = try_continue_train(model, history_dict)

    if first_epoch is None:
        return

    if first_epoch == 1:
        transfer(model, args.config.transfer)

    train(
        dl_train, dl_val, model, history_dict, args.config.epochs,
        args.checkpoint,
        args.config.val_interval,
        args.config.steps_per_train_epoch, args.config.steps_per_val_epoch,
        first_epoch
    )

