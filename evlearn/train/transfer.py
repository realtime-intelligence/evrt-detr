# This file is adapted from uvcgan2 project.
# Copyright (c) 2021-2023, The LS4GAN Project Developers.
# Source: https://github.com/LS4GAN/uvcgan2
# Licensed under BSD-2-Clause License

import os
import logging

import torch

from evlearn.config.args import Args
from evlearn.consts      import ROOT_OUTDIR
from evlearn.models      import construct_model

LOGGER = logging.getLogger('evlearn.train')

# pylint: disable=protected-access
def load_base_model(model, transfer_config):
    if transfer_config.use_last_checkpoint:
        last_epoch = model.find_last_checkpoint_epoch()

        if last_epoch <= 0:
            raise RuntimeError("Failed to find transfer model checkpoints.")

        LOGGER.warning(
            "Loading transfer model from a checkpoint '%d'", last_epoch
        )

        model.load(epoch = last_epoch)

    else:
        model.load(epoch = None)

def get_base_model(transfer_config, device, dtype):
    base_path = os.path.join(ROOT_OUTDIR, transfer_config.base_model)
    base_args = Args.load(base_path)

    model = construct_model(
        base_args.config, device, dtype,
        init_train = transfer_config.load_train_state,
        savedir    = base_args.savedir
    )

    load_base_model(model, transfer_config)

    return model

def transfer_parameters_basic(model, base_model, transfer_config):
    for (dst, src) in transfer_config.transfer_map.items():
        model._nets[dst].load_state_dict(
            base_model._nets[src].state_dict(), strict = transfer_config.strict
        )

def transfer_parameters_ignore_mismatching(model, base_model, transfer_config):
    for (dst, src) in transfer_config.transfer_map.items():
        src_state_dict = base_model._nets[src].state_dict()
        dst_state_dict = model._nets[dst].state_dict()

        if transfer_config.strict:
            assert set(src_state_dict.keys()) == set(dst_state_dict.keys())

        for (k, dst_params) in dst_state_dict.items():
            if k not in src_state_dict:
                continue

            src_params = src_state_dict[k]
            if src_params.shape != dst_params.shape:
                print(
                    f"Mismatching parameter shapes for {k}:"
                    f" {src_params.shape} vs {dst_params.shape}"
                )
                src_state_dict.pop(k)

        model._nets[dst].load_state_dict(src_state_dict, strict = False)

def transfer_parameters_expand_mismatching(model, base_model, transfer_config):
    for (dst, src) in transfer_config.transfer_map.items():
        src_state_dict = base_model._nets[src].state_dict()
        dst_state_dict = model._nets[dst].state_dict()

        if transfer_config.strict:
            assert set(src_state_dict.keys()) == set(dst_state_dict.keys())

        for (k, dst_params) in dst_state_dict.items():
            if k not in src_state_dict:
                continue

            src_params = src_state_dict[k]
            if src_params.shape != dst_params.shape:
                print(
                    f"Mismatching parameter shapes for {k}:"
                    f" {src_params.shape} vs {dst_params.shape}"
                )
                pad_sizes = [
                    d - s
                        for (s, d) in zip(src_params.shape, dst_params.shape)
                ]
                assert all(x >= 0 for x in pad_sizes), pad_sizes

                # Need to be reversed due to torch
                padding = []
                for s in reversed(pad_sizes):
                    padding += [ 0, s ]

                matching_src_params = torch.nn.functional.pad(
                    src_params, padding
                )
                print(f"Padded {k} to {matching_src_params.shape}")

                src_state_dict[k] = matching_src_params

        model._nets[dst].load_state_dict(src_state_dict, strict = False)

def transfer_parameters(model, base_model, transfer_config):
    if transfer_config.fuzzy is None:
        transfer_parameters_basic(model, base_model, transfer_config)

    elif transfer_config.fuzzy == 'ignore-mismatching-shapes':
        transfer_parameters_ignore_mismatching(
            model, base_model, transfer_config
        )

    elif transfer_config.fuzzy == 'expand-mismatching-shapes':
        transfer_parameters_expand_mismatching(
            model, base_model, transfer_config
        )

    else:
        raise ValueError(
            f"Unknown fuzzinness setting: '{transfer_config.fuzzy}'"
        )

def transfer(model, transfer_config):
    if transfer_config is None:
        return

    LOGGER.info(
        "Initiating parameter transfer : '%s'", transfer_config.to_dict()
    )

    base_model = get_base_model(transfer_config, model._device, model._dtype)
    transfer_parameters(model, base_model, transfer_config)

