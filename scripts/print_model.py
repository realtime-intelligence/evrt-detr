#!/usr/bin/env python
import sys

import torch

from evlearn.train.train import eval_epoch
from evlearn.eval.eval   import load_model, load_eval_dset

path = sys.argv[1]

args, model = load_model(path, epoch = -1, device = 'cpu')

for name, net in model._nets.items():
    num_params = 0

    for param in net.parameters():
        num_params += param.numel()

    print(net)

    print(
        '[Network %s] Total number of parameters : %.3f M' % (
            name, num_params / 1e6
        )
    )

import IPython
IPython.embed()

