# This file was modified by Dmitrii Torbunov <dtorbunov@bnl.gov>

import torch.nn as nn 
from ...core import register

CrossEntropyLoss = register(nn.CrossEntropyLoss)

