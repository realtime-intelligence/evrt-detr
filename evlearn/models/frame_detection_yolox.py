# pylint: disable=not-callable
import copy
import itertools

import torch
from torch import nn

from torchvision.transforms import Pad

from evlearn.bundled.leanbase.base.named_dict import NamedDict

from evlearn.bundled.leanbase.torch.select import select_optimizer
from evlearn.bundled.leanbase.torch.schedulers import select_scheduler

from evlearn.nn.backbone import construct_backbone

from evlearn.bundled.yolox.models      import YOLOXHead
from evlearn.bundled.yolox.utils.boxes import postprocess

from evlearn.eval.evaluator    import construct_evaluator
from evlearn.torch.funcs       import clip_gradients, update_ema_model
from evlearn.torch.yolox_funcs import convert_labels_torchvision_to_yolox

from .model_base import ModelBase

def yolox_initializer(
    backbone, head, bn_eps = 1e-3, bn_mom = 0.03, biases = 1e-2
):
    # cf. bundled/yolox/exp/yolox_base.py

    def bn_mod(m):
        if isinstance(m, nn.BatchNorm2d):
            m.eps      = bn_eps
            m.momentum = bn_mom

    backbone.apply(bn_mod)
    head    .apply(bn_mod)

    head.initialize_biases(biases)

class FrameDetectionYoloX(ModelBase):
    # pylint: disable=too-many-instance-attributes
    # pylint: disable=too-many-ancestors

    def __init__(
        self, config, device, init_train, savedir, dtype,
        yolox_postproc_kwargs,
        evaluator    = None,
        pad          = None,
        ema_momentum = None,
        grad_clip    = None,
        yolox_init   = True,
    ):
        # pylint: disable=too-many-arguments
        self._frame_shape = tuple(config.data.train.shapes[0])
        self._ema_mom     = ema_momentum
        self._yolox_init  = yolox_init

        super().__init__(config, device, init_train, savedir, dtype)

        self._grad_clip = grad_clip or {}
        self._evaluator = construct_evaluator(evaluator)

        self.pad = None

        if pad is not None:
            self.pad = Pad(pad)

        self._post_kwargs = yolox_postproc_kwargs

    def _setup_data(self):
        return NamedDict(
            'frame', 'backbone_features', 'labels', 'labels_yolox', 'preds'
        )

    def _setup_losses(self):
        return NamedDict(
            'iou_loss',
            'conf_loss',
            'cls_loss',
            'l1_loss',
            'num_fg',
        )

    def _setup_nets(self):
        nets = {}

        nets['backbone'] = construct_backbone(
            self._config.nets['backbone'], self._frame_shape, self._device
        )
        nets['head'] = YOLOXHead(**self._config.nets['head'].model)

        if self._yolox_init:
            yolox_initializer(nets['backbone'], nets['head'])

        if (self._ema_mom is not None) and (self._ema_mom > 0):
            nets['ema_backbone'] = copy.deepcopy(nets['backbone'])
            nets['ema_head']     = copy.deepcopy(nets['head'])

        return NamedDict(**{
            k : v.to(self._device) for (k, v) in nets.items()
        })

    def _setup_optimizers(self):
        optimizer = select_optimizer(
            itertools.chain(
                self._nets.backbone.parameters(),
                self._nets.head    .parameters(),
            ),
            self._config.optimizers['main']
        )

        return NamedDict(main = optimizer)

    def _setup_schedulers(self):
        sched = select_scheduler(
            self._optimizers.main, self._config.schedulers['main'],
            compose = True
        )

        return NamedDict(main = sched)

    def _set_inputs(self, data):
        (frame, labels) = data

        if self.pad is not None:
            frame, labels = self.pad(frame, labels)

        self._data.frame  = frame.to(
            self._device, dtype = self._dtype, non_blocking = True
        )

        self._data.labels = labels

    def forward(self, use_labels = True):
        # frame : (N, C, H, W)
        frame = self._data.frame

        if use_labels:
            self._data.labels_yolox = convert_labels_torchvision_to_yolox(
                self._data.labels, self._device, self._dtype,
                self._frame_shape[1:]
            )

        backbone = self._nets.backbone

        if (not self._train_state) and self._ema_mom:
            backbone = self._nets.ema_backbone

        self._data.backbone_features = backbone(frame)

    def update_ema_states(self):
        if (self._ema_mom is None) or (self._ema_mom == 0):
            return

        update_ema_model(
            self._nets.ema_backbone, self._nets.backbone, self._ema_mom
        )
        update_ema_model(
            self._nets.ema_head, self._nets.head, self._ema_mom
        )

    def train_step(self):
        self.train()
        self._optimizers.main.zero_grad(set_to_none = True)

        self.forward()

        # predictions : (N_samples_with_labels, anchor, PREDICTION)
        # PREDICTION  : (reg, obj, cls)
        #             = (  4,   1, num_classes)
        loss, iou_loss, conf_loss, cls_loss, l1_loss, num_fg = self._nets.head(
            self._data.backbone_features,
            self._data.labels_yolox,
            self._data.frame,
        )

        self._losses.iou_loss  = iou_loss
        self._losses.conf_loss = conf_loss
        self._losses.cls_loss  = cls_loss
        self._losses.l1_loss   = l1_loss
        self._losses.num_fg    = num_fg

        loss.backward()

        clip_gradients(self._optimizers.main, **self._grad_clip)
        self._optimizers.main.step()

        self.step_scheduler(self._schedulers.main, event = 'batch')
        self.update_ema_states()

    def get_current_losses(self):
        return {
            k : float(v)
                for (k, v) in self._losses.items() if v is not None
        }

    @torch.no_grad()
    def eval_step(self):
        # pylint: disable=too-many-locals
        self.eval()
        self.forward()

        head = self._nets.head
        if self._ema_mom:
            head = self._nets.ema_head

        self._data.preds_yolox = head(self._data.backbone_features)

        preds = postprocess(
            prediction  = self._data.preds_yolox,
            num_classes = self._post_kwargs['num_classes'],
            conf_thre   = self._post_kwargs['confidence_threshold'],
            nms_thre    = self._post_kwargs['nms_threshold']
        )

        self._evaluator.append(
            preds, self._data.labels, preds_format = 'yolox'
        )

        return {}

    def set_postproc_args(
        self, nms_threshold = None, confidence_threshold = None
    ):
        if nms_threshold is not None:
            self._post_kwargs['nms_threshold'] = nms_threshold

        if confidence_threshold is not None:
            self._post_kwargs['confidence_threshold'] = confidence_threshold

    @torch.no_grad()
    def encode_inputs(self):
        raise NotImplementedError

    @torch.no_grad()
    def predict_step(self):
        raise NotImplementedError

    def to_standalone_model(self, fuse_postproc = False):
        raise NotImplementedError

    def epoch_end(self, metrics):
        self.step_scheduler(self._schedulers.main, 'epoch', metrics)

    def eval_epoch_end(self):
        metrics = self._evaluator.evaluate()
        self._evaluator.reset()

        return metrics

