# pylint: disable=not-callable
import itertools

import torch
from torchvision.transforms import Pad

from evlearn.bundled.leanbase.base.named_dict  import NamedDict
from evlearn.bundled.leanbase.torch.select     import select_optimizer
from evlearn.bundled.leanbase.torch.schedulers import select_scheduler

from evlearn.nn.backbone import construct_backbone

from evlearn.bundled.rtdetr_pytorch.zoo.rtdetr.hybrid_encoder import (
    HybridEncoder
)
from evlearn.bundled.rtdetr_pytorch.zoo.rtdetr.rtdetr_decoder import (
    RTDETRTransformer
)
from evlearn.bundled.rtdetr_pytorch.zoo.rtdetr.rtdetr_criterion import (
    SetCriterion
)
from evlearn.bundled.rtdetr_pytorch.zoo.rtdetr.rtdetr_postprocessor import (
    RTDETRPostProcessor
)
from evlearn.bundled.rtdetr_pytorch.zoo.rtdetr.matcher import HungarianMatcher

from evlearn.eval.evaluator import construct_evaluator

from evlearn.torch.funcs        import clip_gradients, update_ema_model
from evlearn.torch.rtdetr_funcs import (
    convert_labels_torchvision_to_rtdetr,
    convert_prediction_postproc_rtdetr_to_torchvision
)

from evlearn.inference_engines.frame_detection_rtdetr import (
    InferenceEngineFrameDetectionRTDETR
)

from .model_base import ModelBase

class FrameDetectionRTDETR(ModelBase):
    # pylint: disable=too-many-instance-attributes
    # pylint: disable=too-many-ancestors

    def __init__(
        self, config, device, init_train, savedir, dtype,
        rtdetr_postproc_kwargs,
        evaluator    = None,
        pad          = None,
        ema_momentum = None,
        grad_clip    = None
    ):
        # pylint: disable=too-many-arguments
        self._frame_shape = tuple(config.data.train.shapes[0])
        self._ema_mom     = ema_momentum

        super().__init__(config, device, init_train, savedir, dtype)

        self._grad_clip = grad_clip or {}
        self._evaluator = construct_evaluator(evaluator)

        self.pad = None

        if pad is not None:
            self.pad = Pad(pad)

        self.postprocessor  \
            = RTDETRPostProcessor(**rtdetr_postproc_kwargs).to(device)

    def _setup_data(self):
        return NamedDict(
            'frame', 'labels', 'labels_rtdetr', 'preds',
            'backbone_features',
            'encoder_output',
            'decoder_output',
        )

    def _setup_losses(self):
        self.matcher   = HungarianMatcher(
            **self._config.losses['matcher']
        ).to(self._device)

        self.criterion = SetCriterion(
            self.matcher,
            **self._config.losses['criterion']
        ).to(self._device)

        return NamedDict('loss_vfl', 'loss_bbox', 'loss_giou')

    def _setup_nets(self):
        nets = {}

        nets['backbone'] = construct_backbone(
            self._config.nets['backbone'], self._frame_shape, self._device
        )
        nets['encoder'] = HybridEncoder(**self._config.nets['encoder'].model)
        nets['decoder'] \
            = RTDETRTransformer(**self._config.nets['decoder'].model)

        if (self._ema_mom is not None) and (self._ema_mom > 0):
            nets['ema_backbone'] = construct_backbone(
                self._config.nets['backbone'], self._frame_shape, self._device
            )
            nets['ema_encoder'] \
                = HybridEncoder(**self._config.nets['encoder'].model)
            nets['ema_decoder'] \
                = RTDETRTransformer(**self._config.nets['decoder'].model)

        return NamedDict(**{
            k : v.to(self._device) for (k, v) in nets.items()
        })

    def _setup_optimizers(self):
        optimizer = select_optimizer(
            itertools.chain(
                self._nets.backbone.parameters(),
                self._nets.encoder .parameters(),
                self._nets.decoder .parameters(),
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
            self._data.labels_rtdetr = convert_labels_torchvision_to_rtdetr(
                self._data.labels, self._device, self._frame_shape[1:]
            )

        backbone = self._nets.backbone
        encoder  = self._nets.encoder
        decoder  = self._nets.decoder

        if (not self._train_state) and self._ema_mom:
            backbone = self._nets.ema_backbone
            encoder  = self._nets.ema_encoder
            decoder  = self._nets.ema_decoder

        self._data.backbone_features = backbone(frame)
        self._data.encoder_output    = encoder(self._data.backbone_features)

        if self._train_state:
            self._data.decoder_output \
                = decoder(self._data.encoder_output, self._data.labels_rtdetr)
        else:
            self._data.decoder_output = decoder(self._data.encoder_output)

    def update_ema_states(self):
        if (self._ema_mom is None) or (self._ema_mom == 0):
            return

        update_ema_model(
            self._nets.ema_backbone, self._nets.backbone, self._ema_mom
        )
        update_ema_model(
            self._nets.ema_encoder, self._nets.encoder, self._ema_mom
        )
        update_ema_model(
            self._nets.ema_decoder, self._nets.decoder, self._ema_mom
        )

    def train_step(self):
        self.train()
        self.criterion.train()

        self._optimizers.main.zero_grad(set_to_none = True)

        self.forward()

        loss_dict = self.criterion(
            self._data.decoder_output, self._data.labels_rtdetr
        )

        self._losses.loss_vfl  = loss_dict['loss_vfl']
        self._losses.loss_bbox = loss_dict['loss_bbox']
        self._losses.loss_giou = loss_dict['loss_giou']

        loss = sum(loss_dict.values())
        loss.backward()

        clip_gradients(self._optimizers.main, **self._grad_clip)
        self._optimizers.main.step()

        self.step_scheduler(self._schedulers.main, event = 'batch')
        self.update_ema_states()

    def get_current_losses(self):
        return {
            k : float(v.detach().cpu().mean())
                for (k, v) in self._losses.items() if v is not None
        }

    @torch.no_grad()
    def eval_step(self):
        # pylint: disable=too-many-locals
        self.eval()
        self.criterion.eval()

        self.forward()

        orig_sizes = torch.stack(
            [ x['orig_size'] for x in self._data.labels_rtdetr ],
            dim = 0
        )

        rtdetr_preds = self.postprocessor(
            self._data.decoder_output, orig_sizes
        )

        self._evaluator.append(
            rtdetr_preds, self._data.labels, preds_format = 'rtdetr'
        )

        return {}

    @torch.no_grad()
    def encode_inputs(self):
        self.eval()
        self.forward(use_labels = False)

        return self._data.decoder_output

    @torch.no_grad()
    def predict_step(self):
        self.eval()
        self.criterion.eval()

        self.forward()

        orig_sizes = torch.stack(
            [ x['orig_size'] for x in self._data.labels_rtdetr ],
            dim = 0
        )

        rtdetr_preds = self.postprocessor(
            self._data.decoder_output, orig_sizes
        )

        self._data.preds = convert_prediction_postproc_rtdetr_to_torchvision(
            rtdetr_preds, self._frame_shape[1:]
        )

        return (self._data.frame, self._data.labels, self._data.preds)

    def construct_inference_engine(self, fuse_postproc = False):
        return InferenceEngineFrameDetectionRTDETR(
            self._device, self._dtype,
            backbone      = self._nets.ema_backbone,
            encoder       = self._nets.ema_encoder,
            decoder       = self._nets.ema_decoder,
            postproc      = self.postprocessor,
            evaluator     = self._evaluator,
            fuse_postproc = fuse_postproc,
            pad           = self.pad,
            frame_shape   = self._frame_shape,
        )

    def epoch_end(self, metrics):
        self.step_scheduler(self._schedulers.main, 'epoch', metrics)

    def eval_epoch_end(self):
        metrics = self._evaluator.evaluate()
        self._evaluator.reset()

        return metrics

