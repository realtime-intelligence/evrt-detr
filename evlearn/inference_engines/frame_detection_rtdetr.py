# pylint: disable=not-callable
from collections import namedtuple
import copy

import torch

from torch import nn
from torch.utils.data import default_collate

from evlearn.bundled.leanbase.base.named_dict import NamedDict

InputSpec = namedtuple('InputSpec', [ 'shape', 'dtype' ])

class TorchModel(nn.Module):

    def __init__(self, backbone, encoder, decoder):
        super().__init__()

        self.backbone = backbone
        self.encoder  = encoder
        self.decoder  = decoder

    def forward(self, frame):
        result = self.decoder(self.encoder(self.backbone(frame)))

        logits = result['pred_logits']
        boxes  = result['pred_boxes']

        return (logits, boxes)

class TorchModelPreproc(nn.Module):

    def __init__(self, backbone, encoder, decoder, postprocessor, frame_shape):
        # pylint: disable=too-many-arguments
        super().__init__()

        self.backbone = backbone
        self.encoder  = encoder
        self.decoder  = decoder

        self._frame_height, self._frame_width = frame_shape[1:]

        self.postprocessor = copy.deepcopy(postprocessor)
        self.postprocessor.deploy()

    def forward(self, frame):
        result = self.decoder(self.encoder(self.backbone(frame)))

        orig_sizes = torch.as_tensor(
            [ self._frame_width, self._frame_height ]
        )
        orig_sizes = orig_sizes.to(frame.device)
        orig_sizes = orig_sizes.unsqueeze(0).expand(frame.shape[0], -1)

        labels, boxes, scores = self.postprocessor(result, orig_sizes)

        return (labels, boxes, scores)

class InferenceEngineFrameDetectionRTDETR:
    # pylint: disable=too-many-instance-attributes

    def __init__(
        self, device, dtype, backbone, encoder, decoder,
        postproc, evaluator,
        fuse_postproc = True,
        pad           = None,
        frame_shape   = None,
    ):
        # pylint: disable=too-many-arguments
        self._frame_shape   = tuple(frame_shape)
        self._device        = device
        self._dtype         = dtype
        self._fuse_postproc = fuse_postproc

        self._data = self._setup_data()
        self._nets = NamedDict(
            backbone = backbone,
            encoder  = encoder,
            decoder  = decoder,
        )

        self.pad = None

        if pad is not None:
            self.pad = copy.deepcopy(pad)

        self.postprocessor = copy.deepcopy(postproc)
        self.evaluator     = copy.deepcopy(evaluator)
        self.evaluator.reset()

    def _setup_data(self):
        return NamedDict('frame', 'labels')

    def set_inputs(self, data):
        (image, labels) = data

        if self.pad is not None:
            image, labels = self.pad(image, labels)

        self._data.image  = image.to(
            self._device, dtype = self._dtype, non_blocking = True
        )

        self._data.labels = labels

    def data_it(self):
        return [ (self._data.image, self._data.labels), ]

    @property
    def input_specs(self):
        named_inputs = [
            ('frame', InputSpec(self._frame_shape, self._dtype)),
        ]
        return named_inputs

    @property
    def output_names(self):
        if self._fuse_postproc:
            return [ 'labels', 'boxes', 'scores' ]
        else:
            return [ 'logits', 'boxes', ]

    def construct_torch_model(self):
        if self._fuse_postproc:
            return TorchModelPreproc(
                self._nets.backbone, self._nets.encoder, self._nets.decoder,
                self.postprocessor, self._frame_shape
            )
        else:
            return TorchModel(
                self._nets.backbone, self._nets.encoder, self._nets.decoder,
            )

    @torch.no_grad()
    def eval_step_standanlone_unfused(self, outputs, labels):
        # pylint: disable=too-many-locals
        logits, boxes = outputs

        if labels is None:
            return {}

        obj_labels  = []
        obj_outputs = []

        for idx, l in enumerate(labels):
            if l is not None:
                obj_labels .append(l)
                obj_outputs.append({
                    'pred_logits' : logits[idx],
                    'pred_boxes'  : boxes [idx],
                })

        n_objects = len(obj_labels)

        if n_objects == 0:
            return {}

        obj_outputs = default_collate(obj_outputs)

        img_height, img_width = self._frame_shape[1:]

        orig_sizes  = torch.as_tensor([ img_width, img_height ])
        orig_sizes  = orig_sizes.to(self._device)
        orig_sizes  = orig_sizes.unsqueeze(0).expand(n_objects, -1)

        rtdetr_preds = self.postprocessor(obj_outputs, orig_sizes)

        self.evaluator.append(
            rtdetr_preds, obj_labels, preds_format = 'rtdetr'
        )

        return {}

    @torch.no_grad()
    def eval_step_standanlone_fused(self, outputs, labels):
        # pylint: disable=too-many-locals
        if labels is None:
            return {}

        pred_labels, pred_boxes, pred_scores = outputs

        obj_labels = []
        obj_preds  = []

        for idx, l in enumerate(labels):
            if l is not None:
                obj_labels.append(l)
                obj_preds .append({
                    'labels' : pred_labels[idx],
                    'boxes'  : pred_boxes[idx],
                    'scores' : pred_scores[idx]
                })

        n_objects = len(obj_labels)

        if n_objects == 0:
            return {}

        #breakpoint()
        #obj_preds = default_collate(obj_preds)
        self.evaluator.append(obj_preds, obj_labels, preds_format = 'rtdetr')

        return {}

    @torch.no_grad()
    def eval_step_standanlone(self, outputs, labels):
        if self._fuse_postproc:
            return self.eval_step_standanlone_fused(outputs, labels)
        else:
            return self.eval_step_standanlone_unfused(outputs, labels)

    def eval_epoch_start(self):
        pass

    def eval_epoch_end(self):
        metrics = self.evaluator.evaluate()
        self.evaluator.reset()

        return metrics

