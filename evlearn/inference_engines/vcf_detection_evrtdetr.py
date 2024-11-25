# pylint: disable=not-callable
from collections import defaultdict, namedtuple
import copy
import re

import torch

from torch import nn
from torch._subclasses.fake_tensor import FakeTensorMode
from torch.utils.data import default_collate

from evlearn.bundled.leanbase.base.named_dict import NamedDict
from evlearn.models.funcs import find_new_video_mask

InputSpec = namedtuple('InputSpec', [ 'shape', 'dtype' ])

RE_LIST = re.compile(r'list\((\d+)\)_(.*)')
RE_DICT = re.compile(r'dict\(([^()]*)\)_(.*)')

def flatten_memory(memory):
    result = []

    if isinstance(memory, (list, tuple)):
        for idx, item in enumerate(memory):
            flattened_item = flatten_memory(item)

            result += [
                (f'list({idx})_' + k, v) for (k, v) in flattened_item
            ]

    elif isinstance(memory, dict):
        for name, item in memory.items():
            flattened_item = flatten_memory(item)

            result += [
                (f'dict({name})_' + k, v) for (k, v) in flattened_item
            ]

    elif isinstance(memory, torch.Tensor):
        result += [ ('memory', memory) ]

    else:
        raise RuntimeError(f'Do not know how to flatten: {type(memory)}')

    result.sort(key = lambda x : x[0])
    return result

def unflatten_memory(flat_memory):
    if len(flat_memory) == 0:
        return []

    first_name = flat_memory[0][0]

    if first_name == 'memory':
        assert len(flat_memory) == 1
        return flat_memory[0][1]

    if first_name.startswith('list'):
        items = defaultdict(list)

        for (name, value) in flat_memory:
            index, subname = RE_LIST.match(name).groups()
            items[int(index)].append((subname, value))

        result = [ [] for _ in range(max(items.keys())+1) ]

        for (index, item) in items.items():
            result[index] = unflatten_memory(item)

        return result

    if first_name.startswith('dict'):
        items = defaultdict(list)

        for (name, value) in flat_memory:
            key, subname = RE_DICT.match(name).groups()
            items[key].append((subname, value))

        result = { k : unflatten_memory(v) for (k, v) in items.items() }

        return result

    raise ValueError(f'Unknown how to unflatten {first_name}')

class TorchModel(nn.Module):

    def __init__(
        self, backbone, encoder, tempenc, decoder, memory_spec
    ):
        # pylint: disable=too-many-arguments
        super().__init__()

        self.backbone = backbone
        self.encoder  = encoder
        self.tempenc  = tempenc
        self.decoder  = decoder

        self._memory_spec = memory_spec

    def forward(self, frame, is_new_frame, *memory):
        encodings = self.encoder(self.backbone(frame))

        flat_memory = list(zip(self._memory_spec, memory))
        memory      = unflatten_memory(flat_memory)

        memory = self.tempenc.reset_mem_by_mask(memory, is_new_frame)
        temp_enc_fp, memory = self.tempenc(encodings, memory)

        result = self.decoder(temp_enc_fp)

        logits = result['pred_logits']
        boxes  = result['pred_boxes']

        flat_memory = flatten_memory(memory)

        return (logits, boxes, *[ x[1] for x in flat_memory ])

class TorchModelPostproc(nn.Module):
    # pylint: disable=too-many-instance-attributes

    def __init__(
        self, backbone, encoder, tempenc, decoder, memory_spec,
        postprocessor, frame_shape
    ):
        # pylint: disable=too-many-arguments
        super().__init__()

        self.backbone = backbone
        self.encoder  = encoder
        self.tempenc  = tempenc
        self.decoder  = decoder

        self._memory_spec = memory_spec

        self._frame_height, self._frame_width = frame_shape[1:]

        self.postprocessor = copy.deepcopy(postprocessor)
        self.postprocessor.deploy()

    def forward(self, frame, is_new_frame, *memory):
        encodings = self.encoder(self.backbone(frame))

        flat_memory = list(zip(self._memory_spec, memory))
        memory      = unflatten_memory(flat_memory)

        memory = self.tempenc.reset_mem_by_mask(memory, is_new_frame)
        temp_enc_fp, memory = self.tempenc(encodings, memory)

        result = self.decoder(temp_enc_fp)

        orig_sizes = torch.as_tensor(
            [ self._frame_width, self._frame_height ]
        )
        orig_sizes = orig_sizes.to(frame.device)
        orig_sizes = orig_sizes.unsqueeze(0).expand(frame.shape[0], -1)

        labels, boxes, scores = self.postprocessor(result, orig_sizes)

        flat_memory = flatten_memory(memory)

        return (labels, boxes, scores, *[ x[1] for x in flat_memory ])

class InferenceEngineVCFDetectionEvRTDETR:
    # pylint: disable=too-many-instance-attributes

    def __init__(
        self, device, dtype, backbone, encoder, tempenc, decoder,
        postproc, evaluator,
        fuse_postproc = True,
        frame_shape   = None,
        batch_first   = False,
    ):
        # pylint: disable=too-many-arguments
        self._video_last_index = None
        self._frame_shape      = tuple(frame_shape)
        self._device           = device
        self._dtype            = dtype
        self._batch_first      = batch_first
        self._fuse_postproc    = fuse_postproc

        self._data = self._setup_data()
        self._nets = NamedDict(
            backbone = backbone,
            encoder  = encoder,
            tempenc  = tempenc,
            decoder  = decoder,
        )

        self._memory_spec  = self.infer_memory_shape()

        self.postprocessor = copy.deepcopy(postproc)
        self.evaluator     = copy.deepcopy(evaluator)
        self.evaluator.reset()

    def infer_memory_shape(self):
        fake_mode  = FakeTensorMode(allow_non_fake_inputs = True)
        fake_frame = fake_mode.from_tensor(
            torch.empty(
                (1, *self._frame_shape),
                device = self._device, dtype = self._dtype
            )
        )

        with fake_mode:
            fake_tempenc_input = self._nets.encoder(
                self._nets.backbone(fake_frame)
            )

            fake_memory = self._nets.tempenc.init_mem(fake_tempenc_input)
            fake_memory = flatten_memory(fake_memory)

            memory_shape = [
                (k, InputSpec(tuple(v.shape[1:]), self._dtype))
                    for (k, v) in fake_memory
            ]

        return memory_shape

    def _setup_data(self):
        return NamedDict(
            'frame',            'clip',            'video',
            'labels_frame',     'labels_clip',     'labels_video',
            'obj_labels_frame', 'obj_labels_clip', 'obj_labels_video',
        )

    def _get_clip_inputs(self, data, prev_index):
        (clip, clip_index, labels_clip)  = data

        if self._batch_first:
            clip        = clip.swapaxes(0, 1)
            clip_index  = clip_index.swapaxes(0, 1)
            labels_clip = list(zip(*labels_clip))

        clip_new_video = find_new_video_mask(clip_index, prev_index)
        last_index     = clip_index[-1]

        # clip: (T, N, C, H, W)
        clip  = clip.to(
            self._device, dtype = self._dtype, non_blocking = True
        )
        clip_new_video = clip_new_video.to(self._device, non_blocking = True)

        return (clip, clip_new_video, last_index, labels_clip)

    def set_inputs(self, data):
        if 'video' in data:
            (
                self._data.video,
                self._data.video_new_video,
                self._video_last_index,
                self._data.labels_video
            ) \
                 = self._get_clip_inputs(data['video'], self._video_last_index)

        if 'clip' in data:
            (
                self._data.clip,
                self._data.clip_new_video,
                _,
                self._data.labels_clip
            ) \
                 = self._get_clip_inputs(data['clip'], prev_index = None)

        if 'frame' in data:
            (frame, frame_index, labels_frame) = data['frame']

            if self._batch_first:
                frame        = frame.unsqueeze(1)
                frame_index  = frame_index.unsqueeze(1)
                labels_frame = [ [ x, ] for x in labels_frame ]
            else:
                frame        = frame.unsqueeze(0)
                frame_index  = frame_index.unsqueeze(0)
                labels_frame = [ labels_frame, ]

            frame_data = (frame, frame_index, labels_frame)

            (
                self._data.frame,
                self._data.frame_new_video,
                _,
                self._data.labels_frame
            ) \
                = self._get_clip_inputs(frame_data, prev_index = None)

    def data_it(self, data_name):
        if data_name == 'video':
            return zip(
                self._data.video,
                self._data.video_new_video,
                self._data.labels_video
            )

        if data_name == 'clip':
            return zip(
                self._data.clip,
                self._data.clip_new_video,
                self._data.labels_clip
            )

        if data_name == 'frame':
            return zip(
                self._data.frame,
                self._data.frame_new_video,
                self._data.labels_frame
            )

        raise ValueError(f"Unknown data name: '{data_name}'")

    @property
    def input_specs(self):
        named_inputs = [
            ('frame',         InputSpec(self._frame_shape, self._dtype)),
            ('is_new_frame',  InputSpec(tuple(),           torch.bool)),
        ]
        named_inputs += self._memory_spec

        return named_inputs

    @property
    def output_names(self):
        if self._fuse_postproc:
            output_names = [ 'labels', 'boxes', 'scores' ]
        else:
            output_names = [ 'logits', 'boxes', ]

        output_names += [ 'out_' + x[0] for x in self._memory_spec ]

        return output_names

    def construct_torch_model(self):
        if self._fuse_postproc:
            return TorchModelPostproc(
                self._nets.backbone,
                self._nets.encoder,
                self._nets.tempenc,
                self._nets.decoder,
                memory_spec   = [ x[0] for x in self._memory_spec ],
                postprocessor = self.postprocessor,
                frame_shape   = self._frame_shape,
            )
        else:
            return TorchModel(
                self._nets.backbone,
                self._nets.encoder,
                self._nets.tempenc,
                self._nets.decoder,
                memory_spec = [ x[0] for x in self._memory_spec ],
            )

    @torch.no_grad()
    def init_mem(self, batch_size):
        return [
            torch.zeros(
                (batch_size, *spec.shape),
                dtype = self._dtype, device = self._device
            )
            for (name, spec) in self._memory_spec
        ]

    @torch.no_grad()
    def eval_step_standanlone_unfused(self, outputs, labels):
        # pylint: disable=too-many-locals
        logits, boxes = outputs[:2]
        flat_memory   = outputs[2:]

        if labels is None:
            return ({}, flat_memory)

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
            return ({}, flat_memory)

        obj_outputs = default_collate(obj_outputs)

        img_height, img_width = self._frame_shape[1:]

        orig_sizes  = torch.as_tensor([ img_width, img_height ])
        orig_sizes  = orig_sizes.to(self._device)
        orig_sizes  = orig_sizes.unsqueeze(0).expand(n_objects, -1)

        rtdetr_preds = self.postprocessor(obj_outputs, orig_sizes)

        self.evaluator.append(
            rtdetr_preds, obj_labels, preds_format = 'rtdetr'
        )

        return ({}, flat_memory)

    @torch.no_grad()
    def eval_step_standanlone_fused(self, outputs, labels):
        # pylint: disable=too-many-locals
        pred_labels, pred_boxes, pred_scores = outputs[:3]
        flat_memory = outputs[3:]

        if labels is None:
            return ({}, flat_memory)

        obj_labels = []
        obj_preds  = []

        for idx, l in enumerate(labels):
            if l is not None:
                obj_labels.append(l)
                obj_preds .append({
                    'labels' : pred_labels[idx],
                    'boxes'  : pred_boxes[idx],
                    'scores' : pred_scores[idx],
                })

        n_objects = len(obj_labels)

        if n_objects == 0:
            return ({}, flat_memory)

        #breakpoint()
        #obj_preds = default_collate(obj_preds)
        self.evaluator.append(obj_preds, obj_labels, preds_format = 'rtdetr')

        return ({}, flat_memory)

    @torch.no_grad()
    def eval_step_standanlone(self, outputs, labels):
        if self._fuse_postproc:
            return self.eval_step_standanlone_fused(outputs, labels)
        else:
            return self.eval_step_standanlone_unfused(outputs, labels)

    def eval_epoch_start(self):
        self._video_last_index = None

    def eval_epoch_end(self):
        metrics = self.evaluator.evaluate()
        self.evaluator.reset()

        return metrics

