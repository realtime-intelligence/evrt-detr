# pylint: disable=not-callable
import torch
from torch.utils.data import default_collate

from evlearn.bundled.leanbase.base.named_dict import NamedDict

from evlearn.bundled.leanbase.torch.select import select_optimizer
from evlearn.bundled.leanbase.torch.schedulers import select_scheduler

from evlearn.nn.backbone import construct_backbone
from evlearn.nn.temporal import construct_temporal

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

from evlearn.eval.evaluator     import construct_evaluator
from evlearn.torch.funcs        import clip_gradients, update_ema_model
from evlearn.torch.rtdetr_funcs import convert_labels_torchvision_to_rtdetr

from evlearn.inference_engines.vcf_detection_evrtdetr import (
    InferenceEngineVCFDetectionEvRTDETR
)

from .model_base import ModelBase
from .funcs      import find_new_video_mask, select_value_schedule_fn

def slice_encoder_outputs(encoder_outputs, batch_index):
    return [ eo[batch_index] for eo in encoder_outputs ]

def collate_encoder_outputs(encoder_outputs_list):
    return default_collate(encoder_outputs_list)

class VCFDetectionEvRTDETR(ModelBase):
    # pylint: disable=too-many-instance-attributes
    # pylint: disable=too-many-ancestors

    def __init__(
        self, config, device, init_train, savedir, dtype,
        rtdetr_postproc_kwargs, video_wsched, clip_wsched,
        frame_shape   = None,
        evaluator     = None,
        batch_first   = False,
        grad_clip     = None,
        ema_momentum  = None,
        use_denoising = False,
    ):
        # pylint: disable=too-many-arguments
        self._frame_shape = tuple(frame_shape)
        self._ema_mom     = ema_momentum
        self._use_denoise = use_denoising

        self._batch_first = batch_first
        self._grad_clip   = grad_clip or {}

        super().__init__(config, device, init_train, savedir, dtype)

        self.postprocessor  \
            = RTDETRPostProcessor(**rtdetr_postproc_kwargs).to(device)

        self._clip_wsched      = select_value_schedule_fn(clip_wsched)
        self._video_wsched     = select_value_schedule_fn(video_wsched)
        self._alpha_clip       = 1
        self._alpha_video      = 1
        self._memory_video     = None
        self._video_last_index = None

        self._evaluator_video = construct_evaluator(evaluator)
        self._evaluator_frame = construct_evaluator(evaluator)
        self._evaluator_clip  = construct_evaluator(evaluator)

    def _setup_data(self):
        return NamedDict(
            'frame',              'clip',              'video',
            'labels_frame',       'labels_clip',       'labels_video',
            'obj_labels_frame',   'obj_labels_clip',   'obj_labels_video',
            'obj_outputs_frame',  'obj_outputs_clip',  'obj_outputs_video',
            'obj_rtdetr_labels_frame',
            'obj_rtdetr_labels_clip',
            'obj_rtdetr_labels_video',
            'frame_memreset_mask',
            'clip_memreset_mask',
            'video_memreset_mask',
        )

    def _setup_losses(self):
        self.matcher   = HungarianMatcher(
            **self._config.losses['matcher']
        ).to(self._device)

        self.criterion = SetCriterion(
            self.matcher,
            **self._config.losses['criterion']
        ).to(self._device)

        return NamedDict(
            'frame_loss_vfl', 'frame_loss_bbox', 'frame_loss_giou',
            'clip_loss_vfl',  'clip_loss_bbox',  'clip_loss_giou',
            'video_loss_vfl', 'video_loss_bbox', 'video_loss_giou',
        )

    def _setup_nets(self):
        nets = {}

        nets['backbone'] = construct_backbone(
            self._config.nets['backbone'], self._frame_shape, self._device
        )
        nets['encoder'] = HybridEncoder(**self._config.nets['encoder'].model)
        nets['decoder'] \
            = RTDETRTransformer(**self._config.nets['decoder'].model)

        nets['temp_enc'] \
            = construct_temporal(self._config.nets['temp_enc'], self._device)
        nets['ema_temp_enc'] \
            = construct_temporal(self._config.nets['temp_enc'], self._device)

        return NamedDict(**{
            k : v.to(self._device) for (k, v) in nets.items()
        })

    def _setup_optimizers(self):
        optimizer = select_optimizer(
            self._nets.temp_enc.parameters(),
            self._config.optimizers['main']
        )

        return NamedDict(main = optimizer)

    def _setup_schedulers(self):
        sched = select_scheduler(
            self._optimizers.main, self._config.schedulers['main'],
            compose = True
        )

        return NamedDict(main = sched)

    def _get_clip_inputs(self, data, prev_index):
        (clip, clip_index, labels_clip)  = data

        if self._batch_first:
            clip        = clip.swapaxes(0, 1)
            clip_index  = clip_index.swapaxes(0, 1)
            labels_clip = list(zip(*labels_clip))

        clip_memreset_mask = find_new_video_mask(clip_index, prev_index)
        last_index         = clip_index[-1]

        # clip: (T, N, C, H, W)
        clip  = clip.to(
            self._device, dtype = self._dtype, non_blocking = True
        )
        clip_memreset_mask = clip_memreset_mask.to(
            self._device, non_blocking = True
        )

        return (clip, clip_memreset_mask, last_index, labels_clip)

    def _set_inputs(self, data):
        if 'video' in data:
            (
                self._data.video,
                self._data.video_memreset_mask,
                self._video_last_index,
                self._data.labels_video
            ) \
                 = self._get_clip_inputs(data['video'], self._video_last_index)

        if 'clip' in data:
            (
                self._data.clip,
                self._data.clip_memreset_mask,
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
                self._data.frame_memreset_mask,
                _,
                self._data.labels_frame
            ) \
                = self._get_clip_inputs(frame_data, prev_index = None)

    @torch.no_grad()
    def encode_clip_frames(self, clip):
        # pylint: disable=too-many-locals
        encoder_outputs_per_timestamp = []

        self._nets.backbone.eval()
        self._nets.encoder.eval()

        # clip :  (T, N, C, H, W)
        # frame:  (N, C, H, W)
        for frame in clip:
            backbone_features = self._nets.backbone(frame)
            encoder_output    = self._nets.encoder(backbone_features)

            encoder_outputs_per_timestamp.append(encoder_output)

        return encoder_outputs_per_timestamp

    def encode_temporal_dependencies(
        self, encoder_outputs_per_timestamp, clip_memreset_mask, memory
    ):
        result   = []
        temp_enc = self._nets.temp_enc

        if (not self._train_state) and self._ema_mom:
            temp_enc = self._nets.ema_temp_enc

        # encoder_outputs -- FPN :  List[ (N, C, H, W) ]
        for (ti, encoder_outputs) in enumerate(encoder_outputs_per_timestamp):
            if memory is None:
                memory = temp_enc.init_mem(encoder_outputs)

            memory = temp_enc.reset_mem_by_mask(memory, clip_memreset_mask[ti])

            tenc_outputs, memory = temp_enc(encoder_outputs, memory)
            result.append(tenc_outputs)

        return result, memory

    def forward_clip_objects(
        self, clip, labels_clip, clip_memreset_mask, memory
    ):
        # pylint: disable=too-many-locals

        #
        # NOTE:
        #   The RT-DETR decoder checks its training state to determine
        #   whether to use denoising and output aux losses.
        #
        #   Since we need to use denoising and aux losses, we keep the decoder
        #   in the training state.
        #
        #   However, we do not calculate decoder gradients nor update its
        #   weights.
        #

        if self._train_state:
            for param in self._nets.decoder.parameters():
                param.requires_grad = False

            #
            # NOTE:
            #   This is a dirty hack to disable denoising if the
            #   decoder in the training state.
            #   Otherwise, it will raise an exception
            #

            if not self._use_denoise:
                self._nets.decoder.num_denoising = 0

        encoder_outputs_per_timestamp = self.encode_clip_frames(clip)

        tenc_outputs_per_timestamp, memory \
            = self.encode_temporal_dependencies(
                encoder_outputs_per_timestamp, clip_memreset_mask, memory
            )

        obj_tenc_outputs = []
        obj_labels       = []

        # Cherry pick only frames with annotations
        for (ti, tenc_outputs) in enumerate(tenc_outputs_per_timestamp):
            for (bi, labels) in enumerate(labels_clip[ti]):
                if labels is None:
                    continue

                obj_tenc_outputs.append(
                    slice_encoder_outputs(tenc_outputs, bi)
                )
                obj_labels.append(labels)

        if len(obj_labels) == 0:
            return (None, None, None, memory)

        obj_rtdetr_labels = convert_labels_torchvision_to_rtdetr(
            obj_labels, self._device, self._frame_shape[1:]
        )
        obj_tenc_outputs = collate_encoder_outputs(obj_tenc_outputs)

        if self._use_denoise and self._train_state:
            obj_dec_outputs \
                = self._nets.decoder(obj_tenc_outputs, obj_rtdetr_labels)
        else:
            obj_dec_outputs = self._nets.decoder(obj_tenc_outputs)

        return (
            obj_dec_outputs,
            obj_labels,
            obj_rtdetr_labels,
            memory
        )

    def forward_video(self):
        (
            self._data.obj_outputs_video,
            self._data.obj_labels_video,
            self._data.obj_rtdetr_labels_video,
            memory
        ) = self.forward_clip_objects(
            self._data.video,
            self._data.labels_video,
            self._data.video_memreset_mask,
            memory = self._memory_video
        )

        self._memory_video = self._nets.temp_enc.detach_mem(memory)

    def forward_clip(self):
        (
            self._data.obj_outputs_clip,
            self._data.obj_labels_clip,
            self._data.obj_rtdetr_labels_clip,
            _memory
        ) = self.forward_clip_objects(
            self._data.clip,
            self._data.labels_clip,
            self._data.clip_memreset_mask,
            memory = None
        )

    def forward_frame(self):
        (
            self._data.obj_outputs_frame,
            self._data.obj_labels_frame,
            self._data.obj_rtdetr_labels_frame,
            _memory
        ) = self.forward_clip_objects(
            self._data.frame,
            self._data.labels_frame,
            self._data.frame_memreset_mask,
            memory = None
        )

    def backward_video(self):
        if self._data.obj_outputs_video is None:
            return

        loss_dict = self.criterion(
            self._data.obj_outputs_video, self._data.obj_rtdetr_labels_video
        )

        self._losses.video_loss_vfl  = loss_dict['loss_vfl']
        self._losses.video_loss_bbox = loss_dict['loss_bbox']
        self._losses.video_loss_giou = loss_dict['loss_giou']

        loss = sum(loss_dict.values())
        loss = self._alpha_video * self._alpha_clip * loss
        loss.backward()

    def backward_clip(self):
        loss_dict = self.criterion(
            self._data.obj_outputs_clip, self._data.obj_rtdetr_labels_clip
        )

        self._losses.clip_loss_vfl  = loss_dict['loss_vfl']
        self._losses.clip_loss_bbox = loss_dict['loss_bbox']
        self._losses.clip_loss_giou = loss_dict['loss_giou']

        loss = sum(loss_dict.values())
        loss = (1 - self._alpha_video) * self._alpha_clip * loss
        loss.backward()

    def backward_frame(self):
        loss_dict = self.criterion(
            self._data.obj_outputs_frame, self._data.obj_rtdetr_labels_frame
        )

        self._losses.frame_loss_vfl  = loss_dict['loss_vfl']
        self._losses.frame_loss_bbox = loss_dict['loss_bbox']
        self._losses.frame_loss_giou = loss_dict['loss_giou']

        loss = sum(loss_dict.values())
        loss = (1 - self._alpha_clip) * loss
        loss.backward()

    def update_ema_states(self):
        update_ema_model(
            self._nets.ema_temp_enc, self._nets.temp_enc, self._ema_mom
        )

    def train_step(self):
        self.train()
        self.criterion.train()
        self._optimizers.main.zero_grad(set_to_none = True)

        if self._data.video is not None:
            self.forward_video()
            self.backward_video()

        if self._data.clip is not None:
            self.forward_clip()
            self.backward_clip()

        if self._data.frame is not None:
            self.forward_frame()
            self.backward_frame()

        clip_gradients(self._optimizers.main, **self._grad_clip)
        self._optimizers.main.step()

        self.step_scheduler(self._schedulers.main, event = 'batch')
        self.update_ema_states()

    def get_current_losses(self):
        result = {
            k : float(v.detach().cpu().mean())
                for (k, v) in self._losses.items() if v is not None
        }

        if self._schedulers.main is not None:
            result['lr'] = float(self._schedulers.main.get_last_lr()[0])

        result['alpha_clip']  = self._alpha_clip
        result['alpha_video'] = self._alpha_video

        result['n_obj_video'] = len(self._data.obj_labels_video  or [])
        result['n_obj_clip']  = len(self._data.obj_labels_clip   or [])
        result['n_obj_frame'] = len(self._data.obj_labels_frame  or [])

        return result

    @torch.no_grad()
    def eval_step_base(self, obj_outputs, obj_labels, evaluator):
        # pylint: disable=too-many-locals
        if obj_labels is None:
            return []

        n_objects = len(obj_labels)

        if n_objects == 0:
            return []

        img_height, img_width = self._frame_shape[1:]

        orig_sizes  = torch.as_tensor([ img_width, img_height ])
        orig_sizes  = orig_sizes.to(self._device)
        orig_sizes  = orig_sizes.unsqueeze(0).expand(n_objects, -1)

        rtdetr_preds = self.postprocessor(obj_outputs, orig_sizes)

        evaluator.append(
            rtdetr_preds, obj_labels, preds_format = 'rtdetr'
        )

        return obj_outputs

    @torch.no_grad()
    def eval_step(self):
        # pylint: disable=too-many-locals
        self.eval()
        self.criterion.eval()

        if self._data.frame is not None:
            self.forward_frame()

            self._data.rtdetr_preds_frame = self.eval_step_base(
                self._data.obj_outputs_frame,
                self._data.obj_labels_frame,
                self._evaluator_frame,
            )

        if self._data.clip is not None:
            self.forward_clip()

            self._data.rtdetr_preds_clip = self.eval_step_base(
                self._data.obj_outputs_clip,
                self._data.obj_labels_clip,
                self._evaluator_clip,
            )

        if self._data.video is not None:
            self.forward_video()

            self._data.rtdetr_preds_video = self.eval_step_base(
                self._data.obj_outputs_video,
                self._data.obj_labels_video,
                self._evaluator_video,
            )

        return {}

    def eval_epoch_start(self):
        self._memory_video     = None
        self._video_last_index = None

    def _epoch_start(self):
        self._alpha_clip       = self._clip_wsched(self._epoch)
        self._alpha_video      = self._video_wsched(self._epoch)
        self._memory_video     = None
        self._video_last_index = None

    def epoch_end(self, metrics):
        self.step_scheduler(self._schedulers.main, 'epoch', metrics)

    def eval_epoch_end(self):
        metrics_frame = self._evaluator_frame.evaluate()
        metrics_clip  = self._evaluator_clip .evaluate()
        metrics_video = self._evaluator_video.evaluate()

        self._evaluator_frame.reset()
        self._evaluator_clip .reset()
        self._evaluator_video.reset()

        result = {}

        for (k, v) in metrics_frame.items():
            result[k + '_frame'] = v

        for (k, v) in metrics_clip.items():
            result[k + '_clip'] = v

        for (k, v) in metrics_video.items():
            result[k + '_video'] = v

        return result

    def predict_step(self):
        raise NotImplementedError

    def construct_inference_engine(self, fuse_postproc = False):
        tempenc = self._nets.temp_enc

        if self._ema_mom:
            tempenc = self._nets.ema_temp_enc

        return InferenceEngineVCFDetectionEvRTDETR(
            self._device, self._dtype,
            backbone      = self._nets.backbone,
            encoder       = self._nets.encoder,
            tempenc       = tempenc,
            decoder       = self._nets.decoder,
            postproc      = self.postprocessor,
            evaluator     = self._evaluator_video,
            fuse_postproc = fuse_postproc,
            frame_shape   = self._frame_shape,
            batch_first   = self._batch_first,
        )

