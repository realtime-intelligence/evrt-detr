from evlearn.train.train import setup_and_train
from evlearn.utils.log   import setup_logging

BATCH_SIZE   = 4
SEQ_LENGTH   = 10
LR_BASE      = 0.0002
DOWNSAMPLING = 2
NUM_CLASSES  = 3
N_QUERIES    = 300
N_DENOISE    = 100
FOCAL        = True
BATCH_FIRST  = False

DTYPE         = 'float32'
INTERPOLATION = 'bilinear'
CANVAS_SIZE   = [ 720, 1280 ]
FRAME_SIZE    = [ 360 + 24, 640 ]

DATA_PATH = "gen4/gen4_preproc_npz"
P_GEOM    = 0.6
P_ERASE   = 0.4
N_QUERY   = 300
N_DENOISE = 100

TRANSFER_PATH = 'models/gen4/frame_rtdetr_presnet50'

TRANSFORM_TRAIN_BASE = [
    'random-flip-horizontal',
]

RESIZE_BASE = [
    # resize from (1280, 720) -> (360, 640)
    {
        'name' : 'resize',
        'size' : (360, 640),
        'interpolation' : INTERPOLATION,
    },
]

PAD_BASE = [
    # pad from (360, 640) -> (384, 640)
    {
        'name'    : 'pad',
        'padding' : (0, 0, 0, 24),
    },
]

TRANSFORM_EVAL_BASE = RESIZE_BASE + PAD_BASE

GEOMETRIC_AUGS = [
    # 'rotation-30' : {
    {
        'name'    : 'random-rotation',
        'degrees' : 30,
        'expand'  : False,
    },
    #'translate-50' : {
    {
        'name'      : 'random-affine',
        'degrees'   : 0,
        'translate' : (0.5, 0.5),
    },
    #'scale-50' : {
    {
        'name'      : 'random-affine',
        'degrees'   : 0,
        'scale'     : (0.5, 1.5),
    },
    #'shear-30' : {
    {
        'name'    : 'random-affine',
        'degrees' : 0,
        'shear'   : 30,
    },
]

def construct_train_video_transform(p_geom = P_GEOM):
    result  = []
    result += RESIZE_BASE + TRANSFORM_TRAIN_BASE

    for transform in GEOMETRIC_AUGS:
        result.append({
            'name' : 'random-apply',
            'p'    : p_geom,
            'transforms' : [ transform ],
        })

    result.append({
        'name' : 'center-crop',
        'size' : tuple(x // DOWNSAMPLING for x in CANVAS_SIZE),
    })

    result += PAD_BASE
    return result

def construct_train_frame_jitter_transform(
    p_drop = 0, p_erase = P_ERASE, m_persp = 0
):
    result = []

    if p_drop > 0:
        result.append({
            'name' : 'channel-dropout',
            'p'    : p_drop,
        })

    if p_erase > 0:
        result.append({
            'name'  : 'random-erasing',
            'p'     : p_erase,
        })

    if m_persp > 0:
        result.append({
            'name'  : 'random-perspective',
            'distortion_scale' : m_persp,
        })

    return result

TRAIN_DATA_CONFIG_RANDCLIP = {
    'batch_size' : BATCH_SIZE,
    'collate' : {
        'name'        : 'video',
        'batch_first' : BATCH_FIRST,
    },
    'dataset' : {
        'name'  : 'ebc-video-frame',
        'path'  : DATA_PATH,
        'data_dtype_list'   : [
            ('frame', DTYPE),
        ],
        'label_dtype_list'  : [
            ('boxes',  None),
            ('labels', 'int32'),
        ],
        'bbox_fmt'     : 'xyxy',
        'canvas_size'  : CANVAS_SIZE,
        'return_index' : True,
    },
    'sampler' : {
        'name'           : 'video-clip',
        'shuffle_videos' : True,
        'shuffle_frames' : False,
        'shuffle_clips'  : True,
        'skip_unlabeled' : True,
        'drop_last'      : False,
        'pad_empty'      : True,
        'seed'           : 1,
        'clip_length'    : SEQ_LENGTH,
        'split_by_video_starts' : False,
    },
    'shapes'  : [ (None, 20, *FRAME_SIZE), 1 ],
    'transform_video' : construct_train_video_transform(),
    'transform_frame' : construct_train_frame_jitter_transform(),
    'transform_labels' : [
        'clamp-bboxes',
        'sanitize-bboxes',
    ],
    'workers' : BATCH_SIZE,
}

TRAIN_DATA_CONFIG_VIDEO = {
    'batch_size' : BATCH_SIZE,
    'collate' : {
        'name'        : 'video',
        'batch_first' : BATCH_FIRST,
    },
    'dataset' : {
        'name'  : 'ebc-video-frame',
        'path'  : DATA_PATH,
        'data_dtype_list'   : [
            ('frame', DTYPE),
        ],
        'label_dtype_list'  : [
            ('boxes',  None),
            ('labels', 'int32'),
        ],
        'bbox_fmt'     : 'xyxy',
        'canvas_size'  : CANVAS_SIZE,
        'return_index' : True,
    },
    'sampler' : {
        'name'           : 'video-clip',
        'shuffle_videos' : True,
        'shuffle_frames' : False,
        'shuffle_clips'  : False,
        'skip_unlabeled' : True,
        'drop_last'      : False,
        'pad_empty'      : True,
        'seed'           : 0,
        'clip_length'    : SEQ_LENGTH,
        'split_by_video_starts' : False,
    },
    'shapes'  : [ (None, 20, *FRAME_SIZE), 1 ],
    'transform_video' : construct_train_video_transform(),
    'transform_frame' : construct_train_frame_jitter_transform(),
    'transform_labels' : [
        'clamp-bboxes',
        'sanitize-bboxes',
    ],
    'workers' : BATCH_SIZE,
}

EVAL_DATA_CONFIG_VIDEO = {
    'batch_size' : BATCH_SIZE,
    'collate' : {
        'name'        : 'video',
        'batch_first' : BATCH_FIRST,
    },
    'dataset' : {
        'name'  : 'ebc-video-frame',
        'path'  : DATA_PATH,
        'data_dtype_list'   : [
            ('frame', DTYPE),
        ],
        'label_dtype_list'  : [
            ('boxes',       None),
            ('labels',      'int32'),
            ('psee_labels', None),
        ],
        'bbox_fmt'     : 'xyxy',
        'canvas_size'  : CANVAS_SIZE,
        'return_index' : True,
    },
    'sampler' : {
        'name'           : 'video-clip',
        'shuffle_videos' : False,
        'shuffle_frames' : False,
        'shuffle_clips'  : False,
        'skip_unlabeled' : False,
        'drop_last'      : False,
        'pad_empty'      : True,
        'seed'           : 0,
        'clip_length'    : SEQ_LENGTH,
        'split_by_video_starts' : True,
    },
    'shapes'  : [ (None, 20, *FRAME_SIZE), 1 ],
    'transform_video' : TRANSFORM_EVAL_BASE,
    'transform_frame' : None,
    'transform_labels' : [
        'clamp-bboxes',
        'sanitize-bboxes',
    ],
    'workers' : 4,
}

EVAL_DATA_CONFIG_CLIP = {
    'batch_size' : BATCH_SIZE,
    'collate' : {
        'name'        : 'video',
        'batch_first' : BATCH_FIRST,
    },
    'dataset' : {
        'name'  : 'ebc-video-frame',
        'path'  : DATA_PATH,
        'data_dtype_list'   : [
            ('frame', DTYPE),
        ],
        'label_dtype_list'  : [
            ('boxes',       None),
            ('labels',      'int32'),
            ('psee_labels', None),
        ],
        'bbox_fmt'     : 'xyxy',
        'canvas_size'  : CANVAS_SIZE,
        'return_index' : True,
    },
    'sampler' : {
        'name'           : 'video-clip',
        'shuffle_videos' : False,
        'shuffle_frames' : False,
        'shuffle_clips'  : False,
        'skip_unlabeled' : True,
        'drop_last'      : False,
        'pad_empty'      : True,
        'seed'           : 0,
        'clip_length'    : SEQ_LENGTH,
        'split_by_video_starts' : True,
    },
    'shapes'  : [ (None, 20, *FRAME_SIZE), 1 ],
    'transform_video' : TRANSFORM_EVAL_BASE,
    'transform_frame' : None,
    'transform_labels' : [
        'clamp-bboxes',
        'sanitize-bboxes',
    ],
    'workers' : 4,
}

EVAL_DATA_CONFIG_FRAME = {
    'batch_size' : BATCH_SIZE,
    'collate' : 'default-with-labels',
    'dataset' : {
        'name'  : 'ebc-video-frame',
        'path'  : DATA_PATH,
        'data_dtype_list'   : [
            ('frame', DTYPE),
        ],
        'label_dtype_list'  : [
            ('boxes',       None),
            ('labels',      'int32'),
            ('psee_labels', None),
        ],
        'bbox_fmt'     : 'xyxy',
        'canvas_size'  : CANVAS_SIZE,
        'return_index' : True,
    },
    'sampler' : {
        'name'           : 'video-element',
        'shuffle_videos' : False,
        'shuffle_frames' : False,
        'skip_unlabeled' : True,
        'drop_last'      : False,
        'pad_empty'      : True,
        'seed'           : 0,
        'split_by_video_starts' : False,
    },
    'shapes'  : [ (20, *FRAME_SIZE), 1 ],
    'transform_video' : None,
    'transform_frame' : TRANSFORM_EVAL_BASE,
    'transform_labels' : [
        'clamp-bboxes',
        'sanitize-bboxes',
    ],
    'workers' : 4,
}

args_dict = {
# Config
    'data' : {
        'train' : {
            'video' : TRAIN_DATA_CONFIG_VIDEO,
            'clip'  : TRAIN_DATA_CONFIG_RANDCLIP,
        },
        'eval'  : {
            'video' : EVAL_DATA_CONFIG_VIDEO,
            'clip'  : EVAL_DATA_CONFIG_CLIP,
            'frame' : EVAL_DATA_CONFIG_FRAME,
        },
    },
    'epochs' : 200,
    'model'  : {
        'name'          : 'vcf-detection-evrtdetr',
        'batch_first'   : BATCH_FIRST,
        'ema_momentum'  : 0,
        'frame_shape'   : (20, *FRAME_SIZE),
        'use_denoising' : True,
        'grad_clip' : {
            'value' : 5.0,
        },
        'evaluator' : {
            'name'        : 'psee',
            'camera'      : 'gen4',
            'image_size'  : FRAME_SIZE,
            'classes'     : [ "pedestrian", "two-wheeler", "car" ],
            'labels_name' : 'psee_labels',
            'downsampling_factor' : DOWNSAMPLING,
        },
        'rtdetr_postproc_kwargs' : {
            'num_top_queries'       : N_QUERIES,
            'num_classes'           : NUM_CLASSES,
            'use_focal_loss'        : FOCAL,
            'remap_mscoco_category' : False,
        },
        'video_wsched': {
            'name'  : 'constant',
            # Equal weight of random clips and videos
            'value' : 0.5,
        },
        'clip_wsched' : {
            'name'  : 'constant',
            # Frames have 0 (=1-value) weight
            'value' : 1,
        },
    },
    'nets' : {
        "temp_enc" : {
            'model' : {
                "name"  : 'conv-lstm',
                "fpn_shapes" : [
                    (256, FRAME_SIZE[0] // 8,  FRAME_SIZE[1] // 8),
                    (256, FRAME_SIZE[0] // 16, FRAME_SIZE[1] // 16),
                    (256, FRAME_SIZE[0] // 32, FRAME_SIZE[1] // 32),
                ],
                'n_layers_list'        : [ 1,   1,   1 ],
                'hidden_features_list' : [ 256, 256, 256 ],
                'kernel_size_list'     : [ 3,   3,   3 ],
                'rezero' : False,
            },
        },
        "backbone": {
            'model' : {
                "name"        : 'presnet-rtdetr',
                'depth'       : 50,
                'variant'     : 'd',
                'num_stages'  : 4,
                'return_idx'  : [ 1, 2, 3 ],
                'act'         : 'relu',
                'freeze_at'   : -1,
                'freeze_norm' : False,
                'pretrained'  : False,
            },
        },
        "encoder" : {
            'model' : {
                  'in_channels'  : [ 512, 1024, 2048 ],
                  'feat_strides' : [ 8,   16,   32],
                  # intra
                  'hidden_dim'         : 256,
                  'use_encoder_idx'    : [2],
                  'num_encoder_layers' : 1,
                  'nhead'              : 8,
                  'dim_feedforward'    : 1024,
                  'dropout'            : 0.,
                  'enc_act'            : 'gelu',
                  'pe_temperature'     : 10000,
                  # cross
                  'expansion'  : 1.0,
                  'depth_mult' : 1,
                  'act'        : 'silu',
                  # eval
                  'eval_spatial_size' : FRAME_SIZE,
            },
        },
        "decoder" : {
            "model" : {
                'num_classes'         : NUM_CLASSES,
                'position_embed_type' : 'sine',
                'num_decoder_points'  : 4,
                'nhead'               : 8,
                'dim_feedforward'     : 1024,
                'dropout'             : 0.,
                'activation'          : "relu",
                'label_noise_ratio'   : 0.5,
                'box_noise_scale'     : 1.0,
                'learnt_init_query'   : False,
                'eps'                 : 1e-2,
                'aux_loss'            : True,
                'feat_channels'       : [256, 256, 256],
                'feat_strides'        : [8, 16, 32],
                'hidden_dim'          : 256,
                'num_levels'          : 3,
                'num_queries'         : N_QUERY,
                'num_decoder_layers'  : 6,
                'num_denoising'       : N_DENOISE,
                'eval_idx'            : -1,
                'eval_spatial_size'   : FRAME_SIZE,
            },
        },
    },
    'losses'     : {
        'matcher' : {
            'weight_dict' : {'cost_class': 2, 'cost_bbox': 5, 'cost_giou': 2},
            'use_focal_loss' : FOCAL,
            'alpha'          : 0.25,
            'gamma'          : 2.0,
        },
        'criterion' : {
            'weight_dict' : { 'loss_vfl': 1, 'loss_bbox': 5, 'loss_giou': 2,},
            'losses'      : [ 'vfl', 'boxes', ],
            'alpha'       : 0.75,
            'gamma'       : 2.0,
            'eos_coef'    : 1e-4,
            'num_classes' : NUM_CLASSES,
        },
    },
    'optimizers' : {
        'main' : {
            'name' : 'AdamW',
            'lr'   : LR_BASE * BATCH_SIZE / 4,
            'weight_decay' : 0,
        },
    },
    'schedulers' : {
        'main' : {
            'name'             : 'one-cycle',
            'max_lr'           : LR_BASE * BATCH_SIZE / 4,
            'div_factor'       : 20,
            'final_div_factor' : 10000 / 20,
            'pct_start'        : 0.005,
            'total_steps'      : 200000,
            'anneal_strategy'  : 'linear',
        },
    },
    'transfer'   : {
        'base_model' : TRANSFER_PATH,
        'transfer_map' : {
            'backbone' : 'ema_backbone',
            'encoder'  : 'ema_encoder',
            'decoder'  : 'ema_decoder',
        },
        'fuzzy'   : None,
        'strict'  : True,
        'load_train_state'    : False,
        'use_last_checkpoint' : False,
    },
    'val_interval'          : 10,
    'steps_per_train_epoch' : 1000,
    'steps_per_val_epoch'   : 1000,
# Args
    'checkpoint' : 10,
    "label"      : 'default',
    "outdir"     : "gen4/video_evrtdetr_presnet50",
}

setup_logging()
setup_and_train(args_dict)

