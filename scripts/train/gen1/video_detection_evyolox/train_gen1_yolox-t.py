from evlearn.train.train import setup_and_train
from evlearn.utils.log   import setup_logging

# cfg: https://github.com/Megvii-BaseDetection/YOLOX/blob/main/exps/default/yolox_x.py
DEPTH = 0.33
WIDTH = 0.375

BATCH_SIZE   = 4
SEQ_LENGTH   = 21
LR_BASE      = 0.0002
DOWNSAMPLING = None
NUM_CLASSES  = 2
BATCH_FIRST  = False

DTYPE       = 'float32'
CANVAS_SIZE = [ 240, 304 ]
FRAME_SIZE  = [ 256, 320 ]


DATA_PATH = "gen1/gen1_preproc_npz"
P_GEOM    = 0.6
P_ERASE   = 0.4
N_QUERY   = 300
N_DENOISE = 100

TRANSFER_PATH = 'models/gen1/frame_detection_yolox-t'

TRANSFORM_TRAIN_BASE = [
    'random-flip-horizontal',
]

RESIZE_BASE = []
PAD_BASE    = [
    # pad from (240, 304) -> (256, 320)
    {
        'name'    : 'pad',
        'padding' : (0, 0, 16, 16),
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

TRANSFORM_EVAL_BASE = RESIZE_BASE + PAD_BASE

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
        'size' : CANVAS_SIZE,
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
        'name'         : 'vcf-detection-evyolox',
        'batch_first'  : BATCH_FIRST,
        'ema_momentum' : 0,
        'frame_shape'  : (20, *FRAME_SIZE),
        'grad_clip' : {
            'value' : 5.0,
        },
        'evaluator' : {
            'name'        : 'psee',
            'camera'      : 'gen1',
            'image_size'  : FRAME_SIZE,
            'classes'     : [ "car", "pedestrian", ],
            'labels_name' : 'psee_labels',
            'downsampling_factor' : DOWNSAMPLING,
        },
        'yolox_postproc_kwargs' : {
            'num_classes'          : NUM_CLASSES,
            'confidence_threshold' : 0.01,
            'nms_threshold'        : 0.65,
        },
        'yolox_init' : True,
        'video_wsched' : {
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
                    (
                        int(256 * WIDTH),
                        FRAME_SIZE[0] // 8,  FRAME_SIZE[1] // 8
                    ),
                    (
                        int(512 * WIDTH),
                        FRAME_SIZE[0] // 16, FRAME_SIZE[1] // 16
                    ),
                    (
                        int(1024 * WIDTH),
                        FRAME_SIZE[0] // 32, FRAME_SIZE[1] // 32
                    ),
                ],
                'n_layers_list'        : [ 1,   1,   1 ],
                'hidden_features_list' : [ 256, 256, 256 ],
                'kernel_size_list'     : [ 3,   3,   3 ],
                'rezero' : False,
            },
        },
        "backbone": {
            'model' : {
                "name"        : 'pafpn-yolox',
                "in_channels" : [ 256, 512, 1024 ],
                "depth"       : DEPTH,
                "width"       : WIDTH,
                "act"         : "silu",
            },
        },
        "head" : {
            'model' : {
                "num_classes" : NUM_CLASSES,
                "width"       : WIDTH,
                "in_channels" : [ 256, 512, 1024 ],
                "act"         : "silu",
            },
        },
    },
    'losses'     : { },
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
    'transfer' : {
        'base_model' : TRANSFER_PATH,
        'transfer_map' : {
            'backbone'     : 'ema_backbone',
            'head'         : 'ema_head',
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
    "outdir"     : "gen1/video_detection_yolox-t",
}

setup_logging()
setup_and_train(args_dict)

