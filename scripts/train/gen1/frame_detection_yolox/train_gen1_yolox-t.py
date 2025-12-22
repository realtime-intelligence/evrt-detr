from evlearn.train.train import setup_and_train
from evlearn.utils.log   import setup_logging

# cfg: https://github.com/Megvii-BaseDetection/YOLOX/blob/main/exps/default/yolox_tiny.py
# cfg: bundled/yolox/exp/yolox_base.py

NUM_CLASSES = 2

DTYPE       = 'float32'
CANVAS_SIZE = [ 240, 304 ]
FRAME_SIZE  = [ 256, 320 ]

DATA_PATH = "gen1/gen1_preproc_npz"
P_GEOM    = 0.6
P_ERASE   = 0.4

# cfg: https://github.com/Megvii-BaseDetection/YOLOX/blob/main/exps/default/yolox_tiny.py
DEPTH     = 0.33
WIDTH     = 0.375

TRANSFORM_TRAIN_BASE = [
    'random-flip-horizontal',
]

PAD_BASE = [
    # pad from (240, 304) -> (256, 320)
    {
        'name'    : 'pad',
        'padding' : (0, 0, 256 - 240, 320 - 304),
    },
]

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

TRANSFORM_EVAL_BASE = PAD_BASE

def construct_train_transform(p_geom = P_GEOM, p_erase = P_ERASE):
    result = []
    result += TRANSFORM_TRAIN_BASE

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

    if p_erase > 0:
        result.append({
            'name'  : 'random-erasing',
            'p'     : p_erase,
        })

    result += PAD_BASE
    return result

args_dict = {
# Config
    'data' : {
        'train' : {
            'batch_size' : 32,
            'collate' : 'default-with-labels',
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
                'bbox_fmt'    : 'xyxy',
                'canvas_size' : CANVAS_SIZE,
            },
            'sampler' : {
                'name'           : 'video-element',
                'shuffle_videos' : True,
                'shuffle_frames' : True,
                'skip_unlabeled' : True,
                'drop_last'      : False,
                'pad_empty'      : False,
                'seed'           : 0,
                'split_by_video_starts' : False,
            },
            'shapes'  : [ (20, *FRAME_SIZE), 1 ],
            'transform_video'  : None,
            'transform_frame'  : construct_train_transform(),
            'transform_labels' : [
                'clamp-bboxes',
                'sanitize-bboxes',
            ],
            'workers' : 8,
        },
        'eval'  : {
            'batch_size' : 32,
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
                'bbox_fmt'    : 'xyxy',
                'canvas_size' : CANVAS_SIZE,
            },
            'collate' : 'default-with-labels',
            'sampler' : {
                'name'           : 'video-element',
                'shuffle_videos' : False,
                'shuffle_frames' : False,
                'skip_unlabeled' : True,
                'drop_last'      : False,
                'pad_empty'      : False,
                'seed'           : 0,
                'split_by_video_starts' : False,
            },
            'shapes'  : [ (20, *FRAME_SIZE), 1 ],
            'transform_video'  : None,
            'transform_frame'  : TRANSFORM_EVAL_BASE,
            'transform_labels' : [
                'clamp-bboxes',
            ],
            'workers' : 8,
        },
    },
    'epochs'     : 400,
    'model'      : {
        'name' : 'frame-detection-yolox',
        'ema_momentum' : 0.9999,
        'yolox_postproc_kwargs' : {
            'num_classes'          : NUM_CLASSES,
            'confidence_threshold' : 0.01,
            'nms_threshold'        : 0.65,
        },
        'yolox_init' : True,
        'grad_clip'  : { 'value' : 5.0, },
        'evaluator' : {
            'name'        : 'psee',
            'camera'      : 'gen1',
            'image_size'  : FRAME_SIZE,
            'classes'     : [ "car", "pedestrian" ],
            'labels_name' : 'psee_labels',
            'downsampling_factor' : None,
        },
    },
    'nets' : {
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
    'losses'     : {},
    'optimizers' : {
        'main' : {
            'name' : 'AdamW',
            'lr'   : 0.0002,
            'weight_decay' : 5e-4,
        },
    },
    'schedulers' : { 'main' : None, },
    'val_interval'          : 10,
    'steps_per_train_epoch' : 1000,
    'steps_per_val_epoch'   : 1000,
# Args
    'checkpoint' : 10,
    "label"      : "default",
    "outdir"     : "gen1/frame_detection_yolox-t",
}

setup_logging()
setup_and_train(args_dict)

