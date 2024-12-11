from evlearn.train.train import setup_and_train
from evlearn.utils.log   import setup_logging

# cfg: https://github.com/lyuwenyu/RT-DETR/blob/main/rtdetr_pytorch/configs/rtdetr/rtdetr_r18vd_6x_coco.yml
# cfg: https://github.com/lyuwenyu/RT-DETR/blob/main/rtdetr_pytorch/configs/rtdetr/include/rtdetr_r50vd.yml

NUM_CLASSES = 2
FOCAL       = True

DTYPE       = 'float32'
CANVAS_SIZE = [ 240, 304 ]
FRAME_SIZE  = [ 256, 320 ]

DATA_PATH = "gen1/gen1_preproc_npz"
P_GEOM    = 0.6
P_ERASE   = 0.4
N_QUERY   = 300
N_DENOISE = 100

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
        'name' : 'frame-detection-rtdetr',
        'ema_momentum' : 0.9999,
        'rtdetr_postproc_kwargs' : {
            'num_top_queries'       : N_QUERY,
            'num_classes'           : NUM_CLASSES,
            'use_focal_loss'        : FOCAL,
            'remap_mscoco_category' : False,
        },
        'grad_clip' : {
            'value' : 5.0,
        },
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
                "name"        : 'presnet-rtdetr',
                'depth'       : 18,
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
                  'in_channels'  : [ 128, 256, 512 ],
                  'feat_strides' : [ 8,   16,  32],
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
                  'expansion'  : 0.5,
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
                'num_decoder_layers'  : 3,
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
            'lr'   : 0.0002,
            'weight_decay' : 0,
        },
    },
    'schedulers' : { 'main' : None, },
    'val_interval' : 10,
    'steps_per_train_epoch' : 1000,
    'steps_per_val_epoch'   : 1000,
# Args
    'checkpoint' : 10,
    "label"      : 'default',
    "outdir"     : "gen1/frame_rtdetr_presnet18"
}

setup_logging()
setup_and_train(args_dict)

