import os

ROOT_DATA   = os.path.join(os.environ.get('EVLEARN_DATA',   'data'))
ROOT_OUTDIR = os.path.join(os.environ.get('EVLEARN_OUTDIR', 'outdir'))

SPLIT_TRAIN = 'train'
SPLIT_VAL   = 'val'
SPLIT_TEST  = 'test'

MODEL_STATE_TRAIN = 'train'
MODEL_STATE_EVAL  = 'eval'
