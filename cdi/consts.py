import os

CONFIG_NAME = 'config.json'

ROOT_DATA   = os.environ.get('CDI_DATA',   'data')
ROOT_OUTDIR = os.environ.get('CDI_OUTDIR', 'outdir')

SPLIT_TRAIN = 'train'
SPLIT_VAL   = 'val'
SPLIT_TEST  = 'test'

MODEL_STATE_TRAIN = 'train'
MODEL_STATE_EVAL  = 'eval'
