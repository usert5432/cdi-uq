from cdi.train.train import setup_and_train
from cdi.utils.logging import setup_logging

EPOCHS        = 500
BATCH_SIZE    = 32
LR            = 1e-4
INIT_FEATURES = 64

DATA_PATH_F11   = 'cbed/f11'
F11_TRAIN_NAMES = [ 'f11a', 'f11b', 'f11c', 'f11e' ]
F11_EVAL_NAMES  = [ 'f11f' ]

IMAGE_SIZE = (320, 320)

TARGET_COLUMNS = [
    'conf_0',
    'conf_1',
    'conf_2',
    'conf_3',
    'conf_4',
    'conf_5',
    'conf_6',
    'conf_7',
    'conf_8',
    'conf_9',
    'conf_10',
    'conf_11',
    'label',
]

TARGET_NORMS = [
    (0.0048, 0.0025),
    (0.0056, 0.0025),
    (0.002,  0.001),
    (0.0047, 0.0025),
    (0.0051, 0.0025),
    (0.002,  0.001),
    (-0.91,  1),
    (0.63,   0.6),
    (2.69,   0.4),
    (2.94,   0.3),
    (-0.03,  0.2),
    (0.015,  0.2),
    (2800,   1000),
]

TRANSFORM_F11_TRAIN = [
    {
        'name'          : 'random-rotation',
        'interpolation' : 'bilinear',
        'degrees'       : 15,
        'expand'        : True,
    },
    {
        'name' : 'center-crop',
        'size' : (400, 400),
    },
    {
        'name'  : 'random-resize-crop',
        'size'  : IMAGE_SIZE,
        'scale' : (0.6, 1.0),
        'ratio' : (1.0, 1.0),
        'interpolation' : 'bilinear',
    },
]

TRANSFORM_F11_EVAL = [
    {
        'name' : 'center-crop',
        'size' : IMAGE_SIZE,
    },
]



args_dict = {
# Config
    'data' : {
        'train' : {
            'f11' : {
                'batch_size' : BATCH_SIZE,
                'dataset' : {
                    'name'       : 'cbed',
                    'path'       : DATA_PATH_F11,
                    'subdirs'    : F11_TRAIN_NAMES,
                    'csv_suffix' : None,
                    'target_columns' : TARGET_COLUMNS,
                },
                'shuffle' : True,
                'drop_last' : True,
                'shapes' : [ (1, *IMAGE_SIZE), (len(TARGET_COLUMNS),) ],
                'transform_image' : TRANSFORM_F11_TRAIN,
                'workers' : 4,
            },
        },
        'eval' : {
            'f11' : {
                'batch_size' : BATCH_SIZE,
                'dataset' : {
                    'name'       : 'cbed',
                    'path'       : DATA_PATH_F11,
                    'subdirs'    : F11_EVAL_NAMES,
                    'csv_suffix' : None,
                    'target_columns' : TARGET_COLUMNS,
                },
                'shuffle' : False,
                'drop_last' : False,
                'shapes' : [(1, *IMAGE_SIZE), (len(TARGET_COLUMNS),) ],
                'transform_image' : TRANSFORM_F11_EVAL,
                'workers' : 4,
            },
        },
    },
    'epochs' : EPOCHS,
    'model'  : {
        'name'          : 'regressor',
        'sched_metric'  : 'val_l2',
        'target_means'  : [ x[0] for x in TARGET_NORMS ],
        'target_stdevs' : [ x[1] for x in TARGET_NORMS ],
    },
    'nets' : {
        "reg": {
            'model' : {
                "name"       : 'tv-efficientnet',
                "model_type" : 'efficientnet-b0',
            },
        },
    },
    'losses'     : { 'reg' : 'mse', },
    'optimizers' : {
        'reg' : {
            'name' : 'AdamW',
            'lr'   : LR,
            'weight_decay' : 0,
        },
    },
    'schedulers' : {
        'reg' : {
            'name'  : 'cosine',
            'T_max' : EPOCHS,
        },
    },
    'val_interval' : 1,
    'steps_per_train_epoch' : 1000,
    'steps_per_val_epoch'   : 100,
# Args
    "label"      : 'optim(cosine)',
    'checkpoint' : 100,
    "outdir"     : (
        "cbed/f11/efficientnet/train_efficientnet_b0_crop320_augs"
    ),
}


setup_logging()
setup_and_train(args_dict)
