from cdi.train.train import setup_and_train
from cdi.utils.logging import setup_logging


args_dict = {
# Config
    'data' : {
        'train' : {
            'toy' : {
                'batch_size' : 256,
                'dataset' : {
                    'name'       : 'gaussian-mixture-2d',
                    'n_samples'  : 32768,
                    'train_seed' : 0,
                    'test_seed'  : 100_000,
                },
                'shuffle'         : True,
                'drop_last'       : True,
                'shapes'          : [ (1,), (2,) ],
                'transform_image' : None,
                'workers'         : 0,
            },
        },
        'eval' : {
            'toy' : {
                'batch_size' : 256,
                'dataset'    : {
                    'name'       : 'gaussian-mixture-2d',
                    'n_samples'  : 4096,
                    'train_seed' : 0,
                    'test_seed'  : 100_000,
                },
                'shuffle'         : False,
                'drop_last'       : False,
                'shapes'          : [ (1,), (2,) ],
                'transform_image' : None,
                'workers'         : 0,
            },
        },
    },
    'epochs' : 100,
    'model' : {
        'name'          : 'cdi',
        'sched_metric'  : 'val_l2',
        'target_means'  : [ 0.0, 0.0 ],
        'target_stdevs' : [ 1.0, 1.0 ],
        'ema_momentum'  : 0.999,
        'seed'          : 0,
        'vsched'        : {
            'name'  : 'linear',
            'T'     : 128,
            'beta1' : 1e-4,
            'betaT' : 0.02,
        },
    },
    'nets' : {
        'encoder' : {
            'model' : {
                'name'            : 'mlp-condition',
                'features'        : 32,
                'hidden_features' : 64,
                'n_hidden'        : 2,
                'activ'           : 'silu',
                'n_tokens'        : 1,
            },
        },
        'gen' : {
            'model' : {
                'name'          : 'param-cond-trans',
                'features_cond' : 32,
                'features'      : 96,
                'features_ffn'  : 192,
                'n_heads'       : 4,
                'n_layers'      : 4,
                'norm_first'    : True,
                'time_embed'    : 'linear',
            },
        },
    },
    'losses' : { 'reg' : 'mse' },
    'optimizers' : {
        'main' : {
            'name' : 'AdamW',
            'lr'   : 1e-3,
            'weight_decay' : 0.0,
        },
    },
    'schedulers' : { 'main' : None },
    'seed'                  : 0,
    'val_interval'          : 1,
    'steps_per_train_epoch' : 128,
    'steps_per_val_epoch'   : 16,
# Args
    'label'      : 'cdi',
    'checkpoint' : 10,
    'outdir'     : 'gaussian_mixture/gaussian_mixture_2d',
}


setup_logging()
setup_and_train(args_dict, device = 'cuda')

