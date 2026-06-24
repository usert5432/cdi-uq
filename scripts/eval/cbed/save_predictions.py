import argparse
from collections import defaultdict
from itertools   import islice
import json
import os

import numpy  as np
import pandas as pd
import tqdm

from cdi.config.data_config import unpack_dataset_config
from cdi.train.train import infer_steps
from cdi.eval.eval   import load_model, load_eval_dset

DEVICE = 'cuda'

def parse_cmdargs():
    parser = argparse.ArgumentParser(description = 'Evaluate simple model')

    parser.add_argument(
        'model',
        metavar  = 'MODEL',
        help     = 'model directory',
        type     = str,
    )

    parser.add_argument(
        '-e', '--epoch',
        default  = None,
        dest     = 'epoch',
        help     = 'epoch',
        type     = int,
    )

    parser.add_argument(
        '--data-name',
        dest     = 'data_name',
        help     = 'name of the dataset to use',
        type     = str,
        required = True,
    )

    parser.add_argument(
        '--data-path',
        default  = None,
        dest     = 'data_path',
        help     = 'path to the new dataset to evaluate',
        type     = str,
    )

    parser.add_argument(
        '--data-config',
        default  = None,
        dest     = 'data_config',
        help     = 'path to the eval data configuration',
        type     = str,
    )

    parser.add_argument(
        '--split',
        default  = 'test',
        dest     = 'split',
        help     = 'dataset split',
        type     = str,
    )

    parser.add_argument(
        '--limit',
        default  = None,
        dest     = 'limit',
        help     = 'maximum number of batches to evaluate',
        type     = int,
    )

    parser.add_argument(
        '--batch-size',
        default  = None,
        dest     = 'batch_size',
        help     = 'batch size for evaluation',
        type     = int,
    )

    parser.add_argument(
        '--workers',
        default  = None,
        dest     = 'workers',
        help     = 'number of workers to use for evaluation',
        type     = int,
    )

    parser.add_argument(
        '--label',
        dest     = 'label',
        help     = 'evaluation output label',
        type     = str,
        required = True,
    )


    return parser.parse_args()

def make_eval_directory(model, savedir, mkdir = True):
    result = os.path.join(savedir, 'evals')

    if model._epoch is None:
        result = os.path.join(result, 'final')
    else:
        result = os.path.join(result, f'epoch_{model._epoch}')

    if mkdir:
        os.makedirs(result, exist_ok = True)

    return result

def load_data_config(path, data_name):
    with open(path, 'rt', encoding = 'utf-8') as f:
        data_config = unpack_dataset_config(json.load(f))

    if isinstance(data_config, dict):
        return data_config

    assert data_name is not None
    return { data_name : data_config }

def eval_preds(model, dl, steps_per_epoch, data_name):
    model.eval()
    model.eval_epoch_start()

    steps   = infer_steps(dl, steps_per_epoch)
    progbar = tqdm.tqdm(
        desc = f'Eval ({data_name})', total = steps, dynamic_ncols = True
    )

    result = defaultdict(list)

    for batch in islice(dl, steps):
        model.set_inputs(batch)

        output = model.predict_step()

        for (k, v) in output.items():
            result[k].append(v.numpy(force = True))

        progbar.update()

    progbar.close()

    return result

def save_results(evaldir, label, results):
    path = os.path.join(evaldir, f'predictions_{label}.csv')
    results.to_csv(path, index = False)

def format_results(results_dict):
    result = {}

    for (k, v) in results_dict.items():
        flat_data = np.concatenate(v, axis = 0)

        if flat_data.shape[1] == 1:
            result[k] = flat_data[:, 0]
        else:
            for col_idx in range(flat_data.shape[1]):
                result[f'{k}_{col_idx}'] = flat_data[:, col_idx]

    return pd.DataFrame(result)

def eval_single_dataset(
    model, args, data_name, data_config, split, limit, evaldir, batch_size,
    workers, data_path, label
):
    # pylint: disable=too-many-arguments

    if batch_size is not None:
        data_config.batch_size = batch_size

    if workers is not None:
        data_config.workers = workers

    if data_path is not None:
        data_config.dataset['path'] = data_path

    args.config.data.eval = { data_name : data_config }
    dl = load_eval_dset(args, split = split)

    results_dict = eval_preds(
        model, dl, steps_per_epoch = limit, data_name = data_name
    )
    results      = format_results(results_dict)

    save_results(evaldir, label, results)

def main():
    cmdargs = parse_cmdargs()

    args, model = load_model(
        cmdargs.model, epoch = cmdargs.epoch, device = DEVICE
    )

    evaldir = make_eval_directory(model, cmdargs.model)

    if cmdargs.data_config is None:
        data_config_dict = args.config.data.eval
    else:
        data_config_dict = load_data_config(
            cmdargs.data_config, cmdargs.data_name
        )

    assert isinstance(data_config_dict, dict)
    split            = cmdargs.split

    eval_single_dataset(
        model, args, cmdargs.data_name, data_config_dict[cmdargs.data_name],
        split, cmdargs.limit, evaldir, cmdargs.batch_size,
        cmdargs.workers, cmdargs.data_path, cmdargs.label
    )

if __name__ == '__main__':
    main()
