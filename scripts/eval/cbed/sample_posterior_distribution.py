import argparse
from collections import defaultdict
import json
import os

import numpy  as np
import pandas as pd
import tqdm
import torch

from cdi.config.data_config import unpack_dataset_config
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
        default  = None,
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
        help     = 'maximum number of observations to evaluate',
        type     = int,
        required = True,
    )

    parser.add_argument(
        '--skip',
        default  = 0,
        dest     = 'skip',
        help     = 'skip samples before evaluation',
        type     = int,
    )

    parser.add_argument(
        '--batch-size',
        default  = None,
        dest     = 'batch_size',
        help     = 'batch size to evaluate',
        type     = int,
    )

    parser.add_argument(
        '--distribution-size',
        default  = None,
        dest     = 'distribution_size',
        help     = 'distribution size to evaluate',
        type     = int,
        required = True,
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

def make_eval_directory(model, savedir, label, mkdir = True):
    result = os.path.join(savedir, 'evals')

    if model._epoch is None:
        result = os.path.join(result, 'final')
    else:
        result = os.path.join(result, f'epoch_{model._epoch}')

    result = os.path.join(result, f'distribution_{label}')

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

def repeat_batch(batch, batch_size):
    if isinstance(batch, (list, tuple)):
        return [ repeat_batch(x, batch_size) for x in batch ]

    if isinstance(batch, dict):
        return { k : repeat_batch(v, batch_size) for (k, v) in batch.items() }

    assert isinstance(batch, torch.Tensor)

    return batch.expand(batch_size, *batch.shape[1:])

def eval_preds(
    model, dl, distribution_size, batch_size, skip, limit, data_name, evaldir
):
    # pylint: disable=too-many-locals
    # pylint: disable=too-many-arguments
    model.eval()
    model.eval_epoch_start()

    sample_idx = 0
    it = iter(dl)

    for _ in range(skip):
        _ = next(it)
        sample_idx += 1

    pbar = tqdm.tqdm(
        desc = f'Eval ({data_name})', total = limit, dynamic_ncols = True
    )

    for _ in range(limit):
        batch        = next(it)
        results_dict = defaultdict(list)

        pbar.set_postfix({ 'curr_eval' : 0, 'sample' : sample_idx })

        for substep in range(0, distribution_size, batch_size):
            curr_batch_size = batch_size

            if (substep + batch_size) > distribution_size:
                curr_batch_size = distribution_size - substep

            curr_batch = repeat_batch(batch, curr_batch_size)

            model.set_inputs(curr_batch)

            output = model.predict_step()

            for (k, v) in output.items():
                results_dict[k].append(v.numpy(force = True))

            pbar.set_postfix({
                'curr_eval' : substep + curr_batch_size,
                'sample'    : sample_idx,
            })

        save_results(evaldir, sample_idx, results_dict)
        pbar.update()
        sample_idx += 1

    pbar.close()

def save_results(evaldir, sample_idx, results_dict):
    # pylint: disable=too-many-arguments
    fname = f'distribution_eval_sample({sample_idx}).csv'
    path  = os.path.join(evaldir, fname)

    results = format_results(results_dict)
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
    model, args, data_name, data_config, split, skip, limit, evaldir,
    distribution_size, batch_size, workers, data_path
):
    # pylint: disable=too-many-arguments

    data_config.batch_size = 1

    if workers is not None:
        data_config.workers = workers

    if data_path is not None:
        data_config.dataset['path'] = data_path

    args.config.data.eval = { data_name : data_config }
    dl = load_eval_dset(args, split = split)

    eval_preds(
        model, dl, distribution_size, batch_size, skip, limit, data_name,
        evaldir
    )

def main():
    cmdargs = parse_cmdargs()

    args, model = load_model(
        cmdargs.model, epoch = cmdargs.epoch, device = DEVICE
    )

    evaldir = make_eval_directory(model, cmdargs.model, cmdargs.label)

    if cmdargs.data_config is None:
        data_config_dict = args.config.data.eval
    else:
        data_config_dict = load_data_config(
            cmdargs.data_config, cmdargs.data_name
        )

    assert isinstance(data_config_dict, dict)
    split            = cmdargs.split

    data_config = data_config_dict[cmdargs.data_name]
    batch_size  = data_config.batch_size

    if cmdargs.batch_size:
        batch_size = cmdargs.batch_size

    eval_single_dataset(
        model, args, cmdargs.data_name, data_config, split, cmdargs.skip,
        cmdargs.limit, evaldir, cmdargs.distribution_size, batch_size,
        cmdargs.workers, cmdargs.data_path
    )

if __name__ == '__main__':
    main()
