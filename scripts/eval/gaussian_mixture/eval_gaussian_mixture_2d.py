#!/usr/bin/env python
import argparse

import matplotlib.pyplot as plt
import numpy as np
import torch
import tqdm

from cdi.eval.eval import load_eval_dset, load_model

def select_eval_config(config):
    for (name, data_cfg) in config.data.eval.items():
        if data_cfg.dataset['name'] == 'gaussian-mixture-2d':
            return (name, data_cfg)

    raise RuntimeError('No gaussian-mixture-2d eval dataset found.')

def select_dataset_from_loader(loader, data_name):
    if hasattr(loader, 'loaders'):
        return loader.loaders[data_name].dataset

    return loader.dataset

def sample_model(model, data_name, h_value, n_samples, batch_size):
    preds  = []
    n_done = 0
    pbar   = tqdm.tqdm(
        total = n_samples, desc = 'Sampling', dynamic_ncols = True
    )

    while n_done < n_samples:
        n_batch = min(batch_size, n_samples - n_done)

        condition = torch.full((n_batch, 1), h_value, dtype = torch.float32)
        target    = torch.zeros((n_batch, 2), dtype = torch.float32)

        model.set_inputs({ data_name : (condition, target) })

        pred_key    = f'preds_{data_name}'
        batch_preds = model.predict_step()[pred_key].detach().cpu().numpy()

        preds.append(batch_preds)

        n_done += n_batch
        pbar.update(n_batch)

    pbar.close()

    return np.concatenate(preds, axis = 0)

def eval_surface(dataset, data_cfg, grid_size):
    xy_range = tuple(float(x) for x in data_cfg.dataset.get(
        'xy_range', (-1.0, 1.0)
    ))
    coord  = np.linspace(*xy_range, grid_size, dtype = np.float32)
    yy, xx = np.meshgrid(coord, coord, indexing = 'ij')
    xy     = np.stack([ xx, yy ], axis = -1)

    gt_values = dataset.eval_surface_np(xy)

    return (xx, yy, gt_values, xy_range)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('savedir', metavar = 'SAVEDIR')

    parser.add_argument('--h-level',    type = float, default = 0.50)
    parser.add_argument('--n-samples',  type = int,   default = 8192)
    parser.add_argument('--batch-size', type = int,   default = 1024)
    parser.add_argument('--output', default = None)

    return parser.parse_args()

def plot_predictions(surface, preds, h_level):
    xx, yy, gt_values, xy_range = surface

    fig, ax = plt.subplots(figsize = (7, 7), constrained_layout = True)

    ax.contourf(xx, yy, gt_values, levels = 48, cmap = 'magma')
    ax.contour(
        xx, yy, gt_values,
        levels = [ h_level ],
        colors = 'cyan',
        linewidths = 2.5,
    )
    ax.scatter(
        preds[:, 0], preds[:, 1],
        s          = 6,
        c          = 'white',
        alpha      = 0.50,
        linewidths = 0,
    )

    ax.set_xlim(xy_range)
    ax.set_ylim(xy_range)
    ax.set_aspect('equal')
    ax.set_title(f'Gaussian mixture CDI samples, h = {h_level:.2f}')
    ax.set_xlabel('x')
    ax.set_ylabel('y')

    return fig


def main():
    args = parse_args()
    model_args, model = load_model(args.savedir, -1, device = 'cuda')
    data_name, data_cfg = select_eval_config(model_args.config)
    eval_loader = load_eval_dset(model_args, split = 'val')

    preds = sample_model(
        model, data_name, args.h_level, args.n_samples, args.batch_size
    )

    dset    = select_dataset_from_loader(eval_loader, data_name)
    surface = eval_surface(dset, data_cfg, grid_size = 512)
    fig     = plot_predictions(surface, preds, args.h_level)

    if args.output is None:
        plt.show()
    else:
        fig.savefig(args.output, dpi = 180)
        print(f'Wrote: {args.output}')


if __name__ == '__main__':
    main()
