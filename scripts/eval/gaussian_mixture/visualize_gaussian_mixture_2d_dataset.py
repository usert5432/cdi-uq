#!/usr/bin/env python
import argparse

import matplotlib.pyplot as plt
import numpy as np

from cdi.data.datasets.gaussian_mixture_2d import GaussianMixture2DDataset

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--split', default = 'train')

    parser.add_argument('--n-samples',  type = int, default = 50000)
    parser.add_argument('--train-seed', type = int, default = 0)
    parser.add_argument('--test-seed',  type = int, default = 100000)

    parser.add_argument(
        '--xy-range', type = float, nargs = 2, default = (-1.0, 1.0)
    )
    parser.add_argument('--grid-size', type = int, default = 256)
    parser.add_argument('--output', default = None)

    return parser.parse_args()

def collect_samples(dset, split, n_samples, xy_range, train_seed, test_seed):
    # pylint: disable=too-many-arguments
    seed    = train_seed if split == 'train' else test_seed
    rng     = np.random.default_rng(seed)
    xy      = rng.uniform(*xy_range, size = (n_samples, 2)).astype(np.float32)
    heights = dset.eval_surface_np(xy)

    return heights, xy

def make_surface_grid(dset, xy_range, grid_size):
    coord     = np.linspace(*xy_range, grid_size, dtype = np.float32)
    yy, xx    = np.meshgrid(coord, coord, indexing = 'ij')
    xy        = np.stack([ xx, yy ], axis = -1)
    gt_values = dset.eval_surface_np(xy)

    return xx, yy, gt_values

def plot_dataset(surface, samples):
    xx, yy, gt_values, xy_range = surface
    heights, xy = samples

    fig, ax = plt.subplots(figsize = (6, 6), constrained_layout = True)

    ax.contourf(xx, yy, gt_values, levels = 48, cmap = 'magma')
    ax.contour(
        xx, yy, gt_values,
        levels = [ 0.25, 0.40, 0.55, 0.70, 0.85 ],
        colors = 'cyan',
        linewidths = 1.2,
    )
    ax.scatter(
        xy[:, 0], xy[:, 1],
        s = 1,
        c = heights,
        cmap = 'viridis',
        alpha = 0.12,
        linewidths = 0,
    )

    ax.set_title('Gaussian mixture height')
    ax.set_xlim(xy_range)
    ax.set_ylim(xy_range)
    ax.set_aspect('equal')
    ax.set_xlabel('x')
    ax.set_ylabel('y')

    return fig

def make_dataset(args, xy_range):
    return GaussianMixture2DDataset(
        split      = args.split,
        n_samples  = args.n_samples,
        xy_range   = xy_range,
        train_seed = args.train_seed,
        test_seed  = args.test_seed,
    )

def main():
    args     = parse_args()
    xy_range = tuple(args.xy_range)
    dset     = make_dataset(args, xy_range)

    samples = collect_samples(
        dset, args.split, args.n_samples, xy_range,
        args.train_seed, args.test_seed,
    )
    xx, yy, gt_values = make_surface_grid(dset, xy_range, args.grid_size)
    surface           = (xx, yy, gt_values, xy_range)
    fig               = plot_dataset(surface, samples)

    if args.output is None:
        plt.show()
    else:
        fig.savefig(args.output, dpi = 180)
        print(f"Wrote: {args.output}")


if __name__ == '__main__':
    main()
