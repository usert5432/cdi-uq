#!/usr/bin/env python

import argparse
import os

import numpy as np
import pandas as pd

def parse_cmdargs():
    parser = argparse.ArgumentParser(
        description     = "Split labels into train/test/val partitions",
        formatter_class = argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        'path',
        help    = 'Path to the labels file',
        metavar = 'PATH',
        type    = str,
    )

    parser.add_argument(
        '--shuffle',
        action  = 'store_true',
        dest    = 'shuffle',
        help    = 'Whether to shuffle data',
    )

    parser.add_argument(
        '--suffix',
        dest     = 'suffix',
        default  = None,
        help     = 'optional suffix to append to the label files',
        type     = str,
    )

    parser.add_argument(
        '-s', '--seed',
        default = 0,
        dest    = 'seed',
        help    = 'seed of the rng',
        type    = int,
    )

    parser.add_argument(
        '--test-size',
        default = 0.2,
        dest    = 'test_size',
        help    = (
            'Size of the test dataset.'
            'If test-size <= 1, then it is interpreted as a fraction'
        ),
        type    = float,
    )

    parser.add_argument(
        '--val-size',
        default = 0.1,
        dest    = 'val_size',
        help    = (
            'Size of the validation dataset.'
            'If val-size <= 1, then it is interpreted as a fraction'
        ),
        type    = float,
    )

    return parser.parse_args()

def load_dataset(path):
    return pd.read_csv(path, index_col = None)

def train_val_test_split(n, val_size, test_size, shuffle, prg):
    indices = np.arange(n)

    if shuffle:
        prg.shuffle(indices)

    if test_size <= 1:
        test_size = len(indices) * test_size

    if val_size <= 1:
        val_size = len(indices) * val_size

    test_size  = int(test_size)
    val_size   = int(val_size)
    train_size = max(0, len(indices) - val_size - test_size)

    train_indices = indices[:train_size]
    val_indices   = indices[train_size:train_size+val_size]
    test_indices  = indices[train_size+val_size:]

    return (train_indices, val_indices, test_indices)

def construct_save_paths(path, suffix):
    path_base, ext = os.path.splitext(path)

    if suffix is None:
        return (
            path_base + '_train' + ext,
            path_base + '_val'   + ext,
            path_base + '_test'  + ext,
        )

    return (
        path_base + '_train_' + suffix + ext,
        path_base + '_val_'   + suffix + ext,
        path_base + '_test_'  + suffix + ext,
    )

def main():
    cmdargs = parse_cmdargs()

    print("Loading labels...")
    df  = pd.read_csv(cmdargs.path)
    prg = np.random.default_rng(cmdargs.seed)

    print("Splitting labels...")
    split_indices = train_val_test_split(
        len(df), cmdargs.val_size, cmdargs.test_size, cmdargs.shuffle, prg
    )

    split_paths = construct_save_paths(cmdargs.path, cmdargs.suffix)

    for (indices, path) in zip(split_indices, split_paths):
        print(f"Saving {path}...")
        curr_df = df.iloc[indices]
        curr_df.to_csv(path)

if __name__ == '__main__':
    main()
