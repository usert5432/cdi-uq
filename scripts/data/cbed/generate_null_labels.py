import argparse
import re
import os

import pandas as pd

DIR_REGEX    = re.compile(r'(?:\w*)?(\d+)(?:_.*)?\.ity$')
SAMPLE_REGEX = re.compile(r'^sample_(\d+)\.npz$')

def parse_cmdargs():
    parser = argparse.ArgumentParser(
        description     = 'Unpack labels',
        formatter_class = argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        '--samples',
        dest     = 'path_samples',
        required = True,
        help     = 'Directory containing unpacked sample file tree',
    )

    parser.add_argument(
        '--output',
        dest     = 'output',
        required = True,
        help     = 'Path where the output CSV file will be saved'
    )

    parser.add_argument(
        '-n', '--nconf',
        dest    = 'n_config',
        type    = int,
        default = 12,
        help    = 'Number of configuration parameters for each sample'
    )

    return parser.parse_args()

def gather_samples(dir_path):
    result = []

    for f in os.listdir(dir_path):
        match = SAMPLE_REGEX.match(f)
        if match:
            idx = int(match.group(1))
            result.append((idx, f))

    result.sort()
    return result

def gather_sample_tree(root_dir):
    result  = []
    subdirs = []

    for d in os.listdir(root_dir):
        match = DIR_REGEX.search(d)
        if match:
            idx = int(match.group(1))
            subdirs.append((idx, d))

    subdirs.sort()

    for dir_idx, dir_name in subdirs:
        dir_path = os.path.join(root_dir, dir_name)

        for sample_idx, file_name in gather_samples(dir_path):
            rel_path = os.path.join(dir_name, file_name)
            result.append((dir_idx, sample_idx, rel_path))

    return result

def generate_null_labels(samples, n_config):
    # samples : List[ (dir_idx, file_idx, rel_path) ]
    result = [
        (rel_path, dir_idx, file_idx) + (float("nan"),) * (n_config + 1)
            for (dir_idx, file_idx, rel_path) in samples
    ]

    header  = [ 'file', 'dir_idx', 'file_idx' ]
    header += [ f'conf_{idx}' for idx in range(n_config) ]
    header += [ 'label' ]

    return pd.DataFrame(result, columns = header)

def main():
    cmdargs = parse_cmdargs()

    print("Collecting samples...")
    samples = gather_sample_tree(cmdargs.path_samples)

    print("Creating null labels...")
    df = generate_null_labels(samples, cmdargs.n_config)

    print("Saving...")
    df.to_csv(cmdargs.output, index = False)

if __name__ == '__main__':
    main()

