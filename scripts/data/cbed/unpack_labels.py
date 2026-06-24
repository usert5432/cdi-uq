import argparse
import re
import os

import pandas as pd

DIR_REGEX    = re.compile(r'(\d+)\.ity$')
SAMPLE_REGEX = re.compile(r'^sample_(\d+)\.npz$')

LABEL_START_REGEX = re.compile(r'^\s*@list:\s*(\d+)(\s+.*)?\s*$')

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
        '--labels',
        dest     = 'path_labels',
        required = True,
        help     = 'Path to the file containing the original labels'
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

def parse_labels_file(path, n_config):
    result = []

    found_labels_start = False
    n_labels_total     = None

    with open(path, 'rt') as f:
        for line in f:
            if not found_labels_start:
                m = LABEL_START_REGEX.match(line)
                if m:
                    n_labels_total     = int(m.group(1))
                    found_labels_start = True

            else:
                tokens          = line.split()
                expected_n_read = int(tokens[0])

                if expected_n_read != len(result):
                    raise RuntimeError(
                        "Malformed file."
                        f" Number of labels read {len(result)}"
                        f", does not match file expectation {expected_n_read}"
                    )

                config = tokens[1:1+n_config]
                data   = [ float(x) for x in tokens[1+n_config:] ]

                result += [ (config, x) for x in data ]

                if len(result) == n_labels_total:
                    break

    return result

def match_labels_samples(samples, labels, n_config):
    # labels  : List[ (config, label) ]
    # samples : List[ dir_idx, file_idx, rel_path ]
    if len(samples) != len(labels):
        raise RuntimeError(
            f"Number of samples {len(samples)} does not match the number"
            f" of labels {len(labels)}"
        )

    result = [
        (rel_path, dir_idx, file_idx, *config, label)
            for ((dir_idx, file_idx, rel_path), (config, label)) in
                zip(samples, labels)
    ]

    header  = [ 'file', 'dir_idx', 'file_idx' ]
    header += [ f'conf_{idx}' for idx in range(n_config) ]
    header += [ 'label' ]

    return pd.DataFrame(result, columns = header)

def main():
    cmdargs = parse_cmdargs()

    print("Parsing labels...")
    labels  = parse_labels_file(cmdargs.path_labels, cmdargs.n_config)

    print("Collecting samples...")
    samples = gather_sample_tree(cmdargs.path_samples)

    print("Merging...")
    df = match_labels_samples(samples, labels, cmdargs.n_config)

    print("Saving...")
    df.to_csv(cmdargs.output, index = False)

if __name__ == '__main__':
    main()

