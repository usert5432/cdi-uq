# CBED Data Preparation

This directory contains small helper scripts for converting raw PM CBED exports
into the CSV/NPZ layout used by the `cbed` dataset loader.

## Raw Data Layout

A raw CBED dataset directory is expected to look like:

```text
raw/f11a/
  f11a.txt
  f11a_cbed.txt
  f11a_cbed0000000.ity
  f11a_cbed0001000.ity
  f11a_cbed0002000.ity
  ...
```

Files:

- `f11a.txt`: PM dataset-generation parameters. Not used by these scripts.
- `f11a_cbed.txt`: PM configuration/label file used to create `labels.csv`.
- `f11a_cbed*.ity`: binary CBED image chunks.

## Unpack ITY Files

Each `.ity` file is unpacked into one directory of `sample_*.npz` files:

```bash
mkdir -p /path/to/unpacked/f11a

for src in /path/to/raw/f11a/*.ity; do
    python scripts/data/cbed/unpack_ity.py "$src" /path/to/unpacked/f11a
done
```

Result:

```text
unpacked/f11a/
  f11a_cbed0000000.ity/
    sample_00000.npz
    sample_00001.npz
    ...
  f11a_cbed0000000.ity_header.json
  f11a_cbed0001000.ity/
    sample_00000.npz
    ...
  f11a_cbed0001000.ity_header.json
```

## Create Labels

Convert the PM label file into the CSV schema used by the dataset loader:

```bash
python scripts/data/cbed/unpack_labels.py \
    --samples /path/to/unpacked/f11a \
    --labels /path/to/raw/f11a/f11a_cbed.txt \
    --output /path/to/unpacked/f11a/labels.csv
```

The output schema is:

```text
file, dir_idx, file_idx, conf_0, ..., conf_11, label
```

The `file` column points to unpacked samples, for example:

```text
f11a_cbed0000000.ity/sample_00000.npz
```

## Generate Null Labels

For real or unlabeled data, create a matching CSV with `NaN` configuration and
label values:

```bash
python scripts/data/cbed/generate_null_labels.py \
    --samples /path/to/unpacked/real_data \
    --output /path/to/unpacked/real_data/labels.csv
```

This keeps the same CSV schema, so unlabeled data can pass through the same
dataset and prediction code paths.

## Split Labels

Split a label CSV into train/validation/test CSV files:

```bash
python scripts/data/cbed/split_labels.py \
    /path/to/unpacked/f11a/labels.csv \
    --shuffle \
    --seed 0
```

This creates:

```text
labels_train.csv
labels_val.csv
labels_test.csv
```

If the full dataset is intendend to be use for a specific split, link or copy
`labels.csv` to the expected split name, e.g:

```bash
ln -s labels.csv /path/to/unpacked/f11a/labels_train.csv
```

Use this for real/unlabeled data or quick inspection runs where a train/val/test
partition is not meaningful.

## Loader Layout

For the `cbed` dataset loader, place data under `$CDI_DATA` like:

```text
$CDI_DATA/cbed/f11/
  f11a/
    labels_train.csv
    labels_val.csv
    labels_test.csv
    f11a_cbed0000000.ity/
      sample_00000.npz
      ...
```

The scripts sort `.ity` directories and `sample_*.npz` files numerically before
matching labels to samples.

