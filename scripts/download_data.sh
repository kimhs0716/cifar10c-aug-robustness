#!/usr/bin/env bash
set -e

DATA_DIR="${1:-./data}"

echo "Downloading CIFAR-10-C to $DATA_DIR ..."
mkdir -p "$DATA_DIR"

if command -v aria2c &> /dev/null; then
    aria2c -x 16 -s 16 -k 1M \
        https://zenodo.org/record/2535967/files/CIFAR-10-C.tar \
        -o "$DATA_DIR/CIFAR-10-C.tar"
else
    curl -L https://zenodo.org/record/2535967/files/CIFAR-10-C.tar \
        -o "$DATA_DIR/CIFAR-10-C.tar"
fi

tar -xf "$DATA_DIR/CIFAR-10-C.tar" -C "$DATA_DIR/"
rm "$DATA_DIR/CIFAR-10-C.tar"

echo "Done. Files saved to $DATA_DIR/CIFAR-10-C/"
