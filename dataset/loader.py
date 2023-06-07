#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Data loader."""

import os

import torch
from dataset.dataset import DataSet


# Default data directory (/path/pycls/pycls/datasets/data)

def _construct_loader(_DATA_DIR, dataset_name, fn, split, transform, batch_size, shuffle, drop_last):
    """Constructs the data loader for the given dataset."""
    # Construct the dataset
    dataset = DataSet(_DATA_DIR, dataset_name, fn, split, transform)
    # Create a loader
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        sampler=None,
        num_workers=4,
        pin_memory=False,
        drop_last=drop_last,
    )
    return loader

def construct_loader(_DATA_DIR, dataset_name, fn, split, transform, batch_size):
    """Test loader wrapper."""
    return _construct_loader(
        _DATA_DIR=_DATA_DIR,
        dataset_name=dataset_name,
        fn=fn,
        split=split,
        transform = transform,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
    )
