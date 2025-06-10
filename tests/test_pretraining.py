from pathlib import Path

import pytest

from pretraining.dataset import Dataset


@pytest.fixture
def pt_files():
    return [str(data_file) for data_file in Path("data/train_10M").glob("[!._]*.train")]


@pytest.fixture
def pt_dataset(pt_files):
    return Dataset(pt_files, tokenizer=None)


def test_pretraining_exists(pt_files):
    assert len(pt_files) != 0, "No pretraining data found in data/train_10M/"


def test_confirm_full_pretraining_data_length(pt_dataset):
    DATASET_LENGTH = 1179020

    assert len(pt_dataset) == DATASET_LENGTH, (
        "Length of pretraining data is not the full 10M tokens for the strict small track."
    )
