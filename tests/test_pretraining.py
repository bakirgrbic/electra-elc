import pytest

from pretraining.pretraining import (
    get_file_names,
    create_dataset,
    create_dataloader,
    pre_train_pipeline,
)

MODEL_NAME = "bsu-slim/electra-tiny"
SMALL_DATASET_SIZE = 100


@pytest.fixture
def pt_files():
    return get_file_names()


@pytest.fixture
def pt_dataset(pt_files):
    return create_dataset(pt_files, MODEL_NAME)


@pytest.fixture
def small_pt_dataset(pt_dataset):
    pt_dataset.decrease_length(SMALL_DATASET_SIZE)

    return pt_dataset


@pytest.fixture
def small_pt_dataloader(small_pt_dataset):
    BATCH_SIZE = 8

    return create_dataloader(small_pt_dataset, BATCH_SIZE)


def test_pretraining_exists(pt_files):
    assert len(pt_files) != 0, "No pretraining data found in data/train_10M/"


def test_confirm_full_pretraining_data_length(pt_dataset):
    FULL_DATASET_LENGTH = 1179020

    assert len(pt_dataset) == FULL_DATASET_LENGTH, (
        "Length of pretraining data is not the full 10M tokens for the strict small track."
    )


def test_decrease_length(small_pt_dataset):
    assert len(small_pt_dataset) == SMALL_DATASET_SIZE, (
        "Length of small pretraining data does not correspond to the desired amount of data."
    )


@pytest.mark.slow
def test_pre_train_pipeline_raise_no_error(small_pt_dataloader):
    EPOCHS = 1
    LEARNING_RATE = 2e-05

    pre_train_pipeline(
        model=MODEL_NAME,
        loader=small_pt_dataloader,
        epochs=EPOCHS,
        learning_rate=LEARNING_RATE,
        model_name=MODEL_NAME,
    )
