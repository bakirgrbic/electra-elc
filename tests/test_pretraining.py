import pytest
import torch
from transformers import AutoTokenizer

from pretraining.pretraining import (
    get_file_names,
    create_dataset,
    create_dataloader,
    pre_train_pipeline,
)
from pretraining.dataset import SpecialTokens

MODEL_NAME = "bsu-slim/electra-tiny"


@pytest.fixture
def tokenizer():
    return AutoTokenizer.from_pretrained(MODEL_NAME)


class TestPretraining:
    SMALL_DATASET_SIZE = 100

    @pytest.fixture
    def pt_files(self):
        return get_file_names()

    @pytest.fixture
    def pt_dataset(self, pt_files):
        return create_dataset(pt_files, MODEL_NAME)

    @pytest.fixture
    def small_pt_dataset(self, pt_dataset):
        pt_dataset.decrease_length(self.SMALL_DATASET_SIZE)

        return pt_dataset

    @pytest.fixture
    def small_pt_dataloader(self, small_pt_dataset):
        BATCH_SIZE = 8

        return create_dataloader(small_pt_dataset, BATCH_SIZE)

    def test_pretraining_exists(self, pt_files):
        assert len(pt_files) != 0, "No pretraining data found in data/train_10M/"

    def test_confirm_full_pretraining_data_length(self, pt_dataset):
        FULL_DATASET_LENGTH = 1179020

        assert len(pt_dataset) == FULL_DATASET_LENGTH, (
            "Length of pretraining data is not the full 10M tokens for the strict small track."
        )

    def test_decrease_length(self, small_pt_dataset):
        assert len(small_pt_dataset) == self.SMALL_DATASET_SIZE, (
            "Length of small pretraining data does not correspond to the desired amount of data."
        )

    @pytest.mark.slow
    def test_pre_train_pipeline_raise_no_error(self, small_pt_dataloader):
        EPOCHS = 1
        LEARNING_RATE = 2e-05

        pre_train_pipeline(
            model=MODEL_NAME,
            loader=small_pt_dataloader,
            epochs=EPOCHS,
            learning_rate=LEARNING_RATE,
            model_name=MODEL_NAME,
        )


class TestDataset:
    SHOULD_MASK_PROB = 0.5
    NO_MASK_PROB = 0.1
    RANDOM_TOKEN_TO_MASK = 848

    @pytest.fixture
    def empty_dataset(self, tokenizer):
        return create_dataset(data_files=[], tokenizer=tokenizer)

    @pytest.mark.parametrize(
        "input_id, mask_prob, expected",
        [
            pytest.param(
                SpecialTokens.PAD.value,
                SHOULD_MASK_PROB,
                SpecialTokens.PAD.value,
                id="pad_no_mask",
            ),
            pytest.param(
                SpecialTokens.PAD.value,
                NO_MASK_PROB,
                SpecialTokens.PAD.value,
                id="pad_no_mask_needed",
            ),
            pytest.param(
                SpecialTokens.UNK.value,
                SHOULD_MASK_PROB,
                SpecialTokens.UNK.value,
                id="unk_no_mask",
            ),
            pytest.param(
                SpecialTokens.UNK.value,
                NO_MASK_PROB,
                SpecialTokens.UNK.value,
                id="unk_no_mask_needed",
            ),
            pytest.param(
                SpecialTokens.CLS.value,
                SHOULD_MASK_PROB,
                SpecialTokens.CLS.value,
                id="cls_no_mask",
            ),
            pytest.param(
                SpecialTokens.CLS.value,
                NO_MASK_PROB,
                SpecialTokens.CLS.value,
                id="cls_no_mask_needed",
            ),
            pytest.param(
                SpecialTokens.SEP.value,
                SHOULD_MASK_PROB,
                SpecialTokens.SEP.value,
                id="sep_no_mask",
            ),
            pytest.param(
                SpecialTokens.SEP.value,
                NO_MASK_PROB,
                SpecialTokens.SEP.value,
                id="sep_no_mask_needed",
            ),
            pytest.param(
                RANDOM_TOKEN_TO_MASK,
                SHOULD_MASK_PROB,
                SpecialTokens.MASK.value,
                id="normal_token_mask",
            ),
            pytest.param(
                RANDOM_TOKEN_TO_MASK,
                NO_MASK_PROB,
                RANDOM_TOKEN_TO_MASK,
                id="normal_token_no_mask_needed",
            ),
        ],
    )
    def test_mask_ids(self, input_id, mask_prob, expected, empty_dataset):
        torch.manual_seed(0)  # Needed so torch generated probabilities are reproducible
        input_id = torch.tensor(input_id)
        expected = torch.tensor(expected)

        empty_dataset._mask_ids(input_id, mask_prob)

        assert input_id == expected
