import pytest
import torch
from transformers import AutoTokenizer

from pretraining.dataset import SpecialTokens
from pretraining.pretraining import (create_dataloader, create_dataset,
                                     get_file_names, pre_train_pipeline)

MODEL_NAME = "bsu-slim/electra-tiny"


@pytest.fixture(scope="module")
def tokenizer():
    return AutoTokenizer.from_pretrained(MODEL_NAME)


class TestPretraining:
    SMALL_DATASET_LENGTH = 100
    FULL_DATASET_LENGTH = 1179020

    @pytest.fixture(scope="class")
    def pt_files(self):
        return get_file_names()

    @pytest.fixture(scope="class")
    def pt_dataset(self, pt_files, tokenizer):
        return create_dataset(pt_files, tokenizer)

    @pytest.fixture(scope="class")
    def small_pt_dataset(self, pt_dataset):
        pt_dataset.decrease_length(self.SMALL_DATASET_LENGTH)

        return pt_dataset

    @pytest.fixture(scope="class")
    def small_pt_dataloader(self, small_pt_dataset):
        BATCH_SIZE = 8

        return create_dataloader(small_pt_dataset, BATCH_SIZE)

    def test_pretraining_exists(self, pt_files):
        assert len(pt_files) != 0, (
            "No pretraining data found in data/train_10M/"
        )

    @pytest.mark.parametrize(
        "dataset, expected_length",
        [
            pytest.param(
                "pt_dataset",
                FULL_DATASET_LENGTH,
                id="full_dataset",
            ),
            pytest.param(
                "small_pt_dataset",
                SMALL_DATASET_LENGTH,
                id="small_dataset",
            ),
        ],
    )
    def test_length(self, dataset, expected_length, request):
        dataset = request.getfixturevalue(dataset)
        assert len(dataset) == expected_length

    @pytest.mark.slow
    def test_pre_train_pipeline_raise_no_error(self, small_pt_dataloader):
        EPOCHS = 1
        LEARNING_RATE = 2e-05

        pre_train_pipeline(
            model_name=MODEL_NAME,
            loader=small_pt_dataloader,
            epochs=EPOCHS,
            learning_rate=LEARNING_RATE,
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
        torch.manual_seed(
            0
        )  # Needed so torch generated probabilities are reproducible
        input_id = torch.tensor(input_id)
        expected = torch.tensor(expected)

        empty_dataset._mask_ids(input_id, mask_prob)

        assert input_id == expected
