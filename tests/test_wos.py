import pytest
from transformers import AutoTokenizer

from evaluation.web_of_science.wos import load_data, create_dataloaders, wos_evaluation

MODEL_NAME = "bsu-slim/electra-tiny"

TRAIN_DATA = 0
TRAIN_LABELS = 1
TEST_DATA = 2
TEST_LABELS = 3


@pytest.fixture
def wos_data():
    return load_data()


@pytest.fixture
def small_wos_data(wos_data):
    DESIRED_TRAIN_SAMPLES = 50
    DESIRED_TEST_SAMPLES = 10

    return (
        wos_data[TRAIN_DATA][:DESIRED_TRAIN_SAMPLES],
        wos_data[TRAIN_LABELS][:DESIRED_TRAIN_SAMPLES],
        wos_data[TEST_DATA][:DESIRED_TEST_SAMPLES],
        wos_data[TEST_LABELS][:DESIRED_TEST_SAMPLES],
    )


@pytest.fixture
def small_wos_dataloaders(small_wos_data):
    TOKENIZER = AutoTokenizer.from_pretrained(MODEL_NAME)
    MAX_LEN = 128
    BATCH_SIZE = 64

    return create_dataloaders(
        small_wos_data[TRAIN_DATA],
        small_wos_data[TRAIN_LABELS],
        small_wos_data[TEST_DATA],
        small_wos_data[TEST_LABELS],
        TOKENIZER,
        MAX_LEN,
        BATCH_SIZE,
    )


def test_load_non_empty_data(wos_data):
    TRAIN_lENGTH = 46000
    TEST_LENGTH = 985
    ERR_MESSAGE = "No evaluation data found in data/web_of_science/"

    assert len(wos_data[TRAIN_DATA]) == TRAIN_lENGTH, ERR_MESSAGE
    assert len(wos_data[TRAIN_LABELS]) == TRAIN_lENGTH, ERR_MESSAGE
    assert len(wos_data[TEST_DATA]) == TEST_LENGTH, ERR_MESSAGE
    assert len(wos_data[TEST_LABELS]) == TEST_LENGTH, ERR_MESSAGE


def test_first_sample_matches_for_train_and_data(wos_data):
    CHARS_TO_TEST = 25
    TRAIN_DATA_25_CHARS = "(2 + 1)-dimensional non-l"
    EXPECTED_TRAIN_LABEL = 0
    TEST_DATA_25_CHARS = "We report a case of CTX-M"
    EXPECTED_TEST_LABEL = 6
    FIRST_SAMPLE = 0

    assert wos_data[TRAIN_DATA][FIRST_SAMPLE][:CHARS_TO_TEST] == TRAIN_DATA_25_CHARS
    assert wos_data[TRAIN_LABELS][FIRST_SAMPLE] == EXPECTED_TRAIN_LABEL
    assert wos_data[TEST_DATA][FIRST_SAMPLE][:CHARS_TO_TEST] == TEST_DATA_25_CHARS
    assert wos_data[TEST_LABELS][FIRST_SAMPLE] == EXPECTED_TEST_LABEL


@pytest.mark.slow
def test_wos_evaluation_raise_no_error(small_wos_dataloaders):
    TRAIN = 0
    TEST = 0
    EPOCHS = 1
    LEARNING_RATE = 2e-05

    wos_evaluation(
        MODEL_NAME,
        small_wos_dataloaders[TRAIN],
        small_wos_dataloaders[TEST],
        EPOCHS,
        LEARNING_RATE,
    )
