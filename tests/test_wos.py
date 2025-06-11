import pytest
from transformers import AutoTokenizer

from evaluation.web_of_science.wos import load_data, create_dataloaders, wos_evaluation

MODEL_NAME = "bsu-slim/electra-tiny"

@pytest.fixture
def wos_data():
    return load_data()


@pytest.fixture
def small_wos_data(wos_data):
    DESIRED_TRAIN_SAMPLES = 50
    DESIRED_TEST_SAMPLES = 10
    train_data, train_labels, test_data, test_labels = wos_data

    return (
        train_data[:DESIRED_TRAIN_SAMPLES],
        train_labels[:DESIRED_TRAIN_SAMPLES],
        test_data[:DESIRED_TEST_SAMPLES],
        test_labels[:DESIRED_TEST_SAMPLES],
    )


@pytest.fixture
def small_wos_dataloaders(small_wos_data):
    TOKENIZER = AutoTokenizer.from_pretrained(MODEL_NAME)
    MAX_LEN = 128
    BATCH_SIZE = 64
    train_data, train_labels, test_data, test_labels = small_wos_data

    return create_dataloaders(
        train_data=train_data,
        train_labels=train_labels,
        test_data=test_data,
        test_labels=test_labels,
        tokenizer=TOKENIZER,
        max_len=MAX_LEN,
        batch_size=BATCH_SIZE,
    )


def test_load_non_empty_data(wos_data):
    TRAIN_lENGTH = 46000
    TEST_LENGTH = 985
    ERR_MESSAGE = "No evaluation data found in data/web_of_science/"
    train_data, train_labels, test_data, test_labels = wos_data

    assert len(train_data) == TRAIN_lENGTH, ERR_MESSAGE
    assert len(train_labels) == TRAIN_lENGTH, ERR_MESSAGE
    assert len(test_data) == TEST_LENGTH, ERR_MESSAGE
    assert len(test_labels) == TEST_LENGTH, ERR_MESSAGE


def test_first_sample_matches_for_train_and_data(wos_data):
    CHARS_TO_TEST = 25
    TRAIN_DATA_25_CHARS = "(2 + 1)-dimensional non-l"
    EXPECTED_TRAIN_LABEL = 0
    TEST_DATA_25_CHARS = "We report a case of CTX-M"
    EXPECTED_TEST_LABEL = 6
    FIRST_SAMPLE = 0
    train_data, train_labels, test_data, test_labels = wos_data

    assert train_data[FIRST_SAMPLE][:CHARS_TO_TEST] == TRAIN_DATA_25_CHARS
    assert train_labels[FIRST_SAMPLE] == EXPECTED_TRAIN_LABEL
    assert test_data[FIRST_SAMPLE][:CHARS_TO_TEST] == TEST_DATA_25_CHARS
    assert test_labels[FIRST_SAMPLE] == EXPECTED_TEST_LABEL


@pytest.mark.slow
def test_wos_evaluation_raise_no_error(small_wos_dataloaders):
    TRAIN = 0
    TEST = 0
    EPOCHS = 1
    LEARNING_RATE = 2e-05

    wos_evaluation(
        model_name=MODEL_NAME,
        training_loader=small_wos_dataloaders[TRAIN],
        testing_loader=small_wos_dataloaders[TEST],
        epochs=EPOCHS,
        learning_rate=LEARNING_RATE,
    )
