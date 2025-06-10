import pytest

from evaluation.web_of_science.wos import load_data


TRAIN_DATA = 0
TRAIN_LABELS = 1
TEST_DATA = 2
TEST_LABELS = 3


@pytest.fixture
def wos_data():
    return load_data()


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
