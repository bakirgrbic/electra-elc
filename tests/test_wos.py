import unittest

from evaluation.web_of_science.wos import load_data


class TestWos(unittest.TestCase):
    def setUp(self):
        self.data = load_data()
        self.TRAIN_DATA = 0
        self.TRAIN_LABELS = 1
        self.TEST_DATA = 2
        self.TEST_LABELS = 3

    def test_load_non_empty_data(self):
        TRAIN_lENGTH = 46000
        TEST_LENGTH = 985
        ERR_MESSAGE = "No evaluation data found in data/web_of_science/"

        self.assertEqual(len(self.data[self.TRAIN_DATA]), TRAIN_lENGTH, ERR_MESSAGE)
        self.assertEqual(len(self.data[self.TRAIN_LABELS]), TRAIN_lENGTH, ERR_MESSAGE)
        self.assertEqual(len(self.data[self.TEST_DATA]), TEST_LENGTH, ERR_MESSAGE)
        self.assertEqual(len(self.data[self.TEST_LABELS]), TEST_LENGTH, ERR_MESSAGE)

    def test_first_sample_matches_for_train_and_data(self):
        CHARS_TO_TEST = 25
        TRAIN_DATA_25_CHARS = "(2 + 1)-dimensional non-l"
        EXPECTED_TRAIN_LABEL = 0
        TEST_DATA_25_CHARS = "We report a case of CTX-M"
        EXPECTED_TEST_LABEL = 6
        FIRST_SAMPLE = 0

        self.assertEqual(
            self.data[self.TRAIN_DATA][FIRST_SAMPLE][:CHARS_TO_TEST],
            TRAIN_DATA_25_CHARS,
        )
        self.assertEqual(
            self.data[self.TRAIN_LABELS][FIRST_SAMPLE], EXPECTED_TRAIN_LABEL
        )
        self.assertEqual(
            self.data[self.TEST_DATA][FIRST_SAMPLE][:CHARS_TO_TEST], TEST_DATA_25_CHARS
        )
        self.assertEqual(self.data[self.TEST_LABELS][FIRST_SAMPLE], EXPECTED_TEST_LABEL)
