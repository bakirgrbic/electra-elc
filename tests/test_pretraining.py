import unittest
from pathlib import Path

from pretraining.dataset import Dataset


class TestPretraining(unittest.TestCase):
    def setUp(self):
        self.data_files = [
            str(data_file) for data_file in Path("data/train_10M").glob("[!._]*.train")
        ]

    def test_load_non_empty_pretraining_data(self):
        self.assertNotEqual(
            len(self.data_files), 0, "No pretraining data found in data/train_10M/"
        )

    def test_confirm_full_pretraining_data_length(self):
        DATASET_LENGTH = 1179020
        dataset = Dataset(self.data_files, tokenizer=None)

        self.assertEqual(
            len(dataset),
            DATASET_LENGTH,
            "Length of pretraining data is not the full 10M tokens for the strict small track.",
        )
