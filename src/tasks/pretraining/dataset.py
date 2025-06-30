"""Dataset class for pre-training."""

from enum import Enum

import torch
import transformers


class SpecialTokens(Enum):
    PAD = 0
    UNK = 100
    CLS = 101
    SEP = 102
    MASK = 103


class Dataset(torch.utils.data.Dataset):
    """Loads and preprocesses the given text data."""

    def __init__(
        self,
        paths: list[str],
        tokenizer: transformers.AutoTokenizer,
    ):
        """Constructor.

        Keyword Arguments:
        paths -- file paths of pre-train data
        tokenizer -- transformer tokenizer
        """
        self.paths = paths
        self.tokenizer = tokenizer
        self.data = self._read_files()

    def __len__(self) -> int:
        """Returns the total sentences to pretrain on."""

        return len(self.data)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        """Returns tokenized sentence at a given index."""

        return self._get_encoding(self.data[index])

    def _read_files(self) -> list[str]:
        """Reads all files in the path of object."""
        data = []

        for path in self.paths:
            data.extend(self._read_file(path))

        return data

    def _read_file(self, path: str) -> list[str]:
        """Reads a given file."""
        with open(path, "r", encoding="utf-8") as f:
            lines = f.read().split("\n")

        return lines

    def _get_encoding(self, line: str) -> dict[str, torch.Tensor]:
        """Creates an encoding for a given line of input."""
        batch = self.tokenizer(
            line, max_length=512, padding="max_length", truncation=True
        )

        labels = torch.tensor(batch["input_ids"])
        mask = torch.tensor(batch["attention_mask"])
        input_ids = labels.detach().clone()

        mask_probability = 0.15
        self._mask_ids(input_ids, mask_probability)

        return {
            "input_ids": input_ids,
            "attention_mask": mask,
            "labels": labels,
        }

    def _mask_ids(
        self, input_ids: torch.Tensor, mask_probability: float
    ) -> None:
        """Masks tokens at random given a probability if they are not special tokens

        Keyword Arguments:
        input_ids -- input_ids from a transformer tokenizer
        mask_probability -- probability to mask any token in input_ids
        """
        rand = torch.rand(input_ids.shape)
        mask_arr = (
            (rand < mask_probability)
            * (input_ids != SpecialTokens.PAD.value)
            * (input_ids != SpecialTokens.UNK.value)
            * (input_ids != SpecialTokens.CLS.value)
            * (input_ids != SpecialTokens.SEP.value)
        )
        input_ids[mask_arr] = SpecialTokens.MASK.value

    def get_data(self) -> list[str]:
        return self.data

    def set_data(self, data: list[str]) -> None:
        self.data = data

    def decrease_length(self, new_data_length: int) -> None:
        """Reduces dataset size to only have up to new_data_length data in it."""
        data_subset = self.get_data()[:new_data_length]
        self.set_data(data_subset)
