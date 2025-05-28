"""Dataset class used for web of science evaluation."""

import torch
import numpy as np


class MultiLabelDataset(torch.utils.data.Dataset):
    """Loads and preprocesses the given text data."""

    def __init__(
        self, text: list[str], labels: np.ndarray, tokenizer, max_len: int
    ) -> None:
        """Constructor.

        Keyword Arguments:
        text -- text to construct dataset
        labels -- corresponding labels for each text
        tokenizer -- transformer tokenizer
        max_len -- controls maximum length of tokenizer to truncate or pad input
        """
        self.text = text
        self.targets = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self) -> int:
        """Returns the total number of documents to finetune on."""
        return len(self.text)

    def __getitem__(self, index: int) -> dict[str, torch.LongTensor]:
        """Returns text from specific index with tokenized information."""
        text = self.text[index]
        inputs = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_token_type_ids=False,
        )

        ids = inputs["input_ids"]
        mask = inputs["attention_mask"]

        return {
            "ids": torch.tensor(ids, dtype=torch.long),
            "mask": torch.tensor(mask, dtype=torch.long),
            "targets": self.targets[index].clone().detach(),
        }
