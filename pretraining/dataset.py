"""Torch dataset class for pre-training."""

import torch


class Dataset(torch.utils.data.Dataset):
    """Loads and preprocesses the given text data."""

    def __init__(self, paths: list[str], tokenizer):
        """Constructor.

        Keyword Arguments:
        paths -- relative file paths to for pre-train data
        tokenizer -- tokenizer from transformer lib
        """
        self.paths = paths
        self.tokenizer = tokenizer
        self.data = self._read_files()

    def __len__(self) -> int:
        """Returns length of entire dataset."""
        return len(self.data)

    def _read_files(self) -> list[str]:
        """Reads all files in the path of object."""
        data = []
        for path in self.paths:
            data.extend(self._read_file(path))
        return data

    def _read_file(self, path: str) -> list[str]:
        """Reads a given file."""
        with open(path, 'r', encoding='utf-8') as f:
            lines = f.read().split('\n')
        return lines

    def get_encoding(self, line: str) -> dict:
        """Creates an encoding for a given line of input."""
        batch = self.tokenizer(line, 
                               max_length=512, 
                               padding='max_length', 
                               truncation=True
        ) # tokenise all text

        labels = torch.tensor(batch['input_ids']) # Ground Truth
        mask = torch.tensor(batch['attention_mask']) # Attention Masks
        input_ids = labels.detach().clone() # Input to be masked
        rand = torch.rand(input_ids.shape)

        mask_arr = ((rand < .15) 
                     * (input_ids != 0) 
                     * (input_ids != 2) 
                     * (input_ids != 3)
        ) # with a probability of 15%, mask a given word, leave out CLS, SEP 
          # and PAD

        input_ids[mask_arr] = 4 # assign token 4 (=MASK)
        return {
            'input_ids': input_ids, 
            'attention_mask': mask, 
            'labels': labels
        }

    def __getitem__(self, index: int) -> dict:
        """Returns tokenized item at a given index."""
        return self.get_encoding(self.data[index])
