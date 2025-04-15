import torch

class MultiLabelDataset(torch.utils.data.Dataset):
    """Loads and preprocesses the given text data."""

    def __init__(self, text, labels, tokenizer, max_len):
        """Constructor.

        Keyword Arguments:
        text -- text to construct dataset
        labels -- corresponding labels for each text
        tokenizer -- tokenizer from transformer lib
        max_len -- controls maximum length of tokenizer to truncate or 
                   pad input
        """
        self.tokenizer = tokenizer
        self.text = text
        self.targets = labels
        self.max_len = max_len

    def __len__(self) -> int:
        return len(self.text)

    def __getitem__(self, int: index) -> dict:
        """Tokenizes text at a given index."""
        text = self.text[index]
        inputs = self.tokenizer.encode_plus(
            text,
            None,
            add_special_tokens=True,
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_token_type_ids=True
        )

        ids = inputs['input_ids']
        mask = inputs['attention_mask']
        token_type_ids = inputs["token_type_ids"]

        return {
            'ids': torch.tensor(ids, dtype=torch.long),
            'mask': torch.tensor(mask, dtype=torch.long),
            'token_type_ids': torch.tensor(token_type_ids, dtype=torch.long),
            'targets': torch.tensor(self.targets[index], dtype=torch.long)
        }
