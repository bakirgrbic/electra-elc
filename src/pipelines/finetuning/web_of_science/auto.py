"""Defines Auto class used for web of science finetuning."""

import torch
from transformers import AutoModel


class AutoClass(torch.nn.Module):
    """Auto class for finetuning pre-trained models."""

    def __init__(self, model_name: str, num_out: int):
        """Constructor.

        Keyword Arguments:
        model_name -- relative file path of pre-trained model or name from
                      huggingface
        num_out -- number of classes to classify
        """
        super().__init__()
        self.transformer_layer = AutoModel.from_pretrained(model_name)
        self.classifier = torch.nn.Linear(196, num_out)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor):
        """Runs a piece of tokenized data through the model.

        Keyword Arguments:
        input_ids -- input_ids from a transformer tokenizer
        attention_mask -- attention mask for input_ids from transformer tokenizer

        Returns the softmax distribution for all classes
        """
        output_transformer = self.transformer_layer(
            input_ids=input_ids, attention_mask=attention_mask
        )
        last_hidden_state = output_transformer.last_hidden_state
        cls_pooler = last_hidden_state[:, 0]
        output = self.classifier(cls_pooler)

        return output
