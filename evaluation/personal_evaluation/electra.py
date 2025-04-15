import torch
from transformers import ElectraModel

class ELECTRAClass(torch.nn.Module):
    """ELECTRA class for finetuning pretrained ELECTRA models"""

    def __init__(self, model_path, num_out):
        """Constructor.

        Keyword Arguments:
        model_path -- relative file path of pre-trained model
        num_out -- number of classes to classify
        """
        super(ELECTRAClass, self).__init__()
        self.l1 = ElectraModel.from_pretrained(model_path)
        self.classifier = torch.nn.Linear(196, num_out)
        self.softmax = torch.nn.Softmax(dim=1)

    def forward(self, input_ids, attention_mask, token_type_ids):
        """Runs a piece of data through the model.

        Keyword Arguments:
        input_ids -- input_ids for a tokenized line of input
        attention_mask -- attention mask for input_ids, also from tokenizer
        token_type_ids -- token type ids as defined by return_outputs attribute

        Returns the softmax distribution for all classes

        """
        output_1 = self.l1(input_ids=input_ids, attention_mask=attention_mask)
        hidden_state = output_1[0]
        pooler = hidden_state[:, 0]
        output = self.classifier(pooler)
        output = self.softmax(output)
        return output
