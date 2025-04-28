# Checkpoints
Stores models after they have finished pre-training.

Models:
- ELECTRA-PT: tiny ELECTRA Model https://huggingface.co/bsu-slim/electra-tiny
- ELECTRA-OE: tiny ELECTRA Model trained only on 1 epoch
- ELECTRA-ELC-OE: modified tiny ELECTRA Model that includeds layer weighing [ELC-BERT zero initialization](https://github.com/ltgoslo/elc-bert/blob/main/models/model_elc_bert_zero.py) trained only on 1 epoch
