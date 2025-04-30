#!/usr/bin/env python3
"""Script that pretrains ELECTRA_ELC model."""

import argparse
from pathlib import Path

from transformers import ElectraTokenizerFast
from transformers import ElectraConfig

from models.electra_elc import ElectraForMaskedLM
from pretraining.pretraining import pre_train


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="""Script that pretrains 
                                                    ELECTRA_ELC model."""
    )
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-04, help="learning rate")
    return parser


def get_args() -> argparse.Namespace:
    parser = get_parser()
    return parser.parse_known_args()[0]

args = get_args()

model_name = "ELECTRA_ELC"
hf_name = "bsu-slim/electra-tiny"
tokenizer = ElectraTokenizerFast.from_pretrained(hf_name)
config = ElectraConfig.from_pretrained(hf_name)
model = ElectraForMaskedLM(config)

pre_train(tokenizer, model, args.epochs, args.lr, model_name)
