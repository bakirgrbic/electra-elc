#!/usr/bin/env python3
"""Script that pretrains models from huggingface."""

import argparse
from pathlib import Path

from transformers import AutoTokenizer
from transformers import AutoConfig
from transformers import AutoModelForMaskedLM

from pretraining.pretraining import pre_train


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="""Script that pretrains models 
                                                    from huggingface."""
    )
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-04, help="learning rate")
    parser.add_argument(
            "--model-config", 
            type=str, 
            default="bsu-slim/electra-tiny", 
            help="name of model to pull from huggingface"
    )
    parser.add_argument("--tokenizer-config", type=str, default="bsu-slim/electra-tiny")
    return parser


def get_args() -> argparse.Namespace:
    parser = get_parser()
    return parser.parse_known_args()[0]

args = get_args()

model_name = Path(args.model_config).parts[-1]
tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_config)
config = AutoConfig.from_pretrained(args.model_config)
model = AutoModelForMaskedLM.from_config(config)

pre_train(tokenizer, model, args.epochs, args.lr, model_name)
