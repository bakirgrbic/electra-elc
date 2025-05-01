#!/usr/bin/env python3
"""Script that runs web of science evaluation pipeline."""

import argparse
from pathlib import Path

from transformers import AutoTokenizer
from transformers import AutoModel

from evaluation.web_of_science.wos import wos_evaluation


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="""Script that runs web of 
                                                    science evaluation pipeline."""
    )
    parser.add_argument("--max-len", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=2e-05, help="learning rate")
    parser.add_argument(
            "--model", 
            type=str, 
            default="bsu-slim/electra-tiny", 
            help="""relative file path of pre-trained model or name from 
                  huggingface"""
    )
    parser.add_argument("--tokenizer-config", type=str, default="bsu-slim/electra-tiny")
    return parser


def get_args() -> argparse.Namespace:
    parser = get_parser()
    return parser.parse_known_args()[0]

args = get_args()

model_name = Path(args.model).parts[-1]
tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_config)
model = AutoModel.from_pretrained(args.model)

wos_evaluation(model, 
               tokenizer, 
               args.max_len, 
               args.batch_size, 
               args.epochs, 
               args.lr, 
               model_name
)
