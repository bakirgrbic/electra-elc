#!/usr/bin/env python3
"""Script that runs web of science evaluation pipeline."""

import argparse

from transformers import AutoTokenizer

from evaluation.web_of_science.wos import load_data, create_dataloaders, wos_evaluation


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="""Script that runs web of
                           science evaluation pipeline."""
    )
    parser.add_argument(
        "--model",
        type=str,
        default="bsu-slim/electra-tiny",
        help="""relative file path of pre-trained model or name from
                    huggingface. Default is electra-tiny.""",
    )
    parser.add_argument(
        "--tokenizer-config",
        type=str,
        default="bsu-slim/electra-tiny",
        help="""relative file path of pre-trained model or name from
                    huggingface. Default is electra-tiny.""",
    )
    parser.add_argument("--max-len", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=2e-05, help="learning rate")

    return parser


def get_args() -> argparse.Namespace:
    parser = get_parser()

    return parser.parse_known_args()[0]


args = get_args()

model_name = args.model
tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_config)

train_data, train_labels, test_data, test_labels = load_data()

training_loader, testing_loader = create_dataloaders(
    train_data,
    train_labels,
    test_data,
    test_labels,
    tokenizer,
    args.max_len,
    args.batch_size,
)


wos_evaluation(
    model_name,
    training_loader,
    testing_loader,
    args.epochs,
    args.lr,
)
