#!/usr/bin/env python3
"""Script that pretrains local or huggingface models."""

import argparse
import logging

from transformers import AutoTokenizer

from src.tasks.pretraining.pretraining import (create_dataloader,
                                               create_dataset, get_file_names,
                                               pre_train_task)
from utils.log import create_save_dir, setup_logger


def get_parser() -> argparse.ArgumentParser:
    """Parser to read cli arguments."""
    parser = argparse.ArgumentParser(
        prog="python3 -m src.pretrain.py",
        description="""Script that pretrains local or huggingface models.""",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "-m",
        "--model_name",
        type=str,
        default="bsu-slim/electra-tiny",
        help="""Name of huggingface model or relative file path
                of a local model.""",
    )
    parser.add_argument(
        "-t",
        "--tokenizer_name",
        type=str,
        default="bsu-slim/electra-tiny",
        help="""Name of pre-trained huggingface tokenizer or relative file path
                to pre-trained local tokenizer. Models should match their tokenizers.""",
    )
    parser.add_argument(
        "-bs",
        "--batch_size",
        type=int,
        default=8,
        help="""Batch size to use before updating model weights.""",
    )
    parser.add_argument(
        "-e",
        "--epochs",
        type=int,
        default=10,
        help="""Number of epochs to pre-train for.""",
    )
    parser.add_argument(
        "-lr",
        "--learning_rate",
        type=float,
        default=1e-04,
        help="Learning rate for optimizer.",
    )

    return parser


def get_args() -> argparse.Namespace:
    """Read cli arguments."""
    parser = get_parser()

    return parser.parse_known_args()[0]


args = get_args()

logger = logging.getLogger("main")
save_dir = create_save_dir(args.model_name)
setup_logger(logger, save_dir)

logger.info("Setting up pre-train task")
tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name)

file_names = get_file_names()
dataset = create_dataset(file_names, tokenizer)
loader = create_dataloader(dataset, args.batch_size)

logger.info(
    f"Hyperparameters: batch_size={args.batch_size}, epochs={args.epochs}, learning_rate={args.learning_rate}"
)
pre_train_task(
    args.model_name,
    loader,
    args.epochs,
    args.learning_rate,
    save_dir=save_dir,
)
logger.info("End of pre-train task")
