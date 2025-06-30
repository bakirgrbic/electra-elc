"""Implements BabyLM strict small data track pre-training task."""

import logging
from pathlib import Path

import numpy as np
import torch
import transformers
from tqdm.auto import tqdm
from transformers import AutoConfig, AutoModelForMaskedLM

from src.tasks.pretraining.dataset import Dataset

logger = logging.getLogger("main." + __name__)


def get_file_names() -> list[str]:
    """Gathers all pre-training data file names from data/train_10M dir."""

    return [
        str(data_file)

        for data_file in Path("data/train_10M").glob("[!._]*.train")
    ]


def create_dataset(
    data_files: list[str],
    tokenizer: transformers.AutoTokenizer,
) -> torch.utils.data.Dataset:
    """Create a datalset for pre-training.

    Parameters
    ----------
    data_files
        List of file names to get data from.
    tokenizer
        Transformer tokenizer.

    Returns
    -------
    dataset
        Overridden torch dataset object.
    """

    return Dataset(data_files, tokenizer=tokenizer)


def create_dataloader(
    dataset: torch.utils.data.Dataset,
    batch_size: int,
) -> torch.utils.data.DataLoader:
    """Create a dataloader for pre-training.

    Parameters
    ----------
    dataset
        Overridden torch Dataset object.
    batch_size
        Batch size to use before updating model weights.

    Returns
    -------
    loader
        Dataloader containing pre-training data.

    """

    return torch.utils.data.DataLoader(dataset, batch_size=batch_size)


def pre_train(
    model: AutoModelForMaskedLM,
    loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Adam,
    device: str,
    epochs: int,
) -> None:
    """Run main training loop.

    Parameters
    ----------
    model
        Transformer model to pretrain.
    loader
        dataloader containing pre-training data.
    optimizer
        Torch optimizer.
    device
        Which hardware device to use.
    epochs
        Number of epochs to pre-train for.
    """

    for epoch in range(epochs):
        loop = tqdm(loader, leave=True)
        model.train()
        losses = []
        logger.info(f"Begining pre-train epoch {epoch}")

        for batch in loop:
            optimizer.zero_grad()

            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(
                input_ids, attention_mask=attention_mask, labels=labels
            )

            loss = outputs.loss
            loss.backward()

            optimizer.step()

            loop.set_description(f"Epoch {epoch}")
            loop.set_postfix(loss=loss.item())
            losses.append(loss.item())

            del input_ids
            del attention_mask
            del labels

        logger.info(f"Epoch {epoch} Mean Training Loss: {np.mean(losses)}")


def pre_train_task(
    model_name: str,
    loader: torch.utils.data.DataLoader,
    epochs: int,
    learning_rate: float,
    save_dir: Path,
) -> None:
    """Run BabyLM pre-training task and logs artifacts.

    Parameters
    ----------
    model_name
        Name of huggingface model or relative file path of a local model.
    loader
        Torch data loader with pre-training data.
    epochs
        Number of epochs to pre-train for.
    learning_rate
        Learning rate for the optimizer.
    save_dir
        Directory to save model artifacts to. Log outputs to
        log/model_name/version_datetime dir.

    Returns
    -------
    None
        Saves model to save_dir.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"

    config = AutoConfig.from_pretrained(model_name)
    model = AutoModelForMaskedLM.from_config(config)
    model.to(device)

    optimizer = torch.optim.Adam(params=model.parameters(), lr=learning_rate)

    logger.info(f"Pre-training start with {device}")
    pre_train(model, loader, optimizer, device, epochs)
    logger.info("Pre-training done!")

    logger.info(f"Saving pre-trained model {model_name} to {save_dir}")
    model.save_pretrained(save_dir)
    logger.info(f"Saved {model_name}!")
