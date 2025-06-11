"""Implements pretraining pipeline on BabyLM strict small data."""

from pathlib import Path

import numpy as np
import transformers
import torch
from tqdm.auto import tqdm

from log.my_logger import get_my_logger, log
from pretraining.dataset import Dataset


def get_file_names() -> list[str]:
    """Gathers all pre-training data file names from data/train_10M dir."""

    return [str(data_file) for data_file in Path("data/train_10M").glob("[!._]*.train")]


def create_dataset(
    data_files: list[str],
    tokenizer: transformers.AutoTokenizer,
):
    """Creates a datalset for pre-training.

    Keyword Arguments:
    data_files -- list of file names to get data from
    tokenizer -- transformer tokenizer
    """

    return Dataset(data_files, tokenizer=tokenizer)


def create_dataloader(
    dataset: torch.utils.data.Dataset,
    batch_size: int,
) -> torch.utils.data.DataLoader:
    """Creates a dataloader for pre-training.

    Keyword Arguments:
    dataset -- overridden torch Dataset object.
    batch_size -- size of batches to be fed to model for finetuning
    """

    return torch.utils.data.DataLoader(dataset, batch_size=batch_size)


def pre_train(model, loader, optimizer, device, epochs, logger):
    """Main training loop.

    Keyword Arguments:
    model -- model to pretrain
    loader -- dataloader containing pre-training data
    optimizer -- torch optimizer
    device -- which hardware device to use
    epochs -- the number of epochs to pre-train model on
    logger -- logging.Logger object to log information
    """

    for epoch in range(epochs):
        loop = tqdm(loader, leave=True)
        model.train()
        losses = []
        log(logger, f"Begining Training Epoch {epoch}")

        for batch in loop:
            optimizer.zero_grad()

            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)

            loss = outputs.loss
            loss.backward()

            optimizer.step()

            loop.set_description(f"Epoch {epoch}")
            loop.set_postfix(loss=loss.item())
            losses.append(loss.item())

            del input_ids
            del attention_mask
            del labels

        log(logger, f"Epoch {epoch} Mean Training Loss: {np.mean(losses)}")
    log(logger, "Pre-training Done!")


def pre_train_pipeline(model, loader, epochs, learning_rate, model_name):
    """Runs pipeline and logs output to logs/model_name folder in project root.
    Also saves model to checkpoints/model_name.

    Keyword Arguments:
    model -- model to pretrain
    loader -- data loader
    epochs -- the number of epochs to pre-train model on
    learning_rate -- learning rate for the optimizer
    model_name -- name to save model by
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    TASK_NAME = "pre_training"

    logger = get_my_logger(model_name, TASK_NAME)
    log(logger, "Background logger started")

    log(logger, "Loading optimizer")
    optimizer = torch.optim.Adam(params=model.parameters(), lr=learning_rate)

    log(logger, f"Using {device} to pre-train for {epochs} epochs!")
    model.to(device)
    pre_train(model, loader, optimizer, device, epochs, logger)

    save_dir = Path("checkpoints") / model_name
    log(logger, f"Saving pre-trained model {model_name} to {save_dir}")
    model.save_pretrained(save_dir)
    log(logger, f"Saved pre-trained model {model_name}!")
