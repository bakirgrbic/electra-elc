"""Implements web of science (wos) document classification finetuning task."""

import logging
from enum import Enum
from pathlib import Path

import numpy as np
import torch
import transformers
from sklearn.metrics import accuracy_score
from tqdm.auto import tqdm

from src.tasks.finetuning.web_of_science.auto import AutoClass
from src.tasks.finetuning.web_of_science.multilabeldataset import \
    MultiLabelDataset

logger = logging.getLogger("main." + __name__)


class DocumentTopics(Enum):
    """Represents class number in multilabel classification."""

    COMPUTER_SCIENCE = 0
    ELECTRICAL_ENGINEERING = 1
    PSYCHOLOGY = 2
    MECHANICAL_ENGINEERING = 3
    CIVIL_ENGINEERING = 4
    MEDICAL_SCIENCE = 5
    BIOCHEMISTRY = 6


def load_data() -> tuple[list[str], np.ndarray, list[str], np.ndarray]:
    """Load and split wos data into train and test sets.

    Returns
    -------
    train_data
        List of documents.
    train_labels
        List of class labels corresponding to training documents.
    test_data
        List of documents.
    test_labels
        List of labels corresponding to testing documents.
    """
    with open("./data/wos/X.txt") as f:
        data = [line.strip() for line in f.readlines()]
    with open("./data/wos/YL1.txt") as f:
        labels = [int(line.strip()) for line in f.readlines()]
    train_data = data[:46000]
    train_labels = np.array(labels[:46000])
    test_data = data[46000:]
    test_labels = np.array(labels[46000:])

    return train_data, train_labels, test_data, test_labels


def create_dataloaders(
    train_data: list[str],
    train_labels: np.ndarray,
    test_data: list[str],
    test_labels: np.ndarray,
    tokenizer: transformers.AutoTokenizer,
    max_len: int,
    batch_size: int,
) -> tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    """Create train and test dataloaders for finetuning.

    Parameters
    ----------
    train_data
        List of documents.
    train_labels
        List of class labels corresponding to training documents.
    test_data
        List of documents.
    test_labels
        List of labels corresponding to testing documents.
    tokenizer
        Transformer tokenizer.
    max_len
        Maximum length of words tokenizer will read for a given text.
    batch_size
        Batch size to use before updating model weights.

    Returns
    -------
    training_loader
        Train data loader.
    testing_loader
        Test data loader.
    """
    train_dataset = MultiLabelDataset(
        train_data, torch.from_numpy(train_labels), tokenizer, max_len
    )
    test_dataset = MultiLabelDataset(
        test_data, torch.from_numpy(test_labels), tokenizer, max_len
    )

    training_loader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
    )
    testing_loader = torch.utils.data.DataLoader(
        dataset=test_dataset, batch_size=batch_size, shuffle=True, num_workers=0
    )

    return training_loader, testing_loader


def loss_fn(outputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Loss function used for finetuning."""

    return torch.nn.CrossEntropyLoss()(outputs, targets)


def train(
    model: AutoClass,
    training_loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Adam,
    device: str,
) -> torch.Tensor:
    """Training loop for wos finetuning task.

    Parameters
    ----------
    model
        Transformer model to pretrain.
    training_loader
        Train data loader.
    optimizer
        Torch optimizer.
    device
        Which hardware device to use.

    Returns
    -------
    loss
        Most recent loss for training epoch.
    """
    model.train()

    for data in tqdm(training_loader):
        ids = data["ids"].to(device, dtype=torch.long)
        mask = data["mask"].to(device, dtype=torch.long)
        targets = data["targets"].to(device, dtype=torch.long)
        outputs = model(ids, mask)
        loss = loss_fn(outputs, targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return loss


def test(
    model: AutoClass, testing_loader: torch.utils.data.DataLoader, device: str
) -> tuple[torch.Tensor, torch.Tensor]:
    """Testing loop for wos finetuning task.

    Parameters
    ----------
    model
        Transformer model to pretrain.
    testing_loader
        Test data loader
    device
        Which hardware device to use.

    Returns
    -------
    fin_outputs
        Model guesses for a document.
    fin_targets
        Labels associated with document.
    """
    model.eval()
    fin_targets = []
    fin_outputs = []
    with torch.no_grad():
        for data in tqdm(testing_loader):
            ids = data["ids"].to(device, dtype=torch.long)
            mask = data["mask"].to(device, dtype=torch.long)
            targets = data["targets"]
            outputs = model(ids, mask)
            outputs = torch.softmax(input=outputs, dim=1).cpu().detach()
            fin_outputs.extend(outputs)
            fin_targets.extend(targets)

    return torch.stack(fin_outputs), torch.stack(fin_targets)


def finetune(
    model: AutoClass,
    training_loader: torch.utils.data.DataLoader,
    testing_loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Adam,
    device: str,
    epochs: int,
) -> None:
    """Run main finetuning loop.

    Parameters
    ----------
    model
        Transformer model to pretrain.
    training_loader
        Train data loader.
    testing_loader
        Test data loader.
    optimizer
        Torch optimizer.
    device
        Which hardware device to use.
    epochs
        Number of epochs to pre-train for.
    """

    for epoch in range(epochs):
        logger.info(f"Begining wos finetuning epoch {epoch}")
        loss = train(model, training_loader, optimizer, device)
        logger.info(f"Epoch: {epoch}, Loss: {loss.item()}")
        guess, targets = test(model, testing_loader, device)
        guesses = torch.argmax(guess, dim=1)
        logger.info(
            f"Epoch {epoch}'s arracy on test set {accuracy_score(guesses, targets)}"
        )


def wos_task(
    model_name: str,
    training_loader: torch.utils.data.DataLoader,
    testing_loader: torch.utils.data.DataLoader,
    epochs: int,
    learning_rate: float,
) -> None:
    """Run wos finetuing task.

    Keyword Arguments:
    model_name
        Name of huggingface model or relative file path of a local model.
    training_loader
        Train data loader.
    testing_loader
        Test data loader.
    epochs
        Number of epochs to pre-train for.
    learning_rate
        Learning rate for the optimizer.

    Returns
    -------
    None
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"

    NUM_OUT = len(DocumentTopics)

    model = AutoClass(model_name, NUM_OUT)
    model.to(device)

    optimizer = torch.optim.Adam(params=model.parameters(), lr=learning_rate)

    logger.info(f"Wos finetuning start with {device}")
    finetune(
        model,
        training_loader,
        testing_loader,
        optimizer,
        device,
        epochs,
    )
    logger.info("Wos finetuning done!")
