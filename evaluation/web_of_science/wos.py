"""Tools to finetune pre-trained models on web of science (wos) document
classification task.
"""

import numpy as np
from sklearn.metrics import accuracy_score
import torch
import transformers
from tqdm.auto import tqdm

from evaluation.web_of_science.auto import AutoClass
from evaluation.web_of_science.multilabeldataset import MultiLabelDataset
from log.my_logger import get_my_logger, log


def load_data() -> tuple[list[str], np.ndarray, list[str], np.ndarray]:
    """Loads and splits wos evaluation data into train and test sets."""
    with open("./data/wos/X.txt") as f:
        data = [line.strip() for line in f.readlines()]
    with open("./data/wos/YL1.txt") as f:
        labels = [int(line.strip()) for line in f.readlines()]
    train_data = data[:46000]
    train_labels = np.array(labels[:46000])
    test_data = data[46000:]
    test_labels = np.array(labels[46000:])

    return train_data, train_labels, test_data, test_labels


def loss_fn(outputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Loss function used for evaluation."""

    return torch.nn.CrossEntropyLoss()(outputs, targets)


def train(
    model: AutoClass,
    training_loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Adam,
    device: str,
) -> torch.Tensor:
    """Training loop for finetuning on wos task.

    Keyword Arguments:
    model -- torch model to train
    training_loader -- train data loader
    optimizer -- torch optimizer
    device -- which hardware device to use

    Returns loss for training epoch
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
    """Testing loop for finetuning on wos task.

    Keyword Arguments:
    model -- torch model to validate
    testing_loader -- test data loader
    device -- which hardware device to use

    Returns tuple of model guesses and actual label
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
            outputs = torch.nn.softmax(input=outputs, dim=1).cpu().detach()
            fin_outputs.extend(outputs)
            fin_targets.extend(targets)

    return torch.stack(fin_outputs), torch.stack(fin_targets)


def evaluate(
    model,
    training_loader,
    testing_loader,
    optimizer: torch.optim.Adam,
    device: str,
    epochs: int,
    logger: log.my_logger,
):
    """Main finetuning loop that runs training and testing loops.

    Keyword Arguments:
    model -- torch model to validate
    training_loader -- train data loader
    testing_loader -- test data loader
    optimizer -- torch optimizer
    device -- which hardware device to use
    epochs -- the number of epochs to finetune model on
    logger -- logging.Logger object to log information
    """

    for epoch in range(epochs):
        log(logger, f"Begining Fine Tuning on Epoch {epoch}")
        loss = train(model, training_loader, optimizer, device)
        log(logger, f"Epoch: {epoch}, Loss:  {loss.item()}")
        guess, targs = test(model, testing_loader, device)
        guesses = torch.argmax(guess, dim=1)
        targets = targs
        log(
            logger,
            f"Epoch {epoch}'s arracy on test set {accuracy_score(guesses, targets)}",
        )
    log(logger, "Fine-tuning Done!")


def wos_evaluation(
    model_name: str,
    tokenizer: transformers.AutoTokenizer,
    max_len: int,
    batch_size: int,
    epochs: int,
    learning_rate: float,
):
    """Sets up and completes finetuning for wos task.

    Keyword Arguments:
    model_name -- relative file path of pre-trained model or name from
                  huggingface
    tokenizer -- transformer tokenizer
    max_len -- maximum length of words tokenizer will read for a given text
    batch_size -- size of batches to be fed to model for finetuning
    epochs -- the number of epochs to finetune model on
    learning_rate -- learning rate for finetuning

    Returns nothing but logs output to logs/model_name folder in project root.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"

    labels = {
        0: "Computer Science",
        1: "Electrical Engineering",
        2: "Psychology",
        3: "Mechanical Engineering",
        4: "Civil Engineering",
        5: "Medical Science",
        6: "Biochemistry",
    }
    NUM_OUT = len(labels)
    TASK_NAME = "wos_evaluation"

    logger = get_my_logger(model_name, TASK_NAME)
    log(logger, "Background logger started")

    log(logger, "Loading model and optimizer")
    model = AutoClass(model_name, NUM_OUT)
    model.to(device)
    optimizer = torch.optim.Adam(params=model.parameters(), lr=learning_rate)

    log(logger, "Loading data for personal evaluation fine-tuning")
    train_data, train_labels, test_data, test_labels = load_data()
    training_data = MultiLabelDataset(
        train_data, torch.from_numpy(train_labels), tokenizer, max_len
    )
    testing_data = MultiLabelDataset(
        test_data, torch.from_numpy(test_labels), tokenizer, max_len
    )
    train_params = {"batch_size": batch_size, "shuffle": True, "num_workers": 0}
    test_params = {"batch_size": batch_size, "shuffle": True, "num_workers": 0}
    training_loader = torch.utils.data.DataLoader(training_data, **train_params)
    testing_loader = torch.utils.data.DataLoader(testing_data, **test_params)

    log(logger, f"Using {device} to fine-tune for {epochs} epochs!")
    evaluate(model, training_loader, testing_loader, optimizer, device, epochs, logger)
