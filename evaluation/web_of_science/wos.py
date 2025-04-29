"""Tools to finetune pre-trained models on web of science document 
classification task. To be used in the root directory of the project.
"""

import numpy as np
from sklearn.metrics import accuracy_score
import torch
from tqdm.auto import tqdm
from transformers import ElectraTokenizerFast

from evaluation.web_of_science.electra import ELECTRAClass
from evaluation.web_of_science.multilabeldataset import MultiLabelDataset
from log.my_logger import get_my_logger, log


def load_data() -> tuple:
    """Loads and splits evaluation data into train and test."""
    with open('./data/web_of_science/X.txt') as f:
        data = [line.strip() for line in f.readlines()]
    with open('./data/web_of_science/YL1.txt') as f:
        labels = [int(line.strip()) for line in f.readlines()]
    train_data = data[:46000]
    train_labels = np.array(labels[:46000])
    test_data = data[46000:]
    test_labels = np.array(labels[46000:])
    return train_data, train_labels, test_data, test_labels


def loss_fn(outputs, targets):
    """Loss function used for evaluation."""
    return torch.nn.CrossEntropyLoss()(outputs, targets)


def finetune(model, training_loader, optimizer, device):
    """Finetunes a given model on training data and optimizer.

    Keyword Arguments:
    model -- torch model to validate
    training_loader -- torch data loader with train data 
    optimizer -- torch optimizer
    device -- which hardware device to use
    
    Returns tuple of model guesses and actual label
    """
    model.train()
    for data in tqdm(training_loader):
        ids = data['ids'].to(device, dtype=torch.long)
        mask = data['mask'].to(device, dtype=torch.long)
        token_type_ids = data['token_type_ids'].to(device, dtype=torch.long)
        targets = data['targets'].to(device, dtype=torch.long)
        outputs = model(ids, mask)
        optimizer.zero_grad()
        loss = loss_fn(outputs, targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return loss


def validation(model, testing_loader, device):
    """Runs a model in evaluation mode to gather guesses and target labels.

    Keyword Arguments:
    model -- torch model to validate
    testing_loader -- torch data loader with test data 
    device -- which hardware device to use
    
    Returns tuple of model guesses and actual label
    """
    model.eval()
    fin_targets = []
    fin_outputs = []
    with torch.no_grad():
        for data in tqdm(testing_loader):
            targets = data['targets']
            ids = data['ids'].to(device, dtype=torch.long)
            mask = data['mask'].to(device, dtype=torch.long)
            token_type_ids = data['token_type_ids'].to(device, dtype=torch.long)
            outputs = model(ids, mask)
            outputs = torch.sigmoid(outputs).cpu().detach()
            fin_outputs.extend(outputs)
            fin_targets.extend(targets)
    return torch.stack(fin_outputs), torch.stack(fin_targets)


def evaluate(model, training_loader, testing_loader, optimizer, device, epochs, logger):
    """Evaluates a model on a discriminative text topic classification task.

    Keyword Arguments:
    model -- torch model to validate
    training_loader -- torch data loader with train data 
    testing_loader -- torch data loader with test data 
    optimizer -- torch optimizer
    device -- which hardware device to use
    epochs -- the number of epochs to finetune model on
    logger -- logging.Logger object to log information
    """
    for epoch in range(epochs):
        log(logger, f"Begining Fine Tuning on Epoch {epoch}")
        loss = finetune(model, training_loader, optimizer, device)
        log(logger, f'Epoch: {epoch}, Loss:  {loss.item()}')
        guess, targs = validation(model, testing_loader, device)
        guesses = torch.argmax(guess, dim=1)
        targets = targs
        log(logger, f"Epoch {epoch}'s arracy on test set {accuracy_score(guesses, targets)}")
    log(logger, "Fine-tuning Done!")


def wos_evaluation(max_len, batch_size, epochs, learning_rate, model_name):
    """Runs pipeline and logs output to logs/model_name folder in project root.

    Keyword Arguments:
    max_len -- controls maximum length of tokenizer to truncate or pad input
    batch_size -- size of batches to be fed to model for finetuning
    epochs -- the number of epochs to finetune model on
    learning_rate -- learning rate for finetuning
    model_name -- pretrained torch model to evaluate
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    labels = {
        0:'Computer Science',
        1:'Electrical Engineering',
        2:'Psychology',
        3:'Mechanical Engineering',
        4:'Civil Engineering',
        5:'Medical Science',
        6:'Biochemistry'
    }
    NUM_OUT = len(labels)  # multilabel task
    TASK_NAME = "wos_evaluation"

    logger = get_my_logger(model_name, TASK_NAME)
    log(logger, "Background logger started")

    log(logger, "Loading tokenizer, model and optimizer")
    tokenizer = ElectraTokenizerFast.from_pretrained("bsu-slim/electra-tiny")
    model = ELECTRAClass(model_name, NUM_OUT)
    model.to(device)
    optimizer = torch.optim.Adam(params=model.parameters(), lr=learning_rate)


    log(logger, "Loading data for personal evaluation fine-tuning")
    train_data, train_labels, test_data, test_labels = load_data()
    training_data = MultiLabelDataset(train_data,
                                      torch.from_numpy(train_labels),
                                      tokenizer, max_len)
    testing_data = MultiLabelDataset(test_data,
                                     torch.from_numpy(test_labels),
                                     tokenizer, max_len)
    train_params = {'batch_size': batch_size,
                'shuffle': True,
                'num_workers': 0
                }
    test_params = {'batch_size': batch_size,
                    'shuffle': True,
                    'num_workers': 0
                }
    training_loader = torch.utils.data.DataLoader(training_data, **train_params)
    testing_loader = torch.utils.data.DataLoader(testing_data, **test_params)

    log(logger, f"Using {device} to fine-tune for {epochs} epochs!")
    evaluate(model, training_loader, testing_loader, optimizer, device, epochs, logger)
