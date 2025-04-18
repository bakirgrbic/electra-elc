"""Tools to finetune pre-trained models on a discriminative text task. This is
a personal evaluation method.
"""

import torch
from sklearn.metrics import accuracy_score
from transformers import ElectraTokenizerFast

def load_data() -> tuple:
    """Loads and splits evaluation data into train and test."""
    X = [line.strip() for line in open('X.txt').readlines()]
    y = [int(line.strip()) for line in open('YL1.txt').readlines()]
    train_X = X[:46000]
    train_y = np.array(y[:46000])
    test_X = X[46000:]
    test_y = np.array(y[46000:])
    return train_X, train_y, test_X, test_y


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
        outputs = model(ids, mask, token_type_ids)
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
            outputs = model(ids, mask, token_type_ids)
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
        loss = train(model, training_loader, optimizer, device)
        log(logger, f'Epoch: {epoch}, Loss:  {loss.item()}')
        guess, targs = validation(model, testing_loader, device)
        guesses = torch.argmax(guess, dim=1)
        targets = targs
        log(logger, f"Epoch {epoch}'s arracy on test set {accuracy_score(guesses, targets)}")
    log(logger, "Fine-tuning Done!")


def personal_evaluation():
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
    MAX_LEN = 128
    BATCH_SIZE = 64
    EPOCHS = 3
    NUM_OUT = len(labels)  # multilabel task
    LEARNING_RATE = 2e-05
    MODEL_NAME = "ELECTRA_pt" # TODO: make this and other hyperparams cli inputs
    TASK_NAME = "personal_evaluation"

    logger = get_my_logger(MODEL_NAME, TASK_NAME)
    log(logger ,"Background logger started")

    log(logger, "Loading data for personal evaluation fine-tuning")
    train_X, train_y, test_X, test_y = load_data()
    training_data = MultiLabelDataset(train_X, torch.from_numpy(train_y), tokenizer, MAX_LEN)
    testing_data = MultiLabelDataset(test_X, torch.from_numpy(test_y), tokenizer, MAX_LEN)
    train_params = {'batch_size': BATCH_SIZE,
                'shuffle': True,
                'num_workers': 0
                }
    test_params = {'batch_size': BATCH_SIZE,
                    'shuffle': True,
                    'num_workers': 0
                }
    training_loader = torch.utils.data.DataLoader(training_data, **train_params)
    testing_loader = torch.utils.data.DataLoader(testing_data, **test_params)

    log(logger, "Loading tokenizer, model and optimizer")
    tokenizer = ElectraTokenizerFast.from_pretrained("bsu-slim/electra-tiny")
    model = ELECTRAClass(NUM_OUT)
    model.to(device)
    optimizer = torch.optim.Adam(params=model.parameters(), lr=LEARNING_RATE)

    log(logger, f"Using {device} to fine-tune for {EPOCHS} epochs!")
    evaluate(model, training_loader, testing_loader, optimizer, device, EPOCHS):
    log(logger, "Fine-tuning Done!")

if __name__ == "__main__":
    main()
