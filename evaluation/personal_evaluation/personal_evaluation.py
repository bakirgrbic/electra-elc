"""Tools to finetune pre-trained models on a discriminative text task. This is
a personal evaluation method.
"""

import torch

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


def evaluate(): # TODO: need data, optimizer, model, and likely device
    for epoch in range(EPOCHS):
        log(f"Begining Fine Tuning on Epoch {EPOCHS}") # should have been printing epoch and not EPOCHS
        loss = train(model, training_loader, optimizer)
        log(f'Epoch: {epoch}, Loss:  {loss.item()}')
        guess, targs = validation(model, testing_loader)
        guesses = torch.argmax(guess, dim=1)
        targets = targs
        log(f"Epoch {epoch}'s arracy on test set {accuracy_score(guesses, targets)}")
    log("Fine-tuning Done!")


def main(): # TODO
    # Define hyperparams and constants
        # Silent output so its not in terminal?
        # What about logging? should this func start a new one or simply be passed one?
    # Get data
    # Set up datasets and loaders
    # set up model and tokenizer
    # call evaluate func

if __name__ == "__main__":
    evaluate()
