"""Tools to finetune pre-trained models using personal evaluation method."""

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


def train(model, training_loader, optimizer):
    # TODO understand, docstring, and rename
    """
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


def validation(model, testing_loader):
    # TODO understand and docstring
    """
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

# TODO multilabel dataset?
# TODO Electra class
# TODO main training loop?

def main():
    # TODO

if __name__ == "__main__":
    main()
