import torch
from torch.utils.data import DataLoader

from tqdm import tqdm
from collections import defaultdict

from config import settings
from utils import calc_batch_accuracy


def train_step(model, data_loader: DataLoader, criterion, optimizer, device: str) -> tuple[float, float]:
    model.train()
    
    train_losses = []
    train_accs = []

    for X, y in data_loader:
        X, y = X.to(device), y.to(device)

        y_logits = model(X)
        loss = criterion(y_logits, y)

        y_pred = y_logits.softmax(dim=1).argmax(dim=1)

        train_losses.append(loss.item())
        train_accs.append(calc_batch_accuracy(y_pred, y))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    mean_train_loss = sum(train_losses) / len(train_losses)
    mean_train_acc = sum(train_accs) / len(train_accs)

    return mean_train_loss, mean_train_acc


def test_step(model, data_loader: DataLoader, criterion, device: str) -> tuple[float, float]:
    model.eval()

    test_losses = []
    test_accs = []

    with torch.inference_mode():
        for X, y in data_loader:
            X, y = X.to(device), y.to(device)
                
            y_logits = model(X)
            loss = criterion(y_logits, y)

            y_pred = y_logits.softmax(dim=1).argmax(dim=1)

            test_losses.append(loss.item())
            test_accs.append(calc_batch_accuracy(y_pred, y))

    mean_test_loss = sum(test_losses) / len(test_losses)
    mean_test_acc = sum(test_accs) / len(test_accs)

    return mean_test_loss, mean_test_acc


def train_model(
        model, 
        criterion, 
        optimizer,
        train_dataloader: DataLoader,
        test_dataloader: DataLoader,
        device: str,
        verbose: bool = True,
) -> dict[str, list[float]]: # type: ignore
    results = defaultdict(list[float])

    for epoch in tqdm(range(settings.epochs)):    
        train_loss, train_acc = train_step(model, train_dataloader, criterion, optimizer, device)

        results['train_losses'].append(train_loss)
        results['train_accuracies'].append(train_acc)

        test_loss, test_acc = test_step(model, test_dataloader, criterion, device)

        results['test_losses'].append(test_loss)
        results['test_accuracies'].append(test_acc)

        if verbose:
            print(
                f'Epoch: {epoch + 1 :4} '
                f'Train loss: {round(train_loss, 2):5} '
                f'Test loss: {round(test_loss, 2):5} '
                f'Train acc: {train_acc :5} '
                f'Test acc: {test_acc :5} '
            )

    torch.save(model.state_dict(), settings.save_weights_path / f'{model.model_name}.pt')

    return results