import torch

from config import settings


def calc_batch_accuracy(y_pred: torch.Tensor, y_true: torch.Tensor) -> float:
    correct = (y_pred == y_true).sum().item()
    return round(100 * correct / len(y_true), 2)


def set_seed() -> None:
    torch.manual_seed(settings.random_seed)
    torch.cuda.manual_seed(settings.random_seed)
