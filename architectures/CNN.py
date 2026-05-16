import torch
from torch import nn


class CNN(nn.Module):
    model_name: str = 'CNN'

    def __init__(
            self,
            n_channels: int,
            n_hidden_units: int,
            n_classes: int
    ) -> None:
        super().__init__()

        kernel_size = 3
        stride = 2

        self.conv_block = nn.Sequential(
            nn.Conv2d(n_channels, n_hidden_units, kernel_size, stride, padding=0),
            nn.ReLU(),
            nn.Conv2d(n_hidden_units, n_hidden_units, kernel_size, stride, padding=0),
            nn.MaxPool2d(kernel_size, stride)
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(4 * n_hidden_units, n_classes)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        conv_block_out = self.conv_block(x)
        prediction = self.classifier(conv_block_out)

        return prediction.squeeze()
