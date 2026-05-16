import torch
from torch import nn


class MLP(nn.Module):
    model_name: str = 'MLP'

    def __init__(
            self, 
            n_channels: int,
            img_size: int, 
            n_hidden_layers: int, 
            n_hidden_units: int, 
            n_classes: int
        ) -> None:
        super().__init__()

        self.input_layer = nn.Sequential(
            nn.Flatten(),
            nn.Linear(n_channels * img_size ** 2, n_hidden_units)
        )

        self.hidden_layers = nn.Sequential(*[
            nn.Sequential(
                nn.Linear(n_hidden_units, n_hidden_units),
                nn.ReLU()
            )
            for _ in range(n_hidden_layers)
        ])

        self.classirier = nn.Linear(n_hidden_units, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        flattened = self.input_layer(x)
        hidden_layers_out = self.hidden_layers(flattened)
        prediction = self.classirier(hidden_layers_out)

        return prediction.squeeze()
    