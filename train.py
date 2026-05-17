import torch

from tqdm import tqdm

from architectures.MLP import MLP
from architectures.CNN import CNN
from architectures.ViT import ViT
from architectures.ResNet import ResNet

from engine import train_model
from plots import plot_results
from utils import set_seed

from data import train_dataloader, test_dataloader
from config import settings, mlp_params, cnn_params, vit_params


set_seed()

models = [MLP(**mlp_params), CNN(**cnn_params), ResNet]

criterion = torch.nn.CrossEntropyLoss()
device = 'cuda' if torch.cuda.is_available() else 'cpu'


for model in tqdm(models):
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=settings.lr)

    print(f'Started training {model.model_name} model...')

    results = train_model(
        model=model, 
        criterion=criterion, 
        optimizer=optimizer, 
        train_dataloader=train_dataloader, 
        test_dataloader=test_dataloader, 
        device=device,
    )

    print(f'Finished training {model.model_name} model')

    plot_results(results, model.model_name, show=False, save_to=settings.save_results_path / f'{model.model_name}.png')
