from dataclasses import dataclass
from pathlib import Path
import os


@dataclass
class Settings:
    project_path: Path
    
    dataset_path: Path
    train_ds_path: Path
    test_ds_path: Path

    save_results_path: Path
    save_weights_path: Path

    random_seed: int = 42

    batch_size: int = 1024
    num_workers: int = os.cpu_count() # type: ignore
    epochs: int = 100
    lr: float = 1e-4

    n_channels: int = 1
    img_size: int = 28
    n_classes: int = 10
    

project_path = Path(__file__).parent

ds_path = project_path / 'dataset'
train_ds_path = ds_path / 'train'
test_ds_path = ds_path / 'test'

save_results_path = project_path / 'results'
save_weights_path = project_path / 'pretrained'

ds_path.mkdir(exist_ok=True)
train_ds_path.mkdir(exist_ok=True)
test_ds_path.mkdir(exist_ok=True)

save_results_path.mkdir(exist_ok=True)
save_weights_path.mkdir(exist_ok=True)

settings = Settings(project_path, ds_path, train_ds_path, test_ds_path, save_results_path, save_weights_path)

mlp_params: dict = {
    'n_channels': settings.n_channels,
    'img_size': settings.img_size,
    'n_hidden_layers': 20,
    'n_hidden_units': 128,
    'n_classes': settings.n_classes
}

cnn_params: dict = {
    'n_channels': settings.n_channels,
    'n_hidden_units': 16,
    'n_classes': settings.n_classes
}

vit_params: dict = {
    'img_size': settings.img_size,
    'n_channels': settings.n_channels,
    'num_layers': 6,
    'patch_size': 4,
    'embedding_dim': 768,
    'mlp_dim': 64,
    'num_heads': 4,
    'attn_dropout': 0.2,
    'num_classes': settings.n_classes
}
