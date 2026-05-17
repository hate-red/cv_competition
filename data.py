from torch.utils.data import DataLoader

from torchvision.datasets import MNIST
from torchvision import transforms

from config import settings


transform = transforms.Compose([
    transforms.ToTensor()
])

train_ds = MNIST(root=settings.train_ds_path, train=True, download=False, transform=transform)
test_ds = MNIST(root=settings.test_ds_path, train=False, download=False, transform=transform)

train_dataloader = DataLoader(
    dataset=train_ds,
    batch_size=settings.batch_size,
    shuffle=True,
    # num_workers=settings.num_workers,
)
test_dataloader = DataLoader(
    dataset=test_ds,
    batch_size=settings.batch_size,
    shuffle=True,
    # num_workers=settings.num_workers
)