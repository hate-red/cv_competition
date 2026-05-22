from torch.utils.data import DataLoader, random_split

from torchvision.datasets import CIFAR10
from torchvision import transforms

from config import settings


transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(32, 4),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
])

train_ds = CIFAR10(root=settings.train_ds_path, train=True, download=True, transform=transform)
test_ds = CIFAR10(root=settings.test_ds_path, train=False, download=True, transform=transform)

# to run experiments before scaling
train_ds, _ = random_split(train_ds, [settings.batch_size, len(train_ds) - settings.batch_size])
test_ds, _ = random_split(test_ds, [settings.batch_size, len(test_ds) - settings.batch_size])

train_dataloader = DataLoader(
    dataset=train_ds,
    batch_size=settings.batch_size,
    shuffle=True,
    pin_memory=True
)
test_dataloader = DataLoader(
    dataset=test_ds,
    batch_size=settings.batch_size,
    shuffle=True,
    pin_memory=True
)