from torchvision.models import resnet18
from torch import nn

from config import settings


ResNet = resnet18()
ResNet.conv1 = nn.Conv2d(settings.n_channels, 64, 7, 2, 3, bias=False)
ResNet.fc = nn.Linear(512, settings.n_classes)

ResNet.model_name = 'ResNet' # type: ignore