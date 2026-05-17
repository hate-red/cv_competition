import torch
from torchmetrics import ConfusionMatrix

import random
from collections import namedtuple

import seaborn as sns
import matplotlib.pyplot as plt
from mlxtend.plotting import plot_confusion_matrix

from config import settings
from architectures.ResNet import ResNet
from data import test_ds


state_dict = torch.load(settings.save_weights_path / 'ResNet.pt')
model = ResNet
model.load_state_dict(state_dict)

Sample = namedtuple('Sample', ['label', 'prediction', 'prob', 'image'])
misclassified = []
preds = []

for X, y in test_ds:
    model.eval()

    y_logit = model(X.unsqueeze(1))

    y_pred = y_logit.softmax(1).argmax(1)

    if y_pred != y:
        prob = y_logit.softmax(1).squeeze().max().item()

        sample = Sample(
            label=y, 
            prediction=y_pred.item(), 
            prob=round(prob, 2), 
            image=X.squeeze().numpy()
        )
        misclassified.append(sample)
    
    preds.append(y_pred)

sns.set_theme('paper')

fig, axes = plt.subplots(3, 4, figsize=(12, 10), num='ResNet Misclassified Samples', dpi=100)

fig.suptitle('Misclassified samples')
axes = axes.flatten()

samples = random.sample(misclassified, 12)

for ax, (label, prediction, prob, image) in zip(axes, samples): # type: ignore
    ax.imshow(image)
    ax.set_title(label)
    ax.set_xlabel(f'Pred: {prediction}, p={prob}')

plt.savefig(settings.save_results_path / 'ResNet Misclassified Samples.png')

confusion_matrix_factory = ConfusionMatrix('multiclass', num_classes=settings.n_classes)
cm = confusion_matrix_factory(preds=torch.cat(preds), target=torch.Tensor(test_ds.targets))

fig, ax = plot_confusion_matrix(cm.numpy(), class_names=test_ds.classes, figsize=(10, 7))
fig.savefig(settings.save_results_path / 'ResNet Confusion Matrix.png') # type: ignore
