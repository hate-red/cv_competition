import matplotlib.pyplot as plt
import seaborn as sns

from math import ceil
from pathlib import Path

from config import settings


def plot_results(
        results: dict[str, list[float]],
        title: str,
        show: bool = True, 
        save_to: Path | None = None
) -> None:
    sns.set_theme('paper')

    fig, axes = plt.subplots(ceil(len(results) / 2), 2, figsize=(12, 7), dpi=120, num=title)
    
    fig.suptitle(title)
    axes = axes.flatten()

    for i, (label, values) in enumerate(results.items()):
        axes[i].plot(range(settings.epochs), values, label=label)
        axes[i].legend()

    if show:
        plt.show()

    if save_to is not None:
        fig.savefig(save_to)
