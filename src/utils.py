import csv
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn as nn

from src.config import EXPERIMENTS_CSV


def accuracy(logits: torch.Tensor, labels: torch.Tensor) -> float:
    return (logits.argmax(1) == labels).float().mean().item()


def save_checkpoint(model: nn.Module, path: Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), path)


def load_checkpoint(model: nn.Module, path: Path, device: torch.device) -> nn.Module:
    model.load_state_dict(torch.load(path, map_location=device, weights_only=True))
    return model


def plot_history(history: dict, save_path: str = "training_curves.png") -> None:
    epochs = range(1, len(history["train_loss"]) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(epochs, history["train_loss"], label="train")
    axes[0].plot(epochs, history["val_loss"], label="val")
    axes[0].set_title("Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()

    axes[1].plot(epochs, history["train_acc"], label="train")
    axes[1].plot(epochs, history["val_acc"], label="val")
    axes[1].set_title("Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(save_path, dpi=100)
    plt.close()
    print(f"Curves saved → {save_path}")


class EarlyStopping:
    def __init__(self, patience: int = 10, min_delta: float = 0.0):
        self.patience = patience
        self.min_delta = min_delta
        self._best = float("inf")
        self._counter = 0

    def __call__(self, val_loss: float) -> bool:
        if val_loss < self._best - self.min_delta:
            self._best = val_loss
            self._counter = 0
        else:
            self._counter += 1
        return self._counter >= self.patience


def log_experiment(
    run_id: str,
    base_champion: str,
    levier: str,
    cv_acc_mean: float,
    cv_acc_std: float,
    temps: float,
    garde: str,
) -> None:
    path = EXPERIMENTS_CSV
    write_header = not Path(path).exists()
    with open(path, "a", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "run_id", "base_champion", "levier_testé",
                "cv_acc_mean", "cv_acc_std", "temps", "gardé",
            ],
        )
        if write_header:
            writer.writeheader()
        writer.writerow({
            "run_id": run_id,
            "base_champion": base_champion,
            "levier_testé": levier,
            "cv_acc_mean": f"{cv_acc_mean:.4f}",
            "cv_acc_std": f"{cv_acc_std:.4f}",
            "temps": f"{temps:.1f}",
            "gardé": garde,
        })
    print(f"Logged → {path}")
