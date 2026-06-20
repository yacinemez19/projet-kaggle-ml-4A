from dataclasses import dataclass
from pathlib import Path
import random
import numpy as np
import torch

ROOT = Path(__file__).parent.parent
DATA_DIR = ROOT / "data"
TRAIN_DIR = DATA_DIR / "train"
TEST_DIR = DATA_DIR / "test"
TRAIN_CSV = DATA_DIR / "train.csv"
FOLDS_CSV = ROOT / "folds.csv"
CHECKPOINTS_DIR = ROOT / "checkpoints"
EXPERIMENTS_CSV = ROOT / "experiments.csv"

# Computed once on the training set (pixel values in [0, 1])
TRAIN_MEAN = (0.497510,)
TRAIN_STD = (0.216788,)


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@dataclass
class Config:
    seed: int = 42
    # Image size: preserves native 3:2 ratio (768×512 → 384×256)
    img_h: int = 256
    img_w: int = 384
    batch_size: int = 32
    # num_workers=0 required on MPS; safe default for all platforms
    num_workers: int = 0
    n_folds: int = 5
    lr: float = 1e-3
    epochs: int = 50
    weight_decay: float = 1e-4
    num_classes: int = 8
    in_channels: int = 1
