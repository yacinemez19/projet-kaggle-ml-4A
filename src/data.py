import os
from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from sklearn.model_selection import StratifiedKFold, train_test_split
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from src.config import (
    FOLDS_CSV,
    TEST_DIR,
    TRAIN_CSV,
    TRAIN_DIR,
    TRAIN_MEAN,
    TRAIN_STD,
    Config,
)


class TildaDataset(Dataset):
    def __init__(self, img_dir: Path, ids: list, labels: list = None, transform=None):
        self.img_dir = Path(img_dir)
        self.ids = ids
        self.labels = labels
        self.transform = transform

    def __len__(self) -> int:
        return len(self.ids)

    def __getitem__(self, idx: int):
        img = Image.open(self.img_dir / f"{self.ids[idx]}.tif")  # mode L (grayscale)
        if self.transform:
            img = self.transform(img)
        label = self.labels[idx] if self.labels is not None else -1
        return img, label


def get_train_transform(cfg: Config):
    return transforms.Compose([
        transforms.Resize((cfg.img_h, cfg.img_w)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=TRAIN_MEAN, std=TRAIN_STD),
    ])


def get_val_transform(cfg: Config):
    return transforms.Compose([
        transforms.Resize((cfg.img_h, cfg.img_w)),
        transforms.ToTensor(),
        transforms.Normalize(mean=TRAIN_MEAN, std=TRAIN_STD),
    ])


def _loader_kwargs(cfg: Config) -> dict:
    """DataLoader perf settings, device-aware.

    On CUDA: enable pin_memory and worker processes (the GPU starves with
    num_workers=0). On MPS/CPU: keep num_workers=0 (MPS requirement) and no pinning.
    A user-set cfg.num_workers > 0 is always respected.
    """
    use_cuda = torch.cuda.is_available()
    num_workers = cfg.num_workers
    if use_cuda and num_workers == 0:
        num_workers = min(8, os.cpu_count() or 0)
    return {
        "num_workers": num_workers,
        "pin_memory": use_cuda,
        "persistent_workers": num_workers > 0,
    }


def make_folds(cfg: Config) -> pd.DataFrame:
    """Generate Stratified 5-fold split and save to folds.csv (idempotent)."""
    if FOLDS_CSV.exists():
        return pd.read_csv(FOLDS_CSV)

    df = pd.read_csv(TRAIN_CSV, sep=";")
    df["fold"] = -1
    skf = StratifiedKFold(n_splits=cfg.n_folds, shuffle=True, random_state=cfg.seed)
    for fold, (_, val_idx) in enumerate(skf.split(df["id"], df["label"])):
        df.loc[val_idx, "fold"] = fold
    df.to_csv(FOLDS_CSV, index=False)
    print(f"folds.csv written ({cfg.n_folds} folds, seed={cfg.seed})")
    return df


def build_dataloaders(fold: int, cfg: Config, folds_df: pd.DataFrame = None):
    """Return (train_loader, val_loader) for a given fold."""
    if folds_df is None:
        folds_df = pd.read_csv(FOLDS_CSV)

    train_df = folds_df[folds_df["fold"] != fold]
    val_df = folds_df[folds_df["fold"] == fold]

    train_ds = TildaDataset(
        TRAIN_DIR,
        train_df["id"].tolist(),
        train_df["label"].tolist(),
        transform=get_train_transform(cfg),
    )
    val_ds = TildaDataset(
        TRAIN_DIR,
        val_df["id"].tolist(),
        val_df["label"].tolist(),
        transform=get_val_transform(cfg),
    )

    kwargs = _loader_kwargs(cfg)
    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, **kwargs)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False, **kwargs)
    return train_loader, val_loader


def stratified_split(cfg: Config, val_size: float = 0.2):
    """Quick train/val split (single fold) for debug iterations."""
    df = pd.read_csv(TRAIN_CSV, sep=";")
    train_df, val_df = train_test_split(
        df, test_size=val_size, stratify=df["label"], random_state=cfg.seed
    )

    train_ds = TildaDataset(
        TRAIN_DIR,
        train_df["id"].tolist(),
        train_df["label"].tolist(),
        transform=get_train_transform(cfg),
    )
    val_ds = TildaDataset(
        TRAIN_DIR,
        val_df["id"].tolist(),
        val_df["label"].tolist(),
        transform=get_val_transform(cfg),
    )

    kwargs = _loader_kwargs(cfg)
    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, **kwargs)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False, **kwargs)
    return train_loader, val_loader
