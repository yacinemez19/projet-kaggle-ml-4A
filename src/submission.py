from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader

from src.config import TEST_DIR, Config
from src.data import TildaDataset, get_val_transform


def predict_test(
    model: torch.nn.Module,
    cfg: Config,
    device: torch.device,
) -> tuple[list, list]:
    """Run inference on all test images; returns (ids, 0-indexed predicted labels)."""
    test_ids = sorted([int(p.stem) for p in Path(TEST_DIR).glob("*.tif")])
    ds = TildaDataset(
        TEST_DIR,
        test_ids,
        labels=None,
        transform=get_val_transform(cfg),
    )
    loader = DataLoader(
        ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=False,
    )
    model.eval()
    all_preds = []
    with torch.no_grad():
        for imgs, _ in loader:
            imgs = imgs.to(device)
            preds = model(imgs).argmax(1).cpu().tolist()
            all_preds.extend(preds)
    return test_ids, all_preds


def write_submission(ids: list, preds: list, path: str = "submission.csv") -> None:
    """Write Kaggle CSV. CSV label k → Kaggle category k+1 (per competition spec)."""
    df = pd.DataFrame({"id": ids, "category": [p + 1 for p in preds]})
    df.to_csv(path, index=False)
    print(f"Submission saved → {path}  ({len(df)} rows)")
    print(df["category"].value_counts().sort_index().to_string())
