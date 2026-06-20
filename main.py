import matplotlib
matplotlib.use("Agg")  # works headless; swap to "TkAgg" / "MacOSX" for interactive display
import matplotlib.pyplot as plt
import pandas as pd
import torch
from torchvision.utils import make_grid

from src.config import CHECKPOINTS_DIR, Config, TRAIN_CSV, TRAIN_MEAN, TRAIN_STD, get_device, set_seed
from src.data import build_dataloaders, make_folds, stratified_split
from src.models import build_lenet
from src.engine import fit
from src.utils import load_checkpoint, log_experiment, plot_history
from src.submission import predict_test, write_submission

# %% ----------------------------------
cfg = Config()
device = get_device()
set_seed(cfg.seed)
print(f"device : {device}")
print(f"config : {cfg}")

# %% ----------------------------------
folds_df = make_folds(cfg)
train_loader, val_loader = build_dataloaders(fold=0, cfg=cfg, folds_df=folds_df)
print(f"train batches : {len(train_loader)}  |  val batches : {len(val_loader)}")
print(f"train samples : {len(train_loader.dataset)}  |  val samples : {len(val_loader.dataset)}")

# %% ----------------------------------
# EDA — class distribution
df = pd.read_csv(TRAIN_CSV, sep=";")
counts = df["label"].value_counts().sort_index()

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].bar(counts.index, counts.values, color="steelblue")
axes[0].set_xlabel("Class")
axes[0].set_ylabel("Count")
axes[0].set_title("Class distribution — train")
axes[0].set_xticks(range(8))
for x, y in zip(counts.index, counts.values):
    axes[0].text(x, y + 3, str(y), ha="center", va="bottom", fontsize=9)

# EDA — sample batch
imgs, labels = next(iter(train_loader))
mean_t = torch.tensor(TRAIN_MEAN).view(1, 1, 1, 1)
std_t = torch.tensor(TRAIN_STD).view(1, 1, 1, 1)
imgs_disp = (imgs * std_t + mean_t).clamp(0, 1)

grid = make_grid(imgs_disp[:8], nrow=4, padding=4, pad_value=0.9)
# grid is [1, H, W] for single-channel
axes[1].imshow(grid.permute(1, 2, 0).numpy(), cmap="gray", vmin=0, vmax=1)
axes[1].set_title(f"Sample batch (labels: {labels[:8].tolist()})")
axes[1].axis("off")

plt.tight_layout()
plt.savefig("eda_batch.png", dpi=100)
print("EDA saved → eda_batch.png")
plt.close()

# %% ----------------------------------
# Sanity check — shapes, dtype, value range
imgs, labels = next(iter(train_loader))
print(f"batch shape : {list(imgs.shape)}")   # [B, 1, H, W]
print(f"dtype       : {imgs.dtype}")
print(f"min / max   : {imgs.min():.4f} / {imgs.max():.4f}")
print(f"labels      : {labels[:16].tolist()}")
assert imgs.shape[1] == 1, "expected 1 channel (grayscale)"
assert imgs.shape[2] == cfg.img_h and imgs.shape[3] == cfg.img_w, "unexpected spatial size"
print("sanity check passed.")

# %% ----------------------------------
# Session 2 — LeNet baseline
model = build_lenet(in_channels=cfg.in_channels, num_classes=cfg.num_classes).to(device)
print(model)
total_params = sum(p.numel() for p in model.parameters())
print(f"Parameters: {total_params:,}")

# %% ----------------------------------
# Train on fold 0
train_loader, val_loader = build_dataloaders(fold=0, cfg=cfg, folds_df=folds_df)
history = fit(model, train_loader, val_loader, cfg, device, run_name="lenet_fold0")

# %% ----------------------------------
# Training curves
plot_history(history, save_path="lenet_curves.png")

# %% ----------------------------------
# Log baseline run
log_experiment(
    run_id="lenet_fold0",
    base_champion="—",
    levier="baseline LeNet (fold 0)",
    cv_acc_mean=history["best_val_acc"],
    cv_acc_std=0.0,
    temps=history["elapsed"],
    garde="oui",
)

# %% ----------------------------------
# Generate Kaggle submission (best checkpoint)
load_checkpoint(model, CHECKPOINTS_DIR / "lenet_fold0.pt", device)
ids, preds = predict_test(model, cfg, device)
write_submission(ids, preds, path="submission.csv")
