import matplotlib
matplotlib.use("Agg")  # works headless; swap to "TkAgg" / "MacOSX" for interactive display
import matplotlib.pyplot as plt
import pandas as pd
import torch
from torchvision.utils import make_grid

from src.config import CHECKPOINTS_DIR, Config, TRAIN_CSV, TRAIN_MEAN, TRAIN_STD, get_device, set_seed
from src.data import build_dataloaders, make_folds, stratified_split
from src.models import build_lenet, build_resnet_scratch, build_vgg_like
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
# Generate Kaggle submission — LeNet baseline
load_checkpoint(model, CHECKPOINTS_DIR / "lenet_fold0.pt", device)
ids, preds = predict_test(model, cfg, device)
write_submission(ids, preds, path="submission_lenet.csv")

# ── Session 3 — ResNet-18 from scratch ────────────────────────────────────────

# %% ----------------------------------
# Build ResNet-18 from scratch (1 channel, 8 classes, Kaiming init)
resnet = build_resnet_scratch(depth=18, in_channels=cfg.in_channels, num_classes=cfg.num_classes).to(device)
print(resnet)
total_params = sum(p.numel() for p in resnet.parameters())
print(f"ResNet-18 parameters: {total_params:,}")

# %% ----------------------------------
# Train ResNet-18 on fold 0 — label smoothing 0.1, weight_decay from config
train_loader, val_loader = build_dataloaders(fold=0, cfg=cfg, folds_df=folds_df)
history_resnet = fit(
    resnet, train_loader, val_loader, cfg, device,
    run_name="resnet18_fold0",
    patience=15,
    label_smoothing=0.1,
)

# %% ----------------------------------
# Training curves — ResNet-18
plot_history(history_resnet, save_path="resnet18_curves.png")

# %% ----------------------------------
# Compare baseline vs ResNet-18
lenet_acc  = history["best_val_acc"]
resnet_acc = history_resnet["best_val_acc"]
print(f"LeNet  best val_acc : {lenet_acc:.4f}")
print(f"ResNet-18 best val_acc : {resnet_acc:.4f}")
print(f"Delta : {resnet_acc - lenet_acc:+.4f}")

# %% ----------------------------------
# Log ResNet-18 run
log_experiment(
    run_id="resnet18_fold0",
    base_champion="lenet_fold0",
    levier="ResNet-18 from scratch (label_smoothing=0.1)",
    cv_acc_mean=history_resnet["best_val_acc"],
    cv_acc_std=0.0,
    temps=history_resnet["elapsed"],
    garde="oui" if history_resnet["best_val_acc"] > history["best_val_acc"] else "non",
)

# %% ----------------------------------
# Generate Kaggle submission — ResNet-18
load_checkpoint(resnet, CHECKPOINTS_DIR / "resnet18_fold0.pt", device)
ids, preds = predict_test(resnet, cfg, device)
write_submission(ids, preds, path="submission.csv")

# ── Session 4 — Levier : cosine LR scheduler ──────────────────────────────────

# %% ----------------------------------
# Champion courant : resnet18_fold0, val_acc=0.6406
# Levier testé : CosineAnnealingLR + 100 epochs (au lieu de lr fixe + 50 epochs)
# Méthode : fold 0 d'abord, puis décision garder/jeter

cfg_cosine = Config(epochs=100, lr=1e-3, weight_decay=1e-4)
set_seed(cfg_cosine.seed)
train_loader, val_loader = build_dataloaders(fold=0, cfg=cfg_cosine, folds_df=folds_df)

resnet_cosine = build_resnet_scratch(
    depth=18, in_channels=cfg_cosine.in_channels, num_classes=cfg_cosine.num_classes
).to(device)

history_cosine = fit(
    resnet_cosine, train_loader, val_loader, cfg_cosine, device,
    run_name="resnet18_cosine_fold0",
    patience=20,
    label_smoothing=0.1,
    use_cosine=True,
)

# %% ----------------------------------
plot_history(history_cosine, save_path="resnet18_cosine_curves.png")

# %% ----------------------------------
print(f"Champion    val_acc : 0.6406")
print(f"Cosine      val_acc : {history_cosine['best_val_acc']:.4f}")
print(f"Delta : {history_cosine['best_val_acc'] - 0.6406:+.4f}")

log_experiment(
    run_id="resnet18_cosine_fold0",
    base_champion="resnet18_fold0",
    levier="CosineAnnealingLR + 100 epochs",
    cv_acc_mean=history_cosine["best_val_acc"],
    cv_acc_std=0.0,
    temps=history_cosine["elapsed"],
    garde="oui" if history_cosine["best_val_acc"] > 0.6406 else "non",
)

# ── Session 4 — Levier : résolution 512×384 ───────────────────────────────────

# %% ----------------------------------
# Champion courant : resnet18_cosine_fold0, val_acc=0.8330
# Levier testé : résolution 384×256 → 512×384 (img_h=384, img_w=512), levier seul
# Hypothèse : + de signal pour les défauts fins (peluches, fils, petits trous).
# batch_size=32 conservé (= champion cosine) → un seul levier propre (résolution).
# Fallback batch_size=16 si OOM (4 Go VRAM, ~2.7-3.0 Go attendus à batch 32).
CHAMPION_VAL_ACC = 0.8330

cfg_hires = Config(img_h=384, img_w=512, batch_size=32, epochs=100, lr=1e-3, weight_decay=1e-4)
set_seed(cfg_hires.seed)
train_loader, val_loader = build_dataloaders(fold=0, cfg=cfg_hires, folds_df=folds_df)

resnet_hires = build_resnet_scratch(
    depth=18, in_channels=cfg_hires.in_channels, num_classes=cfg_hires.num_classes
).to(device)

history_hires = fit(
    resnet_hires, train_loader, val_loader, cfg_hires, device,
    run_name="resnet18_hires_fold0",
    patience=20,
    label_smoothing=0.1,
    use_cosine=True,
)

# %% ----------------------------------
plot_history(history_hires, save_path="resnet18_hires_curves.png")

# %% ----------------------------------
print(f"Champion cosine val_acc : {CHAMPION_VAL_ACC:.4f}")
print(f"Hi-res 512×384  val_acc : {history_hires['best_val_acc']:.4f}")
print(f"Delta : {history_hires['best_val_acc'] - CHAMPION_VAL_ACC:+.4f}")

log_experiment(
    run_id="resnet18_hires_fold0",
    base_champion="resnet18_cosine_fold0",
    levier="résolution 512×384 (img_h=384, img_w=512)",
    cv_acc_mean=history_hires["best_val_acc"],
    cv_acc_std=0.0,
    temps=history_hires["elapsed"],
    garde="oui" if history_hires["best_val_acc"] > CHAMPION_VAL_ACC else "non",
)
