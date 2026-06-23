import contextlib
import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.config import CHECKPOINTS_DIR, Config
from src.utils import EarlyStopping, save_checkpoint


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    scheduler: torch.optim.lr_scheduler._LRScheduler | None = None,
    scaler: torch.amp.GradScaler | None = None,
) -> tuple[float, float]:
    use_amp = scaler is not None and scaler.is_enabled()
    model.train()
    total_loss, total_correct, total = 0.0, 0, 0
    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        amp_ctx = torch.autocast(device_type="cuda") if use_amp else contextlib.nullcontext()
        with amp_ctx:
            logits = model(imgs)
            loss = criterion(logits, labels)
        if use_amp:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()
        total_loss += loss.item() * imgs.size(0)
        total_correct += (logits.argmax(1) == labels).sum().item()
        total += imgs.size(0)
    if scheduler is not None:
        scheduler.step()
    return total_loss / total, total_correct / total


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float]:
    model.eval()
    total_loss, total_correct, total = 0.0, 0, 0
    with torch.no_grad():
        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device)
            logits = model(imgs)
            loss = criterion(logits, labels)
            total_loss += loss.item() * imgs.size(0)
            total_correct += (logits.argmax(1) == labels).sum().item()
            total += imgs.size(0)
    return total_loss / total, total_correct / total


def fit(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    cfg: Config,
    device: torch.device,
    run_name: str = "run",
    patience: int = 10,
    label_smoothing: float = 0.0,
    use_cosine: bool = False,
) -> dict:
    criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay
    )
    scheduler = None
    if use_cosine:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=cfg.epochs, eta_min=cfg.lr * 1e-2
        )
    # Mixed precision: auto-enabled on CUDA only (no-op on MPS/CPU).
    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    early_stop = EarlyStopping(patience=patience)
    ckpt_path = CHECKPOINTS_DIR / f"{run_name}.pt"
    ckpt_state_path = CHECKPOINTS_DIR / f"{run_name}_state.pt"

    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
    best_val_acc = 0.0
    start_epoch = 1

    # Resume from a previous interrupted run if a state checkpoint exists.
    if ckpt_state_path.exists():
        state = torch.load(ckpt_state_path, map_location=device, weights_only=False)
        model.load_state_dict(state["model_state_dict"])
        optimizer.load_state_dict(state["optimizer_state_dict"])
        if scheduler is not None and state["scheduler_state_dict"] is not None:
            scheduler.load_state_dict(state["scheduler_state_dict"])
        if use_amp and state.get("scaler_state_dict") is not None:
            scaler.load_state_dict(state["scaler_state_dict"])
        history = state["history"]
        best_val_acc = state["best_val_acc"]
        start_epoch = state["epoch"] + 1
        print(f"Resuming '{run_name}' from epoch {start_epoch} (best val_acc={best_val_acc:.4f}).")

    t0 = time.time()

    for epoch in range(start_epoch, cfg.epochs + 1):
        train_loss, train_acc = train_one_epoch(
            model, train_loader, optimizer, criterion, device, scheduler, scaler
        )
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            save_checkpoint(model, ckpt_path)

        lr = optimizer.param_groups[0]["lr"]
        print(
            f"Epoch {epoch:3d}/{cfg.epochs} | "
            f"train loss={train_loss:.4f} acc={train_acc:.4f} | "
            f"val loss={val_loss:.4f} acc={val_acc:.4f} | "
            f"lr={lr:.2e}"
        )

        # Persist full state after each completed epoch (overwrite).
        state = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
            "scaler_state_dict": scaler.state_dict() if use_amp else None,
            "history": history,
            "best_val_acc": best_val_acc,
        }
        ckpt_state_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(state, ckpt_state_path)

        if early_stop(val_loss):
            print(f"Early stopping at epoch {epoch}.")
            break

    # Clean, finished run: remove the resume state file.
    if ckpt_state_path.exists():
        ckpt_state_path.unlink()

    elapsed = time.time() - t0
    print(f"\nDone in {elapsed:.1f}s | best val_acc={best_val_acc:.4f} | ckpt → {ckpt_path}")
    history["elapsed"] = elapsed
    history["best_val_acc"] = best_val_acc
    return history
