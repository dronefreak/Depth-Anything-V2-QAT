from pathlib import Path
from typing import Dict

import hydra
from omegaconf import DictConfig, OmegaConf
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from rich.table import Table
import torch
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader

from dataset.hypersim import Hypersim
from dataset.kitti import KITTI
from dataset.vkitti2 import VKITTI2
from depth_anything_v2.dpt import DepthAnythingV2
from quantization.qat_helper import (
    _move_model_to_device_and_verify,
    _warmup_model_for_tracing,
    prepare_model_for_qat_selective,
)
from util.loss import SiLogLoss
from util.metric import eval_depth
from util.utils import RichConsoleManager, set_random_seed

console = RichConsoleManager.get_console()


def get_dataset(cfg: DictConfig, split: str):
    """Factory function to create datasets."""
    size = (cfg.dataset.img_size, cfg.dataset.img_size)

    if cfg.dataset.name == "hypersim":
        if split == "train":
            return Hypersim(cfg.dataset.train_split, split, size=size)
        else:
            return Hypersim(cfg.dataset.val_split, split, size=size)
    elif cfg.dataset.name == "vkitti":
        if split == "train":
            return VKITTI2(cfg.dataset.train_split, split, size=size)
        else:
            return KITTI(cfg.dataset.val_split, split, size=size)
    else:
        raise NotImplementedError(f"Dataset {cfg.dataset.name} not supported")


def create_dataloader(dataset, cfg: DictConfig, split: str):
    """Create dataloader with proper configuration."""
    batch_size = (
        cfg.dataset.train_batch_size if split == "train" else cfg.dataset.val_batch_size
    )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        pin_memory=cfg.dataset.pin_memory,
        num_workers=cfg.dataset.num_workers,
        drop_last=(split == "train"),
        shuffle=(split == "train"),
        persistent_workers=cfg.dataset.num_workers > 0,
    )


def setup_device(cfg: DictConfig) -> torch.device:
    """Setup device with proper error handling."""
    if cfg.device == "cuda" and not torch.cuda.is_available():
        console.print(
            "CUDA requested but not available, falling back to CPU", style="warning"
        )
        device = torch.device("cpu")
    elif cfg.device == "mps" and not torch.backends.mps.is_available():
        console.print(
            "MPS requested but not available, falling back to CPU", style="warning"
        )
        device = torch.device("cpu")
    else:
        device = torch.device(cfg.device)

    return device


def create_model(cfg: DictConfig, device: torch.device):
    """Create and configure the model."""
    model_config = {
        "vitt": {"encoder": "vitt", "features": 32, "out_channels": [24, 48, 96, 192]},
        "vits": {"encoder": "vits", "features": 64, "out_channels": [48, 96, 192, 384]},
        "vitb": {
            "encoder": "vitb",
            "features": 128,
            "out_channels": [96, 192, 384, 768],
        },
        "vitl": {
            "encoder": "vitl",
            "features": 256,
            "out_channels": [256, 512, 1024, 1024],
        },
        "vitg": {
            "encoder": "vitg",
            "features": 384,
            "out_channels": [1536, 1536, 1536, 1536],
        },
    }

    if cfg.model.encoder not in model_config:
        raise ValueError(
            f"Encoder {cfg.model.encoder} not supported."
            f" Choose from: {list(model_config.keys())}"
        )

    model_cfg = model_config[cfg.model.encoder]
    model = DepthAnythingV2(**model_cfg, max_depth=cfg.dataset.max_depth)

    # Load pretrained weights if specified
    if cfg.model.pretrained_from:
        state_dict = torch.load(cfg.model.pretrained_from, map_location="cpu")
        # Filter for pretrained keys only
        pretrained_dict = {k: v for k, v in state_dict.items() if "pretrained" in k}
        model.load_state_dict(pretrained_dict, strict=False)
        console.print(
            f"Loaded pretrained weights from {cfg.model.pretrained_from}!", style="info"
        )
    # Prepare for QAT if enabled
    if cfg.quantization.enabled:
        console.print("Preparing model for Quantization Aware Training", style="info")
        # keep on CPU for prepare_qat internals (observers)
        model = model.to("cpu")
        model = prepare_model_for_qat_selective(model, cfg)

    # Move model to target device and verify
    model = _move_model_to_device_and_verify(model, device)

    # Warm up model to ensure runtime tensors are allocated on the correct device
    # Use user-configured input size if available, else fallback
    input_shape = getattr(cfg.dataset, "example_input_shape", (1, 3, 224, 224))
    warmup_ok = _warmup_model_for_tracing(model, device, input_shape=input_shape)
    if not warmup_ok:
        console.print(
            "[create_model] Warmup failed! "
            "Proceeding but torch.compile may raise device errors.",
            style="warning",
        )

    # Optionally compile after warmup
    if cfg.training.torch_compile:
        model = torch.compile(model)

    return model


def create_optimizer(model, cfg: DictConfig):
    """Create optimizer with parameter groups."""
    pretrained_params = [
        param for name, param in model.named_parameters() if "pretrained" in name
    ]
    new_params = [
        param for name, param in model.named_parameters() if "pretrained" not in name
    ]

    param_groups = [
        {
            "params": pretrained_params,
            "lr": cfg.training.lr * cfg.training.lr_backbone_multiplier,
        },
        {"params": new_params, "lr": cfg.training.lr * cfg.training.lr_head_multiplier},
    ]

    return AdamW(
        param_groups,
        lr=cfg.training.lr,
        betas=cfg.training.betas,
        weight_decay=cfg.training.weight_decay,
    )


def format_metrics(metrics: Dict[str, float]) -> str:
    """Format metrics for display."""
    return (
        f"d1: {metrics.get('d1', 0):.3f} | d2: {metrics.get('d2', 0):.3f}"
        f" | d3: {metrics.get('d3', 0):.3f} | "
        f"abs_rel: {metrics.get('abs_rel', 0):.3f} | rmse: {metrics.get('rmse', 0):.3f}"
    )


def create_progress_display():
    """Create Rich progress display for training."""
    progress = Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        TextColumn("{task.fields[status]}"),
        expand=True,
    )
    return progress


# def poly_lr_step(
#     optimizer, base_lr, iters, total_iters, power=0.9, multipliers=(1.0, 1.0)
# ):
#     """Step-wise poly learning rate update for multiple param groups."""
#     lr = base_lr * (1 - iters / total_iters) ** power
#     optimizer.param_groups[0]["lr"] = lr * multipliers[0]
#     optimizer.param_groups[1]["lr"] = lr * multipliers[1]


def train_one_epoch(
    scaler,
    model,
    trainloader,
    criterion,
    optimizer,
    epoch,
    cfg,
    device,
    progress=None,
    train_task_id=None,
):
    """Train for one epoch with AMP, Poly LR per step, and Rich progress tracking."""
    model.train()
    total_loss = 0.0
    # total_iters = cfg.training.epochs * len(trainloader)

    for i, sample in enumerate(trainloader):
        optimizer.zero_grad()

        # Move data to device
        img = sample["image"].to(device, non_blocking=True)
        depth = sample["depth"].to(device, non_blocking=True)
        valid_mask = sample["valid_mask"].to(device, non_blocking=True)

        # Forward pass with AMP (disabled for quantization-aware training)
        # with torch.amp.autocast(device_type=device.type, enabled=True):
        pred = model(img)
        valid_condition = (
            (valid_mask == 1)
            & (depth >= cfg.dataset.min_depth)
            & (depth <= cfg.dataset.max_depth)
        )
        loss = criterion(pred, depth, valid_condition)

        # Backward pass and optimizer step
        # scaler.scale(loss).backward()
        # scaler.step(optimizer)
        # scaler.update()

        # loss.backward()
        # optimizer.step()

        # # Step-wise Poly LR update
        # iters = epoch * len(trainloader) + i
        # poly_lr_step(
        #     optimizer,
        #     base_lr=cfg.training.lr,
        #     iters=iters,
        #     total_iters=total_iters,
        #     power=cfg.training.scheduler.power,
        #     multipliers=(
        #         cfg.training.lr_backbone_multiplier,
        #         cfg.training.lr_head_multiplier,
        #     ),
        # )

        # Accumulate loss
        total_loss += loss.item()

        # Update progress bar if provided
        if progress and train_task_id is not None:
            current_lr = optimizer.param_groups[0]["lr"]
            progress.update(
                train_task_id,
                advance=1,
                status=f"Loss: {loss.item():.4f} | LR: {current_lr:.2e}",
            )

    return total_loss / len(trainloader)


@torch.no_grad()
def validate(model, valloader, cfg, device, progress=None, val_task_id=None):
    """Validate the model with optional progress tracking."""
    model.eval()
    results = {
        k: 0.0
        for k in [
            "d1",
            "d2",
            "d3",
            "abs_rel",
            "sq_rel",
            "rmse",
            "rmse_log",
            "log10",
            "silog",
        ]
    }
    nsamples = 0

    for i, sample in enumerate(valloader):
        if progress and val_task_id is not None:
            progress.update(val_task_id, advance=1)

        img = sample["image"].to(device, non_blocking=True).float()
        depth = sample["depth"].to(device, non_blocking=True)[0]
        valid_mask = sample["valid_mask"].to(device, non_blocking=True)[0]

        pred = model(img)
        pred = F.interpolate(
            pred[:, None], depth.shape[-2:], mode="bilinear", align_corners=True
        )[0, 0]

        valid_condition = (
            (valid_mask == 1)
            & (depth >= cfg.dataset.min_depth)
            & (depth <= cfg.dataset.max_depth)
        )

        if valid_condition.sum() < 10:
            continue

        cur_results = eval_depth(pred[valid_condition], depth[valid_condition])
        for k in results.keys():
            # Handle both tensor and float returns from eval_depth
            if hasattr(cur_results[k], "item"):
                results[k] += cur_results[k].item()
            else:
                results[k] += float(cur_results[k])
        nsamples += 1

    if nsamples == 0:
        return results, nsamples

    # Average results
    for k in results.keys():
        results[k] /= nsamples

    return results, nsamples


def display_validation_results(
    results: Dict[str, float], epoch: int, total_epochs: int
):
    """Display validation results in a formatted Rich table."""
    table = Table(title=f"Validation Results - Epoch {epoch}/{total_epochs}")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="magenta")

    # Add accuracy metrics (higher is better)
    table.add_row("δ1", f"{results['d1']:.3f}")
    table.add_row("δ2", f"{results['d2']:.3f}")
    table.add_row("δ3", f"{results['d3']:.3f}")

    table.add_section()

    # Add error metrics (lower is better)
    table.add_row("Abs Rel", f"{results['abs_rel']:.3f}")
    table.add_row("Sq Rel", f"{results['sq_rel']:.3f}")
    table.add_row("RMSE", f"{results['rmse']:.3f}")
    table.add_row("RMSE log", f"{results['rmse_log']:.3f}")
    table.add_row("log10", f"{results['log10']:.3f}")
    table.add_row("SILog", f"{results['silog']:.3f}")

    console.print(table)


def save_checkpoint(model, optimizer, epoch, previous_best, cfg, filename="latest.pth"):
    """Save model checkpoint with clean weights only."""
    checkpoint = {
        "model": model.state_dict(),  # Only model weights
        "epoch": epoch,
        "previous_best": previous_best,
        "notes": cfg.training.notes,
        # Remove the config to avoid OmegaConf serialization issues
    }
    save_path = Path(cfg.training.save_path)
    save_path.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, save_path / filename)
    # Save clean model for evaluation
    torch.save(model.state_dict(), save_path / "model_weights.pth")


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig):
    # Setup logging
    console.print(OmegaConf.to_yaml(cfg), style="warning")

    # Set random seeds
    set_random_seed(cfg.seed)

    # Setup device and optimizations
    device = setup_device(cfg)
    cudnn.enabled = True
    cudnn.benchmark = True

    # Create datasets and dataloaders
    trainset = get_dataset(cfg, "train")
    valset = get_dataset(cfg, "val")

    trainloader = create_dataloader(trainset, cfg, "train")
    valloader = create_dataloader(valset, cfg, "val")

    # Create model, loss, and optimizer
    model = create_model(cfg, device)
    criterion = SiLogLoss().to(device)
    optimizer = create_optimizer(model, cfg)

    # Initialize tracking variables
    previous_best = {
        "d1": 0,
        "d2": 0,
        "d3": 0,
        "abs_rel": float("inf"),
        "sq_rel": float("inf"),
        "rmse": float("inf"),
        "rmse_log": float("inf"),
        "log10": float("inf"),
        "silog": float("inf"),
    }

    # Training loop with Rich progress - simplified approach
    console.print(
        f"\n[bold blue]Starting training: {cfg.training.epochs} epochs[/bold blue]"
    )
    console.print(
        f"[cyan]Training samples: {len(trainset)}"
        f" | Validation samples: {len(valset)}[/cyan]"
    )
    console.print(
        f"[cyan]Batch size: {cfg.dataset.train_batch_size}"
        f" | Workers: {cfg.dataset.num_workers}[/cyan]"
    )
    scaler = (
        torch.amp.GradScaler(
            device=device.type, enabled=True, init_scale=1024, growth_interval=2000
        )
        if not cfg.quantization.enabled
        else None
    )
    for epoch in range(cfg.training.epochs):
        # Training progress bar
        console.print(
            f"\n[bold green]Epoch {epoch + 1}/{cfg.training.epochs}"
            " - Training[/bold green]"
        )
        train_progress = Progress(
            SpinnerColumn(),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeElapsedColumn(),
            TimeRemainingColumn(),
            TextColumn("{task.fields[status]}"),
            console=console,
        )

        with train_progress:
            train_task = train_progress.add_task(
                "Training", total=len(trainloader), status="Starting..."
            )

            _ = train_one_epoch(
                scaler,
                model,
                trainloader,
                criterion,
                optimizer,
                epoch,
                cfg,
                device,
                train_progress,
                train_task,
            )

        # Validation progress bar
        console.print(
            f"[bold yellow]Epoch {epoch + 1}/{cfg.training.epochs}"
            " - Validation[/bold yellow]"
        )
        val_progress = Progress(
            SpinnerColumn(),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeElapsedColumn(),
            console=console,
        )

        with val_progress:
            val_task = val_progress.add_task("Validating", total=len(valloader))
            results, nsamples = validate(
                model, valloader, cfg, device, val_progress, val_task
            )

        if nsamples > 0:
            # Display results with Rich
            display_validation_results(results, epoch + 1, cfg.training.epochs)

            # Update best metrics
            improved_metrics = []
            for k in results.keys():
                if k in ["d1", "d2", "d3"]:
                    if results[k] > previous_best[k]:
                        previous_best[k] = results[k]
                        improved_metrics.append(f"{k}: {results[k]:.3f} ↑")
                else:
                    if results[k] < previous_best[k]:
                        previous_best[k] = results[k]
                        improved_metrics.append(f"{k}: {results[k]:.3f} ↓")

            if improved_metrics:
                console.print(
                    "[bold green]Improved metrics:"
                    f" {', '.join(improved_metrics)}[/bold green]"
                )
            else:
                console.print("[yellow]No improvement in metrics this epoch[/yellow]")

            # Save checkpoint
            if cfg.training.save_latest:
                save_checkpoint(model, optimizer, epoch, previous_best, cfg)
                console.print(
                    f"[bold cyan]Checkpoint saved to"
                    f" {cfg.training.save_path}[/bold cyan]"
                )

        # Update learning rate (poly schedule)
        for iter_in_epoch in range(len(trainloader)):
            iters = epoch * len(trainloader) + iter_in_epoch
            lr = (
                cfg.training.lr
                * (1 - iters / (cfg.training.epochs * len(trainloader))) ** 0.9
            )
            optimizer.param_groups[0]["lr"] = lr * cfg.training.lr_backbone_multiplier
            optimizer.param_groups[1]["lr"] = lr * cfg.training.lr_head_multiplier

    console.print(
        "\n[bold green]Training completed!"
        f" Final results saved to {cfg.training.save_path}[/bold green]"
    )


if __name__ == "__main__":
    main()
