import glob
import os
from pathlib import Path
from typing import List

import cv2
import hydra
import matplotlib
import numpy as np
from omegaconf import DictConfig, OmegaConf
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
)
from rich.table import Table
import torch
from torch.utils.data import DataLoader

from dataset.hypersim import Hypersim
from dataset.kitti import KITTI
from depth_anything_v2.dpt import DepthAnythingV2
from util.metric import eval_depth
from util.utils import RichConsoleManager

console = RichConsoleManager.get_console()


def get_model(cfg: DictConfig, device: torch.device):
    """Load and configure the model."""
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
        raise ValueError(f"Encoder {cfg.model.encoder} not supported")

    model_cfg = model_config[cfg.model.encoder]
    model = DepthAnythingV2(**model_cfg, max_depth=cfg.eval.eval.max_depth)

    console.print(f"[cyan]Loading model from: {cfg.eval.input.path}[/cyan]")
    # Load with weights_only=False for trusted checkpoints
    checkpoint = torch.load(cfg.eval.input.path, map_location="cpu", weights_only=False)

    # Extract model state dict from checkpoint
    if "model" in checkpoint:
        state_dict = checkpoint["model"]
    else:
        # Fallback: assume it's already a state dict
        state_dict = checkpoint

    model.load_state_dict(state_dict, strict=True)
    model = model.to(device).eval()
    return model


def get_input_files(input_path: str) -> List[str]:
    """Get list of input files from path (file, directory, or txt list)."""
    if os.path.isfile(input_path):
        if input_path.endswith(".txt"):
            with open(input_path, "r") as f:
                filenames = f.read().splitlines()
        else:
            filenames = [input_path]
    else:
        # Get all image files recursively
        image_extensions = ["*.jpg", "*.jpeg", "*.png", "*.bmp", "*.tiff"]
        filenames = []
        for ext in image_extensions:
            filenames.extend(
                glob.glob(os.path.join(input_path, "**", ext), recursive=True)
            )
            filenames.extend(
                glob.glob(os.path.join(input_path, "**", ext.upper()), recursive=True)
            )

    return sorted(filenames)


def apply_colormap(depth: np.ndarray, grayscale: bool = False) -> np.ndarray:
    """Apply colormap to depth map."""
    if grayscale:
        depth_colored = np.repeat(depth[..., np.newaxis], 3, axis=-1)
    else:
        cmap = matplotlib.colormaps.get_cmap("Spectral")
        depth_colored = (cmap(depth)[:, :, :3] * 255)[:, :, ::-1].astype(np.uint8)

    return depth_colored


def save_visualization(
    raw_image: np.ndarray,
    depth_colored: np.ndarray,
    output_path: str,
    pred_only: bool = False,
):
    """Save visualization with optional side-by-side comparison."""
    if pred_only:
        cv2.imwrite(output_path, depth_colored)
    else:
        # Create split region
        split_region = np.ones((raw_image.shape[0], 50, 3), dtype=np.uint8) * 255
        combined_result = cv2.hconcat([raw_image, split_region, depth_colored])
        cv2.imwrite(output_path, combined_result)


def infer_single_image(
    model, image: np.ndarray, input_size: int, device: torch.device
) -> np.ndarray:
    """Run inference on a single image."""
    with torch.no_grad():
        if hasattr(model, "infer_image"):
            # Use model's built-in inference method if available
            depth = model.infer_image(image, input_size)
        else:
            # Fallback to manual inference
            image_tensor = torch.from_numpy(image).permute(2, 0, 1).float().to(device)
            image_tensor = image_tensor.unsqueeze(0)

            # Resize if needed
            if (
                image_tensor.shape[2] != input_size
                or image_tensor.shape[3] != input_size
            ):
                from torchvision.transforms import Resize

                resize = Resize((input_size, input_size))
                image_tensor = resize(image_tensor)

            depth_pred = model(image_tensor)
            depth_pred = torch.nn.functional.interpolate(
                depth_pred.unsqueeze(1),
                size=image.shape[:2],
                mode="bilinear",
                align_corners=True,
            )
            depth = depth_pred.squeeze().cpu().numpy()

    return depth


@torch.no_grad()
def evaluate_on_dataset(model, dataloader, cfg: DictConfig, device: torch.device):
    """Evaluate model on a dataset with ground truth."""
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

    progress = Progress(
        SpinnerColumn(),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TimeElapsedColumn(),
        console=console,
    )

    with progress:
        task = progress.add_task("Evaluating", total=len(dataloader))

        for i, sample in enumerate(dataloader):
            img = sample["image"].to(device, non_blocking=True).float()
            depth_gt = sample["depth"].to(device, non_blocking=True)[0]
            valid_mask = sample["valid_mask"].to(device, non_blocking=True)[0]

            pred = model(img)
            pred = torch.nn.functional.interpolate(
                pred[:, None], depth_gt.shape[-2:], mode="bilinear", align_corners=True
            )[0, 0]

            valid_condition = (
                (valid_mask == 1)
                & (depth_gt >= cfg.eval.eval.min_depth)
                & (depth_gt <= cfg.eval.eval.max_depth)
            )

            if valid_condition.sum() < 10:
                progress.update(task, advance=1)
                continue

            cur_results = eval_depth(pred[valid_condition], depth_gt[valid_condition])
            for k in results.keys():
                results[k] += float(cur_results[k])
            nsamples += 1

            progress.update(task, advance=1)

    if nsamples == 0:
        console.print("[red]No valid samples found for evaluation![/red]")
        return None, 0

    # Average results
    for k in results.keys():
        results[k] /= nsamples

    return results, nsamples


def get_dataset_for_eval(cfg: DictConfig):
    """Get dataset for evaluation based on config."""
    # This assumes your dataset configs match the training ones
    # You might need to adjust based on your actual dataset structure

    if "hypersim" in cfg.eval.input.path.lower():
        dataset_name = "hypersim"
        val_split = "dataset/splits/hypersim/val.txt"
    elif "vkitti" in cfg.eval.input.path.lower():
        dataset_name = "vkitti"
        val_split = "dataset/splits/kitti/val.txt"
    else:
        # Default to hypersim
        dataset_name = "hypersim"
        val_split = "dataset/splits/hypersim/val.txt"

    size = (cfg.eval.input.input_size, cfg.eval.input.input_size)

    if dataset_name == "hypersim":
        valset = Hypersim(val_split, "val", size=size)
    elif dataset_name == "vkitti":
        valset = KITTI(val_split, "val", size=size)
    else:
        raise NotImplementedError(
            f"Dataset {dataset_name} not supported for evaluation"
        )

    return valset


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig):
    console.print("[bold blue]Starting Depth Anything V2 Evaluation[/bold blue]")
    console.print(f"Configuration:\n{OmegaConf.to_yaml(cfg)}", style="info")

    # Setup device
    if cfg.eval.eval.use_cuda and torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    console.print(f"[cyan]Using device: {device}[/cyan]")

    # Create output directory
    output_dir = Path(cfg.eval.output.dir)  # Fixed: was cfg.output.dir
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load model
    model = get_model(cfg, device)

    # Check if we should evaluate on dataset with ground truth
    if cfg.eval.eval.compute_metrics:
        console.print(
            "[bold yellow]Computing metrics on validation dataset...[/bold yellow]"
        )

        valset = get_dataset_for_eval(cfg)
        valloader = DataLoader(
            valset,
            batch_size=cfg.eval.input.batch_size,  # Fixed: was cfg.input.batch_size
            num_workers=cfg.eval.input.num_workers,  # Fixed: was cfg.input.num_workers
            pin_memory=True,
        )

        results, nsamples = evaluate_on_dataset(model, valloader, cfg, device)

        if results is not None:
            # Display results table
            table = Table(title="Evaluation Metrics")
            table.add_column("Metric", style="cyan")
            table.add_column("Value", style="magenta")

            for metric, value in results.items():
                table.add_row(metric, f"{value:.3f}")

            console.print(table, style="info")

            # Save metrics to file
            if cfg.eval.output.save_metrics:  # Fixed: was cfg.output.save_metrics
                metrics_file = output_dir / "metrics.yaml"
                OmegaConf.save(results, metrics_file)
                console.print(f"[green]Metrics saved to: {metrics_file}[/green]")

        console.print(
            "[bold green]Evaluation completed!"
            f" Processed {nsamples} samples.[/bold green]"
        )

    else:
        # Single image or directory inference
        console.print("[bold yellow]Running inference on input images...[/bold yellow]")

        input_files = get_input_files(cfg.eval.input.path)  # Fixed: was cfg.input.path
        console.print(f"Found {len(input_files)} input files")

        if not input_files:
            console.print("[red]No input files found![/red]")
            return

        progress = Progress(
            SpinnerColumn(),
            BarColumn(),
            TextColumn("[progress.description]{task.description}"),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeElapsedColumn(),
            console=console,
        )

        with progress:
            task = progress.add_task("Processing images", total=len(input_files))

            for k, filename in enumerate(input_files):
                progress.update(task, description=f"Processing: {Path(filename).name}")

                try:
                    raw_image = cv2.imread(filename)
                    if raw_image is None:
                        console.print(f"[red]Failed to load image: {filename}[/red]")
                        continue

                    # Run inference
                    depth = infer_single_image(
                        model, raw_image, cfg.eval.input.input_size, device
                    )  # Fixed: was cfg.input.input_size

                    # Save raw numpy if requested
                    if cfg.eval.output.save_numpy:  # Fixed: was cfg.output.save_numpy
                        output_path = (
                            output_dir / f"{Path(filename).stem}_raw_depth_meter.npy"
                        )
                        np.save(output_path, depth)

                    # Normalize depth for visualization
                    depth_vis = (
                        (depth - depth.min()) / (depth.max() - depth.min()) * 255.0
                    )
                    depth_vis = depth_vis.astype(np.uint8)

                    # Apply colormap
                    depth_colored = apply_colormap(
                        depth_vis, cfg.eval.output.grayscale
                    )  # Fixed: was cfg.output.grayscale

                    # Save visualization
                    output_path = output_dir / f"{Path(filename).stem}.png"
                    save_visualization(
                        raw_image,
                        depth_colored,
                        str(output_path),
                        cfg.eval.output.pred_only,
                    )  # Fixed: was cfg.output.pred_only

                except Exception as e:
                    console.print(f"[red]Error processing {filename}: {e}[/red]")
                    continue

                progress.update(task, advance=1)

        console.print(
            "[bold green]Inference completed!"
            f" Results saved to: {output_dir}[/bold green]"
        )


if __name__ == "__main__":
    main()
