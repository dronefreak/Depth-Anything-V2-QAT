import cv2
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision.transforms import Compose

from dataset.transform import (
    ColorJitter,
    Crop,
    NormalizeImage,
    PrepareForNet,
    RandomGaussianBlur,
    RandomGrayscaleProb,
    RandomHorizontalFlip,
    Resize,
)
from util.utils import RichConsoleManager


def hypersim_distance_to_depth(npyDistance):
    intWidth, intHeight, fltFocal = 1024, 768, 886.81

    npyImageplaneX = (
        np.linspace((-0.5 * intWidth) + 0.5, (0.5 * intWidth) - 0.5, intWidth)
        .reshape(1, intWidth)
        .repeat(intHeight, 0)
        .astype(np.float32)[:, :, None]
    )
    npyImageplaneY = (
        np.linspace((-0.5 * intHeight) + 0.5, (0.5 * intHeight) - 0.5, intHeight)
        .reshape(intHeight, 1)
        .repeat(intWidth, 1)
        .astype(np.float32)[:, :, None]
    )
    npyImageplaneZ = np.full([intHeight, intWidth, 1], fltFocal, np.float32)
    npyImageplane = np.concatenate([npyImageplaneX, npyImageplaneY, npyImageplaneZ], 2)

    npyDepth = npyDistance / np.linalg.norm(npyImageplane, 2, 2) * fltFocal
    return npyDepth


class Hypersim(Dataset):
    def __init__(self, cfg, mode, size=(518, 518)):

        self.mode = mode
        self.size = size
        self.filelist_path = (
            cfg.dataset.train_split if self.mode == "train" else cfg.dataset.val_split
        )
        with open(self.filelist_path, "r") as f:
            self.filelist = f.read().splitlines()

        net_w, net_h = size
        self.transform = Compose(
            [
                Resize(
                    width=net_w,
                    height=net_h,
                    resize_target=True if mode == "train" else False,
                    keep_aspect_ratio=True,
                    ensure_multiple_of=14,
                    resize_method="lower_bound",
                    image_interpolation_method=cv2.INTER_CUBIC,
                ),
                NormalizeImage(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                PrepareForNet(),
            ]
            + ([Crop(size[0])] if self.mode == "train" else [])
        )
        self.train_augmentations = Compose(
            [
                RandomHorizontalFlip(p=cfg.dataset.augmentations.horizontal_flip.p),
                # This is too strong for depth estimation, results in loss of details
                ColorJitter(
                    brightness=cfg.dataset.augmentations.color_jitter.brightness,
                    contrast=cfg.dataset.augmentations.color_jitter.contrast,
                    saturation=cfg.dataset.augmentations.color_jitter.saturation,
                    hue=cfg.dataset.augmentations.color_jitter.hue,
                ),
                RandomGaussianBlur(p=cfg.dataset.augmentations.gaussian_blur.p),
                RandomGrayscaleProb(p=cfg.dataset.augmentations.random_grayscale.p),
            ]
        )

    def __getitem__(self, item):
        img_path = self.filelist[item].split(" ")[0]
        depth_path = self.filelist[item].split(" ")[1]

        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) / 255.0

        depth_fd = h5py.File(depth_path, "r")
        distance_meters = np.array(depth_fd["dataset"])
        depth = hypersim_distance_to_depth(distance_meters)
        # Apply data augmentations to training samples only.
        if self.mode == "train":
            sample = self.train_augmentations({"image": image, "depth": depth})
            image, depth = sample["image"], sample["depth"]

        sample = self.transform({"image": image, "depth": depth})

        sample["image"] = torch.from_numpy(sample["image"])
        sample["depth"] = torch.from_numpy(sample["depth"])

        sample["valid_mask"] = torch.isnan(sample["depth"]) == 0
        sample["depth"][sample["valid_mask"] == 0] = 0

        sample["image_path"] = self.filelist[item].split(" ")[0]

        return sample

    def __len__(self):
        return len(self.filelist)


if __name__ == "__main__":
    console = RichConsoleManager.get_console()
    console.log("Testing Hypersim Dataset")
    dataset_dict = {
        "train": "dataset/splits/hypersim/train.txt",
        "val": "dataset/splits/hypersim/val.txt",
    }
    for key, val in dataset_dict.items():
        console.log(f"Loading {key} dataset from {val}")
        dataset = Hypersim(
            filelist_path=val,
            mode=key,
        )
        console.print(f"{key} Dataset size: {len(dataset)}", style="info")
        if len(dataset) == 0:
            console.log(
                "[danger]No data found! Please check the file paths.[/danger]",
                style="danger",
            )
            exit(1)
        else:
            console.log("[info]Data loaded successfully![/info]")
            sample = dataset[0]
            console.print(f"Sample keys: {list(sample.keys())}", style="info")
            console.print(f"Image shape: {sample['image'].shape}", style="info")
            console.print(f"Depth shape: {sample['depth'].shape}", style="info")
            console.print(
                f"Valid mask shape: {sample['valid_mask'].shape}", style="info"
            )
            console.print(f"Image path: {sample['image_path']}", style="info")
