from pytorch_msssim import ssim
import torch
from torch import nn
import torch.nn.functional as F


class SiLogLoss(nn.Module):
    def __init__(self, lambd=0.5):
        super().__init__()
        self.lambd = lambd

    def forward(self, pred, target, valid_mask):
        valid_mask = valid_mask.detach()
        diff_log = torch.log(target[valid_mask]) - torch.log(pred[valid_mask])
        loss = torch.sqrt(
            torch.pow(diff_log, 2).mean() - self.lambd * torch.pow(diff_log.mean(), 2)
        )

        return loss


# ----------------------
# Gradient helpers
# ----------------------
def gradient_x(img):
    return img[:, :, :-1] - img[:, :, 1:]


def gradient_y(img):
    return img[:, :-1, :] - img[:, 1:, :]


class GradLoss(nn.Module):
    def __init__(self, scales=[1.0]):
        super().__init__()
        self.scales = scales

    def forward(self, pred, target):
        def gradient(x):
            dx = x[..., :, :-1] - x[..., :, 1:]
            dy = x[..., :-1, :] - x[..., 1:, :]
            return dx, dy

        dx_pred, dy_pred = gradient(pred)
        dx_tgt, dy_tgt = gradient(target)
        grad_loss = (
            torch.abs(dx_pred - dx_tgt).mean() + torch.abs(dy_pred - dy_tgt).mean()
        )
        return grad_loss


# ----------------------
# Multi-scale gradient loss with pooling (no interpolate)
# ----------------------
class MultiScaleGradientLoss(torch.nn.Module):
    def __init__(self, scales=[1.0, 0.5, 0.25, 0.125]):
        super().__init__()
        self.scales = scales

    def forward(self, pred, target):
        loss = 0.0

        for s in self.scales:
            if s == 1.0:
                pred_s, target_s = pred, target
            else:
                # Pooling to downsample instead of interpolate
                k = int(1 / s)
                pred_s = F.avg_pool2d(
                    pred.unsqueeze(1), kernel_size=k, stride=k
                ).squeeze(1)
                target_s = F.avg_pool2d(
                    target.unsqueeze(1), kernel_size=k, stride=k
                ).squeeze(1)

            dx_loss = (gradient_x(pred_s) - gradient_x(target_s)).abs().mean()
            dy_loss = (gradient_y(pred_s) - gradient_y(target_s)).abs().mean()
            loss += dx_loss + dy_loss

        return loss / len(self.scales)


# -------------------------------
# Generic Loss Manager
# -------------------------------


class GenericLoss(nn.Module):
    """Generic multi-loss module for monocular depth estimation.

    Supports: SiLog, SSIM, L1 photometric, and multi-scale gradient matching losses.
    """

    def __init__(self, loss_weights: dict, max_depth: float = 20.0, eps: float = 1e-6):
        super().__init__()
        self.loss_weights = loss_weights
        self.max_depth = max_depth
        self.eps = eps

    def forward(self, pred, target, valid_mask=None):
        total_loss = 0.0
        loss_dict = {}

        # ensure float tensors
        pred = pred.float()
        target = target.float()

        if valid_mask is not None:
            valid_mask = valid_mask.bool()
            pred = pred[valid_mask]
            target = target[valid_mask]

        # === SiLog Loss ===
        if "silog" in self.loss_weights:
            silog = self.silog_loss(pred, target)
            total_loss += self.loss_weights["silog"] * silog
            loss_dict["silog"] = silog.item()

        # === L1 Photometric Loss ===
        if "l1_photo" in self.loss_weights:
            l1_photo = torch.abs(pred - target).mean()
            total_loss += self.loss_weights["l1_photo"] * l1_photo
            loss_dict["l1_photo"] = l1_photo.item()

        # === SSIM Loss ===
        if "ssim" in self.loss_weights:
            # Ensure proper shape: (B, C, H, W)
            ssim_pred = pred.unsqueeze(1) if pred.ndim == 3 else pred
            ssim_target = target.unsqueeze(1) if target.ndim == 3 else target
            ssim_val = 1 - ssim(ssim_pred, ssim_target, data_range=1.0)
            total_loss += self.loss_weights["ssim"] * ssim_val
            loss_dict["ssim"] = ssim_val.item()

        # === Gradient Matching Loss ===
        if "grad" in self.loss_weights:
            grad_loss = self.gradient_loss(pred, target)
            total_loss += self.loss_weights["grad"] * grad_loss
            loss_dict["grad"] = grad_loss.item()

        return total_loss, loss_dict

    # ----------------------------------------------------
    # Individual loss components
    # ----------------------------------------------------
    @staticmethod
    def gradient_loss(pred, target):
        def gradient(x):
            dx = x[..., :, :-1] - x[..., :, 1:]
            dy = x[..., :-1, :] - x[..., 1:, :]
            return dx, dy

        dx_pred, dy_pred = gradient(pred)
        dx_tgt, dy_tgt = gradient(target)
        grad_loss = (
            torch.abs(dx_pred - dx_tgt).mean() + torch.abs(dy_pred - dy_tgt).mean()
        )
        return grad_loss
