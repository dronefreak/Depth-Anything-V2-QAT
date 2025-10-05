# quantization/qat_helper.py
# Helper functions for Quantization Aware Training (QAT) using PyTorch's eager mode
# Reference: https://pytorch.org/docs/stable/quantization.html#quantization-aware
import functools

from omegaconf import DictConfig
import torch
from torch.ao.quantization import (
    DeQuantStub,
    QuantStub,
    convert,
    get_default_qconfig,
    prepare_qat,
)


def _move_model_to_device_and_verify(model: torch.nn.Module, device: torch.device):
    model.to(device)

    # Normalize devices for comparison
    def _same_device(a, b):
        da, db = torch.device(a), torch.device(b)
        return da.type == db.type and (
            da.index == db.index or da.index is None or db.index is None
        )

    bad_params = [
        (n, p.device)
        for n, p in model.named_parameters()
        if not _same_device(p.device, device)
    ]
    bad_bufs = [
        (n, b.device)
        for n, b in model.named_buffers()
        if not _same_device(b.device, device)
    ]

    if bad_params or bad_bufs:
        print(
            f"[Auto-fix] Moving {len(bad_params)}"
            f" params and {len(bad_bufs)} buffers to {device}"
        )
        for _, p in bad_params:
            p.data = p.data.to(device)
        for _, b in bad_bufs:
            b.data = b.data.to(device)

    return model


def _warmup_model_for_tracing(
    model: torch.nn.Module, device: torch.device, input_shape=(1, 3, 224, 224)
):
    """Run a single forward on a synthetic tensor to ensure any runtime-created tensors
    land on the target device prior to torch.compile / tracing.

    Returns False if warmup failed (so caller can decide).
    """
    model.eval()
    try:
        dummy = torch.zeros(input_shape, device=device)
        with torch.no_grad():
            _ = model(dummy)
        return True
    except Exception as e:
        print(f"[warmup] model warmup forward failed: {e}")
        return False


def _add_quant_stubs(model):
    """Add QuantStub at input and DeQuantStub at output.

    Safer wrapper using functools.wraps.
    """
    if not hasattr(model, "quant"):
        model.quant = QuantStub()
    if not hasattr(model, "dequant"):
        model.dequant = DeQuantStub()

    # Store original forward if not present
    if not hasattr(model, "_original_forward"):
        model._original_forward = model.forward

    @functools.wraps(model._original_forward)
    def forward_with_quant(x, *args, **kwargs):
        # ensure input is on the same device as model parameters to avoid mixing
        device = (
            next(model.parameters()).device
            if any(p.numel() for p in model.parameters())
            else x.device
        )
        if x.device != device:
            x = x.to(device)
        x = model.quant(x)
        x = model._original_forward(x, *args, **kwargs)
        x = model.dequant(x)
        return x

    model.forward = forward_with_quant
    return model


def prepare_model_for_qat(model, cfg: DictConfig):
    """Prepare model for QAT using eager mode (compatible with DINOv2's dynamic pos
    encoding)."""
    if not cfg.quantization.enabled:
        return model

    # Set quantization backend
    backend = cfg.quantization.backend
    if backend not in ["qnnpack", "fbgemm"]:
        raise ValueError(f"Unsupported backend: {backend}. Use 'qnnpack' or 'fbgemm'")

    torch.backends.quantized.engine = backend
    print(f"Using quantization backend: {backend}")

    # Add quantization stubs
    model = _add_quant_stubs(model)

    # Configure QConfig
    qconfig = get_default_qconfig(backend)
    model.qconfig = qconfig
    print(f"QConfig set: {qconfig}")

    # Prepare for QAT (use inplace=True to keep the same model instance)
    # This avoids subtle issues where prepare_qat returns
    # a new instance without our added attributes
    model_prepared = prepare_qat(model, inplace=True)
    print("Model prepared for QAT (inplace)")

    return model_prepared


def convert_model_to_quantized(model):
    """Convert trained QAT model to fully quantized model."""
    model.eval()

    try:
        quantized_model = convert(model)
        print("Model converted to fully quantized")
    except Exception as e:
        print(f"Warning: Full conversion failed: {e}")
        print("Returning partially quantized model (some ops remain in FP32)")
        quantized_model = model  # Already has quantized weights from QAT

    return quantized_model


def calibrate_model(model, dataloader, num_steps=100):
    """Calibration for QAT (less critical than PTQ, but helps initialize quant
    params)."""
    model.eval()
    print(f"Calibrating model with {num_steps} batches...")

    with torch.no_grad():
        for i, sample in enumerate(dataloader):
            if i >= num_steps:
                break
            img = sample["image"].to(next(model.parameters()).device, non_blocking=True)
            _ = model(img)
            if i % 20 == 0:
                print(f"Calibration step {i}/{num_steps}")

    print("Calibration complete")
