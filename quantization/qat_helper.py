# quantization/qat_helper.py
# Helper functions for Quantization Aware Training (QAT) using PyTorch's eager mode
# Reference: https://pytorch.org/docs/stable/quantization.html#quantization-aware
import functools
from typing import Any

from omegaconf import DictConfig
import torch
from torch.ao.quantization import (
    DeQuantStub,
    MinMaxObserver,
    MovingAverageMinMaxObserver,
    PerChannelMinMaxObserver,
    QConfig,
    QuantStub,
    convert,
    prepare_qat,
)


def _dequantize_output(module, out: Any):
    """Recursively apply module.dequant to all Tensor elements in the output.

    Keeps tuples, lists, and dicts structurally identical.
    """
    if torch.is_tensor(out):
        return module.dequant(out)
    elif isinstance(out, tuple):
        return tuple(_dequantize_output(module, o) for o in out)
    elif isinstance(out, list):
        return [_dequantize_output(module, o) for o in out]
    elif isinstance(out, dict):
        return {k: _dequantize_output(module, v) for k, v in out.items()}
    else:
        return out


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


def prepare_model_for_qat_selective(
    model: torch.nn.Module, cfg: DictConfig, target_submodules=("depth_head", "decoder")
):
    """
    Prepare model for QAT selectively:
      - Use per-channel weight observer for Conv2d/Linear where supported.
      - Use per-tensor weight observer for ConvTranspose* modules
        (unsupported for per-channel).
      - Attach quant stubs at decoder entry/exit and
        run prepare_qat on the decoder only.
    `target_submodules` can be names of top-level modules
     you want quantized (default: depth_head/decoder).
    """

    if not cfg.quantization.enabled:
        return model

    backend = cfg.quantization.backend
    if backend not in ["qnnpack", "fbgemm"]:
        raise ValueError(f"Unsupported backend: {backend}. Use 'qnnpack' or 'fbgemm'")

    torch.backends.quantized.engine = backend
    print(f"Using quantization backend: {backend}")

    # Observers: moving-average for activations, per-channel for weights (where allowed)
    act_observer = MovingAverageMinMaxObserver.with_args(
        reduce_range=False, dtype=torch.qint32
    )
    # Per-channel for weight (works for Conv2d, Linear)
    weight_perch = PerChannelMinMaxObserver.with_args(
        dtype=torch.qint8, qscheme=torch.per_channel_symmetric
    )
    # Fallback per-tensor observer for weight (works for all modules)
    weight_pertensor = MinMaxObserver.with_args(
        dtype=torch.qint8, qscheme=torch.per_tensor_symmetric
    )

    # QConfig variants
    qconfig_per_channel = QConfig(activation=act_observer, weight=weight_perch)
    qconfig_per_tensor = QConfig(activation=act_observer, weight=weight_pertensor)

    # Helper: attach Quant/DeQuant only to the decoder(s) (so encoder stays FP32)
    def _attach_decoder_qstubs(root_model, target_names):
        for sub_name in target_names:
            if hasattr(root_model, sub_name):
                sub = getattr(root_model, sub_name)
                # if already wrapped, skip
                if not hasattr(sub, "quant") and not hasattr(sub, "dequant"):
                    sub.quant = torch.ao.quantization.QuantStub()
                    sub.dequant = torch.ao.quantization.DeQuantStub()
                    # wrap forward safely
                    if not hasattr(sub, "_original_forward"):
                        sub._original_forward = sub.forward

                        @functools.wraps(sub._original_forward)
                        def forward_q(x, *args, **kwargs):
                            # Determine device
                            # (prefer model parameter device, fallback to input)
                            try:
                                device = next(root_model.parameters()).device
                            except StopIteration:
                                device = getattr(x, "device", torch.device("cpu"))

                            # Move input to device if mismatched
                            if torch.is_tensor(x) and str(x.device) != str(device):
                                x = x.to(device)

                            # Quantize -> forward -> dequantize recursively
                            qx = sub.quant(x)
                            out = sub._original_forward(qx, *args, **kwargs)
                            out = _dequantize_output(sub, out)
                            return out

                        sub.forward = forward_q

                        print(
                            "Attached Quant/DeQuant stub and"
                            f" wrapper to submodule: {sub_name}"
                        )
            else:
                print(
                    "[prepare_model_for_qat_selective]"
                    f" root model has no attribute '{sub_name}', skipping."
                )

    # Attach stubs only to the decoder/heads you specified
    _attach_decoder_qstubs(model, target_submodules)

    # Assign qconfig selectively across modules:
    # per-channel for Conv2d/Linear, per-tensor for ConvTranspose
    conv_types_per_channel = (torch.nn.Conv2d, torch.nn.Linear)
    convtranspose_types = (
        torch.nn.ConvTranspose1d,
        torch.nn.ConvTranspose2d,
        torch.nn.ConvTranspose3d,
    )

    for name, module in model.named_modules():
        # Only set qconfig for modules that are
        # inside target_submodules (so encoder unaffected).
        if not any(name == t or name.startswith(f"{t}.") for t in target_submodules):
            # keep as default or None (no qconfig) to avoid quantizing encoder
            module.qconfig = None
            continue

        # Module is inside the target area (decoder/head)
        if isinstance(module, conv_types_per_channel):
            module.qconfig = qconfig_per_channel
        elif isinstance(module, convtranspose_types):
            # ConvTranspose doesn't support per-channel weights yet; force per-tensor
            module.qconfig = qconfig_per_tensor
        else:
            # For other modules (LayerNorm, GELU, etc.)
            # leave None or use activation observer only
            # If you want to enable activation observers for all, use:
            # module.qconfig = QConfig(activation=act_observer, weight=weight_pertensor)
            # But for safety, set None so they remain FP32.
            module.qconfig = None

    # Finally, call prepare_qat only on the submodules we targeted.
    # If target_submodules has multiple entries, prepare each separately.
    for sub_name in target_submodules:
        if hasattr(model, sub_name):
            submodule = getattr(model, sub_name)
            try:
                prepare_qat(submodule, inplace=True)
                print(f"prepare_qat inplace called on {sub_name}")
            except Exception as e:
                print(f"Warning: prepare_qat failed on {sub_name}: {e}")

    print("Selective QAT preparation complete.")
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
    qconfig = QConfig(
        activation=torch.ao.quantization.MovingAverageMinMaxObserver.with_args(
            qscheme=torch.per_tensor_affine
        ),
        weight=torch.ao.quantization.PerChannelMinMaxObserver.with_args(
            dtype=torch.qint8, qscheme=torch.per_channel_symmetric
        ),
    )
    # qconfig = get_default_qconfig(backend)
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
