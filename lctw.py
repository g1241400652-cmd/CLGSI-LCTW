"""Low-confidence task-aligned output-gradient reweighting for CLGSI."""

from __future__ import annotations


TAU = 0.25
RHO = 0.10
NORM_CAP_ABSOLUTE_TOLERANCE = 1e-7
ALIGNMENT_ABSOLUTE_TOLERANCE = 1e-8


class LCTWInputError(ValueError):
    """Raised when inputs do not satisfy the LCTW computation requirements."""


def _autocast_enabled(torch_module):
    enabled = bool(torch_module.is_autocast_enabled())
    try:
        enabled = enabled or bool(torch_module.is_autocast_enabled("cpu"))
    except TypeError:
        if hasattr(torch_module, "is_autocast_cpu_enabled"):
            enabled = enabled or bool(torch_module.is_autocast_cpu_enabled())
    return enabled


def _validate_single_scalar_batch(tensor, name):
    if tensor.ndim == 1:
        batch_size = tensor.shape[0]
    elif tensor.ndim == 2 and tensor.shape[1] == 1:
        batch_size = tensor.shape[0]
    else:
        raise LCTWInputError(
            f"LCTW {name} must have shape [B] or [B,1]"
        )
    if batch_size <= 0 or tensor.numel() != batch_size:
        raise LCTWInputError(
            f"LCTW {name} must contain exactly one scalar per batch element"
        )
    return int(batch_size)


def lctw_surrogate(
    torch_module,
    fused_prediction,
    fusion_label_map,
    gamma_runtime,
):
    """Return a zero-forward surrogate and detached gradient diagnostics."""

    if _autocast_enabled(torch_module):
        raise LCTWInputError("LCTW expects float32 computation without autocast")
    if fused_prediction.dtype != torch_module.float32:
        raise LCTWInputError("LCTW fused predictions must use torch.float32")
    if fusion_label_map.dtype != torch_module.float32:
        raise LCTWInputError("LCTW fusion labels must use torch.float32")

    batch_size = _validate_single_scalar_batch(fused_prediction, "prediction")
    label_batch_size = _validate_single_scalar_batch(fusion_label_map, "label")
    if batch_size != label_batch_size:
        raise LCTWInputError(
            "LCTW prediction and fusion label batch sizes must match"
        )

    prediction = fused_prediction.reshape(batch_size)
    labels = fusion_label_map.reshape(batch_size).detach()
    if prediction.shape != labels.shape:
        raise LCTWInputError("LCTW canonical scalar views must match exactly")
    if not bool(torch_module.all(torch_module.isfinite(prediction))) or not bool(
        torch_module.all(torch_module.isfinite(labels))
    ):
        raise LCTWInputError("LCTW inputs must be finite")

    gamma_tensor = torch_module.as_tensor(
        gamma_runtime, dtype=prediction.dtype, device=prediction.device
    )
    if gamma_tensor.ndim != 0:
        raise LCTWInputError("LCTW official runtime gamma must be scalar")
    if not bool(torch_module.isfinite(gamma_tensor)) or not bool(gamma_tensor > 0):
        raise LCTWInputError(
            "LCTW official runtime gamma must be finite and positive"
        )

    detached_prediction = prediction.detach()
    task_gradient = torch_module.sign(detached_prediction - labels) / batch_size
    compact_weight = torch_module.clamp(
        1.0 - torch_module.abs(detached_prediction) / TAU,
        min=0.0,
        max=1.0,
    ).detach()
    auxiliary_total_gradient = (
        RHO * compact_weight * task_gradient
    ).detach()
    injected_contrastive_gradient = (
        auxiliary_total_gradient / gamma_tensor
    ).detach()

    tensors = (
        task_gradient,
        compact_weight,
        auxiliary_total_gradient,
        injected_contrastive_gradient,
    )
    if not all(bool(torch_module.all(torch_module.isfinite(value))) for value in tensors):
        raise LCTWInputError("LCTW weights and gradients must be finite")

    confident = torch_module.abs(detached_prediction) >= TAU
    if bool(torch_module.any(auxiliary_total_gradient[confident] != 0)):
        raise LCTWInputError(
            "LCTW confident rows must receive zero auxiliary gradient"
        )

    alignment = torch_module.sum(auxiliary_total_gradient * task_gradient)
    task_norm = torch_module.linalg.vector_norm(task_gradient)
    auxiliary_norm = torch_module.linalg.vector_norm(auxiliary_total_gradient)
    if bool(alignment < -ALIGNMENT_ABSOLUTE_TOLERANCE):
        raise LCTWInputError("LCTW auxiliary gradient is task-opposing")
    if bool(
        auxiliary_norm
        > RHO * task_norm + NORM_CAP_ABSOLUTE_TOLERANCE
    ):
        raise LCTWInputError("LCTW auxiliary gradient exceeded its norm cap")

    injected_scalar = torch_module.sum(injected_contrastive_gradient * prediction)
    zero_surrogate = injected_scalar - injected_scalar.detach()
    if not torch_module.equal(
        zero_surrogate, torch_module.zeros_like(zero_surrogate)
    ) or bool(torch_module.signbit(zero_surrogate)):
        raise LCTWInputError(
            "LCTW surrogate must be bitwise float32 +0.0"
        )

    return zero_surrogate, {
        "batch_size": batch_size,
        "prediction_original_shape": list(fused_prediction.shape),
        "label_original_shape": list(fusion_label_map.shape),
        "canonical_shape": list(prediction.shape),
        "gamma_runtime": gamma_tensor.detach(),
        "tau": TAU,
        "rho": RHO,
        "active_rows": int((compact_weight > 0).sum()),
        "confident_rows": int(confident.sum()),
        "compact_weight": compact_weight,
        "task_gradient": task_gradient.detach(),
        "auxiliary_total_gradient": auxiliary_total_gradient,
        "analytic_injected_contrastive_gradient": injected_contrastive_gradient,
        "alignment_dot": alignment.detach(),
        "task_gradient_norm": task_norm.detach(),
        "auxiliary_total_gradient_norm": auxiliary_norm.detach(),
    }


def make_lctw_loss_class(official_loss_class, torch_module, gamma_runtime):
    """Add LCTW to an existing CLGSI contrastive-loss class."""

    class LCTWContrastiveLoss(official_loss_class):

        def forward(self, outputs, label_map):
            official_loss = super().forward(outputs, label_map)
            zero_surrogate, diagnostics = (
                lctw_surrogate(
                    torch_module, outputs["M"], label_map, gamma_runtime
                )
            )
            candidate_loss = official_loss + zero_surrogate
            if not torch_module.equal(candidate_loss, official_loss):
                raise LCTWInputError(
                    "LCTW contrastive forward equality failed"
                )
            self.diagnostics = {
                key: value.detach().cpu() if hasattr(value, "detach") else value
                for key, value in diagnostics.items()
            }
            return candidate_loss

    LCTWContrastiveLoss.__name__ = "ClgsiMethodLCTWContrastiveLoss"
    return LCTWContrastiveLoss
