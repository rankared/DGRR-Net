"""
Training helpers for the proposed E2c model.

The residual and slope-loss definitions are exact Phase-6 definitions.
The epoch loop is a function-level refactor of the successfully executed
Phase-6 cell, preserving AdamW/AMP/gradient-clipping behavior.
"""

from __future__ import annotations

from contextlib import nullcontext

import torch

def masked_l1_loss(
    prediction,
    target,
    valid,
):

    denominator = torch.clamp(
        valid.sum(),
        min=1.0,
    )

    numerator = (
        torch.abs(
            prediction - target
        ) * valid
    ).sum()

    return (
        numerator
        / denominator
    )


def masked_slope_loss(
    prediction,
    target,
    valid,
):

    valid_x = (
        valid[:, :, :, 1:]
        * valid[:, :, :, :-1]
    )

    valid_y = (
        valid[:, :, 1:, :]
        * valid[:, :, :-1, :]
    )

    pred_dx = (
        prediction[:, :, :, 1:]
        - prediction[:, :, :, :-1]
    )

    target_dx = (
        target[:, :, :, 1:]
        - target[:, :, :, :-1]
    )

    pred_dy = (
        prediction[:, :, 1:, :]
        - prediction[:, :, :-1, :]
    )

    target_dy = (
        target[:, :, 1:, :]
        - target[:, :, :-1, :]
    )

    numerator = (
        (
            torch.abs(
                pred_dx - target_dx
            ) * valid_x
        ).sum()
        +
        (
            torch.abs(
                pred_dy - target_dy
            ) * valid_y
        ).sum()
    )

    denominator = torch.clamp(
        valid_x.sum()
        + valid_y.sum(),
        min=1.0,
    )

    return (
        numerator
        / denominator
    )


def build_optimizer_and_scheduler(
    model,
    epochs=80,
    learning_rate=1e-3,
    weight_decay=1e-4,
    eta_min=1e-5,
):
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )

    scheduler = (
        torch.optim.lr_scheduler
        .CosineAnnealingLR(
            optimizer,
            T_max=epochs,
            eta_min=eta_min,
        )
    )

    return (
        optimizer,
        scheduler,
    )


def train_one_epoch(
    model,
    loader,
    optimizer,
    scaler,
    device,
    lambda_slope=0.5,
    gradient_clip_norm=5.0,
    amp_enabled=None,
):
    if amp_enabled is None:
        amp_enabled = (
            device.type == "cuda"
        )

    model.train()

    total_sum = 0.0
    residual_sum = 0.0
    slope_sum = 0.0
    batch_count = 0

    for batch in loader:

        x = batch["x"].to(
            device,
            non_blocking=True,
        )

        y_norm = batch[
            "y_norm"
        ].to(
            device,
            non_blocking=True,
        )

        valid = batch[
            "valid"
        ].to(
            device,
            non_blocking=True,
        )

        optimizer.zero_grad(
            set_to_none=True
        )

        autocast_context = (
            torch.autocast(
                device_type="cuda",
                dtype=torch.float16,
            )
            if amp_enabled
            else nullcontext()
        )

        with autocast_context:

            prediction = model(x)

            residual_loss = (
                masked_l1_loss(
                    prediction,
                    y_norm,
                    valid,
                )
            )

            slope_loss = (
                masked_slope_loss(
                    prediction,
                    y_norm,
                    valid,
                )
            )

            total_loss = (
                residual_loss
                + lambda_slope
                * slope_loss
            )

        if not torch.isfinite(
            total_loss
        ):
            raise RuntimeError(
                "Non-finite training loss."
            )

        scaler.scale(
            total_loss
        ).backward()

        scaler.unscale_(
            optimizer
        )

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=gradient_clip_norm,
        )

        scaler.step(
            optimizer
        )

        scaler.update()

        total_sum += float(
            total_loss.item()
        )

        residual_sum += float(
            residual_loss.item()
        )

        slope_sum += float(
            slope_loss.item()
        )

        batch_count += 1

    denominator = max(
        batch_count,
        1,
    )

    return {
        "total_loss":
            total_sum
            / denominator,

        "residual_loss":
            residual_sum
            / denominator,

        "slope_loss":
            slope_sum
            / denominator,
    }
