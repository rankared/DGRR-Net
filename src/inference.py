"""
Strict checkpoint loading and residual-to-elevation inference helpers.
"""

from __future__ import annotations

from contextlib import nullcontext

import torch


def checkpoint_state_dict(
    checkpoint,
):
    if not isinstance(
        checkpoint,
        dict,
    ):
        raise TypeError(
            f"Unexpected checkpoint type: "
            f"{type(checkpoint)}"
        )

    if "model_state_dict" in checkpoint:

        state_dict = (
            checkpoint[
                "model_state_dict"
            ]
        )

    elif "state_dict" in checkpoint:

        state_dict = (
            checkpoint[
                "state_dict"
            ]
        )

    elif (
        checkpoint
        and all(
            torch.is_tensor(value)
            for value
            in checkpoint.values()
        )
    ):

        state_dict = checkpoint

    else:
        raise KeyError(
            "Checkpoint does not contain "
            "model_state_dict/state_dict "
            "and is not a plain tensor "
            "state dictionary."
        )

    if (
        state_dict
        and all(
            key.startswith(
                "module."
            )
            for key
            in state_dict
        )
    ):

        state_dict = {
            key[
                len("module.") :
            ]: value

            for key, value
            in state_dict.items()
        }

    return state_dict


def load_checkpoint(
    model,
    checkpoint_path,
    device="cpu",
):
    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False,
    )

    result = model.load_state_dict(
        checkpoint_state_dict(
            checkpoint
        ),
        strict=True,
    )

    if (
        result.missing_keys
        or result.unexpected_keys
    ):
        raise RuntimeError(
            "Strict checkpoint load failed. "
            f"Missing={result.missing_keys}, "
            f"unexpected={result.unexpected_keys}"
        )

    return checkpoint


@torch.no_grad()
def predict_batch(
    model,
    batch,
    norm_stats,
    device,
    amp_enabled=None,
):
    device = torch.device(
        device
    )

    if amp_enabled is None:
        amp_enabled = (
            device.type == "cuda"
        )

    x = batch["x"].to(
        device,
        non_blocking=True,
    )

    srtm_bc = batch[
        "srtm_bc"
    ].to(
        device,
        non_blocking=True,
    )

    autocast_context = (
        torch.autocast(
            device_type="cuda",
            dtype=torch.float16,
        )
        if amp_enabled
        else nullcontext()
    )

    model.eval()

    with autocast_context:
        prediction_norm = model(x)

    y_mean = float(
        norm_stats["y_mean"]
    )

    y_std = float(
        norm_stats["y_std"]
    )

    prediction_residual_m = (
        prediction_norm.float()
        * y_std
        + y_mean
    )

    prediction_elevation_m = (
        srtm_bc.float()
        + prediction_residual_m
    )

    return {
        "residual_normalized":
            prediction_norm.float(),

        "residual_m":
            prediction_residual_m,

        "elevation_m":
            prediction_elevation_m,
    }
