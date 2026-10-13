from __future__ import annotations

import os
from typing import Any

import torch
from comfy_api.latest import ComfyExtension, io
from safetensors.torch import load_file as load_safetensors
from typing_extensions import override

from heart_arch import HEART


def _to_channels_first(x: torch.Tensor) -> torch.Tensor:
    if x.dim() != 4:
        raise ValueError(f"Expected 4D image tensor, got shape {tuple(x.shape)}")
    if x.shape[1] in (1, 3, 4) and x.shape[-1] not in (1, 3, 4):
        return x
    if x.shape[-1] in (1, 3, 4):
        return x.permute(0, 3, 1, 2)
    return x


def _to_channels_last(x: torch.Tensor) -> torch.Tensor:
    if x.dim() != 4:
        raise ValueError(f"Expected 4D image tensor, got shape {tuple(x.shape)}")
    if x.shape[-1] in (1, 3, 4):
        return x
    return x.permute(0, 2, 3, 1)


def _load_heart_model(
    model_path: str = "",
    *,
    scale: int = 4,
    img_size: int = 64,
    embed_dim: int = 180,
    window_size: int = 32,
    rank: int = 8,
    rib_hidden_dim: int = 32,
    rib_n_freqs: int = 10,
    attention_freq: int = 2,
    device: torch.device | str | None = None,
) -> HEART:
    model = HEART(
        upscale=scale,
        img_size=img_size,
        embed_dim=embed_dim,
        window_size=window_size,
        rank=rank,
        rib_hidden_dim=rib_hidden_dim,
        rib_n_freqs=rib_n_freqs,
        attention_freq=attention_freq,
    )

    if model_path and os.path.exists(model_path):
        if model_path.lower().endswith(".safetensors"):
            state = load_safetensors(model_path)
        else:
            state = torch.load(model_path, map_location="cpu")
        if isinstance(state, dict) and any(k.startswith("module.") for k in state.keys()):
            state = {k.replace("module.", "", 1): v for k, v in state.items()}
        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]
        model.load_state_dict(state, strict=True)

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    return model


class HEARTModelLoader(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="HEARTModelLoader",
            display_name="HEART Model Loader",
            category="image/upscaling",
            inputs=[
                io.String.Input("model_path", default=""),
                io.Int.Input("scale", default=4, min=1, max=8, step=1),
                io.Int.Input("img_size", default=64, min=32, max=256, step=16),
                io.Int.Input("embed_dim", default=180, min=64, max=512, step=4),
                io.Int.Input("window_size", default=32, min=8, max=64, step=8),
                io.Int.Input("rank", default=8, min=1, max=32, step=1),
                io.Int.Input("rib_hidden_dim", default=32, min=8, max=128, step=4),
                io.Int.Input("rib_n_freqs", default=10, min=0, max=20, step=1),
                io.Int.Input("attention_freq", default=2, min=1, max=8, step=1),
            ],
            outputs=[
                io.Model.Output(),
            ],
        )

    @classmethod
    def execute(
        cls,
        model_path: str,
        scale: int,
        img_size: int,
        embed_dim: int,
        window_size: int,
        rank: int,
        rib_hidden_dim: int,
        rib_n_freqs: int,
        attention_freq: int,
    ) -> io.NodeOutput:
        model = _load_heart_model(
            model_path=model_path,
            scale=scale,
            img_size=img_size,
            embed_dim=embed_dim,
            window_size=window_size,
            rank=rank,
            rib_hidden_dim=rib_hidden_dim,
            rib_n_freqs=rib_n_freqs,
            attention_freq=attention_freq,
        )
        return io.NodeOutput(model)


class HEARTUpscaleImage(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="HEARTUpscaleImage",
            display_name="HEART Upscale Image",
            category="image/upscaling",
            inputs=[
                io.Model.Input("model"),
                io.Image.Input("image"),
            ],
            outputs=[
                io.Image.Output(),
            ],
        )

    @classmethod
    def execute(cls, model: Any, image: torch.Tensor) -> io.NodeOutput:
        if not isinstance(model, torch.nn.Module):
            raise TypeError("Expected a HEART model instance")

        device = next(model.parameters()).device
        x = image.detach().clone().to(device=device)
        if x.dim() == 3:
            x = x.unsqueeze(0)
        x = _to_channels_first(x)
        if x.dtype != torch.float32:
            x = x.float()

        if x.min() < 0.0:
            x = x.clamp(0.0, 1.0)
        if x.max() > 1.0:
            x = x / 255.0

        with torch.no_grad():
            y = model(x)

        y = y.clamp(0.0, 1.0)
        y = _to_channels_last(y)
        return io.NodeOutput(y)


class HeartExtension(ComfyExtension):
    @override
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return [
            HEARTModelLoader,
            HEARTUpscaleImage,
        ]


async def comfy_entrypoint() -> HeartExtension:
    return HeartExtension()
