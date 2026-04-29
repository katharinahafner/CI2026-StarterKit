#!/bin/env python
# -*- coding: utf-8 -*-

import logging
from typing import Dict, Any

import torch
import torch.nn

from starter_kit.model import BaseModel
from starter_kit.layers import InputNormalisation

main_logger = logging.getLogger(__name__)

_normalisation_mean = [
    294.531359, 287.010605, 278.507482, 262.805241, 227.580722, 201.364517,
    209.719502, 0.010667, 0.006922, 0.003784, 0.001229, 0.000088, 0.000003,
    0.000003, -1.412110, -0.914917, 0.431349, 3.504875, 11.699176, 6.758849,
    -1.214763, 0.167424, -0.105374, -0.172138, -0.022648, 0.030789, 0.281048,
    -0.094608, 0.410844, 2129.684371
]
_normalisation_std = [
    62.864550, 61.180621, 58.938862, 56.016099, 47.532073, 32.281805, 38.084321,
    0.006102, 0.004648, 0.003013, 0.001266, 0.000080, 0.000001, 0.000000, 4.661358,
    6.159993, 7.763541, 9.877940, 16.068963, 11.681901, 10.705570, 4.119853, 4.318767,
    4.810067, 6.209760, 10.585627, 5.680168, 2.978756, 0.498762, 3602.712270
]

_N_METEO = 30


def make_positional_encoding(H: int, W: int, device: torch.device) -> torch.Tensor:
    lat_rad = torch.linspace(-torch.pi / 2, torch.pi / 2, H, device=device)
    lon_rad = torch.linspace(-torch.pi, torch.pi, W, device=device)
    lat_grid, lon_grid = torch.meshgrid(lat_rad, lon_rad, indexing="ij")
    return torch.stack([
        lat_grid.sin(), lat_grid.cos(),
        lon_grid.sin(), lon_grid.cos()
    ], dim=0)


class ConvBlock(torch.nn.Module):
    def __init__(self, channels: int, dropout: float = 0.1, kernel_size: int = 3):
        super().__init__()
        padding = kernel_size // 2
        self.block = torch.nn.Sequential(
            torch.nn.Conv2d(channels, channels, kernel_size=kernel_size, padding=padding, padding_mode='circular'),
            torch.nn.GroupNorm(8, channels),
            torch.nn.GELU(),
            torch.nn.Dropout(dropout),
            torch.nn.Conv2d(channels, channels, kernel_size=1),
            torch.nn.GroupNorm(8, channels),
            torch.nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.block(x)


class CNNNetwork(torch.nn.Module):
    def __init__(
            self,
            input_dim: int = 40,
            hidden_dim: int = 128,
            n_blocks: int = 4,
            dropout: float = 0.15,
    ) -> None:
        super().__init__()

        self.normalisation = InputNormalisation(
            mean=torch.tensor(_normalisation_mean),
            std=torch.tensor(_normalisation_std)
        )

        self.input_proj = torch.nn.Sequential(
            torch.nn.Conv2d(input_dim, hidden_dim, kernel_size=1),
            torch.nn.GELU(),
        )

        kernel_sizes = [1, 1, 3, 3]
        self.blocks = torch.nn.Sequential(
            *[ConvBlock(hidden_dim, dropout, kernel_size=kernel_sizes[i] if i < len(kernel_sizes) else 3)
              for i in range(n_blocks)]
        )

        mid = hidden_dim // 2
        self.head = torch.nn.Sequential(
            torch.nn.Conv2d(hidden_dim, mid, kernel_size=1),
            torch.nn.GELU(),
            torch.nn.Conv2d(mid, 1, kernel_size=1),
        )

        torch.nn.init.normal_(self.head[-1].weight, std=1e-4)
        torch.nn.init.constant_(self.head[-1].bias, 0.0)

    def forward(
            self,
            input_level: torch.Tensor,
            input_auxiliary: torch.Tensor
    ) -> torch.Tensor:
        B, _, H, W = input_auxiliary.shape

        flat_level = input_level.reshape(B, -1, H, W)
        all_aux = input_auxiliary

        pos = make_positional_encoding(H, W, input_level.device)
        pos = pos.unsqueeze(0).expand(B, -1, -1, -1)

        x = torch.cat([flat_level, all_aux, pos], dim=1)

        meteo = x[:, :_N_METEO].movedim(1, -1)
        meteo = self.normalisation(meteo)
        meteo = meteo.movedim(-1, 1)
        x = torch.cat([meteo, x[:, _N_METEO:]], dim=1)

        x = self.input_proj(x)
        x = self.blocks(x)
        x = self.head(x)
        return x


class CNNModel(BaseModel):

    def estimate_loss(
            self,
            batch: Dict[str, torch.Tensor]
    ) -> Dict[str, Any]:
        prediction = self.network(
            input_level=batch["input_level"],
            input_auxiliary=batch["input_auxiliary"]
        )
        prediction = prediction.clamp(0., 1.)
        loss = (prediction - batch["target"]).abs()
        loss = loss * self.lat_weights
        loss = loss.mean()
        return {"loss": loss, "prediction": prediction}

    def estimate_auxiliary_loss(
            self,
            batch: Dict[str, torch.Tensor],
            outputs: Dict[str, Any]
    ) -> Dict[str, Any]:
        mse = (outputs["prediction"] - batch["target"]).pow(2)
        mse = (mse * self.lat_weights).mean()
        prediction_bool = (outputs["prediction"] > 0.5).float()
        target_bool = (batch["target"] > 0.5).float()
        accuracy = (prediction_bool == target_bool).float()
        accuracy = (accuracy * self.lat_weights).mean()
        return {"mse": mse, "accuracy": accuracy}