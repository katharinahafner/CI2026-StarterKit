#!/bin/env python
# -*- coding: utf-8 -*-
#
# Built for the CI 2026 hackathon starter kit


import logging
from typing import Dict, Any

# External modules
import torch
import torch.nn


from starter_kit.model import BaseModel
from starter_kit.layers import InputNormalisation


main_logger = logging.getLogger(__name__)

r'''
The normalisation mean and std values are pre-computed from the training data.
As in the MLP, all pressure levels are collapsed into the channels dimension
and only the first two auxiliary fields (land sea mask and geopotential) are
used. For each of these 30 input features we compute the mean and std across
all spatial locations, weighted by the latitude weights, and averaged across
all time steps in the training set. These values are stored in the lists below
and used to initialise the InputNormalisation layer in the MLPNetwork.
'''

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





def make_positional_encoding(H: int, W: int, device: torch.device) -> torch.Tensor:
    
    lat_rad = torch.linspace(-torch.pi / 2, torch.pi / 2, H, device=device)
    lon_rad = torch.linspace(-torch.pi, torch.pi, W, device=device)
    lat_grid, lon_grid = torch.meshgrid(lat_rad, lon_rad, indexing="ij")
    pos = torch.stack([
        lat_grid.sin(), lat_grid.cos(),
        lon_grid.sin(), lon_grid.cos()
    ], dim=0)                         # (4, H, W)
    return pos

# addition of my residual layer 
class ResidualBlock(torch.nn.Module):
    def __init__(self, hidden_dim: int, dropout: float = 0.1):
        super().__init__()
        self.block = torch.nn.Sequential(
            torch.nn.LayerNorm(hidden_dim),
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.GELU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.block(x)


class ResMLPNetwork(torch.nn.Module):
    
    def __init__(
            self,
            input_dim: int = 40,
            hidden_dim: int = 256,
            n_layers: int = 6,
            dropout: float = 0.1,
    ) -> None:
        super().__init__()

       
        self.normalisation = InputNormalisation(
            mean=torch.tensor(_normalisation_mean),
            std=torch.tensor(_normalisation_std)
        )

        self.input_proj = torch.nn.Sequential(
            torch.nn.Linear(input_dim, hidden_dim),
            torch.nn.GELU(),
        )
        self.blocks = torch.nn.Sequential(
            *[ResidualBlock(hidden_dim, dropout) for _ in range(n_layers)]
        )
        mid = hidden_dim // 2
        self.head = torch.nn.Sequential(
            torch.nn.LayerNorm(hidden_dim),
            torch.nn.Linear(hidden_dim, mid),
            torch.nn.GELU(),
            torch.nn.Linear(mid, 1),
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

        
        mlp_input = torch.cat([flat_level, all_aux, pos], dim=1) 

       
        mlp_input = mlp_input.movedim(1, -1)

        
        mlp_input[..., :30] = self.normalisation(mlp_input[..., :30])

        
        x = self.input_proj(mlp_input)
        x = self.blocks(x)
        prediction = self.head(x)

        
        prediction = prediction.movedim(-1, 1)
        return prediction


class ResMLPModel(BaseModel):
  

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