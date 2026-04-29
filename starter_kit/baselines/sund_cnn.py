#!/bin/env python
# -*- coding: utf-8 -*-
#
# Built for the CI 2026 hackathon starter kit

# System modules
import logging
from typing import Dict, Any

# External modules
import torch
import torch.nn

# Internal modules
from starter_kit.model import BaseModel
from starter_kit.layers import InputNormalisation
from .utils import estimate_relative_humidity
from .sundquist import SundquistNetwork

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

_normalisation_mean = torch.Tensor([[
    294.531359,287.010605,278.507482,262.805241,227.580722,201.364517,209.719502,], # T
    #0.010667,0.006922,0.003784,0.001229,0.000088,0.000003, 0.000003, # q
    [0,0,0,0,0,0,0,], # rh
    [-1.412110,-0.914917,0.431349,3.504875,11.699176,6.758849,-1.214763,], # U
    [0.167424,-0.105374,-0.172138,-0.022648,0.030789,0.281048,-0.094608,], # V
    #[0.410844,2129.684371, 0, 0, 0, 0.9645212, 1.6217752, 3.9218636,], #2D
])
_normalisation_std = torch.Tensor([
    [62.864550,61.180621,58.938862,56.016099,47.532073,32.281805,38.084321,], #T
    #0.006102,0.004648,0.003013,0.001266,0.000080,0.000001,0.000000, # specific humidity
    [1,1,1,1,1,1,1,], #relative humidity
    [4.661358,6.159993,7.763541,9.877940,16.068963,11.681901,10.705570,], #U
    [4.119853,4.318767,4.810067,6.209760,10.585627,5.680168,2.978756,], #v
    #[0.498762,3602.712270, 1, 1, 1, 1.2400768, 2.8122723, 6.4065957,], # 2d
])


class SundCNNNetwork(torch.nn.Module):
    r'''
    Multi-layer perceptron operating on flattened pressure-
    level and auxiliary fields.

    Parameters
    ----------
    input_dim : int, optional, default = 30
        Total number of input features after concatenation.
    hidden_dim : int, optional, default = 64
        Width of each hidden layer.
    n_layers : int, optional, default = 4
        Number of hidden Linear+SiLU blocks.
    normalisation : InputNormalisation or None, optional
        Pre-normalisation layer applied before the MLP. When
        provided it must accept the concatenated input tensor
        of shape ``(..., input_dim)``.

    Attributes
    ----------
    normalisation : InputNormalisation or None
        Normalisation layer stored as a sub-module.
    mlp : torch.nn.Sequential
        Sequence of linear layers with SiLU activations.
    '''
    def __init__(
            self,
            input_dim: int = 30,
            hidden_dim: int = 64,
            n_layers: int = 4,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.normalisation = InputNormalisation(
            mean=torch.tensor(_normalisation_mean.T),
            std=torch.tensor(_normalisation_std.T)
        )

        # old version kept in case
        """
        layers = [
            torch.nn.Conv2d(1, hidden_dim, (7,4)),
            torch.nn.SiLU()
        ]
        for i in range(n_layers-1):
            #layers.append(torch.nn.LayerNorm(hidden_dim))
            layers.append(torch.nn.Conv2d(hidden_dim*(i+1), hidden_dim*(i+2),1, padding="same"))
            #layers.append(torch.nn.Dropout(0.1))
            layers.append(torch.nn.SiLU())
        layers.append(torch.nn.Conv2d(hidden_dim*(n_layers), 1, 1, padding="same"))
        layers.append(torch.nn.Flatten())
        self.cnn_ = torch.nn.Sequential(*layers)
        output_layer = torch.nn.Linear(28, 1)
        torch.nn.init.normal_(output_layer.weight, std=1E-4)
        torch.nn.init.normal_(output_layer.bias, std=0.5)
        layers.append(output_layer)
        layers.append(torch.nn.Tanh())
        #layers.append(torch.nn.Hardtanh(min_val=0, max_val=1.0,))
        self.cnn = torch.nn.Sequential(*layers)
        """
        self.cnn = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=1, out_channels=16, kernel_size=(3, 2)),
            torch.nn.ReLU(),

            torch.nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(3, 2)),
            torch.nn.ReLU(),
            torch.nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3, 2), padding='same'),
            torch.nn.ReLU(),

            torch.nn.Flatten(),
            torch.nn.Linear(32 * 3 * 2, 1),
            torch.nn.Tanh()
        )
        print(self.cnn)
        self.register_buffer(
            "pressure_levels",
            torch.tensor(
                [1000_00, 850_00, 700_00, 500_00, 250_00, 100_00, 50_00]
            ).reshape(-1, 1, 1)
        )
        self.sq=SundquistNetwork()


    def forward(
            self,
            input_level: torch.Tensor,
            input_auxiliary: torch.Tensor,
            noise=False
    ) -> torch.Tensor:
        r'''
        Forward pass: concatenate inputs, optionally normalise,
        then apply the MLP.

        Parameters
        ----------
        input_level : torch.Tensor
            Pressure-level fields, shape ``(B, C_l, L, H, W)``.
        input_auxiliary : torch.Tensor
            Auxiliary fields, shape ``(B, C_a, H, W)``.

        Returns
        -------
        torch.Tensor
            Predictions of shape ``(B, 1, H, W)``.
        '''
        sundquist_prediction = self.sq(
            input_level=input_level,
            input_auxiliary=input_auxiliary
        )

        bs = input_level.shape[0]
        level_rh = estimate_relative_humidity(
            temperature=input_level[:, 0:1],
            specific_humidity=input_level[:, 1:2],
            pressure=self.pressure_levels
        )
        input_level[:, 1:2] = level_rh
        
        # We collapse all levels into the channel dimension
        # (time, vars_level, level, lat, lon) 
        
        input_level = input_level.movedim(2,-1)
        input_level = input_level.movedim(1,-1)
        cnn_input = input_level.reshape(
            input_level.shape[0]*input_level.shape[1]*input_level.shape[2],1, *input_level.shape[-2:]
        )
      
        # Apply input normalisation        
        cnn_input = self.normalisation(cnn_input)#4,7
        

        # Apply the MLP (modified by ayoub)
        prediction = self.cnn(cnn_input)
        
        # Move the channel dimension to the expected position
        prediction = prediction.reshape(bs, 64, 64, 1)
        correction = prediction.movedim(-1, 1)

        
        prediction = sundquist_prediction+correction
        prediction = prediction.clamp(0., 1.)
        return prediction


class SundCNNModel(BaseModel):
    r'''
    Model wrapper for an MLP network with standard loss outputs.

    This class delegates forward execution to a hidden MLP network and
    computes a mean absolute error loss together with auxiliary metrics.
    '''

    def estimate_loss(
            self,
            batch: Dict[str, torch.Tensor]
    ) -> Dict[str, Any]:
        r'''
        Compute the primary training loss and prediction output.

        Parameters
        ----------
        batch : Dict[str, torch.Tensor]
            Batch dictionary containing ``input_level``,
            ``input_auxiliary``, and ``target`` tensors.

        Returns
        -------
        Dict[str, Any]
            Dictionary with keys ``loss`` and ``prediction``.
            ``loss`` is the mean absolute error and ``prediction`` is the
            model output clamped to ``[0, 1]``.
        '''
        prediction = self.network(
            input_level=batch["input_level"],
            input_auxiliary=batch["input_auxiliary"]
        )
        
        
        loss = (prediction - batch["target"]).abs()
        loss = loss * self.lat_weights
        mae = loss.mean()
        #crps_loss = self.crps_loss(batch)
        loss = mae #+ crps_loss
        return {"loss": loss, 
                #"mae_loss": mae, 
                #"crps_loss": crps_loss, 
                "prediction": prediction}

    def estimate_auxiliary_loss(
            self,
            batch: Dict[str, torch.Tensor],
            outputs: Dict[str, Any]
    ) -> Dict[str, Any]:
        r'''
        Compute auxiliary regression and classification metrics.

        Parameters
        ----------
        batch : Dict[str, torch.Tensor]
            Batch dictionary containing the ground-truth ``target`` tensor.
        outputs : Dict[str, Any]
            Model outputs from ``estimate_loss`` containing ``prediction``.

        Returns
        -------
        Dict[str, Any]
            Dictionary with keys ``mse`` and ``accuracy``.
            ``mse`` is the mean squared error and ``accuracy`` is the
            thresholded classification accuracy at 0.5.
        '''
        mse = (outputs["prediction"] - batch["target"]).pow(2)
        mse = (mse * self.lat_weights).mean()
        prediction_bool = (outputs["prediction"] > 0.5).float()
        target_bool = (batch["target"] > 0.5).float()
        accuracy = (prediction_bool == target_bool).float()
        accuracy = (accuracy * self.lat_weights).mean()
        return {"mse": mse, "accuracy": accuracy}

    def crps_loss(self,
            batch: Dict[str, torch.Tensor]):
        prediction = self.network(
            input_level=batch["input_level"],
            input_auxiliary=batch["input_auxiliary"]
        )
        n_ensembles = 10
        noise_size = 0.05
    
        prediction_ensembles = torch.zeros((n_ensembles, *prediction.shape)).to(self.device)
        for i in range(n_ensembles):
            noise = noise_size*torch.rand((prediction.shape[1], self.network.input_dim))+1-noise_size/2
            prediction_ensembles[i] = self.network(
                input_level=batch["input_level"],
                input_auxiliary=batch["input_auxiliary"],
                noise=noise.to(self.device)
            )
       
        y_hat_mean = torch.mean(prediction_ensembles * self.lat_weights, dim=0)
        y_hat_asc = torch.sort(prediction_ensembles * self.lat_weights, dim=0)[0]
        y_hat_des = torch.sort(prediction_ensembles * self.lat_weights, dim=0,descending=True)[0]
        
        MAE_term = torch.mean(torch.abs(y_hat_mean-batch["target"]* self.lat_weights), dim=0)
        spread_term = torch.mean(0.5*torch.mean(torch.abs(y_hat_asc - y_hat_des), dim=0), dim=0)
        CRPS_loss=torch.abs(MAE_term-spread_term)
        crps_loss = torch.mean(CRPS_loss)
        return crps_loss
        