from .mlp import MLPModel, MLPNetwork
from .parametric import ParametricModel, ParametricNetwork
from .sundquist import SundquistNetwork
from .all_vars_mlp import AllVarsMLPModel, AllVarsMLPNetwork
from .mlp_res import ResMLPModel, ResMLPNetwork
from .sund_corr import SundMLPModel, SundMLPNetwork
from .sund_cnn import SundCNNModel, SundCNNNetwork
from .CNN import CNNModel, CNNNetwork

__all__ = [
    "MLPModel",
    "MLPNetwork",
    "ParametricModel",
    "ParametricNetwork",
    "SundquistNetwork",
    "AllVarsMLPModel",
    "AllVarsMLPNetwork",
    "ResMLPModel",
    "ResMLPNetwork",
    "SundMLPModel",
    "SundMLPNetwork",
    "SundCNNModel",
    "SundCNNNetwork",
    "CNNModel",
    "CNNNetwork"
    
]
