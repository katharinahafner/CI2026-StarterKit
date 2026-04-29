import xarray as xr 
from IPython import embed

path = '/home/ubuntu/CI2026-StarterKit/data/train_data/train.zarr'
data = xr.open_mfdataset(path)

embed()