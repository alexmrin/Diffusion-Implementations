import torch
import torch.nn as nn
import torch.nn.functional as F

def conv_nd(dims, *args, **kwargs) -> nn.Module:
    if dims == 1:
        return nn.Conv1d(*args, **kwargs)
    elif dims == 2:
        return nn.Conv2d(*args, **kwargs)
    elif dims == 3:
        return nn.Conv3d(*args, **kwargs)
    else:
        raise ValueError(f"Unsupported dim: {dims}")
    
def linear(*args, **kwargs):
    return nn.Linear(*args, **kwargs)

def avg_pool_nd(dims, *args, **kwargs) -> nn.Module:
    if dims == 1:
        return nn.AvgPool1d(*args, **kwargs)
    elif dims == 2:
        return nn.AvgPool2d(*args, **kwargs)
    elif dims == 3:
        return nn.AvgPool3d(*args, **kwargs)
    else:
        raise ValueError(f"Unsupported dim: {dims}")

def update_ema(target_params, source_params, rate=0.999) -> None:
    """
        We use EMA to update the new parameters slower. We want to smooth out the updating from the training.
        The formula to update is: rate * source_params + (1 - rate) * target_params
        The higher the rate, the slower the source_params update.
    """
    for tgt, src in zip(target_params, source_params):
        tgt.cpu().detach().mul_(rate).add_(src.cpu().detach(), alpha=1-rate)

def scale_module(module, scale):
    for p in module.parameters():
        p.detach().mul_(scale)
    return module