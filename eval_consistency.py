import torch.nn as nn
import torch
from typing import List
import math

def sample(
        online_model: nn.Module,
        z_t: torch.Tensor,
        epsilon: float,
        sampling_schedule: List[int] # TODO: Change this to float if int is not possible
        ) -> torch.Tensor:
    """
    TODO:

    Args:
        online_model (nn.Module): _description_
        z_t (torch.Tensor): _description_
        epsilon (float): _description_
        sampling_schedule (_type_): _description_

    Returns:
        torch.Tensor: _description_
    """
    
    x = online_model(z_t, sampling_schedule[0])

    for t in sampling_schedule[1:]:
        z = torch.randn_like(x)
        z_t = x + (math.sqrt((t ** 2) - (epsilon ** 2)) * z)
        x = online_model(z_t, t)

    return x