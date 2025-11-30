import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Tuple

# ┌───────────────────────────────────────────────┐
# │              HELPER CLASSES                   │
# └───────────────────────────────────────────────┘

class TimeAwareConv(nn.Module):
    """
    Replaces 'double_conv'. 
    It takes the image 'x' AND the 'time_emb'.
    It injects the time info into the features so the layer knows the noise level.
    """
    def __init__(self, in_channels, out_channels, time_embedding_dim):
        super().__init__()
        
        # 1. First Convolution
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.GroupNorm(32, out_channels),
            nn.SiLU(),
        )
        
        # 2. Time Projection (The "Time Injection")
        # Projects time_emb from 64 -> out_channels
        self.time_proj = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_embedding_dim, out_channels)
        )

        # 3. Second Convolution
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.GroupNorm(32, out_channels),
            nn.SiLU(),
        )

    def forward(self, x, t_emb):
        # Run first conv
        h = self.conv1(x)
        
        # INJECT TIME: Project time embedding and add to features
        # We reshape time_emb to (Batch, Channels, 1, 1) to broadcast
        time_bias = self.time_proj(t_emb)
        h = h + time_bias[:, :, None, None]
        
        # Run second conv
        h = self.conv2(h)
        return h

# ┌───────────────────────────────────────────────┐
# │               NETWORK DEFINITION              │
# └───────────────────────────────────────────────┘
class ConsistencyUNet(nn.Module):
    def __init__(self, 
                 in_channels: int = 1, 
                 out_channels: int = 1, 
                 time_embedding_dim: int = 64,
                 sigma_data: float = 0.5,
                 epsilon: float = 0.002):
        
        super().__init__()
        self.time_embedding_dim = time_embedding_dim
        self.sigma_data = sigma_data
        self.epsilon = epsilon

        # Initial Projection (Scale up input channels)
        self.init_conv = nn.Conv2d(in_channels, 64, 3, padding=1)

        # Layers (Now using TimeAwareConv)
        # Note: We pass time_embedding_dim to every block
        self.dconv_down1 = TimeAwareConv(64, 64, time_embedding_dim)
        self.dconv_down2 = TimeAwareConv(64, 128, time_embedding_dim)
        self.dconv_down3 = TimeAwareConv(128, 256, time_embedding_dim)

        # FIX: Use AvgPool instead of MaxPool (Preserves info better for generation)
        self.avgpool = nn.AvgPool2d(2) 
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.dconv_up2 = TimeAwareConv(256 + 128, 128, time_embedding_dim)
        self.dconv_up1 = TimeAwareConv(128 + 64, 64, time_embedding_dim)
        
        self.conv_last = nn.Conv2d(64, out_channels, 1)

    def get_scaling_factors(self, t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        c_skip = self.sigma_data ** 2 / ((t - self.epsilon) ** 2 + self.sigma_data ** 2)
        c_out = (t - self.epsilon) * self.sigma_data / ((self.sigma_data ** 2) + (t ** 2)).sqrt()
        c_in = 1 / (self.sigma_data ** 2 + t ** 2).sqrt()
        return c_skip, c_out, c_in

    def forward(self, x: torch.Tensor, time_step: torch.Tensor) -> torch.Tensor:
        # 1. SCALING
        c_skip, c_out, c_in = self.get_scaling_factors(time_step)
        c_skip, c_out, c_in = c_skip.view(-1, 1, 1, 1), c_out.view(-1, 1, 1, 1), c_in.view(-1, 1, 1, 1)

        # 2. TIME EMBEDDING
        t_emb = sinusoidal_embedding(time_step, self.time_embedding_dim)

        # 3. U-NET BACKBONE
        # Scale input
        x = x * c_in
        
        # Initial Conv (No time yet)
        x1 = self.init_conv(x)
        
        # Encoder (Pass t_emb to every block!)
        x1 = self.dconv_down1(x1, t_emb)    # 28x28
        x2 = self.avgpool(x1)
        x2 = self.dconv_down2(x2, t_emb)    # 14x14
        x3 = self.avgpool(x2)
        x3 = self.dconv_down3(x3, t_emb)    # 7x7
        
        # Bottleneck
        # No need for manual concat anymore, the TimeAwareConv handles it if we ran another block here
        # But for simplicity, let's just proceed to upsampling
        
        # Decoder
        x_up = self.upsample(x3)            # 14x14
        x_up = torch.cat([x_up, x2], dim=1)
        x_up = self.dconv_up2(x_up, t_emb)  # Pass t_emb!
        
        x_up = self.upsample(x_up)          # 28x28
        x_up = torch.cat([x_up, x1], dim=1)
        x_up = self.dconv_up1(x_up, t_emb)  # Pass t_emb!

        F_x = self.conv_last(x_up)

        return c_skip * x + c_out * F_x

# Helper for sinusoidal embedding
def sinusoidal_embedding(times: torch.Tensor, dim: int) -> torch.Tensor:
    log_times = torch.log(times) * 0.25 
    half_dim = dim // 2
    emb = math.log(10000) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim, device=times.device) * -emb)
    emb = log_times.view(-1, 1) * emb.view(1, -1)
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
    return emb