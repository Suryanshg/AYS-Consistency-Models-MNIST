import torch
import torch.nn as nn
import math
from typing import Tuple

# ┌───────────────────────────────────────────────┐
# │                 HELPER NETWORK                │
# └───────────────────────────────────────────────┘
class TimeAwareConv(nn.Module):
    def __init__(self, in_channels, out_channels, time_embedding_dim):
        super().__init__()
        
        # First Convolution Block
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.GroupNorm(32, out_channels),
            nn.SiLU(),
        )
        
        # Time Projection
        # Projects time_emb from time_embedding_dim -> out_channels
        self.time_proj = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_embedding_dim, out_channels)
        )

        # Second Convolution Block
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.GroupNorm(32, out_channels),
            nn.SiLU(),
        )


    def forward(self, x, t_emb):
        # x shape: (N, C_in, H, W)
        # t_emb: (N, time_embedding_dim)

        # Forward pass thru first convolution block
        h = self.conv1(x)                                   # (N, C_out, H, W)
        
        # INJECT TIME: Project time embedding and add to features
        # We reshape time_emb to (Batch, Channels, 1, 1) to broadcast
        time_bias = self.time_proj(t_emb)                   # (N, C_out)
        h = h + time_bias[:, :, None, None]                 # (N, C_out, H, W)
        
        # Run second conv
        h = self.conv2(h)                                   # (N, C_out, H, W)
        return h


# ┌───────────────────────────────────────────────┐
# │                 MAIN NETWORK                  │
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

        # Initial Convolution (Scale up from input channels)
        self.init_conv = nn.Conv2d(in_channels, 64, 3, padding=1)

        # Downsampling Layers with time_embedding injection in every block
        self.dconv_down1 = TimeAwareConv(64, 64, time_embedding_dim)
        self.dconv_down2 = TimeAwareConv(64, 128, time_embedding_dim)
        self.dconv_down3 = TimeAwareConv(128, 256, time_embedding_dim)

        self.avgpool = nn.AvgPool2d(2)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        # Upsampling Layers with time_embedding injection in every block
        self.dconv_up2 = TimeAwareConv(256 + 128, 128, time_embedding_dim)
        self.dconv_up1 = TimeAwareConv(128 + 64, 64, time_embedding_dim)
        
        # Final Convolution (Scale down to output channels)
        self.conv_last = nn.Conv2d(64, out_channels, 1)


    def get_scaling_factors(self, t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        c_skip = self.sigma_data ** 2 / ((t - self.epsilon) ** 2 + self.sigma_data ** 2)
        c_out = (t - self.epsilon) * self.sigma_data / ((self.sigma_data ** 2) + (t ** 2)).sqrt()
        c_in = 1 / (self.sigma_data ** 2 + t ** 2).sqrt()
        return c_skip, c_out, c_in


    def forward(self, x: torch.Tensor, time_step: torch.Tensor) -> torch.Tensor:
        # x shape: (N, 1, H, W)
        # time_step shape: (N,)

        # Fetch Scaling Factors using the time step
        c_skip, c_out, c_in = self.get_scaling_factors(time_step)                                               # All have shape: (N,)                                     
        c_skip, c_out, c_in = c_skip.view(-1, 1, 1, 1), c_out.view(-1, 1, 1, 1), c_in.view(-1, 1, 1, 1)         # All have shape: (N, 1, 1, 1)

        # Generate sinusoidal style time embedding
        t_emb = sinusoidal_embedding(time_step, self.time_embedding_dim)        # (N, time_embedding_dim)

        # 3. U-NET BACKBONE
        # Scale input
        x_scaled = x * c_in                                                     # (N, 1, H, W)                                                            
        
        # Initial Conv (No Time Injection)
        x1 = self.init_conv(x_scaled)                                           # (N, 64, H, W)
        
        # Encoder
        x1 = self.dconv_down1(x1, t_emb)                                        # (N, 64, H, W)
        x2 = self.avgpool(x1)                                                   # (N, 64, H/2, W/2)
        x2 = self.dconv_down2(x2, t_emb)                                        # (N, 128, H/2, W/2)
        x3 = self.avgpool(x2)                                                   # (N, 128, H/4, W/4)
        x3 = self.dconv_down3(x3, t_emb)                                        # (N, 256, H/4, W/4) 

        
        # Decoder
        x_up = self.upsample(x3)                                                # (N, 256, H/2, W/2)
        x_up = torch.cat([x_up, x2], dim=1)                                     # (N, 384, H/2, W/2)
        x_up = self.dconv_up2(x_up, t_emb)                                      # (N, 128, H/2, W/2)
        
        x_up = self.upsample(x_up)                                              # (N, 128, H, W)
        x_up = torch.cat([x_up, x1], dim=1)                                     # (N, 192, H, W)
        x_up = self.dconv_up1(x_up, t_emb)                                      # (N, 64, H, W)

        F_x = self.conv_last(x_up)                                              # (N, 1, H, W)

        return c_skip * x + c_out * F_x                                         # (N, 1, H, W)


# ┌───────────────────────────────────────────────┐
# │                HELPER METHODS                 │
# └───────────────────────────────────────────────┘
def sinusoidal_embedding(times: torch.Tensor, dim: int) -> torch.Tensor:
    log_times = torch.log(times) * 0.25 
    half_dim = dim // 2
    emb = math.log(10000) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim, device=times.device) * -emb)
    emb = log_times.view(-1, 1) * emb.view(1, -1)
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
    return emb


if __name__ == '__main__':
    cm = ConsistencyUNet()
    print(cm)