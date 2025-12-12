import torch
import torch.nn as nn
from torch.nn import Sequential
import math
from typing import Tuple
from torchinfo import summary


# ┌───────────────────────────────────────────────┐
# │               NETWORK DEFINITION              │
# └───────────────────────────────────────────────┘
class ConsistencyUNet(nn.Module):
    """
    Definition of a simple implementation of a UNet for Consistency Models.
    """
    def __init__(self, 
                 in_channels: int = 1, 
                 out_channels: int = 1, 
                 time_embedding_dim: int = 64,
                 sigma_data: float = 0.5,
                 epsilon: float = 0.002):
        
        super().__init__()
        self.time_embedding_dim = time_embedding_dim

        # Consistency Parameters
        self.sigma_data = sigma_data
        self.epsilon = epsilon

        # Layers
        self.dconv_down1 = double_conv(in_channels + time_embedding_dim, 64)
        self.dconv_down2 = double_conv(64, 128)
        self.dconv_down3 = double_conv(128, 256)

        # self.maxpool = nn.MaxPool2d(2)
        self.maxpool = nn.AvgPool2d(2)
        # TODO: Look into using mode = 'nearest' later
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.dconv_up2 = double_conv(256 + time_embedding_dim + 128, 128)
        self.dconv_up1 = double_conv(128 + 64, 64)
        self.conv_last = nn.Conv2d(64, out_channels, 1)

    
    def get_scaling_factors(self, t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Computes c_skip and c_out scaling factors.
        Ensures that at epsilon (lowest noise), the model outputs the input exactly.
        Shape of t: (N,)
        """
        # Equations of c_skip and c_out from Consistency Models paper (Pg 26.)
        # c_skip = σ_data^2 / ((t - ϵ)^2 + σ_data^2)
        # c_out  = (t - ϵ) * σ_data / sqrt(t^2 + σ_data^2)
        c_skip = self.sigma_data ** 2 / ((t - self.epsilon) ** 2 + self.sigma_data ** 2)            # (N,)
        c_out = (t - self.epsilon) * self.sigma_data / ((self.sigma_data ** 2) + (t ** 2)).sqrt()   # (N,)

        c_in = 1 / ((self.sigma_data ** 2) + (t ** 2)).sqrt()                                       # (N,)
        
        return c_skip, c_out, c_in


    def forward(self, x: torch.Tensor, time_step: torch.Tensor) -> torch.Tensor:
        # x shape: (N, 1, H, W)
        # time_index shape: (N,)

        # Prepare Scaling Factors
        c_skip, c_out, c_in = self.get_scaling_factors(time_step)          # all have shape (N,)

        # Reshape scaling factors for broadcasting
        c_skip = c_skip.reshape(-1, 1, 1, 1)
        c_out = c_out.reshape(-1, 1, 1, 1)
        c_in = c_in.reshape(-1, 1, 1, 1)

        # Perform embedding of time using sinusoidal embedding
        time_embedding = sinusoidal_embedding(time_step, self.time_embedding_dim)           # (N, time_embedding_dim)

        # Scale input
        x_scaled = x * c_in
        # x_scaled = x

        x_in = torch.cat(
            [x_scaled, time_embedding.unsqueeze(-1).unsqueeze(-1).expand(x.size(0), -1, x.size(2), x.size(3))],
            dim=1
        )                                                           # (N, 1 + time_embedding_dim, H, W)

        # ┌───────────────────────────────────────────────┐
        # │         ENCODER BLOCKS (DOWNSAMPLING)         │
        # └───────────────────────────────────────────────┘
        # First Encoder Block
        conv1 = self.dconv_down1(x_in)                                  # (N, 64, H, W)
        x_enc = self.maxpool(conv1)                                     # (N, 64, H/2, W/2)

        # Second Encoder Block
        conv2 = self.dconv_down2(x_enc)                                 # (N, 128, H/2, W/2)
        x_enc = self.maxpool(conv2)                                     # (N, 128, H/4, W/4)

        # Third Encoder Block
        x_enc = self.dconv_down3(x_enc)                                 # (N, 256, H/4, W/4)


        # ┌───────────────────────────────────────────────┐
        # │         BOTTLENECK WITH TIME EMBEDDING        │
        # └───────────────────────────────────────────────┘
        # Add Time embedding as channel at bottleneck
        x_enc = torch.cat(
            [x_enc, time_embedding.unsqueeze(-1).unsqueeze(-1).expand(x_enc.size(0), -1, x_enc.size(2), x_enc.size(3))],
            dim=1
        )                                                               # (N, 256 + time_embedding_dim, H/4, W/4)
        x_enc = self.upsample(x_enc)                                    # (N, 256 + time_embedding_dim, H/2, W/2)


        # ┌───────────────────────────────────────────────┐
        # │           DECODER BLOCKS (UPSAMPLING)         │
        # └───────────────────────────────────────────────┘
        # First Decoder Block
        x_enc = torch.cat([x_enc, conv2], dim=1)                        # (N, 256 + time_embedding_dim + 128, H/2, W/2)
        x_enc = self.dconv_up2(x_enc)                                   # (N, 128, H/2, W/2)
        x_enc = self.upsample(x_enc)                                    # (N, 128, H, W)

        # Second Decoder Block
        x_enc = torch.cat([x_enc, conv1], dim=1)                        # (N, 128 + 64, H, W)
        x_enc = self.dconv_up1(x_enc)                                   # (N, 64, H, W)

        # Final Decoder Block
        F_x = self.conv_last(x_enc)                                     # (N, 1, H, W)

        # Apply Consistency Formula using scaling factors and return the output
        return (c_skip * x) + (c_out * F_x) 
    

# ┌───────────────────────────────────────────────┐
# │                HELPER METHODS                 │
# └───────────────────────────────────────────────┘
def sinusoidal_embedding(times: torch.Tensor, time_embedding_dim: int = 64) -> torch.Tensor:
    
    # LOGARITHMIC TRANSFORMATION
    log_times = torch.log(times) * 0.25 
    
    # EMBEDDING CALCULATION
    # We use half the dimensions for sin, half for cos
    half_dim = time_embedding_dim // 2
    
    # Calculate frequencies: From 1.0 down to 1/10000
    emb = math.log(10000) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim, device=times.device) * -emb)
    
    # COMBINE log times and embeddings
    # Shape: (batch, 1) * (1, half_dim) -> (batch, half_dim)
    emb = log_times.view(-1, 1) * emb.view(1, -1)
    
    # Compute sin and cos of embeddings and concatenate them
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
        
    return emb


def double_conv(in_channels: int, out_channels: int) -> Sequential:
    """
    Return a Convolutional Block with Two Convolutional Layers with ReLU Activation based on in_channels and
    out_channels. Uses kernel_size = 3 and padding = 1 to preserve spatial dimensions.

    Args:
        in_channels (int): Represents number of input channels
        out_channels (int): Represents number of output channels

    Returns:
        Sequential: represents a Convolutional Block with two Conv Layers.
    """
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, padding=1),     # (N, out_channels, H, W)
        nn.GroupNorm(num_groups=32, num_channels=out_channels),
        nn.SiLU(),

        nn.Conv2d(out_channels, out_channels, 3, padding=1),    # (N, out_channels, H, W)
        nn.GroupNorm(num_groups=32, num_channels=out_channels),
        nn.SiLU()
    )


if __name__ == '__main__':
    cm = ConsistencyUNet()

    input_data = (
        torch.randn(1, 1, 28, 28),      
        torch.rand(1,).clamp_min(1e-6),
    )

    summary(cm, input_data=input_data)