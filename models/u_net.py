import torch
import numpy as np
import torch.nn as nn
from torch.nn import Sequential

# ┌───────────────────────────────────────────────┐
# │               NETWORK DEFINITION              │
# └───────────────────────────────────────────────┘
class UNet(nn.Module):
    """
    Definition of a simple implementation of a UNet for Diffusion & Consistency Models.
    """
    def __init__(self, in_channels: int = 1, out_channels: int = 1, time_embedding_dim: int = 64):
        """
        Constructor for the UNet Class.

        Args:
            in_channels (int): Represents number of input channels
            out_channels (int): Represents number of output channels
            time_embedding_dim (int): Represents dimension of time embedding vector / tensor
        """
        super().__init__()

        self.time_embedding_dim = time_embedding_dim

        self.dconv_down1 = double_conv(in_channels + time_embedding_dim, 64)
        self.dconv_down2 = double_conv(64, 128)
        self.dconv_down3 = double_conv(128, 256)
        self.dconv_down4 = double_conv(256, 512)

        self.maxpool = nn.MaxPool2d(2)

        # TODO: Look into using mode = 'nearest' later
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.dconv_up3 = double_conv(512 + time_embedding_dim + 256, 256)
        self.dconv_up2 = double_conv(256 + 128, 128)
        self.dconv_up1 = double_conv(128 + 64, 64)

        self.conv_last = nn.Conv2d(64, out_channels, 1)


    def forward(self, x: torch.Tensor, time_step: torch.Tensor) -> torch.Tensor:
        """
        Forward Pass Method for the UNet for Diffusion / Consistency Model. Uses Time Embedding as Channel to the Input of both the Encoder
        Blocks and the Decoder Blocks (via the Bottleneck layers).

        Args:
            x (torch.Tensor): represents in the input (usually image / latent code). Should be of shape (N, 1, H, W).
            time_step (torch.Tensor): represents the timestep of the image / latent code. Should be of shape (N,).

        Returns:
            torch.Tensor: represents the output of the UNet Forward Pass. Will be of the same shape as input (N, 1, H, W).
        """
        # x shape: (N, 1, H, W)
        # time_index shape: (N,)

        # Perform embedding of time using sinusoidal embedding
        time_embedding = sinusoidal_embedding(time_step, self.time_embedding_dim)           # (N, time_embedding_dim)
        x = torch.cat(
            [x, time_embedding.unsqueeze(-1).unsqueeze(-1).expand(x.size(0), -1, x.size(2), x.size(3))],
            dim=1
        )                                                           # (N, 1 + time_embedding_dim, H, W)

        # ┌───────────────────────────────────────────────┐
        # │         ENCODER BLOCKS (DOWNSAMPLING)         │
        # └───────────────────────────────────────────────┘
        # First Encoder Block
        conv1 = self.dconv_down1(x)                                 # (N, 64, H, W)
        x = self.maxpool(conv1)                                     # (N, 64, H/2, W/2) --> Maxpool halves spatial dims

        # Second Encoder Block
        conv2 = self.dconv_down2(x)                                 # (N, 128, H/2, W/2)
        x = self.maxpool(conv2)                                     # (N, 128, H/4, W/4)

        # Third Encoder Block
        conv3 = self.dconv_down3(x)                                 # (N, 256, H/4, W/4)
        x = self.maxpool(conv3)                                     # (N, 256, H/8, W/8)

        # Final Encoder Block
        x = self.dconv_down4(x)                                     # (N, 512, H/8, W/8)


        # ┌───────────────────────────────────────────────┐
        # │         BOTTLENECK WITH TIME EMBEDDING        │
        # └───────────────────────────────────────────────┘
        # Add Time embedding as channel at bottleneck
        x = torch.cat(
            [x, time_embedding.unsqueeze(-1).unsqueeze(-1).expand(x.size(0), -1, x.size(2), x.size(3))],
            dim=1
        )                                                           # (N, 512 + time_embedding_dim, H/8, W/8)
        x = self.upsample(x)                                        # (N, 512 + time_embedding_dim, H/4, W//4) --> Upsample doubles spatial dims


        # ┌───────────────────────────────────────────────┐
        # │           DECODER BLOCKS (UPSAMPLING)         │
        # └───────────────────────────────────────────────┘
        # First Decoder Block
        x = torch.cat([x, conv3], dim=1)                            # (N, 512 + time_embedding_dim + 256, H/4, W/4)
        x = self.dconv_up3(x)                                       # (N, 256, H/4, W/4)
        x = self.upsample(x)                                        # (N, 256, H/2, W/2)

        # Second Decoder Block
        x = torch.cat([x, conv2], dim=1)                            # (N, 256 + 128, H/2, W/2)
        x = self.dconv_up2(x)                                       # (N, 128, H/2, W/2)
        x = self.upsample(x)                                        # (N, 128, H, W)

        # Third Decoder Block
        x = torch.cat([x, conv1], dim=1)                            # (N, 128 + 64, H, W)
        x = self.dconv_up1(x)                                       # (N, 64, H, W)

        # Final Decoder Block
        out = self.conv_last(x)                                     # (N, 1, H, W)

        # Return the output of the forward pass
        return out  

# ┌───────────────────────────────────────────────┐
# │                HELPER METHODS                 │
# └───────────────────────────────────────────────┘
def sinusoidal_embedding(times: torch.Tensor, time_embedding_dim: int = 64, T: int = 1000) -> torch.Tensor:
    """
    Consumes a tensor representing timesteps and returns a tensor of sinusoidal embeddings.

    Args:
        times (torch.Tensor): tensor representing timesteps. Should be of shape (batch_size,)
        time_embedding_dim (int): represents the dimension of the time_embedding to use.
        T (int): represents total number of timesteps.

    Returns:
        torch.Tensor: tensor representing sinusoidal embeddings for each timestep
    """
    # Set the min frequency
    embedding_min_frequency = 1.0

    # Compute Frequencies
    frequencies = torch.exp(
        torch.linspace(
            np.log(embedding_min_frequency),
            np.log(T),
            time_embedding_dim // 2
        )
    ).view(1, -1).to(times.device)                      # (1, time_embedding_dim // 2)

    # Convert Frequencies to Angular Speeds (ω = 2 * π * frequency)
    angular_speeds = 2.0 * torch.pi * frequencies       # (1, time_embedding_dim // 2)

    # Convert times to a 2D tensor with float32 dtype
    times = times.view(-1, 1).float()                   # (batch_size, 1)

    # Compute Embeddings by Matrix Multiplication of times tensor with angular speeds
    embeddings = torch.cat(
        [torch.sin(times.matmul(angular_speeds) / T), torch.cos(times.matmul(angular_speeds) / T)], dim=1
    )                                                   # (batch_size, time_embedding_dim)

    # Return the computed sinusoidal embeddings
    return embeddings


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
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, 3, padding=1),    # (N, out_channels, H, W)
        nn.ReLU(inplace=True)
    )