import torch
import torch.nn as nn
import math
from typing import Tuple


# ┌───────────────────────────────────────────────┐
# │            TIME EMBEDDING HELPER              │
# └───────────────────────────────────────────────┘
def sinusoidal_embedding(times: torch.Tensor, dim: int) -> torch.Tensor:
    """
    Creates sinusoidal time embeddings.
    
    Args:
        times (torch.Tensor): Timestep values, shape (batch_size,). Should be in range [0, 1] or [0, num_steps]
        dim (int): Embedding dimension
        
    Returns:
        torch.Tensor: Sinusoidal embeddings of shape (batch_size, dim)
    """
    # Normalize times to [0, 1] if they're large
    # Assuming times are in range [0, num_steps], we scale to reasonable range
    device = times.device
    
    # Create frequency bands
    half_dim = dim // 2
    freqs = torch.exp(
        torch.arange(half_dim, device=device, dtype=torch.float32) * 
        -(math.log(10000.0) / (half_dim - 1))
    )
    
    # Reshape times for broadcasting: (batch_size,) -> (batch_size, 1)
    times_reshaped = times.view(-1, 1)
    
    # Compute sinusoidal embeddings
    args = times_reshaped * freqs.view(1, -1)
    embedding = torch.cat([torch.sin(args), torch.cos(args)], dim=1)
    
    return embedding


# ┌───────────────────────────────────────────────┐
# │           TIME-AWARE CONVOLUTION              │
# └───────────────────────────────────────────────┘
class TimeAwareConv(nn.Module):
    """
    Convolution block with time embedding injection.
    Designed for DDPM noise prediction.
    """
    def __init__(self, in_channels: int, out_channels: int, time_embedding_dim: int):
        super().__init__()
        
        # First convolution block
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.GroupNorm(32, out_channels),
            nn.SiLU(),
        )
        
        # Time projection: time_embedding_dim -> out_channels
        self.time_proj = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_embedding_dim, out_channels)
        )
        
        # Second convolution block
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.GroupNorm(32, out_channels),
            nn.SiLU(),
        )

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch_size, in_channels, H, W)
            t_emb: (batch_size, time_embedding_dim)
            
        Returns:
            (batch_size, out_channels, H, W)
        """
        # First convolution
        h = self.conv1(x)  # (batch_size, out_channels, H, W)
        
        # Inject time embedding
        time_bias = self.time_proj(t_emb)  # (batch_size, out_channels)
        h = h + time_bias[:, :, None, None]  # Broadcast to spatial dimensions
        
        # Second convolution
        h = self.conv2(h)  # (batch_size, out_channels, H, W)
        
        return h


# ┌───────────────────────────────────────────────┐
# │          RESIDUAL BLOCK WITH TIME             │
# └───────────────────────────────────────────────┘
class ResidualBlock(nn.Module):
    """Residual block with time conditioning for DDPM."""
    
    def __init__(self, in_channels: int, out_channels: int, time_embedding_dim: int):
        super().__init__()
        self.same_channels = in_channels == out_channels
        
        self.conv_block = TimeAwareConv(in_channels, out_channels, time_embedding_dim)
        
        if not self.same_channels:
            self.residual_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        
    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch_size, in_channels, H, W)
            t_emb: (batch_size, time_embedding_dim)
            
        Returns:
            (batch_size, out_channels, H, W)
        """
        h = self.conv_block(x, t_emb)
        
        if self.same_channels:
            return h + x
        else:
            return h + self.residual_conv(x)


# ┌───────────────────────────────────────────────┐
# │              DDPM UNET MODEL                  │
# └───────────────────────────────────────────────┘
class DDPMUNet(nn.Module):
    """
    Standard DDPM UNet architecture for noise prediction.
    
    This model is designed to predict noise in the forward diffusion process.
    It takes a noisy image and timestep and predicts the Gaussian noise.
    
    Architecture: Encoder -> Bottleneck -> Decoder with skip connections
    """
    
    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        time_embedding_dim: int = 128,
        feature_map_dim: int = 64,
        max_channels: int = 256,
    ):
        super().__init__()
        
        self.time_embedding_dim = time_embedding_dim
        
        # ──────────────────────────────────────────
        # Time Embedding Projection
        # ──────────────────────────────────────────
        self.time_embedding = nn.Sequential(
            nn.Linear(time_embedding_dim, time_embedding_dim * 2),
            nn.SiLU(),
            nn.Linear(time_embedding_dim * 2, time_embedding_dim),
        )
        
        # ──────────────────────────────────────────
        # Initial Convolution (Project input)
        # ──────────────────────────────────────────
        self.init_conv = nn.Conv2d(in_channels, feature_map_dim, kernel_size=3, padding=1)
        
        # ──────────────────────────────────────────
        # Encoder (Downsampling)
        # ──────────────────────────────────────────
        # Level 1: feature_map_dim -> feature_map_dim (28x28 for MNIST)
        self.down1_res1 = ResidualBlock(feature_map_dim, feature_map_dim, time_embedding_dim)
        self.down1_res2 = ResidualBlock(feature_map_dim, feature_map_dim, time_embedding_dim)
        self.down1_pool = nn.AvgPool2d(2)  # 28x28 -> 14x14
        
        # Level 2: feature_map_dim -> feature_map_dim * 2
        self.down2_res1 = ResidualBlock(feature_map_dim, feature_map_dim * 2, time_embedding_dim)
        self.down2_res2 = ResidualBlock(feature_map_dim * 2, feature_map_dim * 2, time_embedding_dim)
        self.down2_pool = nn.AvgPool2d(2)  # 14x14 -> 7x7
        
        # Level 3: feature_map_dim * 2 -> feature_map_dim * 4 (bottleneck)
        self.down3_res1 = ResidualBlock(feature_map_dim * 2, feature_map_dim * 4, time_embedding_dim)
        self.down3_res2 = ResidualBlock(feature_map_dim * 4, feature_map_dim * 4, time_embedding_dim)
        
        # ──────────────────────────────────────────
        # Decoder (Upsampling)
        # ──────────────────────────────────────────
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')
        
        # Level 2 (going up): feature_map_dim * 4 + feature_map_dim * 2 -> feature_map_dim * 2
        self.up2_res1 = ResidualBlock(
            feature_map_dim * 4 + feature_map_dim * 2, 
            feature_map_dim * 2, 
            time_embedding_dim
        )
        self.up2_res2 = ResidualBlock(feature_map_dim * 2, feature_map_dim * 2, time_embedding_dim)
        
        # Level 1 (going up): feature_map_dim * 2 + feature_map_dim -> feature_map_dim
        self.up1_res1 = ResidualBlock(
            feature_map_dim * 2 + feature_map_dim, 
            feature_map_dim, 
            time_embedding_dim
        )
        self.up1_res2 = ResidualBlock(feature_map_dim, feature_map_dim, time_embedding_dim)
        
        # ──────────────────────────────────────────
        # Final Output Layer
        # ──────────────────────────────────────────
        self.final_conv = nn.Sequential(
            nn.Conv2d(feature_map_dim, feature_map_dim, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Conv2d(feature_map_dim, out_channels, kernel_size=1),
        )

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for DDPM noise prediction.
        
        Args:
            x: Noisy image of shape (batch_size, in_channels, H, W)
            t: Timestep indices of shape (batch_size,)
               Values should be in range [0, num_diffusion_steps)
               
        Returns:
            Predicted noise of shape (batch_size, out_channels, H, W)
        """
        # Generate sinusoidal time embedding
        t_emb_raw = sinusoidal_embedding(t, self.time_embedding_dim)  # (batch_size, time_embedding_dim)
        
        # Project time embedding through MLP
        t_emb = self.time_embedding(t_emb_raw)  # (batch_size, time_embedding_dim)
        
        # Initial convolution
        h = self.init_conv(x)  # (batch_size, feature_map_dim, H, W)
        
        # ──────────────────────────────────────────
        # Encoder with Skip Connections
        # ──────────────────────────────────────────
        h1 = self.down1_res1(h, t_emb)  # (batch_size, feature_map_dim, 28, 28)
        h1 = self.down1_res2(h1, t_emb)
        skip1 = h1
        h1 = self.down1_pool(h1)  # (batch_size, feature_map_dim, 14, 14)
        
        h2 = self.down2_res1(h1, t_emb)  # (batch_size, feature_map_dim*2, 14, 14)
        h2 = self.down2_res2(h2, t_emb)
        skip2 = h2
        h2 = self.down2_pool(h2)  # (batch_size, feature_map_dim*2, 7, 7)
        
        # Bottleneck
        h3 = self.down3_res1(h2, t_emb)  # (batch_size, feature_map_dim*4, 7, 7)
        h3 = self.down3_res2(h3, t_emb)
        
        # ──────────────────────────────────────────
        # Decoder with Skip Connections
        # ──────────────────────────────────────────
        h3 = self.upsample(h3)  # (batch_size, feature_map_dim*4, 14, 14)
        h3 = torch.cat([h3, skip2], dim=1)  # (batch_size, feature_map_dim*4 + feature_map_dim*2, 14, 14)
        h3 = self.up2_res1(h3, t_emb)
        h3 = self.up2_res2(h3, t_emb)
        
        h3 = self.upsample(h3)  # (batch_size, feature_map_dim*2, 28, 28)
        h3 = torch.cat([h3, skip1], dim=1)  # (batch_size, feature_map_dim*2 + feature_map_dim, 28, 28)
        h3 = self.up1_res1(h3, t_emb)
        h3 = self.up1_res2(h3, t_emb)
        
        # Final output
        output = self.final_conv(h3)  # (batch_size, out_channels, 28, 28)
        
        return output


if __name__ == '__main__':
    # Test the model
    model = DDPMUNet()
    
    # Create dummy inputs
    x = torch.randn(4, 1, 28, 28)  # Batch of 4 MNIST images
    t = torch.randint(0, 1000, (4,))  # Random timesteps
    
    # Forward pass
    output = model(x, t)
    
    print(f"Input shape: {x.shape}")
    print(f"Timestep shape: {t.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Output range: [{output.min():.4f}, {output.max():.4f}]")
    
    # Check parameter count
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")
