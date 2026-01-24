"""
DDPM Generation and FID Evaluation Script
Generates samples from trained DDPM model and evaluates using FID score
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from tqdm import tqdm
import matplotlib.pyplot as plt
from torchvision import transforms
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
import os

# ============================================================================
# Device Configuration
# ============================================================================
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Device: {DEVICE}')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}')

# ============================================================================
# Model Components (Same as training)
# ============================================================================
def sinusoidal_embedding(times, dim):
    """Sinusoidal positional embeddings for time steps"""
    half_dim = dim // 2
    freqs = torch.exp(torch.arange(half_dim, device=times.device, dtype=torch.float32) * 
                     -(math.log(10000.0) / (half_dim - 1)))
    args = times.view(-1, 1) * freqs.view(1, -1)
    return torch.cat([torch.sin(args), torch.cos(args)], dim=1)

class TimeAwareConv(nn.Module):
    """Convolution with time-dependent bias"""
    def __init__(self, in_c, out_c, time_dim):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_c, out_c, 3, padding=1),
            nn.GroupNorm(32, out_c),
            nn.SiLU()
        )
        self.time_proj = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_dim, out_c)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_c, out_c, 3, padding=1),
            nn.GroupNorm(32, out_c),
            nn.SiLU()
        )
    
    def forward(self, x, t_emb):
        h = self.conv1(x)
        h = h + self.time_proj(t_emb)[:, :, None, None]
        return self.conv2(h)

class ResidualBlock(nn.Module):
    """Residual block with optional dimension projection"""
    def __init__(self, in_c, out_c, time_dim):
        super().__init__()
        self.same = in_c == out_c
        self.conv = TimeAwareConv(in_c, out_c, time_dim)
        if not self.same:
            self.res = nn.Conv2d(in_c, out_c, 1)
    
    def forward(self, x, t):
        h = self.conv(x, t)
        return h + x if self.same else h + self.res(x)

class DDPMUNet(nn.Module):
    """DDPM UNet for noise prediction"""
    def __init__(self, in_c=1, out_c=1, time_dim=128, feat_dim=64):
        super().__init__()
        self.time_dim = time_dim
        self.time_emb = nn.Sequential(
            nn.Linear(time_dim, time_dim*2),
            nn.SiLU(),
            nn.Linear(time_dim*2, time_dim)
        )
        self.init = nn.Conv2d(in_c, feat_dim, 3, padding=1)
        
        # Encoder - downsampling
        self.down1_1 = ResidualBlock(feat_dim, feat_dim, time_dim)
        self.down1_2 = ResidualBlock(feat_dim, feat_dim, time_dim)
        self.down1_pool = nn.AvgPool2d(2)
        
        self.down2_1 = ResidualBlock(feat_dim, feat_dim*2, time_dim)
        self.down2_2 = ResidualBlock(feat_dim*2, feat_dim*2, time_dim)
        self.down2_pool = nn.AvgPool2d(2)
        
        self.down3_1 = ResidualBlock(feat_dim*2, feat_dim*4, time_dim)
        self.down3_2 = ResidualBlock(feat_dim*4, feat_dim*4, time_dim)
        
        # Decoder - upsampling with skip connections
        self.up = nn.Upsample(scale_factor=2, mode='nearest')
        self.up2_1 = ResidualBlock(feat_dim*6, feat_dim*2, time_dim)
        self.up2_2 = ResidualBlock(feat_dim*2, feat_dim*2, time_dim)
        
        self.up1_1 = ResidualBlock(feat_dim*3, feat_dim, time_dim)
        self.up1_2 = ResidualBlock(feat_dim, feat_dim, time_dim)
        
        self.final = nn.Sequential(
            nn.Conv2d(feat_dim, feat_dim, 3, padding=1),
            nn.SiLU(),
            nn.Conv2d(feat_dim, out_c, 1)
        )
    
    def forward(self, x, t):
        t_e = self.time_emb(sinusoidal_embedding(t, self.time_dim))
        h = self.init(x)
        
        # Encoder
        h1 = self.down1_1(h, t_e)
        h1 = self.down1_2(h1, t_e)
        skip1 = h1
        h1 = self.down1_pool(h1)
        
        h2 = self.down2_1(h1, t_e)
        h2 = self.down2_2(h2, t_e)
        skip2 = h2
        h2 = self.down2_pool(h2)
        
        h3 = self.down3_1(h2, t_e)
        h3 = self.down3_2(h3, t_e)
        
        # Decoder
        h3 = self.up(h3)
        h3 = torch.cat([h3, skip2], dim=1)
        h3 = self.up2_1(h3, t_e)
        h3 = self.up2_2(h3, t_e)
        
        h3 = self.up(h3)
        h3 = torch.cat([h3, skip1], dim=1)
        h3 = self.up1_1(h3, t_e)
        h3 = self.up1_2(h3, t_e)
        
        return self.final(h3)

# ============================================================================
# Diffusion Utilities
# ============================================================================
def get_linear_schedule(n, start=0.0001, end=0.02):
    """Linear schedule for betas"""
    return torch.linspace(start, end, n)

def precompute_diffusion_constants(betas):
    """Precompute constants for forward and reverse diffusion"""
    alphas = 1 - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0)
    return alphas, alphas_cumprod, torch.sqrt(alphas_cumprod), torch.sqrt(1 - alphas_cumprod)

# ============================================================================
# Generation Function
# ============================================================================
@torch.no_grad()
def generate_samples(model, num_samples, num_steps, img_size=28, batch_size=64):
    """
    Generate samples from noise using reverse diffusion process
    
    Args:
        model: Trained DDPM model
        num_samples: Number of samples to generate
        num_steps: Number of diffusion steps
        img_size: Image resolution (default 28 for MNIST)
        batch_size: Batch size for generation
    
    Returns:
        Generated samples tensor of shape (num_samples, 1, img_size, img_size)
    """
    model.eval()
    
    # Setup betas and precomputed constants
    betas = get_linear_schedule(num_steps).to(DEVICE)
    alphas, alphas_cumprod, sqrt_ac, sqrt_1_ac = precompute_diffusion_constants(betas)
    alphas = alphas.to(DEVICE)
    alphas_cumprod = alphas_cumprod.to(DEVICE)
    sqrt_ac = sqrt_ac.to(DEVICE)
    sqrt_1_ac = sqrt_1_ac.to(DEVICE)
    
    # Precompute posterior variance
    posterior_var = (1 - alphas_cumprod[:-1]) / (1 - alphas_cumprod[1:]) * betas[1:]
    
    generated_samples = []
    
    # Generate in batches
    num_batches = (num_samples + batch_size - 1) // batch_size
    for batch_idx in tqdm(range(num_batches), desc='Generating samples'):
        current_batch_size = min(batch_size, num_samples - batch_idx * batch_size)
        
        # Start from pure noise
        x_t = torch.randn(current_batch_size, 1, img_size, img_size, device=DEVICE)
        
        # Reverse diffusion process
        for t in tqdm(range(num_steps-1, -1, -1), desc=f'Batch {batch_idx+1}/{num_batches}', leave=False):
            # Predict noise
            t_tensor = torch.full((current_batch_size,), t, dtype=torch.long, device=DEVICE)
            pred_noise = model(x_t, t_tensor)
            
            # Reverse diffusion step
            alpha_t = alphas[t]
            alpha_cumprod_t = alphas_cumprod[t]
            
            # Mean calculation
            mean = (1.0 / torch.sqrt(alpha_t)) * (
                x_t - (betas[t] / torch.sqrt(1.0 - alpha_cumprod_t)) * pred_noise
            )
            
            # Add noise for t > 0
            if t > 0:
                z = torch.randn_like(x_t)
                x_t = mean + torch.sqrt(posterior_var[t-1]) * z
            else:
                x_t = mean
        
        # Clip to [-1, 1]
        x_t = torch.clamp(x_t, -1, 1)
        generated_samples.append(x_t.cpu())
    
    return torch.cat(generated_samples, dim=0)

# ============================================================================
# FID Calculation using torchmetrics (feature=64)
# ============================================================================
def calculate_fid_score(real_samples, fake_samples):
    """
    Calculate FID score using torchmetrics with feature=64
    
    Args:
        real_samples: Tensor of real images (N, 1, H, W) in [-1, 1]
        fake_samples: Tensor of fake images (N, 1, H, W) in [-1, 1]
    
    Returns:
        FID score
    """
    from torchmetrics.image.fid import FrechetInceptionDistance
    
    # Initialize FID metric with feature=64
    fid_metric = FrechetInceptionDistance(feature=64, normalize=True).to(DEVICE)
    
    # Process real samples in batches
    print('Computing FID features for real samples...')
    for i in tqdm(range(0, len(real_samples), 64)):
        batch = real_samples[i:i+64].to(DEVICE)
        
        # Convert from [-1, 1] to [0, 1]
        batch = (batch * 0.5) + 0.5
        
        # Expand grayscale to RGB (as InceptionV3 expects RGB)
        batch = batch.repeat(1, 3, 1, 1)
        
        # Resize to 299x299 (required by InceptionV3)
        batch = F.interpolate(batch, size=(299, 299), mode='bilinear', align_corners=False)
        
        # Update metric with real=True
        fid_metric.update(batch, real=True)
    
    # Process fake samples in batches
    print('Computing FID features for generated samples...')
    for i in tqdm(range(0, len(fake_samples), 64)):
        batch = fake_samples[i:i+64].to(DEVICE)
        
        # Convert from [-1, 1] to [0, 1]
        batch = (batch * 0.5) + 0.5
        
        # Expand grayscale to RGB
        batch = batch.repeat(1, 3, 1, 1)
        
        # Resize to 299x299
        batch = F.interpolate(batch, size=(299, 299), mode='bilinear', align_corners=False)
        
        # Update metric with real=False
        fid_metric.update(batch, real=False)
    
    # Compute and return FID score
    fid_score = fid_metric.compute()
    return fid_score.item()

# ============================================================================
# Visualization
# ============================================================================
def visualize_samples(samples, num_display=16, save_path='generated_samples.png'):
    """Visualize and save generated samples"""
    fig, axes = plt.subplots(4, 4, figsize=(10, 10))
    
    for idx, ax in enumerate(axes.flat):
        if idx < len(samples):
            img = samples[idx].squeeze().cpu().numpy()
            # Denormalize from [-1, 1] to [0, 1]
            img = (img + 1) / 2
            img = np.clip(img, 0, 1)
            ax.imshow(img, cmap='gray')
        ax.axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=100, bbox_inches='tight')
    print(f'✓ Samples saved to {save_path}')
    plt.show()

# ============================================================================
# Main Execution
# ============================================================================
if __name__ == '__main__':
    # Configuration
    MODEL_PATH = './trained_model_weights/ddpm_model.pth'
    NUM_GENERATE = 1000  # Generate 1000 samples for FID
    NUM_STEPS = 1000
    IMG_SIZE = 28
    
    # Check model exists
    if not os.path.exists(MODEL_PATH):
        print(f'Error: Model not found at {MODEL_PATH}')
        print('Please train the model first using colab_ddpm_training.py')
        exit(1)
    
    # Load model
    print(f'Loading model from {MODEL_PATH}...')
    model = DDPMUNet().to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    print(f'✓ Model loaded: {sum(p.numel() for p in model.parameters()):,} parameters')
    
    # Generate samples
    print(f'\nGenerating {NUM_GENERATE} samples...')
    generated = generate_samples(model, NUM_GENERATE, NUM_STEPS, img_size=IMG_SIZE)
    print(f'✓ Generated shape: {generated.shape}')
    
    # Visualize some samples
    print('\nVisualizing samples...')
    visualize_samples(generated[:16], save_path='generated_samples.png')
    
    # Load real MNIST data for FID comparison
    print('\nLoading MNIST dataset for FID evaluation...')
    transforms_pipeline = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    dataset = MNIST(root='./mnist_data', train=True, download=True, transform=transforms_pipeline)
    
    # Get real samples (same size as generated)
    real_samples = []
    for idx in range(min(NUM_GENERATE, len(dataset))):
        img, _ = dataset[idx]
        real_samples.append(img.unsqueeze(0))
    real_samples = torch.cat(real_samples, dim=0)
    
    print(f'Real samples shape: {real_samples.shape}')
    print(f'Generated samples shape: {generated.shape}')
    
    # Calculate FID (with feature=64)
    print('\n' + '='*60)
    print('Computing FID Score (feature=64)...')
    print('='*60)
    fid_score = calculate_fid_score(real_samples, generated)
    
    print(f'\n✓ FID Score: {fid_score:.4f}')
    print('  (Lower is better; typically 0-50 for MNIST)')
    
    # Save generated samples
    torch.save(generated, 'generated_samples.pt')
    print(f'✓ Generated samples saved to generated_samples.pt')
