import torch.nn as nn
from torch.utils.data import DataLoader
import torch
from tqdm import tqdm
import torch.nn.functional as F
from datasets.mnist_dataloader import get_mnist_dataloader

from models.ConsistencyUNet2 import ConsistencyUNet

from typing import Tuple, List
import math
import matplotlib.pyplot as plt
from torchmetrics.image.fid import FrechetInceptionDistance


# Determine Device for the whole session
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

# Initialize the Inception Model for FID calculation
fid_metric = FrechetInceptionDistance(feature=64, normalize=True).to(DEVICE)


# ┌───────────────────────────────────────────────┐
# │              DIFFUSION SCHEDULE               │
# └───────────────────────────────────────────────┘
def get_linear_schedule(num_steps: int,
                        beta_start: float = 0.0001,
                        beta_end: float = 0.02) -> torch.Tensor:
    """
    Returns a linearly spaced schedule of betas for the forward diffusion process.
    
    Args:
        num_steps (int): Number of diffusion steps
        beta_start (float): Beta value at the start
        beta_end (float): Beta value at the end
        
    Returns:
        torch.Tensor: Tensor of shape (num_steps,) containing beta values
    """
    return torch.linspace(beta_start, beta_end, num_steps)


def get_cosine_schedule(num_steps: int,
                        s: float = 0.008) -> torch.Tensor:
    """
    Returns a cosine-annealed schedule of betas for the forward diffusion process.
    Based on Improved Denoising Diffusion Probabilistic Models (IDPD).
    
    Args:
        num_steps (int): Number of diffusion steps
        s (float): Small offset for numerical stability
        
    Returns:
        torch.Tensor: Tensor of shape (num_steps,) containing beta values
    """
    steps = torch.arange(num_steps + 1)
    alphas_cumprod = torch.cos(((steps / num_steps) + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0.0001, 0.9999)


def precompute_diffusion_constants(betas: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Precomputes alpha values and derived constants for the diffusion process.
    
    Args:
        betas (torch.Tensor): Schedule of betas
        
    Returns:
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]: alphas, alphas_cumprod, sqrt_alphas_cumprod_t, sqrt_one_minus_alphas_cumprod_t
    """
    alphas = 1 - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0)
    sqrt_alphas_cumprod_t = torch.sqrt(alphas_cumprod)
    sqrt_one_minus_alphas_cumprod_t = torch.sqrt(1 - alphas_cumprod)
    
    return alphas, alphas_cumprod, sqrt_alphas_cumprod_t, sqrt_one_minus_alphas_cumprod_t


# ┌───────────────────────────────────────────────┐
# │            EVALUATION FUNCTIONS               │
# └───────────────────────────────────────────────┘
def precompute_real_stats(dataloader: DataLoader, num_batches: int = 10):
    """
    Feeds real images into the FID metric to update the 'real' statistics.
    """
    fid_metric.reset()
    
    count = 0
    for x, _ in dataloader:
        x = x.to(DEVICE)
        
        # Convert range from [-1, 1] to [0, 1]
        x = (x * 0.5) + 0.5

        # Repeat Grayscale to look like RGB Images (as InceptionV3 expects RGB images)
        x = x.repeat(1, 3, 1, 1)

        # Resize to 299x299 (Required by InceptionV3)
        x = F.interpolate(x, size=(299, 299), mode='bilinear')

        # Update metric with real=True
        fid_metric.update(x, real=True)
        
        count += 1
        if count >= num_batches:
            break


def evaluate_fid(model: nn.Module,
                 betas: torch.Tensor,
                 alphas: torch.Tensor,
                 alphas_cumprod: torch.Tensor,
                 timesteps: List[int],
                 num_batches: int = 10,
                 batch_size: int = 128) -> float:
    """
    Generates fake images using DDPM reverse process and computes FID.
    
    Args:
        model (nn.Module): The trained denoising network
        betas (torch.Tensor): Beta schedule
        alphas (torch.Tensor): Alpha values
        alphas_cumprod (torch.Tensor): Cumulative product of alphas
        timesteps (List[int]): Timesteps to use for reverse process
        num_batches (int): Number of batches to generate
        batch_size (int): Batch size for generation
        
    Returns:
        float: FID score
    """
    model.eval()

    with torch.no_grad():
        for _ in range(num_batches):
            # Start with pure noise
            x_t = torch.randn(batch_size, 1, 28, 28).to(DEVICE)
            
            # Reverse diffusion process
            for t_idx in reversed(timesteps):
                t = torch.full((batch_size,), t_idx, dtype=torch.long, device=DEVICE)
                
                # Predict noise
                predicted_noise = model(x_t, t.float())
                
                # Get diffusion parameters for this timestep
                alpha_t = alphas[t_idx]
                alpha_cumprod_t = alphas_cumprod[t_idx]
                beta_t = betas[t_idx]
                
                if t_idx > 0:
                    alpha_cumprod_prev = alphas_cumprod[t_idx - 1]
                else:
                    alpha_cumprod_prev = torch.tensor(1.0, device=DEVICE)
                
                # Predict x_0 from x_t
                pred_x0 = (x_t - torch.sqrt(1 - alpha_cumprod_t) * predicted_noise) / torch.sqrt(alpha_cumprod_t)
                pred_x0 = torch.clamp(pred_x0, -1, 1)
                
                # Compute mean of reverse distribution
                coef1 = torch.sqrt(alpha_cumprod_prev) * beta_t / (1 - alpha_cumprod_t)
                coef2 = torch.sqrt(alpha_t) * (1 - alpha_cumprod_prev) / (1 - alpha_cumprod_t)
                mean = coef1 * pred_x0 + coef2 * x_t
                
                # Add noise
                if t_idx > 0:
                    variance = (1 - alpha_cumprod_prev) / (1 - alpha_cumprod_t) * beta_t
                    noise = torch.randn_like(x_t)
                    x_t = mean + torch.sqrt(variance) * noise
                else:
                    x_t = mean
            
            # Process Fake images
            fake_images = (x_t * 0.5 + 0.5).clamp(0, 1)
            fake_images = fake_images.repeat(1, 3, 1, 1)
            fake_images = F.interpolate(fake_images, size=(299, 299), mode='bilinear')
            
            # Update metric with real=False
            fid_metric.update(fake_images, real=False)

    # Compute final score
    score = fid_metric.compute()
    return score.item()


# ┌───────────────────────────────────────────────┐
# │               TRAINING FUNCTION               │
# └───────────────────────────────────────────────┘
def train(
        model: nn.Module,
        dataloader: DataLoader,
        optimizer: torch.optim.Optimizer,
        num_epochs: int = 10,
        num_diffusion_steps: int = 1000,
        schedule_type: str = "linear"
        ) -> Tuple[nn.Module, List, List]:
    """
    Training loop for standard DDPM diffusion model.

    Args:
        model (nn.Module): The denoising network
        dataloader (DataLoader): Training dataloader
        optimizer (torch.optim.Optimizer): Optimizer for training
        num_epochs (int): Number of training epochs
        num_diffusion_steps (int): Number of diffusion steps
        schedule_type (str): Type of noise schedule ("linear" or "cosine")

    Returns:
        Tuple[nn.Module, List, List]: Trained model, loss history, and FID scores
    """
    
    # Set the model in training mode
    model.train()

    # Get noise schedule
    if schedule_type == "linear":
        betas = get_linear_schedule(num_diffusion_steps)
    else:
        betas = get_cosine_schedule(num_diffusion_steps)
    
    betas = betas.to(DEVICE)
    
    # Precompute diffusion constants
    alphas, alphas_cumprod, sqrt_alphas_cumprod_t, sqrt_one_minus_alphas_cumprod_t = \
        precompute_diffusion_constants(betas)
    
    alphas = alphas.to(DEVICE)
    alphas_cumprod = alphas_cumprod.to(DEVICE)
    sqrt_alphas_cumprod_t = sqrt_alphas_cumprod_t.to(DEVICE)
    sqrt_one_minus_alphas_cumprod_t = sqrt_one_minus_alphas_cumprod_t.to(DEVICE)

    # Init lists to accumulate loss history and FID Scores
    loss_history = []
    fid_scores = []
    
    # Evaluation timesteps (subset for faster FID computation)
    eval_timesteps = list(range(0, num_diffusion_steps, num_diffusion_steps // 50))

    # Iterate num_epochs times
    for epoch in tqdm(range(num_epochs)):

        # Track running_loss
        running_loss = 0.0
        steps = 0

        # For each minibatch in the dataloader
        for x, _ in dataloader:
            # Load x on device
            x = x.to(DEVICE)                                    # (batch_size, 1, H, W)
            batch_size = x.shape[0]

            # Check for NaN in input
            if torch.isnan(x).any():
                print("WARNING: NaN detected in input data!")
                continue

            # Sample random timesteps for each sample in the batch
            t = torch.randint(0, num_diffusion_steps, (batch_size,), device=DEVICE)  # (batch_size,)

            # Sample Gaussian noise
            eps = torch.randn_like(x)                          # (batch_size, 1, H, W)

            # Get coefficients for the forward process
            sqrt_alpha_t = sqrt_alphas_cumprod_t[t].reshape(-1, 1, 1, 1)
            sqrt_one_minus_alpha_t = sqrt_one_minus_alphas_cumprod_t[t].reshape(-1, 1, 1, 1)

            # Forward diffusion process: x_t = sqrt(alpha_bar_t) * x_0 + sqrt(1 - alpha_bar_t) * eps
            x_t = sqrt_alpha_t * x + sqrt_one_minus_alpha_t * eps

            # Check for NaN after forward diffusion
            if torch.isnan(x_t).any():
                print(f"WARNING: NaN detected in x_t at timesteps {t}")
                continue

            # Predict noise using model
            predicted_noise = model(x_t, t.float())

            # Check for NaN in predictions
            if torch.isnan(predicted_noise).any():
                print("WARNING: NaN detected in model predictions!")
                continue

            # Compute MSE loss between predicted and actual noise
            loss = F.mse_loss(predicted_noise, eps)

            # Check for NaN in loss
            if torch.isnan(loss):
                print("WARNING: NaN loss detected! Skipping batch.")
                continue

            # Backward pass and optimizer step
            optimizer.zero_grad()
            loss.backward()
            
            # Check for NaN gradients
            has_nan_grad = False
            for name, param in model.named_parameters():
                if param.grad is not None and torch.isnan(param.grad).any():
                    print(f"WARNING: NaN gradient in {name}")
                    has_nan_grad = True
                    break
            
            if has_nan_grad:
                continue
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)  # Increased from 1.0
            optimizer.step()

            # Accumulate Running Loss
            loss_history.append(loss.item())
            running_loss += loss.item()
            steps += 1

        # Epoch Over
        # Calculate Avg Loss
        if steps > 0:
            avg_loss = running_loss / steps
        else:
            avg_loss = float('nan')

        # Precompute FID Metrics for Real Data
        precompute_real_stats(dataloader, num_batches=10)

        # Compute FID Score
        fid_score = evaluate_fid(model,
                                 betas,
                                 alphas,
                                 alphas_cumprod,
                                 eval_timesteps,
                                 num_batches=10,
                                 batch_size=128)

        fid_scores.append(fid_score)

        # Logging for every epoch
        tqdm.write(f"Epoch {epoch + 1}/{num_epochs}, Avg Loss: {avg_loss:.4f}, FID: {fid_score:.4f}")

    # Return trained model, loss history, and fid scores for each epoch
    return model, loss_history, fid_scores


# ┌───────────────────────────────────────────────┐
# │            VISUALIZATION FUNCTIONS            │
# └───────────────────────────────────────────────┘
def visualize_loss_trajectory(loss_history: List[float]):
    """Visualize training loss trajectory"""
    plt.figure(figsize=(6, 4))
    plt.plot(loss_history, color="blue")
    plt.xlabel("Iterations")
    plt.ylabel("MSE Loss")
    plt.title("DDPM: Loss Trajectory")
    plt.grid(True, linestyle="--", alpha=0.3)
    plt.savefig('viz/ddpm_loss_trajectory.png')
    plt.close()


def visualize_fid_trajectory(fid_scores: List[float]):
    """Visualize FID trajectory across epochs"""
    plt.figure(figsize=(6, 4))
    plt.plot(fid_scores, color="blue")
    plt.xlabel("Epochs")
    plt.ylabel("FID")
    plt.title("DDPM: FID Trajectory")
    plt.grid(True, linestyle="--", alpha=0.3)
    plt.savefig('viz/ddpm_fid_trajectory.png')
    plt.close()


# ┌───────────────────────────────────────────────┐
# │                 DRIVER CODE                   │
# └───────────────────────────────────────────────┘
if __name__ == '__main__':

    # Load MNIST Dataloader
    mnist_dataloader = get_mnist_dataloader(batch_size=128)
    
    # Print Dataset Summary
    print(f"{'-' * 10} DATASET INFO {'-' * 10}\n")
    print(f"Dataset length: {len(mnist_dataloader.dataset)}")
    print(f"X (Image) shape: {mnist_dataloader.dataset[0][0].shape}")
    print(f"{'-' * 34}")

    # Init Model
    model = ConsistencyUNet().to(DEVICE)

    # Init Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    # Call the training loop
    trained_model, loss_history, fid_scores = train(
        model,
        mnist_dataloader,
        optimizer,
        num_epochs=10,
        num_diffusion_steps=1000,
        schedule_type="linear"
    )

    # Visualize the Loss Trajectory
    visualize_loss_trajectory(loss_history)

    # Visualize the FID Trajectory
    visualize_fid_trajectory(fid_scores)

    # Save the Model weights
    torch.save(trained_model.state_dict(), "trained_model_weights/ddpm_model.pth")

    print("Training complete! Model saved to trained_model_weights/ddpm_model.pth")