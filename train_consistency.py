import torch.nn as nn
from torch.utils.data import DataLoader
import torch
from tqdm import tqdm
import torch.nn.functional as F
from datasets.mnist_dataloader import get_mnist_dataloader

# from models.ConsistencyUNet import ConsistencyUNet
from models.ConsistencyUNet2 import ConsistencyUNet

from typing import Tuple, List
import math
import matplotlib.pyplot as plt
from eval_consistency import sample
from torchmetrics.image.fid import FrechetInceptionDistance


# Determine Device for the whole session
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")


# Initialize the Inception Model for FID calculation
# TODO: Test with 2048 features later
fid_metric = FrechetInceptionDistance(feature = 2048, normalize=True).to(DEVICE)


# NOTE: "t" does not just only denote timestep, but also noise level. Higher "t" means high timestep, but also high noise levels.
# Lower "t" means low timestep, but also low noise levels.


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
        # Using bilinear interpolation
        x = F.interpolate(x, size=(299, 299), mode = 'bilinear')

        # Update metric with real=True
        fid_metric.update(x, real=True)
        
        count += 1
        if count >= num_batches:
            break


def evaluate_fid(model: nn.Module, num_batches = 10, batch_size = 128) -> float:
    """
    Generates fake images and computes FID against the pre-computed real stats.
    """
    model.eval()
    
    # Sampling Schedule (Standard 5-step for eval)
    schedule = torch.tensor([80.0, 40.0, 20.0, 10.0, 5.0], device=DEVICE)

    with torch.no_grad():
        for _ in range(num_batches):
            # 1. Generate Noise
            # z = torch.randn(batch_size, 1, 32, 32).to(DEVICE) * 80.0
            
            # 2. Generate Images
            fake_images = sample(model, schedule, DEVICE, shape = (batch_size, 1, 28, 28)) # Returns [-1, 1] or [0, 1] depending on your sample func
            
            # Process Fake images
            fake_images = fake_images.repeat(1, 3, 1, 1)
            fake_images = F.interpolate(fake_images, size=(299, 299), mode='bilinear')
            
            # 4. Update metric with real=False
            fid_metric.update(fake_images, real=False)

    # Compute final score
    score = fid_metric.compute()
    return score.item()

# ┌───────────────────────────────────────────────┐
# │               TRAINING FUNCTION               │
# └───────────────────────────────────────────────┘
def train(
        online_model: nn.Module,
        ema_model: nn.Module,
        dataloader: DataLoader,
        optimizer: torch.optim.Optimizer,
        num_epochs: int = 10,
        initial_N: int = 2,
        final_N: int = 150
        ) -> Tuple[nn.Module, nn.Module, List]:
    """
    TODO

    Args:
        online_model (nn.Module): _description_
        ema_model (nn.Module): _description_
        dataloader (DataLoader): _description_
        optimizer (torch.optim.Optimizer): _description_
        num_epochs (int, optional): _description_. Defaults to 10.
        initial_N (int, optional): _description_. Defaults to 2.
        final_N (int, optional): _description_. Defaults to 150.

    Returns:
        Tuple[nn.Module, List]: _description_
    """
    # Set the online model in training mode
    online_model.train()

    # Init list to accumulate loss history and FID Scores
    loss_history = []
    fid_scores = []

    # Iterate num_epochs times
    for epoch in tqdm(range(num_epochs)):

        # Curriculum Learning
        # Get N for this epoch (for Karras Schedule Calculations)
        N = get_N_for_karras_time_schedule(epoch, num_epochs, initial_N, final_N)

        # Compute the Karras time schedule
        karras_schedule = get_karras_time_schedule(N).to(DEVICE)

        # Calculate EMA Decay Rate (mu) for this epoch
        # As N grows (intervals get smaller), we want the EMA to update slower (higher mu).
        # Formula: exp(initial_N * log(mu_0) / N)
        mu = math.exp(initial_N * math.log(0.9) / N)        # TODO: Remove hardcoding

        # Track running_loss
        running_loss = 0.0
        steps = 0

        # For each minibatch in the dataloader
        for x, _ in dataloader:
            # Load x on device
            x = x.to(DEVICE)                                            # (batch_size, 1, H, W)
            batch_size = x.shape[0]                                     

            # Sample a Standard Gaussian tensor like x
            z = torch.randn_like(x)                                     # (batch_size, 1, H, W)
            
            # Sample a batch of random time indices in interval [0, N-1)
            t = torch.randint(0, N - 1, (batch_size,), device=DEVICE)   # (batch_size,)

            # Map time indices to sigma values
            # sigma_t2 is higher noise
            # sigma_t1 is lower noise (closer to 0)
            sigmas_t2 = karras_schedule[t]                              # (batch_size,)
            sigmas_t1 = karras_schedule[t + 1]                          # (batch_size,)

            # Online Model Forward Pass (High Noise -> Clean)
            # Add noise to x based on sigmas_t2
            z_t2 = x + z * sigmas_t2.reshape(-1, 1, 1, 1)               # (batch_size, 1, H, W)
            online_output = online_model(z_t2, sigmas_t2)

            # EMA Model Forward Pass (Low Noise -> Clean)
            # Add noise to x based on sigmas_t1
            with torch.no_grad():
                z_t1 = x + z * sigmas_t1.reshape(-1, 1, 1, 1)
                ema_output = ema_model(z_t1, sigmas_t1)

            # Compute Consistency Loss (Online should match EMA)
            loss = F.mse_loss(online_output, ema_output)

            # Calculate Gradients, Perform Backpropagation, and update weights
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Update EMA Model Weights
            with torch.no_grad():
                for online_params, ema_params in zip(online_model.parameters(), ema_model.parameters()):

                    # Mathematical equivalent: ema = mu * ema + (1-mu) * online
                    ema_params.mul_(mu).add_(online_params, alpha = 1 - mu)

            # Accumulate Running Loss
            loss_history.append(loss.item())
            running_loss += loss.item()
            steps += 1
            
        # Epoch Over
        # Calculate Avg Loss
        avg_loss = (running_loss / steps)

        # Precompute FID Metrics for Real Data
        precompute_real_stats(dataloader, num_batches=10)

        # Compute FID Score using Online Model
        fid_score = evaluate_fid(online_model, num_batches=10, batch_size=128)

        fid_scores.append(fid_score)

        # Logging for every epoch
        tqdm.write(f"Epoch {epoch + 1}/{num_epochs}, Avg Loss: {avg_loss:.4f}, FID: {fid_score:.4f}, N: {N}, mu: {mu:.4f}")

        # TODO: Eval Model after every epoch for each N and how it performs
        # TODO: Viz about training performance for both models

    # Return trained models, loss history, and fid scores for each epoch
    return online_model, ema_model, loss_history, fid_scores


# ┌───────────────────────────────────────────────┐
# │               HELPER METHODS                  │
# └───────────────────────────────────────────────┘
def get_N_for_karras_time_schedule(epoch: int, 
                                   num_epochs: int, 
                                   initial_N: int = 2, 
                                   final_N: int = 150) -> int:
    """
    Calculates the discretization steps (N) for the current epoch using the 
    curriculum schedule from the Consistency Models paper.

    Args:
        epoch (int): The current training epoch (usually should start from 1).
        num_epochs (int): The total number of training epochs.
        initial_N (int, optional): The starting value of N (usually 2). Defaults to 2.
        final_N (int, optional): The final value of N (usually 150 for simpler datasets). Defaults to 150.

    Returns:
        int: The number of discretization steps (N) to use for this epoch.
    """    
    target_variance = (epoch / num_epochs) * ((final_N ** 2) - (initial_N ** 2)) + (initial_N ** 2)
    current_N = math.ceil(math.sqrt(target_variance) - 1) + 1
    return current_N


def get_karras_time_schedule(N: int,
                             sigma_min: float = 0.002, 
                             sigma_max: float = 80.0, 
                             rho: float = 7
                             ) -> torch.Tensor:
    """
    Returns a Karras-style Discretized Time Schedule.

    Args:
        N (int): Number of descretization steps between sigma_min and sigma_max.
        sigma_min (float, optional): Represents the minimum noise level, which typically happens at the end of the generation process.
            Defaults to 0.002.
        sigma_max (float, optional): Represents the maximum noise level, which typically happens at the start of the generation process. 
            Defaults to 80.0.
        rho (float, optional): Represents the steepness of the timesteps. Defaults to 7, which typically forces vast majority of the
            steps to cluser near zero noise / sigma_min.

    Returns:
        torch.Tensor: Tensor representing the timesteps according to Karras Schedule.
    """
    # Calculate parts of the Karras Schedule Equation ahead of time for readability
    rho_inv = 1.0 / rho
    min_inv = sigma_min ** rho_inv
    max_inv = sigma_max ** rho_inv
    
    # Create a linear grid from 0 to 1 with N elements
    steps = torch.linspace(0, 1, N)
    
    # Compute the Karras Schedule
    # formula: (sigma_max^(1/rho) + steps * (sigma_min^(1/rho) - sigma_max^(1/rho)))^rho
    karras_schedule = (max_inv + steps * (min_inv - max_inv)) ** rho

    # Return the Karras Schedule
    return karras_schedule

def visualize_loss_trajectory(loss_history: List[float]):
    plt.figure(figsize=(6, 4))
    plt.plot(loss_history, color="blue")
    plt.xlabel("Iterations")
    plt.ylabel("MSE Loss")
    plt.title("Consistency Model: Loss Trajectory")
    plt.grid(True, linestyle="--", alpha=0.3)
    plt.savefig('viz/consistency_loss_trajectory.png')


def visualize_fid_trajectory(fid_scores: List[float]):
    plt.figure(figsize=(6, 4))
    plt.plot(fid_scores, color="blue")
    plt.xlabel("Epochs")
    plt.ylabel("FID")
    plt.title("Consistency Model: FID")
    plt.grid(True, linestyle="--", alpha=0.3)
    plt.savefig('viz/consistency_fid_trajectory.png')


# ┌───────────────────────────────────────────────┐
# │                 DRIVER CODE                   │
# └───────────────────────────────────────────────┘
if __name__ == '__main__':

    # Load MNIST Dataloader
    mnist_dataloader = get_mnist_dataloader()
    
    # Print Dataset Summary
    print(f"{'-' * 10} DATASET INFO {'-' * 10}\n")
    print(f"Dataset length: {len(mnist_dataloader.dataset)}")
    print(f"X (Image) shape: {mnist_dataloader.dataset[0][0].shape}")
    print(f"{'-' * 34}")

    # Init UNets
    online_model = ConsistencyUNet().to(DEVICE)
    ema_model = ConsistencyUNet().to(DEVICE)

    # Load EMA weights from Online
    ema_model.load_state_dict(online_model.state_dict())

    # Init Optimizer for Online Model
    optimizer = torch.optim.AdamW(online_model.parameters(), lr = 1e-4) # TODO: Remove hardcoding of learning rate here

    # Call the training loop
    trained_online_model, trained_ema_model, loss_history, fid_scores = train(online_model,
                                                                              ema_model,
                                                                              mnist_dataloader,
                                                                              optimizer,
                                                                              num_epochs = 50)


    # Visualize the Loss Trajectory
    visualize_loss_trajectory(loss_history)

    # Visualize the FID Trajectory
    visualize_fid_trajectory(fid_scores)

    # Save the EMA Model weights
    torch.save(trained_ema_model.state_dict(), "trained_model_weights/consistency_ema_cm2.pth")

    # Save the Online Model weights
    torch.save(trained_online_model.state_dict(), "trained_model_weights/consistency_online_cm2.pth")