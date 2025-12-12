import math
from typing import Tuple, List

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from cm_sampler import ConsistencyModel
from karras import get_karras_N, get_karras_time_schedule


def train_consistency(dataloader: DataLoader, resume_checkpoint=None, lr=1e-4, num_epochs: int = 10, initial_N: int = 2, final_N: int = 15) -> Tuple[nn.Module, nn.Module, List, List]:
    """
    Args:
        dataloader (DataLoader): _description_
        resume_checkpoint (str, optional): _description_. Defaults to None.
        lr (float, optional): _description_. Defaults to 1e-4.
        num_epochs (int, optional): _description_. Defaults to 10.
        initial_N (int, optional): _description_. Defaults to 2.
        final_N (int, optional): _description_. Defaults to 150.

    Returns:
        Tuple[nn.Module, List]: _description_
    """
    # Initialize Online and ema Models + Optimizer
    online_model = ConsistencyModel()
    ema_model = ConsistencyModel()

    if resume_checkpoint is not None:
        online_model.load(resume_checkpoint)

    ema_model.model.load_state_dict(online_model.model.state_dict())
    optimizer = torch.optim.AdamW(online_model.parameters(), lr=lr)
    online_model.initialize_FID(dataloader)

    # Set the online model in training mode
    online_model.train()

    # Init list to accumulate loss history and FID Scores
    loss_history = []
    fid_scores = []

    # Iterate num_epochs times
    for epoch in tqdm(range(num_epochs)):

        # Curriculum Learning
        # Get N for this epoch (for Karras Schedule Calculations)
        N = get_karras_N(epoch, num_epochs, initial_N, final_N)

        # Compute the Karras time schedule
        karras_schedule = get_karras_time_schedule(N).to(online_model.device)

        # Calculate EMA Decay Rate (mu) for this epoch
        # As N grows (intervals get smaller), we want the EMA to update slower (higher mu).
        # Formula: exp(initial_N * log(mu_0) / N)
        mu = math.exp(initial_N * math.log(0.9) / N)

        # Track running_loss
        running_loss = 0.0
        steps = 0

        # For each minibatch in the dataloader
        for x, _ in dataloader:
            # Load x on device
            x = x.to(online_model.device)  # (batch_size, 1, H, W)
            batch_size = x.shape[0]

            # Sample a Standard Gaussian tensor like x
            z = torch.randn_like(x)  # (batch_size, 1, H, W)

            # Sample a batch of random time indices in interval [0, N-1]
            t = torch.randint(0, N - 1, (batch_size,), device=online_model.device)  # (batch_size,)

            # Map time indices to sigma values
            # sigma_t2 is higher noise
            # sigma_t1 is lower noise (closer to 0)
            sigmas_t2 = karras_schedule[t]  # (batch_size,)
            sigmas_t1 = karras_schedule[t + 1]  # (batch_size,)

            # Online Model Forward Pass (High Noise -> Clean)
            # Add noise to x based on sigmas_t2
            z_t2 = x + z * sigmas_t2.reshape(-1, 1, 1, 1)  # (batch_size, 1, H, W)
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
                    ema_params.mul_(mu).add_(online_params, alpha=1 - mu)

            # Accumulate Running Loss
            running_loss += loss.item()
            steps += 1

        # Calculate Avg Loss
        avg_loss = (running_loss / steps)
        loss_history.append(avg_loss)

        # FID Calculation
        fid_score = online_model.evaluate_fid(karras_schedule)
        fid_scores.append(fid_score)

        # Logging
        tqdm.write(f"Epoch {epoch + 1}/{num_epochs}, Avg Loss: {avg_loss:.4f}, FID: {fid_score:.4f}, N: {N}, mu: {mu:.4f}")

        # Save the model weights
        online_model.save(f"online_cm_epoch{epoch+1}")
        ema_model.save(f"ema_model_epoch{epoch+1}")

    # Return trained models, loss history, and fid scores for each epoch
    return online_model, ema_model, loss_history, fid_scores