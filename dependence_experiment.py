import math

import torch
from torch import nn, Tensor

from models.ConsistencyUNet2 import ConsistencyUNet


def propogate_zT(z_t: Tensor, online_model: nn.Module, sampling_schedule: list, epsilon: float = 0.002) -> torch.Tensor:
    image_dim = z_t.shape
    # If z_t is not a batch, turn into [B, C, H, W] form
    if len(image_dim) <= 3:
        z_t = z_t.unsqueeze(0)

    # Produce a batch tensor of first timestep in sampling schedule
    t_tensor = sampling_schedule[0]

    # Predict x_hat using initial noise vector and t_tensor
    x_hat = online_model(z_t, t_tensor)

    # Sample thru remaining multiple timesteps
    for t in sampling_schedule[1:]:
        # Sample a noise tensor
        z = torch.randn_like(x_hat)

        # Add the sampled noise to the current x_hat
        z_t = x_hat + (math.sqrt((t ** 2) - (epsilon ** 2)) * z)

        # Broadcast time "t" to shape (batch_size,)
        t_tensor = t.repeat(image_dim[0])

        # Predict x_hat using noised image and t_tensor
        x_hat = online_model(z_t, t_tensor)

    # De-normalize the image
    x_hat = (x_hat * 0.5 + 0.5)

    # Clamp all values in the range [0, 1]
    x_hat = x_hat.clamp(0, 1)

    return x_hat


def make_batched_zT(N: int, experiment_cluster_size: int, image_dim: tuple = (1, 28, 28), device: torch.device = "cuda") -> Tensor:
    """
    Creates N random z_t tensors of shape (image_dim)
    Each z_t is repeated experiment_cluster_size times.
    Final output will be shape [N * experiement_cluster_size x shape(z_t)]
    """
    # Step 1: Create N distinct z_t vectors
    z_t = torch.randn((N, *image_dim), device=device)

    # Step 2: Repeat each z_t sampled experiment_cluster_size times
    z_t = z_t.repeat_interleave(experiment_cluster_size, dim=0)

    return z_t

def dependence_experiment():
    # Experiment constants
    num_z_t = 10
    num_points = 5
    schedule = [80.0, 40.0, 20.0, 10.0, 5.0, 0.002]
    # Step 1: Load CM model
    cm_model = ConsistencyUNet().to(DEVICE)
    cm_model.load_state_dict(torch.load("trained_model_weights/consistency_online_cm2.pth", map_location=DEVICE))

    # Step 2: Generate an experiment grid
    z_t_grid = make_batched_zT(N=num_z_t, experiment_cluster_size=num_points, device=DEVICE)

    # Step 3: Run CM for all z_t in grid, resulting in x_t_grid
    x_t_grid = propogate_zT(z_t=z_t_grid, online_model=cm_model, sampling_schedule=schedule)
    print(z_t_grid)



if __name__ == '__main__':
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dependence_experiment()