import torch.nn as nn
from torch.utils.data import DataLoader
import torch
from tqdm import tqdm
import torch.nn.functional as F
from datasets.mnist_dataloader import get_mnist_dataloader
from models.u_net import UNet
from typing import Tuple, List

# ┌───────────────────────────────────────────────┐
# │               TRAINING FUNCTION               │
# └───────────────────────────────────────────────┘
def train(
        online_model: nn.Module,
        ema_model: nn.Module,
        dataloader: DataLoader,
        optimizer: torch.optim.Optimizer,
        num_epochs: int = 10
        ) -> Tuple[nn.Module, List]:
    """
    TODO:

    Args:
        online_model (nn.Module): _description_
        ema_model (nn.Module): _description_
        dataloader (DataLoader): _description_
        optimizer (torch.optim.Optimizer): _description_
        num_epochs (int, optional): _description_. Defaults to 10.

    Returns:
        Tuple[nn.Module, List]: _description_
    """
    # Set the online model in training mode
    online_model.train()

    # Init list to accumulate loss history
    loss_history = []

    # Iterate num_epochs times
    for epoch in range(num_epochs):

        # TODO: check what does these do and what is their significance with respect to the paper
        # N = math.ceil(math.sqrt((epoch * (150**2 - 4) / num_epochs) + 4) - 1) + 1
        # boundaries = kerras_boundaries(7.0, 0.002, N, 80.0).to(device)

        # Or can we keep N as a constant
        N = 1000

        # Variable to track loss computed used EMA
        ema_loss = None

        # For each minibatch in the dataloader
        for x, _ in tqdm(dataloader):
            # Load x on device
            x = x.to(DEVICE)

            # Sample a Standard Gaussian tensor like x
            z = torch.randn_like(x)
            
            # Sample a random timestep
            t = torch.randint(0, N - 1, (x.shape[0], 1), device=DEVICE)

            # Get t_0 and t_1 from Kerras Boundaries
            # t_0 = boundaries[t]
            # t_1 = boundaries[t + 1]

            # Calculate Online Loss
            # online_loss = calculate_loss(online_model, ema_model, x, z, t_0, t_1)

            # Calculate Gradients, Perform Backpropagation, and update weights
            optimizer.zero_grad()
            # online_loss.backward()
            optimizer.step()

            # Determine EMA Loss
            # if ema_loss is None:
            #     ema_loss = online_loss.item()
            # else:
            #     ema_loss = (0.9 * ema_loss) + (0.1 * online_loss.item())

            # Update weights of EMA Model
            # with torch.no_grad():
            #     mu = math.exp(2 * math.log(0.95) / N)
            #     for online_params, ema_params in zip(online_model.parameters(), ema_model.parameters()):
            #         ema_params.mul_(mu).add_(online_params, alpha=1 - mu) # TODO: Can this be written simply than this?

        # TODO: Add Logging for every epoch
        # TODO: Accumulate loss for trajectory viz


    # Return trained online model and loss history
    return online_model, loss_history


# ┌───────────────────────────────────────────────┐
# │               HELPER METHODS                  │
# └───────────────────────────────────────────────┘
# TODO: Understand what it does
def calculate_loss(online_model: nn.Module,
                   ema_model: nn.Module,
                   x: torch.Tensor, 
                   z: torch.Tensor, 
                   t1: torch.Tensor, 
                   t2: torch.Tensor) -> torch.Tensor:
    """
    TODO:

    Args:
        online_model (nn.Module): _description_
        ema_model (nn.Module): _description_
        x (torch.Tensor): _description_
        z (torch.Tensor): _description_
        t1 (torch.Tensor): _description_
        t2 (torch.Tensor): _description_

    Returns:
        torch.Tensor: _description_
    """
    x2 = x + z * t2[:, :, None, None]
    x2 = online_model(x2, t2)

    with torch.no_grad():
        x1 = x + z * t1[:, :, None, None]
        x1 = ema_model(x1, t1)

    return F.mse_loss(x1, x2)



# ┌───────────────────────────────────────────────┐
# │                 DRIVER CODE                   │
# └───────────────────────────────────────────────┘
if __name__ == '__main__':

    # Determine Device for the whole session
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {DEVICE}")

    # Load MNIST Dataloader
    mnist_dataloader = get_mnist_dataloader()
    
    # Print Dataset Summary
    print(f"{'-' * 10} DATASET INFO {'-' * 10}\n")
    print(f"Dataset length: {len(mnist_dataloader.dataset)}")
    print(f"X (Image) shape: {mnist_dataloader.dataset[0][0].shape}")
    print(f"{'-' * 34}")

    # Init Online Model
    online_model = UNet().to(DEVICE)

    # Init EMA Model and do load_state_dict from online model
    ema_model = UNet().to(DEVICE)
    ema_model.load_state_dict(online_model.state_dict())

    # Init Optimizer for Online Model
    optimizer = torch.optim.Adam(online_model.parameters(), lr = 1e-4) # TODO: Remove hardcoding of learning rate here

    # TODO: Call the training loop
