import math
import torch


def get_karras_N(epoch: int, num_epochs: int, initial_N: int = 2, final_N: int = 150) -> int:
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


def get_karras_time_schedule(N: int, sigma_min: float = 0.002, sigma_max: float = 80.0, rho: float = 7) -> torch.Tensor:
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