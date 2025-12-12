import math

import numpy as np
import torch
from torch import Tensor, nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchmetrics.image import FrechetInceptionDistance

from models.ConsistencyUNet3 import ConsistencyUNet


class ConsistencyModel(nn.Module):

    def __init__(self):
        super(ConsistencyModel, self).__init__()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = ConsistencyUNet()
        self.fid_metric = FrechetInceptionDistance(feature = 64, normalize=True).to(self.device)

        # AYS Stuff
        self.velocities = None
        self.sigmas = None

    def initialize_FID(self, dataloader: DataLoader, num_real_batches=64):
        print("Generating FID from real...")
        self.fid_metric.reset()

        count = 0
        for x, _ in dataloader:
            x = x.to(self.device)

            # Convert range from [-1, 1] to [0, 1]
            x = (x * 0.5) + 0.5

            # Repeat Grayscale to look like RGB Images (as InceptionV3 expects RGB plots)
            x = x.repeat(1, 3, 1, 1)

            # Resize to 299x299 (Required by InceptionV3)
            # Using bilinear interpolation
            x = F.interpolate(x, size=(299, 299), mode='bilinear')

            # Update metric with real=True
            self.fid_metric.update(x, real=True)

            count += 1
            if count >= num_real_batches:
                break

    def forward(self, z_t, t_tensor):
        return self.model(z_t, t_tensor)

    def sample(self, n_samples: int, schedule: list, image_dim=(1, 28, 28)) -> Tensor:
        z_t_batch = torch.randn((n_samples, *image_dim), device=self.device)
        x_t_batch = self.propagate_zT(z_t_batch, schedule)
        return x_t_batch

    def propagate_zT(self, z_t: Tensor, sampling_schedule: list, epsilon: float = 0.002, deterministic=False) -> torch.Tensor:
        image_dim = z_t.shape
        assert len(image_dim) > 3 # Must be in [B, C, H, W] form

        sampling_schedule = torch.tensor(sampling_schedule, device=self.device)

        # First step (no noise adding)
        z_t = z_t * sampling_schedule[0]
        t_tensor = sampling_schedule[0].repeat(image_dim[0])  # (N,)
        x_hat = self.model(z_t, t_tensor)

        # Remaining Steps
        for t in sampling_schedule[1:]:
            if not deterministic:
                z_t = torch.randn_like(x_hat)
            z_t = x_hat + (math.sqrt((t ** 2) - (epsilon ** 2)) * z_t)

            t_tensor = t.repeat(image_dim[0])
            x_hat = self.model(z_t, t_tensor)

        # Post-process images generated
        x_hat = (x_hat * 0.5 + 0.5)
        x_hat = x_hat.clamp(0, 1)

        return x_hat

    def evaluate_fid(self, schedule, num_batches=10, batch_size=128) -> float:
        assert self.fid_metric is not None

        self.model.eval()
        with torch.no_grad():
            for _ in range(num_batches):
                # 1. Sample Images
                fake_images = self.sample(n_samples=batch_size, schedule=schedule)
                # 2. Process Fake Images
                f_i = fake_images.repeat(1, 3, 1, 1)
                f_i = F.interpolate(f_i, size=(299, 299), mode='bilinear')

                # 3. Update metric with real=False
                self.fid_metric.update(f_i, real=False)

        # Compute final score
        score = self.fid_metric.compute()
        return score.item()

    def load(self, path: str):
        print(f"Loading CM Weights from: {path}")
        self.model.load_state_dict(torch.load(f"weights/{path}", map_location=self.device))
        self.model.to(self.device)

    def save(self, path: str):
        self.model.state_dict()

    # ----------------------------------------------- AYS Integration --------------------------------------------------
    def _init_prediction_velocities(self, num_points=100, sigma_max=80.0, sigma_min=0.002):
        """
        Scans the model from sigma_max down to sigma_min.
        Measures how much the predicted x_0 changes (velocity) at each level.
        """
        print(f"Initializing velocity calculations for current model to perform AYS...")
        # Create a list of sigmas in log-space (High Noise (80.0) -> Low Noise (0.002))
        sigmas = np.exp(np.linspace(np.log(sigma_max), np.log(sigma_min), num_points))

        # Create a fixed Gaussian noise vector to probe consistency model
        # We use a batch of 10 to average out random fluctuations
        # This is important for simulating the PF ODE Trajectory
        fixed_noise = torch.randn(10, 1, 28, 28).to(self.device)  # (10, 1, 28, 28)

        # Init a List to store velocities
        velocities = []

        self.model.eval()
        with torch.no_grad():
            for i in range(len(sigmas) - 1):
                # Extract Current Sigma and Next Sigma values (Consecutive at index i)
                sigma_curr = sigmas[i]  # High Noise Signal
                sigma_next = sigmas[i + 1]  # Low Noise Signal

                # Create inputs for current and next sigma
                # Note: We scale the same fixed noise by the sigma
                z_curr = fixed_noise * sigma_curr  # (10, 1, 28, 28)
                z_next = fixed_noise * sigma_next  # (10, 1, 28, 28)

                t_curr = torch.full((10,), sigma_curr, device=self.device)  # (10, )
                t_next = torch.full((10,), sigma_next, device=self.device)  # (10, )

                # Get predictions
                pred_curr = self.model(z_curr, t_curr)  # (10, 1, 28, 28)
                pred_next = self.model(z_next, t_next)  # (10, 1 ,28, 28)

                # Calculate "Velocity" = change between predictions
                # If the model is perfectly consistent, this should be 0.
                # Large values mean the model is "changing its mind" --> High Curvature.

                # Calculate the raw Difference in predictions
                diff = pred_curr - pred_next

                # Calculate RMSE per image
                # Shape: (Batch_Size,)
                rmse_per_image = diff.pow(2).mean(dim=(1, 2, 3)).sqrt()

                # Calculate velocity per image (Change per unit of sigma)
                # Velocity = Displacement / Time
                # velocity_per_image = rmse_per_image / delta_sigma

                # Average over the batch
                # velocity = velocity_per_image.mean().item()
                velocity = rmse_per_image.mean().item()

                # Acculumulate Velocities
                velocities.append(velocity)

        self.velocities = np.array(velocities)
        self.sigmas = sigmas[1:]
        # Return Velocities and the sigma values
        # return np.array(velocities), sigmas[:-1]
        return self.velocities, self.sigmas

    def get_ays_schedule(self, num_steps=5):
        """
        Uses the velocity (curvature) profile to pick optimal steps.
        We treat velocity as a Probability Density Function (PDF) and sample from it.
        """
        if self.velocities is None:
            self._init_prediction_velocities()

        # Normalize velocities to create a PDF
        pdf = self.velocities / np.sum(self.velocities)

        # Compute Cumulative Distribution Function (CDF)
        cdf = np.cumsum(pdf)

        # Sample 'num_steps' points evenly from the CDF (0 to 1)
        # We want to find sigmas that correspond to cumulative probability 0.2, 0.4, 0.6...
        # target_probs = np.linspace(0, 1, num_steps + 1)
        target_probs = np.linspace(0, 1, num_steps)

        # optimal_sigmas = []

        # # Always include sigma_max
        # optimal_sigmas.append(sigmas[0])

        # # Find the intermediate steps
        # for target in target_probs[1:-1]: # Skip 0 and 1
        #     # Find index where CDF crosses the target
        #     idx = np.searchsorted(cdf, target)
        #     optimal_sigmas.append(sigmas[idx])

        # # Always include sigma_min (epsilon)
        # optimal_sigmas.append(sigmas[-1])

        # Use Linear Interpolation (Inverse CDF Sampling)
        # We ask: "At what exact sigma is the CDF = 0.2?"
        optimal_sigmas = np.interp(target_probs, cdf, self.sigmas)

        # Fix Boundaries
        # Interpolation might slightly miss the exact start/end due to floating point math
        optimal_sigmas[0] = self.sigmas[0]  # Force start to be exactly sigma_max
        optimal_sigmas[-1] = self.sigmas[-1]  # Force end to be exactly sigma_min

        return np.array(optimal_sigmas)