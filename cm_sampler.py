import math

import torch
from torch import Tensor, nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchmetrics.image import FrechetInceptionDistance

from models.ConsistencyUNet2 import ConsistencyUNet


class ConsistencyModel(nn.Module):

    def __init__(self):
        super(ConsistencyModel, self).__init__()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = ConsistencyUNet()
        self.fid_metric = FrechetInceptionDistance(feature = 64, normalize=True).to(self.device)

    def initialize_FID(self, dataloader: DataLoader, num_batches=10):
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
            if count >= num_batches:
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
        x_hat = None

        for i,t in enumerate(sampling_schedule):
            # Broadcast time "t" to shape (batch_size,)
            if i > 0:
                t_tensor = t.repeat(image_dim[0])
            else:
                t_tensor = t

            # Predict x_hat using noised image and t_tensor
            x_hat = self.model(z_t, t_tensor)

            # Sample a new noise tensor to convert ODE to SDE for better generation diversity
            if not deterministic:
                z_t = torch.randn_like(x_hat)

            # Add the sampled noise to the current x_hat
            z_t = x_hat + (math.sqrt((t ** 2) - (epsilon ** 2)) * z_t)

        # Post-process images generated
        x_hat = (x_hat * 0.5 + 0.5)
        x_hat = x_hat.clamp(0, 1)

        return x_hat

    def evaluate_fid(self, schedule, num_batches=10, batch_size=128) -> float:
        assert self.fid_metric

        self.model.eval()
        with torch.no_grad():
            for _ in range(num_batches):
                # 1. Generate Noise
                z = torch.randn(batch_size, 1, 32, 32).to(self.device)

                # 2. Generate Images
                fake_images = self.propagate_zT(z, schedule, deterministic=True)

                # 3. Process Fake Images
                fake_images = fake_images.repeat(1, 3, 1, 1)
                fake_images = F.interpolate(fake_images, size=(299, 299), mode='bilinear')

                # 4. Update metric with real=False
                self.fid_metric.update(fake_images, real=False)

        # Compute final score
        score = self.fid_metric.compute()
        return score.item()

    def load(self, path: str):
        print(f"Loading CM Weights from: {path}")
        self.model.load_state_dict(torch.load(f"weights/{path}", map_location=self.device))
        self.model.to(self.device)

    def save(self, path: str):
        self.model.state_dict()