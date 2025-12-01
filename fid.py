from torchmetrics.image.fid import FrechetInceptionDistance
import torch.nn.functional as F
from datasets.mnist_dataloader import get_mnist_dataloader
import torch

# Determine Device for the whole session
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialize the Inception Model for FID calculation
fid = FrechetInceptionDistance(feature = 2048, normalize=True).to(DEVICE)

def compute_fid_score(real_images, fake_images):
    # Reset Metric States
    fid.reset()

    # Add three channels to Real and Fake Images (for RGB based images)
    # Since Inception Model accepts RGB based images
    real_images = real_images.repeat(1, 3, 1, 1)
    fake_images = fake_images.repeat(1, 3, 1, 1)

    # Create a Bilinear interpolation for real and fake images
    # real_images = F.interpolate(real_images, size=(299, 299), mode="bilinear", align_corners=False)
    # fake_images = F.interpolate(fake_images, size=(299, 299), mode="bilinear", align_corners=False)


    fid.update(real_images, real=True)
    fid.update(fake_images, real=False)

    return fid.compute().item()


if __name__ == '__main__':
    mnist_dataloader = get_mnist_dataloader()
    # Get Real Images and Fake Images for comparison (should be 0 FID)
    real_images, _ = next(iter(mnist_dataloader))
    fake_images, _ = next(iter(mnist_dataloader))
 
    # Convert to [0, 1] range
    real_images = (real_images * 0.5 + 0.5).to(DEVICE)
    fake_images = (fake_images * 0.5 + 0.5).to(DEVICE)

    # Compute FID
    fid_score = compute_fid_score(real_images=real_images, fake_images=fake_images)

    # TODO: This value is not coming out to be 0, please check what can be done about this
    print(f"FID Score: {fid_score: .4f}")
