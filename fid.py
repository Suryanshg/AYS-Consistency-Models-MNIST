from torchmetrics.image.fid import FrechetInceptionDistance
import torch.nn.functional as F
from datasets.mnist_dataloader import get_mnist_dataloader
import torch
from torch.utils.data import DataLoader

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
    real_images = F.interpolate(real_images, size=(299, 299), mode="bilinear")
    fake_images = F.interpolate(fake_images, size=(299, 299), mode="bilinear")


    fid.update(real_images, real=True)
    fid.update(fake_images, real=False)

    return fid.compute().item()


def precompute_real_fid_state(dataloader: DataLoader):
    fid.reset()
    for x, _ in dataloader:
        x = (x.to(DEVICE) * 0.5 + 0.5).repeat(1, 3, 1, 1)
        x = F.interpolate(x, (299, 299), mode="bilinear")
        fid.update(x, real=True)
    torch.save(fid.state_dict(), "real_fid_state.pth")
    

if __name__ == '__main__':
    mnist_dataloader = get_mnist_dataloader()

    precompute_real_fid_state(mnist_dataloader)


    # # Get Real Images and Fake Images for comparison (should be 0 FID)
    # real_images, _ = next(iter(mnist_dataloader))
    # fake_images, _ = next(iter(mnist_dataloader))
 
    # # Convert to [0, 1] range
    # real_images = (real_images * 0.5 + 0.5).to(DEVICE)
    # fake_images = (fake_images * 0.5 + 0.5).to(DEVICE)

    # # Compute FID
    # fid_score = compute_fid_score(real_images=real_images, fake_images=fake_images)

    # print(f"FID Score: {fid_score: .4f}")


    # Sanity: identical-set FID (should be ~0)
    # ident_score = compute_fid_score(real_images, real_images)
    # print(f"FID (identical batch): {ident_score:.6f}")

    # More stable estimate: accumulate multiple batches
    # fid.reset()
    # for real_batch, _ in mnist_dataloader:
    #     real_batch = (real_batch * 0.5 + 0.5).to(DEVICE).repeat(1,3,1,1)
    #     fid.update(F.interpolate(real_batch, (299,299), mode="bilinear"), real=True)
    # for fake_batch, _ in mnist_dataloader:
    #     fake_batch = (fake_batch * 0.5 + 0.5).to(DEVICE).repeat(1,3,1,1)
    #     fid.update(F.interpolate(fake_batch, (299,299), mode="bilinear"), real=False)
    # full_score = fid.compute().item()
    # print(f"FID (full dataset vs itself, expected near 0): {full_score:.6f}")
