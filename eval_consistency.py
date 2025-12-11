import torch
import torch.nn as nn
import math

# from models.ConsistencyUNet import ConsistencyUNet
from models.ConsistencyUNet2 import ConsistencyUNet
# from models.ConsistencyUNet3 import ConsistencyUNet

import matplotlib.pyplot as plt
from datasets.mnist_dataloader import get_mnist_dataloader
from torchmetrics.image.fid import FrechetInceptionDistance
import torch.nn.functional as F

def sample(
        online_model: nn.Module,
        sampling_schedule: torch.Tensor,
        device, # TODO: Add a Data Type here
        epsilon: float = 0.002,
        shape: tuple = (25, 1, 28, 28)
        ) -> torch.Tensor:

    # Sample a noise vector with initial noise
    z_T = torch.randn(shape).to(device) * sampling_schedule[0]

    # Produce a batch tensor of first timestep in sampling schedule
    t_tensor = sampling_schedule[0].repeat(shape[0])                  # (N,)

    # Predict x_hat using initial noise vector and t_tensor
    x_hat = online_model(z_T, t_tensor)

    # Sample thru remaining multiple timesteps
    for t in sampling_schedule[1:]:

        # Sample a noise tensor
        z = torch.randn_like(x_hat)

        # Add the sampled noise to the current x_hat
        z_t = x_hat + (math.sqrt((t ** 2) - (epsilon ** 2)) * z)

        # Broadcast time "t" to shape (batch_size,)
        t_tensor = t.repeat(shape[0])

        # Predict x_hat using noised image and t_tensor
        x_hat = online_model(z_t, t_tensor)
    
    # De-normalize the image
    x_hat = (x_hat * 0.5 + 0.5)

    # TODO: Check if this is needed at all
    # Clamp all values in the range [0, 1]
    x_hat = x_hat.clamp(0, 1) 

    return x_hat


def calculate_test_fid(model, dataloader, sampling_schedule):
    print("Calculating Test FID on 10k Images...")
    
    # Initialize High-Res Metric
    fid_2048 = FrechetInceptionDistance(feature=64, normalize=True).to(DEVICE)
    
    # Feed 10,000 REAL Images
    # We iterate through the dataloader until we have 10k images
    print("Processing Real Images...")
    real_count = 0
    for x, _ in dataloader:
        x = x.to(DEVICE)
        x = (x * 0.5) + 0.5 # [-1,1] -> [0,1]
        x = x.repeat(1, 3, 1, 1)
        x = F.interpolate(x, size=(299, 299), mode="bilinear")
        fid_2048.update(x, real=True)
        real_count += x.shape[0]
        if real_count >= 10000:
            break
            
    # Feed 10,000 FAKE Images
    print("Generating Fake Images...")
    model.eval()
    fake_count = 0
    batch_size = 128
    
    with torch.no_grad():
        while fake_count <= 10000:
            fake = sample(model, sampling_schedule, DEVICE, shape=(batch_size, 1, 28, 28))
            fake = fake.repeat(1, 3, 1, 1)
            fake = F.interpolate(fake, size=(299, 299), mode="bilinear")
            fid_2048.update(fake, real=False)
            fake_count += batch_size
            
    score = fid_2048.compute().item()
    print(f"TEST FID SCORE: {score}")


# ┌───────────────────────────────────────────────┐
# │                 DRIVER CODE                   │
# └───────────────────────────────────────────────┘
if __name__ == '__main__':

    # Determine Device for the whole session
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # print(f"Using device: {DEVICE}")

    # Init MNIST Dataloader
    mnist_dataloader = get_mnist_dataloader(batch_size=128)

    # Init Models
    trained_online_model = ConsistencyUNet().to(DEVICE)
    trained_ema_model = ConsistencyUNet().to(DEVICE)

    # Load Weights into the Consistency Models
    trained_online_model.load_state_dict(torch.load("weights/online_cm_config6.pth", map_location=DEVICE))
    trained_ema_model.load_state_dict(torch.load("weights/ema_cm_config6.pth", map_location=DEVICE))


    # Define Sampling Schedule
    sampling_schedule = torch.tensor([80.0, 40.0, 10.0, 5.0, 0.002], device=DEVICE)
    # sampling_schedule = torch.tensor([80.0, 40.0, 20.0, 10.0, 5.0, 2.5, 1.0, 0.1, 0.01, 0.002], device=DEVICE)
    # sampling_schedule = torch.tensor([80.0000, 4.4465, 0.7459, 0.2391, 0.0021], device=DEVICE)
    # sampling_schedule = torch.tensor([80.0], device=DEVICE)


    # Perform Sampling using Online Model
    trained_online_model.eval()
    with torch.no_grad():
        sampled_imgs_tensor = sample(trained_online_model, sampling_schedule, device=DEVICE)        # (N, 1, H, W)
    sampled_imgs_np = sampled_imgs_tensor.view(25, 28, 28).cpu().detach().numpy()

    # Plot all 25 imgs in a 5 by 5 collage
    fig, axes = plt.subplots(1, 10, figsize=(10, 2))
    # fig, axes = plt.subplots(1, 10)
    for i, ax in enumerate(axes.flatten()):
        ax.imshow(sampled_imgs_np[i], cmap='gray')
        ax.axis('off')

    plt.suptitle(f"25 Generated Digits", fontsize=20)
    plt.savefig('viz/generation_config6_5steps.png')
    print("Saved a collage of 25 Generated Images")

    # Calulcate test FID on 10k images
    # calculate_test_fid(trained_online_model, mnist_dataloader, sampling_schedule)

    # TODO: Visualize the generation quality (FID score) vs number of sampling steps