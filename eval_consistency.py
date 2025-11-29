import torch
import torch.nn as nn
from typing import List
import math
from models.u_net import ConsistencyUNet
import matplotlib.pyplot as plt

def sample(
        online_model: nn.Module,
        z: torch.Tensor,
        sampling_schedule: torch.Tensor,
        epsilon: float = 0.002
        ) -> torch.Tensor:
    
    # Get Batch Size for broadcasting
    batch_size = z.shape[0]

    # Produce a batch tensor of first timestep in sampling schedule
    t_tensor = sampling_schedule[0].repeat(batch_size)                  # (N,)

    # Predict x_hat using initial noise vector and t_tensor
    x_hat = online_model(z, t_tensor)

    # Sample thru remaining multiple timesteps
    for t in sampling_schedule[1:]:

        # Sample a noise tensor
        z = torch.randn_like(x_hat)

        # Add the sampled noise to the current x_hat
        z_t = x_hat + (math.sqrt((t ** 2) - (epsilon ** 2)) * z)

        # Broadcast time "t" to shape (batch_size,)
        t_tensor = t.repeat(batch_size)

        # Predict x_hat using noised image and t_tensor
        x_hat = online_model(z_t, t)
    
    # De-normalize the image
    x_hat = (x_hat * 0.5 + 0.5)

    # TODO: Check if this is needed at all
    # Clamp all values in the range [0, 1]
    x_hat = x_hat.clamp(0, 1) 

    return x_hat

if __name__ == '__main__':

    # Determine Device for the whole session
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {DEVICE}")

    # Init Backbone model
    trained_model = ConsistencyUNet().to(DEVICE)

    # Load Weights into the Consistency Model
    trained_model.load_state_dict(torch.load("trained_model_weights/consistency_ema.pth", map_location=DEVICE))


    # Perform Sampling
    trained_model.eval()
    with torch.no_grad():
        # Start with max noise (80.0)
        z_t = torch.randn(25, 1, 28, 28).to(DEVICE) * 80.0

        # Define Sampling Schedule
        sampling_schedule = torch.tensor([80.0, 40.0, 20.0, 10.0, 5.0], device=DEVICE)

        sampled_imgs_tensor = sample(trained_model, z_t, sampling_schedule)        # (25, 1, 28, 28)
    
    sampled_imgs_np = sampled_imgs_tensor.view(25, 28, 28).cpu().detach().numpy()

    print(sampled_imgs_np.shape)


    # Plot all 25 imgs in a 5 by 5 collage
    # Plot the images
    fig, axes = plt.subplots(5, 5, figsize=(8, 6))
    for i, ax in enumerate(axes.flatten()):
        ax.imshow(sampled_imgs_np[i], cmap='gray')
        ax.axis('off')

    plt.suptitle(f"25 Generated Digits", fontsize=20)
    plt.savefig('viz/consistency_5_steps_generation.png')