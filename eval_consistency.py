import torch
import torch.nn as nn
import math
# from models.ConsistencyUNet import ConsistencyUNet
from models.ConsistencyUNet2 import ConsistencyUNet
import matplotlib.pyplot as plt
from fid import compute_fid_score
from datasets.mnist_dataloader import get_mnist_dataloader


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


# ┌───────────────────────────────────────────────┐
# │                 DRIVER CODE                   │
# └───────────────────────────────────────────────┘
if __name__ == '__main__':

    # Determine Device for the whole session
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # print(f"Using device: {DEVICE}")

    # Init MNIST Dataloader
    mnist_dataloader = get_mnist_dataloader(batch_size=25)

    # Init Models
    trained_online_model = ConsistencyUNet().to(DEVICE)
    trained_ema_model = ConsistencyUNet().to(DEVICE)

    # Load Weights into the Consistency Models
    trained_online_model.load_state_dict(torch.load("trained_model_weights/consistency_online_cm2.pth", map_location=DEVICE))
    trained_ema_model.load_state_dict(torch.load("trained_model_weights/consistency_ema_cm2.pth", map_location=DEVICE))

    # Define Sampling Schedule
    sampling_schedule = torch.tensor([80.0, 40.0, 20.0, 10.0, 5.0, 4.0, 3.0, 2.0, 1.0, 0.5], device=DEVICE)
    # sampling_schedule = torch.tensor([40.0], device=DEVICE)

    # Perform Sampling using Online Model
    trained_online_model.eval()
    with torch.no_grad():

        sampled_imgs_tensor = sample(trained_online_model, sampling_schedule, device=DEVICE)        # (N, 1, H, W)

    # TODO: Calulcate FID
    # Get Real Images for comparison
    real_images, _ = next(iter(mnist_dataloader))
 
    # Convert to [0, 1] range
    real_images = (real_images * 0.5 + 0.5).to(DEVICE)

    # Use sampled images as fake images
    fake_images = sampled_imgs_tensor

    # Compute FID
    fid_score = compute_fid_score(real_images=real_images, fake_images=fake_images)

    print(f"Evaluation FID Score: {fid_score}")

    
    sampled_imgs_np = sampled_imgs_tensor.view(25, 28, 28).cpu().detach().numpy()

    # Plot all 25 imgs in a 5 by 5 collage
    # Plot the images
    fig, axes = plt.subplots(5, 5, figsize=(8, 6))
    for i, ax in enumerate(axes.flatten()):
        ax.imshow(sampled_imgs_np[i], cmap='gray')
        ax.axis('off')

    plt.suptitle(f"25 Generated Digits", fontsize=20)
    plt.savefig('viz/generation.png')

    # Uncomment to perform sampling using EMA Model

    # # Perform Sampling using EMA Model
    # trained_ema_model.eval()
    # with torch.no_grad():

    #     sampled_imgs_tensor = sample(trained_ema_model, sampling_schedule, device=DEVICE)        # (N, 1, H, W)
    
    # sampled_imgs_np = sampled_imgs_tensor.view(25, 28, 28).cpu().detach().numpy()

    # # Plot all 25 imgs in a 5 by 5 collage
    # # Plot the images
    # fig, axes = plt.subplots(5, 5, figsize=(8, 6))
    # for i, ax in enumerate(axes.flatten()):
    #     ax.imshow(sampled_imgs_np[i], cmap='gray')
    #     ax.axis('off')

    # plt.suptitle(f"25 Generated Digits", fontsize=20)
    # plt.savefig('viz/1_step.png')



    # TODO: Visualize the generation quality (FID score) vs number of sampling steps