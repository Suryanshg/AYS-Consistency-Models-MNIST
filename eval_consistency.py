import torch
from typing import List
import math
from models.u_net import UNet
import matplotlib.pyplot as plt

def sample(
        online_model: UNet,
        z_t: torch.Tensor,
        sampling_schedule: List[float],
        epsilon: float = 0.002
        ) -> torch.Tensor:
    """
    TODO:

    Args:
        online_model (nn.Module): _description_
        z_t (torch.Tensor): _description_
        epsilon (float): _description_
        sampling_schedule (_type_): _description_

    Returns:
        torch.Tensor: _description_
    """
    # Sample thru first timestep
    x_hat = online_model(z_t, sampling_schedule[0])

    # Sample thru remaining multiple timesteps
    for t in sampling_schedule[1:]:
        z = torch.randn_like(x_hat)
        z_t = x_hat + (math.sqrt((t ** 2) - (epsilon ** 2)) * z)
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

    trained_consistency_model = UNet().to(DEVICE)

    trained_consistency_model.load_state_dict(torch.load("trained_model_weights/trained_consistency_model.pth"))

    trained_consistency_model.eval()
    with torch.no_grad():
        z_t = torch.randn(25, 1, 28, 28).to(DEVICE) * 80.0
        sampling_schedule = torch.tensor([80.0, 5.0], device=DEVICE)
        sampled_imgs_tensor = sample(trained_consistency_model, z_t, sampling_schedule)        # (25, 1, 28, 28)
    

    sampled_imgs_np = sampled_imgs_tensor.view(25, 28, 28).cpu().detach().numpy()

    print(sampled_imgs_np.shape)


    # Plot all 25 imgs in a 5 by 5 collage
    # Plot the images
    fig, axes = plt.subplots(5, 5, figsize=(8, 6))
    for i, ax in enumerate(axes.flatten()):
        ax.imshow(sampled_imgs_np[i], cmap='gray')
        ax.axis('off')

    plt.suptitle(f"25 Generated Digits", fontsize=20)
    plt.savefig('viz/consistency_2_steps_generation.png')