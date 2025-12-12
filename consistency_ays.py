import torch
import numpy as np
import matplotlib.pyplot as plt
from models.ConsistencyUNet2 import ConsistencyUNet
from eval_consistency import sample


def get_prediction_velocity(model, device, num_points=100, sigma_max=80.0, sigma_min=0.002):
    """
    Scans the model from sigma_max down to sigma_min.
    Measures how much the predicted x_0 changes (velocity) at each level.
    """
    # Create a list of sigmas in log-space (High Noise (80.0) -> Low Noise (0.002))
    sigmas = np.exp(np.linspace(np.log(sigma_max), np.log(sigma_min), num_points))
    
    # Create a fixed Gaussian noise vector to probe consistency model
    # We use a batch of 10 to average out random fluctuations
    # This is important for simulating the PF ODE Trajectory
    fixed_noise = torch.randn(10, 1, 28, 28).to(device)         # (10, 1, 28, 28)
    
    # Init a List to store velocities
    velocities = []
    
    model.eval()
    with torch.no_grad():
        for i in range(len(sigmas) - 1):

            # Extract Current Sigma and Next Sigma values (Consecutive at index i)
            sigma_curr = sigmas[i] # High Noise Signal
            sigma_next = sigmas[i+1] # Low Noise Signal
            
            # Create inputs for current and next sigma
            # Note: We scale the same fixed noise by the sigma
            z_curr = fixed_noise * sigma_curr                       # (10, 1, 28, 28)
            z_next = fixed_noise * sigma_next                       # (10, 1, 28, 28)
            
            t_curr = torch.full((10,), sigma_curr, device=device)   # (10, )
            t_next = torch.full((10,), sigma_next, device=device)   # (10, )
            
            # Get predictions
            pred_curr = model(z_curr, t_curr)                       # (10, 1, 28, 28)     
            pred_next = model(z_next, t_next)                       # (10, 1 ,28, 28)
            
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
            
    # Return Velocities and the sigma values
    # return np.array(velocities), sigmas[:-1] 
    return np.array(velocities), sigmas[:-1]


def compute_ays_schedule(velocities, sigmas, num_steps=5):
    """
    Uses the velocity (curvature) profile to pick optimal steps.
    We treat velocity as a Probability Density Function (PDF) and sample from it.
    """
    # Normalize velocities to create a PDF
    pdf = velocities / np.sum(velocities)
    
    # Compute Cumulative Distribution Function (CDF)
    cdf = np.cumsum(pdf)
    
    # Sample 'num_steps' points evenly from the CDF (0 to 1)
    # We want to find sigmas that correspond to cumulative probability 0.2, 0.4, 0.6...
    # target_probs = np.linspace(0, 1, num_steps + 1)
    target_probs = np.linspace(0, 1, num_steps)

    # Use Linear Interpolation (Inverse CDF Sampling)
    # We ask: "At what exact sigma is the CDF = 0.2?"
    optimal_sigmas = np.interp(target_probs, cdf, sigmas)
    
    # Fix Boundaries
    # Interpolation might slightly miss the exact start/end due to floating point math
    optimal_sigmas[0] = sigmas[0]   # Force start to be exactly sigma_max
    optimal_sigmas[-1] = sigmas[-1] # Force end to be exactly sigma_min
    
    return np.array(optimal_sigmas)


if __name__ == "__main__":
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load the trained Consistency model
    model = ConsistencyUNet().to(DEVICE)
    model.load_state_dict(torch.load("weights/online_cm_config6.pth", map_location=DEVICE))
    
    # Scan the model (PF ODE) for Curvature
    velocities, scan_sigmas = get_prediction_velocity(model, DEVICE, num_points=200, sigma_max=80.0, sigma_min=0.002)

    # Calculate Optimal Steps
    N_STEPS = 5
    optimal_schedule = compute_ays_schedule(velocities, scan_sigmas, num_steps=N_STEPS)

    # Print the optimized schedule for usage
    formatted_schedule = ", ".join([f"{t:.4f}" for t in optimal_schedule])
    print("\n" + "="*40)
    print(f"OPTIMAL AYS SCHEDULE ({N_STEPS} Steps)")
    print("="*40)
    print(f"[{formatted_schedule}]")
    print("="*40)
    
    # Create a Visualization for Velocity Curvature and the Optimized Steps
    plt.figure(figsize=(12, 6))
    
    # Plot the Curvature (Blue Line)
    plt.plot(scan_sigmas, velocities, label="Model Curvature (RMSE)", color='blue', linewidth=2)
    
    regular_sampling_schedule = [80.0, 40.0, 10.0, 2.0, 0.002]
    # regular_sampling_schedule = [80.0, 40.0, 20.0, 10.0, 5.0, 2.5, 1.0, 0.1, 0.01, 0.002]

    # Plot the Regular Steps (Black Dashed Lines)
    # For each sigma in regular sampling schedule
    for i, step_sigma in enumerate(regular_sampling_schedule):
        if i == 0:
            plt.axvline(x=step_sigma, color='black', linestyle='--', alpha=0.8, linewidth=1.5, label = "Regular Steps")
        else:
            plt.axvline(x=step_sigma, color='black', linestyle='--', alpha=0.8, linewidth=1.5)
        
        # Add a text label at the top of the line
        # We place it slightly above the max velocity to keep it clean
        plt.text(step_sigma, max(velocities) * -0.15, f"t_{i + 1}", fontsize=9, ha='center', va='bottom', rotation = 0)


    # Plot the AYS Steps (Red Dashed Lines)
    # For each sigma in optimal schedule
    for i, step_sigma in enumerate(optimal_schedule):
        if i == 0:
            plt.axvline(x=step_sigma, color='red', linestyle='--', alpha=0.8, linewidth=1.5, label = "Optimized Steps")
        else:
            plt.axvline(x=step_sigma, color='red', linestyle='--', alpha=0.8, linewidth=1.5)
        
        # Add a text label at the top of the line
        # We place it slightly above the max velocity to keep it clean
        plt.text(step_sigma, max(velocities) * 1.05, f"t_{i + 1}'", fontsize=9, ha='center', va='bottom', rotation=0)

    # Adding Log Scale
    plt.xscale('log')

    # Inverted X-axis: Noise (80.0) --> Clean (0.002)
    plt.xlim(80, 0.002) 
    
    # Add Custom Ticks for readability
    ticks = [80, 40, 10, 1, 0.1, 0.01, 0.002]
    labels = ["80", "40", "10", "1", "0.1", "0.01", "0.002"]
    plt.xticks(ticks, labels)
    
    # Add title, axis labels, legend and grid
    plt.title(f"AYS Schedule: {N_STEPS} Steps optimized for Consistency Model", fontsize=14, y=1.1)
    plt.xlabel("Sigma (Noise Level) - Log Scale", fontsize=12, labelpad=15)
    plt.ylabel("RMSE (Pixel Shift Velocity)", fontsize=12)
    plt.legend(loc='upper right')
    plt.grid(True, which="both", linestyle=':', alpha=0.4)
    
    # Save the Visualization
    plt.tight_layout()
    plt.savefig('viz/ays_schedule_overlay.png')
    print("Saved visualization to viz/ays_schedule_overlay.png")

    # Prepare the Optimized Sampling Schedule as a Tensor
    optimized_sampling_schedule = torch.tensor(optimal_schedule, device=DEVICE, dtype=torch.float32)

    # Perform Sampling using Online Model and Optimized Sampling Schedule
    model.eval()
    with torch.no_grad():
        sampled_imgs_tensor = sample(model, optimized_sampling_schedule, device=DEVICE)        # (N, 1, H, W)
    sampled_imgs_np = sampled_imgs_tensor.view(25, 28, 28).cpu().detach().numpy()

    # Plot all 25 imgs in a 5 by 5 collage
    fig, axes = plt.subplots(1, 10, figsize=(10, 2))
    for i, ax in enumerate(axes.flatten()):
        ax.imshow(sampled_imgs_np[i], cmap='gray')
        ax.axis('off')

    plt.suptitle(f"25 Generated Digits (CM + AYS)", fontsize=20)
    plt.savefig('viz/generation_config6_optim_5steps.png')
    print("Saved a collage of 25 Generated Images using Optimized Sampling Schedule")