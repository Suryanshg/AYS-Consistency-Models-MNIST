import math
import random
import time

import numpy as np
import torch
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from torch import nn, Tensor
from torch.utils.data import DataLoader
from torchmetrics.image import FrechetInceptionDistance
import torch.nn.functional as F

from datasets.mnist_dataloader import get_mnist_dataloader
from models.ConsistencyUNet2 import ConsistencyUNet


def propogate_zT(z_t: Tensor, online_model: nn.Module, sampling_schedule: list, epsilon: float = 0.002) -> torch.Tensor:
    sampling_schedule = torch.tensor(sampling_schedule, device=DEVICE)
    image_dim = z_t.shape
    # If z_t is not a batch, turn into [B, C, H, W] form
    if len(image_dim) <= 3:
        z_t = z_t.unsqueeze(0)

    # Produce a batch tensor of first timestep in sampling schedule
    t_tensor = sampling_schedule[0]

    # Predict x_hat using initial noise vector and t_tensor
    x_hat = online_model(z_t, t_tensor)

    # Sample thru remaining multiple timesteps
    for t in sampling_schedule[1:]:
        # Sample a noise tensor using deterministic policy
        z = torch.randn_like(x_hat)

        # Add the sampled noise to the current x_hat
        z_t = x_hat + (math.sqrt((t ** 2) - (epsilon ** 2)) * z)

        # Broadcast time "t" to shape (batch_size,)
        t_tensor = t.repeat(image_dim[0])

        # Predict x_hat using noised image and t_tensor
        x_hat = online_model(z_t, t_tensor)

    # De-normalize the image
    x_hat = (x_hat * 0.5 + 0.5)

    # Clamp all values in the range [0, 1]
    x_hat = x_hat.clamp(0, 1)

    return x_hat


def make_batched_zT(N: int, experiment_cluster_size: int, image_dim: tuple = (1, 28, 28)) -> Tensor:
    # Step 1: Create N distinct z_t vectors
    z_t = torch.randn((N, *image_dim), device=DEVICE)

    # Step 2: Repeat each z_t sampled experiment_cluster_size times
    z_t = z_t.repeat_interleave(experiment_cluster_size, dim=0)

    return z_t


def evaluate_fid(model: nn.Module, fid_metric: FrechetInceptionDistance, schedule, num_batches=10, batch_size=128) -> float:
    """
    Generates fake plots and computes FID against the pre-computed real stats.
    """
    model.eval()

    with torch.no_grad():
        for _ in range(num_batches):
            # 1. Generate Noise
            z = torch.randn(batch_size, 1, 32, 32).to(DEVICE) * 80.0

            # 2. Generate Images
            fake_images = propogate_zT(z, model, schedule)  # Returns [-1, 1] or [0, 1] depending on your sample func

            # Process Fake plots
            fake_images = fake_images.repeat(1, 3, 1, 1)
            fake_images = F.interpolate(fake_images, size=(299, 299), mode='bilinear')

            # 4. Update metric with real=False
            fid_metric.update(fake_images, real=False)

    # Compute final score
    score = fid_metric.compute()
    return score.item()


def precompute_real_stats(dataloader: DataLoader, fid_metric: FrechetInceptionDistance, num_batches: int = 10):
    """
    Feeds real plots into the FID metric to update the 'real' statistics.
    """
    fid_metric.reset()

    count = 0
    for x, _ in dataloader:
        x = x.to(DEVICE)

        # Convert range from [-1, 1] to [0, 1]
        x = (x * 0.5) + 0.5

        # Repeat Grayscale to look like RGB Images (as InceptionV3 expects RGB plots)
        x = x.repeat(1, 3, 1, 1)

        # Resize to 299x299 (Required by InceptionV3)
        # Using bilinear interpolation
        x = F.interpolate(x, size=(299, 299), mode='bilinear')

        # Update metric with real=True
        fid_metric.update(x, real=True)

        count += 1
        if count >= num_batches:
            break


def plot_grid_images_pca(x_t_grid, pca_results):
    num_z_t, num_points = x_t_grid.shape[:2]

    # Colors for each cluster
    colors = plt.cm.tab10(range(num_z_t))

    # plot
    plt.figure(figsize=(8, 8))
    idx = 0
    for i in range(num_z_t):
        subset = pca_results[idx: idx + num_points]
        plt.scatter(
            subset[:, 0],
            subset[:, 1],
            color=colors[i],
            label=f"z_t {i}",
            s=40,
            alpha=0.8,
        )
        idx += num_points

    plt.title("Image Clustering (PCA)")
    plt.xlabel("PC 1")
    plt.ylabel("PC 2")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def plot_grid_images(x_t_grid):
    num_z_t, num_points = x_t_grid.shape[:2]

    fig, axes = plt.subplots(num_z_t, num_points, figsize=(num_points * 2.2, num_z_t * 2.2))

    # Handle degenerate cases for rows/columns
    if num_z_t == 1:
        axes = [axes]
    if num_points == 1:
        axes = [[ax] for ax in axes]

    for i in range(num_z_t):
        for j in range(num_points):
            img = x_t_grid[i, j].detach().cpu()

            if img.ndim == 3:
                img = img.permute(1, 2, 0)

            # Handle grayscale plots
            if img.shape[-1] == 1:
                img = img[..., 0]

            # Normalize if not in [0,1]
            img_min, img_max = img.min(), img.max()
            if img_max - img_min > 1e-6:
                img = (img - img_min) / (img_max - img_min)

            axes[i][j].imshow(img, cmap="gray" if img.ndim == 2 else None)
            axes[i][j].axis("off")

    plt.tight_layout()
    plt.show()

def calculate_pca(x_t_grid):
    num_z_t, num_points = x_t_grid.shape[:2]

    flat_imgs = []
    for i in range(num_z_t):
        for j in range(num_points):
            img = x_t_grid[i, j].detach().cpu()
            flat_imgs.append(img.flatten().numpy())

    flat_imgs = torch.tensor(flat_imgs).numpy()  # (N, D)

    pca = PCA(n_components=2)
    pca_results = pca.fit_transform(flat_imgs)  # (N, 2)
    return pca_results

def calculate_point_correlations(x_t_grid):
    num_z_t, num_points = x_t_grid.shape[:2]
    correlations = []

    for i in range(num_z_t):
        # Flatten each image to 1D vector
        points = x_t_grid[i].reshape(num_points, -1)  # (num_points, C*H*W)
        points = points.detach().cpu()

        # Compute correlation matrix
        points_mean = points.mean(dim=1, keepdim=True)
        points_std = points.std(dim=1, keepdim=True) + 1e-8
        points_norm = (points - points_mean) / points_std

        corr_matrix = points_norm @ points_norm.T / points_norm.shape[1]

        mask = 1 - torch.eye(corr_matrix.shape[0])
        overall_corr = (corr_matrix * mask).sum() / mask.sum()
        correlations.append(overall_corr)

    return correlations

def dependence_experiment():
    # Experiment constants
    num_z_t = 10
    num_points = 5
    schedule = [80.0, 40, 10, 0.002]

    # Step 1: Generate an experiment grid
    z_t_grid = make_batched_zT(N=num_z_t, experiment_cluster_size=num_points)

    # Step 2: Run CM for all z_t in grid, resulting in x_t_grid
    x_t_grid = propogate_zT(z_t=z_t_grid, online_model=cm_model, sampling_schedule=schedule)

    # Reshape output
    x_t_grid = x_t_grid.reshape(num_z_t, num_points, *x_t_grid.shape[1:])
    pca_grid = calculate_pca(x_t_grid)

    # Show correlation
    correlations = calculate_point_correlations(x_t_grid)
    correlations_str = [f"z_t_{i} correlation: {corr}" for i,corr in enumerate(correlations)]
    print(correlations_str)
    print(f"Average Correlation: {torch.tensor(correlations).mean().item()}")

    # Visualize outputs
    plot_grid_images_pca(x_t_grid, pca_grid)
    plot_grid_images(x_t_grid)

def schedule_length_experiment():
    schedule_full = [80., 75., 70., 65., 60., 55., 50.,
                     45., 40., 35., 30., 25., 20., 15.,
                     10., 5., 1., 0.5, 0.1, 0.002]
    schedule = [80.]

    fid_scores = []
    time_deltas = []
    while len(schedule) < len(schedule_full):
        # Time how long it takes
        t1 = time.time()
        # Calculate fid score
        fid_score = evaluate_fid(model=cm_model, fid_metric=fid, schedule=schedule)
        t2 = time.time()
        print(f"FID Score for size: {len(schedule)} = {fid_score}")
        print(f"Time Taken: {t2-t1:.3f}s")

        # Append to logs for plots later
        time_deltas.append(t2-t1)
        fid_scores.append(fid_score)
        # Pick random item from schedule_full and add to schedule
        t_to_add = random.choice(schedule_full[1:-1])
        schedule.append(t_to_add)
        schedule.sort(reverse=True)

    return fid_scores, time_deltas

def generate_schedule_length_plot(fid_scores, time_deltas):
    # Normalize 0–1
    fid_norm = (fid_scores - np.min(fid_scores)) / (np.max(fid_scores) - np.min(fid_scores))
    time_norm = (time_deltas - np.min(time_deltas)) / (np.max(time_deltas) - np.min(time_deltas))

    # Compute intersection (sweet spot)
    diff = np.abs(fid_norm - time_norm)
    sweet_idx = np.argmin(diff)+1

    fig, ax = plt.subplots()

    sns.lineplot(x=np.arange(1, len(fid_norm) + 1), y=fid_norm, label="FID", color="blue", ax=ax)
    sns.lineplot(x=np.arange(1, len(time_norm) + 1), y=time_norm, label="Time", color="green", ax=ax)

    # Mark sweet spot
    ax.scatter([sweet_idx], [fid_norm[sweet_idx - 1]], s=120, zorder=10, color="red", label="Ideal N")
    ax.annotate(f"Ideal N @ {sweet_idx}", (sweet_idx, fid_norm[sweet_idx - 1]), textcoords="offset points", xytext=(10,10), zorder=11)

    ax.set_xlabel("N")
    ax.set_ylabel("Normalized Values")
    ax.set_xticks(np.arange(1, len(fid_norm) + 1))
    ax.legend()
    plt.title("Normalized FID vs Time")

    plt.show()

    return sweet_idx


if __name__ == '__main__':
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Generating FID from real...")
    mnist_dl = get_mnist_dataloader(batch_size=64)
    fid = FrechetInceptionDistance(feature = 64, normalize=True).to(DEVICE)
    precompute_real_stats(mnist_dl, fid, num_batches=10)

    print("Loading CM Model...")
    cm_model = ConsistencyUNet().to(DEVICE)
    cm_model.load_state_dict(torch.load("trained_model_weights/consistency_online_cm2.pth", map_location=DEVICE))

    print("Running schedule size experiment...")
    fid_scores, time_deltas = schedule_length_experiment()
    generate_schedule_length_plot(fid_scores, time_deltas)

    print("Running correlation dependence experiment...")
    #dependence_experiment()