import numpy as np
import torch
from sklearn.decomposition import PCA

def make_z_t_grid(N: int, experiment_cluster_size: int, device, image_dim: tuple = (1, 28, 28)):
    # Step 1: Create N distinct z_t vectors
    z_t = torch.randn((N, *image_dim)).to(device)

    # Step 2: Repeat each z_t sampled experiment_cluster_size times
    z_t = z_t.repeat_interleave(experiment_cluster_size, dim=0)

    return z_t

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

def calculate_diversity_scores(x_t_grid):
    # x_t_grid: (num_z_t, num_points, C, H, W)
    B, N, C, H, W = x_t_grid.shape
    diversities = []

    for i in range(B):
        imgs = x_t_grid[i].reshape(N, -1)
        diffs = imgs.unsqueeze(1) - imgs.unsqueeze(0)
        dists = torch.linalg.norm(diffs, dim=2)   # L2 Norm

        triu = torch.triu_indices(N, N, offset=1)
        mean_dist = dists[triu[0], triu[1]].mean().item()
        diversities.append(mean_dist)

    return diversities

def calculate_pca_diversity(pca_grid):
    B, N, D = pca_grid.shape
    diversities = []

    for i in range(B):
        pts = pca_grid[i]                                      # shape: (num_points, D)
        diffs = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]  # pairwise differences
        dists = np.linalg.norm(diffs, axis=2)                  # L2 norm
        triu_indices = np.triu_indices(N, k=1)
        mean_dist = np.mean(dists[triu_indices])
        diversities.append(mean_dist)

    return diversities