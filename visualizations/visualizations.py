import numpy as np
import pandas as pd
import torch
from matplotlib import pyplot as plt
import seaborn as sns

# Visualization of PCA on 2D Grid
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

def schedule_length_plot(fid_scores, time_deltas):
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

def correlation_diversity_plot(data):
    df = pd.DataFrame(data)
    mean_df = df.groupby("N")[["avg_corr", "div_score", "pca_div_score"]].max().reset_index() # Max across candidates
    x_vals = mean_df["N"]

    fig, ax = plt.subplots()

    # Plot correlation on left axis
    sns.lineplot(x=x_vals, y=mean_df["avg_corr"], label="Correlation", color="blue", ax=ax)
    ax.set_ylabel("Correlation (0-1)", color="blue")
    ax.tick_params(axis="y", labelcolor="blue")

    # Plot diversity and PCA diversity on right axis
    ax2 = ax.twinx()
    sns.lineplot(x=x_vals, y=mean_df["div_score"], label="Diversity", color="green", ax=ax2)
    sns.lineplot(x=x_vals, y=mean_df["pca_div_score"], label="PCA Diversity", color="orange", ax=ax2)
    ax2.set_ylabel("Diversity Metrics (0-1 normalized)", color="black")
    ax2.tick_params(axis="y", labelcolor="black")

    ax.set_xlabel("N (schedule size)")
    ax.set_xticks(x_vals)
    fig.suptitle("Average Correlation and Diversity Metrics per N")
    fig.tight_layout()

    # Combine legends
    lines, labels = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(lines + lines2, labels + labels2, loc="upper right")

    plt.show()

def plot_collage(images, collage_dim=(5, 5)):
    rows, cols = collage_dim
    n = rows * cols

    # Trim or pad the images list to match grid_size
    images = list(images)
    if len(images) < n:
        raise ValueError(f"Not enough images ({len(images)}) for grid size {rows}x{cols}")
    images = images[:n]

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2, rows * 2))

    for ax, img in zip(axes.flatten(), images):
        # Convert Torch to Numpy in case bad type requested
        if isinstance(img, torch.Tensor):
            img = img.detach().cpu().numpy()

        # If channel-first, move to channel-last
        if img.ndim == 3 and img.shape[0] in (1, 3):
            img = np.moveaxis(img, 0, -1)

        ax.imshow(img.squeeze())
        ax.axis("off")

    plt.tight_layout()
    plt.show()