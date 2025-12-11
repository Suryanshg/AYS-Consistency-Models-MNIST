import numpy as np
import pandas as pd
import torch
from matplotlib import pyplot as plt
import seaborn as sns

# Visualization of PCA on 2D Grid
def plot_pca(pca_results, num_z_t, num_points):

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

def schedule_length_plot(fid_scores):
    fid_scores = np.array(fid_scores)
    N_values = np.arange(1, len(fid_scores) + 1)

    fig, ax = plt.subplots()
    sns.lineplot(x=N_values, y=fid_scores, label="FID", color="blue", ax=ax)
    best_idx = np.argmin(fid_scores) + 1 # Minimum FID score point

    # Mark best point
    ax.scatter([best_idx], [fid_scores[best_idx - 1]],  s=120, zorder=10, color="red", label="Best N")
    ax.annotate(f"Best N @ {best_idx}", (best_idx, fid_scores[best_idx - 1]), textcoords="offset points", xytext=(10,10), zorder=11)

    ax.set_xlabel("N")
    ax.set_ylabel("FID Score")
    ax.set_xticks(N_values)
    ax.legend()
    plt.title("FID vs Schedule Length (N)")
    plt.show()

def correlation_diversity_plot(data):
    df = pd.DataFrame(data)

    # Average values per N
    mean_df = (
        df.groupby("N")[["avg_corr", "div_score", "pca_div_score"]]
        .mean()
        .reset_index()
        .sort_values("N")
    )

    x_vals = mean_df["N"].values

    # ----------------------------
    # 1. Correlation Plot
    # ----------------------------
    plt.figure(figsize=(7, 4))
    sns.lineplot(x=x_vals, y=mean_df["avg_corr"].values, marker="o")
    plt.title("Average Correlation per N")
    plt.xlabel("N (schedule size)")
    plt.ylabel("Correlation (0–1)")
    plt.xticks(x_vals)
    if mean_df["avg_corr"].max() < 1:
        plt.ylim((0, 1))
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

    # ----------------------------
    # 2. Diversity Plot
    # ----------------------------
    plt.figure(figsize=(7, 4))
    sns.lineplot(x=x_vals, y=mean_df["div_score"], label="Diversity", marker="o")
    sns.lineplot(x=x_vals, y=mean_df["pca_div_score"], label="PCA Diversity", marker="o")

    plt.title("Diversity Metrics per N")
    plt.xlabel("N (schedule size)")
    plt.ylabel("Diversity (0–1 normalized)")
    if mean_df["pca_div_score"].max() < 1:
        plt.ylim((0, 1))
    plt.xticks(x_vals)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
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

        ax.imshow(img.squeeze(), cmap='gray')
        ax.axis("off")

    plt.tight_layout()
    plt.show()