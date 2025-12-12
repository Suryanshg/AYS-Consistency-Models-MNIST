from typing import List

import numpy as np
import pandas as pd
import torch
from matplotlib import pyplot as plt
import seaborn as sns

VIZ_PATH = "visualizations/images"

# Visualization of PCA on 2D Grid
def plot_pca(pca_results, num_z_t, num_points, deterministic):

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

    if deterministic:
        plt.title("CM Sampling Clustering for Deterministic ODE (PCA)")
    else:
        plt.title("CM Sampling Clustering for Stochastic ODE (PCA)")
    plt.xlabel("PC 1")
    plt.ylabel("PC 2")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    if deterministic:
        plt.savefig(f"{VIZ_PATH}/PCAClustering_ODE.png")
    else:
        plt.savefig(f"{VIZ_PATH}/PCAClustering_SDE.png")
    plt.show()


def schedule_length_plot(*fid_lists, labels=None, offset=2):
    """
    Plot one or more FID score lists over N.
    Marks the best N (minimum FID) across all lists.

    Parameters:
        *fid_lists : arbitrary number of lists/arrays of FID scores
        labels     : list of strings for each line (optional)
        colors     : list of colors for each line (optional)
    """
    n_lines = len(fid_lists)
    if labels is None:
        labels = [f"FID {i + 1}" for i in range(n_lines)]

    colors = sns.color_palette("tab10", n_lines)

    # Convert all lists to numpy arrays and compute global minimum
    fid_arrays = [np.array(fid) for fid in fid_lists]
    min_val = min([fid.min() for fid in fid_arrays])
    best_idx = None
    # Find first occurrence of global min
    for fid in fid_arrays:
        if min_val in fid:
            best_idx = np.where(fid == min_val)[0][0] + offset  # +1 for 1-indexed N
            break

    # X-axis values
    N_values = np.arange(offset, len(fid_arrays[0]) + offset)
    fig, ax = plt.subplots(figsize=(8, 5))

    for i, fid_scores in enumerate(fid_arrays):
        sns.lineplot(x=N_values, y=fid_scores, label=labels[i], color=colors[i], ax=ax, marker="o")

    # Mark global best point
    ax.scatter([best_idx], [min_val], s=120, zorder=10, color="red", label="Best N")
    ax.annotate(f"Best N @ {best_idx}",
                (best_idx, min_val),
                textcoords="offset points", xytext=(10, 10), zorder=11, color="red")

    ax.set_xlabel("N")
    ax.set_ylabel("FID Score")
    ax.set_xticks(N_values)
    ax.legend()
    plt.title("FID vs Schedule Length (N)")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{VIZ_PATH}/FID_vs_N.png")
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
    plt.savefig(f"{VIZ_PATH}/Correlation_SDE.png")
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
    plt.savefig(f"{VIZ_PATH}/Diversity_SDE.png")
    plt.show()

def plot_collage(images, title="1", collage_dim=(5, 5)):
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
    plt.savefig(f"{VIZ_PATH}/SAMPLE_{collage_dim[0]}x{collage_dim[1]}_{title}.png")
    plt.show()

def plot_curvature(velocities, sigmas, optimal_schedule=None):
    # Create a Visualization for Velocity Curvature and the Optimized Steps
    plt.figure(figsize=(12, 6))

    # Plot the Curvature (Blue Line)
    plt.plot(sigmas, velocities, label="Model Curvature (RMSE)", color='blue', linewidth=2)

    regular_sampling_schedule = [80.0, 40.0, 10.0, 2.0, 0.002]

    # Plot the Regular Steps (Black Dashed Lines)
    # For each sigma in regular sampling schedule
    for i, step_sigma in enumerate(regular_sampling_schedule):
        if i == 0:
            plt.axvline(x=step_sigma, color='black', linestyle='--', alpha=0.8, linewidth=1.5, label="Regular Steps")
        else:
            plt.axvline(x=step_sigma, color='black', linestyle='--', alpha=0.8, linewidth=1.5)

        # Add a text label at the top of the line
        # We place it slightly above the max velocity to keep it clean
        plt.text(step_sigma, max(velocities) * -0.15, f"t_{i + 1}", fontsize=9, ha='center', va='bottom', rotation=0)

    # Plot the AYS Steps (Red Dashed Lines)
    # For each sigma in optimal schedule
    if optimal_schedule is not None:
        for i, step_sigma in enumerate(optimal_schedule):
            if i == 0:
                plt.axvline(x=step_sigma, color='red', linestyle='--', alpha=0.8, linewidth=1.5, label="Optimized Steps")
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
    plt.title(f"AYS Schedule: {len(optimal_schedule)} Steps optimized for Consistency Model", fontsize=14, y=1.1)
    plt.xlabel("Sigma (Noise Level) - Log Scale", fontsize=12, labelpad=15)
    plt.ylabel("RMSE (Pixel Shift Velocity)", fontsize=12)
    plt.legend(loc='upper right')
    plt.grid(True, which="both", linestyle=':', alpha=0.4)

    # Save the Visualization
    plt.tight_layout()
    plt.savefig(f'{VIZ_PATH}/ays_schedule_overlay.png')
    plt.show()

# ----- TRAINING VIZ -----
def visualize_loss_trajectory(loss_history: List[float]):
    plt.figure(figsize=(6, 4))
    plt.plot(loss_history, color="blue")
    plt.xlabel("Epochs")
    plt.ylabel("MSE Loss")
    plt.title("Consistency Model: Loss Trajectory")
    plt.grid(True, linestyle="--", alpha=0.3)
    plt.savefig(f"{VIZ_PATH}/loss_trajectory.png")
    plt.show()


def visualize_fid_trajectory(fid_scores: List[float]):
    plt.figure(figsize=(6, 4))
    plt.plot(fid_scores, color="blue")
    plt.xlabel("Epochs")
    plt.ylabel("FID")
    plt.title("Consistency Model: FID")
    plt.grid(True, linestyle="--", alpha=0.3)
    plt.savefig(f"{VIZ_PATH}/fid_trajectory.png")
    plt.show()