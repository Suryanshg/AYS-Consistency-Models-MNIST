from typing import List

from matplotlib import pyplot as plt


def visualize_loss_trajectory(loss_history: List[float], path: str):
    plt.figure(figsize=(6, 4))
    plt.plot(loss_history, color="blue")
    plt.xlabel("Epochs")
    plt.ylabel("MSE Loss")
    plt.title("Consistency Model: Loss Trajectory")
    plt.grid(True, linestyle="--", alpha=0.3)
    plt.savefig(path)


def visualize_fid_trajectory(fid_scores: List[float], path: str):
    plt.figure(figsize=(6, 4))
    plt.plot(fid_scores, color="blue")
    plt.xlabel("Epochs")
    plt.ylabel("FID")
    plt.title("Consistency Model: FID")
    plt.grid(True, linestyle="--", alpha=0.3)
    plt.savefig(path)