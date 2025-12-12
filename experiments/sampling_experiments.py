import random
import time

import numpy as np
import torch

from cm_sampler import ConsistencyModel
from datasets.mnist_dataloader import get_mnist_dataloader
from experiments.experiment_helpers import make_z_t_grid, calculate_pca, calculate_diversity_scores, calculate_pca_diversity, calculate_point_correlations
from visualizations.visualizations import schedule_length_plot, correlation_diversity_plot, plot_collage, plot_pca, plot_curvature


def evaluate_dependence(schedule, num_z_t=10, num_points=5, plot=False, deterministic=False):
    # Step 1: Generate an experiment grid
    z_t_grid = make_z_t_grid(N=num_z_t, experiment_cluster_size=num_points, device=cm_model.device)

    # Step 2: Run CM for all z_t in grid, resulting in x_t_grid
    x_t_grid = cm_model.propagate_zT(z_t_grid, schedule, deterministic=deterministic)

    # Reshape output and get 2D PCA
    x_t_grid = x_t_grid.reshape(num_z_t, num_points, *x_t_grid.shape[1:])
    pca_grid = calculate_pca(x_t_grid)

    # Calculate diversity scores
    diversity_scores = calculate_diversity_scores(x_t_grid)
    pca_diversity_scores = calculate_pca_diversity(pca_grid.reshape(num_z_t, num_points, pca_grid.shape[1]))

    # Show correlation
    correlations = calculate_point_correlations(x_t_grid)
    correlations_str = [f"z_t_{i} correlation: {corr}" for i,corr in enumerate(correlations)]
    print(correlations_str)
    avg_corr = torch.tensor(correlations).mean().item()
    div_score = sum(diversity_scores) / len(diversity_scores)
    pca_div_score = sum(pca_diversity_scores) / len(pca_diversity_scores)
    print(f"Average Correlation: {avg_corr}")
    print("Avg Diversity Scores:", div_score)
    print("Avg PCA diversity:", pca_div_score)

    if plot:
        plot_pca(pca_grid, num_z_t, num_points, deterministic)

    return avg_corr, div_score, pca_div_score


def dependence_experiment(optimize_by="div_score", N_points=10, n_candidates=7):
    schedule = [80.0]
    history = []  # records: [{"N": int, "candidate_idx": int, "avg_corr":float, "div_score":float, "pca_div_score":float}]

    while len(schedule) < N_points:
        current_min = min(schedule)

        # Generate n_candidates equally spaced distances below current_max
        candidate_distances = np.linspace(current_min * 0.99, current_min * 0.01, n_candidates).astype(np.float32)
        candidate_distances = [d for d in candidate_distances if d > 0.002 and d not in schedule]

        if len(candidate_distances) == 0:
            print("No valid candidates remaining, stopping iteration.")
            break

        # Evaluate candidates and store metrics with candidate index
        step_results = []
        for idx, d in enumerate(candidate_distances):
            avg_corr, div_score, pca_div_score = evaluate_dependence(sorted(schedule + [d], reverse=True))
            step_results.append({
                "N": len(schedule) + 1,
                "candidate": d,
                "avg_corr": avg_corr,
                "div_score": div_score,
                "pca_div_score": pca_div_score
            })

        # Add step results to history
        history.extend(step_results)

        # Pick candidate with highest correlation
        if optimize_by == "random":
            best_candidate = float(np.random.choice([r["candidate"] for r in step_results]))
        else:
            best_candidate = float(candidate_distances[np.argmax([r[optimize_by] for r in step_results])])
        schedule.append(best_candidate)
        schedule.sort(reverse=True)

    return history, schedule

def random_append_schedule(initial_schedule: list):
    schedule_full = [80., 70., 65., 60., 55.,
                     10., 5., 1., 0.5, 0.002]

    # Pick random item from schedule_full and add to schedule
    t_to_add = random.choice(schedule_full[1:])
    initial_schedule.append(t_to_add)
    initial_schedule.sort(reverse=True)
    return initial_schedule

def schedule_length_experiment(max_points=10):
    schedule = [80.]

    fid_scores = []
    time_deltas = []
    while len(schedule) < max_points:
        # Time how long it takes
        t1 = time.time()
        # Calculate fid score
        fid_score = cm_model.evaluate_fid(schedule)
        t2 = time.time()
        print(f"FID Score for size: {len(schedule)} = {fid_score}")
        print(f"Time Taken: {t2-t1:.3f}s")

        # Append to logs for plots later
        time_deltas.append(t2-t1)
        fid_scores.append(fid_score)
        # Pick random item from schedule_full and add to schedule
        schedule = random_append_schedule(schedule)

    return fid_scores, time_deltas

def ays_fid_experiment(max_points=10):
    # Compute FID for different schedule lengths using AYS now
    non_ays_schedule = [80.]

    fid_normal_scores = []
    fid_ays_scores = []
    for i in range(2, max_points+1):
        # Update AYS and normal schedules
        non_ays_schedule = random_append_schedule(non_ays_schedule)
        ays_schedule = cm_model.get_ays_schedule(num_steps=i).astype(np.float16).tolist()

        # Evaluate FID for both schedules
        fid_norm = cm_model.evaluate_fid(non_ays_schedule)
        fid_ays = cm_model.evaluate_fid(ays_schedule)
        print(f"N: {i} | FID Normal: {fid_norm} | FID AYS: {fid_ays}")

        fid_normal_scores.append(fid_norm)
        fid_ays_scores.append(fid_ays)

    return fid_normal_scores, fid_ays_scores


if __name__ == '__main__':
    print("Loading CM Model...")
    cm_model = ConsistencyModel()
    cm_model.initialize_FID(get_mnist_dataloader(batch_size=64), num_real_batches=64)
    cm_model.load("online_cm_config6.pth")

    print("Sampling some example images...")
    test_schedule = [80., 40., 30., 5., 0.002]
    #results = cm_model.sample(n_samples=25, schedule=test_schedule)
    #plot_collage(results)

    #print("Running AYS FID + Schedule Length experiment...")
    #fid_norms, fid_ays = ays_fid_experiment()
    #schedule_length_plot(fid_norms, fid_ays, labels=["FID Standard", "FID w/ AYS"])

    print("Running AYS Curvature plot...")
    schedule = cm_model.get_ays_schedule(5)
    plot_curvature(cm_model.velocities, cm_model.sigmas, schedule)

    print("Running correlation dependence experiment...")
    evaluate_dependence(test_schedule, plot=True, deterministic=False)
    dependence_experiment(optimize_by="random")
    evaluate_dependence(test_schedule, plot=True, deterministic=True)
    history, _ = dependence_experiment(optimize_by="random")
    correlation_diversity_plot(history)