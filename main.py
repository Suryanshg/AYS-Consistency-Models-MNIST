import numpy as np

from cm_sampler import ConsistencyModel
from cm_trainer import train_consistency
from datasets.mnist_dataloader import get_mnist_dataloader
from experiments.sampling_experiments import ays_fid_experiment, evaluate_dependence, dependence_experiment
from visualizations.visualizations import plot_collage, plot_curvature, schedule_length_plot, \
    correlation_diversity_plot, visualize_loss_trajectory, visualize_fid_trajectory

TRAIN = False
EXPERIMENT = True
SAMPLE = True

def main():
    # 1. Run CM Training if Requested
    if TRAIN:
        mnist_dl = get_mnist_dataloader()
        online_model, ema_model, loss_history, fid_scores = train_consistency(dataloader=mnist_dl)

        # Visualize training trajectory
        visualize_loss_trajectory(loss_history)
        visualize_fid_trajectory(fid_scores)
    # 2. Run Sampling Experiments if Requested
    if EXPERIMENT:
        print("Loading CM Model...")
        cm_model = ConsistencyModel()
        test_schedule = [80., 40., 30., 5., 0.002]
        cm_model.initialize_FID(get_mnist_dataloader(batch_size=64), num_real_batches=64)
        cm_model.load("online_cm_config6.pth")

        print("Running AYS FID + Schedule Length experiment...")
        fid_norms, fid_ays = ays_fid_experiment(cm_model)
        schedule_length_plot(fid_norms, fid_ays, labels=["FID Standard", "FID w/ AYS"])

        print("Running AYS Curvature plot...")
        ays_schedule = cm_model.get_ays_schedule(5).astype(np.float32).tolist()
        plot_curvature(cm_model.velocities, cm_model.sigmas, ays_schedule)

        if SAMPLE:
            print("Sampling some example CM images...")
            plot_collage(cm_model.sample(n_samples=25, schedule=test_schedule), title="CM")

            print("Sampling some example CM + AYS images...")
            plot_collage(cm_model.sample(n_samples=25, schedule=ays_schedule), title="CM_AYS")

        print("Running correlation dependence experiment...")
        evaluate_dependence(cm_model, test_schedule, plot=True, deterministic=False)
        dependence_experiment(cm_model, optimize_by="random")
        evaluate_dependence(cm_model, test_schedule, plot=True, deterministic=True)
        history, _ = dependence_experiment(cm_model, optimize_by="random")
        correlation_diversity_plot(history)

if __name__ == "__main__":
    main()
