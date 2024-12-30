import os
import random

import matplotlib.pyplot as plt
import numpy as np
import scipy.stats
import torch
import torch.nn as nn
import torch.optim as optim
from moments import load_mean_variance
from nnfabrik.utility.nn_helpers import set_random_seed
from sensorium.datasets.mouse_video_loaders import mouse_video_loader
from sensorium.models.make_model import make_video_model
from sklearn.cross_decomposition import CCA
from sklearn.model_selection import KFold, train_test_split
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

from neuralpredictors.layers.encoders.zero_inflation_encoders import ZIGEncoder
from neuralpredictors.measures import zero_inflated_losses
from neuralpredictors.training import LongCycler


def compute_correlations_latents(latents, behaviors):
    """
    Compute the correlation between each latent dimension and each behavior dimension.
    Assumes behaviors is of shape (B, T, 2) and latents is of shape (B, T, n_latents).
    """
    n_latents = latents.shape[2]
    correlations = torch.zeros(
        (n_latents, 2), dtype=torch.float32, device=latents.device
    )  # Two behaviors: pupil, speed

    for i in range(n_latents):
        for j in range(2):
            behavior_data = behaviors[:, :, j].flatten()
            latent_data = latents[:, :, i].flatten()
            correlation = torch.corrcoef(torch.stack((behavior_data, latent_data)))[
                0, 1
            ]
            correlations[i, j] = correlation

    return correlations


def plot_bar(
    correla,
    datakey,
    savedir="plots/cca_future_latent",
    error_bar_speed=None,
    error_bar_pupil=None,
):
    correla_dict = {}
    if isinstance(correla, list):
        correla_dict = {"pupil": correla[0], "speed": correla[1]}
        # labels = ["mouse "+str(i) for i in range(1,6)]
        labels = ["mouse" + mouseid.replace("dynamic", "")[:4] for mouseid in datakey]
        x = np.arange(len(labels))  # the label locations
        width = 0.25  # the width of the bars

        fig, ax = plt.subplots(figsize=(8, 7))
        rects1 = ax.bar(
            x - width / 2,
            correla_dict["pupil"],
            width,
            label="Pupil dilation",
            color="slateblue",
            alpha=0.85,
            yerr=error_bar_pupil,
            capsize=5,
            error_kw={"capthick": 2},
        )
        rects2 = ax.bar(
            x + width / 2,
            correla_dict["speed"],
            width,
            label="Treadmill speed",
            color="darkorange",
            alpha=0.85,
            yerr=error_bar_speed,
            capsize=5,
            error_kw={"capthick": 2},
        )

        # Add some text for labels, title and custom x-axis tick labels, etc.
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.set_ylabel("Average Correlation", fontsize=22)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=45, fontsize=22)
        ax.set_ylim(0, 1)
        ax.tick_params(axis="y", labelsize=22)
        ax.legend(fontsize=22, frameon=False)

        fig.tight_layout()
        plt.savefig(savedir + "/cca_correlation.png")

    else:
        correla_dict["pupil"] = correla[:, 0].cpu().detach().numpy()
        correla_dict["speed"] = correla[:, 1].cpu().detach().numpy()

        indices = np.arange(len(correla_dict["pupil"]))
        combined = list(zip(correla_dict["pupil"], correla_dict["speed"], indices))
        sorted_combined = sorted(combined, key=lambda x: (x[0] + x[1]), reverse=True)
        sorted_behavior1, sorted_behavior2, sorted_indices = zip(*sorted_combined)

        labels = [f"Latent {i+1}" for i in sorted_indices]
        x = np.arange(len(labels))  # the label locations
        width = 0.35  # the width of the bars

        fig, ax = plt.subplots()
        rects1 = ax.bar(
            x - width / 2,
            sorted_behavior1,
            width,
            label="Pupil dilation",
            color="cornflowerblue",
        )
        rects2 = ax.bar(
            x + width / 2,
            sorted_behavior2,
            width,
            label="Treadmill speed",
            color="chocolate",
        )

        # Add some text for labels, title and custom x-axis tick labels, etc.
        ax.set_ylabel("Correlation")
        ax.set_title("Sorted Correlations between Latent Variables and Behaviors")
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=20)
        ax.set_ylim(-1, 1)
        ax.tick_params(axis="y", labelsize=16)
        ax.legend()

        fig.tight_layout()
        plt.savefig(savedir + "/latent_behavior_correlation" + datakey + ".png")


def plot_latents_with_behavior(
    latents,
    behaviors,
    selected_indices,
    behavior_names,
    batch_index=7,
    cca=False,
    savedir="plots/cca_future_latent",
    mice="",
):
    """
    Plots selected latents and their corresponding behaviors.
    """
    color_behavior = "black"  # Change as per your preference
    color_latent = "royalblue"
    time_frames = np.arange(latents.shape[1])
    fig, axes = plt.subplots(2, 1, figsize=(15, 12))  # One subplot for each behavior
    axes = axes.flatten()
    for i in range(2):  # Iterate over each behavior (pupil, speed)
        ax = axes[i]
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        behavior_data = behaviors[batch_index, :, i].cpu().numpy()
        behavior_data = (behavior_data - behavior_data.mean()) / behavior_data.std()
        ax.plot(
            time_frames,
            behavior_data,
            label=f"{behavior_names[i]} Behavior",
            color=color_behavior,
            linewidth=2,
        )

        if cca:
            latent_data = latents[batch_index, :, i].detach().cpu().numpy()
            latent_data = (latent_data - latent_data.mean()) / latent_data.std()
            ax.plot(
                time_frames, latent_data, label="CCA of Latents", color=color_latent
            )
        else:
            label_descriptions = [
                "Max Correlation",
                "2nd Max Correlation",
                "3rd Max Correlation",
            ]
            for j in range(3):
                latent_idx = selected_indices[j, i]
                latent_data = latents[batch_index, :, latent_idx].detach().cpu().numpy()
                latent_data = (latent_data - latent_data.mean()) / latent_data.std()
                ax.plot(
                    time_frames,
                    latent_data,
                    label=f"Latent {latent_idx+1} - {label_descriptions[j]}",
                )

        ax.set_xlabel("Time Frames", fontsize=24)
        ax.set_ylabel("Normalized Behavior/Latent", fontsize=24)
        ax.legend(fontsize=20)
        ax.tick_params(axis="both", which="major", labelsize=20)
        ax.grid(True)

    plt.tight_layout(
        pad=1.0
    )  # Adjust padding to make sure labels and titles do not cut off
    if cca:
        plt.savefig(savedir + "/" + str(batch_index) + "cca_behavior" + mice + ".png")
    else:
        plt.savefig(savedir + "/latent_time_plots" + mice + ".png")


def select_latents_for_plotting(correlations):
    """
    Selects the latents with the highest and lowest correlation for each behavior.
    """
    # Sort indices by correlation for each behavior
    sorted_indices = correlations.abs().argsort(dim=0, descending=True)

    # Extract the top two and bottom two indices for each behavior
    max_indices = sorted_indices[
        :4, :
    ]  # First two rows are the highest and second highest
    # min_indices = sorted_indices[-2:, :]  # Last two rows are the lowest and second lowest

    return max_indices


def perform_cca_analysis(
    latents, behavior, n_folds=5, n_components=1, seed=42, plot_fold=0
):
    if isinstance(latents, torch.Tensor):
        latents = latents.cpu().detach().numpy()
    if isinstance(behavior, torch.Tensor):
        behavior = behavior.cpu().numpy()

    # Define the K-Fold cross-validation
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
    correlations = []

    # Reshape the data
    B, time, n_latents = latents.shape
    latents_reshaped = latents.reshape(B * time, n_latents)
    behavior_reshaped = behavior.flatten()[:, np.newaxis]

    # Standardize features
    scaler_latents = StandardScaler()
    scaler_behavior = StandardScaler()
    latents_scaled = scaler_latents.fit_transform(latents_reshaped)
    behavior_scaled = scaler_behavior.fit_transform(behavior_reshaped)
    # Reshape back to original batch shape for cross-validation along batch dimension
    latents_scaled = latents_scaled  # .reshape(B, time, n_latents)
    behavior_scaled = behavior_scaled  # .reshape(B, time, -1)

    fold_count = 0
    # Cross-validation loop along the batch dimension
    for train_idx, test_idx in kf.split(latents_scaled):
        # Flatten the time dimension for training and testing
        latents_train = latents_scaled[train_idx].reshape(-1, n_latents)
        behavior_train = behavior_scaled[train_idx].reshape(
            -1, behavior_scaled.shape[-1]
        )
        latents_test = latents_scaled[test_idx].reshape(-1, n_latents)
        behavior_test = behavior_scaled[test_idx].reshape(-1, behavior_scaled.shape[-1])

        cca = CCA(n_components=n_components)
        cca.fit(latents_train, behavior_train)

        latents_test_c, behavior_test_c = cca.transform(latents_test, behavior_test)
        correlation = np.corrcoef(latents_test_c[:, 0], behavior_test_c[:, 0])[0, 1]
        correlations.append(correlation)

        if fold_count == plot_fold:
            cca_plot = cca
            test_idx_plot = test_idx

        fold_count += 1

    mean_correlation = np.mean(correlations)
    std_correlation = np.std(correlations)

    return mean_correlation, std_correlation, correlations, cca_plot, test_idx


def cca_analysis(
    model_path,
    encoder_dict,
    decoder_dict,
    save_dir,
    cut=False,
    dropout=True,
    future=False,
    seed=42,
):
    """
    Compute CCA analysis with 5-cross fold validation on given seeds.  We compute the mean correlation and average
    standard deviation of the correlation during the cross validation.
    model_path is path where model is stored
    dropout: If true, masks first half of neuron responses in Encoder input
    future: If true, predicts responses at time t+1 for neuron responses given at time t
    cut: If true, cuts the video down to size 80, if false it keeps whole video length
    """

    set_random_seed(seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    paths = [
        "/mnt/lustre-grete/usr/u11302/Data/dynamic29515-10-12-Video-9b4f6a1a067fe51e15306b9628efea20/",
        "/mnt/lustre-grete/usr/u11302/Data/dynamic29623-4-9-Video-9b4f6a1a067fe51e15306b9628efea20/",
        "/mnt/lustre-grete/usr/u11302/Data/dynamic29712-5-9-Video-9b4f6a1a067fe51e15306b9628efea20/",
        "/mnt/lustre-grete/usr/u11302/Data/dynamic29647-19-8-Video-9b4f6a1a067fe51e15306b9628efea20/",
        "/mnt/lustre-grete/usr/u11302/Data/dynamic29755-2-8-Video-9b4f6a1a067fe51e15306b9628efea20/",
    ]
    print("Loading data..")
    data_loaders = mouse_video_loader(
        paths=paths,
        batch_size=64,
        scale=1,
        max_frame=None,
        frames=80,  # frames has to be > 50. If it fits on your gpu, we recommend 150
        offset=-1,
        include_behavior=True,
        include_pupil_centers=True,
        cuda=device != "cpu",
        to_cut=cut,
    )
    data_loaders_nobehavior = mouse_video_loader(
        paths=paths,
        batch_size=1,
        scale=1,
        max_frame=None,
        frames=80,  # frames has to be > 50. If it fits on your gpu, we recommend 150
        offset=-1,
        include_behavior=False,
        include_pupil_centers=False,
        cuda=device != "cpu",
    )

    ## Load Model

    factorised_3D_core_dict = dict(
        input_channels=1,
        hidden_channels=[32, 64, 128],
        spatial_input_kernel=(11, 11),
        temporal_input_kernel=11,
        spatial_hidden_kernel=(5, 5),
        temporal_hidden_kernel=5,
        stride=1,
        layers=3,
        gamma_input_spatial=10,
        gamma_input_temporal=0.01,
        bias=True,
        hidden_nonlinearities="elu",
        x_shift=0,
        y_shift=0,
        batch_norm=True,
        laplace_padding=None,
        input_regularizer="LaplaceL2norm",
        #     padding=True,
        padding=False,
        final_nonlin=True,
        #     independent_bn_bias=True,
        #     pad_time=False,
        momentum=0.7,
    )
    shifter_dict = dict(
        gamma_shifter=0,
        shift_layers=3,
        input_channels_shifter=2,
        hidden_channels_shifter=5,
    )

    readout_dict = dict(
        bias=False,
        init_mu_range=0.2,
        init_sigma=1.0,
        gamma_readout=0.0,
        gauss_type="full",
        grid_mean_predictor={
            "type": "cortex",
            "input_dimensions": 2,
            "hidden_layers": 1,
            "hidden_features": 30,
            "final_tanh": True,
        },
        # grid_mean_predictor=None,
        share_features=False,
        share_grid=False,
        shared_match_ids=None,
        gamma_grid_dispersion=0.0,
        zig=True,
        out_channels=2,
        kernel_size=(11, 5),
        batch_size=24,
    )
    factorised_3d_model = make_video_model(
        data_loaders_nobehavior,
        seed,
        core_dict=factorised_3D_core_dict,
        core_type="3D_factorised",
        readout_dict=readout_dict.copy(),
        readout_type="gaussian",
        use_gru=False,
        gru_dict=None,
        use_shifter=False,
        shifter_dict=shifter_dict,
        shifter_type="MLP",
        deeplake_ds=False,
    )
    factorised_3d_model
    trainer_fn = "sensorium.training.video_training_loop.standard_trainer"
    trainer_config = {
        "dataloaders": data_loaders,
        "seed": 111,
        "use_wandb": False,
        "wandb_project": "toymodel",
        "wandb_entity": None,
        "wandb_name": "new_file_test",
        "verbose": True,
        "lr_decay_steps": 4,
        "lr_init": 0.0005,
        "device": device,
        "detach_core": False,
        "maximize": True,
        # todo - put this to True if you are using deeplake
        # first connections to deeplake may take up for 10 mins
        "deeplake_ds": False,
        "checkpoint_save_path": "toymodels/",
    }

    # determine maximal number of neurons
    n_neurons = []
    for dataset_name in list(data_loaders["train"].keys()):
        for batch in data_loaders["train"][dataset_name]:
            # reshape the batch to [batch_size*frames,neurons] and apply linear layer to last dimension
            responses = batch.responses
            n_neurons.append(responses.shape[1])
            break
    max_neurons = max(n_neurons)
    encoder_dict["input_dim"] = max_neurons

    # load means and varaince of neurons for Moment fitting
    base_dir = base_dir = "/mnt/lustre-grete/usr/u11302/"
    mean_variance_dict = load_mean_variance(base_dir, device)

    model = ZIGEncoder(
        core=factorised_3d_model.core,
        readout=factorised_3d_model.readout,
        shifter=None,
        k_image_dependent=False,
        loc_image_dependent=False,
        mle_fitting=mean_variance_dict,
        latent=True,
        encoder=encoder_dict,
        decoder=decoder_dict,
        norm_layer="layer_flex",
        non_linearity=True,
        dropout=dropout,
        future_prediction=future,
    ).to(device)

    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()  # For evaluation

    ## Plot model outputs
    batch_ind = 7
    n_iterations = len(LongCycler(data_loaders["oracle"]))
    pupil_cor_mice = []
    speed_cor_mice = []
    pupil_mean_std = []
    speed_mean_std = []
    datakeys = []
    for batch_no, (data_key, data) in tqdm(
        enumerate(LongCycler(data_loaders["oracle"])),
        total=n_iterations,
    ):
        batch_kwargs = data._asdict() if not isinstance(data, dict) else data

        # Plot latents against behavoir
        total_time = batch_kwargs["responses"].shape[
            2
        ]  # total number of time points before core
        latents = model.encoder(
            batch_kwargs["responses"], data_key, model.mice[data_key][0:total_time, :]
        )
        num_latents = latents.shape[2]
        behavior = batch_kwargs["behavior"].permute(0, 2, 1).to(device)

        correlations = compute_correlations_latents(latents, behavior)
        max_indices = select_latents_for_plotting(correlations)

        # Compute CCA of behavior and latents
        means_cor_pupil = []
        stds_cor_pupil = []
        means_cor_speed = []
        stds_cor_speed = []

        for seed in [40, 41, 42, 43, 44]:
            mean_cor_pupil, std_cor_pupil, all_cor_pupil, cca_pupil, test_idx_pupil = (
                perform_cca_analysis(latents, behavior[:, :, 0], seed=seed)
            )
            print(
                "Correlation between the first canonical variable from latents and pupil behavior on test data:",
                all_cor_pupil,
            )
            mean_cor_speed, std_cor_speed, all_cor_speed, cca_speed, test_idx_speed = (
                perform_cca_analysis(latents, behavior[:, :, 1], seed=seed)
            )
            print(
                "Correlation between the first canonical variable from latents and speed behavior on test data:",
                all_cor_speed,
            )
            means_cor_pupil.append(mean_cor_pupil)
            stds_cor_pupil.append(std_cor_pupil)
            means_cor_speed.append(mean_cor_speed)
            stds_cor_speed.append(std_cor_speed)
            # for common_test_idx in  range(10):
            # cca_latent_pupil = torch.einsum('bth,hj->btj', latents.cpu().to(dtype=torch.double),torch.tensor(cca_pupil.x_weights_))
            # cca_latent_speed = torch.einsum('bth,hj->btj', latents.cpu().to(dtype=torch.double),torch.tensor(cca_speed.x_weights_))
            # cca_latents = torch.cat((cca_latent_pupil,cca_latent_speed), dim=2)
            # plot_latents_with_behavior(cca_latents, behavior, None, ['Pupil', 'Speed'],batch_index = common_test_idx, cca=True, mice = data_key,savedir=save_dir)

        # print("Mean Corrrelation of CCA latent and Pupil after cross validation across 5 seeds: ", np.array(means_cor_pupil).mean())
        # print("Pupil, Std of means of cross validation across 5 seeds: ", np.array(means_cor_pupil).std())
        # print("Mean Corrrelation of CCA latent and Speed after cross validation across 5 seeds: ", np.array(means_cor_speed).mean())
        # print("Speed, Std of means of cross validation across 5 seeds: ", np.array(means_cor_speed).std())
        pupil_cor_mice.append(np.array(means_cor_pupil).mean())
        speed_cor_mice.append(np.array(means_cor_speed).mean())
        pupil_mean_std.append(np.array(stds_cor_pupil).mean())
        speed_mean_std.append(np.array(stds_cor_speed).mean())
        datakeys.append(data_key)
        # Plotting
        cca_latent_pupil = torch.einsum(
            "bth,hj->btj",
            latents.cpu().to(dtype=torch.double),
            torch.tensor(cca_pupil.x_weights_),
        )
        cca_latent_speed = torch.einsum(
            "bth,hj->btj",
            latents.cpu().to(dtype=torch.double),
            torch.tensor(cca_speed.x_weights_),
        )
        cca_latents = torch.cat((cca_latent_pupil, cca_latent_speed), dim=2)

        common_test_idx = np.intersect1d(test_idx_pupil, test_idx_speed)[0]
        # plot_latents_with_behavior(latents, behavior, max_indices, ['Pupil', 'Speed'],batch_index = common_test_idx, mice = data_key,savedir=save_dir)
        # plot_bar(correlations,data_key,savedir=save_dir)
        # plot_latents_with_behavior(cca_latents, behavior, None, ['Pupil', 'Speed'],batch_index = common_test_idx, cca=True, mice = data_key,savedir=save_dir)

    print(
        "Average Mean correlation across mice pupil: ", np.array(pupil_cor_mice).mean()
    )
    print(
        "Average Mean correlation across mice speed: ", np.array(speed_cor_mice).mean()
    )
    print("Average std pupil: ", np.array(pupil_mean_std).mean())
    print("Average std correlation speed: ", np.array(speed_mean_std).mean())

    # print("correlation of single latents: ",correlations[:,0].cpu().detach().numpy())
    # print("Canoical Correlation weights: ", cca_pupil.x_weights_.flatten())
    # plot cca for all mice in a bar chart
    plot_bar(
        [pupil_cor_mice, speed_cor_mice],
        datakeys,
        savedir=save_dir,
        error_bar_speed=speed_mean_std,
        error_bar_pupil=pupil_mean_std,
    )


if __name__ == "__main__":
    paths = ["toymodels2/old_testbest.pth"]
    for model_path in paths:
        cut = False
        dropout = True
        future = False
        encoder_dict = {}
        encoder_dict["hidden_dim"] = 42
        encoder_dict["hidden_gru"] = 20
        encoder_dict["output_dim"] = 12
        encoder_dict["hidden_layers"] = 1
        encoder_dict["n_samples"] = 100
        encoder_dict["mice_dim"] = 0  # 18
        encoder_dict["use_cnn"] = False
        encoder_dict["residual"] = False
        encoder_dict["kernel_size"] = [11, 5, 5]
        encoder_dict["channel_size"] = [32, 32, 20]
        encoder_dict["use_resnet"] = False
        encoder_dict["pretrained"] = False

        decoder_dict = {}
        decoder_dict["hidden_layers"] = 1
        decoder_dict["hidden_dim"] = 12
        decoder_dict["use_cnn"] = False
        decoder_dict["kernel_size"] = [5, 11]
        decoder_dict["channel_size"] = [96, 96]

        cca_analysis(
            model_path,
            encoder_dict=encoder_dict,
            decoder_dict=decoder_dict,
            save_dir="plots/cca_nobehav_long",
            cut=cut,
            dropout=dropout,
            future=future,
        )
