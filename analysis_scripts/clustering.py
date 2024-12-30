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
from sensorium.utility.scores import get_correlations
from tqdm import tqdm

from neuralpredictors.layers.encoders.zero_inflation_encoders import ZIGEncoder
from neuralpredictors.measures import zero_inflated_losses
from neuralpredictors.training import LongCycler


def clustering(
    model_path,
    encoder_dict,
    decoder_dict,
    future=False,
    flow=False,
    position_features=None,
    behavior_in_encoder=None,
):
    """
    evaluate model, compute its prediction correlation and log likelihood.
    model_path is path where model is stored
    flow: If true appplies flow model
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(device)
    seed = 42

    set_random_seed(seed)
    loss_function = "ZIGLoss"
    scale_loss = True
    detach_core = False
    deeplake_ds = False

    ## Load data

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
        batch_size=1,
        scale=1,
        max_frame=None,
        frames=80,  # frames has to be > 50. If it fits on your gpu, we recommend 150
        offset=-1,
        include_behavior=False,
        include_pupil_centers=False,
        to_cut=False,
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
        # grid_mean_predictor={
        #'type': 'cortex',
        #'input_dimensions': 2,
        #'hidden_layers': 1,
        #'hidden_features': 30,
        #'final_tanh': True
        # },
        grid_mean_predictor=None,
        share_features=False,
        share_grid=False,
        shared_match_ids=None,
        gamma_grid_dispersion=0.0,
        zig=True,  # set this to True for ZIG/latent model
        out_channels=2,  # set this to 2 for ZIG/latent model
        kernel_size=(11, 5),
        batch_size=8,
    )
    factorised_3d_model = make_video_model(
        data_loaders,
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
    base_dir = base_dir = "/mnt/lustre-grete/usr/u11302/Data/"
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
        non_linearity=True,
        norm_layer="layer_flex",
        flow=flow,
        position_features=position_features,
        behavior_in_encoder=behavior_in_encoder,
    ).to(device)

    model.load_state_dict(torch.load(model_path, map_location=device))

    model.eval()  # For evaluation

    n_iterations = len(LongCycler(data_loaders["oracle"]))
    batch_count = 0
    batch_loglikelihoods = []
    cell_coordinates = {}
    cell_coordinates_normed = {}
    mouse_number = 0
    for batch_no, (data_key, data) in tqdm(
        enumerate(LongCycler(data_loaders["oracle"])),
        total=n_iterations,
    ):
        batch_args = list(data)
        batch_kwargs = data._asdict() if not isinstance(data, dict) else data
        cell_coordinates_path = (
            "/mnt/lustre-grete/usr/u11302/Data/"
            + data_key
            + "/meta/neurons/cell_motor_coordinates.npy"
        )
        cell_coordinates[data_key] = np.load(cell_coordinates_path)
        cell_coordinates[data_key] = torch.tensor(
            cell_coordinates[data_key], device=device, dtype=torch.float32
        )
        mean_coords = cell_coordinates[data_key].mean(
            dim=0, keepdim=True
        )  # Mean value for each dimension xyz
        std_coords = cell_coordinates[data_key].std(
            dim=0, keepdim=True
        )  # Standard deviation for each dimension xyz
        cell_coordinates_normed[data_key] = (
            cell_coordinates[data_key] - mean_coords
        ) / std_coords

        if position_features:
            feature_q = model.mlp_q[data_key](
                cell_coordinates_normed[data_key]
            ).permute(1, 0)
            feature_theta = model.mlp_theta[data_key](
                cell_coordinates_normed[data_key]
            ).permute(1, 0)
        else:
            feature_q = model.latent_feature[data_key + "_q"]
            feature_theta = model.latent_feature[data_key + "_theta"]

        feature_q = (feature_q - feature_q.mean(dim=0, keepdim=True)) / feature_q.std(
            dim=0, keepdim=True
        )
        feature_theta = (
            feature_theta - feature_theta.mean(dim=0, keepdim=True)
        ) / feature_theta.std(dim=0, keepdim=True)

        coordinates = cell_coordinates[data_key].cpu().numpy()
        z_coords = coordinates[:, 2]

        # Iterate over each specified z-coordinate
        specific_z_coordinates = [200, 300, 400]
        for z_value in specific_z_coordinates:
            neuron_indices = np.where(z_coords == z_value)[0]

            if len(neuron_indices) == 0:
                continue

            filtered_coordinates = coordinates[neuron_indices]
            filtered_feature_q = feature_q[:, neuron_indices]
            filtered_feature_theta = feature_theta[:, neuron_indices]

            # Perform SVD on filtered features
            u_q, s_q, vh_q = torch.svd(filtered_feature_q)
            u_theta, s_theta, vh_theta = torch.svd(filtered_feature_theta)

            mouse_plot_dir = f"plots/clustering/mouse_{data_key}/z_{z_value}/"
            os.makedirs(mouse_plot_dir, exist_ok=True)

            # Plot 10 biggest singular value vectors for feature_q and feature_theta
            for i in range(10):
                feature_q_values = (
                    torch.matmul(u_q[:, i], filtered_feature_q).detach().cpu().numpy()
                )
                feature_theta_values = (
                    torch.matmul(u_theta[:, i], filtered_feature_theta)
                    .detach()
                    .cpu()
                    .numpy()
                )

                if z_value == 400:
                    feature_q_values = feature_q_values * -1
                    feature_theta_values = feature_theta_values * -1

                fig_q = plt.figure()
                ax_q = fig_q.add_subplot(111)
                sc_q = ax_q.scatter(
                    filtered_coordinates[:, 0],
                    filtered_coordinates[:, 1],
                    c=feature_q_values,
                    cmap="viridis",
                    s=5,
                )
                ax_q.set_title(f"SVD Feature Q Singular Vector {i+1} (z={z_value})")
                ax_q.set_xlabel("X Coordinate")
                ax_q.set_ylabel("Y Coordinate")
                plt.colorbar(sc_q)
                plt.savefig(
                    f"{mouse_plot_dir}/svd_feature_q_vector_{i+1}_mouse_{data_key}_z_{z_value}.png"
                )
                plt.close(fig_q)

                fig_theta = plt.figure()
                ax_theta = fig_theta.add_subplot(111)
                sc_theta = ax_theta.scatter(
                    filtered_coordinates[:, 0],
                    filtered_coordinates[:, 1],
                    c=feature_theta_values,
                    cmap="viridis",
                    s=5,
                )
                ax_theta.set_title(
                    f"SVD Feature Theta Singular Vector {i+1} (z={z_value})"
                )
                ax_theta.set_xlabel("X Coordinate")
                ax_theta.set_ylabel("Y Coordinate")
                plt.colorbar(sc_theta)
                plt.savefig(
                    f"{mouse_plot_dir}/svd_feature_theta_vector_{i+1}_mouse_{data_key}_z_{z_value}.png"
                )
                plt.close(fig_theta)

            # Plot explained variance
            explained_variance_q = torch.cumsum(s_q**2, dim=0) / torch.sum(s_q**2)
            explained_variance_theta = torch.cumsum(s_theta**2, dim=0) / torch.sum(
                s_theta**2
            )

            # Random matrices and their SVD
            random_matrix_q = torch.empty_like(filtered_feature_q).uniform_(-1, 1)
            random_matrix_theta = torch.empty_like(filtered_feature_theta).uniform_(
                -1, 1
            )
            _, s_random_q, _ = torch.linalg.svd(random_matrix_q)
            _, s_random_theta, _ = torch.linalg.svd(random_matrix_theta)
            explained_variance_random_q = torch.cumsum(
                s_random_q**2, dim=0
            ) / torch.sum(s_random_q**2)
            explained_variance_random_theta = torch.cumsum(
                s_random_theta**2, dim=0
            ) / torch.sum(s_random_theta**2)

            # Plot cumulative explained variance
            fig_var = plt.figure()
            plt.plot(
                np.arange(1, len(explained_variance_q) + 1),
                explained_variance_q.detach().cpu().numpy(),
                label="Feature Q",
            )
            plt.plot(
                np.arange(1, len(explained_variance_theta) + 1),
                explained_variance_theta.detach().cpu().numpy(),
                label="Feature Theta",
            )
            plt.plot(
                np.arange(1, len(explained_variance_random_q) + 1),
                explained_variance_random_q.detach().cpu().numpy(),
                linestyle="dashed",
                label="Random Q",
            )
            plt.plot(
                np.arange(1, len(explained_variance_random_theta) + 1),
                explained_variance_random_theta.detach().cpu().numpy(),
                linestyle="dashed",
                label="Random Theta",
            )
            plt.xlabel("Number of Singular Values")
            plt.ylabel("Cumulative Explained Variance")
            plt.legend(loc="upper right", fontsize=12)
            plt.savefig(
                f"{mouse_plot_dir}/explained_variance_mouse_{data_key}_z_{z_value}.png"
            )
            plt.close(fig_var)

        mouse_number += 1
        if mouse_number > 5:
            break


if __name__ == "__main__":
    # model_path = 'toymodels2/nobehavior_pretrain_felxlayer_12pbest.pth'
    # model_path = 'models_differentlatent/250_200_150latentbest.pth'
    # mlp_dict = {'input_size': 3, 'layer_sizes': [6, 12]}
    # model_path = f"toymodels2/{str(mlp_dict)}_position_featuresbest.pth"
    model_path = "toymodels2/12dim_no_brain_positions_pretrain_nodecoderbest.pth"
    # model_path = 'toymodels2/[6,12]dim_no_brain_positions_pretrain_nodecoderbest.pth'

    encoder_dict = {}
    encoder_dict["hidden_dim"] = 42  # 42
    encoder_dict["hidden_gru"] = 20  # 20
    encoder_dict["output_dim"] = 12  # 12
    encoder_dict["hidden_layers"] = 1
    encoder_dict["n_samples"] = 100
    encoder_dict["mice_dim"] = 0  # 18
    encoder_dict["use_cnn"] = False
    encoder_dict["residual"] = False
    encoder_dict["kernel_size"] = [11, 5, 5]
    encoder_dict["channel_size"] = [32, 32, 20]
    encoder_dict["use_resnet"] = False
    encoder_dict["pretrained"] = True

    decoder_dict = {}
    decoder_dict["hidden_layers"] = 1
    decoder_dict["hidden_dim"] = 150  # 12
    decoder_dict["use_cnn"] = False
    decoder_dict["kernel_size"] = [5, 11]
    decoder_dict["channel_size"] = [96, 96]
    decoder_dict = None

    position_mlp = {}
    position_mlp["input_size"] = 3
    position_mlp["layer_sizes"] = [6, 12]

    behavior_mlp = {}
    behavior_mlp["input_size"] = (
        4  # set to 4 if pupil_center should be included as well otherwise to 2
    )
    behavior_mlp["layer_sizes"] = [4, 6]

    clustering(
        model_path,
        encoder_dict,
        None,
        future=False,
        flow=False,
        position_features=None,
        behavior_in_encoder=None,
    )
