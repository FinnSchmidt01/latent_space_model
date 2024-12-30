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


def plot_response_histogram(responses, transformed_responses, bins=100):
    # Flatten the tensors to make them suitable for histogram plotting
    responses_flat = responses.cpu().numpy().reshape(-1)
    transformed_responses_flat = (
        transformed_responses.cpu().detach().numpy().reshape(-1)
    )

    # Clip the frequency to limit the impact of dominant bars
    max_frequency = 0.05  # Adjust this value to control the maximum frequency displayed

    fig, ax = plt.subplots(1, 2, figsize=(20, 8))

    # Plot the histogram for the original responses
    counts, bin_edges, _ = ax[0].hist(
        responses_flat, bins=bins, alpha=0.7, label="Original Responses", density=True
    )
    counts = np.clip(counts, 0, max_frequency)  # Clip frequencies to max_frequency
    ax[0].cla()  # Clear the axes
    ax[0].bar(
        bin_edges[:-1],
        counts,
        width=np.diff(bin_edges),
        alpha=0.7,
        label="Original Responses",
        align="edge",
    )
    ax[0].set_title("Histogram of Responses", fontsize=24)
    ax[0].spines["top"].set_visible(False)
    ax[0].spines["right"].set_visible(False)
    ax[0].set_xlabel("Response Value", fontsize=22)
    ax[0].set_ylabel("Density", fontsize=22)
    ax[0].tick_params(axis="y", labelsize=22)
    ax[0].tick_params(axis="x", labelsize=22)
    ax[0].legend(fontsize=22, frameon=False)
    ax[0].set_yscale("log")

    # Plot the histogram for the transformed responses
    counts, bin_edges, _ = ax[1].hist(
        transformed_responses_flat,
        bins=bins,
        alpha=0.7,
        color="g",
        label="Transformed Responses",
        density=True,
    )
    counts = np.clip(counts, 0, max_frequency)  # Clip frequencies to max_frequency
    ax[1].cla()  # Clear the axes
    ax[1].bar(
        bin_edges[:-1],
        counts,
        width=np.diff(bin_edges),
        alpha=0.7,
        color="g",
        label="Transformed Responses",
        align="edge",
    )
    ax[1].set_title("Histogram of Transformed Responses", fontsize=24)
    ax[1].spines["top"].set_visible(False)
    ax[1].spines["right"].set_visible(False)
    ax[1].set_xlabel("Response Value", fontsize=22)
    ax[1].set_ylabel("Density", fontsize=22)
    ax[1].tick_params(axis="y", labelsize=22)
    ax[1].tick_params(axis="x", labelsize=22)
    ax[1].legend(fontsize=22, frameon=False)
    ax[1].set_yscale("log")

    # Save the plot for transformed responses
    plt.savefig("plots/transformed_response_distribution.png")


def plot_response_distriburion(
    model_path,
    outchannels,
    zig,
    latent,
    prior,
    encoder_dict,
    decoder_dict,
    cut=False,
    dropout=True,
    future=False,
    non_linearity=False,
    flow=False,
):
    """
    Use the Flow part of model to print histogram of responses and T(r), where T is the flow and r are responses
    model_path is path where model is stored
    outchannels: gives the number of readouts which are used (2 readouts are neccassary for zig models, 1 for Poisson model)
    zig: if True, uses ZIG-model
    prior: If true, evaluation is done with marginalized latent space
    latent: If true, uses latent space in Model
    dropout: If true, masks first half of neuron responses in Encoder input
    future: If true, predicts responses at time t+1 for neuron responses given at time t
    repeats, samples: give the number of of samples for MC-sampling duriong latent marginalization
    cut: If true, cuts the video down to size 80, if false it keeps whole video length
    n_seeds: Number of differnt seeds for evaluation, deviation between seeds are small since sampling behaves nicely for this model
    dropout_prob: float in (0.25,1), if given masks given percentage of neurons. Can be at least 0.25, since first quarter of neurons has to be masked for evaluation on it
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    seed = 42
    print(device)

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
        batch_size=60,
        scale=1,
        max_frame=None,
        frames=80,  # frames has to be > 50. If it fits on your gpu, we recommend 150
        offset=-1,
        include_behavior=False,
        include_pupil_centers=False,
        to_cut=cut,
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
        share_features=False,
        share_grid=False,
        shared_match_ids=None,
        gamma_grid_dispersion=0.0,
        zig=zig,  # set this to True for ZIG/latent model
        out_channels=outchannels,  # set this to 2 for ZIG/latent model
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
        latent=latent,
        encoder=encoder_dict,
        decoder=decoder_dict,
        non_linearity=non_linearity,
        norm_layer="layer_flex",
        dropout=dropout,
        future_prediction=future,
        flow=flow,
    ).to(device)

    if zig:
        model.load_state_dict(torch.load(model_path, map_location=device), strict=False)
    else:
        factorised_3d_model.load_state_dict(
            torch.load(model_path, map_location=device), strict=False
        )
        model = factorised_3d_model

    if model.flow:
        zif_loss_instance = zero_inflated_losses.ZIFLoss()
        criterion = zif_loss_instance.get_slab_logl
    else:
        zig_loss_instance = zero_inflated_losses.ZIGLoss()
        criterion = zig_loss_instance.get_slab_logl

    model.eval()  # For evaluation

    n_iterations = len(LongCycler(data_loaders["oracle"]))
    batch_count = 0
    batch_loglikelihoods = []
    for batch_no, (data_key, data) in tqdm(
        enumerate(LongCycler(data_loaders["oracle"])),
        total=n_iterations,
    ):
        batch_args = list(data)
        batch_kwargs = data._asdict() if not isinstance(data, dict) else data
        original_data = batch_kwargs["responses"]

        loc = 0.005
        comparison_result = original_data <= loc
        zero_mask = comparison_result.int()

        transformed_responses, _ = model.flow[data_key](
            original_data.permute(0, 2, 1), zero_mask.permute(0, 2, 1)
        )
        # Only keep responses where the zero_mask is False
        valid_responses = original_data[zero_mask == 0]
        # valid_transformed_responses = transformed_responses[zero_mask.permute(0,2,1) == 0]

        plot_response_histogram(valid_responses, transformed_responses)
        break


if __name__ == "__main__":
    out_channels = 2
    zig = True
    model_path = "toymodels2/gamma_flowbest.pth"
    latent = True
    prior = False
    samples = [10]
    repeats = [1]

    encoder_dict = {}
    encoder_dict["hidden_dim"] = 42  # 250
    encoder_dict["hidden_gru"] = 20  # 200
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
    decoder_dict["hidden_dim"] = 12
    decoder_dict["use_cnn"] = False
    decoder_dict["kernel_size"] = [5, 11]
    decoder_dict["channel_size"] = [96, 96]

    plot_response_distriburion(
        model_path,
        out_channels,
        zig,
        latent,
        prior,
        encoder_dict=encoder_dict,
        decoder_dict=decoder_dict,
        cut=False,
        flow=True,
    )
