import os
import random

import matplotlib.pyplot as plt
import numpy as np
import scipy.stats
import torch
import torch.nn as nn
import torch.optim as optim
from neuralpredictors.layers.encoders.zero_inflation_encoders import ZIGEncoder
from neuralpredictors.measures import zero_inflated_losses
from neuralpredictors.training import LongCycler
from nnfabrik.utility.nn_helpers import set_random_seed
from tqdm import tqdm

from moments import load_mean_variance
from sensorium.datasets.mouse_video_loaders import mouse_video_loader
from sensorium.models.make_model import make_video_model

seed = 41  # tested seed 40-43
set_random_seed(seed)


loss_function = "ZIGLoss"
scale_loss = True
detach_core = False
deeplake_ds = False
zig_loss_instance = zero_inflated_losses.ZIGLoss()
criterion = zig_loss_instance.get_slab_logl

device = "cuda" if torch.cuda.is_available() else "cpu"

# Load data
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
    frames=80,
    offset=-1,
    include_behavior=False,
    include_pupil_centers=False,
    cuda=device != "cpu",
    to_cut=False,
)


def compute_correlations(model_output, original_data):
    correlations = np.array(
        [
            scipy.stats.pearsonr(
                model_output[batch_ind, :, n].detach().cpu().numpy(),
                original_data[batch_ind, :, n].detach().cpu().numpy(),
            )[0]
            for n in range(model_output.shape[2])
        ]
    )
    return correlations


# Function to plot and save the data for selected neurons and ZIG heatmaps
def plot_and_save_data_for_neurons_with_prior_samples(
    model,
    batch_args,
    batch_kwargs,
    selected_neurons,
    paths,
    savepath,
    dim,
    samples,
    dec,
    batch_index=0,
):
    # model_output = model(batch_args[0].to(device), **batch_kwargs)
    model_output = model.forward_prior(
        batch_args[0].to(device), n_samples=1, **batch_kwargs
    )
    time_frames = np.arange(model_output.shape[1])
    # Use forward_prior to obtain 100 samples for each time point and neuron
    prior_samples = model.forward_prior(
        batch_args[0].to(device), n_samples=samples, out_predicts=False, **batch_kwargs
    )
    theta_samples = prior_samples[0]  # (batch_size, time, neurons, samples)
    q_samples = prior_samples[3]  # (batch_size, time, neurons, samples)
    k = prior_samples[1]
    loc = prior_samples[2]

    for idx, neuron_idx in enumerate(selected_neurons):
        model_data = model_output[batch_index, :, neuron_idx].detach().cpu().numpy() * 2
        time_left = theta_samples.shape[1]
        original_data_slice = (
            batch_args[1]
            .transpose(2, 1)[batch_index, -time_left:, neuron_idx]
            .detach()
            .cpu()
            .numpy()
        )

        # Compute Pearson correlation coefficient
        correlation, _ = scipy.stats.pearsonr(model_data, original_data_slice)

        fig, ax1 = plt.subplots(figsize=(15, 6))
        ax1.spines["top"].set_visible(False)
        ax1.spines["right"].set_visible(False)
        # Plot Model Output and Actual Data
        ax1.plot(time_frames, model_data, label="Model Output", color="cyan")
        ax1.plot(time_frames, original_data_slice, label="Original Data", color="black")
        # ax1.set_xlabel('Time Frames')
        # ax1.set_ylabel('Responses')
        # ax1.set_title(f'{dim} latent space, decoder {dec}, {samples} samples')
        # ax1.set_title(f'{dim} latent space, {samples} samples, normed')
        # ax1.legend(loc='upper left')
        ax1.set_ylim(bottom=0, top=20)
        ax1.set_yticks([])
        ax1.set_xticks([])
        # Add a second y-axis for the ZIG distribution heatmaps
        ax2 = ax1.twinx()

        # Compute and plot the average ZIG distribution heatmap
        zig_values_avg = []

        # Evaluate ZIG distribution over a range of response values
        response_range = torch.linspace(0, 20, steps=1000).to(device)
        response_values = response_range.unsqueeze(0).repeat(samples, 1)

        # Gather theta and q values for each sample
        theta_slice_samples = theta_samples[batch_index, :, neuron_idx, :].to(device)
        q_slice_samples = q_samples[batch_index, :, neuron_idx, :].to(device)
        k_slice = k[batch_index, :, neuron_idx]
        loc_slice = loc[batch_index, :, neuron_idx]

        zig_distributions_at_t = []
        for value in response_range:
            # Compute ZIG distribution values using the criterion
            # create zero, non zero masks
            comparison_result = value >= loc_slice
            nonzero_mask = comparison_result.int()

            comparison_result = value < loc_slice
            zero_mask = comparison_result.int()
            zig_distribution_values = (
                criterion(
                    theta_slice_samples,
                    k_slice.unsqueeze(-1),
                    loc=loc_slice.unsqueeze(-1),
                    q=q_slice_samples,
                    target=value.unsqueeze(-1).repeat(
                        q_slice_samples.shape[0], samples
                    ),
                    zero_mask=zero_mask.unsqueeze(-1),
                    nonzero_mask=nonzero_mask.unsqueeze(-1),
                )[0]
                .detach()
                .cpu()
                .numpy()
            )

            # Average ZIG values for this value
            zig_values_avg.append(
                torch.tensor(zig_distribution_values, device=device)
                .exp()
                .mean(axis=1)
                .detach()
                .cpu()
                .numpy()
            )

        # zig_values_avg should be transposed to have the correct orientation

        zig_values_avg = np.log(
            np.array(zig_values_avg) + 1e-5
        )  # Shape should be (400, 281) before slicing

        # Slice zig_values_avg to make it compatible with pcolormesh (should be one less in each dimension)
        # zig_values_avg = zig_values_avg[:-1, :-1]  # Shape will be (399, 280)

        # To create the boundary grid for pcolormesh, we need one more point in each dimension than zig_values_avg
        time_edges = np.linspace(
            0, len(time_frames), len(time_frames)
        )  # 281 points for 280 time values
        response_edges = np.linspace(
            response_range.min().cpu().numpy(),
            response_range.max().cpu().numpy(),
            len(response_range),
        )  # 401 points for 400 response values

        # Create the mesh grid for boundaries
        time_mesh, response_mesh = np.meshgrid(time_edges, response_edges)

        # Plot the heatmap using pcolormesh(), which allows each value to be colored based on its coordinate in the mesh
        im = ax2.pcolormesh(
            time_mesh, response_mesh, zig_values_avg, cmap="hot", alpha=0.5
        )

        # Set labels for the second axis
        ax2.set_ylabel("Log-Likelihood", fontsize=22, labelpad=20)
        # ax2.set_xlabel('Time Frames')
        cbar = plt.colorbar(im, ax=ax2, orientation="vertical", pad=0.05)
        cbar.ax.tick_params(labelsize=18)
        ax2.spines["top"].set_visible(False)
        ax2.spines["right"].set_visible(False)
        ax2.set_yticks([])
        plt.grid(True)

        if not os.path.exists(paths[idx]):
            os.makedirs(paths[idx])
        plt.savefig(f"{paths[idx]}/zig_heatmap_neuron_{neuron_idx}_{savepath}.png")
        plt.close()


# Main Evaluation Loop

samples = 200
dim = 12
dec = False
savepath = "12dim_200samples_nodec"


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

# dict containing parameters of 3dconv on top of Gaussian readout
conv_out = {}
conv_out["hidden_channels"] = [4, 2]
conv_out["input_kernel"] = 5
conv_out["hidden_kernel"] = [(11, 11), (15, 15)]


readout_dict = dict(
    bias=False,
    init_mu_range=0.2,
    init_sigma=1.0,
    gamma_readout=0.0,
    gauss_type="full",
    grid_mean_predictor=(
        {
            "type": "cortex",
            "input_dimensions": 2,
            "hidden_layers": 1,
            "hidden_features": 30,
            "final_tanh": True,
        }
        if dec
        else None
    ),
    share_features=False,
    share_grid=False,
    shared_match_ids=None,
    gamma_grid_dispersion=0.0,
    zig=True,
    out_channels=2,
    kernel_size=(11, 5),
    batch_size=8,
    # conv_out = conv_out
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
    use_shifter=False,  # set to True if behavior is included
    shifter_dict=shifter_dict,
    shifter_type="MLP",
    deeplake_ds=False,
)
factorised_3d_model
trainer_fn = "sensorium.training.video_training_loop.standard_trainer"
trainer_config = {
    "dataloaders": data_loaders,
    "seed": 111,
    "use_wandb": True,
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

# load means and varaince of neurons for Moment fitting
base_dir = base_dir = "/mnt/lustre-grete/usr/u11302/Data/"
mean_variance_dict = load_mean_variance(base_dir, device)

# determine maximal number of neurons
n_neurons = []
tier = "train"
for dataset_name in list(data_loaders[tier].keys()):
    for batch in data_loaders[tier][dataset_name]:
        # reshape the batch to [batch_size*frames,neurons] and apply linear layer to last dimension
        responses = batch.responses
        n_neurons.append(responses.shape[1])
        break
max_neurons = max(n_neurons)


dropout = "across_time"
dropout_prob = 0.5
encoder_dict = {}
encoder_dict["input_dim"] = max_neurons
encoder_dict["hidden_dim"] = 42 if dim == 12 else 250
encoder_dict["hidden_gru"] = 20 if dim == 12 else 200
encoder_dict["output_dim"] = dim
encoder_dict["hidden_layers"] = 1
encoder_dict["n_samples"] = 100
encoder_dict["mice_dim"] = 0
encoder_dict["use_cnn"] = False
encoder_dict["residual"] = False
encoder_dict["kernel_size"] = [11, 5, 5]
encoder_dict["channel_size"] = [32, 32, 20]
encoder_dict["use_resnet"] = False
encoder_dict["pretrained"] = True

decoder_dict = {}
decoder_dict["hidden_dim"] = dim
decoder_dict["hidden_layers"] = 1
decoder_dict["use_cnn"] = False
decoder_dict["kernel_size"] = [5, 11]
decoder_dict["channel_size"] = [12, 12]

position_mlp = {}
position_mlp["input_size"] = 3
position_mlp["layer_sizes"] = [6, 12]

behavior_mlp = {}
behavior_mlp["input_size"] = (
    4  # set to 4 if pupil_center should be included as well otherwise to 2
)
behavior_mlp["layer_sizes"] = [4, 6]

model = ZIGEncoder(
    core=factorised_3d_model.core,
    readout=factorised_3d_model.readout,
    # shifter = factorised_3d_model.shifter,
    shifter=None,
    k_image_dependent=False,
    loc_image_dependent=False,
    mle_fitting=mean_variance_dict,
    latent=True,
    encoder=encoder_dict,
    decoder=decoder_dict if dec else None,
    norm_layer="layer_flex",
    non_linearity=True,
    dropout=dropout,
    dropout_prob=dropout_prob,
    future_prediction=False,
    flow=False,
    # position_features = position_mlp,
    # behavior_in_encoder = behavior_mlp
)
if dim == 12 and dec:
    model.load_state_dict(
        torch.load("toymodels2/old_testbest.pth", map_location=device)
    )
elif dim == 150 and dec:
    model.load_state_dict(
        torch.load(
            "models_differentlatent/250_200_150latentbest.pth", map_location=device
        )
    )
elif dim == 150:
    model.load_state_dict(
        torch.load(
            "toymodels2/150dim_no_brain_positions_pretrain_nodecoderbest.pth",
            map_location=device,
        )
    )
elif dim == 12 and (not dec):
    model.load_state_dict(
        torch.load(
            "toymodels2/12dim_no_brain_positions_pretrain_nodecoderbest.pth",
            map_location=device,
        )
    )
else:
    model.load_state_dict(
        torch.load(
            "toymodels2/12dim_no_brain_positions_pretrain_nodecoderbest.pth",
            map_location=device,
        )
    )

model.to(device)

batch_ind = 0
n_iterations = len(LongCycler(data_loaders["oracle"]))

for batch_no, (data_key, data) in tqdm(
    enumerate(LongCycler(data_loaders["oracle"])),
    total=n_iterations,
):
    batch_args = list(data)
    batch_kwargs = data._asdict() if not isinstance(data, dict) else data
    batch_kwargs["data_key"] = (
        data_key  # Adding data_key explicitly to batch_kwargs for easier access
    )

    # Compute correlations
    model_output = model(batch_args[0].to(device), **batch_kwargs)
    time_left = model_output.shape[1]
    original_data = batch_args[1].transpose(2, 1)[:, -time_left:, :].to(device)

    # valid_indices = (torch.sum(original_data, axis=1) > 250)
    # model_output = model_output[:, :, valid_indices.flatten()]
    # original_data = original_data[:, :, valid_indices.flatten()]
    # correlations = compute_correlations(model_output, original_data)

    # Sort indices by correlation, handling NaN values
    # sorted_indices = np.argsort(-np.nan_to_num(correlations))

    # Select neurons for plotting
    # best_idx = sorted_indices[0]
    # second_best_idx = sorted_indices[1]
    # worst_idx = sorted_indices[-1]
    random_idxs = np.random.choice(631, 10, replace=False)

    # Paths for each category
    # paths = [
    #   "plots/latent_nobeh/best",
    #   "plots/latent_nobeh/2ndbest",
    #   "plots/latent_nobeh/worst",
    #   "plots/latent_nobeh/random1",
    #   "plots/latent_nobeh/random2"
    # ]
    paths = ["plots/latent_nobeh/random"] * 40
    # selected_neurons = [best_idx, second_best_idx, worst_idx] + list(random_idxs)
    selected_neurons = random_idxs

    total_activity = (
        original_data[batch_ind].sum(axis=0).detach().cpu().numpy()
    )  # Sum over time, assuming batch_index = 0
    # Find indices of the neurons with the highest and second highest total activity
    sorted_indices_by_activity = np.argsort(-total_activity)  # - for descending order
    selected_neurons = sorted_indices_by_activity[1000:1040]
    # Plot and save the figures with averaged ZIG heatmaps
    plot_and_save_data_for_neurons_with_prior_samples(
        model,
        batch_args,
        batch_kwargs,
        selected_neurons,
        paths,
        savepath,
        dim,
        samples,
        dec,
        batch_index=batch_ind,
    )
    break
