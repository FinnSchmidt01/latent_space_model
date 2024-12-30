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
from tqdm import tqdm

from neuralpredictors.layers.encoders.zero_inflation_encoders import ZIGEncoder
from neuralpredictors.measures import zero_inflated_losses
from neuralpredictors.training import LongCycler

seed = 41  # tested seed 40-43
set_random_seed(seed)

loss_function = "ZIGLoss"
scale_loss = True
detach_core = False
deeplake_ds = False
zig_loss_instance = zero_inflated_losses.ZIGLoss()
criterion = zig_loss_instance.get_slab_logl


def full_objective(model, dataloader, data_key, *args, **kwargs):
    loss_scale = (
        np.sqrt(len(dataloader[data_key].dataset) / args[0].shape[0])
        if scale_loss
        else 1.0
    )
    # todo - think how to avoid sum in model.core.regularizer()
    if not isinstance(model.core.regularizer(), tuple):
        regularizers = int(
            not detach_core
        ) * model.core.regularizer() + model.readout.regularizer(data_key)
    else:
        regularizers = int(not detach_core) * sum(
            model.core.regularizer()
        ) + model.readout.regularizer(data_key)
    if deeplake_ds:
        for k in kwargs.keys():
            if k not in ["id", "index"]:
                kwargs[k] = torch.Tensor(np.asarray(kwargs[k])).to(device)

    if loss_function == "ZIGLoss":
        # one entry in a tuple corresponds to one paramter of ZIG
        # the output is (theta,k,loc,q)
        model_output = model(
            args[0].to(device), data_key=data_key, out_predicts=False, **kwargs
        )
        theta = model_output[0]
        k = model_output[1]
        loc = model_output[2]
        q = model_output[3]
        time_left = model_output[0].shape[1]

        original_data = args[1].transpose(2, 1)[:, -time_left:, :].to(device)
        # create zero, non zero masks
        comparison_result = original_data >= loc
        nonzero_mask = comparison_result.int()

        comparison_result = original_data <= loc
        zero_mask = comparison_result.int()

        return (
            -1
            * loss_scale
            * criterion(
                theta,
                k,
                loc=loc,
                q=q,
                target=original_data,
                zero_mask=zero_mask,
                nonzero_mask=nonzero_mask,
            )[0].sum()
            + regularizers
        )
    else:
        model_output = model(args[0].to(device), data_key=data_key, **kwargs)
        time_left = model_output.shape[1]

        original_data = args[1].transpose(2, 1)[:, -time_left:, :].to(device)

        return (
            loss_scale
            * criterion(
                model_output,
                original_data,
            )
            + regularizers
        )


## Load data

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
    batch_size=1,
    scale=1,
    max_frame=None,
    frames=80,  # frames has to be > 50. If it fits on your gpu, we recommend 150
    offset=-1,
    include_behavior=False,
    include_pupil_centers=False,
    cuda=device != "cpu",
    to_cut=False,
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
    zig=True,
    out_channels=2,
    kernel_size=(11, 5),
    batch_size=24,
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

encoder_dict = {}
encoder_dict["input_dim"] = max_neurons
encoder_dict["hidden_dim"] = 42
encoder_dict["hidden_gru"] = 20
encoder_dict["output_dim"] = 12  # 3
encoder_dict["hidden_layers"] = 1
encoder_dict["n_samples"] = 100
encoder_dict["mice_dim"] = 0  # 18

decoder_dict = {}
decoder_dict["hidden_dim"] = 12
decoder_dict["hidden_layers"] = 1
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
).to(device)
model.load_state_dict(
    torch.load(
        "toymodels2/nobehavior_pretrain_felxlayer_12pbest.pth", map_location=device
    )
)
# model.load_state_dict(torch.load('toymodels/mle_fittingfinal.pth', map_location=device),strict=False)
# model.load_state_dict(torch.load('toymodels/dec_layerN_indv_linear_mice_29param_out12final.pth', map_location=device),strict=False)
# model.load_state_dict(torch.load('toymodels/layerN_60hidden_18mice_out3final.pth', map_location=device),strict=False)

factorised_3d_model.eval()
model.eval()  # For evaluation

## Helper plotting functions


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


# Function to plot and save the data for selected neurons
def plot_and_save_data_for_neurons(
    model_output,
    original_data,
    selected_neurons,
    paths,
    batch_index=7,
    theta=None,
    q=None,
    latent_theta=None,
    latent_q=None,
    latent=None,
):
    time_frames = np.arange(model_output.shape[1])
    for idx, neuron_idx in enumerate(selected_neurons):
        model_data = (
            10 * model_output[batch_index, :, neuron_idx].detach().cpu().numpy()
        )
        # model_data = (model_data - model_data.mean()) / model_data.std()
        original_data_slice = (
            original_data[batch_index, :, neuron_idx].detach().cpu().numpy()
        )
        # original_data_slice = (original_data_slice - original_data_slice.mean())/original_data_slice.std()

        # Compute Pearson correlation coefficient
        correlation, _ = scipy.stats.pearsonr(model_data, original_data_slice)

        plt.figure(figsize=(12, 6))
        plt.plot(time_frames, model_data, label="Model Output")
        plt.plot(time_frames, original_data_slice, label="Original Data")
        plt.title(
            f"Neuron {neuron_idx} responses - Model vs. Original\nCorrelation: {correlation:.2f}"
        )
        plt.xlabel("Time Frames")
        plt.ylabel("Responses")
        plt.legend()
        plt.grid(True)

        if not os.path.exists(paths[idx]):
            os.makedirs(paths[idx])
        plt.savefig(f"{paths[idx]}/lowdim_neuron_{neuron_idx}.png")
        plt.close()
        if (
            q is not None
        ):  # if model predictions for q and theta are given as well as their latent_features of the neuron
            if len(theta.shape) == 4:
                theta_slice = theta[batch_index, :, neuron_idx, 0]
                q_slice = q[batch_index, :, neuron_idx, 0]
            else:
                theta_slice = theta[batch_index, :, neuron_idx]
                q_slice = q[batch_index, :, neuron_idx]

            if latent is not None:
                theta_feature_slice = latent_theta[:, neuron_idx]
                q_feature_slice = latent_q[:, neuron_idx]
                latent_impact_theta = (
                    torch.einsum(
                        "ij,j->i",
                        latents[batch_index, :, :],
                        latent_theta[:, neuron_idx],
                    )
                    .cpu()
                    .detach()
                    .numpy()
                )  # Latent impact for theta
                latent_impact_q = (
                    torch.einsum(
                        "ij,j->i", latents[batch_index, :, :], latent_q[:, neuron_idx]
                    )
                    .cpu()
                    .detach()
                    .numpy()
                )  # Latent impact for q
            # theta plotting
            plt.figure(figsize=(12, 6))
            plt.plot(time_frames, model_data, label="Model Output", color="blue")
            plt.plot(time_frames, theta_slice, label="Theta", color="red")
            if latent is not None:
                plt.plot(
                    time_frames,
                    latent_impact_theta[-len(time_frames) :],
                    label="Latent Impact on Theta",
                    color="green",
                    linestyle="--",
                )
            plt.title(f"Neuron {neuron_idx} - Model Output, Theta, and Latent Impact")
            plt.xlabel("Time Frames")
            plt.legend()
            plt.grid(True)
            plt.savefig(f"{paths[idx]}/lowdim_neuron_{neuron_idx}_theta.png")
            plt.close()
            # q plotting
            plt.figure(figsize=(12, 6))
            plt.plot(time_frames, model_data, label="Model Output", color="blue")
            plt.plot(time_frames, q_slice, label="q", color="red")
            if latent is not None:
                plt.plot(
                    time_frames,
                    latent_impact_q[-len(time_frames) :],
                    label="Latent Impact on q",
                    color="green",
                    linestyle="--",
                )
            plt.title(f"Neuron {neuron_idx} - Model Output, q, and Latent Impact")
            plt.xlabel("Time Frames")
            plt.legend()
            plt.grid(True)
            plt.savefig(f"{paths[idx]}/lowdim_neuron_{neuron_idx}_q.png")
            plt.close()


def plot_bar(correla, datakey, savedir="plots/latent_behavior_correlation/"):
    correla_dict = {}
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
    rects1 = ax.bar(x - width / 2, sorted_behavior1, width, label="Pupil dilation")
    rects2 = ax.bar(x + width / 2, sorted_behavior2, width, label="Treadmill speed")

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel("Average Correlation")
    ax.set_title("Sorted Correlations between Latent Variables and Behaviors")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45)
    ax.legend()

    fig.tight_layout()
    plt.savefig(savedir + "/latent_behavior_correlation" + datakey + ".png")


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


def plot_latents_with_behavior(
    latents,
    behaviors,
    selected_indices,
    behavior_names,
    batch_index=7,
    cca=False,
    savedir="plots/latent_behavior_correlation/",
):
    """
    Plots selected latents and their corresponding behaviors.
    """
    time_frames = np.arange(latents.shape[1])
    fig, axes = plt.subplots(2, 1, figsize=(15, 12))  # One subplot for each behavior

    for i in range(2):  # Iterate over each behavior (pupil, speed)
        ax = axes[i]
        behaviors[batch_index, :, i] = (
            behaviors[batch_index, :, i] - behaviors[batch_index, :, i].mean()
        )
        behavior_data = 10 * behaviors[batch_index, :, i].cpu().numpy()
        ax.plot(
            time_frames,
            behavior_data,
            label=f"{behavior_names[i]} Behavior",
            color="k",
            linewidth=2,
        )
        if cca:
            print("..")
        else:
            for j in range(
                3
            ):  # Plot the four selected latents (highest, second highest, lowest, second lowest)
                latent_idx = selected_indices[j, i]
                latent_data = latents[batch_index, :, latent_idx].detach().cpu().numpy()
                label_descriptions = [
                    "Max Correlation",
                    "2nd Max Correlation",
                    "3rd Max Correlation",
                ]
                ax.plot(
                    time_frames,
                    latent_data,
                    label=f"Latent {latent_idx+1} - {label_descriptions[j]}",
                )

            ax.set_title(f"{behavior_names[i]} Behavior and Corresponding Latents")
            ax.set_xlabel("Time Frames")
            ax.set_ylabel("Activity/Value")
            ax.legend()
            ax.grid(True)

    plt.tight_layout()
    if cca:
        plt.savefig(savedir + "cca_behavior.png")
    else:
        plt.savefig(savedir + "latent_time_plots.png")


## Plot model outputs
batch_ind = 0
n_iterations = len(LongCycler(data_loaders["oracle"]))

for batch_no, (data_key, data) in tqdm(
    enumerate(LongCycler(data_loaders["oracle"])),
    total=n_iterations,
):
    batch_args = list(data)
    batch_kwargs = data._asdict() if not isinstance(data, dict) else data
    ## Evaluate model
    torch.backends.cudnn.enabled = False
    model_output = model(batch_args[0].to(device), data_key=data_key, **batch_kwargs)
    time_left = model_output.shape[1]
    original_data = batch_args[1].transpose(2, 1)[:, -time_left:, :].to(device)

    model_params = model(
        batch_args[0].to(device), data_key=data_key, out_predicts=False, **batch_kwargs
    )
    theta = model_params[0].detach().cpu().numpy()
    q = model_params[3].detach().cpu().numpy()
    if hasattr(model, "encoder"):
        latent_theta = model.latent_feature[data_key + "_theta"]
        latent_q = model.latent_feature[data_key + "_q"]
        latents = model.encoder(
            batch_kwargs["responses"], data_key, model.mice[data_key][0:80, :]
        )
        # if hasattr(model, 'decoder'):
        # latents = model.decoder(latents)[0]

    # After computing correlations
    valid_indices = torch.sum(original_data, axis=1) > 250
    model_output = model_output[:, :, valid_indices.flatten()]
    original_data = original_data[:, :, valid_indices.flatten()]
    correlations = compute_correlations(model_output, original_data)

    # Sort indices by correlation, handling NaN values
    sorted_indices = np.argsort(-np.nan_to_num(correlations))  # - for descending order

    # Select indices
    best_idx = sorted_indices[0]
    second_best_idx = sorted_indices[1]
    worst_idx = sorted_indices[-1]
    second_worst_idx = sorted_indices[-2]
    random_idxs = np.random.choice(len(correlations), 2, replace=False)

    # Paths for each category
    paths = [
        "plots/latent_nobeh/best",
        "plots/latent_nobeh/2ndbest",
        "plots/latent_nobeh/3rdbest",
        "plots/latent_nobeh/4thbest",
        "plots/latent_nobeh/random1",
        "plots/latent_nobeh/random2",
    ]

    selected_neurons = [
        best_idx,
        second_best_idx,
        sorted_indices[2],
        sorted_indices[3],
    ] + list(random_idxs)

    # Plot and save the figures
    plot_and_save_data_for_neurons(
        model_output, original_data, selected_neurons, paths, batch_index=0
    )
    # plot_and_save_data_for_neurons(model_output, original_data,selected_neurons, paths, batch_index=7)#,theta=theta,q=q)#,latent_theta=latent_theta,latent_q=latent_q,latent = latents)
    # plot_and_save_data_for_neurons(model_output, original_data,[7722,2674,3131,6594], paths, batch_index=7)
    # plot_and_save_data_for_neurons(model_output, original_data,[5857,3640,1987,1867], paths, batch_index=7)#,theta=theta,q=q,latent_theta=latent_theta,latent_q=latent_q,latent = latents)
    # plot_and_save_data_for_neurons(model_output, original_data,[2031,501,1686,7512], paths, batch_index=7)
    # Compute total activity for each neuron
    total_activity = (
        original_data[batch_ind].sum(axis=0).detach().cpu().numpy()
    )  # Sum over time, assuming batch_index = 0

    # Find indices of the neurons with the highest and second highest total activity
    sorted_indices_by_activity = np.argsort(-total_activity)  # - for descending order
    highest_activity_idx = sorted_indices_by_activity[0]
    second_highest_activity_idx = sorted_indices_by_activity[1]

    # Define paths for saving
    activity_labels = [
        "plots/latent_nobeh/highest_activity",
        "plots/nobeh_latent/2nd_highest_activity",
        "plots/nobeh_latent/3rd_highest_activity",
        "plots/nobeh_latent/4th_highest_activity",
    ]
    activity_indices = [
        highest_activity_idx,
        second_highest_activity_idx,
        sorted_indices_by_activity[2],
        sorted_indices_by_activity[3],
    ]

    # Plot and save the data for these neurons
    # plot_and_save_data_for_neurons(model_output, original_data, activity_indices, activity_labels,batch_index = batch_ind,theta=theta,q=q,latent_theta=latent_theta,latent_q=latent_q,latent = latents)
    plot_and_save_data_for_neurons(
        model_output,
        original_data,
        activity_indices,
        activity_labels,
        batch_index=batch_ind,
    )  # ,theta=theta,q=q)

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
    # selected_indices = torch.cat((max_indices, min_indices), dim=0)  # Shape: [2, 2]

    plot_latents_with_behavior(
        latents, behavior, max_indices, ["Pupil", "Speed"], batch_index=batch_ind
    )
    plot_bar(correlations, data_key)
    break
