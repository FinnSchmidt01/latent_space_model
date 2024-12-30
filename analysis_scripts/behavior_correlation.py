seed = 40

import os

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from moments import load_mean_variance
from nnfabrik.utility.nn_helpers import set_random_seed
from sensorium.datasets.mouse_video_loaders import mouse_video_loader
from sensorium.models.make_model import make_video_model
from tqdm import tqdm

from neuralpredictors.layers.encoders.zero_inflation_encoders import ZIGEncoder
from neuralpredictors.training import LongCycler

set_random_seed(seed)


def plot_bar(correla, savedir, datakey):
    correla["behavior1"] = correla["behavior1"].cpu().detach().numpy()
    correla["behavior2"] = correla["behavior2"].cpu().detach().numpy()

    indices = np.arange(len(correla["behavior1"]))
    combined = list(zip(correla["behavior1"], correla["behavior2"], indices))
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
    plt.savefig(savedir + "/64b_latent_behavior_correlation" + datakey + ".png")


device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)

save_dir = "./plots/latent_behavior_correlation"


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
)
print("Data loaded")

factorised_3D_core_dict = dict(
    input_channels=3,
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
    use_shifter=True,
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
for dataset_name in list(data_loaders["train"].keys()):
    for batch in data_loaders["train"][dataset_name]:
        # reshape the batch to [batch_size*frames,neurons] and apply linear layer to last dimension
        responses = batch.responses
        n_neurons.append(responses.shape[1])
        break
max_neurons = max(n_neurons)

encoder_dict = {}
encoder_dict["input_dim"] = max_neurons
encoder_dict["hidden_dim"] = 91  # 60
encoder_dict["hidden_gru"] = 19  # 20
encoder_dict["output_dim"] = 12  # 3
encoder_dict["hidden_layers"] = 1
encoder_dict["n_samples"] = 100
encoder_dict["mice_dim"] = 29  # 18

model = ZIGEncoder(
    core=factorised_3d_model.core,
    readout=factorised_3d_model.readout,
    shifter=factorised_3d_model.shifter,
    k_image_dependent=False,
    loc_image_dependent=False,
    mle_fitting=mean_variance_dict,
    latent=True,
    encoder=encoder_dict,
    norm_layer="layer",
)

state_dict = torch.load(
    "toymodels/layer_indv_linear_mice_param_out12final.pth", map_location=device
)
if "encoder.sigma" in state_dict:
    del state_dict["encoder.sigma"]

model.load_state_dict(state_dict)
model.eval()
model.to(device)

n_iterations = len(LongCycler(data_loaders["oracle"]))
correlations = {
    "behavior1": torch.zeros((12, 5), device=device),
    "behavior2": torch.zeros((12, 5), device=device),
}
total_correlations = {
    "behavior1": torch.zeros(12, device=device),
    "behavior2": torch.zeros(12, device=device),
}
count_batches = 0

for batch_no, (data_key, data) in tqdm(
    enumerate(LongCycler(data_loaders["oracle"])),
    total=n_iterations,
):
    batch_args = list(data)
    batch_kwargs = data._asdict() if not isinstance(data, dict) else data

    total_time = batch_kwargs["responses"].shape[
        2
    ]  # total number of time points before core
    latents = model.encoder(
        batch_kwargs["responses"], data_key, model.mice[data_key][0:total_time, :]
    )
    # model_output = model(batch_args[0].to(device), data_key=data_key,out_predicts = True, **batch_kwargs)
    num_latents = latents.shape[2]
    behavior = batch_kwargs["behavior"].permute(0, 2, 1).to(device)

    mouse_correlations = {
        "behavior1": torch.zeros(12, device=device),
        "behavior2": torch.zeros(12, device=device),
    }
    for i in range(num_latents):  # Assuming 12 latent dimensions
        for j in range(2):  # Two behaviors
            behavior_data = behavior[:, :, j]
            latent_data = latents[:, :, i]
            correlation = torch.corrcoef(
                torch.cat((behavior_data.flatten(), latent_data.flatten())).view(2, -1)
            )[0, 1]

            if j == 0:
                correlations["behavior1"][i, count_batches] += correlation
                mouse_correlations["behavior1"][i] += correlation
                total_correlations["behavior1"][i] += correlation
            else:
                correlations["behavior2"][i, count_batches] += correlation
                mouse_correlations["behavior2"][i] += correlation
                total_correlations["behavior2"][i] += correlation
    plot_bar(mouse_correlations, save_dir, data_key)
    count_batches += 1

# for key in total_correlations:
# correlations[key] /= count_batches

x_labels = ["Mouse1", "Mouse2", "Mouse3", "Mouse4", "Mouse5"]
latent_labels = [f"Latent {i+1}" for i in range(12)]

# Create the plot
plt.figure(figsize=(12, 8))
for i in range(9, 12):  # 12 latents
    plt.plot(
        x_labels,
        correlations["behavior1"][i].cpu().detach().numpy(),
        label=latent_labels[i],
    )  # Plot each latent

# Adding labels and title
plt.xlabel("Batch (Mouse)")
plt.ylabel("Correlation")
plt.title("Correlation of Latents Across Batches")
plt.legend(title="Latents", loc="upper right")

# Optional: Set the ylim for better visualization if needed
# plt.ylim(-1, 1)

# Save the plot
plt.savefig(f"{save_dir}/latent_correlations_across_batches.png")


# Plotting
