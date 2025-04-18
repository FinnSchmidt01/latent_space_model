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
from neuralpredictors.measures.np_functions import corr

from neuralpredictors.layers.encoders.zero_inflation_encoders import ZIGEncoder
from neuralpredictors.measures import zero_inflated_losses
from neuralpredictors.training import LongCycler

group_main = {'dev': {'dynamic29156-11-10-Video-8744edeac3b4d1ce16b680916b5267ce': {
    'ssIew7vsoDwR9wnG3uWk': [40, 109, 130, 131, 150, 176, 268, 290, 547, 670],
    '/FbOh2ePgMqrdXpJC07P': [7,  53,  84, 194, 226, 264, 286, 323, 360, 692],
    'QdH3ZwzfCrhC3hw15J29': [17,  73, 173, 197, 301, 429, 544, 552, 593, 650],
    'MAF2j097Q4BHKMgtWlWe': [58, 168, 221, 413, 436, 555, 602, 609, 639, 663],
    'oIC+bpSFZVqMyXuw7FzU': [5,  51, 135, 183, 225, 285, 287, 394, 619, 689],
    'B1hl1Lfb+NWPIqMXK4Ya': [19,  22, 158, 233, 374, 422, 575, 605, 712, 719]},
    'dynamic29228-2-10-Video-8744edeac3b4d1ce16b680916b5267ce': {
    'aLRPqBKV3Hf5siSSHmeC': [102, 114, 181, 250, 342, 385, 393, 439, 487, 586],
    'A/SBLsELSUsBelyv+BWg': [20,  60,  75, 109, 190, 233, 529, 599, 631, 648],
    'x5IhS8u87xOEcMtLynwD': [101, 211, 212, 298, 313, 360, 372, 390, 468, 520],
    'af1Vnuj0UJIxBh2LduXb': [28,  76,  99, 139, 222, 225, 333, 377, 539, 642],
    'usKAtMHbsh05G8B6aBjK': [24, 136, 320, 329, 365, 423, 496, 504, 552, 574],
    'eBkKTNkWPAxeDI7ALWkE': [110, 133, 184, 339, 530, 548, 572, 595, 637, 644]},
    'dynamic29234-6-9-Video-8744edeac3b4d1ce16b680916b5267ce': {
    'yGA23llYm0Ftzh9bwyy5': [70,  82, 223, 412, 427, 443, 516, 550, 559, 628],
    '0+1ayawyVQ4Akj2DSqbo': [18,  21, 146, 215, 343, 386, 526, 554, 656, 661],
    'edHuKTOVWNPxaMAAhu/O': [7,  49,  77, 176, 208, 242, 262, 297, 331, 636],
    'W98svESnqbEhvA27hl36': [9,  53, 139, 150, 226, 269, 351, 359, 541, 639],
    'RlSN8nU2OBR57BvGKozk': [40,  93, 175, 233, 330, 388, 492, 530, 545, 662],
    'SMu+HtH1qDOoPc4LWCQH': [87,  92, 192, 210, 260, 314, 352, 587, 615, 673]},
    'dynamic29513-3-5-Video-8744edeac3b4d1ce16b680916b5267ce': {
    'uODHIDAmzGEdNIR/7CU+': [4,  19, 141, 180, 191, 198, 263, 338, 461, 526],
    'BWYoVyTZmAAeGb/+FAN5': [131, 149, 231, 241, 291, 345, 418, 440, 570, 653],
    '4f2CDNju0w5mNFOkcHh6': [110, 133, 184, 339, 530, 548, 572, 595, 637, 644],
    'iJDjTZ10zhFV+ILMn1ZP': [89, 146, 193, 273, 353, 355, 443, 544, 560, 613],
    'G1FQEbvKcBqLVy10JNC6': [82, 189, 195, 252, 295, 321, 470, 531, 573, 604],
    'qJyt7U28CfgWpTGF66Lb': [9,  48, 118, 153, 221, 406, 433, 449, 467, 486]},
    'dynamic29514-2-9-Video-8744edeac3b4d1ce16b680916b5267ce': {
    'OUnuqMzSI/k8J4O7ao/+': [7,  53,  84, 194, 226, 264, 286, 323, 360, 691],
    'EN+BEEINDVoYTogjZuSu': [77,  89, 242, 453, 468, 486, 565, 600, 610, 683],
    '97DG676R2J1wTHpad6uq': [4,  20,  50, 162, 260, 355, 481, 591, 671, 685],
    'TC6qD2MF/2yNeQ1lJrXQ': [40, 109, 130, 131, 150, 176, 268, 290, 547, 669],
    '1onnD5tvnWs55ui1fddC': [33,  63, 122, 139, 178, 184, 190, 265, 377, 507],
    '78vrQM1WMAroVSRnSo1k': [13,  90, 146, 175, 205, 460, 482, 664, 686, 695]}},

    'final': {'dynamic29156-11-10-Video-8744edeac3b4d1ce16b680916b5267ce': {
        'CZ+DhtMljSpNJoGC7uee': [74,  79, 114, 116, 227, 231, 298, 415, 551, 641],
        'Jeu3DMpQlmPiqhMW1SKl': [152, 201, 219, 371, 428, 487, 529, 580, 638, 643],
        '760N+YRVyXPlY7zG6Dcs': [33,  63, 122, 139, 178, 184, 190, 265, 377, 507],
        'oIHPXud6r92YFKjXxvz2': [95, 101, 210, 228, 284, 342, 383, 640, 668, 733],
        '9NUudHK3AYroIBWxG5ny': [4,  20,  50, 162, 260, 355, 481, 592, 672, 686],
        '30rqODn0zoFKgVy4rcbr': [34, 213, 246, 257, 350, 378, 423, 466, 612, 705]},
    'dynamic29228-2-10-Video-8744edeac3b4d1ce16b680916b5267ce': {
        'lyuefUNnplbYN+QkTXon': [11,  15,  29,  36, 173, 177, 260, 387, 389, 432],
        '7HuSlLAHKBHL+xMlI/hT': [131, 149, 231, 241, 291, 345, 418, 440, 570, 653],
        'wWJ/Ve5GIXFs9p+FMjYJ': [18, 107, 201, 304, 391, 414, 519, 545, 569, 646],
        'ui1SwoX4WBvQtAA4Kbge': [89, 146, 193, 273, 353, 355, 443, 544, 560, 613],
        'knL1XdNd2VTq5yCQKuYI': [13,  31,  45,  67, 223, 425, 476, 481, 641, 655],
        'F4lMA0Un+T22TkvpduV2': [9,  48, 118, 153, 221, 406, 433, 449, 467, 486]},
    'dynamic29234-6-9-Video-8744edeac3b4d1ce16b680916b5267ce': {
        'Sb2Jjz5IbmYlromle+Qj': [4,  19,  46, 149, 238, 326, 438, 542, 618, 630],
        'GKJOalC+FWQTU8mSxXvB': [178, 197, 202, 232, 309, 450, 459, 490, 593, 678],
        'RKYc2wP07YUs0OgX9o7B': [16,  67, 158, 179, 277, 393, 495, 503, 543, 597],
        'hxYOd5cdaRTPLQRt+Fw0': [5,  47, 123, 166, 207, 261, 263, 360, 566, 633],
        'h3xGHFGClfLgJSyzuoCl': [31,  57, 112, 127, 163, 167, 172, 243, 346, 462],
        'Cn+BFK0m9afMoXg5Rsd5': [37, 100, 118, 119, 138, 161, 246, 266, 498, 616]},
    'dynamic29513-3-5-Video-8744edeac3b4d1ce16b680916b5267ce': {
        'IzeO9dDRaB9s5jEem3NS': [152, 240, 246, 303, 450, 458, 534, 571, 581, 617],
        'ru1ZEKHmUXylYmbkH/wr': [28,  76,  99, 139, 222, 225, 333, 377, 539, 642],
        'Iw9luXyddOe6iA1gViVr': [13,  31,  45,  67, 223, 425, 476, 481, 641, 655],
        '8CNUwkIDKg8J640AIzPQ': [102, 114, 181, 250, 342, 385, 393, 439, 487, 586],
        'f624gHhnXnp8YBXSU2IP': [20,  60,  75, 109, 190, 233, 529, 599, 631, 648],
        'YmfSjn7YgOfV2k6RbHMX': [18, 107, 201, 304, 391, 414, 519, 545, 569, 646]},
    'dynamic29514-2-9-Video-8744edeac3b4d1ce16b680916b5267ce': {
        '6E86vxKi5MnZieARC5KE': [9,  59, 151, 163, 245, 293, 382, 393, 590, 694],
        'IH4V2RFaQAvgYnlqV25d': [43, 102, 193, 254, 359, 424, 541, 578, 594, 719],
        'X4ON5q2WFxEawHsaLiRx': [196, 215, 220, 252, 337, 493, 504, 539, 645, 737],
        '1pZGWbf83r7c60rZ+GWY': [218, 263, 349, 364, 404, 494, 516, 533, 598, 655],
        'EUDYW+miI0Ha9F4KTzFp': [17,  73, 173, 197, 301, 429, 544, 552, 592, 649],
        'IREheNuMTpATrV9/euLP': [19,  22, 158, 233, 374, 422, 575, 604, 711, 718]}}}

def load_data(device, cut=False):
    """
    Load data using the mouse_video_loader.
    """
    paths = [
    "/scratch-grete/projects/nim00012/original_sensorium_2023/dynamic29156-11-10-Video-8744edeac3b4d1ce16b680916b5267ce/",
    "/scratch-grete/projects/nim00012/original_sensorium_2023/dynamic29228-2-10-Video-8744edeac3b4d1ce16b680916b5267ce/",
    "/scratch-grete/projects/nim00012/original_sensorium_2023/dynamic29234-6-9-Video-8744edeac3b4d1ce16b680916b5267ce/",
    "/scratch-grete/projects/nim00012/original_sensorium_2023/dynamic29513-3-5-Video-8744edeac3b4d1ce16b680916b5267ce/",
    "/scratch-grete/projects/nim00012/original_sensorium_2023/dynamic29514-2-9-Video-8744edeac3b4d1ce16b680916b5267ce/",
]

    print("Loading data...")
    data_loaders = mouse_video_loader(
        paths=paths,
        batch_size=1,
        scale=1,
        max_frame=None,
        frames=80,
        offset=-1,
        include_behavior=False,
        include_pupil_centers=False,
        to_cut=cut,
        cuda=device != "cpu",
    )


    cell_coordinates = {}
    data_keys = [
    "dynamic29156-11-10-Video-8744edeac3b4d1ce16b680916b5267ce/",
    "dynamic29228-2-10-Video-8744edeac3b4d1ce16b680916b5267ce/",
    "dynamic29234-6-9-Video-8744edeac3b4d1ce16b680916b5267ce/",
    "dynamic29513-3-5-Video-8744edeac3b4d1ce16b680916b5267ce/",
    "dynamic29514-2-9-Video-8744edeac3b4d1ce16b680916b5267ce/",
]

    for data_key in data_keys:
        cell_coordinates_path = f"/scratch-grete/projects/nim00012/original_sensorium_2023/{data_key}/meta/neurons/cell_motor_coordinates.npy"
        coords = np.load(cell_coordinates_path)
        coords = torch.tensor(coords, device=device, dtype=torch.float32)
        mean_coords = coords.mean(dim=0, keepdim=True)
        std_coords = coords.std(dim=0, keepdim=True)
        cell_coordinates[data_key] = (coords - mean_coords) / std_coords

    return data_loaders, cell_coordinates


def load_model(
    model_path,
    encoder_dict,
    decoder_dict,
    latent,
    flow,
    outchannels,
    dropout,
    dropout_prob,
    device,
    data_loaders,
    data_loaders_nobehavior,
    grid_mean_predictor=None,
    position_mlp=None,
    behavior_mlp=None,
    channels = 128
):
    """
    Load the model using the specified parameters.
    """
    if not isinstance(model_path, str):
        print("Model is not loaded from a path")
        return model_path

    factorised_3D_core_dict = dict(
        input_channels=1,
        hidden_channels=[32, 64, channels],
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
        grid_mean_predictor=(
            {
                "type": "cortex",
                "input_dimensions": 2,
                "hidden_layers": 1,
                "hidden_features": 30,
                "final_tanh": True,
            }
            if grid_mean_predictor
            else None
        ),
        share_features=False,
        share_grid=False,
        shared_match_ids=None,
        gamma_grid_dispersion=0.0,
        zig=True if outchannels == 2 else False,
        out_channels=outchannels,
        kernel_size=(11, 5),
        batch_size=8,
    )
    factorised_3d_model = make_video_model(
        data_loaders_nobehavior,
        seed=42,
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

    # load means and varaince of neurons for Moment fitting
    base_dir = base_dir = "/scratch-grete/projects/nim00012/original_sensorium_2023/"
    mean_variance_dict = load_mean_variance(base_dir, device)

    model = ZIGEncoder(
        core=factorised_3d_model.core,
        readout=factorised_3d_model.readout,
        # shifter = factorised_3d_model.shifter,
        shifter=None,
        k_image_dependent=False,
        loc_image_dependent=False,
        mle_fitting=mean_variance_dict,
        latent=latent,
        encoder=encoder_dict,
        decoder=decoder_dict,
        norm_layer="layer_flex",
        non_linearity=True,
        dropout=dropout,
        dropout_prob=dropout_prob,
        future_prediction=False,
        flow=False,
        position_features=position_mlp,
        behavior_in_encoder=behavior_mlp,
    ).to(device)

    # Compare the keys between the loaded state_dict and the model's state_dict
    loaded_state_dict = torch.load(model_path, map_location=device)

    if out_channels > 1:  # for ZIG
        current_state_dict = model.state_dict()
    else:
        current_state_dict = factorised_3d_model.state_dict()

    loaded_keys = set(loaded_state_dict.keys())
    current_keys = set(current_state_dict.keys())
    for key in loaded_keys:
        if key.startswith("encoder.cnn"):
            print("Loaded key", key)
    for key in current_keys:
        if key.startswith("encoder.cnn"):
            print("Current key", key)

    # Find keys that are in current state_dict but not in loaded state_dict (Missing Keys)
    missing_keys = current_keys - loaded_keys
    # Find keys that are in loaded state_dict but not in current state_dict (Unexpected Keys)
    unexpected_keys = loaded_keys - current_keys

    print("\nMissing keys in loaded state_dict that are in model's state_dict:")
    for key in missing_keys:
        print(key)
        print(current_state_dict[key].shape)

    print("\nUnexpected keys in loaded state_dict that are not in model's state_dict:")
    for key in unexpected_keys:
        print(key)
        print(loaded_state_dict[key].shape)

    if outchannels > 1:  # this is for ZIG/ latent
        model.load_state_dict(torch.load(model_path, map_location=device))
    else:
        factorised_3d_model.load_state_dict(
            torch.load(model_path, map_location=device), strict=False
        )
        model = factorised_3d_model

    return model


# Define the vectorized get_correlations function
def get_correlations(targets, outputs, tier=None):
    """
    Computes single-trial correlation between model predictions and true responses.

    Args:
        targets (torch.Tensor): Shape [repeats, time, neurons].
        outputs (torch.Tensor): Shape [repeats, time, neurons].
        tier (optional): Additional parameter (not used in current implementation).

    Returns:
        np.ndarray: Correlation values with shape [repeats, repeats, number_neurons_half].
            Each element [i, j, k] represents the correlation between target i and output j for neuron k.
    """
    num_targets, time, neurons = targets.shape
    num_outputs = outputs.shape[0]

    number_neurons_half = neurons // 2  # Using half of the neurons

    # Slice to first half of neurons
    targets_sliced = targets[:, :, :number_neurons_half].cpu().numpy()  # Shape: [repeats, time, neurons_half]
    outputs_sliced = outputs[:, :, :number_neurons_half].cpu().numpy()  # Shape: [repeats, time, neurons_half]

    correlations_array = np.empty((num_targets, num_outputs, number_neurons_half))

    for i in range(num_targets):
        for j in range(num_outputs):
            # Compute correlations for all neurons at once
            correlation_values = corr(targets_sliced[i], outputs_sliced[j], axis=0)  # Shape: [neurons_half]
            nan_mask = np.isnan(correlation_values)
            if np.any(nan_mask):
                nan_percentage = nan_mask.mean() * 100
                warnings.warn(
                    f"{nan_percentage:.2f}% NaNs encountered in correlations between target {i} and output {j}. NaNs will be set to zero."
                )
                correlation_values[nan_mask] = 0.0
            correlations_array[i, j, :] = correlation_values

    return correlations_array

# Define the analyze_correlations function
def analyze_correlations(correlation_tensor):
    """
    Analyzes the correlation tensor to compute statistics for diagonal and non-diagonal entries.

    Args:
        correlation_tensor (np.ndarray): Tensor of shape (repeats, repeats, neurons_half).

    Returns:
        dict: Dictionary containing mean and std for diagonal and non-diagonal correlations.
    """
    repeats = correlation_tensor.shape[0]

    # Create a mask for diagonal elements
    diagonal_mask = np.eye(repeats, dtype=bool)

    # Extract diagonal and non-diagonal correlations
    diagonal_correlations = correlation_tensor[:, diagonal_mask].reshape(-1)
    non_diagonal_correlations = correlation_tensor[:, ~diagonal_mask].reshape(-1)

    # Compute statistics
    stats = {
        "diagonal_mean": np.mean(diagonal_correlations),
        "diagonal_std": np.std(diagonal_correlations),
        "non_diagonal_mean": np.mean(non_diagonal_correlations),
        "non_diagonal_std": np.std(non_diagonal_correlations),
    }

    return stats


if __name__ == "__main__":
    device =  "cuda" if torch.cuda.is_available() else "cpu"
    paths = [
            "models/32_core_channels_latent_mice6_10best.pth",
            "models/64_core_channels_latent_mice6_10best.pth",
            "models/64_core_channels_latent_mice6_10_2best.pth",
            "models/128_core_channels_latent_mice6_10best.pth",
            "models/256_core_channels_latent_mice6_10best.pth"    
    ]  #  latent dim is 12
    channels = [32,64,64,128,256]
    data_loaders, cell_coordinates = load_data(device, cut=False)
    print(paths)
    n_neurons = []
    for dataset_name in list(data_loaders["train"].keys()):
        for batch in data_loaders["train"][dataset_name]:
            # reshape the batch to [batch_size*frames,neurons] and apply linear layer to last dimension
            responses = batch.responses
            n_neurons.append(responses.shape[1])
            break
    max_neurons = max(n_neurons)
    
    repeats_dict = group_main["dev"]


    for model_path, i  in zip(paths,channels):

        samples = 100  # number of samples drawn from prior for computing approximate posterior and correlation
        out_channels = (
            2  # number of feature vectors after readout -> 2 for ZIG, 1 for Poisson
        )
        latent = True
        latent_dim = [42, 20, 12]  # for low-dim latent model used in workshop paper
        #latent_dim = [250, 200, 150]  # for high-dim model
        neuron_position_info = True  # False for models without infomation about neurons postion -> important for cortical maps models

        encoder_dict = {
            "input_dim": max_neurons,
            "hidden_dim": latent_dim[0],  # 42
            "hidden_gru": latent_dim[1],
            "output_dim": latent_dim[2],  # 12
            "hidden_layers": 1,
            "n_samples": 100,
            "mice_dim": 0,  # 18
            "use_cnn": False,
            "residual": False,
            "kernel_size": [11, 5, 5],
            "channel_size": [32, 32, 20],
            "use_resnet": False,
            "pretrained": True,
        }

        decoder_dict = {
            "hidden_layers": 1,
            "hidden_dim": latent_dim[2],
            "use_cnn": False,
            "kernel_size": [5, 11],
            "channel_size": [12, 12],
        }
        decoder_dict = None

        # position_mlp = {
        #    "input_size": 3,
        #    "layer_sizes": [6, 12]
        # }
        position_mlp = None

        # Load model
        model = load_model(
            model_path,
            encoder_dict,
            decoder_dict,
            latent,
            flow = False,
            outchannels = 2,
            dropout = True,
            dropout_prob=None,
            device = device,
            data_loaders = data_loaders,
            data_loaders_nobehavior = data_loaders,
            grid_mean_predictor=True,
            position_mlp=None,
            behavior_mlp=None,
            channels = i
        )
    # Put model in eval mode
        model.eval()

        # Wrap entire inference in no_grad context
        with torch.no_grad():
            # Initialize lists to store overall correlations
            

            # Iterate over each mouse in the training set
            diag_mean_average = 0
            off_mean_average = 0
            diag_std_average = 0
            off_std_average = 0
            for mouse in data_loaders['train'].keys():
                # Iterate over each hash and its corresponding neuron IDs
                for hash_key, neuron_ids in repeats_dict[mouse].items():
                    all_correlations = []
                    local_videos = []
                    target_responses = []
                    predicted_responses = []

                    # Load videos and responses for the current hash
                    for id_ in neuron_ids:
                        video_path = f"/scratch-grete/projects/nim00012/original_sensorium_2023/{mouse}/data/videos/{id_}.npy"
                        if os.path.exists(video_path):
                            element = data_loaders['train'][mouse].dataset.__getitem__(id_)  # Returns a named tuple with videos, pupil_center, responses, behavior
                            # element.videos: tensor of shape [channels, frames, height, width]
                            # element.responses: tensor of shape [time, neurons]
                            local_videos.append(element.videos)
                            target_responses.append(element.responses)


                    # Compute the average video across the repeats
                    video = torch.stack(local_videos, dim=0).mean(dim=0).unsqueeze(0)  # Shape: [1, channels, frames, height, width]

                    # Generate predictions for each repeat
                    for responses in target_responses:
                        batch_kwargs = {}
                        batch_kwargs["responses"] = responses.unsqueeze(0)  # Shape: [1, neurons, time]
                        prediction = model(
                            video.to(device),
                            data_key=mouse,
                            out_predicts=True,
                            sample_prior=True,
                            **batch_kwargs,
                        )
                        predicted_responses.append(prediction.cpu())

                    # Ensure that the number of predictions matches the number of targets
                    if len(predicted_responses) != len(target_responses):
                        warnings.warn(f"Mismatch in number of predictions and targets for hash {hash_key} in mouse {mouse}. Skipping.")
                        continue

                    # Convert lists to tensors
                    # Assuming each prediction is a tensor of shape [1, neurons, time]
                    targets_tensor = torch.stack(target_responses)  # Shape: [repeats, neurons, time]
                    predictions_tensor = torch.stack(predicted_responses)  # Shape: [repeats, neurons, time]

                    # Transpose tensors to shape [repeats, time, neurons] for correlation
                    targets_tensor = targets_tensor.permute(0, 2, 1)  # Shape: [repeats, time, neurons]
                    predictions_tensor = predictions_tensor.squeeze(1)  # Shape: [repeats, time, neurons]

                    time_points = predictions_tensor.shape[1]
                    targets_tensor = targets_tensor[:,-time_points:] #chop of first time points

                    # Compute pairwise correlations using the vectorized function
                    correlations = get_correlations(targets_tensor, predictions_tensor)  # Shape: [repeats, repeats, neurons_half]

                    # Append to overall correlations list
                    all_correlations.append(correlations)

                # After processing all mice and hashes, aggregate the correlations
                if not all_correlations:
                    print("No correlations were computed.")
                else:
                    # all_correlations is a list of TENSORS, each shaped [repeats, repeats, neurons]
                    # We'll process them to get per-hash diagonal/off-diagonal stats.

                    diag_list = []  # will hold [2, neurons] for each hash (diagonal mean/std)
                    off_list = []   # will hold [2, neurons] for each hash (off-diagonal mean/std)
                    
                    for correlation_array in all_correlations:
                        # correlation_array: shape [repeats, repeats, neurons]
                        repeats, _, neurons = correlation_array.shape

                        # Extract diagonal elements => shape [repeats, neurons]
                        diag_vals = []
                        for i in range(repeats):
                            diag_vals.append(correlation_array[i, i, :])
                        diag_vals = np.stack(diag_vals, axis=0)  # shape [repeats, neurons]

                        # Mean and std across repeats => shape [neurons] each
                        diag_mean = diag_vals.mean(axis=0)  # shape [neurons]
                        diag_std = diag_vals.std(axis=0)    # shape [neurons]

                        # Combine them => shape [2, neurons]
                        # row 0 = mean per neuron, row 1 = std per neuron
                        diag_tensor = np.stack([diag_mean, diag_std], axis=0)

                        # Extract off-diagonal => shape [repeats*(repeats-1), neurons]
                        off_vals = []
                        for i in range(repeats):
                            for j in range(repeats):
                                if i != j:
                                    off_vals.append(correlation_array[i, j, :])
                        off_vals = np.stack(off_vals, axis=0)  # shape [repeats*(repeats-1), neurons]

                        off_mean = off_vals.mean(axis=0)  # shape [neurons]
                        off_std = off_vals.std(axis=0)    # shape [neurons]
                        off_tensor = np.stack([off_mean, off_std], axis=0)  # shape [2, neurons]

                        diag_list.append(diag_tensor)  # each entry => shape [2, neurons]
                        off_list.append(off_tensor)

                    # Now diag_list and off_list each have length = number of hashes processed
                    # Convert them to np.array => shape [num_hashes, 2, neurons]
                    diag_list = np.stack(diag_list, axis=0)  # shape [num_hashes, 2, neurons]
                    off_list = np.stack(off_list, axis=0)    # shape [num_hashes, 2, neurons]

                    # Transpose to shape [2, neurons, num_hashes]
                    diag_list = diag_list.transpose(1, 2, 0)
                    off_list = off_list.transpose(1, 2, 0)

                    # diag_list[0] => shape [neurons, num_hashes], each element is diagonal MEAN for that neuron/hash
                    # diag_list[1] => shape [neurons, num_hashes], each element is diagonal STD
                    diag_means = diag_list[0]  # shape [neurons, num_hashes]
                    diag_stds  = diag_list[1]

                    # Similarly for off_list
                    off_means = off_list[0]
                    off_stds  = off_list[1]

                    # Compute final "average of means" across neurons & hashes, and "average of stds" across neurons & hashes
                    diag_mean_of_means = diag_means.mean()
                    diag_mean_of_stds  = diag_stds.mean()

                    off_mean_of_means = off_means.mean()
                    off_mean_of_stds  = off_stds.mean()

                    # Print in desired format: "mouse: diag mean(std), off mean(std)"
                    # Because we are in the loop for each 'mouse', you can do:
                    diag_mean_average += diag_mean_of_means
                    off_mean_average += diag_mean_of_stds
                    diag_std_average += off_mean_of_means
                    off_std_average += off_mean_of_stds
                    print(
                        f"{mouse}: diag {diag_mean_of_means:.4f}({diag_mean_of_stds:.4f}), "
                        f"off {off_mean_of_means:.4f}({off_mean_of_stds:.4f})"
                    )

            print(f"Average: diag {diag_mean_average/5:.4f}({diag_std_average/5:.4f}), "
                    f"off {off_mean_average/5:.4f}({off_std_average/5:.4f})"
                )