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
from sensorium.utility.scores import get_correlations


def full_objective(
    model,
    data_key,
    device,
    criterion,
    *args,
    repeats=1,
    n_samples=100,
    flow=False,
    cell_coordinates=None,
    **kwargs,
):

    if model.position_features:
        positions = cell_coordinates[data_key]
    else:
        positions = None

    if model.latent:
        theta, k, loc, q, n_samples = model.forward_prior(
            args[0][:, 0:1].to(device),
            data_key=data_key,
            out_predicts=False,
            repeats=repeats,
            n_samples=n_samples,
            positions=positions,
            **kwargs,
        )
    else:
        theta, k, loc, q = model.forward_prior(
            args[0][:, 0:1].to(device),
            data_key=data_key,
            out_predicts=False,
            repeats=repeats,
            n_samples=n_samples,
            positions=positions,
            **kwargs,
        )

    # the output is (theta,k,loc,q)
    time_left = k.shape[1]  # time points which are left after CNN core

    original_data = args[1].transpose(2, 1)[:, -time_left:, :].to(device)
    # create zero, non zero masks
    comparison_result = original_data >= loc
    nonzero_mask = comparison_result.int()

    comparison_result = original_data <= loc
    zero_mask = comparison_result.int()

    if model.latent:  # log-likelihood for latent model
        k = k.unsqueeze(-1)
        loc = loc.unsqueeze(-1)
        zero_mask = zero_mask.unsqueeze(-1)
        nonzero_mask = nonzero_mask.unsqueeze(-1)
        original_data = original_data.unsqueeze(-1)

        if model.flow:
            if model.flow_base == "Gaussian":
                zig_loss, _ = criterion(
                    model,
                    data_key,
                    targets=original_data,
                    rho=loc,
                    qs=q,
                    means=theta,
                    psi_diag=model.psi[data_key],
                )

            else:
                original_data = original_data.squeeze()
                zero_mask = zero_mask.squeeze()

                original_data, log_det = model.flow[data_key](original_data, zero_mask)

                original_data = original_data.unsqueeze(-1)
                zero_mask = zero_mask.unsqueeze(-1)

                zig_loss = criterion(
                    theta,
                    k,
                    loc=loc,
                    q=q,
                    target=original_data,
                    zero_mask=zero_mask,
                    nonzero_mask=nonzero_mask,
                )[0]
                zig_loss = zig_loss + log_det.unsqueeze(-1)

        else:
            zig_loss = criterion(
                theta,
                k,
                loc=loc,
                q=q,
                target=original_data,
                zero_mask=zero_mask,
                nonzero_mask=nonzero_mask,
            )[0]

        log_likelihood = torch.logsumexp(zig_loss, 3) - torch.log(
            torch.tensor(n_samples, device=device)
        )  # do logsumexp trick, to do MC sampling for log likelihood
        averagedlog_likelihood = (
            log_likelihood.mean() * 1 / torch.log(torch.tensor(2.0, device=device))
        )  # average log likelihodd across time,neuron and batch size

    else:
        log_likelihood = criterion(
            theta,
            k,
            loc=loc,
            q=q,
            target=original_data,
            zero_mask=zero_mask,
            nonzero_mask=nonzero_mask,
        )[0]
        averagedlog_likelihood = (
            log_likelihood.mean() * 1 / torch.log(torch.tensor(2.0, device=device))
        )

    return averagedlog_likelihood.detach().cpu().numpy()


def load_data(device, cut=False):
    """
    Load data using the mouse_video_loader.
    """
    paths = [
        "/mnt/lustre-grete/usr/u11302/Data/dynamic29515-10-12-Video-9b4f6a1a067fe51e15306b9628efea20/",
        "/mnt/lustre-grete/usr/u11302/Data/dynamic29623-4-9-Video-9b4f6a1a067fe51e15306b9628efea20/",
        "/mnt/lustre-grete/usr/u11302/Data/dynamic29712-5-9-Video-9b4f6a1a067fe51e15306b9628efea20/",
        "/mnt/lustre-grete/usr/u11302/Data/dynamic29647-19-8-Video-9b4f6a1a067fe51e15306b9628efea20/",
        "/mnt/lustre-grete/usr/u11302/Data/dynamic29755-2-8-Video-9b4f6a1a067fe51e15306b9628efea20/",
    ]

    print("Loading data...")
    data_loaders = mouse_video_loader(
        paths=paths,
        batch_size=1,
        scale=1,
        max_frame=None,
        frames=80,
        offset=-1,
        include_behavior=True,
        include_pupil_centers=True,
        to_cut=cut,
        cuda=device != "cpu",
    )

    data_loaders_nobehavior = mouse_video_loader(
        paths=paths,
        batch_size=8,
        scale=1,
        max_frame=None,
        frames=80,
        offset=-1,
        include_behavior=False,
        include_pupil_centers=False,
        cuda=device != "cpu",
    )

    cell_coordinates = {}
    data_keys = [
        "dynamic29515-10-12-Video-9b4f6a1a067fe51e15306b9628efea20",
        "dynamic29623-4-9-Video-9b4f6a1a067fe51e15306b9628efea20",
        "dynamic29712-5-9-Video-9b4f6a1a067fe51e15306b9628efea20",
        "dynamic29647-19-8-Video-9b4f6a1a067fe51e15306b9628efea20",
        "dynamic29755-2-8-Video-9b4f6a1a067fe51e15306b9628efea20",
    ]

    for data_key in data_keys:
        cell_coordinates_path = f"/mnt/lustre-grete/usr/u11302/Data/{data_key}/meta/neurons/cell_motor_coordinates.npy"
        coords = np.load(cell_coordinates_path)
        coords = torch.tensor(coords, device=device, dtype=torch.float32)
        mean_coords = coords.mean(dim=0, keepdim=True)
        std_coords = coords.std(dim=0, keepdim=True)
        cell_coordinates[data_key] = (coords - mean_coords) / std_coords

    return data_loaders, data_loaders_nobehavior, cell_coordinates


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
):
    """
    Load the model using the specified parameters.
    """
    if not isinstance(model_path, str):
        print("Model is not loaded from a path")
        return model_path

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
    base_dir = base_dir = "/mnt/lustre-grete/usr/u11302/Data/"
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


def eval_model(
    model_path,
    outchannels,
    latent,
    prior,
    encoder_dict,
    decoder_dict,
    cut=False,
    n_samples=100,
    flow=False,
    device=None,
    dropout=True,
    dropout_prob=None,
    grid_mean_predictor=True,
    position_features=None,
    behavior_in_encoder=None,
):
    """
    Evaluate model, compute its prediction correlation, and log-likelihood.
    model_path: Str, path of saved model weights
    outchannels: Int,  Number of feature vectors of core-readout, should be two for ZIG, one for Poisson
    latent: Boolean, If true uses latent
    prior: Boolean, if true prior correlation is computed else conditoned correlation is computed
    encoder_dict: Dict, contains all information about encoder architecture
    decoder_dict: Dict, contains all information about decoder architecture
    cut: Boolean, If True cut video size down to training size (roughly 80) from original (roughly 300)
    n_samples: Int, determines how many samples from prior are drawn to compute log-likelihood is computed
    flow: Boolean, if True appends a flow to the responses
    dropout: Boolean, if True masks half of neuron responses if False neurons are not masked
    dropout_prob: float in (0.25,1), if None simply first half of neurons is masked else given portion of neurons is masked
    grid_mean_predictor: Boolean, If true applies postion based grid_mean_predictor to neurons in Readout, if False predictor is None
    position_features: Dict, if given computes latent feature vector from cortical position and dict contains information about MLP
                             if None feature vectors are model parameters
    behavior_in_encoder: Dict, if given applies behavior as additional channels to encoder input, proccesses them with a MLP first
                               if None behavior is not included at all
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    set_random_seed(42)

    # load data with and without behavior
    data_loaders, data_loaders_nobehavior, cell_coordinates = load_data(device, cut)

    # determine maximal number of neurons for input dim of encoder
    n_neurons = []
    for dataset_name in list(data_loaders["train"].keys()):
        for batch in data_loaders["train"][dataset_name]:
            # reshape the batch to [batch_size*frames,neurons] and apply linear layer to last dimension
            responses = batch.responses
            n_neurons.append(responses.shape[1])
            break
    max_neurons = max(n_neurons)
    encoder_dict["input_dim"] = max_neurons

    # Load model
    model = load_model(
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
        grid_mean_predictor=grid_mean_predictor,
        position_mlp=position_features,
        behavior_mlp=behavior_in_encoder,
    )

    log_likelihoods = []
    # Get log likelihood function. It is either zero-inflated Gaussian for a Gaussian flow or zero-inflated Gamma in any other case
    if model.flow:
        if model.flow_base == "Gaussian":
            zif_loss_instance = zero_inflated_losses.ZIFLoss()
            criterion = zif_loss_instance.get_slab_logl
        else:
            zig_loss_instance = zero_inflated_losses.ZIGLoss()
            criterion = zig_loss_instance.get_slab_logl
    else:
        zig_loss_instance = zero_inflated_losses.ZIGLoss()
        criterion = zig_loss_instance.get_slab_logl

    model.eval()

    # Compute correlations
    with torch.no_grad():
        correlation = get_correlations(
            model,
            data_loaders["oracle"],
            device=device,
            as_dict=False,
            per_neuron=False,
            deeplake_ds=False,
            forward_prior=prior,
            n_samples=n_samples,
            dropout_prob=dropout_prob,
            flow=model.flow,
            cell_coordinates=cell_coordinates if model.position_features else None,
        )

        n_iterations = len(LongCycler(data_loaders["oracle"]))
        batch_count = 0
        batch_loglikelihoods = []

        for batch_no, (data_key, data) in tqdm(
            enumerate(LongCycler(data_loaders["oracle"])),
            total=n_iterations,
        ):
            batch_args = list(data)
            batch_kwargs = data._asdict() if not isinstance(data, dict) else data

            if outchannels == 1:  # Poisson model doesn't have a likelihood
                log_likelihood = 0
            else:
                log_likelihood = full_objective(
                    model,
                    data_key,
                    device,
                    criterion,
                    *batch_args,
                    n_samples=n_samples,
                    flow=flow,
                    cell_coordinates=(
                        cell_coordinates if model.position_features else None
                    ),
                    **batch_kwargs,
                )
            batch_loglikelihoods.append(log_likelihood)
            batch_count += 1

    return np.mean(batch_loglikelihoods), correlation


if __name__ == "__main__":
    # set decoder_dict to None for model with neuron postions and neuron_position_info to False, and use position_mlp dict
    # paths = ['models/[6,12]dim_no_brain_positions_pretrain_nodecoderbest.pth'] #model which assign latent feature vectors based on position
    # paths = ['models/baseline_sensorium_nobehaviorbest.pth'] #Poisson Model
    # paths = ['models/zig_best.pth'] #Pure ZIG model, no latent
    # paths = ['models/latent_12dimbest.pth'] #latent model as in workshop paper, latent dim is 12
    paths = ["models/250_200_200latentbest.pth"]

    for model_path in paths:

        samples = 100  # number of samples drawn from prior for computing approximate posterior and correlation
        out_channels = (
            2  # number of feature vectors after readout -> 2 for ZIG, 1 for Poisson
        )
        latent = True
        # latent_dim = [42,20,12] #for low-dim latent model used in workshop paper
        latent_dim = [250, 200, 200]  # for high-dim model
        neuron_position_info = True  # False for models without infomation about neurons postion -> important for cortical maps models

        encoder_dict = {
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
        # decoder_dict = None

        # position_mlp = {
        #    "input_size": 3,
        #    "layer_sizes": [6, 12]
        # }
        position_mlp = None

        # behavior_mlp = {
        #    "input_size": 4,  # Set to 4 if pupil_center should be included, otherwise set to 2
        #    "layer_sizes": [4, 6]
        # }

        correlations = []

        for prior in [True, False]:
            # If latent is not used, we need to compute correlation only once as there is no conditioned correlation
            if (not latent) and prior:
                continue

            log_likelihood, correlation = eval_model(
                model_path,
                out_channels,
                latent,
                prior,
                encoder_dict=encoder_dict,
                decoder_dict=decoder_dict,
                cut=False,
                n_samples=samples,
                flow=False,
                dropout_prob=None,
                grid_mean_predictor=neuron_position_info,
                position_features=position_mlp,
                behavior_in_encoder=None,
            )

            correlations.append(correlation)

        if out_channels > 1:
            print("Log_likelihood", log_likelihood)

        print("Prior Correlation", correlations[0])
        if latent:
            print("Conditioned Correlation", correlations[1])
