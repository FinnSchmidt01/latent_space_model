import os
import random

import matplotlib.pyplot as plt
import numpy as np
import scipy.stats
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from neuralpredictors.layers.encoders.zero_inflation_encoders import ZIGEncoder
from neuralpredictors.measures import zero_inflated_losses
from neuralpredictors.training import LongCycler
from nnfabrik.utility.nn_helpers import set_random_seed
from tqdm import tqdm

from moments import load_mean_variance
from sensorium.datasets.mouse_video_loaders import mouse_video_loader
from sensorium.models.make_model import make_video_model
from sensorium.utility.scores import get_correlations

from sklearn.model_selection import KFold
from sklearn.cross_decomposition import CCA

import numpy as np
import numpy as np
from scipy.linalg import block_diag

def sample_block_diag_gp_independently(
    num_blocks=281,
    block_size=281,
    lengthscale=10.0,
    variance=1.0,
    target_mean=0.8,
    target_std=0.4,
    min_val=0.05,
    random_state=None
):
    """
    Generates data for 'num_blocks' independent GPs of length 'block_size'.
    Each block is an RBF kernel with the given lengthscale/variance.
    Then shift/scale/clip each independently and concatenate into a single array.
    """
    rng = np.random.default_rng(random_state)

    y_list = []
    for b in range(num_blocks):
        # Build a single 281x281 RBF kernel
        t = np.arange(block_size)
        dist_sq = (t[:, None] - t[None, :]) ** 2
        K_block = variance * np.exp(-0.5 * dist_sq / (lengthscale ** 2))

        # Sample from N(0, K_block)
        y_block = rng.multivariate_normal(mean=np.zeros(block_size), cov=K_block)

        # Shift & scale
        y_block -= y_block.mean()
        current_std = y_block.std(ddof=1)
        if current_std > 1e-12:
            y_block *= (target_std / current_std)
        y_block += target_mean

        # Clip
        y_block = np.maximum(min_val, y_block)

        y_list.append(y_block)

    # Concatenate => shape [num_blocks * block_size,]
    y_full = np.concatenate(y_list, axis=0)
    return y_full



##############################################
# 1) CREATE DATA LOADERS & LOAD COORDINATES #
##############################################

def create_data_loader_and_coords(paths, device="cpu"):
    """
    1) Creates the data_loaders via mouse_video_loader.
    2) Iterates once through 'oracle' to discover data_keys and load cell coordinates for each.
    
    Returns:
        data_loaders (dict): containing 'train', 'oracle', etc.
        cell_coordinates (dict): { data_key: torch.Tensor([num_neurons, 3]) }
    """
    print("Creating data loaders..")
    data_loaders = mouse_video_loader(
        paths=paths,
        batch_size=1,
        scale=1,
        max_frame=None,
        frames=80,  # frames must be > 50 if GPU can handle it
        offset=-1,
        include_behavior=False,
        include_pupil_centers=False,
        to_cut=False,
        cuda=(device != "cpu"),
    )

    print("Loading neuron coordinates..")
    cell_coordinates = {}
    loader_oracle = data_loaders["oracle"]
    
    # We only do one pass to read out the data_key for each path.
    n_iterations = len(LongCycler(loader_oracle))
    
    for _, (data_key, data) in enumerate(LongCycler(loader_oracle)):
        # Convert to dict-like
        batch_kwargs = data._asdict() if not isinstance(data, dict) else data

        # Build the path for coordinates
        cell_coordinates_path = os.path.join(
            "/scratch-grete/projects/nim00012/original_sensorium_2023/",
            data_key,
            "meta/neurons/cell_motor_coordinates.npy"
        )
        
        coords = np.load(cell_coordinates_path)  # shape [num_neurons, 3]
        coords_torch = torch.tensor(coords, device=device, dtype=torch.float32)
        cell_coordinates[data_key] = coords_torch
    
    return data_loaders, cell_coordinates


#################################
# 2) LOAD MODEL (ZIG or 3D Net) #
#################################

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
    grid_mean_predictor=True,
    position_mlp=None,
    behavior_mlp=None,
):
    """
    Load the model using the specified parameters.
    Returns either a ZIGEncoder or the base factorised_3d_model.
    """
    n_neurons = []
    for dataset_name in list(data_loaders["train"].keys()):
        for batch in data_loaders["train"][dataset_name]:
            # reshape the batch to [batch_size*frames,neurons] and apply linear layer to last dimension
            responses = batch.responses
            n_neurons.append(responses.shape[1])
            break
    max_neurons = max(n_neurons)
    encoder_dict["input_dim"] = max_neurons
    

    if not isinstance(model_path, str):
        print("Model is not loaded from a path; returning the given model object.")
        return model_path

    # Example core dict
    factorised_3D_core_dict = dict(
        input_channels=1,
        hidden_channels=[32, 64, 16],
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
        padding=False,
        final_nonlin=True,
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
        zig=(outchannels == 2),  # If outchannels=2 => ZIG
        out_channels=outchannels,
        kernel_size=(11, 5),
        batch_size=8,
    )

    # Build the base model
    factorised_3d_model = make_video_model(
        data_loaders_nobehavior,
        seed=42,
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

    # Load means & variance for MLE
    base_dir = "/scratch-grete/projects/nim00012/original_sensorium_2023/"
    mean_variance_dict = load_mean_variance(base_dir, device)

    # Construct the ZIGEncoder
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
        norm_layer="layer_flex",
        non_linearity=True,
        dropout=dropout,
        dropout_prob=dropout_prob,
        future_prediction=False,
        flow=flow,
        position_features=position_mlp,
        behavior_in_encoder=behavior_mlp,
    ).to(device)

    # Load weights
    loaded_state_dict = torch.load(model_path, map_location=device)
    if outchannels > 1:
        current_state_dict = model.state_dict()
    else:
        current_state_dict = factorised_3d_model.state_dict()

    # Check missing/unexpected keys for debug
    loaded_keys = set(loaded_state_dict.keys())
    current_keys = set(current_state_dict.keys())
    missing_keys = current_keys - loaded_keys
    unexpected_keys = loaded_keys - current_keys

    print("\nMissing keys in loaded state_dict that are in model's state_dict:")
    for key in missing_keys:
        print("  ", key)
    print("\nUnexpected keys in loaded state_dict that are not in model's state_dict:")
    for key in unexpected_keys:
        print("  ", key)

    # Finally load
    if outchannels > 1:
        model.load_state_dict(loaded_state_dict)
    else:
        factorised_3d_model.load_state_dict(loaded_state_dict, strict=False)
        model = factorised_3d_model

    return model


##################################
# 3) COMPUTE LATENTS PER BATCH  #
##################################

def compute_latents(model, data_loaders):
    """
    Iterate over the 'oracle' loader, compute latents for each batch
    using model.encoder(...).

    Returns a dict keyed by data_key:
      latents_dict[data_key] = list of latents (one for each batch).
    
    NOTE: Adjust indexing and shapes to match your data layout!
    """
    latents_dict = {}
    loader_oracle = data_loaders["oracle"]
    n_iterations = len(LongCycler(loader_oracle))

    for batch_no, (data_key, data) in tqdm(
        enumerate(LongCycler(loader_oracle)),
        total=n_iterations,
        desc="Computing latents",
    ):
        batch_kwargs = data._asdict() if not isinstance(data, dict) else data

        # total_time => # time points before the core; adapt if your shape is different
        # E.g., if 'responses' is shape [batch_size=1, #neurons, #time], 
        # you might need to rearrange or confirm dimension indexing.
        total_time = batch_kwargs["responses"].shape[2]
        number_neurons = batch_kwargs["responses"].shape[1]
        batch_kwargs["responses"][:, : number_neurons // 2, :] = 0 #mask one half to correlate mean activtiy only there
        # If your model expects model.mice[data_key] to exist, adapt accordingly:
        latents = model.encoder(
            batch_kwargs["responses"],
            data_key,
            model.mice[data_key][0:total_time, :]  # example indexing
        )

        if data_key not in latents_dict:
            latents_dict[data_key] = []
        latents_dict[data_key].append(latents.detach().cpu())

    return latents_dict


########################################################
# 4) BIN NEURONS INTO GRIDS & COMPUTE MEAN CONTRASTS #
########################################################

def compute_local_contrast(video, device="cuda"):
    """
    Computes local contrast in three successive average-pooling 'cone' layers.
    The kernel sizes are (temporal, height, width) = (11, 11), (5, 5), (5, 5),
    but factorized as a 3D pooling with kernel_size=(t, h, w).

    Args:
        video: torch.Tensor of shape [batch_size, 1, T, H, W]
            The raw videos in (B, C=1, Time, Height, Width).
        device: the torch device.

    Returns:
        local_contrast: torch.Tensor of shape [batch_size, 1, T_out, H_out, W_out]
    """
    video = video.to(device)

    # x^2
    video_sq = video * video

    # Three successive average-pooling layers
    # 1) kernel_size=(11,11,11)
    video_1  = F.avg_pool3d(video,    kernel_size=(11, 11, 11), stride=1, padding=0)
    video_sq_1 = F.avg_pool3d(video_sq, kernel_size=(11, 11, 11), stride=1, padding=0)

    # 2) kernel_size=(5,5,5)
    video_2  = F.avg_pool3d(video_1,    kernel_size=(5, 5, 5), stride=1, padding=0)
    video_sq_2 = F.avg_pool3d(video_sq_1, kernel_size=(5, 5, 5), stride=1, padding=0)

    # 3) kernel_size=(5,5,5)
    video_3  = F.avg_pool3d(video_2,    kernel_size=(5, 5, 5), stride=1, padding=0)
    video_sq_3 = F.avg_pool3d(video_sq_2, kernel_size=(5, 5, 5), stride=1, padding=0)

    # local_contrast = avgpool(x^2) - [avgpool(x)]^2
    local_contrast = video_sq_3 - (video_3 ** 2)

    return local_contrast

import torch.nn.functional as F

def extract_neuron_contrast(local_contrast, neuron_grid, align_corners=True):
    """
    For each neuron, sample the local_contrast at its (x,y) location across time.

    Args:
        local_contrast: torch.Tensor of shape [B, 1, T_out, H_out, W_out]
            The final local-contrast volume after the 3 pooling steps.
        neuron_grid: torch.Tensor of shape [N, 2]  (the readout grid).
            Typically these are normalized coords in [-1, 1].
            neuron_grid[n] = (x_n, y_n).
        align_corners: bool, passed to F.grid_sample.

    Returns:
        neuron_contrast: torch.Tensor of shape [B, T_out, N]
            The local-contrast time series for each neuron.
    """
    B, _, T_out, H_out, W_out = local_contrast.shape
    N = neuron_grid.shape[0]

    # We need to construct a sampling grid of shape [B, T_out, N, 2].
    # - For each time step, we use the same (x,y) for that neuron, i.e. no shift in time dimension.
    # - The time dimension is "ignored" by grid_sample in the sense that
    #   we treat local_contrast as [B, C=1, T_out, H_out, W_out], so the "depth" dimension is time.
    #   We want to sample along (height, width) for each "slice" in time.
    #   However, PyTorch's grid_sample for 5D input expects a grid of shape [B, T_out, H_out, W_out, 3]
    #   if we consider D=3 (Depth=Time, Height, Width).
    #
    #   Another approach is to loop over time or to reorder the dimensions so that we treat T_out
    #   as if it were "batch" dimension. The simplest is a time loop or unrolling. We'll do a loop for clarity.

    # A simpler approach if your net lumps time into the "height dimension" is to rearrange, but let's do a loop.

    # Initialize an output array
    neuron_contrast_list = []

    for t in range(T_out):
        # local_contrast[:, :, t, :, :] -> shape [B, 1, H_out, W_out]
        # We'll sample from 2D (H_out, W_out). So we can treat this as [B, 1, H_out, W_out].
        # and a grid of shape [B, N, 2].
        # Then the output is [B, 1, N, 1], if we treat N as the "spatial dimension" in the grid.

        # Expand the readout grid to [B, N, 2], repeating across the batch dimension
        # We'll treat each item in the batch the same. Or you can store data_key-specific grids differently.
        repeated_grid = neuron_grid.unsqueeze(0).expand(B, -1, -1)  # [B, N, 2]
        # For 2D grid_sample, we need shape [B, N, 1, 2].
        repeated_grid_2D = repeated_grid.unsqueeze(2)  # [B, N, 1, 2]

        # Now call grid_sample on the slice local_contrast[:, :, t, :, :].
        # shape => [B, 1, H_out, W_out]
        # sample => [B, 1, N, 1]
        slice_t = local_contrast[:, :, t, :, :]  # [B, 1, H_out, W_out]

        # grid_sample wants input in BHWC for 4D, but actually in PyTorch it's B,C,H,W for 4D.
        # The grid must be B, H_out', W_out', 2. Here we do B, N, 1, 2.
        sampled_t = F.grid_sample(
            input=slice_t,
            grid=repeated_grid_2D,       # shape [B, N, 1, 2]
            mode='bilinear', 
            padding_mode='border',
            align_corners=align_corners
        )
        # 'sampled_t' => [B, 1, N, 1]
        # Squeeze the single spatial dimension => [B, N]
        sampled_t = sampled_t.squeeze(-1).squeeze(1)  # => [B, N]

        # Accumulate in a list over time
        neuron_contrast_list.append(sampled_t)

    # Stack across time => shape [T_out, B, N]. We want [B, T_out, N].
    neuron_contrast_tensor = torch.stack(neuron_contrast_list, dim=0).permute(1,0,2)
    # => shape [B, T_out, N]

    return neuron_contrast_tensor

import torch
import numpy as np
from tqdm import tqdm
from neuralpredictors.training import LongCycler
from sklearn.cross_decomposition import CCA
from sklearn.model_selection import KFold

###############################
# 1) COMPUTE LOCAL CONTRAST  #
###############################
# [ Already defined: compute_local_contrast(video), extract_neuron_contrast(local_contrast, grid), etc. ]


#######################################################
# 2) AGGREGATE NEURON-SPECIFIC LOCAL CONTRAST PER BATCH
#######################################################

def compute_local_contrast_for_batches(model, data_loader, device="cuda"):
    """
    For each batch/key in the data_loader:
      1) Load raw video x,
      2) compute local contrast,
      3) sample for each neuron using the readout's grid,
      4) store time-series in a dictionary.

    Returns:
        local_contrast_dict: { data_key: list of [T, N] tensors }
    """
    model.eval()
    local_contrast_dict = {}

    with torch.no_grad():
        loader_oracle = data_loaders["oracle"]
        n_iterations = len(LongCycler(loader_oracle))


        print("Accumulating responses & latents..")
        for _, (data_key, data) in enumerate(LongCycler(loader_oracle)):
            # data might be a namedtuple or dict
            batch_kwargs = data._asdict() if not isinstance(data, dict) else data

            # The raw frames typically in batch_kwargs["videos"]
            # Suppose shape [B=1, C=1, T, H, W].
            video = batch_kwargs["videos"].to(device)  # shape: (1,1,T,H,W)

            # 1) compute local contrast
            loc_contrast = compute_local_contrast(video, device=device)  
            # => [1,1,T_out,H_out,W_out]

            # 2) get the readout grid for this data_key
            #    shape [N, 2], typically in normalized coords in [-1,1].
            readout_grid = model.readout[data_key].grid.squeeze()

            # 3) sample per neuron => shape [1, T_out, N]
            neuron_contrast = extract_neuron_contrast(loc_contrast, readout_grid)
            # => shape [1, T_out, N]

            # remove batch dim => shape [T_out, N]
            neuron_contrast = neuron_contrast.squeeze(0).cpu()

            if data_key not in local_contrast_dict:
                local_contrast_dict[data_key] = []
            local_contrast_dict[data_key].append(neuron_contrast)
            
    return local_contrast_dict


##########################################################
# 3) BIN NEURONS INTO NxN GRIDS & AGGREGATE OVER TIME AXIS
##########################################################

def create_spatial_grids_and_average_activities(
    x_coords: np.ndarray,
    y_coords: np.ndarray,
    signal: torch.Tensor,
    max_grid_size: int = 13,
):
    """
    Generic function to bin neurons into NxN grids for N=1..max_grid_size.
    Then average the time course (T,) across all neurons in each bin.

    Args:
        x_coords, y_coords: arrays of length N_neurons, specifying each neuron's location.
        signal: torch.Tensor of shape [T, N_neurons]
        max_grid_size: largest NxN bin size.

    Returns:
        grid_activities_dict: {N: np.ndarray of shape [T, N*N]}
          each column is the average time course for that bin.
    """
    if signal.is_cuda:
        signal = signal.cpu()
    signal_np = signal.detach().numpy()  # shape [T, N_neurons]

    min_x, max_x = x_coords.min(), x_coords.max()
    min_y, max_y = y_coords.min(), y_coords.max()

    grid_activities_dict = {}

    for N in range(1, max_grid_size + 1):
        x_bins = np.linspace(min_x, max_x, N + 1)
        y_bins = np.linspace(min_y, max_y, N + 1)

        cell_mean_list = []
        # We want N*N bins
        for row in range(N):
            for col in range(N):
                x_low, x_high = x_bins[col], x_bins[col + 1]
                y_low, y_high = y_bins[row], y_bins[row + 1]

                in_cell = (
                    (x_coords >= x_low) & (x_coords < x_high) &
                    (y_coords >= y_low) & (y_coords < y_high)
                )
                neuron_indices = np.where(in_cell)[0]

                if len(neuron_indices) == 0:
                    mean_sig = np.zeros((signal_np.shape[0],), dtype=np.float32)
                else:
                    mean_sig = signal_np[:, neuron_indices].mean(axis=1)

                cell_mean_list.append(mean_sig)

        # stack => [N*N, T], transpose => [T, N*N]
        cell_mean_stack = np.stack(cell_mean_list, axis=0).T
        grid_activities_dict[N] = cell_mean_stack

    return grid_activities_dict


############################################
# 4) 5-FOLD CCA CROSS-VALIDATION ON TIME AXIS
############################################

def perform_cca_cross_validation(
    X, 
    Y, 
    block_size=30, 
    n_folds=5, 
    n_components=2, 
    random_state=42
):
    """
    Same function you had before for cross-validated CCA along time blocks.
    """
    T = X.shape[0]

    # Number of full blocks
    n_blocks = T // block_size
    effective_length = n_blocks * block_size

    # Discard leftover time points
    X = X[:effective_length]
    Y = Y[:effective_length]

    # Reshape into blocks
    dX = X.shape[1]
    dY = Y.shape[1]
    X_blocks = X.reshape(n_blocks, block_size, dX)
    Y_blocks = Y.reshape(n_blocks, block_size, dY)

    block_indices = np.arange(n_blocks)
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=random_state)

    all_fold_correlations = []
    first_comp_weights_x = []
    first_comp_weights_y = []
    fold_corr_matrices = []

    for train_idx, test_idx in kf.split(block_indices):
        X_train = X_blocks[train_idx].reshape(-1, dX)
        Y_train = Y_blocks[train_idx].reshape(-1, dY)
        X_test  = X_blocks[test_idx].reshape(-1, dX)
        Y_test  = Y_blocks[test_idx].reshape(-1, dY)

        cca = CCA(n_components=n_components)
        cca.fit(X_train, Y_train)

        X_c, Y_c = cca.transform(X_test, Y_test)

        fold_correlations = []
        for comp_idx in range(n_components):
            corr_comp = np.corrcoef(X_c[:, comp_idx], Y_c[:, comp_idx])[0, 1]
            fold_correlations.append(corr_comp)
        
        all_fold_correlations.append(fold_correlations)
        
        # 2) Full matrix of correlations between (X_c^i, Y_c^j)
        fold_matrix = np.zeros((n_components, n_components))
        for i in range(n_components):
            for j in range(n_components):
                fold_matrix[i, j] = np.corrcoef(X_c[:, i], Y_c[:, j])[0, 1]
        fold_corr_matrices.append(fold_matrix)

    data_key_results = {}
    # Average the diagonal across folds as before
    data_key_results["fold_correlations"] = all_fold_correlations,  # diagonal correlations
    
    # Compute the average correlation matrix across folds
    avg_corr_matrix = np.mean(fold_corr_matrices, axis=0)  # shape [n_comp, n_comp]
    data_key_results["avg_corr_matrix"] = avg_corr_matrix

        
    first_comp_weights_x.append(cca.x_weights_[:, 0])
    first_comp_weights_y.append(cca.y_weights_[:, 0])

    return data_key_results, first_comp_weights_x, first_comp_weights_y


###########################################################
# 5) MAIN PIPELINE: LOCAL CONTRAST -> NxN BIN -> CCA vs. LATENTS
###########################################################

def run_grid_cca_analysis_contrast(
    data_loader_oracle,
    latents_dict,
    local_contrast_dict,
    cell_coordinates,
    max_grid=5
):
    """
    1) Concatenate the local contrast [T, N_neurons] across all batches.
    2) Concatenate latents [T, latent_dim].
    3) Bin neurons in NxN by x,y coords and average their local contrast time courses.
    4) Do 5-fold block-based CCA with latents.
    """

    # We'll accumulate local-contrast for each data_key:
    #   aggregator[data_key]["contrast_list"] -> list of shape [T_batch_i, N]
    #   aggregator[data_key]["latents_list"]  -> list of shape [T_batch_i, latent_dim]
    #   aggregator[data_key]["coords_xyz"]    -> from cell_coordinates

    aggregator = {}
    oracle_cycler = LongCycler(data_loader_oracle)
    n_iterations = len(oracle_cycler)

    for _, (data_key, data) in enumerate(oracle_cycler):
        batch_kwargs = data._asdict() if not isinstance(data, dict) else data
        # shape note: might be [1, N, T] or [T, N], adapt as needed
        # We'll assume [1, N, T], so we might do .squeeze(0).T => [T, N]
        # Adjust to match your shape in practice.
        responses_raw = batch_kwargs["responses"]  # e.g., shape [1, N, T]
        responses_tensor = responses_raw.squeeze(0).permute(1, 0)  # => [T, N]
        # -- local contrast times for this batch --
        #   we stored them in local_contrast_dict[data_key] in the same order as oracle_cycler
        if data_key not in local_contrast_dict or len(local_contrast_dict[data_key]) == 0:
            continue
        contrast_this_batch = local_contrast_dict[data_key].pop(0)  # shape [T_b, N]

        # -- latents --
        if data_key not in latents_dict or len(latents_dict[data_key]) == 0:
            continue
        latents_this_batch = latents_dict[data_key].pop(0)  # shape [T_b, latent_dim]

        if data_key not in aggregator:
            aggregator[data_key] = {
                "contrast_list": [],
                "latents_list": [],
                "coords_xyz": cell_coordinates[data_key].cpu().numpy(),
                "responses_list": []
            }

        aggregator[data_key]["contrast_list"].append(contrast_this_batch.cpu())
        time_points = contrast_this_batch.shape[0]
        aggregator[data_key]["latents_list"].append(latents_this_batch[:, -time_points:, :].squeeze().cpu())
        aggregator[data_key]["responses_list"].append(responses_tensor[ -time_points:, :].cpu())

    # Now run NxN grid-based CCA
    results = {}

    for data_key, struct in aggregator.items():
        # Concatenate across time
        local_contrast_cat = torch.cat(struct["contrast_list"], dim=0)  # [T_total, N]
        latents_cat = torch.cat(struct["latents_list"], dim=0)   # [T_total, latent_dim]
        responses_cat = torch.cat(struct["responses_list"], dim=0)

        coords_xyz = struct["coords_xyz"]
        x_coords = coords_xyz[:, 0]
        y_coords = coords_xyz[:, 1]

        # Bin the local contrast into NxN grids
        grid_activities_dict = create_spatial_grids_and_average_activities(
            x_coords=x_coords,
            y_coords=y_coords,
            signal=local_contrast_cat,
            max_grid_size=max_grid
        )

        response_activities_dict = create_spatial_grids_and_average_activities(
            x_coords=x_coords,
            y_coords=y_coords,
            signal=responses_cat,
            max_grid_size=max_grid
        )
        # For each grid dimension, do 5-fold CCA
        data_key_results = {}
        for N in range(1, max_grid + 1):
            # shape => [T_total, N*N]
            X = response_activities_dict[N]
            
            '''
            # The shape of real data [T_total, N*N]
            T, M = grid_activities_dict[N].shape
            # Create GP samples with desired mean/std for each of the M columns
            gp_samples = []
            for m in range(M):
                y = sample_block_diag_gp_independently(
                        num_blocks=T // 281,
                        block_size=281,
                        lengthscale=30.0,
                        variance=1.0,
                        target_mean=0.8,
                        target_std=0.4,
                        min_val=0.05,
                        random_state=None
                    )
                gp_samples.append(y)
            # Stack columns => shape [T, M]
            X = np.column_stack(gp_samples)
            '''


            # shape => [T_total, latent_dim]
            Y = latents_cat.numpy()
            #Y = response_activities_dict[N]

            # n_components can be up to min(N*N, latent_dim). We'll just do 2 or something smaller:
            n_comp = min(N*N, 12)

            all_fold_corrs, x_w_first, y_w_first = perform_cca_cross_validation(
                X, Y,
                block_size=300,
                n_folds=5,
                n_components=n_comp,
                random_state=42
            )

            data_key_results[N] = {
                "fold_correlations": all_fold_corrs["fold_correlations"],
                "correlation_matrix": all_fold_corrs["avg_corr_matrix"],
                "x_weights_first": x_w_first,
                "y_weights_first": y_w_first,
            }

        results[data_key] = data_key_results

    return results

from sklearn.model_selection import KFold
from sklearn.linear_model import LinearRegression
import numpy as np

def perform_linear_regression_cv(
    Z,        # shape [T, d]
    Y,        # shape [T, N]
    block_size=30, 
    n_folds=5, 
    random_state=42
):
    """
    Perform linear regression of Y ~ Z (multi-output) with block-based
    cross-validation along the time axis.

    1) Partition T into blocks of size block_size, ignoring any leftover.
    2) KFold over the block indices => get train blocks / test blocks.
    3) Fit a single linear model on the training data:
          Y = ZW + b
       Then evaluate test R^2 on the test data.
    4) Return the average R^2 across folds (and optionally the per-fold or per-neuron R^2).

    Returns:
        mean_r2: float, average R^2 over all folds and all neurons
        fold_r2_list: list of length n_folds, each is the R^2 (averaged over neurons)
    """

    T, d = Z.shape
    _, N = Y.shape

    # 1) Break T into consecutive blocks of length 'block_size'
    n_blocks = T // block_size
    effective_length = n_blocks * block_size

    # Truncate leftover
    Z = Z[:effective_length]
    Y = Y[:effective_length]

    # Reshape => [n_blocks, block_size, ...]
    Z_blocks = Z.reshape(n_blocks, block_size, d)    # [n_blocks, block_size, d]
    Y_blocks = Y.reshape(n_blocks, block_size, N)    # [n_blocks, block_size, N]

    block_indices = np.arange(n_blocks)
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=random_state)

    fold_r2_list = []

    for train_idx, test_idx in kf.split(block_indices):
        # Combine train blocks
        Z_train = Z_blocks[train_idx].reshape(-1, d)  # shape [train_blocks*block_size, d]
        Y_train = Y_blocks[train_idx].reshape(-1, N)  # shape [train_blocks*block_size, N]

        # Combine test blocks
        Z_test  = Z_blocks[test_idx].reshape(-1, d)   # shape [test_blocks*block_size, d]
        Y_test  = Y_blocks[test_idx].reshape(-1, N)   # shape [test_blocks*block_size, N]

        # 2) Fit multi-output linear regression
        # We'll use scikit-learn's LinearRegression with fit_intercept=True
        linreg = LinearRegression(fit_intercept=True)
        #linreg.fit(Z_train, Y_train)
        linreg.fit(Y_train, Z_train) #regress from contrast to latents

        # 3) Predict on test
        #Y_pred = linreg.predict(Z_test)  # shape [test_points, N]
        Z_pred = linreg.predict(Y_test)  # shape [test_points, k]

        # 4) Compute R^2 for each neuron => average
        # R^2_i = 1 - sum((Y_test_i - Y_pred_i)^2) / sum((Y_test_i - mean(Y_test_i))^2)
        # We'll do an average across neurons
        # ss_res = np.sum((Y_test - Y_pred)**2, axis=0)  # shape [N,]
        # y_mean = Y_test.mean(axis=0)                   # shape [N,]
        # ss_tot = np.sum((Y_test - y_mean)**2, axis=0)  # shape [N,]
        ss_res = np.sum((Z_test - Z_pred)**2, axis=0)  # shape [k,]
        Z_mean = Z_test.mean(axis=0)                   # shape [k,]
        ss_tot = np.sum((Z_test - Z_mean)**2, axis=0)  # shape [k,]
        r2_per_neuron = 1.0 - ss_res / (ss_tot + 1e-12) # add small epsilon to avoid /0
        r2_fold = r2_per_neuron.mean()                 # average across N

        fold_r2_list.append(r2_fold)

    mean_r2 = np.mean(fold_r2_list)
    std_r2 = np.std(fold_r2_list)
    return mean_r2, std_r2


def run_linear_regression_analysis(
    data_loader_oracle,
    latents_dict,
    local_contrast_dict,
    cell_coordinates,
    block_size=30,
    n_folds=5
):
    """
    For each data_key, do a multi-output linear regression of local_contrast
    on latents, with 5-fold block-based cross-validation. Compute average R^2.
    """
    from neuralpredictors.training import LongCycler

    aggregator = {}
    oracle_cycler = LongCycler(data_loader_oracle)
    n_iterations = len(oracle_cycler)


    for _, (data_key, data) in enumerate(oracle_cycler):
        batch_kwargs = data._asdict() if not isinstance(data, dict) else data
        # shape note: might be [1, N, T] or [T, N], adapt as needed
        # We'll assume [1, N, T], so we might do .squeeze(0).T => [T, N]
        # Adjust to match your shape in practice.
        responses_raw = batch_kwargs["responses"]  # e.g., shape [1, N, T]
        responses_tensor = responses_raw.squeeze(0).permute(1, 0)  # => [T, N]

        # shape [1, N, T] => e.g. [1, N, T]
        if data_key not in latents_dict or len(latents_dict[data_key]) == 0:
            continue
        latents_this_batch = latents_dict[data_key].pop(0)  # shape [T_batch, d]

        if data_key not in local_contrast_dict or len(local_contrast_dict[data_key]) == 0:
            continue
        contrast_this_batch = local_contrast_dict[data_key].pop(0)  # shape [T_batch, N]

        if data_key not in aggregator:
            aggregator[data_key] = {
                "latents_list": [],
                "contrast_list": [],
                "responses_list": []
            }
        time_points = contrast_this_batch.shape[0]
        aggregator[data_key]["latents_list"].append(latents_this_batch[:, -time_points:, :].squeeze().cpu())
        aggregator[data_key]["contrast_list"].append(contrast_this_batch.cpu())
        aggregator[data_key]["responses_list"].append(responses_tensor[ -time_points:, :].cpu())

    # 2) For each data_key, concat in time and run cross-validation
    results = {}
    for data_key, dct in aggregator.items():
        latents_cat = torch.cat(dct["latents_list"], dim=0).numpy()     # shape [T_total, d]
        contrast_cat = torch.cat(dct["contrast_list"], dim=0).numpy()   # shape [T_total, N]
        responses_cat = torch.cat(dct["responses_list"], dim=0)

        mean_r2, std_r2 = perform_linear_regression_cv(
            latents_cat,
            contrast_cat,
            block_size=block_size,
            n_folds=n_folds,
            random_state=42
        )
        results[data_key] = {
            "mean_r2": mean_r2,
            "std_r2": std_r2
        }

    return results

import os
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

def plot_cca_heatmaps(results_cca_contrast):
    """
    For each data_key (mouse) and each grid size, plot a heatmap of the
    averaged CCA correlation matrix (n_comp x n_comp) on the test set
    and save the figure under:
      plots/response_vs_activity/<data_key>/grid_<N>x<N>.png
    """
    for data_key, grid_dict in results_cca_contrast.items():
        for N, result_info in grid_dict.items():
            avg_corr_matrix = result_info["correlation_matrix"]  # shape [n_comp, n_comp]
            n_comp = avg_corr_matrix.shape[0]
            
            # Create a figure and axes
            fig, ax = plt.subplots(figsize=(6, 5))

            # Plot the heatmap
            sns.heatmap(
                avg_corr_matrix,
                annot=True,           # Show numeric values
                fmt=".2f",            # Round to 2 decimal places
                cmap="RdBu_r",        # Divergent colormap from red to blue
                vmin=-1, vmax=1,      # Correlations range from -1 to 1
                square=True,
                ax=ax
            )

            #ax.set_title(f"CCA Correlations\nData Key: {data_key}, Grid Size: {N}x{N}", fontsize=16)
            ax.set_xlabel("Y_c components", fontsize=14)
            ax.set_ylabel("X_c components", fontsize=14)

            # Label the ticks more clearly
            ax.set_xticks(np.arange(n_comp) + 0.5)
            ax.set_yticks(np.arange(n_comp) + 0.5)
            ax.set_xticklabels([f"Y_c{i}" for i in range(n_comp)])
            ax.set_yticklabels([f"X_c{i}" for i in range(n_comp)], rotation=0)

            plt.tight_layout()

            # Create the output directory if it doesn't exist
            out_dir = f"plots/response_vs_activity/{data_key}"
            os.makedirs(out_dir, exist_ok=True)

            # Build the file path for saving
            out_path = os.path.join(out_dir, f"grid_{N}x{N}.png")

            # Save the figure
            plt.savefig(out_path)

            # Close the figure to free memory, especially important for many plots
            plt.close(fig)



#################################
# 6) PUTTING IT ALL TOGETHER
#################################

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Paths to your data
    data_keys = data_paths = [
    "/scratch-grete/projects/nim00012/original_sensorium_2023/dynamic29156-11-10-Video-8744edeac3b4d1ce16b680916b5267ce/",
    "/scratch-grete/projects/nim00012/original_sensorium_2023/dynamic29228-2-10-Video-8744edeac3b4d1ce16b680916b5267ce/",
    "/scratch-grete/projects/nim00012/original_sensorium_2023/dynamic29234-6-9-Video-8744edeac3b4d1ce16b680916b5267ce/",
    "/scratch-grete/projects/nim00012/original_sensorium_2023/dynamic29513-3-5-Video-8744edeac3b4d1ce16b680916b5267ce/",
    "/scratch-grete/projects/nim00012/original_sensorium_2023/dynamic29514-2-9-Video-8744edeac3b4d1ce16b680916b5267ce/",
]
    # 1) Create data loaders & load coordinates
    data_loaders, cell_coords = create_data_loader_and_coords(data_paths, device=device)

    # 2) Load model
    latent_dim = [42,20,12] 

    encoder_dict = {
        "hidden_dim": latent_dim[0],  # 42
        "hidden_gru": latent_dim[1],
        "output_dim": latent_dim[2],  # 12
        "hidden_layers": 1,
        "n_samples": 100,
        "mice_dim": 0,  
        "use_cnn": False,
        "residual": False,
        "kernel_size": None,
        "channel_size": None,
        "use_resnet": False,
        "pretrained": True,
    }

    decoder_dict = {
        "hidden_layers": 1,
        "hidden_dim": latent_dim[2],
        "use_cnn": False,
        "kernel_size": None,
        "channel_size": None,
    }
    decoder_dict = None

    position_mlp = {
            "input_size": 3,
            "layer_sizes": [6, 12]
         }
    model_path = "models/16_core_channels_latent_mice6_10best.pth"
    #model_path = "toymodels2/[6,12]dim_no_brain_positions_pretrain_nodecoderbest.pth"
    #model_path = "toymodels2/12dim_no_brain_positions_pretrain_nodecoderbest.pth"
    #model_path = "toymodels2/150dim_no_brain_positions_pretrain_nodecoderbest.pth"
    print(model_path)

    model = load_model(
        model_path=model_path,
        encoder_dict=encoder_dict,
        decoder_dict=decoder_dict,
        latent=True,
        flow=False,
        outchannels=2,      # For ZIG
        dropout=False,
        dropout_prob=0.0,
        device=device,
        data_loaders=data_loaders,
        data_loaders_nobehavior=data_loaders,  # or separate if needed
        grid_mean_predictor=True,
        position_mlp=None,
        behavior_mlp=None,
    )

    # 3) COMPUTE LATENTS PER BATCH
    latents_dict = compute_latents(model, data_loaders)

    # 4) COMPUTE LOCAL CONTRAST PER BATCH, PER NEURON
    local_contrast_dict = compute_local_contrast_for_batches(
        model=model,
        data_loader=data_loaders["oracle"], 
        device=device
    )
    '''
    # 5) RUN NxN BIN & 5-FOLD CCA
    results_cca_contrast = run_grid_cca_analysis_contrast(
        data_loader_oracle=data_loaders["oracle"],
        latents_dict=latents_dict,
        local_contrast_dict=local_contrast_dict,
        cell_coordinates=cell_coords,
        max_grid=10
    )

    # 6) PRINT RESULTS
    for data_key, grid_dict in results_cca_contrast.items():
        print(f"\n=== CCA Results for data_key: {data_key} ===")
        for N, result_info in grid_dict.items():
            fold_correlations = result_info["fold_correlations"]
            
            # Compute mean±std across folds for each canonical component
            fold_array = np.array(fold_correlations).squeeze(0)  # shape [n_folds, n_components]
            avg_cor = fold_array.mean(axis=0)
            std_cor = fold_array.std(axis=0)

            comp_strings = [
                f"{avg_cor[i]:.3f} ({std_cor[i]:.3f})"
                for i in range(len(avg_cor))
            ]
            print(f"  Grid {N}x{N}: {comp_strings}")

    plot_cca_heatmaps(results_cca_contrast)

    print("\nCCA analysis done!")
    '''
    # 5) Run the linear regression analysis
    linear_regression_results = run_linear_regression_analysis(
        data_loader_oracle=data_loaders["oracle"],
        latents_dict=latents_dict,
        local_contrast_dict=local_contrast_dict,
        cell_coordinates=cell_coords,
        block_size=300,
        n_folds=5
    )

    # 6) Print
    mean_avg = 0
    std_avg = 0
    for data_key, dct in linear_regression_results.items():
        mean_r2 = dct["mean_r2"]
        std_r2= dct["std_r2"]
        mean_avg += dct["mean_r2"]
        std_avg += dct["std_r2"]
        print(f"Data_key {data_key}: Mean R^2 = {mean_r2:.3f} ({std_r2:.3f})")
    print(f"Average R^2: {mean_avg/5} ({std_avg/5})")
