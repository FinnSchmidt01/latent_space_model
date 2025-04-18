from operator import itemgetter

import torch
from nnfabrik.utility.nn_helpers import set_random_seed
from torch import nn

from neuralpredictors.layers.cores import RotationEquivariant2dCore, Stacked2dCore

# imports for 3d cores and gru
from neuralpredictors.layers.cores.conv3d import Basic3dCore, Factorized3dCore
from neuralpredictors.layers.rnn_modules.gru_module import GRU_Module
from neuralpredictors.layers.shifters import MLPShifter, StaticAffine2dShifter
from neuralpredictors.utils import get_module_output

from .readouts import MultipleFullFactorized2d, MultipleFullGaussian2d
from .utility import get_dims_for_loader_dict, prepare_grid
from .video_encoder import VideoFiringRateEncoder


def make_video_model(
    dataloaders,
    seed,
    core_dict,
    core_type,
    readout_dict,
    readout_type,
    use_gru,
    gru_dict,
    use_shifter,
    shifter_dict,
    shifter_type,
    elu_offset=0.0,
    nonlinearity_type="elu",
    nonlinearity_config=None,
    deeplake_ds=False,
    n_neurons_dict=None,
    mean_activity_dict=None,
    readout_dim=None,
    experanto=False,
):
    """
    Model class of a stacked2dCore (from neuralpredictors) and a pointpooled (spatial transformer) readout
    Args:
        dataloaders: a dictionary of dataloaders, one loader per session
            in the format {'data_key': dataloader object, .. }
        seed: random seed
        grid_mean_predictor: if not None, needs to be a dictionary of the form
            {
            'type': 'cortex',
            'input_dimensions': 2,
            'hidden_layers':0,
            'hidden_features':20,
            'final_tanh': False,
            }
            In that case the datasets need to have the property `neurons.cell_motor_coordinates`
        share_features: whether to share features between readouts. This requires that the datasets
            have the properties `neurons.multi_match_id` which are used for matching. Every dataset
            has to have all these ids and cannot have any more.
        all other args: See Documentation of Stacked2dCore in neuralpredictors.layers.cores and
            PointPooled2D in neuralpredictors.layers.readouts
    Returns: An initialized model which consists of model.core and model.readout
    """
    if not experanto:
        if "train" in dataloaders.keys():
            dataloaders = dataloaders["train"]

    # Obtain the named tuple fields from the first entry of the first dataloader in the dictionary

    if n_neurons_dict is None:
        session_shape_dict = get_dims_for_loader_dict(dataloaders, deeplake_ds)
        n_neurons_dict = {k: v['responses'][1] for k, v in session_shape_dict.items()}

    set_random_seed(seed)

    if core_type == "2D_equivariant":
        core = RotationEquivariant2dCore(**core_dict)
    elif core_type == "2D":
        core = Stacked2dCore(**core_dict)
    elif core_type == "3D_factorised":
        core = Factorized3dCore(**core_dict)
    elif core_type == "3D":
        if core_dict["spatial_input_kernel"] is not None:
            core_dict["input_kernel"] = (
                core_dict["in_channels"],
                core_dict["spatial_input_kernel"],
                core_dict["spatial_input_kernel"],
            )
        else:
            core_dict["input_kernel"] = (
                core_dict["num_frames"],
                core_dict["input_kernel"],
                core_dict["input_kernel"],
            )

        core_dict["hidden_kernel"] = (
            core_dict["num_frames"],
            core_dict["hidden_kernel"],
            core_dict["hidden_kernel"],
        )

        del core_dict["num_frames"]
        del core_dict["spatial_input_kernel"]
        core = Basic3dCore(**core_dict)
    else:
        raise NotImplementedError(f"core type {core_type} is not implemented")
    if experanto:
        in_shapes_dict = {
            k: (readout_dim, 18, 46)
            for k in list(n_neurons_dict.keys())
        }
    else:
        subselect = itemgetter(0, 2, 3)
        in_shapes_dict = {
            k: subselect(tuple(get_module_output(core, v['videos'])[1:]))
            for k, v in session_shape_dict.items()
        }
    if mean_activity_dict is None:
        mean_activity_dict = {
            k: next(iter(dataloaders[k]))[1].mean(0).mean(-1)
            for k in dataloaders.keys()
        }

    readout_dict["in_shape_dict"] = in_shapes_dict
    readout_dict["n_neurons_dict"] = n_neurons_dict
    # todo - check if its needed at al?
    # readout_dict["loaders"] = dataloaders

    if readout_type == "gaussian":
        grid_mean_predictor, grid_mean_predictor_type, source_grids = prepare_grid(
            readout_dict["grid_mean_predictor"], dataloaders, deeplake_ds
        )

        readout_dict["mean_activity_dict"] = mean_activity_dict
        readout_dict["grid_mean_predictor"] = grid_mean_predictor
        readout_dict["grid_mean_predictor_type"] = grid_mean_predictor_type
        readout_dict["source_grids"] = source_grids
        readout = MultipleFullGaussian2d(**readout_dict)
    else:
        raise NotImplementedError(f"readout type {readout_type} is not implemented")

    if use_gru:
        gru_module = GRU_Module(**gru_dict)
    else:
        gru_module = None

    shifter = None
    if use_shifter:
        # todo 
        data_keys = list(n_neurons_dict.keys())
        shifter_dict["data_keys"] = data_keys
        if shifter_type == "MLP":
            shifter = MLPShifter(**shifter_dict)

        elif shifter_type == "StaticAffine":
            shifter = StaticAffine2dShifter(**shifter_dict)
        else:
            raise NotImplementedError(f"shifter type {shifter_type} is not implemented")

    twoD_core = "2D" in core_type

    model = VideoFiringRateEncoder(
        core=core,
        readout=readout,
        shifter=shifter,
        modulator=None,
        elu_offset=elu_offset,
        nonlinearity_type=nonlinearity_type,
        nonlinearity_config=nonlinearity_config,
        use_gru=use_gru,
        gru_module=gru_module,
        twoD_core=twoD_core,
    )

    return model
