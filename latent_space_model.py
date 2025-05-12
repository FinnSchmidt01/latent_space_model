seed = 42
import sys

sys.path.append("/srv/user/turishcheva/sensorium_replicate/sensorium_2023/")
# sys.path.append('/srv/user/turishcheva/sensorium_replicate/neuralpredictors/')
# sys.path.append('/srv/user/turishcheva/from_ayush/ayush_april/new_neuropred_code/neuralpredictors')
# sys.path.append('/srv/user/turishcheva/from_ayush/ayush_april/sensorium_2023')

from functools import partial

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from nnfabrik.utility.nn_helpers import set_random_seed
from torch.distributions.kl import kl_divergence
from torch.distributions.normal import Normal
from torch.utils.data import DataLoader
from torch.utils.data.dataloader import default_collate

set_random_seed(seed)

import json

import matplotlib.pyplot as plt
import optuna
from moments import load_mean_variance
from nnfabrik.builder import get_trainer
from nnfabrik.utility.nn_helpers import set_random_seed
from optuna.visualization import (
    plot_contour,
    plot_optimization_history,
    plot_param_importances,
    plot_slice,
)
from sensorium.datasets.mouse_video_loaders import mouse_video_loader
from sensorium.models.make_model import make_video_model
from sensorium.models.video_encoder import VideoFiringRateEncoder
from sensorium.utility import scores
from sensorium.utility.scores import get_correlations, get_poisson_loss
from tqdm import tqdm

import wandb
from neuralpredictors.layers.cores.conv2d import Stacked2dCore
from neuralpredictors.layers.encoders.mean_variance_functions import fitted_zig_mean
from neuralpredictors.layers.encoders.zero_inflation_encoders import ZIGEncoder
from neuralpredictors.measures import modules, zero_inflated_losses
from neuralpredictors.training import LongCycler, early_stopping

device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)
# torch.cuda.set_device(device)

import os

### New code


def kl_divergence_gaussian(mu, sigma):
    """
    Compute the KL divergence D_KL(q || p) where:
    q(z) = N(mu, diag(sigma^2)) and p(z) = N(0, I)

    Parameters:
    - mu (torch.Tensor): Mean vector of the Gaussian distribution q(z), shape (B,time,hidden).
    - sigma (torch.Tensor): Standard deviation vector of q(z), not squared, shape (B,time,).

    Returns:
    - kl_div (torch.Tensor): The KL divergence.
    """

    sigma_squared = sigma**2
    inverse_sigma_squared = 1.0 / sigma_squared

    # Trace term: Sum of inverse variances (since trace of a diagonal matrix is the sum of its diagonal elements)
    dim = mu.shape[0] * mu.shape[1] * mu.shape[2]

    trace_term = (
        inverse_sigma_squared * dim
    )  # sigma is a scalar which is constant acrosss all dims

    # Quadratic term: (mu^T * Sigma_0^{-1} * mu)
    quadratic_term = torch.sum(mu**2) * inverse_sigma_squared

    # Log-determinant term: 2 * sum(log(sigma))
    log_det_term = (
        2 * torch.log(sigma) * dim
    )  # sigma is a scalar which is constant acrosss all dims

    kl_div = 0.5 * (trace_term + quadratic_term - dim + log_det_term)
    return kl_div


# compute exponentail moving average of correlation
def calculate_ema(data, alpha):
    ema = torch.zeros(len(data))
    ema[0] = data[0]

    for t in range(1, len(data)):
        ema[t] = alpha * data[t] + (1 - alpha) * ema[t - 1]

    return ema


class WarmupCosineAnnealingLR(torch.optim.lr_scheduler._LRScheduler):
    def __init__(
        self,
        optimizer,
        warmup_steps,
        max_lr,
        min_lr,
        T_max,
        total_batches=225,
        last_epoch=-1,
        repeat_warmup=False,
    ):
        self.warmup_steps = warmup_steps
        self.max_lr = max_lr
        self.min_lr = min_lr
        self.T_max = T_max
        self.total_batches = total_batches
        self.batch_step = 0  # to track batch steps during warmup
        self.repeat_warmup = repeat_warmup
        super().__init__(optimizer, last_epoch)

    def step(self, batch_increment=None):
        if batch_increment is not None:
            self.batch_step += 5
            self.last_epoch = (
                self.batch_step / self.total_batches
            )  # Update epoch count after full batch cycle
        super().step()

    def get_lr(self):
        if self.repeat_warmup:
            lr = (self.max_lr - self.min_lr) * (
                (self.batch_step % (5 * self.warmup_steps)) / (5 * self.warmup_steps)
            ) + self.min_lr
        else:
            if self.batch_step < 5 * self.warmup_steps:
                # Warmup phase based on batch steps
                lr = (self.max_lr - self.min_lr) * (
                    self.batch_step / (5 * self.warmup_steps)
                ) + self.min_lr
            else:
                # Cosine annealing phase based on epoch count
                lr = self.min_lr + 0.5 * (self.max_lr - self.min_lr) * (
                    1
                    + torch.cos(
                        torch.tensor(
                            torch.pi
                            * (
                                (
                                    self.last_epoch
                                    - self.warmup_steps * 5 / self.total_batches
                                )
                                % self.T_max
                            )
                            / self.T_max
                        )
                    )
                )
        return [lr for _ in self.base_lrs]


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
    batch_size=8,
    scale=1,
    max_frame=None,
    frames=80,  # frames has to be > 50. If it fits on your gpu, we recommend 150
    offset=-1,
    include_behavior=True,
    include_pupil_centers=True,
    cuda=device != "cpu",
)

data_loaders_nobehavior = mouse_video_loader(
    paths=paths,
    batch_size=8,
    scale=1,
    max_frame=None,
    frames=80,  # frames has to be > 50. If it fits on your gpu, we recommend 150
    offset=-1,
    include_behavior=False,
    include_pupil_centers=False,
    cuda=device != "cpu",
)

print("Data loaded")

cell_coordinates = {}
data_keys = [
    "dynamic29515-10-12-Video-9b4f6a1a067fe51e15306b9628efea20",
    "dynamic29623-4-9-Video-9b4f6a1a067fe51e15306b9628efea20",
    "dynamic29712-5-9-Video-9b4f6a1a067fe51e15306b9628efea20",
    "dynamic29647-19-8-Video-9b4f6a1a067fe51e15306b9628efea20",
    "dynamic29755-2-8-Video-9b4f6a1a067fe51e15306b9628efea20",
]
for data_key in data_keys:
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
    cell_coordinates[data_key] = (cell_coordinates[data_key] - mean_coords) / std_coords


def standard_trainer(
    model,
    dataloaders,
    seed,
    avg_loss=False,
    scale_loss=True,
    loss_function="PoissonLoss",
    stop_function="get_correlations",
    loss_accum_batch_n=None,
    device="cuda",
    verbose=True,
    interval=1,
    patience=5,
    epoch=0,
    lr_init=0.005,
    max_iter=200,
    maximize=True,
    tolerance=1e-6,
    restore_best=True,
    lr_decay_steps=3,
    lr_decay_factor=0.3,
    min_lr=0.0001,
    warmup_steps=5,
    T_max=5,
    cb=None,
    detach_core=False,
    use_wandb=True,
    wandb_project="factorised_core_parameter_search",
    wandb_entity="movies_parameter_search",
    wandb_name=None,
    wandb_model_config=None,
    wandb_dataset_config=None,
    print_step=1000,
    save_checkpoints=True,
    checkpoint_save_path="local/",
    chpt_save_step=15,
    deeplake_ds=False,
    k_reg=False,
    hyper=False,
    ema_span=0.3,  # ema for validation correlation
    scheduler_patience=6,  # patience for decaying learning rate
    latent=False,
    **kwargs,
):
    """

    Args:
        model: model to be trained
        dataloaders: dataloaders containing the data to train the model with
        seed: random seed
        avg_loss: whether to average (or sum) the loss over a batch
        scale_loss: whether to scale the loss according to the size of the dataset
        loss_function: loss function to use
        stop_function: the function (metric) that is used to determine the end of the training in early stopping
        loss_accum_batch_n: number of batches to accumulate the loss over
        device: device to run the training on
        verbose: whether to print out a message for each optimizer step
        interval: interval at which objective is evaluated to consider early stopping
        patience: number of times the objective is allowed to not become better before the iterator terminates
        epoch: starting epoch
        lr_init: initial learning rate
        max_iter: maximum number of training iterations
        maximize: whether to maximize or minimize the objective function
        tolerance: tolerance for early stopping
        restore_best: whether to restore the model to the best state after early stopping
        lr_decay_steps: how many times to decay the learning rate after no improvement
        lr_decay_factor: factor to decay the learning rate with
        min_lr: minimum learning rate
        warmup_steps: number of batch steps for linear lr warump
        T_max: epoch periodicity of cosine annealing schedluer
        cb: whether to execute callback function
        zig : True if ZIG encoder is used as model
        k_reg: is a dictonary containg the fitted k_values for each mice, applies regularization to size of shape parameter k of gamma distribution if k_reg is not None but a dictionary,
        hyper: is optuna trial if hyperparameter search is active, otherwise False
        ema-Span: alpha factor of exponential moving avaerage of validation correlation
        **kwargs:

    Returns:

    """
    print(loss_function)

    def full_objective(model, dataloader, data_key, *args, k_regu=k_reg, **kwargs):
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

        if loss_function == "ZIGLoss" or loss_function == "combinedLoss":
            # one entry in a tuple corresponds to one paramter of ZIG
            # the output is (theta,k,loc,q)
            if model.position_features:
                positions = cell_coordinates[data_key]
            else:
                positions = None
            # args[0][0:1] removes behavior from the video input data.
            model_output = model(
                args[0][:, 0:1].to(device),
                data_key=data_key,
                out_predicts=False,
                train=True,
                positions=positions,
                **kwargs,
            )
            theta = model_output[0]
            k = model_output[1]
            loc = model_output[2]
            q = model_output[3]
            time_left = k.shape[1]

            original_data = args[1].transpose(2, 1)[:, -time_left:, :].to(device)

            # create zero, non zero masks
            comparison_result = original_data >= loc
            nonzero_mask = comparison_result.int()

            comparison_result = original_data <= loc
            zero_mask = comparison_result.int()

            # if model.flow: #transform orginal data with flow, if flow is applied to model
            # _, log_det = model.flow[data_key](original_data,zero_mask)
            # else:
            # log_det = 0

            if k_regu:
                k_fitted = k_regu[data_key + "fitted_k"]
                # k is constant over time and has shape (batch_size,time,num_neurons)
                k_output = k[0, 0, :]
                # punish values of k that are far away from the fitted k value, scale the regularization to the size of zig loss
                k_regularized = ((k_fitted - k_output) ** 2).mean() * 7 * 10**7

            else:
                k_regularized = 0

            if (
                len(model_output) > 4
            ):  # that is the case only for the latent space model
                k = k.unsqueeze(-1)
                loc = loc.unsqueeze(-1)
                zero_mask = zero_mask.unsqueeze(-1)
                nonzero_mask = nonzero_mask.unsqueeze(-1)
                original_data = original_data.unsqueeze(-1)
                means = model_output[4]
                sigma_squared = model_output[5]
                n_samples = model_output[7]

                sigma = torch.sqrt(sigma_squared)
                # Mask neurons, which were given in Encoder
                neuron_mask = model_output[8].to(means.device)
                neuron_mask = neuron_mask.unsqueeze(-1).repeat(1, 1, 1, n_samples)

                if model.flow:
                    if model.flow_base == "Gaussian":
                        zig_loss, log_det = criterion(
                            model,
                            data_key,
                            targets=original_data,
                            rho=loc,
                            qs=q,
                            means=theta,
                            psi_diag=model.psi[data_key],
                        )
                        zig_loss = (
                            -1 * loss_scale * zig_loss
                        )  # the gaussian log likelihood already computes and sums the log_det
                    else:
                        original_data, log_det = model.flow[data_key](
                            original_data.squeeze(-1), zero_mask.squeeze(-1)
                        )
                        original_data = original_data.unsqueeze(-1)
                        log_det = log_det.unsqueeze(-1)
                        zig_loss = criterion(
                            theta,
                            k,
                            loc=loc,
                            q=q,
                            target=original_data,
                            zero_mask=zero_mask,
                            nonzero_mask=nonzero_mask,
                        )[0]
                        zig_loss = -1 * loss_scale * (zig_loss + log_det)
                else:
                    zig_loss = (
                        -1
                        * loss_scale
                        * (
                            criterion(
                                theta,
                                k,
                                loc=loc,
                                q=q,
                                target=original_data,
                                zero_mask=zero_mask,
                                nonzero_mask=nonzero_mask,
                            )[0]
                        )
                    )

                zig_loss.masked_fill_(~neuron_mask, 0)
                zig_loss = zig_loss.sum() + regularizers

                zig_loss = zig_loss / (
                    n_samples
                )  # zigloss is in that case an MC approximate for the mean of log p(y|x,z)
                # calculate KL divergence between Gaussian prior and approximate posterior

                kl_divergence = kl_divergence_gaussian(
                    means, sigma
                )  # * q.shape[2] #kl_divergence is constant across neuron dimension since latent is the same for all neurons

                differences = means[:, 1:] - means[:, :-1]
                neighbor_loss = torch.norm(differences, p=2, dim=2).sum()

                # average loss over batch_size and time
                zig_loss = (
                    1
                    / (means.shape[0] * means.shape[1])
                    * (zig_loss + 5 * kl_divergence)
                )  # loss is ElBO = p(y|x,z) + KL(q(z|x),p(z)), log_det=0 if no flow is applied

            else:
                zig_loss = (
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
                    + k_regularized
                )

            if loss_function == "combinedLoss":
                pos_los = (
                    loss_scale
                    * criterion_pos(
                        fitted_zig_mean(theta, k, loc, q),
                        original_data,
                    )
                    + regularizers
                )

                loss = zig_loss + 200 * pos_los
                return loss, zig_loss, pos_los

            else:  # only zig loss
                if len(model_output) > 4:
                    if model.flow:
                        if model.flow_base == "Gaussian":
                            return (
                                zig_loss,
                                kl_divergence,
                                log_det.unsqueeze(-1)
                                .repeat(1, 1, 1, n_samples)
                                .masked_fill_(~neuron_mask, 0)
                                .sum(),
                            )
                        else:
                            return (
                                zig_loss,
                                kl_divergence,
                                log_det.repeat(1, 1, 1, n_samples)
                                .masked_fill_(~neuron_mask, 0)
                                .sum(),
                            )
                    else:
                        return zig_loss, kl_divergence
                else:
                    return zig_loss
            """
            return (
                -1*loss_scale
                * criterion(theta, k,
                            loc=loc, 
                            q=q, 
                            target=original_data, 
                            zero_mask=zero_mask, 
                            nonzero_mask=nonzero_mask)[0].sum()
                + regularizers
            )
            """
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

    ##### Model training ####################################################################################################
    model.to(device)
    set_random_seed(seed)
    model.train()
    if loss_function == "ZIGLoss" or loss_function == "combinedLoss":
        if model.flow:
            print(model.flow_base)
            if model.flow_base == "Gaussian":
                zif_loss_instance = zero_inflated_losses.ZIFLoss()
                criterion = zif_loss_instance.get_slab_logl
                print("Gaussian Loss")
            else:
                zig_loss_instance = zero_inflated_losses.ZIGLoss()
                criterion = zig_loss_instance.get_slab_logl
        else:
            zig_loss_instance = zero_inflated_losses.ZIGLoss()
            criterion = zig_loss_instance.get_slab_logl

        if loss_function == "combinedLoss":
            criterion_pos = getattr(modules, "PoissonLoss")(avg=avg_loss)
    else:
        criterion = getattr(modules, loss_function)(avg=avg_loss)
    stop_closure = partial(
        getattr(scores, stop_function),
        dataloaders=dataloaders["oracle"],
        device=device,
        per_neuron=False,
        avg=True,
        deeplake_ds=deeplake_ds,
        flow=model.flow,
        cell_coordinates=cell_coordinates if model.position_features else None,
    )

    n_iterations = len(LongCycler(dataloaders["train"]))

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr_init)
    # Define the optimizer to only include parameters that require gradients
    # optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=lr_init)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max" if maximize else "min",
        factor=lr_decay_factor,
        patience=scheduler_patience,
        threshold=tolerance,
        min_lr=min_lr,
        verbose=verbose,
        threshold_mode="abs",
    )

    # scheduler = WarmupCosineAnnealingLR(optimizer, warmup_steps=warmup_steps, max_lr=lr_init, min_lr=min_lr, T_max=T_max,repeat_warmup = False)

    # set the number of iterations over which you would like to accummulate gradients
    optim_step_count = (
        len(dataloaders["train"].keys())
        if loss_accum_batch_n is None
        else loss_accum_batch_n
    )
    print(f"optim_step_count = {optim_step_count}")

    if use_wandb:
        wandb.init(
            project=wandb_project,
            entity=wandb_entity,
            # We pass a run name (otherwise it’ll be randomly assigned, like sunshine-lollypop-10)
            name=wandb_name,
            # Track hyperparameters and run metadata
            config={
                "learning_rate": lr_init,
                "architecture": wandb_model_config,
                "dataset": wandb_dataset_config,
                "cur_epochs": max_iter,
                "starting epoch": epoch,
                "lr_decay_steps": lr_decay_steps,
                "lr_decay_factor": lr_decay_factor,
                "min_lr": min_lr,
            },
        )

        wandb.define_metric(name="Epoch", hidden=True)
        wandb.define_metric(name="Batch", hidden=True)

    batch_no_tot = 0
    ema_values = []
    best_validation_correlation = 0
    # train over epochs
    for epoch, val_obj in early_stopping(
        model,
        stop_closure,
        interval=interval,
        patience=patience,
        start=epoch,
        max_iter=max_iter,
        maximize=maximize,
        tolerance=tolerance,
        restore_best=restore_best,
        scheduler=scheduler,
        lr_decay_steps=lr_decay_steps,
    ):

        # executes callback function if passed in keyword args
        if cb is not None:
            cb()

        # train over batches
        optimizer.zero_grad(set_to_none=True)
        epoch_loss = 0
        epoch_val_loss = 0
        for batch_no, (data_key, data) in tqdm(
            enumerate(LongCycler(dataloaders["train"])),
            total=n_iterations,
            desc="Epoch {}".format(epoch),
        ):
            batch_no_tot += 1
            batch_args = list(data)

            batch_kwargs = data._asdict() if not isinstance(data, dict) else data
            loss = full_objective(
                model,
                dataloaders["train"],
                data_key,
                *batch_args,
                **batch_kwargs,
                detach_core=detach_core,
            )[0]
            if loss_function == "combinedLoss":
                loss = loss[0]

            loss.backward()
            epoch_loss += loss.detach()
            if (batch_no + 1) % optim_step_count == 0:
                optimizer.step()
                # scheduler.step(batch_increment=1)
                # lr = optimizer.param_groups[0]['lr']
                # print(lr,"lr")
                optimizer.zero_grad(set_to_none=True)

        model.eval()
        ###
        print(epoch_loss / 225, "loss", epoch, "epoch")
        lr = optimizer.param_groups[0]["lr"]
        if model.position_features:
            positions = cell_coordinates[data_key]
        else:
            positions = None

        model_output2 = model(
            batch_args[0][:, 0:1].to(device),
            data_key=data_key,
            out_predicts=False,
            positions=positions,
            **batch_kwargs,
        )
        theta = model_output2[0]
        k = model_output2[1]
        loc = model_output2[2]
        q = model_output2[3]
        if len(model_output2) > 4:
            latent_means = model_output2[4]
            sigma_squared2 = model_output2[5]
            # if not model.position_features:
            # latent_feature_q = model_output2[6][data_key+"_q"]
            # latent_feature_theta = model_output2[6][data_key+"_theta"]
            n_samples = model_output2[7]

        print("theta")
        print(theta)
        print(torch.mean(theta))
        print("q")
        print(q)
        print(torch.mean(q))
        print(" ")
        if len(model_output2) > 4:
            print("means")
            print(latent_means)
            print(torch.mean(latent_means))
            print("sigma")
            print(sigma_squared2)
            # print("latent_feature_q")
            # print(latent_feature_q)
            # print(torch.mean(latent_feature_q))
            # print("latent_feature_theta")
            # print(latent_feature_theta)
            # print(torch.mean(latent_feature_theta))

        ###
        ## after - epoch-analysis

        validation_correlation = get_correlations(
            model,
            dataloaders["oracle"],
            device=device,
            as_dict=False,
            per_neuron=False,
            deeplake_ds=deeplake_ds,
            flow=model.flow,
            cell_coordinates=cell_coordinates if model.position_features else None,
        )

        if save_checkpoints:
            if validation_correlation > best_validation_correlation:
                torch.save(
                    # model.state_dict(), f"{checkpoint_save_path}best.pth"
                    {"model_state_dict": model.state_dict()},
                    "model.pth",
                )
                best_validation_correlation = validation_correlation

        if loss_function == "PoissonLoss" or (not model.latent):
            val_loss = full_objective(
                model,
                dataloaders["oracle"],
                data_key,
                *batch_args,
                **batch_kwargs,
                detach_core=detach_core,
            )
        else:
            if model.flow:
                val_loss, kl_div, log_det = full_objective(
                    model,
                    dataloaders["oracle"],
                    data_key,
                    *batch_args,
                    **batch_kwargs,
                    detach_core=detach_core,
                )
            else:
                val_loss, kl_div = full_objective(
                    model,
                    dataloaders["oracle"],
                    data_key,
                    *batch_args,
                    **batch_kwargs,
                    detach_core=detach_core,
                )

        # torch.save(
        # model.state_dict(), f"toymodels2/temp_save.pth"
        # )

        print(
            f"Epoch {epoch}, Batch {batch_no}, Train loss {loss}, Validation loss {val_loss}"
        )
        print(f"EPOCH={epoch}  validation_correlation={validation_correlation}")

        ema_values.append(validation_correlation)
        ema = calculate_ema(torch.tensor(ema_values), ema_span)[-1]

        linear_layer = model.encoder.linear[data_key]
        reg_term = linear_layer.weight.abs().sum()

        if use_wandb:
            wandb_dict = {
                "Epoch Train loss": epoch_loss,
                "Batch": batch_no_tot,
                "Epoch": epoch,
                "validation_correlation": validation_correlation,
                # "log_det": log_det,
                "Epoch validation loss": val_loss,
                "EMA validation loss": ema,
                # "Poisson Loss": pos_loss,
                "ZIG Loss": val_loss,
                "Epoch": epoch,
                # "Theta First": theta_first,
                # "Theta Last": theta_last,
                "Theta Mean": torch.mean(theta),
                # "Loc First": loc_first,
                # "Loc Last": loc_last,
                # "Loc Mean": torch.mean(loc),
                # "K First": k_first,
                # "K Last": k_last,
                # "K Mean": torch.mean(k),
                # "Q First": q_first,
                # "Q Last": q_last,
                "Q Mean": torch.mean(q),
                "Learning rate": lr,
                "kl_divergence": kl_div,
                "latent_means": torch.mean(latent_means),
                "latent_variance": latent_means.var(),
                "latent_max": latent_means.max(),
                "latent_min": latent_means.min(),
                "latent_sigma": sigma_squared2,
                # "latent_feature_q": torch.norm(latent_feature_q, dim = 0).mean(),
                # "latent_feature_theta": torch.norm(latent_feature_theta, dim = 0).mean(),
                "regularization term": reg_term,
                # "log_likelihood": log_likelihood,
                # "prior_correlation": validation_correlation_uncut,
            }
            wandb.log(wandb_dict)

            # for hyperparameter search
            if hyper:
                hyper.report(ema, epoch)
                if hyper.should_prune():
                    if use_wandb:
                        wandb.finish()
                    raise optuna.exceptions.TrialPruned()

        model.train()

    ##### Model evaluation ####################################################################################################
    model.eval()
    # if save_checkpoints:
    # torch.save(model.state_dict(), f"{checkpoint_save_path}final.pth")

    # Compute avg validation and test correlation
    validation_correlation = get_correlations(
        model,
        dataloaders["oracle"],
        device=device,
        as_dict=False,
        per_neuron=False,
        deeplake_ds=deeplake_ds,
        flow=model.flow,
        cell_coordinates=cell_coordinates if model.position_features else None,
    )
    print(f"\n\n FINAL validation_correlation {validation_correlation} \n\n")

    output = {}
    output["validation_corr"] = validation_correlation

    score = np.mean(validation_correlation)
    if use_wandb:
        wandb.finish()

    # removing the checkpoints except the last one
    # to_clean = os.listdir(checkpoint_save_path)
    to_clean = os.listdir("toymodels")
    for f2c in to_clean:
        if "epoch" in f2c:
            os.remove(os.path.join("toymodels", f2c))

    if model.encoder.elu:
        non_linearity = True
    else:
        non_linearity = False

    return score


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
# conv_out = {}
# conv_out["hidden_channels"] = [4,2]
# conv_out["input_kernel"] = 5
# conv_out["hidden_kernel"] = [(11,11),(15,15)]


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
    # grid_mean_predictor = None,
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
    data_loaders_nobehavior,
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

output_dim = 12
dropout = "across_time"
dropout_prob = 0.5
encoder_dict = {}
encoder_dict["input_dim"] = max_neurons
encoder_dict["hidden_dim"] = 42  # 42
encoder_dict["hidden_gru"] = 20  # 20
encoder_dict["output_dim"] = output_dim
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
decoder_dict["hidden_dim"] = output_dim
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

for data_key in data_keys:
    cell_coordinates[data_key] = cell_coordinates[data_key][
        :, 0 : position_mlp["input_size"]
    ]  # adapt cell_coordinates to MLP input (either xy corrdinates or xyz)


zig_model = ZIGEncoder(
    core=factorised_3d_model.core,
    readout=factorised_3d_model.readout,
    # shifter = factorised_3d_model.shifter,
    shifter=None,
    k_image_dependent=False,
    loc_image_dependent=False,
    mle_fitting=mean_variance_dict,
    latent=True,
    encoder=encoder_dict,
    # decoder = decoder_dict,
    norm_layer="layer_flex",
    non_linearity=True,
    dropout=dropout,
    dropout_prob=dropout_prob,
    future_prediction=False,
    flow=False,
    # position_features = position_mlp,
    # behavior_in_encoder = behavior_mlp
)
zig_model.load_state_dict(
    torch.load("models/zig_best.pth", map_location=device), strict=False
)
# Print all keys in the loaded state dictionary


print("Out_dim", encoder_dict["output_dim"])
lr_inint = 5e-3
min_lr = 1e-5


def round_to_two_non_zeros(x):
    import math

    # Find the first non-zero digit
    first_non_zero = int(math.floor(-math.log10(abs(x)))) + 1
    # Shift decimal point to make first two significant digits before the decimal
    shifted = x * (10**first_non_zero)
    # Round to two significant figures
    rounded = round(shifted, 1)
    # Shift back
    final_value = str(rounded) + "e-" + str(first_non_zero)
    return final_value


validation_score = standard_trainer(
    zig_model,
    data_loaders,
    111,
    use_wandb=True,
    wandb_project="toymodel",
    wandb_name="12dim_test_nodecoder",
    wandb_entity=None,
    # loss_function= "combinedLoss",
    loss_function="ZIGLoss",
    # loss_function= "PoissonLoss",
    verbose=True,
    lr_decay_steps=4,
    lr_init=lr_inint,
    min_lr=min_lr,
    # T_max=T_max,
    # warmup_steps= warmup_steps,
    device=device,
    patience=12,  # 12#8,
    scheduler_patience=10,  # 10#6,
    # k_reg = mean_variance_dict,
    checkpoint_save_path="toymodels2/12dim_test_nodecoder",
)
validation_score
torch.cuda.empty_cache()
torch.cuda.empty_cache()
import datetime

print(datetime.datetime.now())
print("mew")


"""
print("latent_dec")
def objective(trial):
    # Hyperparameters to tune
    lr_inint = 0.005 #trial.suggest_float("lr", 9e-4, 9e-3, log=False)
    #scheduler_patience  =  trial.suggest_int("decay_patience",2,9)
    #lr_decay_factor = trial.suggest_float("decay_factor",0.1,0.9, log = True)
    print(lr_inint, "lr")


    #warmup_steps = 225 #trial.suggest_int("warmup_steps", 200, 1000)
    #T_max = trial.suggest_int("T_max", 1, 5)
    #min_lr_factor = trial.suggest_float("min_lr", 9e-3, 5e-1, log=True)
    #min_lr = lr_inint * min_lr_factor #find the best scale between min and max lr
    encoder_dict = {}
    encoder_dict["input_dim"] = max_neurons
    encoder_dict["hidden_dim"] =  42 #trial.suggest_int("hidden_dim", 200, 400)
    encoder_dict["hidden_gru"] = 20 #trial.suggest_int("hidden_gru", 100, encoder_dict["hidden_dim"])
    encoder_dict["output_dim"] = 12 #trial.suggest_int("hidden_output", 5, encoder_dict["hidden_gru"])
    encoder_dict["n_samples"] = 100 #250 // encoder_dict["output_dim"] * 3
    encoder_dict["mice_dim"] = 0 #encoder_dict["hidden_dim"] // 3 #trial.suggest_int("mice_dim",0, encoder_dict["hidden_dim"]-1)
    # CNN-specific hyperparameters
    use_cnn = True  # Use CNN for encoder
    residual = trial.suggest_categorical("residual", [True, False])  # Use residual connections
    n_layers = trial.suggest_int("n_layers", 2, 12)  # Number of CNN layers
    kernel_size = trial.suggest_int("kernel_size", 2, 16)  # Kernel size (first layer is double)
    channel_size_first_half = trial.suggest_int("channel_size_first_half", 20, 128)
    channel_size_second_half = trial.suggest_int("channel_size_second_half", 20, 128)

    encoder_dict["use_cnn"] = use_cnn
    encoder_dict["use_resnet"] = False
    encoder_dict["pretrained"] = False #for resnet pretraining
    encoder_dict["residual"] = residual
    encoder_dict["hidden_layers"] = n_layers 
    encoder_dict["kernel_size"] = [kernel_size * 2] + [kernel_size] * (n_layers - 1)
    encoder_dict["channel_size"] = [channel_size_first_half] * (n_layers // 2) + [channel_size_second_half] * (n_layers // 2 -1) + [encoder_dict["hidden_gru"]]
    
    decoder_dict = {}
    decoder_dict["hidden_dim"] = 12
    decoder_dict["hidden_layers"] = 2
    decoder_dict["use_cnn"] = True
    decoder_dict["kernel_size"] = [5,11]
    decoder_dict["channel_size"] = [12,12]

    normal_layer = "layer_flex"  # Fixed normal layer


    factorised_3d_model = make_video_model(
    data_loaders,
    seed,
    core_dict=factorised_3D_core_dict,
    core_type='3D_factorised',
    readout_dict=readout_dict.copy(),
    readout_type='gaussian',               
    use_gru=False,
    gru_dict=None,
    use_shifter=False,
    shifter_dict=shifter_dict,
    shifter_type='MLP',
    deeplake_ds=False,
)


    zig_model = ZIGEncoder(core = factorised_3d_model.core, 
                        readout=factorised_3d_model.readout, 
                        shifter = None,
                        k_image_dependent=False,
                        loc_image_dependent = False,
                        mle_fitting = mean_variance_dict,
                        latent = True,
                        encoder = encoder_dict,
                        decoder = decoder_dict,
                        norm_layer = normal_layer,
                        dropout = True
                        )
    zig_model.load_state_dict(torch.load('toymodels/zig_nobehaviorbest.pth', map_location=device),strict=False)

    print("new model created")
    def round_to_two_non_zeros(x):
        import math
        # Find the first non-zero digit
        first_non_zero = int(math.floor(-math.log10(abs(x)))) + 1
        # Shift decimal point to make first two significant digits before the decimal
        shifted = x * (10 ** first_non_zero)
        # Round to two significant figures
        rounded = round(shifted, 1)
        # Shift back
        final_value = str(rounded)+"e-"+str(first_non_zero)
        return final_value


    validation_score = standard_trainer(zig_model,data_loaders,111,
                                                                use_wandb = True,
                                                                wandb_project="hyperparameter",
                                                                wandb_name='CNN'+"layers"+str(n_layers)+"channels1_"+str(channel_size_first_half)+"channels2_"+str(channel_size_second_half)+"kernel"+str(kernel_size)+"res"+str(residual),
                                                                wandb_entity = None,
                                                                #loss_function= "combinedLoss",
                                                                loss_function="ZIGLoss",
                                                                #loss_function= "PoissonLoss",
                                                                verbose = True,
                                                                lr_decay_steps = 5,
                                                                lr_init = lr_inint,
                                                                #lr_decay_factor = lr_decay_factor,
                                                                #scheduler_patience = scheduler_patience,
                                                                min_lr=0.00001, 
                                                                #T_max=T_max,
                                                                #warmup_steps= warmup_steps,
                                                                device = device,
                                                                patience= 10,
                                                                max_iter = 30, #train at most .. epochs each run in hyperparameter search
                                                                save_checkpoints= False,
                                                                hyper = trial,
                                                                checkpoint_save_path= 'toymodels/fef')

    return validation_score

study = optuna.create_study(direction="maximize",pruner=optuna.pruners.MedianPruner(n_startup_trials=2, n_warmup_steps=7 ))
study.optimize(objective, n_trials=15)

# Plot Optimization History
fig_opt_history = plot_optimization_history(study)
fig_opt_history.write_image('hyperparameters/cnn.png')

# Plot Parameter Importances
fig_param_importances = plot_param_importances(study)
fig_param_importances.write_image('hyperparameters/cnn.png')

# Plot Slice
fig_slice = plot_slice(study)
fig_slice.write_image('hyperparameters/cnn.png')


# Save the best hyperparameters
best_params = study.best_trial.params
with open("hyperparameters/cnn.json", "w") as f:
    json.dump(best_params, f)

print("Best Hyperparameters:", best_params)

#Plot counter
fig_counter = plot_contour(study)
fig_counter.write_image('hyperparameters/cnn.png')
"""
