import os

import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.optimize import root
from scipy.special import digamma
from scipy.stats import gamma


def load_mean_variance(base_dir, device):
    mean_variance_dict = {}

    # List all folders in the base directory
    folders = os.listdir(base_dir)
    for folder in folders:
        if folder.startswith("dynamic"):
            folder_path = os.path.join(base_dir, folder)

            if os.path.isdir(folder_path):
                # Construct paths for mean and variance files
                mean_file_path = os.path.join(folder_path, "new_mean.npy")
                variance_file_path = os.path.join(folder_path, "new_variance.npy")
                k_file_path = os.path.join(folder_path, "k_fitted.npy")
                # Check if the files exist before loading
                if os.path.exists(mean_file_path):
                    mean_array = np.load(mean_file_path)
                    # Convert to PyTorch tensors
                    mean_tensor = torch.tensor(mean_array, dtype=torch.float32).to(
                        device
                    )

                    mean_variance_dict[folder + "_mean"] = mean_tensor

                if os.path.exists(variance_file_path):
                    variance_array = np.load(variance_file_path)
                    variance_tensor = torch.tensor(
                        variance_array, dtype=torch.float32
                    ).to(device)
                    mean_variance_dict[folder + "_variance"] = variance_tensor

                if os.path.exists(k_file_path):
                    k_array = np.load(k_file_path)
                    k_tensor = torch.tensor(k_array, dtype=torch.float32).to(device)
                    mean_variance_dict[folder + "fitted_k"] = k_tensor

    return mean_variance_dict


if __name__ == "__main__":
    # Base directory containing the folders
    base_dir = "/mnt/lustre-grete/usr/u11302/Data/"

    # List of folders
    folders = os.listdir(base_dir)

    # Loop through each folder
    for folder in folders:
        print(folder)
        if folder.startswith("dynamic"):
            folder_path = os.path.join(base_dir, folder) + "/data/responses"

            if os.path.isdir(folder_path):
                # List all .npy files in the folder
                npy_files = [f for f in os.listdir(folder_path) if f.endswith(".npy")]

                # Initialize lists to store the arrays
                arrays = []

                # Load each .npy file and append the array to the list
                for npy_file in npy_files:
                    file_path = os.path.join(folder_path, npy_file)
                    array = np.load(file_path)
                    arrays.append(array)

                # Concatenate arrays along the second dimension (axis 1)
                concatenated_array = np.concatenate(arrays, axis=1)

                # Filter out values smaller than 0.005, this is roughly the threshold of the ZIG distribution. Only values bigger than this threshold are considered in the gamma distribution
                filtered_array = np.where(
                    concatenated_array < 0.005, np.nan, concatenated_array
                )
                filtered_array = np.minimum(filtered_array, 80)  # cut outliers off
                first_row_valid_values = filtered_array[
                    1, ~np.isnan(filtered_array[0, :])
                ]
                k_params = []

                for i in range(filtered_array.shape[0]):
                    neuron_valid_values = filtered_array[
                        i, ~np.isnan(filtered_array[i, :])
                    ]
                    # Fix the loc parameter to 0.05
                    params = gamma.fit(neuron_valid_values, floc=0.005)

                    # Get the shape and scale parameters
                    k, loc, theta = params
                    k_params.append(k)

                # Get the shape, location, and scale parameters
                k, loc, theta = params
                x = np.linspace(
                    min(first_row_valid_values), max(first_row_valid_values), 1000
                )
                fitted_gamma = gamma.pdf(x, k, loc, theta)

                # Plot the valid values of the first row with a histogram and the fitted gamma distribution
                plt.figure(figsize=(10, 6))
                plt.hist(
                    first_row_valid_values,
                    bins=50,
                    density=True,
                    edgecolor="black",
                    alpha=0.6,
                    label="Histogram of Valid Values",
                )
                # plt.plot(x, fitted_gamma, 'r-', label=f'Fitted Gamma Distribution\nk={k:.2f}, loc={loc:.2f}, theta={theta:.2f}')
                plt.xlabel("Value")
                plt.ylabel("Density")
                plt.title("Histogram of Valid Values of the second neuron")
                plt.legend()

                plt.savefig("capped_responses" + folder + ".png")
                plt.clf()

                # Calculate mean and variance along the first dimension (axis 1) ignoring nan values
                mean_array = np.nanmean(filtered_array, axis=1)
                variance_array = np.nanvar(filtered_array, axis=1)

                # Save the mean and variance arrays
                mean_file_path = os.path.join(base_dir + folder, "new_mean.npy")
                variance_file_path = os.path.join(base_dir + folder, "new_variance.npy")

                np.save(mean_file_path, mean_array)
                np.save(variance_file_path, variance_array)

                k_file_path = os.path.join(base_dir, folder, "k_fitted.npy")
                # np.save(k_file_path, np.array(k_params))

    print("Mean and variance files have been saved for each folder.")
