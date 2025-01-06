# dynamic-brain-state

This repository provides code for evaluating and training  a latent space model. The model and supporting functionality are implemented using the `neuralpredictors` library. Follow the instructions below to set up and run the evaluation.

The eval.py file demonstrates how the latent space model is loaded and evaluated.

## Installation
The conda environment YAML file for this repository is located in the `infrastructure/scripts` folder.

To get started, clone the `neuralpredictors` repository and install the `latent_space_model` branch to complete the setup:

```bash
# Clone the neuralpredictors repository
git clone git@github.com:FinnSchmidt01/neuralpredictors.git

# Navigate to the repository
cd neuralpredictors

# Checkout the latent_space_model branch
git checkout latent_space_model

# Install the repository in editable mode
pip install -e .

