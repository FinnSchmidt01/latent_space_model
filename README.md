# dynamic-brain-state

This repository provides code for evaluating and training  a latent space model. The model and supporting functionality are implemented using the `neuralpredictors` library. Follow the instructions below to set up and run the evaluation.

The eval.py file demonstrates how the latent space model is loaded and evaluated.

## Installation

To get started, one needs to additionally clone the `neuralpredictors` repository and install the required branch (`latent_space_model`):

```bash
# Clone the neuralpredictors repository
git clone git@github.com:FinnSchmidt01/neuralpredictors.git

# Navigate to the repository
cd neuralpredictors

# Checkout the latent_space_model branch
git checkout latent_space_model

# Install the repository in editable mode
pip install -e .

