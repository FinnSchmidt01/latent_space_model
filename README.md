
# Modeling Dynamic Neural Activity by combining Naturalistic Video Stimuli and Stimulus-independent Latent Factors

## Requirements

To install requirements:
```setup
conda env create -f environment.yml
```
After setting up the environment one has to install two additional repositories as packages. Download the neuralpredictors repository [here](https://anonymous.4open.science/r/neuralpredictors-D2FE/):
```bash
# Clone the nnfabrik repository
git clone git@github.com:sinzlab/nnfabrik.git

# Install Neuralpredictors in editable mode
cd neuralpredictors
git checkout latent_space_model
pip install -e .

# Install nnfabrik in editable mode
cd nnfabrik
pip install -e .
```
Before using the training/evaluation scripts, download the mice data [here](https://gin.g-node.org/pollytur/sensorium_2023_dataset) and run 
```
python moments.py 
```
to create numpy files containing the means and variances of neurons responses. 
Data paths must be adapted before running any python script. 

## Training

To train the model(s) in the paper, run the `latent_space_model.py` file. Aside from the adjustments listed below, all other parameters are identical to those used in the paper’s models.

- For training the Poisson baseline model, use the `factorised_3d_model` and `loss_function=PoissonLoss` in the `standard_trainer` function. Set `out_channels=1` and `zig=False` in the `readout_dict`. No other parameters have to be changed. 

- For training the video-only ZIG model, use the `zig_model` and set `latent = False` in the `ZIGEncoder` class. Further, use `loss_function=ZIGLoss` in the standard trainer function.
  
- For training a latent model, use the `zig_model` and set `latent = True` in the `ZIGEncoder` class. Further, use `loss_function=ZIGLoss` in the standard trainer function. To adjust the latent dimension $$k$$ change `output_dim=12`, by default $$k=12$$.
  
- For training a latent model, where the mapping of the latent feature vectors is determined based on cortical positions, additionally set `position_features = position_mlp` in the ZIGEncoder class.



## Evaluation

To evaluate the models in terms of log_likelihood and correlation, run

```eval
python eval.py 
```
- To evaluate the Poisson baseline model, set `out_channels = 1`.
  
- To evaluate  the video-only ZIG model, keep `out_channels = 2` but set `latent = False`.
  
- To evaluate a latent model, keep `latent = True` and adjust the `latent_dim` if needed.
  
- To evaluate latent models without cortical positions, set `neuron_position_info = False`.  For the latent model, which maps the latent feature vectors based on cortical positions, set
```
position_mlp = {
             "input_size": 3,
             "layer_sizes": [6, 12]
          }
```

## Pre-trained Models

You can download pretrained models here:

- [My awesome model](https://drive.google.com/mymodel.pth) trained on ImageNet using parameters x,y,z. 
