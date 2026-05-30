
# Modeling Dynamic Neural Activity by combining Naturalistic Video Stimuli and Stimulus-independent Latent Factors

NeurIPS 2025. We propose a probabilistic model that predicts the joint distribution of the neuronal responses from video stimuli and stimulus-independent latent factors. After training and testing our model on mouse V1 neuronal responses, we find that it out- performs video-only models in terms of log-likelihood and achieves improvements in likelihood and correlation when conditioned on responses from other neurons. Furthermore, we find that the learned latent factors strongly correlate with mouse behavior and that they exhibit patterns related to the neurons’ position on the visual cortex, although the model was trained without behavior and cortical coordinates. Our findings demonstrate that unsupervised learning of latent factors from population responses can reveal biologically meaningful structure that bridges sensory processing and behavior, without requiring explicit behavioral annotations during training.

---

## Setup

### 1. Clone this repository

```bash
git clone https://github.com/FinnSchmidt01/latent_space_model.git
cd latent_space_model
```

### 2. Create a Python environment and install dependencies

We recommend Python 3.10 or 3.11. Using a conda environment:

```bash
conda create -n latent_space python=3.10 -y
conda activate latent_space
pip install -r requirements.txt
```

### 3. Install the neuralpredictors fork

This repository depends on a custom fork of [neuralpredictors](https://github.com/FinnSchmidt01/neuralpredictors) on the `latent_space_model` branch. The PyPI release does **not** include the required `ZIGEncoder` implementation.

```bash
git clone https://github.com/FinnSchmidt01/neuralpredictors.git
cd neuralpredictors
git checkout latent_space_model
pip install -e .
cd ..
```

### 4. Fix nnfabrik compatibility with Python 3.10+

`nnfabrik 0.2.1` has two import bugs on Python ≥ 3.10 that must be patched manually. Find the installed file:

```bash
python -c "import nnfabrik; print(nnfabrik.__file__)"
# e.g. /path/to/site-packages/nnfabrik/__init__.py
# patch is in: /path/to/site-packages/nnfabrik/utility/dj_helpers.py
```

Open `nnfabrik/utility/dj_helpers.py` and apply two edits:

**Fix 1** — `Iterable` and `Mapping` were moved to `collections.abc` in Python 3.10:
```python
# Replace:
from collections import OrderedDict, Iterable, Mapping

# With:
from collections import OrderedDict
from collections.abc import Iterable, Mapping
```

**Fix 2** — `datajoint 2.x` moved `Schema`; extend the existing try/except to add a third fallback:
```python
# Replace:
try:
    from datajoint.schema import Schema
except:
    from datajoint.schemas import Schema

# With:
try:
    from datajoint.schema import Schema
except ImportError:
    try:
        from datajoint.schemas import Schema
    except ImportError:
        from datajoint import Schema
```

---

## Data

Download the Sensorium 2023 dataset from [GIN](https://gin.g-node.org/pollytur/sensorium_2023_dataset). The expected folder structure is:

```
data/
├── dynamic29515-10-12-Video-9b4f6a1a067fe51e15306b9628efea20/
├── dynamic29623-4-9-Video-9b4f6a1a067fe51e15306b9628efea20/
├── dynamic29647-19-8-Video-9b4f6a1a067fe51e15306b9628efea20/
├── dynamic29712-5-9-Video-9b4f6a1a067fe51e15306b9628efea20/
└── dynamic29755-2-8-Video-9b4f6a1a067fe51e15306b9628efea20/
```

Each session folder must contain `data/responses/`, `data/videos/`, `data/behavior/`, and `meta/neurons/cell_motor_coordinates.npy`.

---

## Preprocessing: compute neuron statistics

Before running evaluation or training, compute per-neuron mean, variance, and shape parameter `k` of the Gamma distribution. These are required by the ZIG model for moment-matched initialisation.

```bash
python moments.py --data_dir /path/to/data
```

This writes `new_mean.npy`, `new_variance.npy`, and `k_fitted.npy` into each session folder. Only needs to be run once.

---

## Evaluation

Run evaluation with:

```bash
python eval.py --data_dir /path/to/data --model_path /path/to/checkpoint.pth [options]
```

**Key arguments:**

| Argument | Default | Description |
|---|---|---|
| `--data_dir` | required | Root folder containing the `dynamic*` session folders |
| `--model_path` | required | Path to a `.pth` checkpoint |
| `--latent_dim H G O` | `42 20 12` | Encoder dims: hidden, GRU hidden, output (latent k) |
| `--no_latent` | off | Evaluate a pure ZIG model without latent space |
| `--with_decoder` | off | Enable the GRU decoder (use when the checkpoint was trained with one) |

Outputs **prior correlation** (latent sampled from p(z)), **conditioned correlation** (latent inferred from portion of neurons, which are not used in evaluation), and **log-likelihood**.

**Pretrained models** (included in `models/`):

| File | Type | Latent dim |
|---|---|---|
| `models/zig_best.pth` | ZIG baseline (no latent) | — |
| `models/latent_12dim_zigbest.pth` | ZIG + 12-dim latent | 42 → 20 → 12 |

**Example commands:**

```bash
# ZIG baseline — no latent space
python eval.py --data_dir data/ --model_path models/zig_best.pth --no_latent

# 12-dim latent model (decoder weights in checkpoint are discarded at inference)
python eval.py --data_dir data/ --model_path models/latent_12dim_zigbest.pth
```

---

## Training

Edit `latent_space_model.py` to set `paths` (data directories) and `base_dir` (for moment loading) to your data location, then run:

```bash
python latent_space_model.py
```

The script trains a latent ZIG model and logs to [Weights & Biases](https://wandb.ai). Set `use_wandb=False` in the `standard_trainer` call to disable logging.

**Model variants** (configure inside the script):

| Model | `zig` in readout | `latent` in ZIGEncoder | `loss_function` |
|---|---|---|---|
| Poisson baseline | `False` (`out_channels=1`) | — | `"PoissonLoss"` |
| ZIG video-only | `True` (`out_channels=2`) | `False` | `"ZIGLoss"` |
| ZIG + latent | `True` (`out_channels=2`) | `True` | `"ZIGLoss"` |

The best checkpoint (by oracle correlation) is saved to `model.pth` in the working directory. To adjust the latent dimension change `output_dim` in `encoder_dict`.
