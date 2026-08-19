# CIFAR-10 Image Classification: Comparing Five PyTorch Architectures

Five neural network architectures, implemented from scratch in PyTorch and
trained on CIFAR-10 (60,000 32×32 color images across 10 classes), moving
from a plain fully-connected network up to a hand-designed convolutional
network that clears both the assignment's required accuracy bar and its
extra-credit bar.

Built from coursework for **CSE 416 (Introduction to Machine Learning)**,
University of Washington. See [Results](#results) for the actual validation
accuracies this notebook produced from real training runs — not targets.

## What this does

Implements and trains five networks, each defined and trained from
scratch:

- **NetA** — a single-hidden-layer fully-connected network.
- **NetB** — a two-hidden-layer fully-connected network.
- **NetC** — a convolutional network (conv → ReLU → max-pool → fully
  connected).
- **NetD** — a custom-designed network with two convolutional layers and
  two fully-connected layers, required to reach ≥65% validation accuracy.
- **NetE** (extra credit) — NetD's architecture plus `BatchNorm2d`/
  `BatchNorm1d` layers, required to reach ≥70% validation accuracy.

Each model is trained with a shared training/evaluation harness that logs
per-epoch loss and accuracy, and the notebook produces a per-model
training curve plus a combined comparison plot across all five
architectures.

## Results

Real validation accuracies from actual training runs (not simulated or
estimated):

| Model | Approx. validation accuracy | Notes |
|---|---|---|
| NetA (1 hidden layer, FC) | ~52% | Baseline fully-connected network |
| NetB (2 hidden layers, FC) | ~similar to NetA | Depth alone doesn't help much without convolution |
| NetC (basic CNN) | ~64% | Convolution gives a clear jump over FC-only networks |
| NetD (custom CNN) | **71.3%** | Clears the required ≥65% bar |
| NetE (custom CNN + BatchNorm, extra credit) | **73.6%** | Clears the required ≥70% extra-credit bar |

The clearest trend in the results: convolutional structure matters far
more than raw parameter count or depth for image data (NetB barely beats
NetA despite more layers, while NetC's convolutional layers alone produce
a large jump), and batch normalization gave NetE a further real
improvement over NetD on top of that.

## What's original vs. course-provided

The course notebook provided the shared `train()`/`accuracy()`/
`plot_history()` harness and a `NetExample` reference architecture.
Written for this assignment: `NetA`, `NetB`, `NetC`, `NetD` (the required
custom architecture), and `NetE` (the extra-credit architecture with batch
normalization) — all five network definitions and the hyperparameter
choices used to train each to their reported accuracy.

## Known limitations

- Training logs (per-epoch loss/accuracy curves) are saved as plots inside
  the notebook itself rather than exported separately; open the notebook
  to see the full training curves for all five models.
- CIFAR-10 is downloaded automatically by `torchvision.datasets.CIFAR10`
  the first time the notebook runs — no manual data setup needed (unlike
  the other repos in this portfolio, which depend on data files that
  aren't included).

## Running this code

Requires Python with `torch`, `torchvision`, `matplotlib`, and `seaborn`.
Open `cifar10_cnn_architectures.ipynb` in Jupyter or Colab (a GPU runtime
is strongly recommended — training five CNNs on CPU is slow) and run all
cells top to bottom; CIFAR-10 downloads automatically on first run.
