# CIFAR-10 CNN Architectures in PyTorch

This repository contains a Jupyter notebook that compares five PyTorch image-classification models on CIFAR-10: two fully connected networks and three convolutional networks, including a convolutional model with batch normalization. The notebook includes shared data-loading, training, accuracy-evaluation, and training-history plotting code.

The project uses Python, Jupyter, PyTorch, torchvision, Matplotlib, and Seaborn. CIFAR-10 is downloaded automatically into a local `data/` directory when the notebook runs. The code uses a CUDA GPU when one is available and otherwise runs on the CPU.

## Running

Install the dependencies:

```bash
pip install -r requirements.txt
```

Open `cifar10_cnn_architectures.ipynb` in Jupyter or another compatible notebook environment and run the cells from top to bottom. The notebook trains each model for the configured number of epochs and plots its training and validation accuracy histories.

The course context recorded in the repository is CSE 416, Introduction to Machine Learning, at the University of Washington.

## Credits

Implemented by Sparsh Dadhich.
