# qytorch
aka Quaternion Torch

The `qytorch` package is a specialized deep learning library built on top of PyTorch. It introduces Quaternion neural network layers for several commonly used modules in `torch.nn`. These quaternion layers offer a significant advantage by reducing the number of weights to just one-fourth of those in their real-valued counterparts, while maintaining full feature support.

## Installation

To install the `qytorch` package, use the following command:
```bash
pip install qytorch
```

Note that the `qytorch` package requires `torch` and `numpy` as dependencies. If you don't have them installed, you can install them using the following commands:
```bash
pip install torch numpy
```

## Features

- **QLinear**: Quaternion version of the linear layer.
- **QConv1d, QConv2d, QConv3d**: Quaternion versions of 1D, 2D, and 3D convolutional layers.
- **QMultiheadAttention**: Quaternion multi-head attention mechanism.
- **QTransformerEncoderLayer, QTransformerDecoderLayer**: Quaternion transformer encoder and decoder layers.
- **get_qtransformer**: Utility function to create quaternion transformers.

## Usage

Here's an example of how to use the `qytorch` package to create a quaternion linear layer:

```python
import torch
import qytorch

# Create a quaternion linear layer
qlinear = qytorch.QLinear(in_features=10, out_features=5)

# Create some input data
x = torch.randn(1, 10)

# Pass the input data through the quaternion linear layer
output = qlinear(x)

print(output)
```
