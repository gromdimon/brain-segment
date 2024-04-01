# The Investigation
This document contains the results of our investigations into each of the architectures, assessing their suitability for our specific requirements.
The candidates are the following:

- U-Net
- UNETR
- ResNet

## U-Net

TODO

## UNETR

### Overview

UNETR, short for UNet with Transformer, is a novel architecture that combines the strengths of both UNet and Transformer models. It aims to improve semantic segmentation tasks, especially in scenarios where there are long-range dependencies between pixels.

### Sources and Useful Resources

- [Original paper](https://arxiv.org/abs/2103.10504)
- [Code implementation](https://github.com/tamasino52/UNETR)

### Key Features

1. **Integration of UNet and Transformer**: UNETR integrates the architecture of UNet, known for its effective feature extraction and localization, with the self-attention mechanism of the Transformer, which excels in capturing long-range dependencies.

2. **Self-Attention Mechanism**: By incorporating self-attention mechanisms, UNETR can effectively capture global context information, allowing it to make more informed segmentation decisions, especially in cases where distant pixels influence each other.

3. **Hierarchical Downsampling and Upsampling**: Similar to UNet, UNETR employs a hierarchical architecture with downsampling and upsampling paths, enabling it to capture features at multiple scales while maintaining high-resolution segmentation outputs.

4. **Positional Encoding**: To enable the Transformer to handle spatial information, UNETR incorporates positional encoding techniques, allowing it to effectively process the spatial relationships between pixels.

### Performance

UNETR is much larger than traditional UNet models due to the inclusion of Transformer layers (self-Attention). This increased complexity allows UNETR to capture long-range dependencies and spatial relationships more effectively, leading to improved segmentation performance. However, this comes at the cost of increased computational resources, higher memory usage and training time.

### Possible Imrovements

1. **Precision of Floats**: The precision of the floats used in the model can be reduced to reduce the memory usage and training time. float16 can be used instead of float32.
2. **Model Pruning**: Model pruning can be used to reduce the number of parameters in the model, which can help reduce memory usage and training time.
3. **Quantization**: Quantization can be used to reduce the precision of the weights and activations in the model, which can help reduce memory usage and training time.
4. **Distillation**: Knowledge distillation can be used to train a smaller model to mimic the behavior of the larger model, which can help reduce memory usage and training time.


## ResNet

### Sources and Useful Resources
- [Original paper](https://arxiv.org/abs/1512.03385)
- https://medium.com/@ibtedaazeem/understanding-resnet-architecture-a-deep-dive-into-residual-neural-network-2c792e6537a9
- https://en.wikipedia.org/wiki/Residual_neural_network
- https://cv-tricks.com/keras/understand-implement-resnets/


### Overview
ResNet is a deep learning model used for computer vision applications where the weight layers learn residual functions with reference to the layer inputs.

The key innovation consists in the introduction of residual connections (or skip connections). Whereas traditionally in neural networks, each layer tries to learn a specfic mapping from its input to its output, these connections allow the network to learn residual functions instead.

Thereby, the usual problem of vanishing gradients or degradation with increasing network depth is addressed by introducing the aforementioned residual connections, which bypass one or more layers. Instead of learning the mappings directly, it learns the residual mappings, which are the difference between the input and the output of the layer.

### History

### Residual Learning

The key component of ResNet is residual learning. In a normal neural network, the input would be transformed by a set of convolutional layers before being passed to the activation function. In a residual network however, the input to the block is added to the output of the block, creating a residual connection. We denote the output of the residual block as $H(x)$, where $x$ is the input. The output of $H(x)$ can be represented by:

$H(x) = F(x) + x$

$F(x)$ represents the residual mapping learned by the network. The presence of identity term $x$ allows the gradient to flow more easily.

### Complexity
ResNet comes in numerous variations such as ResNet-50 (the original architecture), ResNet-18, -34, -101, 110 and many more.

The number denotes the number of neural network layers in that specific architecture.

### Performance
![alt text](img/resnetperformance.png)

The above figure shows the training/validation error rate after training on the dataset Imagenet. Left is a plain network, right is ResNet. The thick curve represents validation error, whereas the thin curve represents training error.