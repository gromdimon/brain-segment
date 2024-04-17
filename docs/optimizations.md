## Table of Contents
- [Introduction](#introduction)
- [Data Set](#data-set)
    - [Augment our training set](#augment-our-training-set)
    - [Finding more data](#finding-more-data)
- [Hyperparameters](#hyperparameters)
    - [Epochs](#epochs)
    - [Activation function](#activation-function)
    - [Loss function](#loss-function)
    - [Optimizer](#optimizer)
    - [Learning rate scheduler](#learning-rate-scheduler)
    - [Regularization techniques](#regularization-techniques)
    - [Architecture Adjustments](#architecture-adjustments)
- [Tracking Progress with Weights & Biases](#tracking-progress-with-weights--biases)

## Introduction
Now that we have decided on an architecture, in this document we'll be diving into the core of the subject – namely the optimizations we could make in order to improve this model. We will be discussing what potential modifications we can make to the code in order to improve the performance.

Please remember to use our Weights & Biases to track how each of these changes affect results.

## Data Set
### Augment our training set
One way to augment our training set is by artificially transform our data, including applying:
- Skewing
- Downscaling
- Rotating
- Increase/Decrease contrast (careful with this one)
- Translation
- Reflection
- Adding in noise

to the images. This will help us protect our model against overfitting.

We can use torchvision.transforms in order to help us with this process. Here's an example of some transformations being applied:

```python
import torchvision.transforms as transforms

# Define the mean and standard deviation for normalization
mean = [0.485, 0.456, 0.406]  # ImageNet mean
std = [0.229, 0.224, 0.225]    # ImageNet standard deviation

# Define the image transforms including normalization
transform = transforms.Compose([
    transforms.Resize((256, 256)),       # Resize images to 256x256
    transforms.CenterCrop(224),           # Crop the center 224x224 portion
    transforms.ToTensor(),                # Convert PIL Image to tensor (0-1)
    transforms.Normalize(mean, std)       # Normalize with ImageNet mean and std
])
```

### Finding more data

In addition to the training set provided by MONAI, we should consider how our model performs on different training/validation sets and try switching around training/validation sets.

## Hyperparameters

To start, our parameters look like the following:

```python
# Setup the training parameters
max_epochs = 30
val_interval = 1
VAL_AMP = True
device = torch.device("cuda:0") if torch.cuda.device_count() else "cpu"

# Create UNet
# TODO: Adjust the parameters of the UNet
model = UNet(
    spatial_dims=3,   # 3D images
    in_channels=4,   # Input chanels based on our images
    out_channels=3,   # We predict 3 output classes
    channels=(16, 32, 64),
    strides=(2, 2),
    num_res_units=2,
    act='PRELU',
    norm='INSTANCE',
    dropout=0.1,
    bias=True,
    adn_ordering='NDA'
)

# Create loss function and optimizer
loss_function = DiceLoss(smooth_nr=0, smooth_dr=1e-5, squared_pred=True, to_onehot_y=False, sigmoid=True)
optimizer = torch.optim.Adam(model.parameters(), 1e-4, weight_decay=1e-5)
lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epochs)

dice_metric = DiceMetric(include_background=True, reduction="mean")
dice_metric_batch = DiceMetric(include_background=True, reduction="mean_batch")

post_trans = Compose([Activations(sigmoid=True), AsDiscrete(threshold=0.5)])
```

Of these, we need to modify a few and compare results.

### Epochs
We can change ```max_epochs``` to different numbers and see which yields the best results. We have to keep in mind the risk of overfitting. As we are using data augmentation, the risk of this can be reduced. It is necessary to find the equilibrium between data augmentation and max_epochs where our model is not underfitted due to too heavy transformations applied to the images, but is also not overfitted due to too many epochs.

### Activation function
By default, we're using Parametric ReLU (PReLU) function. The difference to ReLU is that the slope for negative inputs is not fixed but is learned during the training process.

This function is more likely than not already a better option than the alternatives, especially the classic sigmoid/tanh functions, however it could still be worth a try to see if another function yields better results. Some options would be
1. LeakyReLU (Try different values for the constant)
2. Maybe a simple ReLU would work as well

### Loss function
Currently, our model is using DiceLoss as the loss function. We should definitely try using other classification-based functions, and no less than the following:
1. Weighted Cross-Entropy Loss
2. Focal Loss
3. Balanced Cross-Entropy Loss
4. Jaccard/IoU Loss
5. Boundary loss
6. Lovász-Softmax loss

### Optimizer
Here we need to test out 2 kinds of modifications:
1. Find the optimal parameters using Adam as the optimizer
2. Trying out a different optimizer altogether

**Let's explore our options for point 1:**

[Helpful resource](https://www.kdnuggets.com/2022/12/tuning-adam-optimizer-parameters-pytorch.html)

Our code currently looks like this:
```python
optimizer = torch.optim.Adam(model.parameters(), 1e-4, weight_decay=1e-5)
```

We are interested in the 2 last arguments.

The learning rate (1e-4) should be experimented with and set to the optimal value.

The learning rate decay (```weight_decay```) is the value by which the learning rate decreases as the parameter values converge towards the global minimum in order to avoid overshooting it. The default value is 0, however, different values should be tried out.

**Now for point 2:**

Alternatives to Adam that could be worth looking into are:
1. RMSprop
2. AdaGrad
3. Adamax

Although Adam is a combination of RMSprop and AdaGrad, it could be worth looking into how each perform on its own. Here is an example using RMSprop:

```python
from torch.optim import RMSprop

optimizer = RMSprop(model.parameters(), lr=1e-4)
```

### Learning rate scheduler

```python
lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epochs)
```

Options besides cosine annealing should also be explored. Options here include
- StepLR
- ReduceLROnPlateau

Below is an example of what using the StepLR would look like:

```python
from torch.optim.lr_scheduler import StepLR

lr_scheduler = StepLR(optimizer, step_size=10, gamma=0.1)
```

More information can be found [here](https://medium.com/@theom/a-very-short-visual-introduction-to-learning-rate-schedulers-with-code-189eddffdb00).

## Regularization techniques

**Dropout layer**

In order to filter out statistical noise in our training data and thereby avoid overfitting, we use a dropout value. A dropout layer essentially deactivates certain neurons at random. The dropout value defines the probability by which a dropout layer is added in a cycle.

Our model has a dropout set in the model definition
```python
model = UNet(
    ...
    dropout=0.1  # Add dropout layer with 10% probability
)
```
Experimenting with different values could change the result and help the model generalize better.

**Batch normalization**

Additionally, we can add a batch normalization to stabilize the training process:

```python
model = UNet(
    ...
    norm='BATCH'  # Example: Use batch normalization
)
```

### Architecture adjustments

This part needs further investigation, however, we should consider increasing/decreasing the number of channels and layers in order to find out the necessary complexity in order to catch the tumor in its entirety. Below is an example:

```python
model = UNet(
    ...
    channels=(32, 64, 128, 256),  # Example: Increase number of channels
    strides=(2, 2, 2)              # Example: Increase number of layers
)

```

## Tracking Progress with Weights & Biases
It is essential that we use Weights & Biases to track our project. Once the server is up and running, it should be incorporated immediately.

```python
import wandb

# Initialize W&B project
wandb.init(project='your_project_name', config={'max_epochs': max_epochs, 'lr': 1e-4, 'optimizer': 'Adam'})

# Log training metrics during training loop
for epoch in range(max_epochs):
    # Training loop
    ...
    # Log metrics
    wandb.log({'loss': loss.item(), 'dice_coefficient': dice_coefficient})
```
