# Brain Segmentation

![header](assets/header.png)

## Introduction

The Brain Segmentation project is focused on developing a lightweight model for the segmentation of brain images. This project involves comparing different neural network architectures, selecting the UNet model, building a training script, integrating a W&B tracker, exploring model optimizations, and conducting the final training.

## Table of Contents

- [Brain Segmentation](#brain-segmentation)
  - [Introduction](#introduction)
  - [Table of Contents](#table-of-contents)
  - [Project Overview](#project-overview)
    - [Steps Taken in the Project](#steps-taken-in-the-project)
  - [Dataset](#dataset)
  - [Resources](#resources)
  - [License](#license)
  - [Contributors](#contributors)
  - [Acknowledgements](#acknowledgements)

## Project Overview

### Steps Taken in the Project

1. **Comparing Neural Network Architectures**
   - We evaluated various neural network architectures and ultimately chose the UNet model for its superior performance in segmentation tasks.

2. **Building a Training Script**
   - We developed a robust training script to streamline the training process.

3. **Adding W&B Tracker**
   - Integrated Weights & Biases (W&B) tracker for monitoring and visualizing the training process.

4. **Exploring Model Optimizations**
   - Implemented and tested various optimizations to enhance model performance. For detailed steps, refer to [Optimizations](optimizations.md).

5. **Final Training**
   - Conducted the final training phase using the optimized model and training script.

For detailed steps, refer to the following markdown files:

- [NN Architectures Investigation](docs/nn_architectures_investigation.md)
- [Investigation Results](docs/investigation_results.md)
- [Setup Jupyter](docs/setup_jupyter.md)
- [Pipenv Environment Setup](docs/pipenv_env.md)
- [Train Job](docs/train_job.md)
- [GitHub Workflow](docs/github_workflow.md)

## Dataset

We used the brain tumor segmentation dataset from a BRCA competition. This dataset provided the necessary data for training and evaluating our model.

## Resources

The compute resources for this project were provided by the Berlin Institute of Health (BIH) during our internship.

## License

Distributed under the MIT License. See `LICENSE` for more information.

## Contributors

- Dzmitry Hramyka
- Mathias Husted
- Evin Demir
- Nils Bender

## Acknowledgements

We would like to thank the following individuals and organizations for their support and contributions:

- Berlin Institute of Health (BIH)
- Soren Lukassen (Supervisor)
