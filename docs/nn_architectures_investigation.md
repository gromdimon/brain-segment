# Brain Segmentation Project - Architecture Research Plan

The goal of this research plan is to systematically assess the viability of three different neural network architectures for 3D brain segmentation: **U-Net**, **ResNet**, and **UNETR**. Each team member will be responsible for gathering information on the following aspects of each architecture.

## General Workflow

This section outlines a comprehensive workflow for this task, guiding you through research, implementation, documentation, and evaluation phases. The workflow leverages the MONAI framework, which is specifically designed for healthcare imaging. Ensure to read this doc first and choose the best options for you.

### Step 1: Initial Research and Setup

- **Architecture Overview**: Conduct initial research on the architecture, focusing on their design, applications, and performance in medical imaging tasks.
- **Environment Setup**: Ensure all team members have a consistent `conda` environment. 
- **MONAI Installation**: Install MONAI and any other necessary libraries (PyTorch, NumPy). Hint: You should install everything with mamba and befor pushing the changes to GitHub, DO NOT FORGET to update the `environment.yml` file. Doc `env_setup.md` can be used as a reference.

### Step 2: Dive into MONAI Tutorials

- **Explore MONAI Tutorials**: You should start by exploring [MONAI tutorials](https://docs.monai.io/en/latest/tutorials.html), focusing on those relevant to 3D image segmentation and the specific architectures in question.
- **Hands-on Practice**: Replicate a tutorial closest to our project’s needs, possibly adapting it to work with a sample 3D brain segmentation dataset. Try to run the tutorial on your local machine or a cloud environment (Jupyter Hub).

### Step 3: Model Research and Implementation

- **Model-Specific Research**: Delve deeper into the architecture, focusing on aspects listed in the research plan (see below).
- **Implementation**: Start with a basic implementation of the assigned model for brain segmentation (e.x. From MONAI tutorial). Utilize existing resources, code repositories, and documentation to aid in this process.

### Step 4: Documentation of Steps and Findings

- **Documentation Practice**: Document every step of the process, from environment setup to model implementation. Include code snippets, setup instructions, and explanations of decisions made. You can use the `docs` folder for this purpose (with markdown file similar to existing ones). Good documentation is key!
- **Findings and Results**: Summarize findings on model performance, challenges encountered, and solutions devised. This should include quantitative results (e.g., accuracy, training time) and qualitative insights.

### Step 5: Review and Refine

- **Oversight and Feedback**: Regularly review each other's work, providing feedback and suggestions for improvement. This will help ensure consistency and quality across all research efforts.


This workflow is designed to be iterative, allowing for continuous improvement and adaptation as the project progresses. The ultimate goal is to not only choose the best architecture for our brain segmentation project but also to gain a deep understanding of the capabilities and nuances of each option. Enjoy the journey!


## Aspects to Investigate

Here are the key aspects to collect for future comparison. Note, that the list is not exhaustive and can (should) be extended if needed!:

### 1. General Overview

- Briefly describe architecture.
- Historical context and development background.
- Common applications and notable achievements in the field of medical imaging.
- Examples of successful implementations in 3D brain segmentation tasks.

### 2. Model Complexity

- **Number of Parameters**: How large is the model in terms of the total number of trainable parameters?
- **Number of Layers**: Detail the depth of architecture, including variations if applicable (e.g., ResNet-50 vs. ResNet-101).

### 3. Performance

- **Accuracy**: How accurate is model in the context of 3D brain segmentation? Include benchmarks and case studies.
- **Training Time**: Estimate the training duration for the model on comparable hardware setups.
- **Inference Time**: Evaluate how fast model can perform inference on new data.

### 3. Hardware Requirements

- Specify the hardware needed to train model efficiently. Consider GPU memory requirements and processing power.
- Discuss any specific hardware optimizations available (e.g., TensorRT for NVIDIA GPUs).

### 4. Model Optimizations

- **Available Optimizations**: Investigate optimizations like quantization, pruning, and knowledge distillation for each architecture.
- **Frameworks and Tools**: Identify the major deep learning frameworks and tools supporting these optimizations (e.g., TensorFlow, PyTorch).

### 5. Flexibility and Scalability

- Assess how model adapts to different sizes and types of datasets.
- Discuss the scalability of the architecture for increasing image resolution and dataset sizes.

### 6. Implementation Challenges

- Identify common pitfalls and challenges in implementing the model for 3D brain segmentation.
- Suggest possible solutions or workarounds for these challenges.

### 7. Useful Resources

- Provide links to relevant papers, tutorials, and code repositories for further exploration.
- Include any additional resources that might be helpful for understanding the architecture.

