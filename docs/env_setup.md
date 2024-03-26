We use conda for managing the Python environment. If you don't have conda installed, you can download it from [here](https://docs.conda.io/en/latest/miniconda.html). 
Once you have conda installed, you need to install mamba, which is a faster alternative to conda. You can install mamba by running the following command:

```bash
conda install mamba -n base -c conda-forge
```

Now we can create a new conda environment from the `environment.yml` file. You can do this by running the following command:

```bash
conda env create -f environment.yml
```
or with mamba:
```bash
mamba env create -f environment.yml
```

This will create a new conda environment called `brain-segm`. To activate the environment, you can run the following command:

```bash
conda activate brain-segm
```

To install the package in the environment, we preferably use mamaba. You can install the package by running the following command:

```bash
mamba install <the rest depending on the package>
```

If there are any updates to the environment, you can update the environment by running the following command:

```bash
mamba env update -f environment.yml --prune
```

And to push new chages to the environment.yml file, you can run the following command:

```bash
mamba env export > environment.yml
```

To deactivate the environment, you can run the following command:

```bash
conda deactivate
```

If you have any trouble with the installation, please refer to the [conda documentation](https://conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html#activating-an-environment).

