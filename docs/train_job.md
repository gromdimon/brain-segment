# How to run a training job on the server

## Note
It is preferred to run the whole SSH from within VS Code in order to more easily be able to make changes to the code in real time: In order to set up the environment in VSCode, check out [this tutorial.](https://code.visualstudio.com/docs/remote/ssh)

## Step 1: Connect to the server

```bash
ssh <username>@s-sc-frontend1.charite.de
```

In case DNS is still unavailable to you and the above domain name doesn't work, please refer to Slack to find the address for either frontend1 or frontend2 and insert it as the IP address. Then enter your username and password
```bash
ssh <username>@<ip-address>
```


Note: Replace `<username>` with your username and <ip-address> with the corresponding one from Slack!

## Step 2: Go to the project directory
If you want to create your own instance in your user directory, you can clone the repo as follows:
```git clone https://github.com/gromdimon/brain-segment.git``` and ```cd``` into that.

Or use the common folder:
```bash
cd /sc-projects/sc-proj-dh-ag-eils-ml/brain_segm/brain-segment/
```

## Step 3: Start tmux session

```bash
tmux
```

## Step 4 (Optional): Ensure that the conda environment exists

This step is only necessary the first time you set up the environment, or you want to update the environment.
Important: Make sure that the conda environment exists. If not, create it by running the following commands:

```bash
conda create -n brain-segm python=3.10
conda activate brain-segm
```

Then, install the required packages:

```bash
conda install -c conda-forge mamba
mamba install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
mamba install ignite -c pytorch
mamba install -c conda-forge albumentations
mamba install gdown h5py imageio lmdb matplotlib mlflow -y
mamba install jupyter monai nibabel nilearn pandas pydicom scikit-image scikit-learn scipy seaborn tensorboard tqdm transformers pydantic -y
mamba install black flake8 mypy isort -y
mamba install wandb -y
```

## Step 5: Run the iterative job

```bash
srun --pty --mem=4G --gres=gpu -p gpu --time=10:00:00 --job-name="example_brain_segment_train" bash
```

Note: Replace `--job-name="example_brain_segment_train"` with the desired job name!

## Step 6: Activate the conda environment

```bash
conda activate brain-segm
```

## Step 7: Log in to Weights and Biases
```bash
wandb login
```
You will be prompted for your username and password along with the API key, which you can find on the brain-segm project page on https://wandb.ai .

## Step 8: Run the training script

**Important note**
If you are doing test runs (not the full training) for optimization purposes, you should limit the number of epochs and batches to process (otherwise each run could take hours). To accomplish this, you should edit `train.py` - specifically the max_epochs variable, along with a limiter on the batch_loader loop. After the start of the loop on the line
```python
for batch_data in train_loader:
```
You can add a condition which skips the entire loop in case you reach a batch number limit (e.g. 10):
```python
for batch_data in train_loader:
    if step >= 10:
        print(f"CAUTION: Skipping rest of the training set for testing purposes, REMOVE THIS WHEN DONE")
        continue
    ...
```

Then, start the training script.

```bash
python -m src.train
```

Note: You can also use make to run the training script:

```bash
make train
```

A link to Weights & Biases will appear in your terminal where your current run will be tracked.

## Step 9: Detach from the tmux session

Press `Ctrl + B` and then `D` to detach from the tmux session.
You can reattach to the session by running `tmux attach` later on.
Please read this article: [How to use tmux](https://linuxize.com/post/getting-started-with-tmux/).

Note: It is useful to use tmux, as it allows you to run the training job in the background.
For example, you can detach from the tmux session and close the terminal window without stopping the training job.
Then you can ssh to the server again and reattach to the tmux session to check the training progress.
