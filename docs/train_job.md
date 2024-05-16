# How to run a training job on the server

## Step 1: Connect to the server

```bash
ssh <username>@s-sc-frontend1.charite.de
```

Note: Replace `<username>` with your username!

## Step 2: Go to the project directory

```bash
cd /sc-projects/sc-proj-dh-ag-eils-ml/brain_segm/brain-segment/
```

## Step 3: Start tmux session

```bash
tmux
```

## Step 4 (Optional): Ensure that the conda environment exists

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

## Step 7: Run the training script

```bash
python -m src.train
```

Note: You can also use make to run the training script:

```bash
make train
```

## Step 8: Detach from the tmux session

Press `Ctrl + B` and then `D` to detach from the tmux session.
You can reattach to the session by running `tmux attach` later on.
Please read this article: [How to use tmux](https://linuxize.com/post/getting-started-with-tmux/).

Note: It is useful to use tmux, as it allows you to run the training job in the background.
For example, you can detach from the tmux session and close the terminal window without stopping the training job.
Then you can ssh to the server again and reattach to the tmux session to check the training progress.
