# Guide to setting up SSH to run JupyterLab remotely

In this guide we'll discuss how to get set up to run JupyterLab remotely via SSH in case you prefer working that way. Note: This is not strictly necessary in order to get started, but JupyterLab provides a convinient environment for you to execute the code bit by bit in. If you just want to get up and running, however, you should check out [this](./train_job.md) document.

## Setting up the remote environment

Refer to the step from [train_job.md](./train_job.md)

## Running JupyterLab
Great! Now you have your environment set up. The next step is running JupyterLab from the remote environment.

1. Make sure you are either in the shared folder or your own local copy of this repository. Then start JupyterLab by running
    ```bash
    make jupyterlab
    ```
    This will start JupyterLab on port 8888 by default. If you have problems, you can try specifying the port on which Jupyter should run
    ```bash
    jupyter notebook --no-browser --port=8888
    ```
    That being said, ```make jupyterlab``` is the preferred method.
2. Open a **new** terminal window and run
    ```bash
    ssh -NL localhost:1234:localhost:8888 [USERNAME]@[SERVER]
    ```
    where the username and server are the same as you used for logging in to SSH above. This will make your **local** machine, on its own port 1234, listen to the **remote** machine's port 8888. You can change 1234 to whatever suits you the best.
3. Now you will be able to check if it is up and running by going to your browser and going to http://localhost:1234/lab