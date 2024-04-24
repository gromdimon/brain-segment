# Guide to setting up SSH to run JupyterLab remotely

In this guide we'll discuss how to get set up to run JupyterLab remotely via SSH, in order to get started coding and access the computing resources.

## Setting up the remote environment

First you will need to set up your SSH access, clone this Git repository and create a Conda environment there.

1. Start the VPN service (OpenVPN or another protocol you're using)
2. Start an SSH session
    ```bash
    ssh [YOUR-USERNAME]@[IP-ADDRESS]
    ```
    As DNS is currently unavailable to us, please refer to Slack to find the address for either frontend1 or frontend2 and insert it as the IP address. Then enter your username and password
3. Once you're logged in, you'll need to change to an interactive shell in order for Conda not to time out
    ```bash
    srun --pty bash
    ```
4. In order to set up the Conda environment, ```cd``` into the shared directory (path provided in Slack), and make sure you are in ```brain-segm```, where the environment.yml file is located. Make sure you have the permissions to access the shared folder - in case you don't, please ask Ben to add you. Alternatively, you can run ```git clone https://github.com/gromdimon/brain-segment.git``` and ```cd``` into that.
5. Run
    ```bash
    conda env create -f environment.yml
    ```
    and wait for it to complete.

    This step has caused problems in the past, so in case this does not work (and only then), use the alternative yml file:

    ```bash
    conda env create -f environment_server.yml
    ```
    and wait for it to complete.

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

## Setting up the environment in your VS Code

In order to set up the environment in VSCode, you can apply the information from above to [this tutorial.](https://code.visualstudio.com/docs/remote/ssh)