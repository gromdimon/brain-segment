# Guide to setting up SSH to run JupyterLab remotely

## Setting up the remote environment
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
    and wait for it to complete

## Running JupyterLab
tbd

## Setting up the environment in your VS Code
tbd