#!/bin/bash

set -xe

# Navigate to the project directory
cd ~/cmaes_stuff/turtwig/ || { echo "Unable to navigate to turtwig"; exit 1; }

uv --version || { echo "Unable to find uv; make sure its loaded"; exit 1; }
uv venv --python 3.12 || { echo "Unable to find create venv via uv"; exit 1; }
source .venv/bin/activate || { echo "Unable to activate venv"; exit 1; }

uv pip install -r requirements.txt || { echo "Failed to install requirements"; exit 1; }
uv pip install -e ../RobotSwarmSimulator || { echo "Failed to install swarmsim"; exit 1; }
