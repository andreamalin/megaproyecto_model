#!/bin/bash

# Update the package manager
sudo apt-get update && sudo apt-get upgrade -y

sudo apt-get install -y git

# Install Python and pip
sudo apt-get install -y python3 python3-pip

# Upgrade pip
sudo pip3 install --upgrade pip

# Install the required Python libraries
pip3 install pandas numpy matplotlib scikit-learn mediapipe tensorflow keras fastapi uvicorn httpx
