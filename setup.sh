#!/bin/bash

# Update the package manager
sudo apt-get update && sudo apt-get upgrade -y

# Install Python, pip, Git, necessary libraries for OpenCV, and FFmpeg (which includes ffprobe)
sudo apt-get install -y python3 python3-pip git libgl1-mesa-glx ffmpeg unzip

# Upgrade pip
sudo pip3 install --upgrade pip

# Install the required Python libraries
pip3 install pandas numpy matplotlib scikit-learn mediapipe tensorflow keras fastapi uvicorn httpx requests opencv-python
