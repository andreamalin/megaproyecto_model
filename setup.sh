#!/bin/bash

# Update the package manager
sudo apt-get update && sudo apt-get upgrade -y

# Install Python, pip, Git, necessary libraries for OpenCV, and FFmpeg (which includes ffprobe)
sudo apt-get install -y python3 python3-pip git libgl1-mesa-glx ffmpeg unzip python3-opencv

# Upgrade pip
sudo pip3 install --upgrade pip

# Install the required Python libraries
pip3 install scikit-learn==1.0.2
pip3 install pandas numpy matplotlib mediapipe tensorflow keras fastapi uvicorn httpx requests opencv-python google-cloud-storage
