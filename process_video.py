import pandas as pd
import os

import urllib.request
from Mediapipe import MediapipeHands
from Models import PretrainedModels


df = pd.DataFrame()
frames = []
path = f'{os.getcwd()}/cdn_input/downloaded_video.mp4'

def download_video(url_link):
    urllib.request.urlretrieve(url_link, path) 

if __name__ == '__main__':
    url_link = "https://storage.googleapis.com/deaflens-cdn/6503af38030fa3fe788ff709.mp4"
    download_video(url_link)


    mediapipeHands = MediapipeHands()
    mediapipeHands.extract_coordinates_from_path(path)

    df = mediapipeHands.get_padded_data()
    del df["sequence_id"] 
    del df["target"] 
    del df["file"] 

    pretainedModels = PretrainedModels()
    results = pretainedModels.get_predictions(df)

    print("----Resultados----")
    print(results)
    results = list(set(results.values()))
    print(results)