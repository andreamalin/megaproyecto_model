import pandas as pd
import os

import urllib.request
from Mediapipe import MediapipeHands
from Models import PretrainedModels, PretrainedModelLetters

letters_results = []

def download(url_link, path='', is_image=False):
    if (is_image):
        urllib.request.urlretrieve(url_link, path) 
    else:
        urllib.request.urlretrieve(url_link, path) 

def process_video(url_link, id):
    path = f'{os.getcwd()}/cdn_input/{id}.mp4'
    download(url_link, path)

    mediapipeHands = MediapipeHands()
    mediapipeHands.extract_coordinates_from_path(path, id)

    print('>>> WUU')
    df = mediapipeHands.get_padded_data()
    del df["sequence_id"] 
    del df["target"] 
    del df["file"] 

    pretainedModels = PretrainedModels()
    results = pretainedModels.get_predictions(df)

    print("----Resultados----")
    print(results)
    results = pretainedModels.get_unique_pred()
    print(results)
    return results


def process_image(url_link, groupId, id):
    image_path = f'{os.getcwd()}/cdn_input/{groupId}/{id}.png'
    download(url_link, image_path)  

    mediapipeHands = MediapipeHands()
    mediapipeHands.extract_image_coordinates_from_path(image_path)

    df = mediapipeHands.get_padded_data()
    del df["sequence_id"] 
    del df["target"] 
    del df["file"] 

    pretainedModel = PretrainedModelLetters()
    letters_results.append(pretainedModel.get_predictions(df))
