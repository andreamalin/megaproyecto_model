from flask import Flask, render_template, request, send_from_directory
import pandas as pd
import os

import urllib.request
from Mediapipe import MediapipeHands
from Models import PretrainedModels, PretrainedModelLetters

import time

def process_video(id, path, is_asl):
    mediapipeHands = MediapipeHands()
    mediapipeHands.extract_coordinates_from_path(path, id)

    df = mediapipeHands.get_padded_data()
    del df["sequence_id"] 
    del df["target"] 
    del df["file"] 

    pretainedModels = PretrainedModels(is_asl=is_asl)
    results = pretainedModels.get_predictions(df)

    return results[0]


app = Flask(__name__)

# Set the upload folder
UPLOAD_FOLDER = 'cdn_input'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST' and 'file' in request.files:
        file = request.files['file']
        selected_option = request.form.get('option')
        print(selected_option)

        if file and file.filename != '':
            filename = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(filename)
            
            result = process_video(1, filename, selected_option == "asl")

            return render_template('index.html', original_filename=file.filename, edited_filename=file.filename, edited_words=result)

    return render_template('index.html', filename=None)


@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)


@app.route('/edited/<filename>')
def edited_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], f'mediapipe-video-final-1.mp4')

if __name__ == '__main__':
    app.run(debug=True, port=4444, host="0.0.0.0")
