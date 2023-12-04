from flask import Flask, render_template, request, redirect, url_for
import os
import pandas as pd
import os

import urllib.request
from Mediapipe import MediapipeHands
from Models import PretrainedModels, PretrainedModelLetters

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg', 'gif'}

def process_image(image_path):
    mediapipeHands = MediapipeHands()
    mediapipeHands.extract_image_coordinates_from_path(image_path)

    df = mediapipeHands.get_padded_data()
    if (df.empty):
        return [" "]

    del df["sequence_id"] 
    del df["target"] 
    del df["file"] 

    pretainedModel = PretrainedModelLetters()
    return pretainedModel.get_predictions(df)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    if 'file' not in request.files:
        return redirect(request.url)

    files = request.files.getlist('file')

    uploaded_files = []
    file_names = []  # Added to store file names

    for file in files:
        if file and allowed_file(file.filename):
            filename = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(filename)
            uploaded_files.append(filename)
            result = process_image(filename)

            file_names.append(result[0])

    return render_template('index.html', uploaded_files=uploaded_files, file_names=file_names, resultado="".join(file_names))

if __name__ == '__main__':
    app.run(debug=True, port=4444)
