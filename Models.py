import pickle
import tensorflow as tf
import pandas as pd
import numpy as np
from utils import flat_X

class Models():
    def __init__(self) -> None:
        self.max_seq_length = 30 # Frames per video
        self.num_features = 84 # 21 rows x, 21 rows y left and right = 84
    
    def get_three_dimensions(self, df: pd.DataFrame):
        num_samples = int(len(df)/self.max_seq_length)
        return df.values.reshape(num_samples, self.max_seq_length, self.num_features)

    def get_flat_X(self, df: pd.DataFrame):
        return flat_X(self.max_seq_length, df)

    def save_model(self, model_name, trained_model):
        with open(f'{model_name}.pkl','wb') as f:
            pickle.dump(trained_model, f)

class PretrainedModels():
    def __init__(self) -> None:
        self.max_seq_length = 30 # Frames per video
        self.num_samples = 1 # One video processed
        self.num_features = 84 # 21 rows x, 21 rows y left and right = 84

        self.unique_pred = []
        self.load_label_encoder()
        self.load_three_models()
    
    def load_label_encoder(self):
        # Load the label encoder
        self.label_encoder = self.load_model('label_encoder')

    def load_three_models(self):
        self.svm_model = self.load_model("svm")
        self.tree_model = self.load_model("tree")
        self.cnn_model = tf.keras.models.load_model("cnn")

    def predict_with_cnn(self, data: pd.DataFrame):
        X_val_cnn = data.values.reshape(self.num_samples, self.max_seq_length, self.num_features)
        predicted_cnn = self.cnn_model.predict(X_val_cnn)
        most_likely_predictions = np.argmax(predicted_cnn, axis=1)
        predicted_cnn = self.label_encoder.inverse_transform(most_likely_predictions)[0]

        return predicted_cnn
    
    def predict_with_tree(self, data: pd.DataFrame):
        X_val = flat_X(self.max_seq_length, data)
        predicted_tree = self.tree_model.predict(X_val)[0]

        return predicted_tree

    def predict_with_svm(self, data: pd.DataFrame):
        X_val = flat_X(self.max_seq_length, data)
        predicted_svm = self.svm_model.predict(X_val)[0]

        return predicted_svm

    def load_model(self, model_name):
        # load
        with open(f'{model_name}.pkl', 'rb') as f:
            return pickle.load(f) 
    
    def get_predictions(self, data: pd.DataFrame):
        self.unique_pred = []
        results = []
        chunks = [data[i:i + self.max_seq_length] for i in range(0, len(data), self.max_seq_length)]

        for chunk in chunks:
            results.append(
                {
                    "SVM": self.predict_with_svm(chunk),
                    "CNN": self.predict_with_cnn(chunk),
                    "TREE": self.predict_with_tree(chunk),
                }
            )
            self.unique_pred.append(list(set(results[-1].values())))

        return results
    
    def get_unique_pred(self):
        return self.unique_pred
