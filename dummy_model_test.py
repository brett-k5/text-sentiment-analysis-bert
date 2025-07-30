
try:
    from google.colab import files
except ImportError:
    pass  # Not running in Colab
import pandas as pd
import numpy as np
from sklearn.dummy import DummyClassifier
from model_utils import evaluate_model



if __name__ == '__main__':
    # Load training data (ensure this file is in the working directory)
    try:
        with np.load('embedded_features.npz') as data:
            train_features = data['X_train']
            test_features = data['X_test']
        train_target = pd.read_csv('train_target.csv')['pos']
        test_target = pd.read_csv('test_target.csv')['pos']
    except FileNotFoundError:
        pass # Not running from within project directory

    try:
        uploaded = files.upload()
        with np.load('embedded_features.npz') as data:
            train_features = data['X_train']
            test_features = data['X_test']
        train_target = pd.read_csv('train_target.csv')['pos']
        test_target = pd.read_csv('test_target.csv')['pos']
    except Exception as e:
        pass # Not running in google colab


    dummy = DummyClassifier(strategy='most_frequent')

    dummy.fit(train_features, train_target)

    evaluate_model(dummy, train_features, train_target, test_features, test_target)

