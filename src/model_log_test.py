# Standard library imports
import joblib

# Third-party imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import metrics

# Local application imports
from src.model_utils import evaluate_model
from src.data_pre_processing import train_target, test_target

model_log = joblib.load('models/model_log.pkl')

with np.load('embedded_features.npz') as data:
    train_features = data['X_train']
    test_features = data['X_test']

evaluate_model(model_log, train_features, train_target, test_features, test_target)