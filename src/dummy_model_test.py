
# Third-party imports
import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier

# Local application imports
from src.model_utils import evaluate_model
from src.data_pre_processing import train_target, test_target



if __name__ == '__main__':

    with np.load('embedded_features.npz') as data:
        train_features = data['X_train']
        test_features = data['X_test']

    dummy = DummyClassifier(strategy='most_frequent')

    dummy.fit(train_features, train_target)

    evaluate_model(dummy, train_features, train_target, test_features, test_target)

