# Standard library imports
import joblib

# Third-party imports
import numpy as np
import lightgbm as lgb
from lightgbm import LGBMClassifier
from sklearn.model_selection import GridSearchCV

# Local application imports
def in_colab():
    try:
        import google.colab
        return True
    except ImportError:
        return False

if in_colab():
    from data_pre_processing import train_target
else:
    from src.data_pre_processing import train_target

# We need to make sure we have a version of LGBMClassifier that can run on GPUs
print("LightGBM version:", lgb.__version__)

try:
    model = lgb.LGBMClassifier(device='gpu')
    print("GPU support is enabled for LightGBM.")
except Exception as e:
    print("GPU support is NOT enabled:", e)

model_light = LGBMClassifier(random_state=12345,
                             num_leaves=31,
                             n_estimators=200,
                             device='gpu')

param_grid = {
    'learning_rate': [0.01, 0.1],
    'min_split_gain': [0.01, 0.1],
    'max_depth': [5, 12],
}

if in_colab():
    from google.colab import files
    print("Upload embedded_features.npz")
    uploaded = files.upload()
with np.load('embedded_features.npz') as data:
    train_features = data['X_train']

grid_search_light = GridSearchCV(
    estimator=model_light,
    param_grid=param_grid,
    scoring='f1',
    cv=3,
    verbose=1,
    refit=True
)

with np.load('embedded_features.npz') as data:
    train_features = data['X_train']

grid_search_light.fit(train_features, train_target)


print("Best hyperparameters:", grid_search_light.best_params_)
print(f"Best F1 score: {grid_search_light.best_score_:.4f}")

if in_colab():
    joblib.dump(grid_search_light.best_estimator_, 'model_light.pkl')
    files.download('model_light.pkl')
else:
    joblib.dump(grid_search_light.best_estimator_, 'models/model_light.pkl')