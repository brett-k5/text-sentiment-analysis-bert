# Third-party imports
import joblib
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
    from model_utils import joblib_save
else:
    from src.data_pre_processing import train_target
    from src.model_utils import joblib_save

# Check if GPU support is available in LightGBM
print("LightGBM version:", lgb.__version__)

try:
    _ = lgb.LGBMClassifier(device='gpu')
    print("GPU support is enabled for LightGBM.")
except Exception as e:
    print("GPU support is NOT enabled:", e)

model_light = LGBMClassifier(random_state=12345,
                             num_leaves=31,
                             n_estimators=200,
                             device='gpu')

param_grid_light = {
    'learning_rate': [0.01, 0.1],
    'min_split_gain': [0.01, 0.1],
    'max_depth': [5, 12],
}

if in_colab():
    from google.colab import files
    print("Upload embedded_features.npz")
    uploaded = files.upload()

grid_search_light = GridSearchCV(
    estimator=model_light,
    param_grid=param_grid_light,
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
    joblib_save(grid_search_light,
            'grid_search_light.pkl',
            param_grid_light,
            'param_grid_light.pkl',
            grid_search_light.best_estimator_,
            'model_light.pkl')
    files.download('model_light.pkl')
    files.download('grid_search_light.pkl')
    files.download('param_grid_light.pkl')
else:
    joblib_save(grid_search_light,
            'cv_tuning_results/grid_search_light.pkl',
            param_grid_light,
            'param_grids/param_grid_light.pkl',
            grid_search_light.best_estimator_,
            'models/model_light.pkl')