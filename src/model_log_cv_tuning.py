# Standard library imports
import joblib

# Third-party imports
import numpy as np
from sklearn.linear_model import LogisticRegression
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


model_log = LogisticRegression(random_state=12345, max_iter=2000, penalty='l2', solver='liblinear')

param_grid = {
    'C': [0.01, 1], # Divides regularization coefficient (larger C = less reg, smaller = more reg)
}

grid_search_log = GridSearchCV(
    estimator=model_log,
    param_grid=param_grid,
    scoring='f1',     # use 'f1_macro' or 'f1_weighted' for multiclass
    cv=5,
    n_jobs=-1,
    verbose=1,
    refit=True
)

if in_colab():
    from google.colab import files
    print("Upload embedded_features.npz")
    uploaded = files.upload()
with np.load('embedded_features.npz') as data:
    train_features = data['X_train']

grid_search_log.fit(train_features, train_target)

print("Best hyperparameters:", grid_search_log.best_params_)
print(f"Best F1 score: {grid_search_log.best_score_:.4f}")


if in_colab():
    joblib.dump(grid_search_log.best_estimator_, 'model_log.pkl')
    files.download('model_log.pkl')
else:
    joblib.dump(grid_search_log.best_estimator_, 'models/model_log.pkl')