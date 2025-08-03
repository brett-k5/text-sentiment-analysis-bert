
# Third-party imports
import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV

# Local application imports
from src.data_pre_processing import train_target
from src.model_utils import joblib_save


model_log = LogisticRegression(random_state=12345, max_iter=2000, penalty='l2', solver='liblinear')

param_grid_log = {
    'C': [0.01, 1], # Divides regularization coefficient (larger C = less reg, smaller = more reg)
}

grid_search_log = GridSearchCV(
    estimator=model_log,
    param_grid=param_grid_log,
    scoring='f1',     # use 'f1_macro' or 'f1_weighted' for multiclass
    cv=5,
    n_jobs=-1,
    verbose=1,
    refit=True
)

with np.load('embedded_features.npz') as data:
    train_features = data['X_train']

grid_search_log.fit(train_features, train_target)

print("Best hyperparameters:", grid_search_log.best_params_)
print(f"Best F1 score: {grid_search_log.best_score_:.4f}")

joblib_save(grid_search_log,
            'cv_tuning_results/grid_search_log.pkl',
            param_grid_log,
            'param_grids/param_grid_log.pkl',
            grid_search_log.best_estimator_,
            'models/model_log.pkl',)

