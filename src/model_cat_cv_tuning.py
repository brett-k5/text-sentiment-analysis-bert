# Standard library imports
import joblib

# Third-party imports
import numpy as np
from catboost import CatBoostClassifier
from sklearn.model_selection import GridSearchCV

# Local application imports
from src.data_pre_processing import train_target


model_cat = CatBoostClassifier(
    loss_function='Logloss',     # still optimizing binary cross-entropy during training
    eval_metric='F1',            # monitor F1 score on validation for early stopping and best iteration
    early_stopping_rounds=30,
    iterations=200,
    random_seed=12345,
    verbose=1,
    task_type='GPU'
)

param_grid = {
    'learning_rate': [0.05, 0.3],        # low and high learning rates
    'depth': [3, 8],                     # shallow vs. relatively deep trees
    'min_data_in_leaf': [5, 40],         # small leaf size vs. large leaf size
    'l2_leaf_reg': [1, 10],              # low and strong regularization
}

grid_search_cat = GridSearchCV(
    estimator=model_cat,
    param_grid=param_grid,
    scoring='f1',
    cv=3,
    verbose=1,
    refit=True
)

with np.load('embedded_features.npz') as data:
    train_features = data['X_train']
    
grid_search_cat.fit(train_features, train_target)

print("Best hyperparameters:", grid_search_cat.best_params_)
print(f"Best F1 score: {grid_search_cat.best_score_:.4f}")

joblib.dump(grid_search_cat.best_estimator_, 'models/model_cat.pkl')