
# Third-party imports
import joblib
import numpy as np
from catboost import CatBoostClassifier
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


model_cat = CatBoostClassifier(
    loss_function='Logloss',     # still optimizing binary cross-entropy during training
    eval_metric='F1',            # monitor F1 score on validation for early stopping and best iteration
    early_stopping_rounds=30,
    iterations=200,
    random_seed=12345,
    verbose=1,
    task_type='GPU'
)


param_grid_cat = {
    'learning_rate': [0.05, 0.3],        # low and high learning rates
    'depth': [3, 8],                     # shallow vs. relatively deep trees
    'min_data_in_leaf': [5, 40],         # small leaf size vs. large leaf size
    'l2_leaf_reg': [1, 10],              # low and strong regularization
}

if in_colab():
    from google.colab import files
    print("Upload embedded_features.npz")
    uploaded = files.upload()

grid_search_cat = GridSearchCV(
    estimator=model_cat,
    param_grid=param_grid_cat,
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

if in_colab():
    joblib_save(grid_search_cat,
            'grid_search_cat.pkl',
            param_grid_cat,
            'param_grid_cat.pkl')
    best_model = grid_search_cat.best_estimator_
    best_model.save_model('model_cat.json', format='json')
    files.download('model_cat.json')
    files.download('grid_search_cat.pkl')
    files.download('param_grid_cat.pkl')
else:
    joblib_save(grid_search_cat,
            'cv_tuning_results/grid_search_cat.pkl',
            param_grid_cat,
            'param_grids/param_grid_cat.pkl')
    best_model = grid_search_cat.best_estimator_
    best_model.save_model('models/model_cat.json', format='json')
