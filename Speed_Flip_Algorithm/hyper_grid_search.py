import xgboost as xgb
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import f1_score, make_scorer, precision_recall_curve
from ML_model import X, y


def hyperparameter_grid_search(X, y):
    """
    Performs hyperparameter tuning for XGBoost using Grid Search with 80-20 training-validation split.
    
    Parameters:
    - X: Feature matrix
    - y: Target variable (labels)
    
    Returns:
    - best_model: The XGBoost model with the best F1-score
    - best_params: The optimal hyperparameters found
    """
    # ✅ Step 1: Split into 80% training and 20% test set (final test set)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # ✅ Step 2: Further split training data (80-20) for validation within Grid Search
    X_train_sub, X_val, y_train_sub, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42, stratify=y_train)

    # ✅ Step 3: Define XGBoost parameter grid
    param_grid = {
        'n_estimators': [100, 300, 500],  # Number of boosting rounds
        'max_depth': [4, 6, 8],  # Tree depth
        'learning_rate': [0.01, 0.05, 0.1],  # How much each tree contributes
        'subsample': [0.7, 0.8, 1.0],  # % of data used per tree
        'colsample_bytree': [0.7, 0.8, 1.0],  # Feature sampling per tree
        'scale_pos_weight': [1, 5, 10]  # Adjust for class imbalance
    }

    # ✅ Step 4: Define XGBoost model
    xgb_model = xgb.XGBClassifier(random_state=42, objective='binary:logistic', eval_metric='logloss')

    # ✅ Step 5: Perform Grid Search with F1-score as the optimization metric
    grid_search = GridSearchCV(
        estimator=xgb_model,
        param_grid=param_grid,
        scoring=make_scorer(f1_score),  # Optimize for F1-score
        cv=3,  # Cross-validation folds
        verbose=1,
        n_jobs=-1  # Use all CPU cores
    )

    # ✅ Step 6: Train models and find the best parameters
    grid_search.fit(X_train_sub, y_train_sub)

    # ✅ Step 7: Print best parameters & return best model
    best_params = grid_search.best_params_
    print(f"🔥 Best Hyperparameters: {best_params}")

    best_model = grid_search.best_estimator_
    return best_model, best_params, X_test, y_test

# Example usage:
#Load your dataset


# Run hyperparameter tuning
best_model, best_params, X_test, y_test = hyperparameter_grid_search(X, y)

#Make predictions with the best model
y_pred = best_model.predict(X_test)

# Evaluate final F1-score
final_f1 = f1_score(y_test, y_pred)
print(f"🚀 Final Test F1-Score: {final_f1:.4f}")


#Finding optimized decison theeshold

probabilities = best_model.predict_proba(X_test)[:, 1]

# Compute Precision-Recall curve
precision, recall, thresholds = precision_recall_curve(y_test, probabilities)

# Compute F1-score for each threshold
f1_scores = 2 * (precision * recall) / (precision + recall + 1e-6)

# Find the threshold with the best F1-score
best_f1_threshold = thresholds[np.argmax(f1_scores)]
print(f"✅ Optimal Decision Threshold for Best F1-score: {best_f1_threshold:.4f}")

# Find the threshold where precision > recall
best_precision_threshold = thresholds[np.argmax(precision > recall)]
print(f"🎯 Best Threshold for Higher Precision: {best_precision_threshold:.4f}")

# Find the threshold where recall > precision
best_recall_threshold = thresholds[np.argmax(recall > precision)]
print(f"🔍 Best Threshold for Higher Recall: {best_recall_threshold:.4f}")

#optimized f1 score is 0.5099 when 50% sampling strategy
#optimized f1 score is 0.5364 when sampling strategy is 30%
# Best Hyperparameters for 30%: {'colsample_bytree': 1.0, 'learning_rate': 0.05, 'max_depth': 8, 'n_estimators': 500, 'scale_pos_weight': 5, 'subsample': 0.8}