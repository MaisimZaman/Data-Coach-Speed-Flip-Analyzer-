#ML Model using Random Forest Classifier
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report
import pandas as pd
import joblib
from pathlib import Path
from imblearn.over_sampling import SMOTE
from sklearn.metrics import precision_score, precision_recall_curve, f1_score
import matplotlib.pyplot as plt


dir_path = Path('Training_data')

def build_training_df(dir_path):
    training_dfs = []
    
    for file_path in dir_path.rglob('*'):  # '*' matches all files and directories
        if file_path.is_file():  # This check ensures only files are considered
            training_df = pd.read_csv(file_path)
            training_dfs.append(training_df)
    
    final_df = pd.concat(training_dfs, ignore_index=True)
    
    return final_df            

training_df = build_training_df(dir_path)

# Define features and target variable
features = [
    "CarSteer", "CarPositionZ", "CarRotationX", "CarRotationY","CarRotationZ",
    "CarRotationW", "CarLinearVelocityX", "CarLinearVelocityY", "CarLinearVelocityZ",
    "CarAngularVelocityX", "CarAngularVelocityY", "CarAngularVelocityZ",
    "CarSpeed", "CarDodgeActive","CarBoostAmount",  "CarJumpActive"
]



X = training_df[features]  # Feature set
y = training_df["SpeedFlip"]  # Target variable

#training the model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

def build_random_forest_model(X_train, y_train, class_weight=None):

    # Check if class_weight was provided, otherwise use balanced weights
    if class_weight is None:
        class_weight = {0: 1, 1: 3}  # Default weights, can be adjusted

    # Applying SMOTE to balance the dataset
    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X_train, y_train)

    # Creating the Random Forest model
    rf_model = RandomForestClassifier(
        n_estimators=500,            # More trees
        max_depth=15,                # Maximum depth of each tree
        min_samples_split=10,        # Minimum number of samples required to split an internal node
        min_samples_leaf=4,          # Minimum number of samples required to be at a leaf node
        max_features='sqrt',         # Number of features to consider when looking for the best split
        class_weight=class_weight,   # Class weights
        random_state=42              # Random state for reproducibility
    )

    # Training the model on the balanced dataset
    rf_model.fit(X_resampled, y_resampled)

    return rf_model

def build_xgb_model(X_train, y_train):
    # Calculate scale_pos_weight
    class_counts = y_train.value_counts()
    # Increase the scale_pos_weight significantly to put a very high emphasis on Class 1
    scale_pos_weight = (class_counts[0] / class_counts[1]) * 1.2

    # Applying SMOTE to balance the dataset
    smote = SMOTE(random_state=42)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

    xgb_model = xgb.XGBClassifier(
        n_estimators=500,        # More boosting rounds
        max_depth=8,             # More complexity allowed
        learning_rate=0.05,      # Lower learning rate for better generalization
        colsample_bytree=0.8,    # Reduces overfitting
        scale_pos_weight=scale_pos_weight,  # Significantly adjusted to favor the minority class
        random_state=42
    )
    # Fit the model on the balanced training data
    xgb_model.fit(X_train_resampled, y_train_resampled)
    
    return xgb_model




rf_model = build_random_forest_model(X_train, y_train)

xgb_model = build_xgb_model(X_train, y_train)



joblib.dump(xgb_model, "ML_Models/xgb_speed_flip_model.pkl")

joblib.dump(rf_model, "ML_Models/rf_speed_flip_model.pkl")

xgb_pred = xgb_model.predict(X_test)
rf_pred = rf_model.predict(X_test)

report_xgb = classification_report(y_test, xgb_pred, target_names=['Class 0', 'Class 1'])
report_rf = classification_report(y_test, rf_pred, target_names=['Class 0', 'Class 1'])

print("Classification Report for XGBoost Model:")
print(report_xgb)
print("\nClassification Report for Random Forest Model:")
print(report_rf)
