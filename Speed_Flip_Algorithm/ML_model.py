#ML Model using Random Forest Classifier
from sklearn.ensemble import RandomForestClassifier, IsolationForest
import xgboost as xgb
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_auc_score, precision_recall_curve, auc
import pandas as pd
import joblib
from pathlib import Path
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_score, precision_recall_curve, f1_score
import matplotlib.pyplot as plt


dir_path = Path('Training_data')

def build_training_df(dir_path):
    training_dfs = []
    
    for file_path in dir_path.rglob('*'):  # '*' matches all files and directories
        if file_path.is_file():  # This check ensures only files are considered
            training_df = pd.read_csv(file_path)
            df_class_0 = training_df[training_df['SpeedFlip'] == 0]
            df_class_1 = training_df[training_df['SpeedFlip'] == 1]
            df_class_0_under = df_class_0.sample(n=len(df_class_1), random_state=42)  # ensure reproducibility with a random state
            df_balanced = pd.concat([df_class_0_under, df_class_1], axis=0)
            training_dfs.append(df_balanced)
    
    final_df = pd.concat(training_dfs, ignore_index=True)
    
    return final_df            

training_df = build_training_df(dir_path)

# Define features and target variable

features = [
    "SecondsRemaining", "CarSteer", "CarPositionZ", "CarRotationX", "CarRotationY","CarRotationZ",
    "CarRotationW", "CarLinearVelocityX", "CarLinearVelocityY", "CarLinearVelocityZ",
    "CarAngularVelocityX", "CarAngularVelocityY", "CarAngularVelocityZ",
    "CarSpeed", "CarDodgeActive","CarBoostAmount",  "CarJumpActive"
]







X = training_df[features]  # Feature set
y = training_df["SpeedFlip"]  # Target variable



#training the model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42, stratify=y)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)  # Assuming X_train is your training data
X_test_scaled = scaler.transform(X_test)

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



#Balanced dataset calssification report
report_xgb = classification_report(y_test, xgb_pred, target_names=['No speed flip (0)', 'Speed flip (1)'])
report_rf = classification_report(y_test, rf_pred, target_names=['No speed flip (0)', 'Speed flip (1)'])


print("Classification Report for XGBoost Model:")
print(report_xgb)


#Testing on unbalanced data

# Assuming you have a trained model 'model'
predictions = xgb_model.predict(X_test_scaled)

# Print classification report
print(classification_report(y_test, predictions, target_names=['Class 0', 'Class 1']))

# Additional useful metrics
print("Confusion Matrix:\n", confusion_matrix(y_test, predictions))
print("Accuracy:", accuracy_score(y_test, predictions))

# If your model can output probabilities and you want to assess ROC AUC or Precision-Recall AUC
probabilities = xgb_model.predict_proba(X_test_scaled)[:, 1]  # Probabilities for class 1
print("ROC AUC Score:", roc_auc_score(y_test, probabilities))

precision, recall, _ = precision_recall_curve(y_test, probabilities)
print("Precision-Recall AUC:", auc(recall, precision))


