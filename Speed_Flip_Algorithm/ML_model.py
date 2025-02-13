#ML Model using Random Forest Classifier
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import pandas as pd
import joblib
from pathlib import Path

dir_path = Path('Training_data')

def build_traing_df(dir_path):
    training_dfs = []
    
    for file_path in dir_path.rglob('*'):  # '*' matches all files and directories
        if file_path.is_file():  # This check ensures only files are considered
            training_df = pd.read_csv(file_path)
            training_dfs.append(training_df)
    
    final_df = pd.concat(training_dfs, ignore_index=True)
    
    return final_df            

training_df = build_traing_df(dir_path)

# Define features and target variable
features = [
    "CarSteer", "CarPositionZ", "CarRotationX", "CarRotationY","CarRotationZ",
    "CarRotationW", "CarLinearVelocityX", "CarLinearVelocityY", "CarLinearVelocityZ",
    "CarAngularVelocityX", "CarAngularVelocityY", "CarAngularVelocityZ",
    "CarSpeed", "CarBoostAmount", "CarDodgeActive", "CarJumpActive"
]

X = training_df[features]  # Feature set
y = training_df["SpeedFlip"]  # Target variable

#training the model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

def build_random_forest_model(X_train, y_train):
    rf_model = RandomForestClassifier(
        n_estimators=300,         # More trees (higher accuracy)
        max_depth=20,             # Deeper trees (captures more patterns)
        min_samples_split=2,      # Allow frequent splitting
        min_samples_leaf=1,       # Smaller leaves help detect rare cases
        max_features="sqrt",      # Improves generalization
        class_weight="balanced",  # Handles imbalanced classes
        random_state=42
    )
    
    rf_model.fit(X_train, y_train)
    return rf_model

def build_xgb_model(X_train, y_train):
    xgb_model = xgb.XGBClassifier(
        n_estimators=500,        # More boosting rounds
        max_depth=8,             # More complexity allowed
        learning_rate=0.05,      # Lower learning rate for better generalization
        colsample_bytree=0.8,    # Reduces overfitting
        scale_pos_weight=1.2,    # Helps with class imbalance
        random_state=42
    )
    
    xgb_model.fit(X_train, y_train)
    return xgb_model



rf_model = build_random_forest_model(X_train, y_train)

xgb_model = build_xgb_model(X_train, y_train)


joblib.dump(xgb_model, "ML_Models/xgb_speed_flip_model.pkl")

joblib.dump(rf_model, "ML_Models/rf_speed_flip_model.pkl")

y_pred = xgb_model.predict(X_test)
y_pred2 = rf_model.predict(X_test)


xgb_classification_report = classification_report(y_test, y_pred)

xgb_accuracy = accuracy_score(y_test, y_pred)
rf_accuracy = accuracy_score(y_test, y_pred2)


print("xgb accuracy")
print(xgb_accuracy)

print("rf accuracy")
print(rf_accuracy)