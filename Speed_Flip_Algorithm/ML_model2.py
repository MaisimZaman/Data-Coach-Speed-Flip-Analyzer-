# ML Model for Speed Flip Detection - Original Version Restored

from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
import lightgbm as lgb
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, precision_recall_curve, f1_score
import pandas as pd
import shap
import joblib
from pathlib import Path

# ✅ Define data directory
dir_path = Path('Training_data')


#ML model with no SMOTE in testing datasets

# ✅ Function to load and process a single file
def load_training_data(file_path):
    df = pd.read_csv(file_path)

    # Ensure dataset has both classes
    if df['SpeedFlip'].nunique() < 2:
        return None  # Skip if only one class is present

    return df

# ✅ Load all datasets and create training/testing splits


def build_training_testing_dfs(dir_path, test_size=0.2):
    all_dfs = []
    for file_path in dir_path.rglob('*'):
        if file_path.is_file():
            df = load_training_data(file_path)
            if df is not None:
                all_dfs.append(df)

    full_df = pd.concat(all_dfs, ignore_index=True)
    
    # ✅ Keep a copy of ALL original features before modifications
    original_features = [
        "SecondsRemaining", "CarSteer", "CarThrottle", "CarRotationX", "CarRotationY", "CarRotationZ",
        "CarRotationW", "CarLinearVelocityX", "CarLinearVelocityY", "CarLinearVelocityZ",
        "CarAngularVelocityX", "CarAngularVelocityY", "CarAngularVelocityZ",
        "CarSpeed", "CarDodgeActive", "CarJumpActive"
    ]

    # ✅ Train-test split BEFORE any balancing (test remains unbalanced)
    X = full_df[original_features]  # Keep original features!
    y = full_df['SpeedFlip']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y
    )

    return X_train, X_test, y_train, y_test



# ✅ Build final training and testing datasets
X_train, X_test, y_train, y_test = build_training_testing_dfs(dir_path)

# ✅ Train XGBoost Model
def build_xgb_model(X_train, y_train):
    xgb_model = xgb.XGBClassifier(
        n_estimators=600,
        max_depth=8,
        learning_rate=0.08,
        colsample_bytree=1.0,
        scale_pos_weight=8,  # 🔥 Forces XGBoost to focus more on Class 1
        random_state=42,
        subsample=0.8
    )

    xgb_model.fit(X_train, y_train)
    return xgb_model

# ✅ Train LightGBM Model
def build_lgb_model(X_train, y_train):
    lgb_model = lgb.LGBMClassifier(
        n_estimators=500,
        max_depth=8,
        learning_rate=0.05,
        class_weight='balanced',  # 🔥 Auto-balances the classes dynamically
        random_state=42
    )

    lgb_model.fit(X_train, y_train)
    return lgb_model

# ✅ Train models
xgb_model = build_xgb_model(X_train, y_train)
lgb_model = build_lgb_model(X_train, y_train)

# ✅ Save trained models
joblib.dump(xgb_model, "ML_Models/xgb_speed_flip_model.pkl")
joblib.dump(lgb_model, "ML_Models/lgb_speed_flip_model.pkl")

# ✅ Find Best Decision Threshold for XGBoost
y_probs_xgb = xgb_model.predict_proba(X_test)[:, 1]
precision, recall, thresholds = precision_recall_curve(y_test, y_probs_xgb)
f1_scores = 2 * (precision * recall) / (precision + recall + 1e-6)
best_threshold_xgb = thresholds[np.argmax(f1_scores)]

print(f"🚀 Best Decision Threshold for XGBoost: {best_threshold_xgb:.4f}")

y_pred_xgb = (y_probs_xgb >= best_threshold_xgb).astype(int)

# ✅ Find Best Decision Threshold for LightGBM
y_probs_lgb = lgb_model.predict_proba(X_test)[:, 1]
precision, recall, thresholds = precision_recall_curve(y_test, y_probs_lgb)
f1_scores = 2 * (precision * recall) / (precision + recall + 1e-6)
best_threshold_lgb = thresholds[np.argmax(f1_scores)]

print(f"🚀 Best Decision Threshold for LightGBM: {best_threshold_lgb:.4f}")

y_pred_lgb = (y_probs_lgb >= best_threshold_lgb).astype(int)

# ✅ Classification Reports
report_xgb = classification_report(y_test, y_pred_xgb, target_names=['No speed flip (0)', 'Speed flip (1)'])
report_lgb = classification_report(y_test, y_pred_lgb, target_names=['No speed flip (0)', 'Speed flip (1)'])

print("🚀 Classification Report for XGBoost Model:")
print(report_xgb)
print("🚀 Classification Report for LightGBM Model:")
print(report_lgb)




'''
explainer = shap.Explainer(xgb_model)
shap_values = explainer(X_test)

shap.summary_plot(shap_values, X_test) 
'''