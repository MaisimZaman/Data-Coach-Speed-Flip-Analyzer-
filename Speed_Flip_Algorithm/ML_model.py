#ML Model using Random Forest Classifier
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import pandas as pd
import joblib

file_path = "training_data.csv"
file_path2 = "training_data_2.csv"


training_df1 = pd.read_csv(file_path)
training_df2 = pd.read_csv(file_path2)

training_df = pd.concat([training_df1, training_df2], ignore_index=True)

# Define features and target variable
features = [
    "CarPositionX", "CarPositionY", "CarPositionZ",
    "CarRotationX", "CarRotationY", "CarRotationZ", "CarRotationW",
    "CarLinearVelocityX", "CarLinearVelocityY", "CarLinearVelocityZ",
    "CarAngularVelocityX", "CarAngularVelocityY", "CarAngularVelocityZ",
    "CarSpeed", "CarDodgeActive"
]

X = training_df[features]  # Feature set
y = training_df["SpeedFlip"]  # Target variable

#training the model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

def build_random_forest_model(X_train, y_train):
    rf_model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, class_weight="balanced")
    rf_model.fit(X_train, y_train)
    
    return rf_model

def build_xgb_model(X_train, y_train):
    xgb_model = xgb.XGBClassifier(n_estimators=200, max_depth=6, learning_rate=0.1, random_state=42)

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