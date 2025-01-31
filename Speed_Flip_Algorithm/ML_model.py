#ML Model using Random Forest Classifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import pandas as pd
import joblib

file_path = "training_data.csv"


training_df = pd.read_csv(file_path)

# Define features and target variable
features = [
    "CarPositionX", "CarPositionY", "CarPositionZ",
    "CarRotationX", "CarRotationY", "CarRotationZ", "CarRotationW",
    "CarLinearVelocityX", "CarLinearVelocityY", "CarLinearVelocityZ",
    "CarAngularVelocityX", "CarAngularVelocityY", "CarAngularVelocityZ",
    "CarSpeed"
]

X = training_df[features]  # Feature set
y = training_df["SpeedFlip"]  # Target variable

#training the model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42, stratify=y)

rf_model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, class_weight="balanced")
rf_model.fit(X_train, y_train)



joblib.dump(rf_model, "speed_flip_model.pkl")

y_pred = rf_model.predict(X_test)

rf_accuracy = accuracy_score(y_test, y_pred)
rf_classification_report = classification_report(y_test, y_pred)

print(rf_accuracy)