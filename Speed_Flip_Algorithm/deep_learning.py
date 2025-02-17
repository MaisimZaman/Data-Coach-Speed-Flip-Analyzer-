import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from ML_model import X_train, X_test, y_train, y_test
from sklearn.metrics import classification_report
from sklearn.metrics import roc_curve, auc
import numpy as np
from sklearn.metrics import precision_recall_curve, auc
import matplotlib.pyplot as plt

def train_and_evaluate_model(X_train, y_train, X_test=None, y_test=None):
    # Define the input dimension based on the training data
    input_dim = X_train.shape[1]

    # Initialize the model
    model = Sequential([
        Dense(128, activation='relu', input_shape=(input_dim,)),
        Dropout(0.2),  # Dropout for regularization
        Dense(64, activation='relu'),
        Dropout(0.2),
        Dense(1, activation='sigmoid')  # Sigmoid for binary classification
    ])

    # Compile the model
    model.compile(optimizer='adam',  # Adam is generally a good starting point
                  loss='binary_crossentropy',  # Suitable for binary classification
                  metrics=['accuracy', tf.keras.metrics.AUC(name='auc')])

    # Train the model
    model.fit(X_train, y_train, epochs=50, validation_split=0.2, batch_size=32,
              class_weight={0:1, 1:3})  # Adjust class weights if necessary

    # If test data is provided, evaluate the model on it
    if X_test is not None and y_test is not None:
        test_results = model.evaluate(X_test, y_test, verbose=1)
        print(f"Test Loss: {test_results[0]}, Test Accuracy: {test_results[1]}, AUC: {test_results[2]}")
    
    return model

model = train_and_evaluate_model(X_train, y_train, X_test, y_test)
# Predict the probabilities for the test set
y_pred_probs = model.predict(X_test)
# Convert probabilities to binary labels based on a threshold
y_pred = (y_pred_probs > 0.5).astype('int32')

# Calculate the ROC curve to find the optimal threshold
fpr, tpr, thresholds = roc_curve(y_test, y_pred_probs)
optimal_idx = np.argmax(tpr - fpr)
optimal_threshold = thresholds[optimal_idx]

print("Optimal threshold:", optimal_threshold)

# Apply the optimal threshold
y_pred_optimal = (y_pred_probs > optimal_threshold).astype(int)
print(classification_report(y_test, y_pred_optimal))

precision, recall, thresholds = precision_recall_curve(y_test, y_pred_probs)

# Calculate area under the precision-recall curve
pr_auc = auc(recall, precision)

# Plotting the precision-recall curve
plt.figure()
plt.plot(recall, precision, label=f'PR Curve (area = {pr_auc:.2f})')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall curve')
plt.legend(loc="best")
plt.show()

# Optionally, finding the optimal threshold by F1 score
f1_scores = 2 * recall * precision / (recall + precision)
optimal_idx = np.argmax(f1_scores)
optimal_threshold = thresholds[optimal_idx]

print("Optimal threshold by F1:", optimal_threshold)




