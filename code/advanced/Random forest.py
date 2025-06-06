import pandas as pd
from sklearn import datasets
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

iris = datasets.load_iris()
inputs = iris.data
targets = iris.target

X_train, X_val, y_train, y_val = train_test_split(inputs, targets, test_size=0.25, random_state=7)

rf_model = RandomForestClassifier(n_estimators=120, max_depth=4, random_state=10)

rf_model.fit(X_train, y_train)

predicted_labels = rf_model.predict(X_val)

model_accuracy = accuracy_score(y_val, predicted_labels)
details = classification_report(y_val, predicted_labels)

print(f"Validation Accuracy: {model_accuracy:.3f}")
print("Performance Summary:\n", details)