import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns


#  Load features and labels

X = np.load(r"/Radar-based-People-Counting-Using-ML-Capstone-Project-/feature extraction\X_features.npy")
y = np.load(r"/preprocessing/y.npy")

print("Feature matrix shape:", X.shape)
print("Labels shape:", y.shape)


# Split dataset

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print("Train shape:", X_train.shape, y_train.shape)
print("Test shape:", X_test.shape, y_test.shape)


# Train baseline SVM

clf = SVC(kernel='rbf', C=10, gamma='scale')
clf.fit(X_train, y_train)


#  Evaluate accuracy

# Training accuracy
y_train_pred = clf.predict(X_train)
train_acc = accuracy_score(y_train, y_train_pred)
print("Training accuracy:", train_acc)

# Testing accuracy
y_test_pred = clf.predict(X_test)
test_acc = accuracy_score(y_test, y_test_pred)
print("Testing accuracy:", test_acc)


#  evaluation

print("\nClassification Report (Test Set):")
print(classification_report(y_test, y_test_pred))

cm = confusion_matrix(y_test, y_test_pred)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix (Test Set)")
plt.show()
