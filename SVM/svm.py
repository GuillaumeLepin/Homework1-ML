import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# digits data
digits = load_digits()
X = digits.data
y = digits.target

# split
X_train, X_val, y_train, y_val = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# C grid
C_values = np.logspace(-4, 4, 9)

rbf_accuracies = []

print("\n--- Part 4: RBF kernel SVM experiment ---")

for C in C_values:
    rbf_model = make_pipeline(
        StandardScaler(),
        SVC(kernel="rbf", C=C, gamma="scale", random_state=42)
    )

    rbf_model.fit(X_train, y_train)
    y_pred = rbf_model.predict(X_val)

    acc = accuracy_score(y_val, y_pred)
    rbf_accuracies.append(acc)

    print(f"RBF SVM - C={C:.4f}, validation accuracy={acc:.4f}")

# best run
best_rbf_idx = np.argmax(rbf_accuracies)
best_rbf_C = C_values[best_rbf_idx]
best_rbf_acc = rbf_accuracies[best_rbf_idx]

print("\nBest RBF SVM result:")
print(f"Best C = {best_rbf_C}")
print(f"Best validation accuracy = {best_rbf_acc:.4f}")

# retrain best one
best_rbf_model = make_pipeline(
    StandardScaler(),
    SVC(kernel="rbf", C=best_rbf_C, gamma="scale", random_state=42)
)

best_rbf_model.fit(X_train, y_train)
best_rbf_pred = best_rbf_model.predict(X_val)

print("\nClassification report for best RBF SVM:")
print(classification_report(y_val, best_rbf_pred))

print("Confusion matrix for best RBF SVM:")
print(confusion_matrix(y_val, best_rbf_pred))

# plot acc
plt.figure(figsize=(8, 5))
plt.plot(C_values, rbf_accuracies, marker="o")
plt.xscale("log")
plt.xlabel("C")
plt.ylabel("Validation Accuracy")
plt.title("RBF SVM: Validation Accuracy vs C")
plt.grid(True)
plt.tight_layout()
plt.show()

# compare with linear result from part 3
best_linear_acc = 0.9694
best_linear_C = 0.1

print("\n--- Comparison with linear SVM ---")
print(f"Best linear SVM accuracy: {best_linear_acc:.4f} at C = {best_linear_C}")
print(f"Best RBF SVM accuracy:    {best_rbf_acc:.4f} at C = {best_rbf_C}")

if best_rbf_acc > best_linear_acc:
    print("The RBF kernel performs better than the linear SVM.")
    print("This suggests that the digits dataset is not perfectly linearly separable.")
    print("The kernel trick helps by modeling nonlinear decision boundaries.")
elif best_rbf_acc < best_linear_acc:
    print("The linear SVM performs better than the RBF SVM.")
    print("This suggests that a linear boundary is already sufficient for this dataset.")
else:
    print("The RBF kernel and the linear SVM achieve the same validation accuracy.")
    print("This suggests that the dataset is close to linearly separable.")
