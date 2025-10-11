from sklearn.datasets import fetch_openml
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import accuracy_score
from sklearn.model_selection import learning_curve



# Load MNIST dataset
mnist = fetch_openml('mnist_784', version=1, as_frame=False)
print(f'mnist keys : {mnist.keys()}\n')
for key in mnist.keys():
    print(f"\nThe {key}'s data type is {type(key)}")

X, y = mnist["data"], mnist["target"]
X_train, X_test, y_train, y_test = X[:60000], X[60000:], y[:60000], y[60000:]

# Reduce memory usage
X_train = X_train.astype(np.float32)
y_train = y_train.astype(np.float32)
X_test = X_test.astype(np.float32)
y_test = y_test.astype(np.float32)
# Small subset for quicker testing
X_train_small = X_train[:10000]
y_train_small = y_train[:10000]

# -------------------------
# KNN CLASSIFIER (small set)
# -------------------------
knn = KNeighborsClassifier()
knn.fit(X_train_small, y_train_small)

# Cross-validation on small set
cv_scores_knn = cross_val_score(knn, X_train_small, y_train_small, cv=5, scoring='accuracy', n_jobs=-1)
print("KNN CV Accuracy scores:", cv_scores_knn)
print("KNN Mean CV Accuracy:", cv_scores_knn.mean())

# Confusion matrix before grid search
y_pred = knn.predict(X_train_small)
cm = confusion_matrix(y_train_small, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot(cmap=plt.cm.Blues)
plt.title("Confusion Matrix - KNN (Before Grid Search)")
plt.show()

# -------------------------
# SGD CLASSIFIER (small set)
# -------------------------
sgd = SGDClassifier(random_state=42)
sgd.fit(X_train_small, y_train_small)

# Confusion matrix for SGD (small set)
y_pred = sgd.predict(X_train_small)
cm = confusion_matrix(y_train_small, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot(cmap=plt.cm.Blues)
plt.title("Confusion Matrix - SGD (Before Grid Search)")
plt.show()

# -------------------------
# GRID SEARCH - KNN (full set)
# -------------------------
param_grid_knn = {
    'n_neighbors': [3, 5, 7],
    'weights': ['uniform', 'distance'],
    'metric': ['euclidean', 'manhattan', 'minkowski'],
    'p': [1, 2]
}

knn_grid = GridSearchCV(knn, param_grid_knn, cv=3, n_jobs=-1, verbose=3)
knn_grid.fit(X_train, y_train)
knn_best = knn_grid.best_estimator_

print(f'KNN best params = {knn_grid.best_params_}')
print(f"KNN best estimator = {knn_best}\n")

knn_cv_score_grid = cross_val_score(knn_best, X_train, y_train, cv=3, n_jobs=-1, scoring='accuracy')
print("KNN CV grid scores:", knn_cv_score_grid)
print("KNN CV grid mean:", knn_cv_score_grid.mean())

y_pred = knn_best.predict(X_train)
cm = confusion_matrix(y_train, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot(cmap=plt.cm.Blues)
plt.title("Confusion Matrix - KNN (Best Estimator)")
plt.show()

# -------------------------
# GRID SEARCH - SGD (full set)
# -------------------------
param_grid_sgd = {
    'loss': ['hinge', 'log_loss'],
    'penalty': ['l2'],
    'alpha': [1e-4, 1e-3],
    'max_iter': [1000],
}

sgd_grid = GridSearchCV(sgd, param_grid_sgd, cv=3, n_jobs=-1, verbose=3)
sgd_grid.fit(X_train, y_train)
sgd_best = sgd_grid.best_estimator_

print(f'SGD best params = {sgd_grid.best_params_}')
print(f"SGD best estimator = {sgd_best}\n")

sgd_cv_score = cross_val_score(sgd_best, X_train, y_train, cv=3, n_jobs=-1, scoring='accuracy')
print("SGD CV grid scores:", sgd_cv_score)
print("SGD CV grid mean:", sgd_cv_score.mean())

y_pred = sgd_best.predict(X_train)
cm = confusion_matrix(y_train, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot(cmap=plt.cm.Blues)
plt.title("Confusion Matrix - SGD (Best Estimator)")
plt.show()

# -------------------------
# GRID SEARCH - RANDOM FOREST (full set)
# -------------------------
print("\nStarting Grid Search for Random Forest Classifier...")
rfc = RandomForestClassifier(random_state=42)

param_grid_rfc = {
    'n_estimators': [50, 100, 150],
    'max_features': ['sqrt', 'log2'],
    'min_samples_split': [2, 5],
}

rfc_grid = GridSearchCV(rfc, param_grid_rfc, cv=3, n_jobs=-1, verbose=3, scoring='accuracy')
rfc_grid.fit(X_train, y_train)
rfc_best = rfc_grid.best_estimator_

print(f'Random Forest best params = {rfc_grid.best_params_}')
print(f"Random Forest best estimator = {rfc_best}\n")

rfc_cv_score = cross_val_score(rfc_best, X_train, y_train, cv=3, n_jobs=-1, scoring='accuracy')
print("Random Forest CV grid scores:", rfc_cv_score)
print("Random Forest CV grid mean:", rfc_cv_score.mean())

y_pred = rfc_best.predict(X_train)
cm = confusion_matrix(y_train, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot(cmap=plt.cm.Blues)
plt.title("Confusion Matrix - Random Forest (Best Estimator)")
plt.show()

# Using our best KNN model on our test set
print("Evaluating the best KNN model on the test set...")
best_knn_model = KNeighborsClassifier(metric='euclidean', n_neighbors=3, p=1, weights='distance')

best_knn_model.fit(X_train, y_train)
# Make predictions on the test data
y_test_pred_knn = best_knn_model.predict(X_test)
# Calculate the accuracy
knn_test_accuracy = accuracy_score(y_test, y_test_pred_knn)
print(f"Final KNN Classifier Accuracy on Test Set: {knn_test_accuracy:.4f}")

# --- Using SGD classifier on our test set ---#
# Evaluate the best SGD model on the test set
print("\nEvaluating the best SGD model on the test set for comparison...")
best_sgd_model = SGDClassifier(alpha=0.001, loss='log_loss', random_state=42)
best_sgd_model.fit(X_train, y_train)
# Make predictions on the test data
y_test_pred_sgd = best_sgd_model.predict(X_test)
# Calculate the accuracy
sgd_test_accuracy = accuracy_score(y_test, y_test_pred_sgd)
print(f"Final SGD Classifier Accuracy on Test Set: {sgd_test_accuracy:.4f}")

# --- Using RFC classifier on our test set 

print("\nEvaluating the best SGD model on the test set for comparison...")
best_rfc_model = RandomForestClassifier(n_estimators=150, random_state=42)
best_rfc_model.fit(X_train, y_train)
# Make predictions on the test data
y_test_pred_rfc = best_rfc_model.predict(X_test)
# Calculate the accuracy
rfc_test_accuracy = accuracy_score(y_test, y_test_pred_rfc)
print(f"Final RFC Classifier Accuracy on Test Set: {rfc_test_accuracy:.4f}")

# Accuracy values for each model
models = ['KNN', 'SGD', 'Random Forest']
accuracies = [knn_test_accuracy, sgd_test_accuracy, rfc_test_accuracy]

# Create bar chart
plt.figure(figsize=(6, 4))
plt.bar(models, accuracies, color=['skyblue', 'lightcoral', 'lightgreen'])
plt.title('Model Accuracy Comparison on MNIST Test Set')
plt.ylabel('Accuracy')
plt.ylim(0, 1)  # accuracy goes from 0 to 1
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()

#=== Plotting learning curves===

# Define a function to plot learning curves
def plot_learning_curve(estimator, X, y, title, cv=3, train_sizes=np.linspace(0.1, 1.0, 5)):
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=-1, train_sizes=train_sizes, scoring='accuracy'
    )
    
    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    test_mean = np.mean(test_scores, axis=1)
    test_std = np.std(test_scores, axis=1)
    
    plt.figure(figsize=(6,4))
    plt.plot(train_sizes, train_mean, 'o-', color='blue', label='Training Accuracy')
    plt.plot(train_sizes, test_mean, 'o-', color='green', label='Validation Accuracy')
    plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, color='blue', alpha=0.1)
    plt.fill_between(train_sizes, test_mean - test_std, test_mean + test_std, color='green', alpha=0.1)
    plt.title(title)
    plt.xlabel('Training Set Size')
    plt.ylabel('Accuracy')
    plt.legend(loc='lower right')
    plt.grid(True)
    plt.show()

# Create estimators with best found parameters
best_knn = KNeighborsClassifier(metric='euclidean', n_neighbors=3, p=1, weights='distance')
best_sgd = SGDClassifier(alpha=0.001, loss='log_loss', random_state=42)
best_rfc = RandomForestClassifier(n_estimators=150, random_state=42)

# Plot learning curves
plot_learning_curve(best_knn, X_train, y_train, "Learning Curve - KNN")
plot_learning_curve(best_sgd, X_train, y_train, "Learning Curve - SGD")
plot_learning_curve(best_rfc, X_train, y_train, "Learning Curve - Random Forest")