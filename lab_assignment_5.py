from sklearn.datasets import fetch_openml
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC, SVC
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score
import time
import matplotlib.pyplot as plt
import pandas as pd

# ======================
# Load and prepare dataset
# ======================
mnist = fetch_openml('mnist_784', version=1, as_frame=False)
X, y = mnist["data"], mnist["target"]

X_train, X_test, y_train, y_test = X[:5000], X[60000:], y[:5000], y[60000:]

# === Scale your data ===
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# ======================
# Grid search function
# ======================
def grid_searching(model_name, param_grid, X_train, X_test, y_train, y_test):
    start_time = time.time()
    grid_search = GridSearchCV(model_name, param_grid, cv=3, n_jobs=-1, verbose=3)
    grid_search.fit(X_train, y_train)
    training_time = time.time() - start_time

    best_model = grid_search.best_estimator_
    print(f'\nBest estimator: {best_model}')

    y_pred = grid_search.predict(X_test)

    return {
        'Model': str(model_name),
        'Best_Params': grid_search.best_params_,
        'CV_Score': grid_search.best_score_,
        'Test_Accuracy': accuracy_score(y_test, y_pred),
        'Precision': precision_score(y_test, y_pred, average='macro'),
        'Recall': recall_score(y_test, y_pred, average='macro'),
        'F1': f1_score(y_test, y_pred, average='macro'),
        'Training_Time': training_time
    }

# ======================
# Parameter grids
# ======================
param_grid_linear = {
    'C': [0.1, 1, 10],
    'loss': ['squared_hinge'],
    'dual': [False],
    'tol': [1e-3],
    'max_iter': [2000]
}

param_grid_rbf = {
    'C': [1, 10],
    'gamma': ['scale', 0.1]
}

param_grid_poly = {
    'C': [1, 10],
    'degree': [2, 3],
    'gamma': ['scale'],
    'coef0': [0, 1]
}

# ======================
# Run models
# ======================
best_linear = grid_searching(LinearSVC(), param_grid_linear, X_train, X_test, y_train, y_test)
best_rbf = grid_searching(SVC(kernel='rbf'), param_grid_rbf, X_train, X_test, y_train, y_test)
best_poly = grid_searching(SVC(kernel='poly'), param_grid_poly, X_train, X_test, y_train, y_test)

results = [
    {**best_linear, 'Model': 'Linear SVC'},
    {**best_rbf, 'Model': 'RBF SVC'},
    {**best_poly, 'Model': 'Polynomial SVC'}
]

# ======================
# DataFrame + Summary
# ======================
results_df = pd.DataFrame(results)
print("\n=== Summary of All Models ===")
print(results_df[['Model', 'Test_Accuracy', 'Precision', 'Recall', 'F1', 'Training_Time']])

# ======================
# Performance Bar Chart
# ======================
metrics = ['Test_Accuracy', 'Precision', 'Recall', 'F1']
ax = results_df.set_index('Model')[metrics].plot(
    kind='bar',
    figsize=(14, 8),
    width=0.8
)

plt.title('Model Performance Comparison', fontsize=28, fontweight='bold')
plt.ylabel('Score', fontsize=22, fontweight='bold')
plt.xlabel('Model', fontsize=22, fontweight='bold')
plt.xticks(rotation=45, fontsize=20, fontweight='bold')
plt.yticks(fontsize=20, fontweight='bold')
plt.ylim(0, 1.0)

plt.legend(title='Metrics', fontsize=18, title_fontsize=20, loc='lower right')
plt.grid(axis='y', linestyle='--', linewidth=1.5, alpha=0.6)
plt.tight_layout()
plt.show()

# ======================
# Training Time Bar Chart
# ======================
plt.figure(figsize=(12, 7))
plt.bar(results_df['Model'], results_df['Training_Time'],
        color=['#5DADE2', '#F1948A', '#58D68D'], width=0.6, edgecolor='black')

plt.title('Training Time Comparison', fontsize=28, fontweight='bold')
plt.ylabel('Time (seconds)', fontsize=22, fontweight='bold')
plt.xlabel('Model', fontsize=22, fontweight='bold')
plt.xticks(rotation=45, fontsize=20, fontweight='bold')
plt.yticks(fontsize=20, fontweight='bold')
plt.grid(axis='y', linestyle='--', linewidth=1.5, alpha=0.6)
plt.tight_layout()
plt.show()
