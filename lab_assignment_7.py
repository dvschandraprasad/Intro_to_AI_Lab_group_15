# MNIST MLP Hyperparameter Tuning Assignment
# Optimized to achieve 98%+ accuracy

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import keras_tuner as kt
import numpy as np
import matplotlib.pyplot as plt
import time
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# STEP 1: LOAD AND PREPARE MNIST DATA
print("\nSTEP 1: LOADING MNIST DATASET")

(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

# Flatten images from 28x28 to 784 and normalize to 0-1 range
x_train = x_train.reshape(-1, 784).astype("float32") / 255.0
x_test = x_test.reshape(-1, 784).astype("float32") / 255.0

print(f"Training samples: {x_train.shape[0]}")
print(f"Test samples: {x_test.shape[0]}")
print(f"Input features per image: {x_train.shape[1]}")

# STEP 2: CREATE AND TRAIN BASELINE MODEL
print("\nSTEP 2: BASELINE MODEL")

# Simple baseline MLP with 2 hidden layers
baseline_model = keras.Sequential([
    layers.Dense(128, activation='relu', input_shape=(784,)),
    layers.Dense(64, activation='relu'),
    layers.Dense(10, activation='softmax')
])

baseline_model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

print("\nBaseline Model Architecture:")
baseline_model.summary()

# Train baseline model
print("\nTraining baseline model...")
baseline_start = time.time()
baseline_history = baseline_model.fit(
    x_train, y_train,
    epochs=10,
    batch_size=32,
    validation_split=0.1,
    verbose=1
)
baseline_time = time.time() - baseline_start

# Evaluate baseline on test set
baseline_loss, baseline_accuracy = baseline_model.evaluate(x_test, y_test, verbose=0)

print(f"\nBaseline Test Accuracy: {baseline_accuracy*100:.2f}%")
print(f"Baseline Test Loss: {baseline_loss:.4f}")
print(f"Baseline Training Time: {baseline_time:.2f} seconds")

# STEP 3: HYPERPARAMETER TUNING WITH KERAS TUNER (ALL PARAMETERS)
print("\nSTEP 3: HYPERPARAMETER TUNING (RANDOM SEARCH)")
print("Tuning: Layers, Units, Learning rate, Dropout, and Batch size")

# Define model-building function with all tunable hyperparameters
def build_tunable_model(hp):
    """Build MLP model with tunable hyperparameters including batch size"""
    model = keras.Sequential()
    
    # Input layer - optimized range (256-512)
    input_units = hp.Int('input_units', min_value=256, max_value=512, step=64)
    model.add(layers.Dense(input_units, activation='relu', input_shape=(784,)))
    
    # Hidden layers - focus on 2-3 layers (optimal for MNIST)
    num_layers = hp.Int('num_layers', min_value=2, max_value=3)
    
    for i in range(num_layers):
        units = hp.Int(f'layer_{i}_units', min_value=128, max_value=256, step=32)
        model.add(layers.Dense(units, activation='relu'))
        
        # Add dropout for regularization
        dropout = hp.Float(f'dropout_{i}', min_value=0.1, max_value=0.3, step=0.1)
        model.add(layers.Dropout(dropout))
    
    # Output layer
    model.add(layers.Dense(10, activation='softmax'))
    
    # Learning rate - narrowed range for better convergence
    learning_rate = hp.Float('learning_rate', min_value=5e-4, max_value=3e-3, sampling='log')
    
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

# Custom tuner class to include batch_size in hyperparameter search
class MyHyperModel(kt.HyperModel):
    def build(self, hp):
        return build_tunable_model(hp)
    
    def fit(self, hp, model, *args, **kwargs):
        # Add batch_size as a tunable hyperparameter
        batch_size = hp.Choice('batch_size', [32, 64, 128, 256])
        return model.fit(*args, batch_size=batch_size, **kwargs)

# Create tuner with custom hypermodel
print("\nInitializing Random Search Tuner (40 trials)...")
tuner = kt.RandomSearch(
    MyHyperModel(),
    objective='val_accuracy',
    max_trials=40,
    executions_per_trial=1,
    directory='mnist_tuning_final',
    project_name='mlp_tuning_complete',
    overwrite=True
)

# Run hyperparameter search
print("\nStarting hyperparameter search...")
print("This may take 30-60 minutes depending on your hardware...\n")

tuning_start = time.time()
tuner.search(
    x_train, y_train,
    epochs=15,
    validation_split=0.15,
    callbacks=[
        keras.callbacks.EarlyStopping(
            monitor='val_accuracy',
            patience=5,
            restore_best_weights=True
        )
    ],
    verbose=1
)
tuning_time = time.time() - tuning_start

print(f"\nHyperparameter search completed in {tuning_time/60:.2f} minutes")

# Get best hyperparameters
best_hp = tuner.get_best_hyperparameters(num_trials=1)[0]

# STEP 4: DISPLAY BEST HYPERPARAMETERS
print("\nBEST HYPERPARAMETERS FOUND")
print(f"Input layer units: {best_hp.get('input_units')}")
print(f"Number of hidden layers: {best_hp.get('num_layers')}")
for i in range(best_hp.get('num_layers')):
    print(f"  Hidden layer {i+1} units: {best_hp.get(f'layer_{i}_units')}")
    print(f"  Hidden layer {i+1} dropout: {best_hp.get(f'dropout_{i}'):.2f}")
print(f"Learning rate: {best_hp.get('learning_rate'):.6f}")
print(f"Batch size: {best_hp.get('batch_size')}")

# STEP 5: TRAIN FINAL MODEL WITH BEST HYPERPARAMETERS
print("\nSTEP 5: TRAINING FINAL MODEL")

print("Building final model with all optimized hyperparameters...")
final_model = build_tunable_model(best_hp)

print("\nFinal Model Architecture:")
final_model.summary()

# Count total parameters
total_params = final_model.count_params()
print(f"\nTotal parameters: {total_params:,}")

# Train final model with best hyperparameters
print("\nTraining final model...")
final_start = time.time()
final_history = final_model.fit(
    x_train, y_train,
    epochs=25,
    batch_size=best_hp.get('batch_size'),
    validation_split=0.15,
    callbacks=[
        keras.callbacks.EarlyStopping(
            monitor='val_accuracy',
            patience=7,
            restore_best_weights=True
        )
    ],
    verbose=1
)
final_time = time.time() - final_start

# STEP 6: EVALUATE FINAL MODEL ON TEST SET
print("\nSTEP 6: FINAL MODEL EVALUATION")

# Evaluate on test set
test_loss, test_accuracy = final_model.evaluate(x_test, y_test, verbose=0)

print(f"\nTest Accuracy: {test_accuracy*100:.2f}%")
print(f"Test Loss: {test_loss:.4f}")
print(f"Training Time: {final_time:.2f} seconds ({final_time/60:.2f} minutes)")
print(f"Total Epochs Trained: {len(final_history.history['accuracy'])}")

# Get predictions for detailed analysis
y_pred = final_model.predict(x_test, verbose=0)
y_pred_classes = np.argmax(y_pred, axis=1)

# Classification report
print("\nCLASSIFICATION REPORT (Per-Digit Performance)")
print(classification_report(y_test, y_pred_classes, 
                          target_names=[f'Digit {i}' for i in range(10)],
                          digits=4))

# STEP 7: COMPARISON WITH BASELINE MODEL
print("\nSTEP 7: BASELINE vs TUNED MODEL COMPARISON")

improvement = (test_accuracy - baseline_accuracy) * 100

print(f"\n{'Metric':<30} {'Baseline':<20} {'Tuned Model':<20}")
print("-" * 70)
print(f"{'Test Accuracy':<30} {baseline_accuracy*100:.2f}%{'':<16} {test_accuracy*100:.2f}%")
print(f"{'Test Loss':<30} {baseline_loss:.4f}{'':<16} {test_loss:.4f}")
print(f"{'Training Time':<30} {baseline_time:.2f}s{'':<16} {final_time:.2f}s")
print(f"{'Epochs Trained':<30} {len(baseline_history.history['accuracy']):<20} {len(final_history.history['accuracy'])}")

print(f"\nAccuracy Improvement: +{improvement:.2f}%")

if test_accuracy >= 0.98:
    print("TARGET ACHIEVED: Test accuracy >= 98%")
else:
    print(f"Test accuracy: {test_accuracy*100:.2f}% (Target: 98%)")

# STEP 8: CONVERGENCE ANALYSIS
print("\nSTEP 8: CONVERGENCE RATE ANALYSIS")

convergence_threshold = 0.95

# Find convergence epochs
baseline_conv_epoch = None
final_conv_epoch = None

for epoch, acc in enumerate(baseline_history.history['val_accuracy'], 1):
    if acc >= convergence_threshold and baseline_conv_epoch is None:
        baseline_conv_epoch = epoch

for epoch, acc in enumerate(final_history.history['val_accuracy'], 1):
    if acc >= convergence_threshold and final_conv_epoch is None:
        final_conv_epoch = epoch

print(f"\nEpochs to reach {convergence_threshold*100}% validation accuracy:")
print(f"  Baseline Model: {baseline_conv_epoch if baseline_conv_epoch else 'Not reached'} epochs")
print(f"  Tuned Model: {final_conv_epoch if final_conv_epoch else 'Not reached'} epochs")

if baseline_conv_epoch and final_conv_epoch:
    conv_improvement = baseline_conv_epoch - final_conv_epoch
    print(f"  Convergence Speed Improvement: {conv_improvement} epochs faster")

# STEP 9: VISUALIZATIONS
print("\nSTEP 9: GENERATING VISUALIZATIONS")

# Graph 1: Training Accuracy and Loss Comparison
plt.figure(figsize=(14, 6))

# Accuracy subplot
plt.subplot(1, 2, 1)
plt.plot(baseline_history.history['accuracy'], 'b-', label='Baseline Train', linewidth=2)
plt.plot(baseline_history.history['val_accuracy'], 'b--', label='Baseline Val', linewidth=2)
plt.plot(final_history.history['accuracy'], 'r-', label='Tuned Train', linewidth=2)
plt.plot(final_history.history['val_accuracy'], 'r--', label='Tuned Val', linewidth=2)
plt.title('Model Accuracy Comparison', fontsize=24, fontweight='bold')
plt.xlabel('Epoch', fontsize=20)
plt.ylabel('Accuracy', fontsize=20)
plt.legend(loc='lower right', fontsize=18)
plt.grid(True, alpha=0.3)
plt.tick_params(axis='both', labelsize=18)

# Loss subplot
plt.subplot(1, 2, 2)
plt.plot(baseline_history.history['loss'], 'b-', label='Baseline Train', linewidth=2)
plt.plot(baseline_history.history['val_loss'], 'b--', label='Baseline Val', linewidth=2)
plt.plot(final_history.history['loss'], 'r-', label='Tuned Train', linewidth=2)
plt.plot(final_history.history['val_loss'], 'r--', label='Tuned Val', linewidth=2)
plt.title('Model Loss Comparison', fontsize=24, fontweight='bold')
plt.xlabel('Epoch', fontsize=20)
plt.ylabel('Loss', fontsize=20)
plt.legend(loc='upper right', fontsize=18)
plt.grid(True, alpha=0.3)
plt.tick_params(axis='both', labelsize=18)

plt.tight_layout()
plt.savefig('training_comparison.png', dpi=300, bbox_inches='tight')
print("\nSaved: training_comparison.png")
plt.show()

# Graph 2: Confusion Matrix
plt.figure(figsize=(10, 8))
cm = confusion_matrix(y_test, y_pred_classes)
ax = sns.heatmap(
    cm, annot=True, fmt='d', cmap='Blues',
    xticklabels=range(10), yticklabels=range(10),
    cbar_kws={'label': 'Number of Predictions'},
    annot_kws={'size': 20, 'weight': 'bold'}
)

plt.title('Confusion Matrix - Tuned Model', fontsize=24, fontweight='bold')
plt.xlabel('Predicted Digit', fontsize=20)
plt.ylabel('True Digit', fontsize=20)

# Make axis tick labels large
ax.set_xticklabels(ax.get_xticklabels(), fontsize=18)
ax.set_yticklabels(ax.get_yticklabels(), fontsize=18)

# Make colorbar tick labels and label large
cbar = ax.collections[0].colorbar
cbar.ax.tick_params(labelsize=18)
cbar.ax.yaxis.label.set_size(20)

plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
print("Saved: confusion_matrix.png")
plt.show()


if test_accuracy >= 0.98:
    print(f"\nSUCCESS: Achieved target accuracy of >= 98%")
else:
    print(f"\nAchieved {test_accuracy*100:.2f}% accuracy")

