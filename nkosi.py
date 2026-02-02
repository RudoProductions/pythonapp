import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import os

# Deep Learning
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, callbacks
from tensorflow.keras.utils import to_categorical

# Machine Learning
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, StandardScaler
from sklearn.metrics import (classification_report, confusion_matrix, 
                           roc_curve, auc, accuracy_score, 
                           precision_recall_fscore_support, roc_auc_score)

# Create directories for saving models and visualizations
os.makedirs('saved_models', exist_ok=True)
os.makedirs('visualizations', exist_ok=True)

print("Directories created: 'saved_models/', 'visualizations/'")

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

print("Libraries imported successfully!")
print(f"TensorFlow version: {tf.__version__}")

# Load dataset 
df = pd.read_csv("IoT_Dataset.csv")

print("Dataset loaded successfully!")
print(f"Shape: {df.shape}")
print(f"Columns: {list(df.columns)}")
print("\nFirst few rows:")
print(df.head())

print("\nDataset Info:")
df.info()

print("\nLabel distribution:")
print(df['Label'].value_counts())

print("\nAttack categories:")
print(df['Cat'].value_counts())

print("Starting preprocessing...")

# Separate features and target
y = df["Label"]
X = df.drop(columns=["Flow_ID","Src_IP","Dst_IP","Timestamp","Cat","Sub_Cat","Label"], errors="ignore")

print(f"Features shape: {X.shape}")
print(f"Target shape: {y.shape}")

# Check for missing values
print("\nMissing values per column:")
print(X.isnull().sum().sum())

# Check data types
print("\nData types:")
print(X.dtypes.value_counts())

# Encoding target (same as AE+XGBoost)
le = LabelEncoder()
y_encoded = le.fit_transform(y)
print("Classes:", le.classes_)
print("Encoded labels:", np.unique(y_encoded, return_counts=True))

# Train-test split (same parameters as AE+XGBoost)
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)

print(f"Train set: {X_train.shape}, {y_train.shape}")
print(f"Test set: {X_test.shape}, {y_test.shape}")

print("Handling infinite values and missing data...")

# Step A: Replacng infinities with NaN (same as AE+XGBoost)
X_train = X_train.replace([np.inf, -np.inf], np.nan)
X_test = X_test.replace([np.inf, -np.inf], np.nan)

# Step B: Drop rows with NaN
mask_train = X_train.notna().all(axis=1)
mask_test = X_test.notna().all(axis=1)

print(f"Rows to remove - Train: {(~mask_train).sum()}, Test: {(~mask_test).sum()}")

X_train = X_train[mask_train]
y_train = y_train[mask_train]
X_test = X_test[mask_test]
y_test = y_test[mask_test]

print(f"After cleaning - Train: {X_train.shape}, Test: {X_test.shape}")

# Scaling features (same approach as AE+XGBoost)
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"Scaled features - Train: {X_train_scaled.shape}, Test: {X_test_scaled.shape}")
print(f"Feature range after scaling: [{X_train_scaled.min():.3f}, {X_train_scaled.max():.3f}]")

def create_temporal_sequences(X, y, sequence_length=10, step=1):
    """
    Create temporal sequences from tabular data for LSTM
    
    Args:
        X: Feature matrix
        y: Labels
        sequence_length: Number of consecutive samples per sequence
        step: Step size for sliding window (1 = no overlap)
    
    Returns:
        X_seq: 3D array (n_sequences, sequence_length, n_features)
        y_seq: 1D array (n_sequences,)
    """
    sequences_X = []
    sequences_y = []
    
    # Create sliding windows
    for i in range(0, len(X) - sequence_length + 1, step):
        sequences_X.append(X[i:i + sequence_length])
        # Use label of the last sample in the sequence
        sequences_y.append(y[i + sequence_length - 1])
    
    return np.array(sequences_X), np.array(sequences_y)

# Create sequences
sequence_length = 15  # different values (10, 15, 20) for experimentation
step_size = 5  # Creates overlapping sequences for more training data

print(f"Creating sequences with length={sequence_length}, step={step_size}...")

X_train_seq, y_train_seq = create_temporal_sequences(X_train_scaled, y_train, 
                                                    sequence_length, step_size)
X_test_seq, y_test_seq = create_temporal_sequences(X_test_scaled, y_test, 
                                                  sequence_length, step_size)

print(f"Sequence shapes:")
print(f"X_train_seq: {X_train_seq.shape}")  # (n_sequences, sequence_length, n_features)
print(f"X_test_seq: {X_test_seq.shape}")
print(f"y_train_seq: {y_train_seq.shape}")
print(f"y_test_seq: {y_test_seq.shape}")

print(f"\nClass distribution in sequences:")
print(f"Train: {np.bincount(y_train_seq)}")
print(f"Test: {np.bincount(y_test_seq)}")

# Calculating class weights to handle imbalance
class_weights = {}
unique_labels, counts = np.unique(y_train_seq, return_counts=True)
total_samples = len(y_train_seq)

for label, count in zip(unique_labels, counts):
    class_weights[label] = total_samples / (len(unique_labels) * count)

print("Class weights:", class_weights)

def build_cnn_model(input_shape, num_classes=1):
    """
    Build CNN-only model for intrusion detection
    Uses 1D convolutions to extract spatial patterns
    """
    
    model = models.Sequential()
    
    # Input layer
    model.add(layers.Input(shape=input_shape))
    
    # CNN Block 1 - Small kernel for local patterns
    model.add(layers.Conv1D(filters=128, kernel_size=3, activation='relu', padding='same'))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling1D(pool_size=2))
    model.add(layers.Dropout(0.3))
    
    # CNN Block 2 - Medium kernel
    model.add(layers.Conv1D(filters=64, kernel_size=5, activation='relu', padding='same'))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling1D(pool_size=2))
    model.add(layers.Dropout(0.3))
    
    # CNN Block 3 - Larger kernel for broader patterns
    model.add(layers.Conv1D(filters=32, kernel_size=7, activation='relu', padding='same'))
    model.add(layers.BatchNormalization())
    model.add(layers.GlobalMaxPooling1D())  # Reduce to fixed size
    
    # Dense layers for classification
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.Dropout(0.4))
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dropout(0.3))
    model.add(layers.Dense(32, activation='relu'))
    model.add(layers.Dropout(0.3))
    
    # Output layer
    model.add(layers.Dense(num_classes, activation='sigmoid'))
    
    return model

# Building CNN model
input_shape = (sequence_length, X_train_scaled.shape[1])
print(f"Building CNN model with input shape: {input_shape}")

cnn_model = build_cnn_model(input_shape)

# Compiling
cnn_model.compile(
    optimizer=optimizers.Adam(learning_rate=0.001),
    loss='binary_crossentropy',
    metrics=['accuracy', 'precision', 'recall']
)

print("\nCNN Model Architecture:")
cnn_model.summary()

def build_lstm_model(input_shape, num_classes=1):
    """
    Build LSTM-only model for intrusion detection
    Focuses purely on temporal sequence modeling
    """
    
    model = models.Sequential()
    
    # Input layer
    model.add(layers.Input(shape=input_shape))
    
    # LSTM Block 1 - Bidirectional for better context
    model.add(layers.Bidirectional(layers.LSTM(128, return_sequences=True, 
                                               dropout=0.3, recurrent_dropout=0.2)))
    model.add(layers.BatchNormalization())
    
    # LSTM Block 2
    model.add(layers.Bidirectional(layers.LSTM(64, return_sequences=True,
                                               dropout=0.3, recurrent_dropout=0.2)))
    model.add(layers.BatchNormalization())
    
    # LSTM Block 3 - Final sequence encoding
    model.add(layers.LSTM(32, dropout=0.3, recurrent_dropout=0.2))
    
    # Dense layers for classification
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dropout(0.4))
    model.add(layers.Dense(32, activation='relu'))
    model.add(layers.Dropout(0.3))
    
    # Output layer
    model.add(layers.Dense(num_classes, activation='sigmoid'))
    
    return model

# Building LSTM model
print(f"\nBuilding LSTM model with input shape: {input_shape}")

lstm_model = build_lstm_model(input_shape)

# Compiling
lstm_model.compile(
    optimizer=optimizers.Adam(learning_rate=0.001),
    loss='binary_crossentropy',
    metrics=['accuracy', 'precision', 'recall']
)

print("\nLSTM Model Architecture:")
lstm_model.summary()

def build_cnn_lstm_hybrid(input_shape, num_classes=1):
    """
    Build CNN+LSTM hybrid model
    CNN extracts features, LSTM models temporal dependencies
    """
    
    model = models.Sequential()
    
    # Input layer
    model.add(layers.Input(shape=input_shape))
    
    # CNN Block 1 - Extract local patterns
    model.add(layers.Conv1D(filters=64, kernel_size=3, activation='relu', padding='same'))
    model.add(layers.BatchNormalization())
    model.add(layers.Dropout(0.2))
    
    # CNN Block 2 - Multi-scale feature extraction
    model.add(layers.Conv1D(filters=32, kernel_size=5, activation='relu', padding='same'))
    model.add(layers.BatchNormalization())
    model.add(layers.Dropout(0.2))
    
    # CNN Block 3 - Higher level features
    model.add(layers.Conv1D(filters=16, kernel_size=3, activation='relu', padding='same'))
    model.add(layers.BatchNormalization())
    
    # LSTM Block 1 - Temporal modeling
    model.add(layers.LSTM(64, return_sequences=True, dropout=0.3, recurrent_dropout=0.2))
    
    # LSTM Block 2 - Final sequence encoding
    model.add(layers.LSTM(32, dropout=0.3, recurrent_dropout=0.2))
    
    # Dense layers for classification
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dropout(0.4))
    model.add(layers.Dense(32, activation='relu'))
    model.add(layers.Dropout(0.3))
    
    # Output layer
    model.add(layers.Dense(num_classes, activation='sigmoid'))
    
    return model

# Building CNN+LSTM hybrid model
print(f"\nBuilding CNN+LSTM Hybrid model with input shape: {input_shape}")

cnn_lstm_model = build_cnn_lstm_hybrid(input_shape)

# Compiling
cnn_lstm_model.compile(
    optimizer=optimizers.Adam(learning_rate=0.001),
    loss='binary_crossentropy',
    metrics=['accuracy', 'precision', 'recall']
)

print("\nCNN+LSTM Hybrid Model Architecture:")
cnn_lstm_model.summary()

# Define callbacks (same for all models)
def get_callbacks(model_name):
    """Create callbacks for model training"""
    return [
        callbacks.EarlyStopping(
            monitor='val_loss', 
            patience=10, 
            restore_best_weights=True,
            verbose=1
        ),
        callbacks.ReduceLROnPlateau(
            monitor='val_loss', 
            factor=0.5, 
            patience=5, 
            min_lr=1e-6,
            verbose=1
        ),
        callbacks.ModelCheckpoint(
            f'saved_models/best_{model_name}_model.h5',
            monitor='val_loss',
            save_best_only=True,
            verbose=1
        )
    ]

# Training parameters
epochs = 100
batch_size = 32
validation_split = 0.2

print("Training Configuration:")
print(f"  Epochs: {epochs}")
print(f"  Batch Size: {batch_size}")
print(f"  Validation Split: {validation_split}")
print(f"  Class Weights: {class_weights}")

print("\n" + "="*60)
print("TRAINING MODEL 1: CNN Only")
print("="*60)

history_cnn = cnn_model.fit(
    X_train_seq, y_train_seq,
    epochs=epochs,
    batch_size=batch_size,
    validation_split=validation_split,
    class_weight=class_weights,
    callbacks=get_callbacks('cnn'),
    verbose=1
)

print("CNN model training completed!")

print("\n" + "="*60)
print("TRAINING MODEL 2: LSTM Only")
print("="*60)

history_lstm = lstm_model.fit(
    X_train_seq, y_train_seq,
    epochs=epochs,
    batch_size=batch_size,
    validation_split=validation_split,
    class_weight=class_weights,
    callbacks=get_callbacks('lstm'),
    verbose=1
)

print("LSTM model training completed!")

print("\n" + "="*60)
print("TRAINING MODEL 3: CNN+LSTM Hybrid")
print("="*60)

history_hybrid = cnn_lstm_model.fit(
    X_train_seq, y_train_seq,
    epochs=epochs,
    batch_size=batch_size,
    validation_split=validation_split,
    class_weight=class_weights,
    callbacks=get_callbacks('cnn_lstm_hybrid'),
    verbose=1
)

print("CNN+LSTM Hybrid model training completed!")

def plot_all_training_histories(history_cnn, history_lstm, history_hybrid, save_path=None):
    """Compare training histories of all three models"""
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Loss
    axes[0,0].plot(history_cnn.history['loss'], label='CNN - Train', alpha=0.7)
    axes[0,0].plot(history_cnn.history['val_loss'], label='CNN - Val', alpha=0.7)
    axes[0,0].plot(history_lstm.history['loss'], label='LSTM - Train', alpha=0.7)
    axes[0,0].plot(history_lstm.history['val_loss'], label='LSTM - Val', alpha=0.7)
    axes[0,0].plot(history_hybrid.history['loss'], label='Hybrid - Train', alpha=0.7)
    axes[0,0].plot(history_hybrid.history['val_loss'], label='Hybrid - Val', alpha=0.7)
    axes[0,0].set_title('Training & Validation Loss Comparison')
    axes[0,0].set_xlabel('Epoch')
    axes[0,0].set_ylabel('Loss')
    axes[0,0].legend()
    
    # Accuracy
    axes[0,1].plot(history_cnn.history['accuracy'], label='CNN - Train', alpha=0.7)
    axes[0,1].plot(history_cnn.history['val_accuracy'], label='CNN - Val', alpha=0.7)
    axes[0,1].plot(history_lstm.history['accuracy'], label='LSTM - Train', alpha=0.7)
    axes[0,1].plot(history_lstm.history['val_accuracy'], label='LSTM - Val', alpha=0.7)
    axes[0,1].plot(history_hybrid.history['accuracy'], label='Hybrid - Train', alpha=0.7)
    axes[0,1].plot(history_hybrid.history['val_accuracy'], label='Hybrid - Val', alpha=0.7)
    axes[0,1].set_title('Training & Validation Accuracy Comparison')
    axes[0,1].set_xlabel('Epoch')
    axes[0,1].set_ylabel('Accuracy')
    axes[0,1].legend()
    
    # Precision
    axes[1,0].plot(history_cnn.history['precision'], label='CNN - Train', alpha=0.7)
    axes[1,0].plot(history_cnn.history['val_precision'], label='CNN - Val', alpha=0.7)
    axes[1,0].plot(history_lstm.history['precision'], label='LSTM - Train', alpha=0.7)
    axes[1,0].plot(history_lstm.history['val_precision'], label='LSTM - Val', alpha=0.7)
    axes[1,0].plot(history_hybrid.history['precision'], label='Hybrid - Train', alpha=0.7)
    axes[1,0].plot(history_hybrid.history['val_precision'], label='Hybrid - Val', alpha=0.7)
    axes[1,0].set_title('Training & Validation Precision Comparison')
    axes[1,0].set_xlabel('Epoch')
    axes[1,0].set_ylabel('Precision')
    axes[1,0].legend()
    
    # Recall
    axes[1,1].plot(history_cnn.history['recall'], label='CNN - Train', alpha=0.7)
    axes[1,1].plot(history_cnn.history['val_recall'], label='CNN - Val', alpha=0.7)
    axes[1,1].plot(history_lstm.history['recall'], label='LSTM - Train', alpha=0.7)
    axes[1,1].plot(history_lstm.history['val_recall'], label='LSTM - Val', alpha=0.7)
    axes[1,1].plot(history_hybrid.history['recall'], label='Hybrid - Train', alpha=0.7)
    axes[1,1].plot(history_hybrid.history['val_recall'], label='Hybrid - Val', alpha=0.7)
    axes[1,1].set_title('Training & Validation Recall Comparison')
    axes[1,1].set_xlabel('Epoch')
    axes[1,1].set_ylabel('Recall')
    axes[1,1].legend()
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Training history plot saved to: {save_path}")
    
    plt.show()

# Plot all training histories and save
plot_all_training_histories(history_cnn, history_lstm, history_hybrid, 
                           save_path='visualizations/training_history_comparison.png')

print("Making predictions with all models...")

# CNN predictions
y_pred_proba_cnn = cnn_model.predict(X_test_seq, verbose=0)
y_pred_cnn = (y_pred_proba_cnn > 0.5).astype(int).flatten()

# LSTM predictions
y_pred_proba_lstm = lstm_model.predict(X_test_seq, verbose=0)
y_pred_lstm = (y_pred_proba_lstm > 0.5).astype(int).flatten()

# CNN+LSTM Hybrid predictions
y_pred_proba_hybrid = cnn_lstm_model.predict(X_test_seq, verbose=0)
y_pred_hybrid = (y_pred_proba_hybrid > 0.5).astype(int).flatten()

print("Predictions completed for all models!")

def evaluate_model(y_true, y_pred, y_pred_proba, model_name):
    """Evaluate a model and return metrics"""
    
    print(f"\n{'='*60}")
    print(f"EVALUATION: {model_name}")
    print('='*60)
    
    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    print("\nConfusion Matrix:")
    print(cm)
    
    # Classification Report
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=le.classes_))
    
    # Calculate metrics
    accuracy = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='weighted')
    auc_score = roc_auc_score(y_true, y_pred_proba)
    
    print(f"\nSummary Metrics:")
    print(f"  Accuracy:  {accuracy:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall:    {recall:.4f}")
    print(f"  F1-Score:  {f1:.4f}")
    print(f"  AUC Score: {auc_score:.4f}")
    
    return {
        'Model': model_name,
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall,
        'F1': f1,
        'AUC': auc_score,
        'Confusion_Matrix': cm
    }

# Evaluate all models
results_cnn = evaluate_model(y_test_seq, y_pred_cnn, y_pred_proba_cnn, "CNN Only")
results_lstm = evaluate_model(y_test_seq, y_pred_lstm, y_pred_proba_lstm, "LSTM Only")
results_hybrid = evaluate_model(y_test_seq, y_pred_hybrid, y_pred_proba_hybrid, "CNN+LSTM Hybrid")

# comparison DataFrame
comparison_df = pd.DataFrame([results_cnn, results_lstm, results_hybrid])
comparison_df = comparison_df[['Model', 'Accuracy', 'Precision', 'Recall', 'F1', 'AUC']]

print("\n" + "="*70)
print("COMPREHENSIVE MODEL COMPARISON")
print("="*70)
print(comparison_df.to_string(index=False))
print("="*70)

# Visualize comparison and save
plt.figure(figsize=(12, 6))
comparison_df.set_index('Model')[['Accuracy', 'Precision', 'Recall', 'F1', 'AUC']].plot(
    kind='bar', figsize=(12, 6), width=0.8
)
plt.title('Model Performance Comparison: CNN vs LSTM vs CNN+LSTM Hybrid', fontsize=14, fontweight='bold')
plt.ylabel('Score', fontsize=12)
plt.xlabel('Model', fontsize=12)
plt.xticks(rotation=45, ha='right')
plt.ylim(0, 1.1)
plt.legend(loc='lower right')
plt.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig('visualizations/model_performance_comparison.png', dpi=300, bbox_inches='tight')
plt.show()
print("Model performance comparison plot saved to: visualizations/model_performance_comparison.png")

# Confusion matrices visualization
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# CNN Confusion Matrix
sns.heatmap(results_cnn['Confusion_Matrix'], annot=True, fmt="d", cmap="Blues",
            xticklabels=le.classes_, yticklabels=le.classes_, ax=axes[0])
axes[0].set_title('CNN Only')
axes[0].set_xlabel('Predicted')
axes[0].set_ylabel('Actual')

# LSTM Confusion Matrix
sns.heatmap(results_lstm['Confusion_Matrix'], annot=True, fmt="d", cmap="Blues",
            xticklabels=le.classes_, yticklabels=le.classes_, ax=axes[1])
axes[1].set_title('LSTM Only')
axes[1].set_xlabel('Predicted')
axes[1].set_ylabel('Actual')

# CNN+LSTM Hybrid Confusion Matrix
sns.heatmap(results_hybrid['Confusion_Matrix'], annot=True, fmt="d", cmap="Blues",
            xticklabels=le.classes_, yticklabels=le.classes_, ax=axes[2])
axes[2].set_title('CNN+LSTM Hybrid')
axes[2].set_xlabel('Predicted')
axes[2].set_ylabel('Actual')

plt.tight_layout()
plt.savefig('visualizations/confusion_matrices.png', dpi=300, bbox_inches='tight')
plt.show()
print("Confusion matrices plot saved to: visualizations/confusion_matrices.png")

# ROC curves visualization
plt.figure(figsize=(10, 8))

# CNN ROC
fpr_cnn, tpr_cnn, _ = roc_curve(y_test_seq, y_pred_proba_cnn)
auc_cnn = auc(fpr_cnn, tpr_cnn)
plt.plot(fpr_cnn, tpr_cnn, linewidth=2, label=f'CNN Only (AUC = {auc_cnn:.3f})')

# LSTM ROC
fpr_lstm, tpr_lstm, _ = roc_curve(y_test_seq, y_pred_proba_lstm)
auc_lstm = auc(fpr_lstm, tpr_lstm)
plt.plot(fpr_lstm, tpr_lstm, linewidth=2, label=f'LSTM Only (AUC = {auc_lstm:.3f})')

# CNN+LSTM Hybrid ROC
fpr_hybrid, tpr_hybrid, _ = roc_curve(y_test_seq, y_pred_proba_hybrid)
auc_hybrid = auc(fpr_hybrid, tpr_hybrid)
plt.plot(fpr_hybrid, tpr_hybrid, linewidth=2, label=f'CNN+LSTM Hybrid (AUC = {auc_hybrid:.3f})')

# Diagonal reference line
plt.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random Classifier')

plt.xlabel('False Positive Rate', fontsize=12)
plt.ylabel('True Positive Rate', fontsize=12)
plt.title('ROC Curves Comparison: CNN vs LSTM vs CNN+LSTM Hybrid', fontsize=14, fontweight='bold')
plt.legend(loc='lower right', fontsize=11)
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('visualizations/roc_curves_comparison.png', dpi=300, bbox_inches='tight')
plt.show()
print("ROC curves plot saved to: visualizations/roc_curves_comparison.png")

def get_per_class_metrics(y_true, y_pred, model_name):
    """Get per-class precision, recall, f1"""
    prec, rec, f1, support = precision_recall_fscore_support(y_true, y_pred, labels=[0, 1])
    
    results = []
    for i, class_name in enumerate(le.classes_):
        results.append({
            'Model': model_name,
            'Class': class_name,
            'Precision': prec[i],
            'Recall': rec[i],
            'F1-Score': f1[i],
            'Support': support[i]
        })
    return results

# Collect per-class metrics
per_class_results = []
per_class_results.extend(get_per_class_metrics(y_test_seq, y_pred_cnn, "CNN"))
per_class_results.extend(get_per_class_metrics(y_test_seq, y_pred_lstm, "LSTM"))
per_class_results.extend(get_per_class_metrics(y_test_seq, y_pred_hybrid, "Hybrid"))

per_class_df = pd.DataFrame(per_class_results)

print("\nPer-Class Performance Comparison:")
print(per_class_df.to_string(index=False))

# Visualize per-class performance and save
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

for i, metric in enumerate(['Precision', 'Recall', 'F1-Score']):
    pivot_df = per_class_df.pivot(index='Class', columns='Model', values=metric)
    pivot_df.plot(kind='bar', ax=axes[i], width=0.8)
    axes[i].set_title(f'{metric} by Class', fontsize=12, fontweight='bold')
    axes[i].set_xlabel('Class', fontsize=11)
    axes[i].set_ylabel(metric, fontsize=11)
    axes[i].set_ylim(0, 1.1)
    axes[i].legend(title='Model')
    axes[i].grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('visualizations/per_class_performance.png', dpi=300, bbox_inches='tight')
plt.show()
print("Per-class performance plot saved to: visualizations/per_class_performance.png")

# Get model parameters
model_info = {
    'Model': ['CNN Only', 'LSTM Only', 'CNN+LSTM Hybrid'],
    'Total Parameters': [
        cnn_model.count_params(),
        lstm_model.count_params(),
        cnn_lstm_model.count_params()
    ],
    'Trainable Parameters': [
        sum([tf.size(w).numpy() for w in cnn_model.trainable_weights]),
        sum([tf.size(w).numpy() for w in lstm_model.trainable_weights]),
        sum([tf.size(w).numpy() for w in cnn_lstm_model.trainable_weights])
    ]
}

model_complexity_df = pd.DataFrame(model_info)
print("\n" + "="*60)
print("MODEL COMPLEXITY COMPARISON")
print("="*60)
print(model_complexity_df.to_string(index=False))
print("="*60)

# Visualize model complexity and save
fig, ax = plt.subplots(figsize=(10, 6))
x = np.arange(len(model_complexity_df['Model']))
width = 0.35

bars1 = ax.bar(x - width/2, model_complexity_df['Total Parameters'], width, 
               label='Total Parameters', alpha=0.8)
bars2 = ax.bar(x + width/2, model_complexity_df['Trainable Parameters'], width,
               label='Trainable Parameters', alpha=0.8)

ax.set_xlabel('Model', fontsize=12)
ax.set_ylabel('Number of Parameters', fontsize=12)
ax.set_title('Model Complexity Comparison', fontsize=14, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(model_complexity_df['Model'])
ax.legend()
ax.grid(axis='y', alpha=0.3)

# value labels on bars
for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height):,}',
                ha='center', va='bottom', fontsize=9)

plt.tight_layout()
plt.savefig('visualizations/model_complexity.png', dpi=300, bbox_inches='tight')
plt.show()
print("Model complexity plot saved to: visualizations/model_complexity.png")

# training time from history
def get_training_epochs(history):
    """Get actual number of epochs trained"""
    return len(history.history['loss'])

training_info = {
    'Model': ['CNN Only', 'LSTM Only', 'CNN+LSTM Hybrid'],
    'Epochs Trained': [
        get_training_epochs(history_cnn),
        get_training_epochs(history_lstm),
        get_training_epochs(history_hybrid)
    ],
    'Final Train Loss': [
        history_cnn.history['loss'][-1],
        history_lstm.history['loss'][-1],
        history_hybrid.history['loss'][-1]
    ],
    'Final Val Loss': [
        history_cnn.history['val_loss'][-1],
        history_lstm.history['val_loss'][-1],
        history_hybrid.history['val_loss'][-1]
    ]
}

training_df = pd.DataFrame(training_info)
print("\n" + "="*60)
print("TRAINING ANALYSIS")
print("="*60)
print(training_df.to_string(index=False))
print("="*60)

# Radar chart visualization and save
from math import pi

categories = ['Accuracy', 'Precision', 'Recall', 'F1', 'AUC']
N = len(categories)

# Prepare data
values_cnn = comparison_df[comparison_df['Model'] == 'CNN Only'][categories].values.flatten().tolist()
values_lstm = comparison_df[comparison_df['Model'] == 'LSTM Only'][categories].values.flatten().tolist()
values_hybrid = comparison_df[comparison_df['Model'] == 'CNN+LSTM Hybrid'][categories].values.flatten().tolist()

# Close the plot
values_cnn += values_cnn[:1]
values_lstm += values_lstm[:1]
values_hybrid += values_hybrid[:1]

# Compute angle for each axis
angles = [n / float(N) * 2 * pi for n in range(N)]
angles += angles[:1]

# Initialize plot
fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))

# Plot data
ax.plot(angles, values_cnn, 'o-', linewidth=2, label='CNN Only', color='#1f77b4')
ax.fill(angles, values_cnn, alpha=0.15, color='#1f77b4')

ax.plot(angles, values_lstm, 'o-', linewidth=2, label='LSTM Only', color='#ff7f0e')
ax.fill(angles, values_lstm, alpha=0.15, color='#ff7f0e')

ax.plot(angles, values_hybrid, 'o-', linewidth=2, label='CNN+LSTM Hybrid', color='#2ca02c')
ax.fill(angles, values_hybrid, alpha=0.15, color='#2ca02c')

# Fix axis to go in the right order
ax.set_xticks(angles[:-1])
ax.set_xticklabels(categories, size=12)

# Set y-axis limits
ax.set_ylim(0, 1)
ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
ax.set_yticklabels(['0.2', '0.4', '0.6', '0.8', '1.0'], size=10)

# Add legend and title
plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=11)
plt.title('Overall Model Performance - Radar Chart', size=14, fontweight='bold', pad=20)

plt.tight_layout()
plt.savefig('visualizations/radar_chart_performance.png', dpi=300, bbox_inches='tight')
plt.show()
print("Radar chart saved to: visualizations/radar_chart_performance.png")

final_comparison = comparison_df.copy()

# Adding rankings
for metric in ['Accuracy', 'Precision', 'Recall', 'F1', 'AUC']:
    final_comparison[f'{metric}_Rank'] = final_comparison[metric].rank(ascending=False).astype(int)

# Calculate average rank
rank_columns = [col for col in final_comparison.columns if '_Rank' in col]
final_comparison['Avg_Rank'] = final_comparison[rank_columns].mean(axis=1)
final_comparison['Overall_Rank'] = final_comparison['Avg_Rank'].rank().astype(int)

print("\n" + "="*80)
print("FINAL COMPARISON WITH RANKINGS (1 = Best)")
print("="*80)
print(final_comparison.to_string(index=False))
print("="*80)

# Visualize rankings and save
fig, ax = plt.subplots(figsize=(12, 6))

models = final_comparison['Model']
metrics = ['Accuracy', 'Precision', 'Recall', 'F1', 'AUC']
x = np.arange(len(metrics))
width = 0.25

for i, model in enumerate(models):
    ranks = [final_comparison[final_comparison['Model'] == model][f'{m}_Rank'].values[0] 
             for m in metrics]
    ax.bar(x + i*width, ranks, width, label=model, alpha=0.8)

ax.set_xlabel('Metrics', fontsize=12)
ax.set_ylabel('Rank (Lower is Better)', fontsize=12)
ax.set_title('Model Rankings by Metric', fontsize=14, fontweight='bold')
ax.set_xticks(x + width)
ax.set_xticklabels(metrics)
ax.set_yticks([1, 2, 3])
ax.set_ylim(0, 3.5)
ax.legend()
ax.invert_yaxis()  # Lower rank is better
ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('visualizations/model_rankings.png', dpi=300, bbox_inches='tight')
plt.show()
print("Model rankings plot saved to: visualizations/model_rankings.png")

# Saving comparison results to CSV
comparison_df.to_csv('visualizations/model_comparison_results.csv', index=False)
print("\nComparison results saved to: 'visualizations/model_comparison_results.csv'")

# Save per-class results
per_class_df.to_csv('visualizations/per_class_comparison_results.csv', index=False)
print("Per-class results saved to: 'visualizations/per_class_comparison_results.csv'")

# Save training history data for dashboard
training_history_data = {
    'cnn': history_cnn.history,
    'lstm': history_lstm.history,
    'hybrid': history_hybrid.history
}

import json
with open('visualizations/training_history.json', 'w') as f:
    # Convert numpy arrays to lists for JSON serialization
    json_data = {}
    for model_name, history in training_history_data.items():
        json_data[model_name] = {}
        for key, values in history.items():
            json_data[model_name][key] = [float(x) for x in values]
    json.dump(json_data, f)
print("Training history data saved to: 'visualizations/training_history.json'")

# Save models in the saved_models folder
cnn_model.save('saved_models/cnn_model_final.h5')
lstm_model.save('saved_models/lstm_model_final.h5')
cnn_lstm_model.save('saved_models/cnn_lstm_hybrid_model_final.h5')

print("\nFinal models saved in 'saved_models/' folder:")
print("  - saved_models/cnn_model_final.h5")
print("  - saved_models/lstm_model_final.h5")
print("  - saved_models/cnn_lstm_hybrid_model_final.h5")

print("\nBest models saved during training:")
print("  - saved_models/best_cnn_model.h5")
print("  - saved_models/best_lstm_model.h5")
print("  - saved_models/best_cnn_lstm_hybrid_model.h5")

print("\n" + "="*80)
print("KEY INSIGHTS AND ANALYSIS")
print("="*80)

# best model
best_model_idx = final_comparison['Overall_Rank'].idxmin()
best_model = final_comparison.loc[best_model_idx, 'Model']
best_accuracy = final_comparison.loc[best_model_idx, 'Accuracy']
best_f1 = final_comparison.loc[best_model_idx, 'F1']
best_auc = final_comparison.loc[best_model_idx, 'AUC']

print(f"\n🏆 BEST OVERALL MODEL: {best_model}")
print(f"   - Accuracy: {best_accuracy:.4f}")
print(f"   - F1-Score: {best_f1:.4f}")
print(f"   - AUC Score: {best_auc:.4f}")

print("\n📊 MODEL CHARACTERISTICS:")

print("\n1. CNN Only:")
print("   ✓ Strengths: Fast inference, fewer parameters")
print("   ✓ Best for: Spatial pattern recognition in flow features")
print("   ✓ Limitation: No temporal sequence modeling")

print("\n2. LSTM Only:")
print("   ✓ Strengths: Excellent temporal dependency modeling")
print("   ✓ Best for: Sequential attack patterns over time")
print("   ✓ Limitation: More parameters, slower training")

print("\n3. CNN+LSTM Hybrid:")
print("   ✓ Strengths: Combines spatial AND temporal modeling")
print("   ✓ Best for: Complex attacks with both spatial and temporal patterns")
print("   ✓ Limitation: Highest complexity, longest training time")

print("\n💡 RECOMMENDATIONS:")
print("   → For real-time deployment: Consider CNN if speed is critical")
print("   → For maximum accuracy: Use the best-performing model based on rankings")
print("   → For production: Evaluate trade-off between accuracy and computational cost")

print("\n📁 ALL VISUALIZATIONS SAVED TO 'visualizations/' FOLDER:")
print("   - training_history_comparison.png")
print("   - model_performance_comparison.png")
print("   - confusion_matrices.png")
print("   - roc_curves_comparison.png")
print("   - per_class_performance.png")
print("   - model_complexity.png")
print("   - radar_chart_performance.png")
print("   - model_rankings.png")
print("   - model_comparison_results.csv")
print("   - per_class_comparison_results.csv")
print("   - training_history.json")

print("\n" + "="*80)
print("ANALYSIS COMPLETE")
print("="*80)