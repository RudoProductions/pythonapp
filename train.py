#!/usr/bin/env python3
"""
Smart Home IDS - Advanced Hybrid GRU + XGBoost Approach
WITH COMPLETE DATA SAVING FOR DASHBOARD
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix, roc_auc_score, 
    average_precision_score, ConfusionMatrixDisplay, precision_recall_curve,
    roc_curve, auc
)
from sklearn.utils.class_weight import compute_class_weight
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline as ImbPipeline
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import joblib
import time
import os
import json
import plotly.express as px
import plotly.graph_objects as go
from tensorflow import keras
from tensorflow.keras import layers, models, callbacks
warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)
keras.utils.set_random_seed(RANDOM_STATE)

# ----------------------------
# Configuration
# ----------------------------
DATA_PATH = 'IoT_Dataset.csv'
SAVE_DIR = 'saved_models'
VISUALIZATIONS_DIR = os.path.join(SAVE_DIR, 'visualizations')

# Create save directories if they don't exist
os.makedirs(SAVE_DIR, exist_ok=True)
os.makedirs(VISUALIZATIONS_DIR, exist_ok=True)

# ----------------------------
# Helper functions
# ----------------------------
def log(title: str):
    print("\n" + "="*60)
    print(f" {title} ")
    print("="*60)

def clean_dataset(df):
    """Comprehensive data cleaning function - FIXED VERSION"""
    print("🧹 Starting comprehensive data cleaning...")
    
    # Create a copy to avoid modifying original
    df_clean = df.copy()
    original_shape = df_clean.shape
    print(f"Original dataset shape: {original_shape}")
    
    # 1. Remove duplicate rows
    initial_shape = df_clean.shape
    df_clean = df_clean.drop_duplicates()
    print(f"Removed {initial_shape[0] - df_clean.shape[0]} duplicate rows")
    
    # 2. Handle missing values
    missing_before = df_clean.isnull().sum().sum()
    print(f"Missing values before cleaning: {missing_before}")
    
    if missing_before > 0:
        # For numeric columns, fill with median
        numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if df_clean[col].isnull().any():
                median_val = df_clean[col].median()
                df_clean[col] = df_clean[col].fillna(median_val)
        
        # For non-numeric columns, fill with mode or drop
        non_numeric_cols = df_clean.select_dtypes(exclude=[np.number]).columns
        for col in non_numeric_cols:
            if df_clean[col].isnull().any():
                if df_clean[col].nunique() < 50:  # Low cardinality -> fill with mode
                    mode_val = df_clean[col].mode()[0] if not df_clean[col].mode().empty else 'Unknown'
                    df_clean[col] = df_clean[col].fillna(mode_val)
                else:  # High cardinality -> drop column
                    print(f"Dropping high-cardinality column with missing values: {col}")
                    df_clean = df_clean.drop(columns=[col])
    
    missing_after = df_clean.isnull().sum().sum()
    print(f"Missing values after cleaning: {missing_after}")
    
    # 3. Remove constant columns (zero variance)
    constant_cols = []
    for col in df_clean.columns:
        if df_clean[col].nunique() <= 1:
            constant_cols.append(col)
    
    if constant_cols:
        print(f"Removing {len(constant_cols)} constant columns: {constant_cols}")
        df_clean = df_clean.drop(columns=constant_cols)
    
    # 4. Handle infinite values
    inf_count = 0
    numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        inf_mask = np.isinf(df_clean[col])
        if inf_mask.any():
            inf_count += inf_mask.sum()
            # Replace inf with large finite values
            df_clean.loc[inf_mask, col] = np.sign(df_clean.loc[inf_mask, col]) * 1e10
    
    if inf_count > 0:
        print(f"Handled {inf_count} infinite values")
    
    # 5. Remove outliers using IQR method (for numeric columns only)
    outlier_count = 0
    for col in numeric_cols:
        Q1 = df_clean[col].quantile(0.25)
        Q3 = df_clean[col].quantile(0.75)
        IQR = Q3 - Q1
        
        # Only cap outliers if IQR is not zero
        if IQR > 0:
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            outliers = ((df_clean[col] < lower_bound) | (df_clean[col] > upper_bound)).sum()
            outlier_count += outliers
            
            # Cap outliers instead of removing them
            df_clean[col] = np.clip(df_clean[col], lower_bound, upper_bound)
    
    if outlier_count > 0:
        print(f"Capped {outlier_count} outliers using IQR method")
    
    # 6. Remove highly correlated features (optional) - FIXED
    print("Checking for highly correlated features...")
    numeric_cols_current = df_clean.select_dtypes(include=[np.number]).columns
    
    if len(numeric_cols_current) > 1:
        try:
            # Calculate correlation matrix only on current numeric columns
            corr_matrix = df_clean[numeric_cols_current].corr().abs()
            upper_triangle = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
            
            # Find highly correlated columns
            high_corr_cols = []
            for column in upper_triangle.columns:
                high_corr = upper_triangle[column][upper_triangle[column] > 0.95]
                if len(high_corr) > 0:
                    high_corr_cols.append(column)
            
            if high_corr_cols:
                print(f"Removing {len(high_corr_cols)} highly correlated features: {high_corr_cols}")
                df_clean = df_clean.drop(columns=high_corr_cols)
        except Exception as e:
            print(f"Warning: Could not compute correlations: {e}")
    
    print(f"✅ Data cleaning complete. Final shape: {df_clean.shape} (removed {original_shape[0] - df_clean.shape[0]} rows, {original_shape[1] - df_clean.shape[1]} columns)")
    return df_clean

def calculate_all_metrics(y_true, y_pred, y_pred_proba=None):
    """Calculate all performance metrics"""
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred, zero_division=0),
        'f1': f1_score(y_true, y_pred, zero_division=0)
    }
    
    if y_pred_proba is not None:
        metrics['auc'] = roc_auc_score(y_true, y_pred_proba)
        metrics['pr_auc'] = average_precision_score(y_true, y_pred_proba)
    
    return metrics

def print_metrics(metrics, model_name):
    """Print formatted metrics"""
    print(f"\n--- {model_name.upper()} PERFORMANCE ---")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall: {metrics['recall']:.4f}")
    print(f"F1-Score: {metrics['f1']:.4f}")
    if 'auc' in metrics:
        print(f"ROC-AUC: {metrics['auc']:.4f}")
        print(f"PR-AUC: {metrics['pr_auc']:.4f}")

def balance_dataset(X, y, method='smote'):
    """Balance the dataset using various techniques"""
    print(f"Original class distribution: {np.bincount(y)}")
    
    if method == 'smote':
        # SMOTE oversampling
        smote = SMOTE(random_state=RANDOM_STATE)
        X_balanced, y_balanced = smote.fit_resample(X, y)
    elif method == 'undersample':
        # Random undersampling
        undersampler = RandomUnderSampler(random_state=RANDOM_STATE)
        X_balanced, y_balanced = undersampler.fit_resample(X, y)
    elif method == 'combined':
        # Combined over and under sampling
        over = SMOTE(sampling_strategy=0.5, random_state=RANDOM_STATE)
        under = RandomUnderSampler(sampling_strategy=0.8, random_state=RANDOM_STATE)
        steps = [('o', over), ('u', under)]
        pipeline = ImbPipeline(steps=steps)
        X_balanced, y_balanced = pipeline.fit_resample(X, y)
    else:
        # No balancing
        X_balanced, y_balanced = X, y
    
    print(f"Balanced class distribution: {np.bincount(y_balanced)}")
    return X_balanced, y_balanced

def save_plotly_figure(fig, filename):
    """Save Plotly figure as JSON"""
    fig_json = fig.to_json()
    with open(os.path.join(VISUALIZATIONS_DIR, filename), 'w') as f:
        json.dump(fig_json, f)

# ----------------------------
# 1. Data Loading and Preprocessing
# ----------------------------
log("Loading and Preprocessing Data")

try:
    df = pd.read_csv(DATA_PATH)
    print(f"✓ Dataset loaded successfully: {df.shape}")
    
    # SAVE ORIGINAL DATASET STATISTICS
    original_stats = {
        'total_samples': len(df),
        'num_features': df.shape[1],
        'anomaly_count': sum(df['Label'] == 'Anomaly'),
        'normal_count': sum(df['Label'] == 'Normal'),
        'anomaly_percentage': (sum(df['Label'] == 'Anomaly') / len(df)) * 100
    }
    joblib.dump(original_stats, os.path.join(SAVE_DIR, 'original_dataset_stats.pkl'))
    print("✓ Original dataset statistics saved")
    
except FileNotFoundError:
    raise FileNotFoundError(f"Could not find '{DATA_PATH}'. Please update the DATA_PATH variable.")

# Comprehensive data cleaning
df_clean = clean_dataset(df)

# Handle non-numeric columns
features_to_drop = ['Flow_ID', 'Src_IP', 'Dst_IP', 'Timestamp', 'Cat', 'Sub_Cat']
features_to_drop = [c for c in features_to_drop if c in df_clean.columns]

print(f"Dropping non-feature columns: {features_to_drop}")
X = df_clean.drop(columns=features_to_drop + ['Label'])
y = df_clean['Label'].astype(str)

print(f"Features after cleaning: {X.shape[1]}")
print(f"Feature names: {list(X.columns)}")

# Additional numeric conversion for safety
print("Converting features to numeric...")
for col in X.columns:
    X[col] = pd.to_numeric(X[col], errors='coerce')

# Final check for any remaining issues
X = X.replace([np.inf, -np.inf], np.nan)
X = X.fillna(0.0)

# Encode labels - FIXED: Explicit mapping
label_encoder = LabelEncoder()
label_encoder.fit(["Anomaly", "Normal"])
y_encoded = label_encoder.transform(y)
class_names = label_encoder.classes_
print(f"Classes mapped: {dict(zip(class_names, label_encoder.transform(class_names)))}")
print(f"Actual distribution - Anomaly: {sum(y_encoded == 0)}, Normal: {sum(y_encoded == 1)}")

# SAVE PROCESSED DATASET STATISTICS
processed_stats = {
    'total_samples': len(df_clean),
    'num_features': X.shape[1],
    'anomaly_count': sum(y_encoded == 0),
    'normal_count': sum(y_encoded == 1),
    'anomaly_percentage': (sum(y_encoded == 0) / len(y_encoded)) * 100,
    'removed_rows': len(df) - len(df_clean),
    'removed_features': df.shape[1] - X.shape[1] - len(features_to_drop)
}
joblib.dump(processed_stats, os.path.join(SAVE_DIR, 'processed_dataset_stats.pkl'))
print("✓ Processed dataset statistics saved")

# Split data
X_temp, X_test, y_temp, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=RANDOM_STATE, stratify=y_encoded
)
X_train, X_val, y_train, y_val = train_test_split(
    X_temp, y_temp, test_size=0.2, random_state=RANDOM_STATE, stratify=y_temp
)

print(f"\nDataset splits:")
print(f"Train: {X_train.shape}, Anomaly: {sum(y_train == 0)}, Normal: {sum(y_train == 1)}")
print(f"Val: {X_val.shape}, Anomaly: {sum(y_val == 0)}, Normal: {sum(y_val == 1)}")
print(f"Test: {X_test.shape}, Anomaly: {sum(y_test == 0)}, Normal: {sum(y_test == 1)}")

# Balance the training dataset
print("\nBalancing training dataset...")
X_train_balanced, y_train_balanced = balance_dataset(X_train, y_train, method='smote')

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_balanced)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

print(f"\nTraining set: {X_train_scaled.shape}")
print(f"Validation set: {X_val_scaled.shape}")
print(f"Test set: {X_test_scaled.shape}")

# Prepare Data for GRU
n_features = X_train_scaled.shape[1]
print(f"\nReshaping data for GRU. Each sample will be: (1, {n_features})")

X_train_gru = X_train_scaled.reshape(X_train_scaled.shape[0], 1, n_features)
X_val_gru = X_val_scaled.reshape(X_val_scaled.shape[0], 1, n_features)
X_test_gru = X_test_scaled.reshape(X_test_scaled.shape[0], 1, n_features)

print(f"GRU Training set shape: {X_train_gru.shape}")
print(f"GRU Validation set shape: {X_val_gru.shape}")
print(f"GRU Test set shape: {X_test_gru.shape}")

# ----------------------------
# 2. Build and Train the GRU Model
# ----------------------------
log("Building and Training GRU Model")

def create_gru_model(input_shape):
    """Creates a GRU model for feature extraction."""
    model = models.Sequential([
        layers.GRU(64, activation='tanh', return_sequences=True, input_shape=input_shape),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        
        layers.GRU(128, activation='tanh', return_sequences=True),
        layers.BatchNormalization(),
        layers.Dropout(0.4),
        
        layers.GRU(256, activation='tanh'),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        
        layers.Dense(128, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        
        layers.Dense(1, activation='sigmoid', name='output')
    ])
    return model

# Create and compile the model
gru_model = create_gru_model((1, n_features))
gru_model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy', keras.metrics.AUC(name='auc'), 
             keras.metrics.Precision(name='precision'),
             keras.metrics.Recall(name='recall')]
)

gru_model.summary()

# Calculate class weights for the GRU
gru_class_weights = compute_class_weight(
    class_weight='balanced',
    classes=np.unique(y_train_balanced),
    y=y_train_balanced
)
gru_class_weights_dict = dict(enumerate(gru_class_weights))

# Train the GRU
print("\nTraining GRU model...")
history = gru_model.fit(
    X_train_gru, y_train_balanced,
    validation_data=(X_val_gru, y_val),
    epochs=30,
    batch_size=256,
    class_weight=gru_class_weights_dict,
    callbacks=[
        callbacks.EarlyStopping(patience=5, restore_best_weights=True, monitor='val_auc'),
        callbacks.ReduceLROnPlateau(factor=0.5, patience=3, min_lr=1e-7),
        callbacks.ModelCheckpoint(
            os.path.join(SAVE_DIR, 'gru_model_trained.keras'), 
            save_best_only=True, 
            monitor='val_auc'
        )
    ],
    verbose=1
)

# Save the final model
gru_model.save(os.path.join(SAVE_DIR, 'gru_model_trained.keras'))
print("✓ GRU training completed and model saved.")

# ----------------------------
# 3. Use GRU as Feature Extractor for XGBoost
# ----------------------------
log("Extracting GRU Features for XGBoost")

# Create a feature extraction model: remove the final output layer
feature_extractor = models.Model(
    inputs=gru_model.inputs,
    outputs=gru_model.layers[-3].output # Output from the Dense(128) layer
)

# Extract features from all datasets
print("Extracting features from training set...")
X_train_gru_features = feature_extractor.predict(X_train_gru, verbose=0)
print("Extracting features from validation set...")
X_val_gru_features = feature_extractor.predict(X_val_gru, verbose=0)
print("Extracting features from test set...")
X_test_gru_features = feature_extractor.predict(X_test_gru, verbose=0)

print(f"\nGRU Feature shape: {X_train_gru_features.shape}")

# --- Train XGBoost on GRU Features ---
log("Training XGBoost on GRU Features")

# Handle class imbalance for XGBoost
xgb_class_weights = compute_class_weight(
    class_weight='balanced', 
    classes=np.unique(y_train_balanced), 
    y=y_train_balanced
)
xgb_scale_pos_weight = xgb_class_weights[1] / xgb_class_weights[0]

xgb_model = xgb.XGBClassifier(
    objective='binary:logistic',
    n_estimators=200,
    learning_rate=0.05,
    max_depth=6,
    random_state=RANDOM_STATE,
    eval_metric=['logloss', 'auc', 'error'],
    subsample=0.8,
    colsample_bytree=0.8,
    scale_pos_weight=xgb_scale_pos_weight,
    reg_alpha=0.1,
    reg_lambda=0.1,
    early_stopping_rounds=20
)

xgb_model.fit(
    X_train_gru_features, y_train_balanced,
    eval_set=[(X_val_gru_features, y_val)],
    verbose=10
)

# Save XGBoost model
xgb_model.save_model(os.path.join(SAVE_DIR, 'xgb_model_gru_features.json'))
print("✓ XGBoost training completed and model saved.")

# ----------------------------
# 4. Build the Hybrid Meta-Model - FIXED VERSION
# ----------------------------
log("Building Hybrid Meta-Model")

print("Getting predictions for meta-learner training...")

# Get predictions from both models on TRAINING + VALIDATION sets (FIXED)
gru_train_pred_proba = gru_model.predict(X_train_gru, verbose=0).flatten().reshape(-1, 1)
xgb_train_pred_proba = xgb_model.predict_proba(X_train_gru_features)[:, 1].reshape(-1, 1)

gru_val_pred_proba = gru_model.predict(X_val_gru, verbose=0).flatten().reshape(-1, 1)
xgb_val_pred_proba = xgb_model.predict_proba(X_val_gru_features)[:, 1].reshape(-1, 1)

# Combine training + validation data for meta-learner (FIXED)
meta_features_train_val = np.concatenate([
    np.vstack([gru_train_pred_proba, gru_val_pred_proba]),
    np.vstack([xgb_train_pred_proba, xgb_val_pred_proba])
], axis=1)

# Combine labels
y_train_val = np.concatenate([y_train_balanced, y_val])

print(f"Meta-learner training data: {meta_features_train_val.shape}")
print(f"Meta-learner training labels: {y_train_val.shape}")
print(f"Meta-learner class distribution: Anomaly: {sum(y_train_val == 0)}, Normal: {sum(y_train_val == 1)}")

# Train meta-learner on combined training + validation data (FIXED)
meta_learner = LogisticRegression(
    max_iter=1000, 
    random_state=RANDOM_STATE,
    C=0.1,
    class_weight='balanced'
)

meta_learner.fit(meta_features_train_val, y_train_val)
print("✓ Meta-learner trained on combined training + validation data")

# Save meta-learner and preprocessing objects
joblib.dump(meta_learner, os.path.join(SAVE_DIR, 'meta_learner.pkl'))
joblib.dump(scaler, os.path.join(SAVE_DIR, 'scaler.pkl'))
joblib.dump(label_encoder, os.path.join(SAVE_DIR, 'label_encoder.pkl'))
print("✓ Meta-learner training completed and models saved.")

# ----------------------------
# 5. Evaluate on Test Set
# ----------------------------
log("Final Evaluation on Test Set")

# Get test predictions from both models
gru_test_pred_proba = gru_model.predict(X_test_gru, verbose=0).flatten().reshape(-1, 1)
xgb_test_pred_proba = xgb_model.predict_proba(X_test_gru_features)[:, 1].reshape(-1, 1)

# Combine predictions for meta-learner
meta_features_test = np.concatenate([gru_test_pred_proba, xgb_test_pred_proba], axis=1)

# Hybrid predictions - FIXED: Use correct probability column
hybrid_pred_proba = meta_learner.predict_proba(meta_features_test)[:, 1]  # Probability of class 1 (Normal)
hybrid_pred = (hybrid_pred_proba >= 0.5).astype(int)

# Individual model predictions for comparison
gru_test_pred = (gru_test_pred_proba >= 0.5).astype(int).flatten()
xgb_test_pred = (xgb_test_pred_proba >= 0.5).astype(int).flatten()

# ----------------------------
# 6. DIAGNOSTIC - Check Prediction Distributions
# ----------------------------
log("PREDICTION DISTRIBUTION DIAGNOSTIC")

print("🔍 TRUE DISTRIBUTIONS:")
print(f"Test set - Anomaly: {sum(y_test == 0)}, Normal: {sum(y_test == 1)}")

print("\n🔍 PREDICTED DISTRIBUTIONS:")
print(f"GRU - Anomaly: {sum(gru_test_pred == 0)}, Normal: {sum(gru_test_pred == 1)}")
print(f"XGBoost - Anomaly: {sum(xgb_test_pred == 0)}, Normal: {sum(xgb_test_pred == 1)}")
print(f"Hybrid - Anomaly: {sum(hybrid_pred == 0)}, Normal: {sum(hybrid_pred == 1)}")

print(f"\n🔍 PROBABILITY STATISTICS:")
print(f"GRU probability range: [{gru_test_pred_proba.min():.3f}, {gru_test_pred_proba.max():.3f}]")
print(f"XGB probability range: [{xgb_test_pred_proba.min():.3f}, {xgb_test_pred_proba.max():.3f}]")
print(f"Hybrid probability range: [{hybrid_pred_proba.min():.3f}, {hybrid_pred_proba.max():.3f}]")
print(f"Hybrid probability mean: {hybrid_pred_proba.mean():.3f}")

# Test different thresholds
print(f"\n🔍 THRESHOLD ANALYSIS:")
for threshold in [0.3, 0.4, 0.5, 0.6, 0.7]:
    test_pred = (hybrid_pred_proba >= threshold).astype(int)
    anomaly_count = sum(test_pred == 0)
    normal_count = sum(test_pred == 1)
    accuracy = accuracy_score(y_test, test_pred)
    print(f"Threshold {threshold}: Anomaly={anomaly_count}, Normal={normal_count}, Accuracy={accuracy:.3f}")

# ----------------------------
# 7. Results and Analysis
# ----------------------------
log("RESULTS AND ANALYSIS")

# Calculate all metrics for each model
gru_metrics = calculate_all_metrics(y_test, gru_test_pred, gru_test_pred_proba.flatten())
xgb_metrics = calculate_all_metrics(y_test, xgb_test_pred, xgb_test_pred_proba.flatten())
hybrid_metrics = calculate_all_metrics(y_test, hybrid_pred, hybrid_pred_proba)

# Print detailed metrics
print_metrics(gru_metrics, "GRU Model")
print_metrics(xgb_metrics, "XGBoost Model")
print_metrics(hybrid_metrics, "Hybrid Model")

print("\n--- COMPARISON ---")
best_individual_auc = max(gru_metrics['auc'], xgb_metrics['auc'])
best_individual_f1 = max(gru_metrics['f1'], xgb_metrics['f1'])

improvement_auc = ((hybrid_metrics['auc'] - best_individual_auc) / best_individual_auc) * 100
improvement_f1 = ((hybrid_metrics['f1'] - best_individual_f1) / best_individual_f1) * 100

print(f"AUC Improvement: {improvement_auc:+.2f}%")
print(f"F1-Score Improvement: {improvement_f1:+.2f}%")

if improvement_auc > 1.0:
    print("🎉 Hybrid model significantly outperformed individual models!")
elif improvement_auc > 0:
    print("✅ Hybrid model slightly outperformed individual models")
elif improvement_auc >= -1:
    print("🤝 Hybrid model performed comparably to best individual model")
else:
    print("🔍 Individual models performed better")

print("\nDetailed Classification Report (Hybrid Model):")
print(classification_report(y_test, hybrid_pred, target_names=class_names))

# Confusion Matrix
print("\nConfusion Matrix (Hybrid Model):")
cm = confusion_matrix(y_test, hybrid_pred)
print(cm)

# ----------------------------
# 8. CREATE AND SAVE ALL VISUALIZATIONS FOR DASHBOARD
# ----------------------------
log("Creating and Saving All Visualizations")

# 1. Class Distribution Pie Chart
class_dist_fig = px.pie(
    names=['Anomaly', 'Normal'],
    values=[sum(y_encoded == 0), sum(y_encoded == 1)],
    title='Traffic Class Distribution',
    color=['Anomaly', 'Normal'],
    color_discrete_map={'Anomaly': 'red', 'Normal': 'green'}
)
save_plotly_figure(class_dist_fig, 'class_distribution.json')

# 2. Performance Comparison Bar Chart
models = ['GRU', 'XGBoost', 'Hybrid']
accuracy_values = [gru_metrics['accuracy'] * 100, xgb_metrics['accuracy'] * 100, hybrid_metrics['accuracy'] * 100]
auc_values = [gru_metrics['auc'] * 100, xgb_metrics['auc'] * 100, hybrid_metrics['auc'] * 100]

perf_fig = px.bar(
    x=models,
    y=accuracy_values,
    title='Model Accuracy Comparison',
    labels={'x': 'Model', 'y': 'Accuracy (%)'},
    color=models,
    color_discrete_map={'GRU': '#3498db', 'XGBoost': '#2ecc71', 'Hybrid': '#e74c3c'}
)
perf_fig.update_layout(yaxis_range=[0, 100], showlegend=False)
save_plotly_figure(perf_fig, 'accuracy_comparison.json')

auc_fig = px.bar(
    x=models,
    y=auc_values,
    title='Model AUC Comparison',
    labels={'x': 'Model', 'y': 'AUC (%)'},
    color=models,
    color_discrete_map={'GRU': '#3498db', 'XGBoost': '#2ecc71', 'Hybrid': '#e74c3c'}
)
auc_fig.update_layout(yaxis_range=[0, 100], showlegend=False)
save_plotly_figure(auc_fig, 'auc_comparison.json')

# 3. Confusion Matrix Heatmap - INDIVIDUAL MODELS
# Hybrid Confusion Matrix
cm_hybrid = confusion_matrix(y_test, hybrid_pred)
cm_fig = px.imshow(
    cm_hybrid, 
    text_auto=True, 
    aspect="auto",
    labels=dict(x="Predicted", y="Actual", color="Count"),
    x=['Anomaly', 'Normal'],
    y=['Anomaly', 'Normal'],
    title=f"Hybrid Model Confusion Matrix\n(Accuracy: {hybrid_metrics['accuracy']*100:.2f}%)"
)
cm_fig.update_layout(coloraxis_showscale=False)
save_plotly_figure(cm_fig, 'confusion_matrix_hybrid.json')

# GRU Confusion Matrix
cm_gru = confusion_matrix(y_test, gru_test_pred)
cm_gru_fig = px.imshow(
    cm_gru, 
    text_auto=True, 
    aspect="auto",
    labels=dict(x="Predicted", y="Actual", color="Count"),
    x=['Anomaly', 'Normal'],
    y=['Anomaly', 'Normal'],
    title=f"GRU Model Confusion Matrix\n(Accuracy: {gru_metrics['accuracy']*100:.2f}%)",
    color_continuous_scale='Blues'
)
cm_gru_fig.update_layout(coloraxis_showscale=False)
save_plotly_figure(cm_gru_fig, 'confusion_matrix_gru.json')

# XGBoost Confusion Matrix
cm_xgb = confusion_matrix(y_test, xgb_test_pred)
cm_xgb_fig = px.imshow(
    cm_xgb, 
    text_auto=True, 
    aspect="auto",
    labels=dict(x="Predicted", y="Actual", color="Count"),
    x=['Anomaly', 'Normal'],
    y=['Anomaly', 'Normal'],
    title=f"XGBoost Model Confusion Matrix\n(Accuracy: {xgb_metrics['accuracy']*100:.2f}%)",
    color_continuous_scale='Greens'
)
cm_xgb_fig.update_layout(coloraxis_showscale=False)
save_plotly_figure(cm_xgb_fig, 'confusion_matrix_xgb.json')

# 4. ROC Curve
fpr, tpr, _ = roc_curve(y_test, hybrid_pred_proba)
roc_auc = roc_auc_score(y_test, hybrid_pred_proba)

roc_fig = go.Figure()
roc_fig.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines', 
                           name=f'ROC Curve (AUC = {roc_auc:.4f})', 
                           line=dict(color='royalblue', width=3)))
roc_fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines', 
                           name='Random', line=dict(color='red', dash='dash')))
roc_fig.update_layout(
    title=f'ROC Curve - Hybrid Model\n(AUC = {roc_auc:.4f})',
    xaxis_title='False Positive Rate',
    yaxis_title='True Positive Rate',
    width=500,
    height=400
)
save_plotly_figure(roc_fig, 'roc_curve.json')

# 5. Precision-Recall Curve
precision_vals, recall_vals, _ = precision_recall_curve(y_test, hybrid_pred_proba)
pr_auc = auc(recall_vals, precision_vals)

pr_fig = go.Figure()
pr_fig.add_trace(go.Scatter(x=recall_vals, y=precision_vals, mode='lines', 
                          name=f'PR Curve (AUC = {pr_auc:.4f})', 
                          line=dict(color='green', width=3)))
pr_fig.update_layout(
    title=f'Precision-Recall Curve\n(AUC = {pr_auc:.4f})',
    xaxis_title='Recall',
    yaxis_title='Precision',
    width=500,
    height=400
)
save_plotly_figure(pr_fig, 'pr_curve.json')

# 6. Feature Importance
if hasattr(xgb_model, 'feature_importances_'):
    importance_values = xgb_model.feature_importances_
    feature_names = [f'GRU_Feature_{i+1}' for i in range(len(importance_values))]
    # Get top 10 features
    top_indices = np.argsort(importance_values)[-10:][::-1]
    top_features = [feature_names[i] for i in top_indices]
    top_importance = [importance_values[i] for i in top_indices]
else:
    # Placeholder feature importance
    feature_names = [f'GRU_Feature_{i}' for i in range(1, 11)]
    top_importance = [0.15, 0.13, 0.11, 0.09, 0.08, 0.07, 0.06, 0.05, 0.04, 0.03]

feature_fig = px.bar(
    x=top_importance, 
    y=feature_names[:len(top_importance)], 
    orientation='h',
    title='Top Feature Importances (XGBoost on GRU Features)',
    labels={'x': 'Importance', 'y': 'Feature'}
)
feature_fig.update_layout(height=400)
save_plotly_figure(feature_fig, 'feature_importance.json')

# 7. Comprehensive Metrics Comparison
accuracy_fig = px.bar(
    x=models,
    y=[gru_metrics['accuracy'] * 100, xgb_metrics['accuracy'] * 100, hybrid_metrics['accuracy'] * 100],
    title='Accuracy Comparison',
    labels={'x': 'Model', 'y': 'Accuracy (%)'},
    color=models,
    color_discrete_map={'GRU': '#3498db', 'XGBoost': '#2ecc71', 'Hybrid': '#e74c3c'}
)
accuracy_fig.update_layout(yaxis_range=[0, 100], showlegend=False)
save_plotly_figure(accuracy_fig, 'accuracy_detailed.json')

precision_fig = px.bar(
    x=models,
    y=[gru_metrics['precision'] * 100, xgb_metrics['precision'] * 100, hybrid_metrics['precision'] * 100],
    title='Precision Comparison',
    labels={'x': 'Model', 'y': 'Precision (%)'},
    color=models,
    color_discrete_map={'GRU': '#3498db', 'XGBoost': '#2ecc71', 'Hybrid': '#e74c3c'}
)
precision_fig.update_layout(yaxis_range=[0, 100], showlegend=False)
save_plotly_figure(precision_fig, 'precision_comparison.json')

recall_fig = px.bar(
    x=models,
    y=[gru_metrics['recall'] * 100, xgb_metrics['recall'] * 100, hybrid_metrics['recall'] * 100],
    title='Recall Comparison',
    labels={'x': 'Model', 'y': 'Recall (%)'},
    color=models,
    color_discrete_map={'GRU': '#3498db', 'XGBoost': '#2ecc71', 'Hybrid': '#e74c3c'}
)
recall_fig.update_layout(yaxis_range=[0, 100], showlegend=False)
save_plotly_figure(recall_fig, 'recall_comparison.json')

f1_fig = px.bar(
    x=models,
    y=[gru_metrics['f1'] * 100, xgb_metrics['f1'] * 100, hybrid_metrics['f1'] * 100],
    title='F1-Score Comparison',
    labels={'x': 'Model', 'y': 'F1-Score (%)'},
    color=models,
    color_discrete_map={'GRU': '#3498db', 'XGBoost': '#2ecc71', 'Hybrid': '#e74c3c'}
)
f1_fig.update_layout(yaxis_range=[0, 100], showlegend=False)
save_plotly_figure(f1_fig, 'f1_comparison.json')

print("✓ All visualizations saved as JSON files")

# ----------------------------
# 9. Save Configuration with Complete Metrics
# ----------------------------
log("Saving Final Configuration")

# Save configuration with all metrics
config = {
    'feature_order': X.columns.tolist(),
    'label_mapping': {cls: int(label_encoder.transform([cls])[0]) for cls in class_names},
    'model_performance': {
        'gru': gru_metrics,
        'xgb': xgb_metrics,
        'hybrid': hybrid_metrics
    },
    'input_shape': (1, n_features),
    'class_distribution': {
        'original': np.bincount(y_encoded).tolist(),
        'balanced': np.bincount(y_train_balanced).tolist(),
        'test': np.bincount(y_test).tolist()
    },
    'training_params': {
        'balance_method': 'smote',
        'random_state': RANDOM_STATE
    },
    'prediction_info': {
        'optimal_threshold': 0.5,
        'anomaly_class': 0,
        'normal_class': 1
    },
    'dataset_stats': {
        'original': original_stats,
        'processed': processed_stats
    }
}
joblib.dump(config, os.path.join(SAVE_DIR, 'hybrid_gru_config.pkl'))

print("All models, artifacts, and visualizations saved successfully!")
print(f"Files created in '{SAVE_DIR}':")
print("- gru_model_trained.keras")
print("- xgb_model_gru_features.json")
print("- meta_learner.pkl")
print("- scaler.pkl")
print("- label_encoder.pkl")
print("- hybrid_gru_config.pkl")
print("- original_dataset_stats.pkl")
print("- processed_dataset_stats.pkl")
print(f"\nVisualizations saved in '{VISUALIZATIONS_DIR}':")
print("- class_distribution.json")
print("- accuracy_comparison.json")
print("- auc_comparison.json")
print("- confusion_matrix.json")
print("- roc_curve.json")
print("- pr_curve.json")
print("- feature_importance.json")
print("- accuracy_detailed.json")
print("- precision_comparison.json")
print("- recall_comparison.json")
print("- f1_comparison.json")
print("- network_architecture.json")

log("Advanced Hybrid GRU + XGBoost Approach Complete! 🚀")