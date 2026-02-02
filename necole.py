# -*- coding: utf-8 -*-
"""Enhanced Autoencoder + XGBoost with Visualization Saving"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
import os
import joblib
from datetime import datetime

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_auc_score, precision_recall_fscore_support,
    accuracy_score, roc_curve, precision_recall_curve
)

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from xgboost import XGBClassifier
import plotly.graph_objects as go
import plotly.express as px

class AEXGBVisualization:
    def __init__(self, output_dir='ae_xgb_visualizations'):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
    def save_plotly_figure(self, fig, filename):
        """Save Plotly figure as JSON and PNG"""
        # Save as JSON
        fig_json = fig.to_json()
        with open(os.path.join(self.output_dir, f'{filename}.json'), 'w') as f:
            json.dump(fig_json, f)
        
        # Save as PNG
        fig.write_image(os.path.join(self.output_dir, f'{filename}.png'))
        
    def save_matplotlib_figure(self, filename, dpi=150):
        """Save matplotlib figure"""
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, f'{filename}.png'), 
                   dpi=dpi, bbox_inches='tight')
        plt.close()

def train_ae_xgboost_ensemble():
    """Train AE + XGBoost ensemble and save all visualizations"""
    
    # Initialize visualization saver
    viz = AEXGBVisualization()
    
    # Load and preprocess data
    df = pd.read_csv("IoT_Dataset.csv")
    
    # Data preprocessing
    drop_cols = ["Flow_ID", "Src_IP", "Dst_IP", "Timestamp", "Cat", "Sub_Cat"]
    df = df.drop(columns=drop_cols, errors="ignore")
    
    X = df.drop(columns=["Label"])
    y = df["Label"]
    
    # Encode labels: Normal=1, Anomaly=0
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )
    
    # Clean and scale data
    mask_train = np.all(np.isfinite(X_train), axis=1)
    X_train, y_train = X_train[mask_train], y_train[mask_train]
    mask_test = np.all(np.isfinite(X_test), axis=1)
    X_test, y_test = X_test[mask_test], y_test[mask_test]
    
    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # =========================================================================
    # 1. Autoencoder Training
    # =========================================================================
    normal_idx = (y_train == 1)
    X_train_normal = X_train_scaled[normal_idx]
    input_dim = X_train_normal.shape[1]
    
    autoencoder = keras.Sequential([
        layers.Input(shape=(input_dim,)),
        layers.Dense(32, activation="relu"),
        layers.Dense(16, activation="relu"),
        layers.Dense(32, activation="relu"),
        layers.Dense(input_dim, activation="sigmoid")
    ])
    
    autoencoder.compile(optimizer="adam", loss="mse")
    
    history = autoencoder.fit(
        X_train_normal, X_train_normal,
        epochs=10, batch_size=256,
        validation_split=0.2,
        verbose=1
    )
    
    # Save training history plot
    plt.figure(figsize=(10, 6))
    plt.plot(history.history["loss"], label="Train Loss", linewidth=2)
    plt.plot(history.history["val_loss"], label="Val Loss", linewidth=2)
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Autoencoder Training History")
    plt.legend()
    plt.grid(True, alpha=0.3)
    viz.save_matplotlib_figure('ae_training_history')
    
    # =========================================================================
    # 2. Autoencoder Only Evaluation
    # =========================================================================
    X_test_recon = autoencoder.predict(X_test_scaled)
    test_recon_error = np.mean(np.square(X_test_scaled - X_test_recon), axis=1)
    
    X_train_recon = autoencoder.predict(X_train_normal)
    train_recon_error = np.mean(np.square(X_train_normal - X_train_recon), axis=1)
    threshold = np.percentile(train_recon_error, 95)
    
    y_pred_ae = (test_recon_error < threshold).astype(int)
    
    # AE Only Metrics
    ae_accuracy = accuracy_score(y_test, y_pred_ae)
    ae_precision, ae_recall, ae_f1, _ = precision_recall_fscore_support(y_test, y_pred_ae, average="weighted")
    ae_auc = roc_auc_score(y_test, y_pred_ae)
    
    # AE Confusion Matrix
    cm_ae = confusion_matrix(y_test, y_pred_ae)
    fig = px.imshow(cm_ae, text_auto=True, aspect="auto",
                   labels=dict(x="Predicted", y="Actual", color="Count"),
                   x=['Normal', 'Anomaly'], y=['Normal', 'Anomaly'],
                   title=f'Autoencoder Only - Confusion Matrix<br>Accuracy: {ae_accuracy:.3f}')
    fig.update_layout(coloraxis_showscale=False)
    viz.save_plotly_figure(fig, 'confusion_matrix_ae')
    
    # AE ROC Curve
    fpr_ae, tpr_ae, _ = roc_curve(y_test, y_pred_ae)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=fpr_ae, y=tpr_ae, mode='lines', 
                            name=f'AE Only (AUC = {ae_auc:.3f})', line=dict(width=3)))
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines', 
                            name='Random', line=dict(dash='dash', color='gray')))
    fig.update_layout(title='ROC Curve - Autoencoder Only',
                     xaxis_title='False Positive Rate',
                     yaxis_title='True Positive Rate',
                     height=500)
    viz.save_plotly_figure(fig, 'roc_curve_ae')
    
    # =========================================================================
    # 3. XGBoost Only Evaluation
    # =========================================================================
    xgb = XGBClassifier(n_estimators=100, max_depth=5, learning_rate=0.1, 
                       use_label_encoder=False, eval_metric="logloss")
    xgb.fit(X_train_scaled, y_train)
    y_pred_xgb = xgb.predict(X_test_scaled)
    y_pred_xgb_proba = xgb.predict_proba(X_test_scaled)[:, 1]
    
    # XGBoost Metrics
    xgb_accuracy = accuracy_score(y_test, y_pred_xgb)
    xgb_precision, xgb_recall, xgb_f1, _ = precision_recall_fscore_support(y_test, y_pred_xgb, average="weighted")
    xgb_auc = roc_auc_score(y_test, y_pred_xgb_proba)
    
    # XGBoost Confusion Matrix
    cm_xgb = confusion_matrix(y_test, y_pred_xgb)
    fig = px.imshow(cm_xgb, text_auto=True, aspect="auto",
                   labels=dict(x="Predicted", y="Actual", color="Count"),
                   x=['Normal', 'Anomaly'], y=['Normal', 'Anomaly'],
                   title=f'XGBoost Only - Confusion Matrix<br>Accuracy: {xgb_accuracy:.3f}')
    fig.update_layout(coloraxis_showscale=False)
    viz.save_plotly_figure(fig, 'confusion_matrix_xgb')
    
    # XGBoost ROC Curve
    fpr_xgb, tpr_xgb, _ = roc_curve(y_test, y_pred_xgb_proba)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=fpr_xgb, y=tpr_xgb, mode='lines', 
                            name=f'XGBoost (AUC = {xgb_auc:.3f})', 
                            line=dict(width=3, color='green')))
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines', 
                            name='Random', line=dict(dash='dash', color='gray')))
    fig.update_layout(title='ROC Curve - XGBoost Only',
                     xaxis_title='False Positive Rate',
                     yaxis_title='True Positive Rate',
                     height=500)
    viz.save_plotly_figure(fig, 'roc_curve_xgb')
    
    # XGBoost Feature Importance
    feature_importance = xgb.feature_importances_
    feature_names = [f'Feature_{i}' for i in range(len(feature_importance))]
    
    # Get top 20 features
    indices = np.argsort(feature_importance)[-20:]
    fig = px.bar(x=feature_importance[indices], y=[feature_names[i] for i in indices],
                orientation='h', title='XGBoost Feature Importance (Top 20)',
                labels={'x': 'Importance', 'y': 'Features'})
    fig.update_layout(height=600)
    viz.save_plotly_figure(fig, 'feature_importance_xgb')
    
    # =========================================================================
    # 4. Hybrid AE + XGBoost Evaluation
    # =========================================================================
    X_train_re = np.mean(np.square(X_train_scaled - autoencoder.predict(X_train_scaled)), axis=1)
    X_test_re  = np.mean(np.square(X_test_scaled - autoencoder.predict(X_test_scaled)), axis=1)
    
    X_train_hybrid = np.hstack([X_train_scaled, X_train_re.reshape(-1,1)])
    X_test_hybrid  = np.hstack([X_test_scaled, X_test_re.reshape(-1,1)])
    
    xgb_hybrid = XGBClassifier(n_estimators=100, max_depth=5, learning_rate=0.1, 
                              use_label_encoder=False, eval_metric="logloss")
    xgb_hybrid.fit(X_train_hybrid, y_train)
    y_pred_hybrid = xgb_hybrid.predict(X_test_hybrid)
    y_pred_hybrid_proba = xgb_hybrid.predict_proba(X_test_hybrid)[:, 1]
    
    # Hybrid Metrics
    hybrid_accuracy = accuracy_score(y_test, y_pred_hybrid)
    hybrid_precision, hybrid_recall, hybrid_f1, _ = precision_recall_fscore_support(y_test, y_pred_hybrid, average="weighted")
    hybrid_auc = roc_auc_score(y_test, y_pred_hybrid_proba)
    
    # Hybrid Confusion Matrix
    cm_hybrid = confusion_matrix(y_test, y_pred_hybrid)
    fig = px.imshow(cm_hybrid, text_auto=True, aspect="auto",
                   labels=dict(x="Predicted", y="Actual", color="Count"),
                   x=['Normal', 'Anomaly'], y=['Normal', 'Anomaly'],
                   title=f'Hybrid AE+XGBoost - Confusion Matrix<br>Accuracy: {hybrid_accuracy:.3f}')
    fig.update_layout(coloraxis_showscale=False)
    viz.save_plotly_figure(fig, 'confusion_matrix_hybrid')
    
    # Hybrid ROC Curve
    fpr_hybrid, tpr_hybrid, _ = roc_curve(y_test, y_pred_hybrid_proba)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=fpr_hybrid, y=tpr_hybrid, mode='lines', 
                            name=f'Hybrid AE+XGBoost (AUC = {hybrid_auc:.3f})', 
                            line=dict(width=3, color='orange')))
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines', 
                            name='Random', line=dict(dash='dash', color='gray')))
    fig.update_layout(title='ROC Curve - Hybrid AE+XGBoost',
                     xaxis_title='False Positive Rate',
                     yaxis_title='True Positive Rate',
                     height=500)
    viz.save_plotly_figure(fig, 'roc_curve_hybrid')
    
    # =========================================================================
    # 5. Comparison Visualizations
    # =========================================================================
    
    # ROC Curves Comparison
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=fpr_ae, y=tpr_ae, mode='lines', 
                            name=f'AE Only (AUC = {ae_auc:.3f})', line=dict(width=3)))
    fig.add_trace(go.Scatter(x=fpr_xgb, y=tpr_xgb, mode='lines', 
                            name=f'XGBoost Only (AUC = {xgb_auc:.3f})', 
                            line=dict(width=3, color='green')))
    fig.add_trace(go.Scatter(x=fpr_hybrid, y=tpr_hybrid, mode='lines', 
                            name=f'Hybrid AE+XGBoost (AUC = {hybrid_auc:.3f})', 
                            line=dict(width=3, color='orange')))
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines', 
                            name='Random', line=dict(dash='dash', color='gray')))
    fig.update_layout(title='ROC Curves Comparison',
                     xaxis_title='False Positive Rate',
                     yaxis_title='True Positive Rate',
                     height=500)
    viz.save_plotly_figure(fig, 'roc_curves_comparison')
    
    # Metrics Comparison Bar Chart
    models = ['Autoencoder Only', 'XGBoost Only', 'Hybrid AE+XGBoost']
    accuracy_scores = [ae_accuracy, xgb_accuracy, hybrid_accuracy]
    precision_scores = [ae_precision, xgb_precision, hybrid_precision]
    recall_scores = [ae_recall, xgb_recall, hybrid_recall]
    f1_scores = [ae_f1, xgb_f1, hybrid_f1]
    auc_scores = [ae_auc, xgb_auc, hybrid_auc]
    
    # Create metrics comparison
    metrics_data = []
    for i, model in enumerate(models):
        metrics_data.extend([
            {'Model': model, 'Metric': 'Accuracy', 'Value': accuracy_scores[i]},
            {'Model': model, 'Metric': 'Precision', 'Value': precision_scores[i]},
            {'Model': model, 'Metric': 'Recall', 'Value': recall_scores[i]},
            {'Model': model, 'Metric': 'F1-Score', 'Value': f1_scores[i]},
            {'Model': model, 'Metric': 'AUC', 'Value': auc_scores[i]}
        ])
    
    metrics_df = pd.DataFrame(metrics_data)
    
    fig = px.bar(metrics_df, x='Metric', y='Value', color='Model', barmode='group',
                title='Model Performance Comparison',
                labels={'Value': 'Score', 'Metric': 'Performance Metric'})
    fig.update_layout(height=500)
    viz.save_plotly_figure(fig, 'metrics_comparison')
    
    # Radar Chart Comparison
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'AUC']
    
    fig = go.Figure()
    
    # AE Only
    ae_values = [ae_accuracy, ae_precision, ae_recall, ae_f1, ae_auc]
    ae_values += ae_values[:1]  # Close the radar
    angles = [n / float(len(metrics)) * 2 * np.pi for n in range(len(metrics))]
    angles += angles[:1]
    
    fig.add_trace(go.Scatterpolar(
        r=ae_values,
        theta=metrics + [metrics[0]],
        fill='toself',
        name='AE Only',
        line=dict(color='blue', width=2)
    ))
    
    # XGBoost Only
    xgb_values = [xgb_accuracy, xgb_precision, xgb_recall, xgb_f1, xgb_auc]
    xgb_values += xgb_values[:1]
    fig.add_trace(go.Scatterpolar(
        r=xgb_values,
        theta=metrics + [metrics[0]],
        fill='toself',
        name='XGBoost Only',
        line=dict(color='green', width=2)
    ))
    
    # Hybrid
    hybrid_values = [hybrid_accuracy, hybrid_precision, hybrid_recall, hybrid_f1, hybrid_auc]
    hybrid_values += hybrid_values[:1]
    fig.add_trace(go.Scatterpolar(
        r=hybrid_values,
        theta=metrics + [metrics[0]],
        fill='toself',
        name='Hybrid AE+XGBoost',
        line=dict(color='orange', width=2)
    ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=[0.8, 1.0])
        ),
        showlegend=True,
        title='Model Comparison - Radar Chart',
        height=500
    )
    viz.save_plotly_figure(fig, 'radar_chart_comparison')
    
    # =========================================================================
    # 6. Save Model Metrics for Dashboard
    # =========================================================================
    model_metrics = {
        'ae_only': {
            'accuracy': ae_accuracy * 100,
            'precision': ae_precision * 100,
            'recall': ae_recall * 100,
            'f1': ae_f1 * 100,
            'auc': ae_auc
        },
        'xgb_only': {
            'accuracy': xgb_accuracy * 100,
            'precision': xgb_precision * 100,
            'recall': xgb_recall * 100,
            'f1': xgb_f1 * 100,
            'auc': xgb_auc
        },
        'hybrid': {
            'accuracy': hybrid_accuracy * 100,
            'precision': hybrid_precision * 100,
            'recall': hybrid_recall * 100,
            'f1': hybrid_f1 * 100,
            'auc': hybrid_auc
        },
        'training_time': 145.2,  # Example value
        'inference_time': 2.8,   # Example value
        'parameters': 167890,    # Example value
        'confusion_matrix': cm_hybrid.tolist()
    }
    
    # Save metrics
    with open(os.path.join(viz.output_dir, 'model_metrics.json'), 'w') as f:
        json.dump(model_metrics, f, indent=4)
    
    # Save models and scaler
    joblib.dump(scaler, os.path.join(viz.output_dir, 'scaler.pkl'))
    joblib.dump(xgb_hybrid, os.path.join(viz.output_dir, 'xgb_hybrid_model.pkl'))
    autoencoder.save(os.path.join(viz.output_dir, 'autoencoder_model.h5'))
    
    print("✅ AE + XGBoost training completed!")
    print("📊 Visualizations saved to:", viz.output_dir)
    print("📈 Model metrics saved")
    
    return model_metrics

if __name__ == "__main__":
    metrics = train_ae_xgboost_ensemble()
    print("\nFinal Model Performance:")
    print(f"AE Only - Accuracy: {metrics['ae_only']['accuracy']:.2f}%")
    print(f"XGBoost Only - Accuracy: {metrics['xgb_only']['accuracy']:.2f}%")
    print(f"Hybrid AE+XGBoost - Accuracy: {metrics['hybrid']['accuracy']:.2f}%")