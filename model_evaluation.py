"""
Model evaluation and metrics calculation for lung cancer detection models.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report,
    roc_curve, precision_recall_curve
)
from typing import Dict, List, Tuple, Optional
import streamlit as st
from logger import logger

class ModelEvaluator:
    """Comprehensive model evaluation for medical AI systems."""
    
    @staticmethod
    def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_prob: Optional[np.ndarray] = None) -> Dict:
        """Calculate comprehensive evaluation metrics."""
        try:
            metrics = {}
            
            # Basic classification metrics
            metrics['accuracy'] = accuracy_score(y_true, y_pred)
            metrics['precision'] = precision_score(y_true, y_pred, average='binary')
            metrics['recall'] = recall_score(y_true, y_pred, average='binary')
            metrics['f1_score'] = f1_score(y_true, y_pred, average='binary')
            
            # Specificity (True Negative Rate)
            tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
            metrics['specificity'] = tn / (tn + fp) if (tn + fp) > 0 else 0
            metrics['sensitivity'] = metrics['recall']  # Same as recall
            
            # Positive and Negative Predictive Values
            metrics['ppv'] = tp / (tp + fp) if (tp + fp) > 0 else 0  # Same as precision
            metrics['npv'] = tn / (tn + fn) if (tn + fn) > 0 else 0
            
            # AUC-ROC if probabilities provided
            if y_prob is not None:
                metrics['auc_roc'] = roc_auc_score(y_true, y_prob)
            
            # Medical AI specific metrics
            metrics['false_positive_rate'] = fp / (fp + tn) if (fp + tn) > 0 else 0
            metrics['false_negative_rate'] = fn / (fn + tp) if (fn + tp) > 0 else 0
            
            # Confusion matrix components
            metrics['true_positives'] = int(tp)
            metrics['true_negatives'] = int(tn)
            metrics['false_positives'] = int(fp)
            metrics['false_negatives'] = int(fn)
            
            logger.info(f"Calculated evaluation metrics: {metrics}")
            return metrics
            
        except Exception as e:
            logger.error(f"Error calculating metrics: {e}")
            return {}
    
    @staticmethod
    def plot_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, class_names: List[str] = None) -> plt.Figure:
        """Create confusion matrix visualization."""
        if class_names is None:
            class_names = ['Healthy', 'Cancer']
        
        cm = confusion_matrix(y_true, y_pred)
        
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=class_names, yticklabels=class_names, ax=ax)
        
        ax.set_title('Confusion Matrix')
        ax.set_xlabel('Predicted Label')
        ax.set_ylabel('True Label')
        
        return fig
    
    @staticmethod
    def plot_roc_curve(y_true: np.ndarray, y_prob: np.ndarray) -> plt.Figure:
        """Create ROC curve visualization."""
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        auc = roc_auc_score(y_true, y_prob)
        
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {auc:.2f})')
        ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random')
        
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title('Receiver Operating Characteristic (ROC) Curve')
        ax.legend(loc="lower right")
        ax.grid(True, alpha=0.3)
        
        return fig
    
    @staticmethod
    def plot_precision_recall_curve(y_true: np.ndarray, y_prob: np.ndarray) -> plt.Figure:
        """Create Precision-Recall curve visualization."""
        precision, recall, _ = precision_recall_curve(y_true, y_prob)
        
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.plot(recall, precision, color='blue', lw=2)
        ax.fill_between(recall, precision, alpha=0.2, color='blue')
        
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('Recall')
        ax.set_ylabel('Precision')
        ax.set_title('Precision-Recall Curve')
        ax.grid(True, alpha=0.3)
        
        return fig
    
    @staticmethod
    def generate_classification_report(y_true: np.ndarray, y_pred: np.ndarray, class_names: List[str] = None) -> str:
        """Generate detailed classification report."""
        if class_names is None:
            class_names = ['Healthy', 'Cancer']
        
        return classification_report(y_true, y_pred, target_names=class_names)
    
    @staticmethod
    def medical_ai_evaluation_summary(metrics: Dict) -> Dict:
        """Create medical AI specific evaluation summary."""
        summary = {
            'clinical_performance': {
                'sensitivity': metrics.get('sensitivity', 0),
                'specificity': metrics.get('specificity', 0),
                'ppv': metrics.get('ppv', 0),
                'npv': metrics.get('npv', 0)
            },
            'diagnostic_accuracy': {
                'overall_accuracy': metrics.get('accuracy', 0),
                'balanced_accuracy': (metrics.get('sensitivity', 0) + metrics.get('specificity', 0)) / 2
            },
            'error_analysis': {
                'false_positive_rate': metrics.get('false_positive_rate', 0),
                'false_negative_rate': metrics.get('false_negative_rate', 0),
                'missed_cancers': metrics.get('false_negatives', 0),
                'false_alarms': metrics.get('false_positives', 0)
            }
        }
        
        return summary
    
    @staticmethod
    def display_evaluation_results(metrics: Dict, y_true: np.ndarray = None, y_pred: np.ndarray = None, y_prob: np.ndarray = None):
        """Display comprehensive evaluation results in Streamlit."""
        st.subheader("📊 Model Evaluation Results")
        
        # Key metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Accuracy", f"{metrics.get('accuracy', 0):.3f}")
            st.metric("Sensitivity", f"{metrics.get('sensitivity', 0):.3f}")
        
        with col2:
            st.metric("Precision", f"{metrics.get('precision', 0):.3f}")
            st.metric("Specificity", f"{metrics.get('specificity', 0):.3f}")
        
        with col3:
            st.metric("F1 Score", f"{metrics.get('f1_score', 0):.3f}")
            st.metric("PPV", f"{metrics.get('ppv', 0):.3f}")
        
        with col4:
            if 'auc_roc' in metrics:
                st.metric("AUC-ROC", f"{metrics.get('auc_roc', 0):.3f}")
            st.metric("NPV", f"{metrics.get('npv', 0):.3f}")
        
        # Medical AI summary
        summary = ModelEvaluator.medical_ai_evaluation_summary(metrics)
        
        st.subheader("🏥 Clinical Performance Summary")
        
        clinical_df = pd.DataFrame([
            {"Metric": "Sensitivity (Recall)", "Value": f"{summary['clinical_performance']['sensitivity']:.3f}", "Description": "Ability to detect cancer cases"},
            {"Metric": "Specificity", "Value": f"{summary['clinical_performance']['specificity']:.3f}", "Description": "Ability to identify healthy cases"},
            {"Metric": "Positive Predictive Value", "Value": f"{summary['clinical_performance']['ppv']:.3f}", "Description": "Probability cancer when predicted positive"},
            {"Metric": "Negative Predictive Value", "Value": f"{summary['clinical_performance']['npv']:.3f}", "Description": "Probability healthy when predicted negative"}
        ])
        
        st.dataframe(clinical_df, use_container_width=True)
        
        # Visualizations
        if y_true is not None and y_pred is not None:
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Confusion Matrix")
                fig_cm = ModelEvaluator.plot_confusion_matrix(y_true, y_pred)
                st.pyplot(fig_cm)
            
            with col2:
                if y_prob is not None:
                    st.subheader("ROC Curve")
                    fig_roc = ModelEvaluator.plot_roc_curve(y_true, y_prob)
                    st.pyplot(fig_roc)

class ModelComparison:
    """Compare multiple models for lung cancer detection."""
    
    @staticmethod
    def compare_models(model_results: Dict[str, Dict]) -> pd.DataFrame:
        """Compare multiple models and return comparison DataFrame."""
        comparison_data = []
        
        for model_name, metrics in model_results.items():
            comparison_data.append({
                'Model': model_name,
                'Accuracy': metrics.get('accuracy', 0),
                'Sensitivity': metrics.get('sensitivity', 0),
                'Specificity': metrics.get('specificity', 0),
                'Precision': metrics.get('precision', 0),
                'F1 Score': metrics.get('f1_score', 0),
                'AUC-ROC': metrics.get('auc_roc', 0)
            })
        
        return pd.DataFrame(comparison_data)
    
    @staticmethod
    def plot_model_comparison(comparison_df: pd.DataFrame) -> plt.Figure:
        """Create model comparison visualization."""
        metrics_to_plot = ['Accuracy', 'Sensitivity', 'Specificity', 'Precision', 'F1 Score']
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        x = np.arange(len(comparison_df))
        width = 0.15
        
        for i, metric in enumerate(metrics_to_plot):
            ax.bar(x + i * width, comparison_df[metric], width, label=metric)
        
        ax.set_xlabel('Models')
        ax.set_ylabel('Score')
        ax.set_title('Model Performance Comparison')
        ax.set_xticks(x + width * 2)
        ax.set_xticklabels(comparison_df['Model'])
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        return fig

# Global evaluator instance
model_evaluator = ModelEvaluator()
model_comparison = ModelComparison()