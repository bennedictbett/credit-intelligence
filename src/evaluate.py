import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (roc_auc_score, roc_curve, 
                              confusion_matrix, classification_report)
from scipy.stats import ks_2samp


def compute_metrics(y_test, y_pred_proba) -> dict:
    """Compute all credit risk evaluation metrics."""
    auc = roc_auc_score(y_test, y_pred_proba)
    gini = 2 * auc - 1
    
    defaulters = y_pred_proba[y_test == 1]
    non_defaulters = y_pred_proba[y_test == 0]
    ks_stat, _ = ks_2samp(defaulters, non_defaulters)
    
    metrics = {
        'ROC-AUC': round(auc, 4),
        'Gini': round(gini, 4),
        'KS-Statistic': round(ks_stat, 4)
    }
    
    for name, value in metrics.items():
        print(f"{name}: {value}")
    
    return metrics


def plot_roc_curve(y_test, y_pred_proba, save_path=None):
    """Plot ROC curve."""
    auc = roc_auc_score(y_test, y_pred_proba)
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    
    plt.figure(figsize=(10, 7))
    plt.plot(fpr, tpr, color='darkorange', lw=2,
             label=f'ROC Curve (AUC = {auc:.4f})')
    plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend()
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
    plt.show()


def plot_confusion_matrix(y_test, y_pred, save_path=None):
    """Plot confusion matrix."""
    import seaborn as sns
    cm = confusion_matrix(y_test, y_pred)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Repaid', 'Defaulted'],
                yticklabels=['Repaid', 'Defaulted'])
    plt.title('Confusion Matrix')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
    plt.show()
    
    print(classification_report(y_test, y_pred,
                                target_names=['Repaid', 'Defaulted']))