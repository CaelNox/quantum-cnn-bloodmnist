import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import f1_score, roc_auc_score, confusion_matrix, accuracy_score

def compute_metrics(y_true, y_pred, y_probs, num_classes=8):
    """
    Computes Multi-Class F1, Accuracy, and One-vs-Rest ROC-AUC scores.
    """
    macro_f1 = f1_score(y_true, y_pred, average='macro')
    weighted_f1 = f1_score(y_true, y_pred, average='weighted')
    acc = accuracy_score(y_true, y_pred)
    
    # One-vs-Rest Multi-Class ROC-AUC
    try:
        auc_score = roc_auc_score(y_true, y_probs, multi_class='ovr', average='macro')
    except Exception:
        auc_score = 0.0

    return {
        'accuracy': acc,
        'macro_f1': macro_f1,
        'weighted_f1': weighted_f1,
        'roc_auc': auc_score
    }

def plot_comparison_matrix(y_true, q_pred, c_pred, class_names, save_path="outputs/confusion_matrices.png"):
    """
    Generates side-by-side Confusion Matrix comparison for Quantum vs. Classical CNN.
    """
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    cm_q = confusion_matrix(y_true, q_pred)
    cm_c = confusion_matrix(y_true, c_pred)

    sns.heatmap(cm_q, annot=True, fmt='d', cmap='Blues', ax=axes[0],
                xticklabels=list(class_names.values()), yticklabels=list(class_names.values()))
    axes[0].set_title("Quantum CNN (Quanv) Confusion Matrix")
    axes[0].set_xlabel("Predicted")
    axes[0].set_ylabel("True")

    sns.heatmap(cm_c, annot=True, fmt='d', cmap='Oranges', ax=axes[1],
                xticklabels=list(class_names.values()), yticklabels=list(class_names.values()))
    axes[1].set_title("Classical CNN Baseline Confusion Matrix")
    axes[1].set_xlabel("Predicted")
    axes[1].set_ylabel("True")

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Confusion matrices saved to {save_path}")