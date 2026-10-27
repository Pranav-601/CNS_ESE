"""
evaluate.py
-----------
Comprehensive Model Evaluation, Metrics Generation, and Publication Figures.
Generates:
1. Classification metrics table (Accuracy, Precision, Recall, F1, ROC-AUC, FPR, FNR)
2. Loss curve comparison (Custom Decay vs Fixed LR)
3. Learning rate schedule step-curve
4. High-resolution Confusion Matrix heatmaps
5. Multi-model ROC Curves
6. Class-wise Precision/Recall performance comparison
7. MLP Architecture Schematic Diagram
"""

import os
from pathlib import Path
from typing import Dict, Any, List, Tuple
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report, roc_curve
)

# Publication aesthetic settings
plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.titlesize": 14,
    "figure.dpi": 300
})


def evaluate_model(model: Any, X: np.ndarray, y: np.ndarray, model_name: str = "Model") -> Dict[str, Any]:
    """
    Computes all standard intrusion detection evaluation metrics on the provided dataset.
    """
    y_pred = model.predict(X)
    
    # Check if predict_proba is available for ROC-AUC
    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X)[:, 1]
    elif hasattr(model, "decision_function"):
        y_prob = model.decision_function(X)
    else:
        y_prob = y_pred

    cm = confusion_matrix(y, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    acc = accuracy_score(y, y_pred)
    prec = precision_score(y, y_pred, pos_label=1, zero_division=0)
    rec = recall_score(y, y_pred, pos_label=1, zero_division=0)
    f1 = f1_score(y, y_pred, pos_label=1, zero_division=0)
    auc = roc_auc_score(y, y_prob)

    # Security-critical rates
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0  # False alarm rate
    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0  # Missed attack rate

    report_dict = classification_report(y, y_pred, target_names=["Normal", "Attack"], output_dict=True)

    fpr_curve, tpr_curve, _ = roc_curve(y, y_prob)

    return {
        "model_name": model_name,
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1_score": f1,
        "roc_auc": auc,
        "false_positive_rate": fpr,
        "false_negative_rate": fnr,
        "true_negatives": int(tn),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "true_positives": int(tp),
        "confusion_matrix": cm,
        "classification_report": report_dict,
        "roc_curve": (fpr_curve, tpr_curve)
    }


def plot_loss_curves(
    decay_history: Dict[str, Any],
    fixed_history: Dict[str, Any],
    output_path: Path
):
    """
    Plots training loss over 100 epochs comparing custom LR decay vs fixed LR.
    Highlights step decay milestones.
    """
    epochs = decay_history["epoch"]
    
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(epochs, decay_history["train_loss"], label="MLP with Custom LR Decay", color="#1f77b4", linewidth=2.2)
    ax.plot(epochs, fixed_history["train_loss"], label="MLP with Fixed LR (0.001)", color="#ff7f0e", linestyle="--", linewidth=2.0)

    # Highlight decay milestone epochs
    milestones = [20, 40, 60, 80]
    for m in milestones:
        ax.axvline(x=m, color="#7f7f7f", linestyle=":", alpha=0.7)
        ax.text(m + 1, ax.get_ylim()[1] * 0.75 if ax.get_ylim()[1] else 0.05, f"Decay @ Ep {m}", rotation=90, fontsize=8, color="#555555")

    ax.set_title("Training Loss Convergence Across 100 Epochs", fontweight="bold", pad=12)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Cross-Entropy Training Loss")
    ax.grid(True, linestyle="--", alpha=0.6)
    ax.legend(frameon=True, loc="upper right")
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300)
    plt.close()


def plot_learning_rate_schedule(decay_history: Dict[str, Any], output_path: Path):
    """
    Plots the step-wise learning rate decay schedule across epochs.
    """
    epochs = decay_history["epoch"]
    lrs = decay_history["learning_rate"]

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.step(epochs, lrs, where="post", color="#2ca02c", linewidth=2.5, label="Scheduled Learning Rate")
    ax.scatter(epochs, lrs, color="#2ca02c", s=15, alpha=0.5)

    ax.set_yscale("log")
    ax.set_title("Prescribed Adam Learning-Rate Decay Schedule", fontweight="bold", pad=12)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Learning Rate (log scale)")
    ax.grid(True, which="both", linestyle="--", alpha=0.6)
    ax.legend(loc="upper right")

    # Annotate plateau values
    plateaus = [(10, 0.001, "0.001"), (30, 0.0005, "5e-4"), (50, 0.00025, "2.5e-4"),
                (70, 0.0001, "1e-4"), (90, 0.00005, "5e-5")]
    for ep, lr_val, text_lbl in plateaus:
        ax.text(ep, lr_val * 1.15, text_lbl, ha="center", fontsize=9, fontweight="bold", color="#1c6b1c")

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300)
    plt.close()


def plot_confusion_matrices(eval_results: List[Dict[str, Any]], output_path: Path):
    """
    Plots side-by-side confusion matrix heatmaps for models.
    """
    n_models = len(eval_results)
    fig, axes = plt.subplots(1, n_models, figsize=(5.5 * n_models, 4.8))
    if n_models == 1:
        axes = [axes]

    for ax, res in zip(axes, eval_results):
        cm = res["confusion_matrix"]
        cm_perc = cm / cm.sum() * 100.0

        # Annotations with counts and percentages
        labels = np.array([
            [f"{cm[0,0]:,}\n({cm_perc[0,0]:.1f}%)", f"{cm[0,1]:,}\n({cm_perc[0,1]:.1f}%)"],
            [f"{cm[1,0]:,}\n({cm_perc[1,0]:.1f}%)", f"{cm[1,1]:,}\n({cm_perc[1,1]:.1f}%)"]
        ])

        sns.heatmap(cm, annot=labels, fmt="", cmap="Blues", cbar=False, ax=ax,
                    xticklabels=["Normal", "Attack"], yticklabels=["Normal", "Attack"])
        ax.set_title(f"{res['model_name']}\n(Acc: {res['accuracy']*100:.2f}%)", fontweight="bold")
        ax.set_xlabel("Predicted Class")
        ax.set_ylabel("True Class")

    plt.suptitle("Test Set Confusion Matrices on KDDTest+", fontweight="bold", y=1.03)
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300)
    plt.close()


def plot_roc_curves(eval_results: List[Dict[str, Any]], output_path: Path):
    """
    Plots combined Receiver Operating Characteristic (ROC) curves on test set.
    """
    fig, ax = plt.subplots(figsize=(7.5, 6))
    ax.plot([0, 1], [0, 1], "k--", alpha=0.6, label="Random Guess (AUC = 0.5000)")

    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]
    for res, color in zip(eval_results, colors):
        fpr_c, tpr_c = res["roc_curve"]
        ax.plot(fpr_c, tpr_c, label=f"{res['model_name']} (AUC = {res['roc_auc']:.4f})", linewidth=2.0, color=color)

    ax.set_xlim([-0.02, 1.02])
    ax.set_ylim([-0.02, 1.02])
    ax.set_title("Test Set Receiver Operating Characteristic (ROC) Curves", fontweight="bold", pad=12)
    ax.set_xlabel("False Positive Rate (1 - Specificity)")
    ax.set_ylabel("True Positive Rate (Recall / Sensitivity)")
    ax.grid(True, linestyle="--", alpha=0.6)
    ax.legend(loc="lower right", frameon=True)

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300)
    plt.close()


def plot_classwise_metrics(eval_results: List[Dict[str, Any]], output_path: Path):
    """
    Plots class-wise precision and recall bar chart across models.
    """
    model_names = [res["model_name"] for res in eval_results]
    n_models = len(model_names)
    indices = np.arange(n_models)
    width = 0.2

    norm_prec = [res["classification_report"]["Normal"]["precision"] for res in eval_results]
    norm_rec = [res["classification_report"]["Normal"]["recall"] for res in eval_results]
    att_prec = [res["classification_report"]["Attack"]["precision"] for res in eval_results]
    att_rec = [res["classification_report"]["Attack"]["recall"] for res in eval_results]

    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.bar(indices - 1.5 * width, norm_prec, width, label="Normal Precision", color="#4c72b0")
    ax.bar(indices - 0.5 * width, norm_rec, width, label="Normal Recall", color="#8da0cb")
    ax.bar(indices + 0.5 * width, att_prec, width, label="Attack Precision", color="#c44e52")
    ax.bar(indices + 1.5 * width, att_rec, width, label="Attack Recall", color="#e78ac3")

    ax.set_ylabel("Metric Score (0.0 - 1.0)")
    ax.set_title("Class-Wise Precision & Recall Breakdown on KDDTest+", fontweight="bold", pad=12)
    ax.set_xticks(indices)
    ax.set_xticklabels(model_names, rotation=15, ha="right", fontweight="bold")
    ax.set_ylim([0.0, 1.1])
    ax.grid(True, axis="y", linestyle="--", alpha=0.6)
    ax.legend(loc="lower left", frameon=True, ncol=2)

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300)
    plt.close()


from matplotlib.patches import FancyBboxPatch

def plot_architecture_diagram(output_path: Path):
    """
    Draws a clean, publication-ready schematic diagram of the 4-layer MLP network architecture.
    """
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.axis("off")

    layers = [
        {"name": "Input Layer", "nodes": 122, "type": "Features", "color": "#d9e2ec", "desc": "122 Dims\n(38 Numeric + 84 OHE)"},
        {"name": "Hidden Layer 1", "nodes": 64, "type": "Dense + ReLU", "color": "#b8c0ff", "desc": "64 Neurons\nReLU Activation"},
        {"name": "Hidden Layer 2", "nodes": 32, "type": "Dense + ReLU", "color": "#c8b6ff", "desc": "32 Neurons\nReLU Activation"},
        {"name": "Hidden Layer 3", "nodes": 16, "type": "Dense + ReLU", "color": "#e7c6ff", "desc": "16 Neurons\nReLU Activation"},
        {"name": "Output Layer", "nodes": 1, "type": "Binary Output", "color": "#ffd6a5", "desc": "1 Output\nNormal vs Attack"}
    ]

    x_positions = np.linspace(0.1, 0.9, len(layers))
    y_center = 0.5

    for i, (l, x) in enumerate(zip(layers, x_positions)):
        # Draw box
        box_width = 0.13
        box_height = 0.55
        rect = FancyBboxPatch((x - box_width/2, y_center - box_height/2), box_width, box_height,
                              boxstyle="round,pad=0.02,rounding_size=0.02",
                              facecolor=l["color"], edgecolor="#333333", linewidth=1.5,
                              transform=ax.transAxes)
        ax.add_patch(rect)

        # Labels
        ax.text(x, y_center + 0.16, l["name"], transform=ax.transAxes, ha="center", va="center",
                fontsize=10, fontweight="bold")
        ax.text(x, y_center + 0.05, f"[{l['nodes']}]", transform=ax.transAxes, ha="center", va="center",
                fontsize=12, fontweight="bold", color="#111111")
        ax.text(x, y_center - 0.12, l["desc"], transform=ax.transAxes, ha="center", va="center",
                fontsize=8.5, color="#222222")

        # Arrows between layers
        if i < len(layers) - 1:
            next_x = x_positions[i+1]
            ax.annotate("", xy=(next_x - box_width/2, y_center), xytext=(x + box_width/2, y_center),
                        xycoords="axes fraction", textcoords="axes fraction",
                        arrowprops=dict(arrowstyle="->", lw=2.2, color="#444444"))

    ax.text(0.5, 0.92, "Proposed 4-Layer Feed-Forward MLP Architecture [122 -> 64 -> 32 -> 16 -> 1]",
            transform=ax.transAxes, ha="center", fontsize=13, fontweight="bold")
    ax.text(0.5, 0.85, "Progressive dimensionality compression with Adam optimizer & custom LR step-decay",
            transform=ax.transAxes, ha="center", fontsize=9.5, style="italic", color="#555555")

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300)
    plt.close()
