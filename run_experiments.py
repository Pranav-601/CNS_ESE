"""
run_experiments.py
------------------
Master Execution Script for Member 3 (Model Lead).
Runs full end-to-end pipeline:
1. Data loading & verified preprocessing (no leakage).
2. Training 4-layer MLP with Custom Adam Learning Rate Decay (100 epochs).
3. Training Ablation 4-layer MLP with Fixed Learning Rate (100 epochs) [RQ4].
4. Training Baseline Logistic Regression & Decision Tree [RQ2].
5. Computing comprehensive evaluation metrics on KDDTest+.
6. Generating all publication figures (Loss Curve, LR Schedule, Confusion Matrix, ROC, Bar Charts, Architecture).
7. Exporting results tables (CSV/JSON) and trained models (joblib).
"""

import json
import time
from pathlib import Path
import numpy as np
import pandas as pd
import joblib

from src.data_pipeline import prepare_data
from src.model import (
    train_mlp_custom_schedule,
    train_mlp_fixed_schedule,
    train_baseline_logistic_regression,
    train_baseline_decision_tree
)
from src.evaluate import (
    evaluate_model,
    plot_loss_curves,
    plot_learning_rate_schedule,
    plot_confusion_matrices,
    plot_roc_curves,
    plot_classwise_metrics,
    plot_architecture_diagram
)


def main():
    print("=" * 75)
    print("   CNS ESE - GROUP 5: MULTILAYER PERCEPTRON FOR NETWORK ANOMALY DETECTION   ")
    print("                      MEMBER 3 MODEL LEAD PIPELINE                          ")
    print("=" * 75)

    base_dir = Path(__file__).resolve().parent
    outputs_dir = base_dir / "outputs"
    fig_dir = outputs_dir / "figures"
    tab_dir = outputs_dir / "tables"
    model_dir = outputs_dir / "models"

    for d in [fig_dir, tab_dir, model_dir]:
        d.mkdir(parents=True, exist_ok=True)

    # -------------------------------------------------------------
    # 1. DATA PREPARATION & INTEGRITY VERIFICATION
    # -------------------------------------------------------------
    print("\n[STEP 1/6] Loading data and executing preprocessing pipeline...")
    prep_path = model_dir / "preprocessor.joblib"
    X_train, y_train, X_val, y_val, X_test, y_test, preprocessor, meta = prepare_data(
        save_preprocessor_path=prep_path
    )
    print(f"-> Train samples: {X_train.shape[0]:,} | Features: {X_train.shape[1]}")
    print(f"-> Validation samples: {X_val.shape[0]:,} | Test samples: {X_test.shape[0]:,}")
    print(f"-> Preprocessor successfully saved to {prep_path.name}")

    # -------------------------------------------------------------
    # 2. PROPOSED MODEL: 4-LAYER MLP WITH CUSTOM ADAM LR DECAY
    # -------------------------------------------------------------
    print("\n[STEP 2/6] Training Proposed 4-Layer MLP with Custom Adam LR Decay (100 Epochs)...")
    mlp_decay, decay_history = train_mlp_custom_schedule(
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        epochs=100,
        random_state=42,
        batch_size=128,
        verbose=True
    )
    joblib.dump(mlp_decay, model_dir / "mlp_custom_decay.joblib")
    print(f"-> Proposed MLP training complete in {decay_history['total_training_time_sec']:.2f}s")

    # -------------------------------------------------------------
    # 3. ABLATION MODEL: 4-LAYER MLP WITH FIXED LEARNING RATE (RQ4)
    # -------------------------------------------------------------
    print("\n[STEP 3/6] Training Ablation 4-Layer MLP with Fixed LR = 0.001 (100 Epochs) for RQ4...")
    mlp_fixed, fixed_history = train_mlp_fixed_schedule(
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        epochs=100,
        learning_rate=0.001,
        random_state=42,
        batch_size=128,
        verbose=False
    )
    joblib.dump(mlp_fixed, model_dir / "mlp_fixed_lr.joblib")
    print(f"-> Fixed LR MLP training complete in {fixed_history['total_training_time_sec']:.2f}s")

    # -------------------------------------------------------------
    # 4. BASELINE MODELS: LOGISTIC REGRESSION & DECISION TREE (RQ2)
    # -------------------------------------------------------------
    print("\n[STEP 4/6] Training Baseline Classifiers (Logistic Regression & Decision Tree)...")
    lr_model, lr_time = train_baseline_logistic_regression(X_train, y_train, random_state=42)
    joblib.dump(lr_model, model_dir / "logistic_regression.joblib")
    print(f"-> Baseline 1 (Logistic Regression) fit in {lr_time:.2f}s")

    dt_model, dt_time = train_baseline_decision_tree(X_train, y_train, max_depth=10, random_state=42)
    joblib.dump(dt_model, model_dir / "decision_tree.joblib")
    print(f"-> Baseline 2 (Decision Tree) fit in {dt_time:.2f}s")

    # -------------------------------------------------------------
    # 5. TEST SET EVALUATION & BENCHMARKING
    # -------------------------------------------------------------
    print("\n[STEP 5/6] Evaluating all models on unseen official benchmark test set (KDDTest+)...")
    models = [
        ("Proposed MLP (Custom LR Decay)", mlp_decay, decay_history["total_training_time_sec"]),
        ("Ablation MLP (Fixed LR 0.001)", mlp_fixed, fixed_history["total_training_time_sec"]),
        ("Baseline (Logistic Regression)", lr_model, lr_time),
        ("Baseline (Decision Tree)", dt_model, dt_time)
    ]

    eval_results = []
    comparison_rows = []

    for name, model, train_t in models:
        # Measure inference latency on test set
        t_start_inf = time.time()
        res = evaluate_model(model, X_test, y_test, model_name=name)
        inf_time_ms = (time.time() - t_start_inf) * 1000.0

        res["training_time_sec"] = train_t
        res["inference_time_ms"] = inf_time_ms
        eval_results.append(res)

        comparison_rows.append({
            "Model": name,
            "Accuracy": round(res["accuracy"], 4),
            "Precision (Attack)": round(res["precision"], 4),
            "Recall (Attack)": round(res["recall"], 4),
            "F1-Score": round(res["f1_score"], 4),
            "ROC-AUC": round(res["roc_auc"], 4),
            "False Alarm Rate (FPR)": round(res["false_positive_rate"], 4),
            "Miss Rate (FNR)": round(res["false_negative_rate"], 4),
            "Training Time (s)": round(train_t, 2),
            "Inference Time (ms)": round(inf_time_ms, 2)
        })

    df_comparison = pd.DataFrame(comparison_rows)
    df_comparison.to_csv(tab_dir / "model_comparison_test.csv", index=False)
    print("\n" + "=" * 95)
    print("                           KDDTest+ FINAL PERFORMANCE BENCHMARK                           ")
    print("=" * 95)
    print(df_comparison.to_string(index=False))
    print("=" * 95)

    # Save training loss histories and detailed metadata
    with open(tab_dir / "mlp_training_history.json", "w", encoding="utf-8") as f:
        json.dump({"custom_decay": decay_history, "fixed_lr": fixed_history}, f, indent=2)

    # Clean JSON serializable metrics export
    json_export = []
    for r in eval_results:
        export_r = dict(r)
        del export_r["confusion_matrix"]
        del export_r["roc_curve"]
        json_export.append(export_r)
    with open(tab_dir / "detailed_test_metrics.json", "w", encoding="utf-8") as f:
        json.dump(json_export, f, indent=2)

    # -------------------------------------------------------------
    # 6. PUBLICATION FIGURE GENERATION
    # -------------------------------------------------------------
    print("\n[STEP 6/6] Generating high-resolution publication figures...")
    plot_loss_curves(decay_history, fixed_history, fig_dir / "loss_curve.png")
    print("-> Saved: outputs/figures/loss_curve.png")

    plot_learning_rate_schedule(decay_history, fig_dir / "lr_schedule.png")
    print("-> Saved: outputs/figures/lr_schedule.png")

    plot_confusion_matrices(eval_results, fig_dir / "confusion_matrices.png")
    print("-> Saved: outputs/figures/confusion_matrices.png")

    plot_roc_curves(eval_results, fig_dir / "roc_curves.png")
    print("-> Saved: outputs/figures/roc_curves.png")

    plot_classwise_metrics(eval_results, fig_dir / "classwise_metrics.png")
    print("-> Saved: outputs/figures/classwise_metrics.png")

    plot_architecture_diagram(fig_dir / "architecture_diagram.png")
    print("-> Saved: outputs/figures/architecture_diagram.png")

    print("\n" + "=" * 75)
    print(" [SUCCESS] ALL EXPERIMENTS, METRICS, AND FIGURES GENERATED SUCCESSFULLY!")
    print("=" * 75)


if __name__ == "__main__":
    main()
