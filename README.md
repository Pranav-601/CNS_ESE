# CNS ESE - Group 5: Multilayer Perceptron for Network Anomaly Detection
## Member 3 Deliverables: Model Implementation, Training, Ablation, Evaluation & Paper Material

This directory contains the complete, reproducible implementation for **Member 3 (Model Lead)** using the NSL-KDD benchmark.

---

### Project Structure

```
CNS_ese/
├── NSL_KDD_Common_Dataset/
│   └── nsl-kdd-binary-common-v1/      # Official shared benchmark dataset
├── notebooks/
│   └── MLP_Anomaly_Detection_Experiments.ipynb   # Interactive Jupyter notebook
├── src/
│   ├── __init__.py
│   ├── data_pipeline.py               # Preprocessing pipeline (StandardScaler + OneHotEncoder)
│   ├── model.py                       # 4-layer MLP, custom Adam LR decay, and baseline classifiers
│   └── evaluate.py                    # Evaluation metrics, ROC, confusion matrix, and publication figures
├── outputs/
│   ├── figures/                       # High-resolution (300 DPI) publication figures
│   │   ├── loss_curve.png             # Figure 1: Training loss convergence over 100 epochs
│   │   ├── lr_schedule.png            # Figure 2: Scheduled Adam learning rate step-decay curve
│   │   ├── confusion_matrices.png     # Figure 3: Multi-model test set confusion matrices
│   │   ├── roc_curves.png             # Figure 4: ROC curves with AUC scores
│   │   ├── classwise_metrics.png      # Figure 5: Class-wise Precision/Recall comparison
│   │   └── architecture_diagram.png   # Figure 6: Schematic of 4-layer feed-forward network
│   ├── models/                        # Serialized models and fitted preprocessors
│   │   ├── mlp_custom_decay.joblib    # Trained proposed MLP model
│   │   ├── mlp_fixed_lr.joblib        # Trained ablation fixed-rate MLP model
│   │   ├── logistic_regression.joblib # Trained baseline Logistic Regression
│   │   ├── decision_tree.joblib       # Trained baseline Decision Tree
│   │   └── preprocessor.joblib        # Fitted ColumnTransformer pipeline
│   └── tables/                        # Formatted benchmark results & training logs
│       ├── model_comparison_test.csv  # Final benchmark comparison on KDDTest+
│       ├── detailed_test_metrics.json # Full classification reports & confusion matrix raw counts
│       └── mlp_training_history.json  # Epoch-by-epoch loss and learning rate trajectory
├── run_experiments.py                 # Master execution script (reproduces all results in ~1 min)
├── MEMBER_3_PAPER_SECTIONS.md         # Complete draft for Paper Sections 5, 6, and 7
└── CNS Research Work.pdf              # Group assignment specification & rubric
```

---

### How to Run

1. **Run Full End-to-End Experiment Pipeline:**
   ```bash
   python run_experiments.py
   ```
   This will automatically load the data, fit transformers on training data only, train the 4 models, evaluate on `KDDTest+`, save all models, and generate all 6 publication figures.

2. **Open Interactive Notebook:**
   ```bash
   jupyter notebook notebooks/MLP_Anomaly_Detection_Experiments.ipynb
   ```

---

### Summary of Final Experimental Results (KDDTest+)

| Model | Accuracy | Precision (Attack) | Recall (Attack) | F1-Score | ROC-AUC | False Alarm Rate (FPR) | Miss Rate (FNR) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Proposed MLP (Custom LR Decay)** | **82.21%** | **96.90%** | **71.02%** | **0.8196** | **0.8603** | **3.01%** | **28.98%** |
| **Ablation MLP (Fixed LR 0.001)** | 77.28% | 95.65% | 62.95% | 0.7593 | 0.8505 | 3.78% | 37.05% |
| **Baseline (Logistic Regression)** | 75.32% | 91.77% | 62.23% | 0.7417 | 0.7962 | 7.37% | 37.77% |
| **Baseline (Decision Tree)** | 78.19% | 96.69% | 63.87% | 0.7693 | 0.7592 | 2.89% | 36.13% |

* **Key Finding:** Custom learning-rate decay delivers an **+8.07% attack recall improvement** and **+4.93% accuracy boost** over the fixed-rate counterpart, while outperforming Logistic Regression by **+6.89% accuracy** and cutting false alarms by more than half.
