"""
model.py
--------
Member 3 Model Implementation:
- 4-layer feed-forward Multilayer Perceptron (MLP): [Input -> 64 -> 32 -> 16 -> Output]
- ReLU activation in hidden layers
- Adam optimizer
- Custom epoch-wise learning rate decay schedule:
    Epochs 1-20:   0.001
    Epochs 21-40:  0.0005
    Epochs 41-60:  0.00025
    Epochs 61-80:  0.0001
    Epochs 81-100: 0.00005
- Controlled fixed-rate MLP for rigorous RQ4 empirical ablation
- Baseline classifiers (Logistic Regression & Decision Tree) for RQ2 comparative analysis
"""

import time
import warnings
from typing import Dict, Any, Optional, Tuple
import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import log_loss, accuracy_score


# Documented custom learning-rate decay table (0-indexed epochs)
DEFAULT_DECAY_TABLE = {
    0: 0.001,      # Epochs 1-20
    20: 0.0005,    # Epochs 21-40
    40: 0.00025,   # Epochs 41-60
    60: 0.0001,    # Epochs 61-80
    80: 0.00005,   # Epochs 81-100
}


def build_mlp_classifier(
    hidden_layer_sizes: Tuple[int, ...] = (64, 32, 16),
    activation: str = "relu",
    solver: str = "adam",
    initial_lr: float = 0.001,
    random_state: int = 42,
    batch_size: Any = 128
) -> MLPClassifier:
    """
    Constructs the 4-layer MLPClassifier conforming to Group 5 specifications:
    - Input dimension: 122 (preprocessed features)
    - Hidden layers: (64, 32, 16)
    - Activation: ReLU
    - Solver: Adam
    - max_iter: 1 with warm_start=True for custom epoch loop
    """
    return MLPClassifier(
        hidden_layer_sizes=hidden_layer_sizes,
        activation=activation,
        solver=solver,
        learning_rate_init=initial_lr,
        batch_size=batch_size,
        max_iter=1,
        warm_start=True,
        random_state=random_state,
        shuffle=True
    )


def train_mlp_custom_schedule(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: Optional[np.ndarray] = None,
    y_val: Optional[np.ndarray] = None,
    epochs: int = 100,
    decay_table: Optional[Dict[int, float]] = None,
    random_state: int = 42,
    batch_size: Any = 128,
    verbose: bool = True
) -> Tuple[MLPClassifier, Dict[str, Any]]:
    """
    Trains the 4-layer MLP over specified epochs using the custom learning-rate decay schedule.
    Records epoch-by-epoch loss, learning rate, validation metrics, and elapsed time.
    """
    if decay_table is None:
        decay_table = DEFAULT_DECAY_TABLE

    mlp = build_mlp_classifier(
        initial_lr=decay_table.get(0, 0.001),
        random_state=random_state,
        batch_size=batch_size
    )

    history = {
        "epoch": [],
        "learning_rate": [],
        "train_loss": [],
        "val_loss": [],
        "val_accuracy": [],
        "epoch_duration_sec": []
    }

    current_lr = decay_table.get(0, 0.001)
    start_total_time = time.time()

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=ConvergenceWarning)

        for epoch in range(epochs):
            t_epoch_start = time.time()

            # Apply custom decay at scheduled checkpoints
            if epoch in decay_table:
                current_lr = decay_table[epoch]
                mlp.learning_rate_init = current_lr

            # One full epoch pass over training data
            mlp.fit(X_train, y_train)

            epoch_dur = time.time() - t_epoch_start
            train_loss = float(mlp.loss_)

            history["epoch"].append(epoch + 1)
            history["learning_rate"].append(current_lr)
            history["train_loss"].append(train_loss)
            history["epoch_duration_sec"].append(epoch_dur)

            # Evaluate on validation split if provided
            if X_val is not None and y_val is not None:
                val_probs = mlp.predict_proba(X_val)
                val_preds = mlp.predict(X_val)
                v_loss = float(log_loss(y_val, val_probs))
                v_acc = float(accuracy_score(y_val, val_preds))
                history["val_loss"].append(v_loss)
                history["val_accuracy"].append(v_acc)

            if verbose and ((epoch + 1) % 10 == 0 or epoch == 0 or epoch == epochs - 1):
                val_str = f" | Val Loss: {history['val_loss'][-1]:.5f} | Val Acc: {history['val_accuracy'][-1]:.4f}" if X_val is not None else ""
                print(f"[Epoch {epoch+1:03d}/{epochs:03d}] LR: {current_lr:.5f} | Train Loss: {train_loss:.5f}{val_str} ({epoch_dur:.2f}s)")

    total_training_time = time.time() - start_total_time
    history["total_training_time_sec"] = total_training_time

    return mlp, history


def train_mlp_fixed_schedule(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: Optional[np.ndarray] = None,
    y_val: Optional[np.ndarray] = None,
    epochs: int = 100,
    learning_rate: float = 0.001,
    random_state: int = 42,
    batch_size: Any = 128,
    verbose: bool = False
) -> Tuple[MLPClassifier, Dict[str, Any]]:
    """
    Ablation model: Trains an identical MLP with a constant learning rate (0.001).
    Used for rigorous empirical comparison to address Research Question 4 (RQ4).
    """
    fixed_table = {0: learning_rate}
    return train_mlp_custom_schedule(
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        epochs=epochs,
        decay_table=fixed_table,
        random_state=random_state,
        batch_size=batch_size,
        verbose=verbose
    )


def train_baseline_logistic_regression(
    X_train: np.ndarray,
    y_train: np.ndarray,
    random_state: int = 42
) -> Tuple[LogisticRegression, float]:
    """
    Trains standard Logistic Regression as a linear benchmark classifier.
    """
    t0 = time.time()
    model = LogisticRegression(
        max_iter=1000,
        random_state=random_state,
        solver="lbfgs"
    )
    model.fit(X_train, y_train)
    duration = time.time() - t0
    return model, duration


def train_baseline_decision_tree(
    X_train: np.ndarray,
    y_train: np.ndarray,
    max_depth: int = 10,
    random_state: int = 42
) -> Tuple[DecisionTreeClassifier, float]:
    """
    Trains a Decision Tree classifier as a non-linear tree-based benchmark.
    """
    t0 = time.time()
    model = DecisionTreeClassifier(
        max_depth=max_depth,
        random_state=random_state
    )
    model.fit(X_train, y_train)
    duration = time.time() - t0
    return model, duration
