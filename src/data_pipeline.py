"""
data_pipeline.py
----------------
Member 3 Data Loading, Preprocessing, and Validation Pipeline.
Strictly adheres to NSL-KDD benchmark contract:
- Loads official splits: train (100,779), validation (25,194), test (22,544).
- 41 canonical predictors (3 categorical, 38 numeric).
- One-hot encodes 'protocol_type', 'service', 'flag' with handle_unknown='ignore'.
- Scales 38 numeric features using StandardScaler.
- Fits all transformations strictly on TRAINING data only (no data leakage).
- Verified feature alignment: 38 numeric + 84 one-hot = 122 input features.
"""

import json
from pathlib import Path
from typing import Dict, Tuple, Optional
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import joblib


def get_default_data_dir() -> Path:
    """Returns the path to the official common dataset release."""
    base_dir = Path(__file__).resolve().parent.parent
    data_dir = base_dir / "NSL_KDD_Common_Dataset" / "nsl-kdd-binary-common-v1"
    if not data_dir.exists():
        raise FileNotFoundError(f"Dataset directory not found at: {data_dir}")
    return data_dir


def load_raw_dataset(data_dir: Optional[Path] = None) -> Dict[str, Tuple[pd.DataFrame, pd.Series]]:
    """
    Loads raw feature and label CSVs for train, validation, and test partitions.
    Validates row_id integrity and one-to-one correspondence.
    """
    if data_dir is None:
        data_dir = get_default_data_dir()

    schema_path = data_dir / "schema.json"
    if not schema_path.exists():
        raise FileNotFoundError(f"Schema file not found at: {schema_path}")
    
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    predictor_order = schema["predictor_order"]

    partitions = {}
    for part in ["train", "validation", "test"]:
        feat_path = data_dir / f"{part}_features.csv"
        label_path = data_dir / f"{part}_labels.csv"

        if not feat_path.exists() or not label_path.exists():
            raise FileNotFoundError(f"Missing {part} CSV files in {data_dir}")

        df_feat = pd.read_csv(feat_path, float_precision="round_trip")
        df_label = pd.read_csv(label_path)

        # Integrity check: Row IDs must be unique and match in exact order
        assert df_feat["row_id"].is_unique, f"Duplicate row_ids found in {part} features"
        assert df_label["row_id"].is_unique, f"Duplicate row_ids found in {part} labels"
        assert (df_feat["row_id"].values == df_label["row_id"].values).all(), (
            f"Row ID misalignment detected between features and labels in {part}"
        )

        # Ensure no unexpected NaNs
        assert df_feat[predictor_order].isna().sum().sum() == 0, f"NaNs detected in {part} features"
        assert df_label["target"].isna().sum() == 0, f"NaNs detected in {part} labels"

        partitions[part] = (df_feat[predictor_order], df_label["target"])

    return partitions, schema


def build_preprocessor(schema: dict) -> ColumnTransformer:
    """
    Constructs a ColumnTransformer that:
    1. Standardizes 38 numeric features.
    2. One-hot encodes 3 categorical features with handle_unknown='ignore'.
    """
    numeric_cols = schema["numeric"]
    categorical_cols = schema["categorical"]

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical_cols),
        ],
        remainder="drop"
    )
    return preprocessor


def prepare_data(data_dir: Optional[Path] = None, save_preprocessor_path: Optional[Path] = None):
    """
    End-to-end data preparation:
    - Loads raw datasets.
    - Fits preprocessor strictly on X_train.
    - Transforms X_train, X_val, X_test.
    - Returns processed NumPy arrays and summary metadata.
    """
    partitions, schema = load_raw_dataset(data_dir)
    X_train_raw, y_train = partitions["train"]
    X_val_raw, y_val = partitions["validation"]
    X_test_raw, y_test = partitions["test"]

    preprocessor = build_preprocessor(schema)

    # STRICT: Fit ONLY on training data to avoid any data leakage
    preprocessor.fit(X_train_raw)

    X_train = preprocessor.transform(X_train_raw)
    X_val = preprocessor.transform(X_val_raw)
    X_test = preprocessor.transform(X_test_raw)

    # Verification of shapes
    num_cat_features = len(preprocessor.named_transformers_["cat"].get_feature_names_out(schema["categorical"]))
    total_features = len(schema["numeric"]) + num_cat_features
    assert X_train.shape[1] == total_features, f"Expected {total_features} features, got {X_train.shape[1]}"
    assert X_val.shape[1] == total_features
    assert X_test.shape[1] == total_features

    # Verify no NaN or Inf
    assert not np.isnan(X_train).any(), "NaN found in X_train"
    assert not np.isnan(X_val).any(), "NaN found in X_val"
    assert not np.isnan(X_test).any(), "NaN found in X_test"

    if save_preprocessor_path is not None:
        save_preprocessor_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(preprocessor, save_preprocessor_path)

    metadata = {
        "num_raw_predictors": len(schema["predictor_order"]),
        "num_numeric_features": len(schema["numeric"]),
        "num_categorical_features": len(schema["categorical"]),
        "num_encoded_categorical_features": num_cat_features,
        "total_model_features": total_features,
        "train_samples": len(y_train),
        "validation_samples": len(y_val),
        "test_samples": len(y_test),
        "train_class_distribution": y_train.value_counts().to_dict(),
        "validation_class_distribution": y_val.value_counts().to_dict(),
        "test_class_distribution": y_test.value_counts().to_dict(),
    }

    return (
        X_train, y_train.values,
        X_val, y_val.values,
        X_test, y_test.values,
        preprocessor,
        metadata
    )


if __name__ == "__main__":
    print("Testing data pipeline execution...")
    X_train, y_train, X_val, y_val, X_test, y_test, preprocessor, meta = prepare_data()
    print("Pipeline executed successfully!")
    print(f"X_train shape: {X_train.shape}, y_train: {y_train.shape}")
    print(f"X_val shape:   {X_val.shape}, y_val: {y_val.shape}")
    print(f"X_test shape:  {X_test.shape}, y_test: {y_test.shape}")
    print("\nMetadata Summary:")
    print(json.dumps(meta, indent=2))
