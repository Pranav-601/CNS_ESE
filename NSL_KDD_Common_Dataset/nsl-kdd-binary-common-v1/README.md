# Correct shared dataset — NSL-KDD binary common release

USE THIS RELEASE for the assignment. The earlier KDD Cup 99 binary-common-v1 and
Common_Dataset.zip belong to another dataset; their model scores do not apply here.

Normal=0, Attack=1. This release uses full KDDTrain+.txt and full KDDTest+.txt from
the supplied DATASET folder. KDDTrain+_20Percent and KDDTest-21 are subsets and are
not appended. The nested nsl-kdd folder and ARFF files are alternate copies/formats.

Shared files: train_features.csv, validation_features.csv, test_features.csv each
contain row_id then 41 canonical predictors. Matching *_labels.csv files contain
row_id,target in exactly the same row order. Use schema.json as the predictor
allow-list. source_label, difficulty, row IDs, source positions and audit flags are
METADATA ONLY, never model inputs or sample weights.

The original test set is preserved in its supplied order. Training/validation come
only from KDDTrain+, using the first of five StratifiedGroupKFold splits, shuffle=True,
seed=42; identical feature vectors stay together in this development split.

All supplied benchmark rows are retained. Unlike the earlier KDD Cup 99 protocol,
there is no global deduplication or binary-conflict removal. This preserves the
NSL-KDD benchmark populations. Feature-only duplicates and contradictory binary
labels (after excluding difficulty and original label) are flagged in membership.csv.
There is measured exact-feature overlap between the supplied development and test
sets; see quality_report.json. It is disclosed, not removed or claimed to be zero.

Categories remain text and numerical values retain their original units. No encoding,
scaling, balancing, imputation or feature selection is fitted centrally. Fit your
model's preprocessing on TRAIN ONLY and apply it unchanged to validation/test.
Unknown categories must be handled explicitly. The full test includes attack labels
not present in development data, all mapped to Attack by the stored allow-list.

Recommended: RF/XGBoost one-hot categories and unscaled numbers; SVM/MLP one-hot plus
training-fitted numerical scaling; LightGBM native training-derived categories.
Each group must keep the same partitions and report its adapter/interventions.

```python
import json
from pathlib import Path
import pandas as pd
release = Path("nsl-kdd-binary-common-v1")  # adjust to extracted folder
schema = json.loads((release / "schema.json").read_text())
f = pd.read_csv(release / "train_features.csv", float_precision="round_trip")
y = pd.read_csv(release / "train_labels.csv")
assert f.row_id.is_unique and y.row_id.is_unique
assert f.row_id.tolist() == y.row_id.tolist()
joined = f.merge(y, on="row_id", validate="one_to_one", sort=False)
X_train = joined[schema["predictor_order"]]
y_train = joined["target"]
```

Repeat for validation/test. Use validation to choose settings and freeze the model
before final-test scoring. No models have been retrained on this corrected release.
Report class supports, precision/recall/F1, macro-F1, accuracy, balanced accuracy,
Normal false-positive rate and confusion matrices with true rows/predicted columns.

Verify from the project: python -m group3 verify --release PATH --source DATASET
The --source folder is optional and additionally checks exact raw-source values.
manifest.json contains every release artifact checksum. Share the COMPLETE release.
