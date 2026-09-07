# ==========================================
# CELL 16: HEMOGLOBIN FEASIBILITY & CONFOUND AUDIT
# Standalone audit of db.csv + Cell 6 manifest; no model training.
# ==========================================

import json
from pathlib import Path
from typing import Dict, Iterable, Tuple

import numpy as np
import pandas as pd


DATASET_ROOT = Path(r"C:\Users\VICTUS\HemoVision\data\vitalscan-clinic\MCD-rPPG")
DB_CSV_PATH = DATASET_ROOT / "db.csv"
MANIFEST_PATH = DATASET_ROOT / "manifest.json"
RESULTS_PATH = DATASET_ROOT / "hemoglobin_feasibility_audit.json"
TARGET_COLUMN = "hemoglobin"


def _video_key(value: object) -> str:
    return Path(str(value)).stem


def _pearson_or_nan(target: np.ndarray, prediction: np.ndarray) -> float:
    if len(target) < 2 or np.std(target) <= 1e-12 or np.std(prediction) <= 1e-12:
        return float("nan")
    return float(np.corrcoef(target, prediction)[0, 1])


def _mae(target: np.ndarray, prediction: np.ndarray) -> float:
    return float(np.mean(np.abs(target - prediction)))


def _fit_mean_by_group(
    train: pd.DataFrame,
    evaluation: pd.DataFrame,
    group_columns: Iterable[str],
    target_column: str,
) -> np.ndarray:
    """Train-only group-mean predictor with train-global fallback."""
    group_columns = list(group_columns)
    global_mean = float(train[target_column].mean())
    group_mean = train.groupby(group_columns, dropna=False)[target_column].mean()
    predictions = []

    for _, row in evaluation.iterrows():
        key = tuple(row[column] for column in group_columns)
        if len(group_columns) == 1:
            key = key[0]
        predictions.append(float(group_mean.get(key, global_mean)))

    return np.asarray(predictions, dtype=np.float64)


def _design_matrix(
    train: pd.DataFrame,
    evaluation: pd.DataFrame,
) -> Tuple[np.ndarray, np.ndarray]:
    """Build a train-fitted demographic/context design matrix without leakage."""
    numeric_columns = [
        column for column in ("age", "bmi", "weight", "height")
        if column in train.columns
    ]
    categorical_columns = [
        column for column in ("sex", "condition", "camera_type")
        if column in train.columns
    ]

    train_parts = [np.ones((len(train), 1), dtype=np.float64)]
    evaluation_parts = [np.ones((len(evaluation), 1), dtype=np.float64)]

    for column in numeric_columns:
        train_value = pd.to_numeric(train[column], errors="coerce")
        median = float(train_value.median())
        train_parts.append(train_value.fillna(median).to_numpy(dtype=np.float64)[:, None])
        evaluation_value = pd.to_numeric(evaluation[column], errors="coerce")
        evaluation_parts.append(
            evaluation_value.fillna(median).to_numpy(dtype=np.float64)[:, None]
        )

    for column in categorical_columns:
        known_categories = sorted(train[column].fillna("<missing>").astype(str).unique())
        # One category is omitted as a reference column.
        for category in known_categories[1:]:
            train_parts.append(
                (train[column].fillna("<missing>").astype(str) == category)
                .to_numpy(dtype=np.float64)[:, None]
            )
            evaluation_parts.append(
                (evaluation[column].fillna("<missing>").astype(str) == category)
                .to_numpy(dtype=np.float64)[:, None]
            )

    return np.hstack(train_parts), np.hstack(evaluation_parts)


def _fit_demographic_context_baseline(
    train: pd.DataFrame,
    evaluation: pd.DataFrame,
    target_column: str,
) -> np.ndarray:
    """Ordinary least-squares baseline fit on train subjects only."""
    train_x, evaluation_x = _design_matrix(train, evaluation)
    train_y = train[target_column].to_numpy(dtype=np.float64)
    coefficients, *_ = np.linalg.lstsq(train_x, train_y, rcond=None)
    return evaluation_x @ coefficients


def _run_cell16_self_test() -> None:
    """Known-answer test for train-only group means and OLS design construction."""
    train = pd.DataFrame(
        {"condition": ["before", "after"], "age": [20.0, 30.0], "hemoglobin": [10.0, 20.0]}
    )
    evaluation = pd.DataFrame({"condition": ["before", "unknown"], "age": [25.0, 25.0]})
    group_prediction = _fit_mean_by_group(train, evaluation, ["condition"], "hemoglobin")
    assert np.allclose(group_prediction, np.asarray([10.0, 15.0]))
    train_x, evaluation_x = _design_matrix(train, evaluation)
    assert train_x.shape[0] == 2 and evaluation_x.shape[0] == 2


_run_cell16_self_test()


if not DB_CSV_PATH.exists() or not MANIFEST_PATH.exists():
    raise FileNotFoundError("db.csv or manifest.json is missing under DATASET_ROOT.")

db = pd.read_csv(DB_CSV_PATH)
with open(MANIFEST_PATH, "r", encoding="utf-8") as file:
    manifest = pd.DataFrame(json.load(file))

if TARGET_COLUMN not in db.columns:
    raise KeyError(f"db.csv does not contain {TARGET_COLUMN!r}.")

db = db.copy()
manifest = manifest.copy()
db["video_key"] = db["video"].map(_video_key)
manifest["video_key"] = manifest["roi_cache_path"].map(_video_key)

if db["video_key"].duplicated().any():
    raise ValueError("db.csv video keys are not unique.")
if manifest["video_key"].duplicated().any():
    raise ValueError("Manifest video keys are not unique.")

joined = manifest.merge(
    db,
    on="video_key",
    how="left",
    validate="one_to_one",
    suffixes=("_manifest", "_db"),
)

if joined[TARGET_COLUMN].isna().any():
    raise ValueError("Some manifest rows did not join to a valid hemoglobin label.")
if not (joined["subject_id"].astype(str) == joined["patient_id"].astype(str)).all():
    raise ValueError("Manifest/db patient identifiers disagree after join.")

if "camera" in joined.columns:
    joined["camera_type"] = joined["camera"]

splits = {}
for split_name in ("train", "val", "test"):
    split = joined[joined["split"] == split_name].copy()
    if split.empty:
        raise ValueError(f"Manifest split {split_name!r} is empty.")
    splits[split_name] = split

split_subjects = {
    split_name: set(frame["subject_id"].astype(str))
    for split_name, frame in splits.items()
}
for first, second in (("train", "val"), ("train", "test"), ("val", "test")):
    overlap = split_subjects[first] & split_subjects[second]
    if overlap:
        raise AssertionError(f"Subject leakage between {first} and {second}: {sorted(overlap)[:5]}")

train = splits["train"]
test = splits["test"]

label_by_session = joined.groupby(["patient_id", "step"], dropna=False)[TARGET_COLUMN]
session_range = label_by_session.max() - label_by_session.min()

print("=" * 70)
print("HEMOGLOBIN FEASIBILITY & CONFOUND AUDIT")
print("=" * 70)
print(f"Joined recordings: {len(joined)}")
print("Subject-disjoint rows / subjects:")
for split_name, frame in splits.items():
    print(f"  {split_name:<5} {len(frame):4d} / {frame['subject_id'].nunique():3d}")

print("\nHemoglobin distribution by split:")
split_distribution = {}
for split_name, frame in splits.items():
    values = frame[TARGET_COLUMN].to_numpy(dtype=np.float64)
    summary = {
        "n": int(len(values)),
        "mean": float(values.mean()),
        "std": float(values.std(ddof=0)),
        "min": float(values.min()),
        "p05": float(np.quantile(values, 0.05)),
        "median": float(np.median(values)),
        "p95": float(np.quantile(values, 0.95)),
        "max": float(values.max()),
    }
    split_distribution[split_name] = summary
    print(f"  {split_name:<5} mean={summary['mean']:.3f}, std={summary['std']:.3f}, "
          f"range=[{summary['min']:.3f}, {summary['max']:.3f}]")

print("\nCamera-view label repetition within (patient, state):")
print(f"  sessions: {len(session_range)}")
print(f"  exact-repeat fraction: {float((session_range == 0).mean()):.3f}")
print(f"  maximum within-session label range: {float(session_range.max()):.6f}")

print("\nHeld-out non-video baselines for hemoglobin:")
test_target = test[TARGET_COLUMN].to_numpy(dtype=np.float64)
baseline_predictions = {
    "train_mean": np.full(len(test), float(train[TARGET_COLUMN].mean())),
    "state_only": _fit_mean_by_group(train, test, ["condition"], TARGET_COLUMN),
    "camera_only": _fit_mean_by_group(train, test, ["camera_type"], TARGET_COLUMN),
    "demographic_context_ols": _fit_demographic_context_baseline(train, test, TARGET_COLUMN),
}

baseline_results = {}
for name, prediction in baseline_predictions.items():
    result = {"mae": _mae(test_target, prediction), "pearson_r": _pearson_or_nan(test_target, prediction)}
    baseline_results[name] = result
    print(f"  {name:<24} MAE={result['mae']:.4f} | r={result['pearson_r']:+.4f}")

print("\nTest hemoglobin by camera/state (label balance check):")
strata_columns = [column for column in ("camera_type", "condition") if column in test.columns]
strata = (
    test.groupby(strata_columns, dropna=False)[TARGET_COLUMN]
    .agg(["count", "mean", "std"])
    .reset_index()
)
print(strata.to_string(index=False))

skin_tone_available = any(
    "skin" in column.lower() or "fitzpatrick" in column.lower() or "tone" in column.lower()
    for column in joined.columns
)
print(f"\nSkin-tone proxy available in joined manifest/db: {skin_tone_available}")

audit = {
    "target": TARGET_COLUMN,
    "joined_recordings": int(len(joined)),
    "split_distribution": split_distribution,
    "session_label_repetition": {
        "sessions": int(len(session_range)),
        "exact_repeat_fraction": float((session_range == 0).mean()),
        "max_within_session_range": float(session_range.max()),
    },
    "heldout_non_video_baselines": baseline_results,
    "skin_tone_proxy_available": bool(skin_tone_available),
}

with open(RESULTS_PATH, "w", encoding="utf-8") as file:
    json.dump(audit, file, indent=2, allow_nan=True)

print(f"\nAudit saved: {RESULTS_PATH}")
