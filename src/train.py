from pathlib import Path
import json

import joblib
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


# ============================================================
# Project configuration
# ============================================================

RANDOM_STATE = 42
DEVELOPMENT_RATIO = 0.80
N_CV_SPLITS = 5

TARGET = "y"

LEAKAGE_FEATURES = [
    "duration",
]

FEATURES = [
    "age",
    "job",
    "marital",
    "education",
    "default",
    "housing",
    "loan",
    "contact",
    "month",
    "day_of_week",
    "campaign",
    "pdays",
    "previous",
    "poutcome",
    "emp.var.rate",
    "cons.price.idx",
    "cons.conf.idx",
    "euribor3m",
    "nr.employed",
]

NUMERIC_FEATURES = [
    "age",
    "campaign",
    "previous",
    "emp.var.rate",
    "cons.price.idx",
    "cons.conf.idx",
    "euribor3m",
    "nr.employed",
]

CATEGORICAL_FEATURES = [
    "job",
    "marital",
    "education",
    "default",
    "housing",
    "loan",
    "contact",
    "month",
    "day_of_week",
    "poutcome",
    "pdays",
]

LOGISTIC_PARAM_GRID = {
    "model__C": [
        0.001,
        0.01,
        0.1,
        1.0,
        10.0,
        100.0,
    ],
    "model__class_weight": [
        None,
        "balanced",
    ],
}


# ============================================================
# Paths
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "bank-additional-full.csv"
)

MODEL_DIR = PROJECT_ROOT / "models"
REPORT_DIR = PROJECT_ROOT / "reports"

MODEL_PATH = (
    MODEL_DIR
    / "bank_subscription_pipeline.joblib"
)

METADATA_PATH = (
    MODEL_DIR
    / "model_metadata.json"
)

METRICS_PATH = (
    REPORT_DIR
    / "final_metrics.json"
)


# ============================================================
# Data loading
# ============================================================

def load_data():
    """Load the original Bank Marketing dataset."""

    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATA_PATH}"
        )

    data = pd.read_csv(
        DATA_PATH,
        sep=";",
    )

    print(
        f"Loaded dataset: "
        f"{data.shape[0]} rows x "
        f"{data.shape[1]} columns"
    )

    return data


# ============================================================
# Validation
# ============================================================

def validate_dataset(data):
    """Check that required columns are available."""

    required_columns = (
        set(FEATURES)
        | set(LEAKAGE_FEATURES)
        | {TARGET}
    )

    missing_columns = (
        required_columns
        - set(data.columns)
    )

    if missing_columns:
        raise ValueError(
            "Missing required columns: "
            f"{sorted(missing_columns)}"
        )

    if TARGET in FEATURES:
        raise ValueError(
            "Target must not appear "
            "inside FEATURES."
        )

    leakage_in_features = (
        set(LEAKAGE_FEATURES)
        .intersection(FEATURES)
    )

    if leakage_in_features:
        raise ValueError(
            "Leakage features found in "
            f"model features: {leakage_in_features}"
        )

    defined_features = set(
        NUMERIC_FEATURES
        + CATEGORICAL_FEATURES
    )

    if defined_features != set(FEATURES):
        raise ValueError(
            "Preprocessing feature groups "
            "do not match FEATURES."
        )

    overlap = (
        set(NUMERIC_FEATURES)
        .intersection(
            CATEGORICAL_FEATURES
        )
    )

    if overlap:
        raise ValueError(
            "Features appear in both "
            f"preprocessing groups: {overlap}"
        )


# ============================================================
# Chronological development / test split
# ============================================================

def split_data(data):
    """
    Reserve the final 20% of chronologically ordered
    observations as the held-out test period.
    """

    split_index = int(
        len(data) * DEVELOPMENT_RATIO
    )

    dev_df = (
        data.iloc[:split_index]
        .copy()
    )

    test_df = (
        data.iloc[split_index:]
        .copy()
    )

    X_dev = dev_df[FEATURES].copy()

    y_dev = (
        dev_df[TARGET]
        .map({
            "no": 0,
            "yes": 1,
        })
    )

    X_test = test_df[FEATURES].copy()

    y_test = (
        test_df[TARGET]
        .map({
            "no": 0,
            "yes": 1,
        })
    )

    if (
        y_dev.isna().any()
        or y_test.isna().any()
    ):
        raise ValueError(
            "Unexpected target values found."
        )

    print(
        f"Development rows: {len(X_dev)}"
    )

    print(
        f"Held-out test rows: {len(X_test)}"
    )

    print(
        "Development positive rate: "
        f"{y_dev.mean():.4f}"
    )

    return (
        X_dev,
        y_dev,
        X_test,
        y_test,
    )


# ============================================================
# Preprocessing
# ============================================================

def build_preprocessor():
    """Create the reproducible preprocessing pipeline."""

    numeric_pipeline = Pipeline(
        steps=[
            (
                "scaler",
                StandardScaler(),
            ),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            (
                "onehot",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=False,
                ),
            ),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "num",
                numeric_pipeline,
                NUMERIC_FEATURES,
            ),
            (
                "cat",
                categorical_pipeline,
                CATEGORICAL_FEATURES,
            ),
        ],
        remainder="drop",
    )

    return preprocessor


# ============================================================
# Model pipeline
# ============================================================

def build_model_pipeline():
    """Create an unfitted preprocessing + model pipeline."""

    preprocessor = build_preprocessor()

    model = LogisticRegression(
        max_iter=2000,
        random_state=RANDOM_STATE,
    )

    pipeline = Pipeline(
        steps=[
            (
                "preprocessor",
                preprocessor,
            ),
            (
                "model",
                model,
            ),
        ]
    )

    return pipeline


# ============================================================
# Hyperparameter tuning
# ============================================================

def tune_model(
    pipeline,
    X_dev,
    y_dev,
):
    """Tune Logistic Regression using time-ordered CV."""

    time_cv = TimeSeriesSplit(
        n_splits=N_CV_SPLITS
    )

    cv_splits = list(
        time_cv.split(X_dev)
    )

    grid_search = GridSearchCV(
        estimator=pipeline,
        param_grid=LOGISTIC_PARAM_GRID,
        scoring="average_precision",
        refit=True,
        cv=cv_splits,
        n_jobs=-1,
        return_train_score=True,
    )

    grid_search.fit(
        X_dev,
        y_dev,
    )

    print(
        "\nBest parameters:"
    )

    print(
        grid_search.best_params_
    )

    print(
        "Best development CV "
        "Average Precision: "
        f"{grid_search.best_score_:.4f}"
    )

    return grid_search


# ============================================================
# Final held-out evaluation
# ============================================================

def evaluate_model(
    fitted_pipeline,
    X_test,
    y_test,
):
    """Evaluate the selected model once on the final test set."""

    probabilities = (
        fitted_pipeline
        .predict_proba(X_test)[:, 1]
    )

    predictions = (
        probabilities >= 0.5
    ).astype(int)

    positive_rate = float(
        y_test.mean()
    )

    average_precision = float(
        average_precision_score(
            y_test,
            probabilities,
        )
    )

    roc_auc = float(
        roc_auc_score(
            y_test,
            probabilities,
        )
    )

    precision = float(
        precision_score(
            y_test,
            predictions,
            zero_division=0,
        )
    )

    recall = float(
        recall_score(
            y_test,
            predictions,
            zero_division=0,
        )
    )

    f1 = float(
        f1_score(
            y_test,
            predictions,
            zero_division=0,
        )
    )

    accuracy = float(
        accuracy_score(
            y_test,
            predictions,
        )
    )

    tn, fp, fn, tp = (
        confusion_matrix(
            y_test,
            predictions,
            labels=[0, 1],
        )
        .ravel()
    )

    metrics = {
        "test_positive_rate":
            positive_rate,

        "average_precision":
            average_precision,

        "no_skill_ap_baseline":
            positive_rate,

        "ap_gain_over_baseline":
            average_precision
            - positive_rate,

        "ap_lift_over_baseline":
            average_precision
            / positive_rate,

        "roc_auc":
            roc_auc,

        "reference_threshold":
            0.5,

        "precision_at_0_5":
            precision,

        "recall_at_0_5":
            recall,

        "f1_at_0_5":
            f1,

        "accuracy_at_0_5":
            accuracy,

        "confusion_matrix": {
            "true_negative":
                int(tn),

            "false_positive":
                int(fp),

            "false_negative":
                int(fn),

            "true_positive":
                int(tp),
        },
    }

    return metrics


# ============================================================
# Save artifacts
# ============================================================

def save_artifacts(
    fitted_pipeline,
    grid_search,
    metrics,
):
    """Save fitted pipeline, metadata, and final metrics."""

    MODEL_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    REPORT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    joblib.dump(
        fitted_pipeline,
        MODEL_PATH,
    )

    fitted_model = (
        fitted_pipeline
        .named_steps["model"]
    )

    metadata = {
        "model_type":
            "LogisticRegression",

        "target":
            TARGET,

        "positive_class":
            1,

        "input_feature_count":
            len(FEATURES),

        "input_features":
            FEATURES,

        "numerical_features":
            NUMERIC_FEATURES,

        "categorical_features":
            CATEGORICAL_FEATURES,

        "excluded_leakage_features":
            LEAKAGE_FEATURES,

        "prediction_timing":
            (
                "Immediately before a "
                "specific marketing contact"
            ),

        "primary_metric":
            "Average Precision",

        "development_cv_mean_average_precision":
            float(
                grid_search.best_score_
            ),

        "best_parameters": {
            "C":
                float(fitted_model.C),

            "class_weight":
                fitted_model.class_weight,
        },

        "custom_decision_threshold_selected":
            False,

        "reference_threshold":
            0.5,
    }

    with open(
        METADATA_PATH,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            metadata,
            file,
            indent=4,
        )

    with open(
        METRICS_PATH,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            metrics,
            file,
            indent=4,
        )

    print(
        "\nSaved pipeline:"
    )

    print(
        MODEL_PATH
    )

    print(
        "\nSaved metadata:"
    )

    print(
        METADATA_PATH
    )

    print(
        "\nSaved final metrics:"
    )

    print(
        METRICS_PATH
    )


# ============================================================
# Display results
# ============================================================

def print_final_results(metrics):
    """Print a concise final evaluation summary."""

    print(
        "\n"
        + "=" * 60
    )

    print(
        "FINAL HELD-OUT TEST RESULTS"
    )

    print(
        "=" * 60
    )

    print(
        "Positive prevalence: "
        f"{metrics['test_positive_rate']:.4f}"
    )

    print(
        "Average Precision: "
        f"{metrics['average_precision']:.4f}"
    )

    print(
        "No-skill AP baseline: "
        f"{metrics['no_skill_ap_baseline']:.4f}"
    )

    print(
        "AP lift over baseline: "
        f"{metrics['ap_lift_over_baseline']:.2f}x"
    )

    print(
        "ROC-AUC: "
        f"{metrics['roc_auc']:.4f}"
    )

    print(
        "\nReference threshold = 0.5"
    )

    print(
        "Precision: "
        f"{metrics['precision_at_0_5']:.4f}"
    )

    print(
        "Recall: "
        f"{metrics['recall_at_0_5']:.4f}"
    )

    print(
        "F1: "
        f"{metrics['f1_at_0_5']:.4f}"
    )

    print(
        "Accuracy: "
        f"{metrics['accuracy_at_0_5']:.4f}"
    )

    print(
        "\nConfusion matrix counts:"
    )

    for name, value in (
        metrics[
            "confusion_matrix"
        ].items()
    ):
        print(
            f"  {name}: {value}"
        )

    print(
        "=" * 60
    )


# ============================================================
# Main workflow
# ============================================================

def main():
    print(
        "=" * 60
    )

    print(
        "Bank Term Deposit "
        "Subscription Prediction"
    )

    print(
        "=" * 60
    )

    data = load_data()

    validate_dataset(
        data
    )

    (
        X_dev,
        y_dev,
        X_test,
        y_test,
    ) = split_data(
        data
    )

    pipeline = (
        build_model_pipeline()
    )

    grid_search = tune_model(
        pipeline,
        X_dev,
        y_dev,
    )

    final_pipeline = (
        grid_search
        .best_estimator_
    )

    metrics = evaluate_model(
        final_pipeline,
        X_test,
        y_test,
    )

    print_final_results(
        metrics
    )

    save_artifacts(
        final_pipeline,
        grid_search,
        metrics,
    )

    print(
        "\nEnd-to-end training "
        "completed successfully."
    )


if __name__ == "__main__":
    main()