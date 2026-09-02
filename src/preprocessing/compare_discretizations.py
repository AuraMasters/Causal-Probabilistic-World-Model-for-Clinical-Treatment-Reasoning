from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    log_loss,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

# ============================================================
# Paths
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

CANDIDATES = {
    "sparse": PROCESSED_DIR / "sparse",
    "rich": PROCESSED_DIR / "rich",
}

OUTPUT_PATH = (
    PROCESSED_DIR / "discretization_comparison.csv"
)


# ============================================================
# Variables
# ============================================================

TARGET = "label"

FEATURES = [
    "age",
    "wtkg",
    "hemo",
    "homo",
    "drugs",
    "karnof",
    "oprior",
    "z30",
    "preanti",
    "race",
    "gender",
    "strat",
    "symptom",
    "cd40",
    "cd80",
    "trt",
]


# ============================================================
# Configuration
# ============================================================

RANDOM_STATE = 42
N_SPLITS = 5

# Minimum number of observations required for a state
# to be considered adequately supported.
MIN_STATE_SUPPORT = 10


# ============================================================
# Helper: Expected Calibration Error
# ============================================================

def expected_calibration_error(
    y_true: pd.Series,
    probabilities: np.ndarray,
    n_bins: int = 10,
) -> float:
    """
    Calculate Expected Calibration Error (ECE).

    Predictions are divided into probability bins.
    ECE is the weighted average absolute difference between
    confidence and observed frequency.
    """

    y_true = np.asarray(y_true)
    probabilities = np.asarray(probabilities)

    bin_edges = np.linspace(
        0.0,
        1.0,
        n_bins + 1,
    )

    ece = 0.0

    for i in range(n_bins):

        lower = bin_edges[i]
        upper = bin_edges[i + 1]

        if i == n_bins - 1:
            mask = (
                (probabilities >= lower)
                & (probabilities <= upper)
            )
        else:
            mask = (
                (probabilities >= lower)
                & (probabilities < upper)
            )

        if not np.any(mask):
            continue

        bin_probabilities = probabilities[mask]
        bin_labels = y_true[mask]

        mean_confidence = np.mean(
            bin_probabilities
        )

        observed_frequency = np.mean(
            bin_labels
        )

        bin_weight = (
            np.sum(mask) / len(probabilities)
        )

        ece += (
            bin_weight
            * abs(
                mean_confidence
                - observed_frequency
            )
        )

    return float(ece)


# ============================================================
# State-support analysis
# ============================================================

def analyze_state_support(
    df: pd.DataFrame,
) -> dict:
    """
    Analyze the number and support of categorical states.

    This is especially relevant for Bayesian-network CPT
    sparsity and parameter estimation.
    """

    total_states = 0
    rare_states = 0
    minimum_state_count = np.inf

    state_counts = {}

    for column in FEATURES:

        counts = (
            df[column]
            .value_counts(dropna=False)
        )

        state_counts[column] = {
            str(state): int(count)
            for state, count in counts.items()
        }

        total_states += len(counts)

        rare_states += int(
            (counts < MIN_STATE_SUPPORT).sum()
        )

        if len(counts) > 0:
            minimum_state_count = min(
                minimum_state_count,
                int(counts.min()),
            )

    if minimum_state_count == np.inf:
        minimum_state_count = 0

    return {
        "total_states": total_states,
        "rare_states": rare_states,
        "minimum_state_count": int(
            minimum_state_count
        ),
        "state_counts": state_counts,
    }


# ============================================================
# State-space complexity
# ============================================================

def calculate_state_space(
    df: pd.DataFrame,
) -> int:
    """
    Calculate the theoretical number of joint states
    across all modeling variables.

    This is not the number of observed rows. It is the
    product of the cardinalities of all variables.
    """

    state_space = 1

    for column in FEATURES:
        state_count = df[column].nunique(
            dropna=False
        )

        state_space *= state_count

    return int(state_space)


# ============================================================
# Complete discretized-row collapse
# ============================================================

def calculate_collapsed_rows(
    df: pd.DataFrame,
) -> tuple[int, float]:
    """
    Calculate how many complete discretized records
    are repeated.

    These are NOT considered erroneous duplicates.
    They represent patients occupying the same discretized
    state configuration.
    """

    collapsed_rows = int(
        df.duplicated().sum()
    )

    collapsed_percentage = (
        collapsed_rows / len(df) * 100
    )

    return (
        collapsed_rows,
        float(collapsed_percentage),
    )


# ============================================================
# Build predictive pipeline
# ============================================================

def build_pipeline() -> Pipeline:
    """
    Build the preliminary predictive model.

    Logistic Regression is used only as a common reference
    model for comparing Sparse vs Rich representations.
    """

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "categorical",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=True,
                ),
                FEATURES,
            )
        ],
        remainder="drop",
    )

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
# Evaluate one representation
# ============================================================

def evaluate_candidate(
    name: str,
    directory: Path,
) -> dict:

    development_path = (
        directory / "development.csv"
    )

    if not development_path.exists():
        raise FileNotFoundError(
            f"Development dataset not found:\n"
            f"{development_path}"
        )

    df = pd.read_csv(
        development_path
    )

    X = df[FEATURES]
    y = df[TARGET]

    # --------------------------------------------------------
    # State-support analysis
    # --------------------------------------------------------

    support = analyze_state_support(
        df
    )

    state_space = calculate_state_space(
        df
    )

    collapsed_rows, collapsed_percentage = (
        calculate_collapsed_rows(df)
    )

    # --------------------------------------------------------
    # Build model
    # --------------------------------------------------------

    pipeline = build_pipeline()

    cv = StratifiedKFold(
        n_splits=N_SPLITS,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    # --------------------------------------------------------
    # Cross-validated probabilities
    # --------------------------------------------------------

    proba_matrix = np.asarray(
        cross_val_predict(
            pipeline,
            X,
            y,
            cv=cv,
            method="predict_proba",
        )
    )
    probabilities = proba_matrix[:, 1]

    predictions = (
        probabilities >= 0.5
    ).astype(int)

    # --------------------------------------------------------
    # Predictive metrics
    # --------------------------------------------------------

    log_loss_value = log_loss(
        y,
        probabilities,
    )

    brier_value = brier_score_loss(
        y,
        probabilities,
    )

    roc_auc_value = roc_auc_score(
        y,
        probabilities,
    )

    accuracy_value = accuracy_score(
        y,
        predictions,
    )

    ece_value = expected_calibration_error(
        y,
        probabilities,
        n_bins=10,
    )

    # --------------------------------------------------------
    # One-hot feature count
    # --------------------------------------------------------

    fitted_pipeline = build_pipeline()

    fitted_pipeline.fit(
        X,
        y,
    )

    preprocessor = (
        fitted_pipeline
        .named_steps["preprocessor"]
    )

    encoded_feature_count = (
        len(
            preprocessor
            .get_feature_names_out()
        )
    )

    # --------------------------------------------------------
    # Return results
    # --------------------------------------------------------

    return {
        "representation": name,

        # Dataset information
        "rows": len(df),
        "variables": len(FEATURES),

        # State complexity
        "total_states": support[
            "total_states"
        ],
        "rare_states": support[
            "rare_states"
        ],
        "minimum_state_count": support[
            "minimum_state_count"
        ],
        "joint_state_space": state_space,

        # Information collapse
        "collapsed_rows": collapsed_rows,
        "collapsed_percentage": (
            collapsed_percentage
        ),

        # Encoded predictive complexity
        "encoded_feature_count": (
            encoded_feature_count
        ),

        # Predictive performance
        "log_loss": log_loss_value,
        "brier_score": brier_value,
        "roc_auc": roc_auc_value,
        "accuracy": accuracy_value,

        # Calibration
        "ece": ece_value,
    }


# ============================================================
# Comparison interpretation
# ============================================================

def print_interpretation(
    results_df: pd.DataFrame,
) -> None:

    print("\n" + "=" * 70)
    print("PRELIMINARY INTERPRETATION")
    print("=" * 70)

    sparse = results_df[
        results_df["representation"] == "sparse"
    ].iloc[0]

    rich = results_df[
        results_df["representation"] == "rich"
    ].iloc[0]

    # --------------------------------------------------------
    # Log loss
    # --------------------------------------------------------

    print("\nLog loss:")
    print(
        f"  Sparse: {sparse['log_loss']:.4f}"
    )
    print(
        f"  Rich:   {rich['log_loss']:.4f}"
    )

    if sparse["log_loss"] < rich["log_loss"]:
        print("  → Sparse has lower log loss.")
    elif rich["log_loss"] < sparse["log_loss"]:
        print("  → Rich has lower log loss.")
    else:
        print("  → Equal log loss.")

    # --------------------------------------------------------
    # Brier
    # --------------------------------------------------------

    print("\nBrier score:")
    print(
        f"  Sparse: {sparse['brier_score']:.4f}"
    )
    print(
        f"  Rich:   {rich['brier_score']:.4f}"
    )

    if sparse["brier_score"] < rich["brier_score"]:
        print("  → Sparse has better Brier score.")
    elif rich["brier_score"] < sparse["brier_score"]:
        print("  → Rich has better Brier score.")
    else:
        print("  → Equal Brier score.")

    # --------------------------------------------------------
    # ROC-AUC
    # --------------------------------------------------------

    print("\nROC-AUC:")
    print(
        f"  Sparse: {sparse['roc_auc']:.4f}"
    )
    print(
        f"  Rich:   {rich['roc_auc']:.4f}"
    )

    if sparse["roc_auc"] > rich["roc_auc"]:
        print("  → Sparse has higher ROC-AUC.")
    elif rich["roc_auc"] > sparse["roc_auc"]:
        print("  → Rich has higher ROC-AUC.")
    else:
        print("  → Equal ROC-AUC.")

    # --------------------------------------------------------
    # Calibration
    # --------------------------------------------------------

    print("\nExpected Calibration Error:")
    print(
        f"  Sparse: {sparse['ece']:.4f}"
    )
    print(
        f"  Rich:   {rich['ece']:.4f}"
    )

    if sparse["ece"] < rich["ece"]:
        print(
            "  → Sparse has lower calibration error."
        )
    elif rich["ece"] < sparse["ece"]:
        print(
            "  → Rich has lower calibration error."
        )
    else:
        print("  → Equal calibration error.")

    # --------------------------------------------------------
    # Complexity
    # --------------------------------------------------------

    print("\nJoint state-space size:")
    print(
        f"  Sparse: {sparse['joint_state_space']:,}"
    )
    print(
        f"  Rich:   {rich['joint_state_space']:,}"
    )

    print("\nEncoded feature count:")
    print(
        f"  Sparse: {sparse['encoded_feature_count']}"
    )
    print(
        f"  Rich:   {rich['encoded_feature_count']}"
    )

    print("\nCollapsed discretized rows:")
    print(
        f"  Sparse: "
        f"{sparse['collapsed_rows']} "
        f"({sparse['collapsed_percentage']:.2f}%)"
    )
    print(
        f"  Rich:   "
        f"{rich['collapsed_rows']} "
        f"({rich['collapsed_percentage']:.2f}%)"
    )

    print("\nNOTE:")
    print(
        "These results are preliminary predictive evidence."
    )

    print(
        "They do NOT determine the final Bayesian-network "
        "representation."
    )

    print(
        "Final selection will also consider Bayesian-network "
        "structure, CPT sparsity, calibration, and "
        "model complexity."
    )


# ============================================================
# Main
# ============================================================

def main():

    print("=" * 70)
    print(
        "ACTG175 SPARSE vs RICH "
        "PRELIMINARY COMPARISON"
    )
    print("=" * 70)

    print(
        "\nEvaluation setup:"
    )

    print(
        f"  Cross-validation folds: {N_SPLITS}"
    )

    print(
        f"  Random state: {RANDOM_STATE}"
    )

    print(
        "  Evaluation dataset: Development only"
    )

    print(
        "  Test set: NOT USED"
    )

    print(
        "\nLogistic Regression is used only as "
        "a common preliminary reference model."
    )

    results = []

    # --------------------------------------------------------
    # Evaluate candidates
    # --------------------------------------------------------

    for name, directory in CANDIDATES.items():

        print(
            "\n" + "-" * 70
        )

        print(
            f"Evaluating {name.upper()} representation..."
        )

        result = evaluate_candidate(
            name=name,
            directory=directory,
        )

        results.append(
            result
        )

        print(
            f"  Log loss: "
            f"{result['log_loss']:.4f}"
        )

        print(
            f"  Brier: "
            f"{result['brier_score']:.4f}"
        )

        print(
            f"  ROC-AUC: "
            f"{result['roc_auc']:.4f}"
        )

        print(
            f"  Accuracy: "
            f"{result['accuracy']:.4f}"
        )

        print(
            f"  ECE: "
            f"{result['ece']:.4f}"
        )

    # --------------------------------------------------------
    # Create results dataframe
    # --------------------------------------------------------

    results_df = pd.DataFrame(
        results
    )

    print(
        "\n" + "=" * 70
    )

    print(
        "RESULTS"
    )

    print(
        "=" * 70
    )

    display_columns = [
        "representation",
        "rows",
        "variables",
        "total_states",
        "rare_states",
        "minimum_state_count",
        "joint_state_space",
        "collapsed_rows",
        "collapsed_percentage",
        "encoded_feature_count",
        "log_loss",
        "brier_score",
        "roc_auc",
        "accuracy",
        "ece",
    ]

    print(
        results_df[
            display_columns
        ]
        .round(4)
        .to_string(index=False)
    )

    # --------------------------------------------------------
    # Save results
    # --------------------------------------------------------

    PROCESSED_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    results_df.to_csv(
        OUTPUT_PATH,
        index=False,
    )

    print(
        "\nSaved comparison to:"
    )

    print(
        OUTPUT_PATH
    )

    # --------------------------------------------------------
    # Interpretation
    # --------------------------------------------------------

    print_interpretation(
        results_df
    )

    # --------------------------------------------------------
    # Final message
    # --------------------------------------------------------

    print(
        "\n" + "=" * 70
    )

    print(
        "PRELIMINARY COMPARISON COMPLETE"
    )

    print(
        "=" * 70
    )


# ============================================================
# Entry point
# ============================================================

if __name__ == "__main__":
    main()