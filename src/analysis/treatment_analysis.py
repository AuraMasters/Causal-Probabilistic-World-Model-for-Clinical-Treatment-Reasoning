from pathlib import Path
import json

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, roc_auc_score, brier_score_loss
from sklearn.model_selection import StratifiedKFold, cross_val_predict


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "sparse"
    / "development.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "results"
    / "analysis"
    / "treatment"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# VARIABLES
# ============================================================

TARGET = "label"
TREATMENT = "trt"

BASELINE_FEATURES = [
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
]

ALL_FEATURES = BASELINE_FEATURES + [TREATMENT]


# ============================================================
# UTILITY FUNCTIONS
# ============================================================

def build_logistic_pipeline(features):
    """
    Build a common logistic-regression pipeline.

    All variables are treated as categorical because the
    Bayesian-network representation uses discrete states.
    """

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "categorical",
                OneHotEncoder(
                    handle_unknown="ignore"
                ),
                features,
            )
        ]
    )

    model = LogisticRegression(
        max_iter=3000,
        random_state=42,
    )

    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model),
        ]
    )


def calculate_bic(y, probabilities, parameter_count):
    """
    Bernoulli-model BIC:

        BIC = -2 log(L) + k log(n)

    where:
        L = likelihood
        k = number of estimated parameters
        n = sample size
    """

    probabilities = np.clip(
        probabilities,
        1e-12,
        1 - 1e-12,
    )

    y = np.asarray(y)

    log_likelihood = np.sum(
        y * np.log(probabilities)
        + (1 - y) * np.log(1 - probabilities)
    )

    n = len(y)

    bic = (
        -2 * log_likelihood
        + parameter_count * np.log(n)
    )

    return float(bic)


def evaluate_model(df, features, name):
    """
    Evaluate a logistic model using 5-fold stratified
    cross-validation on development data only.
    """

    X = df[features]
    y = df[TARGET]

    pipeline = build_logistic_pipeline(features)

    cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=42,
    )

    probabilities = cross_val_predict(
        pipeline,
        X,
        y,
        cv=cv,
        method="predict_proba",
    )[:, 1]

    predictions = (
        probabilities >= 0.5
    ).astype(int)

    # Fit once to determine encoded parameter count.
    pipeline.fit(X, y)

    encoded_features = (
        pipeline
        .named_steps["preprocessor"]
        .get_feature_names_out()
    )

    parameter_count = (
        len(encoded_features) + 1
    )

    result = {
        "model": name,
        "features": len(features),
        "encoded_parameters": parameter_count,
        "log_loss": log_loss(
            y,
            probabilities,
        ),
        "brier_score": brier_score_loss(
            y,
            probabilities,
        ),
        "roc_auc": roc_auc_score(
            y,
            probabilities,
        ),
        "accuracy": np.mean(
            predictions == y
        ),
        "bic": calculate_bic(
            y,
            probabilities,
            parameter_count,
        ),
    }

    return result, probabilities


# ============================================================
# TREATMENT OUTCOME ANALYSIS
# ============================================================

def treatment_outcome_analysis(df):

    print("\n" + "=" * 70)
    print("1. TREATMENT vs OUTCOME")
    print("=" * 70)

    table = pd.crosstab(
        df[TREATMENT],
        df[TARGET],
    )

    print("\nTreatment × Outcome:")
    print(table)

    percentages = pd.crosstab(
        df[TREATMENT],
        df[TARGET],
        normalize="index",
    ) * 100

    print("\nTreatment × Outcome percentages:")
    print(percentages.round(2))

    treatment_rates = (
        df.groupby(TREATMENT)[TARGET]
        .mean()
        .reset_index()
    )

    treatment_rates["positive_rate"] = (
        treatment_rates[TARGET] * 100
    )

    treatment_rates = treatment_rates.drop(
        columns=[TARGET]
    )

    print("\nObserved positive outcome rates:")
    print(
        treatment_rates.to_string(
            index=False
        )
    )

    output_path = (
        OUTPUT_DIR
        / "treatment_outcome_rates.csv"
    )

    treatment_rates.to_csv(
        output_path,
        index=False,
    )

    return table, percentages, treatment_rates


# ============================================================
# TREATMENT EFFECT RELATIVE TO CONTROL
# ============================================================

def treatment_effect_analysis(df):

    print("\n" + "=" * 70)
    print("2. TREATMENT EFFECT RELATIVE TO TRT=0")
    print("=" * 70)

    rates = (
        df.groupby(TREATMENT)[TARGET]
        .mean()
    )

    control_rate = rates.loc[0]

    rows = []

    for treatment in sorted(rates.index):

        rate = rates.loc[treatment]

        absolute_difference = (
            rate - control_rate
        )

        relative_risk = (
            rate / control_rate
            if control_rate > 0
            else np.nan
        )

        rows.append(
            {
                "treatment": int(treatment),
                "positive_rate": float(rate),
                "absolute_difference_vs_trt0": float(
                    absolute_difference
                ),
                "relative_risk_vs_trt0": float(
                    relative_risk
                ),
            }
        )

    result = pd.DataFrame(rows)

    print(result.round(4).to_string(index=False))

    output_path = (
        OUTPUT_DIR
        / "treatment_effects.csv"
    )

    result.to_csv(
        output_path,
        index=False,
    )

    return result


# ============================================================
# PREDICTIVE MODEL COMPARISON
# ============================================================

def predictive_comparison(df):

    print("\n" + "=" * 70)
    print("3. TREATMENT PREDICTIVE VALUE")
    print("=" * 70)

    print(
        "\nModel A: baseline variables WITHOUT treatment"
    )

    baseline_result, baseline_prob = evaluate_model(
        df,
        BASELINE_FEATURES,
        "baseline_without_treatment",
    )

    print(
        pd.Series(baseline_result)
        .to_string()
    )

    print(
        "\nModel B: baseline variables + treatment"
    )

    treatment_result, treatment_prob = evaluate_model(
        df,
        ALL_FEATURES,
        "baseline_plus_treatment",
    )

    print(
        pd.Series(treatment_result)
        .to_string()
    )

    comparison = pd.DataFrame(
        [
            baseline_result,
            treatment_result,
        ]
    )

    print("\nModel comparison:")
    print(
        comparison.round(4).to_string(
            index=False
        )
    )

    output_path = (
        OUTPUT_DIR
        / "treatment_model_comparison.csv"
    )

    comparison.to_csv(
        output_path,
        index=False,
    )

    return (
        comparison,
        baseline_prob,
        treatment_prob,
    )


# ============================================================
# TREATMENT-SPECIFIC PREDICTIONS
# ============================================================

def treatment_specific_predictions(df):

    print("\n" + "=" * 70)
    print("4. TREATMENT-SPECIFIC PREDICTIONS")
    print("=" * 70)

    features = ALL_FEATURES

    X = df[features]
    y = df[TARGET]

    pipeline = build_logistic_pipeline(
        features
    )

    pipeline.fit(X, y)

    rows = []

    for treatment in sorted(
        df[TREATMENT].unique()
    ):

        modified = df[features].copy()

        modified[TREATMENT] = treatment

        probabilities = (
            pipeline.predict_proba(
                modified
            )[:, 1]
        )

        rows.append(
            {
                "treatment": int(treatment),
                "mean_predicted_P_label_1": float(
                    probabilities.mean()
                ),
            }
        )

    result = pd.DataFrame(rows)

    print(
        result.round(4).to_string(
            index=False
        )
    )

    output_path = (
        OUTPUT_DIR
        / "treatment_specific_predictions.csv"
    )

    result.to_csv(
        output_path,
        index=False,
    )

    return result


# ============================================================
# DECISION LOGIC
# ============================================================

def make_recommendation(
    comparison,
    treatment_effects,
):

    print("\n" + "=" * 70)
    print("5. STRUCTURE DECISION")
    print("=" * 70)

    baseline = comparison.iloc[0]
    treatment = comparison.iloc[1]

    log_loss_change = (
        treatment["log_loss"]
        - baseline["log_loss"]
    )

    brier_change = (
        treatment["brier_score"]
        - baseline["brier_score"]
    )

    auc_change = (
        treatment["roc_auc"]
        - baseline["roc_auc"]
    )

    bic_change = (
        treatment["bic"]
        - baseline["bic"]
    )

    print(
        f"\nLog-loss change: {log_loss_change:+.6f}"
    )

    print(
        f"Brier-score change: {brier_change:+.6f}"
    )

    print(
        f"ROC-AUC change: {auc_change:+.6f}"
    )

    print(
        f"BIC change: {bic_change:+.6f}"
    )

    # BIC is lower when the model is preferred.
    treatment_improves_bic = (
        bic_change < 0
    )

    treatment_improves_logloss = (
        log_loss_change < 0
    )

    treatment_improves_brier = (
        brier_change < 0
    )

    treatment_improves_auc = (
        auc_change > 0
    )

    improvements = sum(
        [
            treatment_improves_bic,
            treatment_improves_logloss,
            treatment_improves_brier,
            treatment_improves_auc,
        ]
    )

    if improvements >= 3:
        recommendation = (
            "INVESTIGATE_ADDING_TRT_TO_OUTCOME"
        )
    elif improvements >= 2:
        recommendation = (
            "INVESTIGATE_FURTHER"
        )
    else:
        recommendation = (
            "KEEP_CURRENT_DAG_FOR_NOW"
        )

    print(
        "\nRecommendation:"
    )

    print(
        recommendation
    )

    print(
        "\nIMPORTANT:"
    )

    print(
        "This analysis does NOT automatically modify "
        "the final DAG."
    )

    print(
        "A treatment edge should only be added after "
        "structural and statistical justification."
    )

    summary = {
        "log_loss_change": float(
            log_loss_change
        ),
        "brier_change": float(
            brier_change
        ),
        "roc_auc_change": float(
            auc_change
        ),
        "bic_change": float(
            bic_change
        ),
        "treatment_improves_bic": bool(
            treatment_improves_bic
        ),
        "treatment_improves_logloss": bool(
            treatment_improves_logloss
        ),
        "treatment_improves_brier": bool(
            treatment_improves_brier
        ),
        "treatment_improves_auc": bool(
            treatment_improves_auc
        ),
        "number_of_improvements": int(
            improvements
        ),
        "recommendation": recommendation,
    }

    output_path = (
        OUTPUT_DIR
        / "treatment_analysis_summary.json"
    )

    with open(
        output_path,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            summary,
            f,
            indent=2,
        )

    return summary


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("ACTG175 PHASE-12")
    print("TREATMENT / CAUSAL STRUCTURE ANALYSIS")
    print("=" * 70)

    print(
        f"\nDevelopment dataset:\n{DATA_PATH}"
    )

    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found:\n{DATA_PATH}"
        )

    df = pd.read_csv(
        DATA_PATH
    )

    print(
        f"Shape: {df.shape}"
    )

    required_columns = (
        BASELINE_FEATURES
        + [TREATMENT, TARGET]
    )

    missing_columns = [
        col
        for col in required_columns
        if col not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            "Missing required columns: "
            + str(missing_columns)
        )

    missing_values = (
        df[required_columns]
        .isna()
        .sum()
        .sum()
    )

    print(
        f"Missing values: {missing_values}"
    )

    if missing_values != 0:
        raise ValueError(
            "Missing values detected."
        )

    # --------------------------------------------------------
    # 1. Treatment/outcome association
    # --------------------------------------------------------

    (
        table,
        percentages,
        treatment_rates,
    ) = treatment_outcome_analysis(
        df
    )

    # --------------------------------------------------------
    # 2. Treatment effects
    # --------------------------------------------------------

    treatment_effects = (
        treatment_effect_analysis(
            df
        )
    )

    # --------------------------------------------------------
    # 3. Predictive comparison
    # --------------------------------------------------------

    (
        comparison,
        baseline_prob,
        treatment_prob,
    ) = predictive_comparison(
        df
    )

    # --------------------------------------------------------
    # 4. Treatment-specific predictions
    # --------------------------------------------------------

    treatment_predictions = (
        treatment_specific_predictions(
            df
        )
    )

    # --------------------------------------------------------
    # 5. Recommendation
    # --------------------------------------------------------

    summary = make_recommendation(
        comparison,
        treatment_effects,
    )

    # --------------------------------------------------------
    # Metadata
    # --------------------------------------------------------

    metadata = {
        "phase": 12,
        "dataset": str(DATA_PATH),
        "rows": int(len(df)),
        "development_only": True,
        "test_set_used": False,
        "treatment_variable": TREATMENT,
        "target_variable": TARGET,
        "baseline_features": BASELINE_FEATURES,
        "all_features": ALL_FEATURES,
        "cv_folds": 5,
        "random_state": 42,
        "recommendation": summary[
            "recommendation"
        ],
    }

    metadata_path = (
        OUTPUT_DIR
        / "treatment_analysis_metadata.json"
    )

    with open(
        metadata_path,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            metadata,
            f,
            indent=2,
        )

    print("\n" + "=" * 70)
    print("PHASE-12 COMPLETE")
    print("=" * 70)

    print(
        "\nSaved results:"
    )

    print(
        OUTPUT_DIR
        / "treatment_outcome_rates.csv"
    )

    print(
        OUTPUT_DIR
        / "treatment_effects.csv"
    )

    print(
        OUTPUT_DIR
        / "treatment_model_comparison.csv"
    )

    print(
        OUTPUT_DIR
        / "treatment_specific_predictions.csv"
    )

    print(
        OUTPUT_DIR
        / "treatment_analysis_summary.json"
    )

    print(
        OUTPUT_DIR
        / "treatment_analysis_metadata.json"
    )

    print(
        "\nDo NOT modify the final DAG yet."
    )


if __name__ == "__main__":
    main()