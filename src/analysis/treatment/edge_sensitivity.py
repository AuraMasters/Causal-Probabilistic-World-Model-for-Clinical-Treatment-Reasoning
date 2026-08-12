from pathlib import Path
import json

import numpy as np
import pandas as pd

from pgmpy.models import DiscreteBayesianNetwork
from pgmpy.estimators import BayesianEstimator
from pgmpy.inference import VariableElimination

from sklearn.metrics import (
    log_loss,
    brier_score_loss,
    roc_auc_score,
    accuracy_score,
)


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[3]

DEVELOPMENT_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "sparse"
    / "development.csv"
)

TEST_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "sparse"
    / "test.csv"
)

FINAL_DAG_PATH = (
    PROJECT_ROOT
    / "results"
    / "structure_learning"
    / "final"
    / "final_dag_edges.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "results"
    / "analysis"
    / "treatment"
    / "edge_sensitivity"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# CONFIGURATION
# ============================================================

TARGET = "label"
TREATMENT = "trt"

ESS = 10

BASELINE_VARS = [
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

ALL_VARS = BASELINE_VARS + [
    TREATMENT,
    TARGET,
]

FORBIDDEN_EDGE = (
    "label",
    "trt",
)


# ============================================================
# HELPERS
# ============================================================

def load_edges(path):
    df = pd.read_csv(path)

    required = {"source", "target"}

    if not required.issubset(df.columns):
        raise ValueError(
            f"DAG file must contain columns: {required}"
        )

    return [
        (row["source"], row["target"])
        for _, row in df.iterrows()
    ]


def validate_dag(model):
    """
    pgmpy 1.1.2 compatibility:
    use nx.is_directed_acyclic_graph().
    """

    import networkx as nx

    if not nx.is_directed_acyclic_graph(
        model
    ):
        raise ValueError(
            "Model is not a DAG."
        )

    if FORBIDDEN_EDGE in model.edges():
        raise ValueError(
            "Forbidden edge label -> trt detected."
        )


def build_model(edges):
    model = DiscreteBayesianNetwork(
        edges
    )

    # Ensure all variables exist, including
    # isolated treatment variables.
    model.add_nodes_from(
        ALL_VARS
    )

    validate_dag(model)

    return model


def learn_parameters(model, development):
    """
    Learn BDeu parameters using development data only.
    """

    estimator = BayesianEstimator(
        model,
        development,
    )

    model.fit(
        development,
        estimator=BayesianEstimator,
        prior_type="BDeu",
        equivalent_sample_size=ESS,
    )

    return model


def check_model(model):
    """
    pgmpy 1.1.2 compatibility.
    """

    try:
        result = model.check_model()
    except Exception as exc:
        raise ValueError(
            f"Model consistency check failed: {exc}"
        )

    if result is not True:
        raise ValueError(
            "Model consistency check returned False."
        )


def probability_label_1(
    inference,
    evidence,
):
    result = inference.query(
        variables=[TARGET],
        evidence=evidence,
        show_progress=False,
    )

    states = list(
        result.state_names[TARGET]
    )

    if 1 in states:
        index = states.index(1)
    elif "1" in states:
        index = states.index("1")
    else:
        raise ValueError(
            f"Could not find label=1 in states: {states}"
        )

    return float(
        result.values[index]
    )


def calculate_ece(
    y_true,
    probabilities,
    bins=10,
):
    """
    Expected Calibration Error.
    """

    y_true = np.asarray(y_true)
    probabilities = np.asarray(
        probabilities
    )

    ece = 0.0

    edges = np.linspace(
        0.0,
        1.0,
        bins + 1,
    )

    for i in range(bins):

        if i == bins - 1:
            mask = (
                (probabilities >= edges[i])
                & (probabilities <= edges[i + 1])
            )
        else:
            mask = (
                (probabilities >= edges[i])
                & (probabilities < edges[i + 1])
            )

        if not np.any(mask):
            continue

        confidence = probabilities[
            mask
        ].mean()

        accuracy = y_true[
            mask
        ].mean()

        ece += (
            mask.mean()
            * abs(
                confidence - accuracy
            )
        )

    return float(ece)


def count_cpt_entries(model):
    total = 0

    for cpd in model.get_cpds():

        values = np.asarray(
            cpd.values
        )

        total += values.size

    return int(total)


def evaluate_model(
    model,
    test,
    model_name,
):
    """
    Evaluate the Bayesian network on the
    untouched test set.
    """

    inference = VariableElimination(
        model
    )

    probabilities = []

    print(
        f"\nEvaluating {model_name}..."
    )

    for index, row in test.iterrows():

        evidence = {}

        for variable in ALL_VARS:

            if variable == TARGET:
                continue

            evidence[variable] = row[
                variable
            ]

        probability = probability_label_1(
            inference,
            evidence,
        )

        probabilities.append(
            probability
        )

        if (
            (index + 1) % 50 == 0
            or index + 1 == len(test)
        ):
            print(
                f"Processed {index + 1}/{len(test)}"
            )

    probabilities = np.asarray(
        probabilities
    )

    y = test[TARGET].to_numpy()

    predictions = (
        probabilities >= 0.5
    ).astype(int)

    metrics = {
        "model": model_name,
        "rows": int(len(test)),
        "positive_rate": float(
            y.mean()
        ),
        "mean_predicted_probability": float(
            probabilities.mean()
        ),
        "log_loss": float(
            log_loss(
                y,
                probabilities,
            )
        ),
        "brier_score": float(
            brier_score_loss(
                y,
                probabilities,
            )
        ),
        "roc_auc": float(
            roc_auc_score(
                y,
                probabilities,
            )
        ),
        "accuracy": float(
            accuracy_score(
                y,
                predictions,
            )
        ),
        "ece": calculate_ece(
            y,
            probabilities,
        ),
        "cpt_entries": count_cpt_entries(
            model
        ),
        "edges": int(
            len(model.edges())
        ),
    }

    prediction_df = pd.DataFrame(
        {
            "observed_label": y,
            "predicted_probability": probabilities,
            "predicted_label": predictions,
        }
    )

    prediction_path = (
        OUTPUT_DIR
        / f"{model_name}_predictions.csv"
    )

    prediction_df.to_csv(
        prediction_path,
        index=False,
    )

    return metrics, prediction_df


def treatment_diagnostic(
    model,
    development,
    model_name,
):
    """
    Calculate model predictions under
    each possible treatment while keeping
    each patient's other evidence fixed.
    """

    inference = VariableElimination(
        model
    )

    rows = []

    for treatment in sorted(
        development[TREATMENT].unique()
    ):

        probabilities = []

        for _, row in development.iterrows():

            evidence = {}

            for variable in ALL_VARS:

                if variable in (
                    TARGET,
                    TREATMENT,
                ):
                    continue

                evidence[variable] = row[
                    variable
                ]

            evidence[TREATMENT] = treatment

            probability = probability_label_1(
                inference,
                evidence,
            )

            probabilities.append(
                probability
            )

        probabilities = np.asarray(
            probabilities
        )

        rows.append(
            {
                "model": model_name,
                "treatment": int(
                    treatment
                ),
                "mean_P_label_1": float(
                    probabilities.mean()
                ),
                "min_P_label_1": float(
                    probabilities.min()
                ),
                "max_P_label_1": float(
                    probabilities.max()
                ),
            }
        )

    result = pd.DataFrame(rows)

    return result


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("ACTG175 PHASE-13")
    print("TREATMENT EDGE SENSITIVITY ANALYSIS")
    print("=" * 70)

    print(
        "\nIMPORTANT:"
    )

    print(
        "The existing final DAG is NOT modified."
    )

    print(
        "We compare the current DAG against a"
        " treatment-edge sensitivity DAG."
    )

    # --------------------------------------------------------
    # Load data
    # --------------------------------------------------------

    development = pd.read_csv(
        DEVELOPMENT_PATH
    )

    test = pd.read_csv(
        TEST_PATH
    )

    print(
        f"\nDevelopment shape: {development.shape}"
    )

    print(
        f"Test shape: {test.shape}"
    )

    print(
        f"Development missing: "
        f"{development.isna().sum().sum()}"
    )

    print(
        f"Test missing: "
        f"{test.isna().sum().sum()}"
    )

    # --------------------------------------------------------
    # Load final DAG
    # --------------------------------------------------------

    final_edges = load_edges(
        FINAL_DAG_PATH
    )

    print(
        f"\nOriginal final DAG edges: "
        f"{len(final_edges)}"
    )

    # --------------------------------------------------------
    # Model A: current final DAG
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("MODEL A — CURRENT FINAL DAG")
    print("=" * 70)

    current_model = build_model(
        final_edges
    )

    print(
        f"Edges: {len(current_model.edges())}"
    )

    print(
        "DAG validation: PASSED"
    )

    current_model = learn_parameters(
        current_model,
        development,
    )

    check_model(
        current_model
    )

    print(
        "Parameter learning: PASSED"
    )

    current_metrics, current_predictions = (
        evaluate_model(
            current_model,
            test,
            "current_dag",
        )
    )

    # --------------------------------------------------------
    # Model B: add trt -> label
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("MODEL B — CURRENT DAG + TRT -> LABEL")
    print("=" * 70)

    sensitivity_edges = list(
        final_edges
    )

    treatment_edge = (
        TREATMENT,
        TARGET,
    )

    if treatment_edge not in sensitivity_edges:
        sensitivity_edges.append(
            treatment_edge
        )

    sensitivity_model = build_model(
        sensitivity_edges
    )

    print(
        f"Edges: {len(sensitivity_model.edges())}"
    )

    print(
        "Added edge: trt -> label"
    )

    print(
        "DAG validation: PASSED"
    )

    sensitivity_model = learn_parameters(
        sensitivity_model,
        development,
    )

    check_model(
        sensitivity_model
    )

    print(
        "Parameter learning: PASSED"
    )

    sensitivity_metrics, sensitivity_predictions = (
        evaluate_model(
            sensitivity_model,
            test,
            "treatment_edge_dag",
        )
    )

    # --------------------------------------------------------
    # Compare metrics
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("TEST-SET COMPARISON")
    print("=" * 70)

    comparison = pd.DataFrame(
        [
            current_metrics,
            sensitivity_metrics,
        ]
    )

    print(
        comparison.round(6).to_string(
            index=False
        )
    )

    # --------------------------------------------------------
    # Calculate changes
    # --------------------------------------------------------

    current = current_metrics
    treatment = sensitivity_metrics

    changes = {
        "log_loss_change": (
            treatment["log_loss"]
            - current["log_loss"]
        ),
        "brier_change": (
            treatment["brier_score"]
            - current["brier_score"]
        ),
        "roc_auc_change": (
            treatment["roc_auc"]
            - current["roc_auc"]
        ),
        "accuracy_change": (
            treatment["accuracy"]
            - current["accuracy"]
        ),
        "ece_change": (
            treatment["ece"]
            - current["ece"]
        ),
        "cpt_entry_change": (
            treatment["cpt_entries"]
            - current["cpt_entries"]
        ),
        "edge_count_change": (
            treatment["edges"]
            - current["edges"]
        ),
    }

    print(
        "\nMetric changes "
        "(Treatment-edge model - Current model):"
    )

    for key, value in changes.items():

        print(
            f"{key}: {value:+.6f}"
        )

    # --------------------------------------------------------
    # Treatment diagnostic
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("TREATMENT-SPECIFIC DIAGNOSTIC")
    print("=" * 70)

    current_treatment = treatment_diagnostic(
        current_model,
        development,
        "current_dag",
    )

    sensitivity_treatment = treatment_diagnostic(
        sensitivity_model,
        development,
        "treatment_edge_dag",
    )

    treatment_diagnostic_df = pd.concat(
        [
            current_treatment,
            sensitivity_treatment,
        ],
        ignore_index=True,
    )

    print(
        treatment_diagnostic_df.round(6)
        .to_string(index=False)
    )

    treatment_diagnostic_path = (
        OUTPUT_DIR
        / "treatment_diagnostic.csv"
    )

    treatment_diagnostic_df.to_csv(
        treatment_diagnostic_path,
        index=False,
    )

    # --------------------------------------------------------
    # Decision logic
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("MODEL DECISION")
    print("=" * 70)

    improved_logloss = (
        changes["log_loss_change"] < 0
    )

    improved_brier = (
        changes["brier_change"] < 0
    )

    improved_auc = (
        changes["roc_auc_change"] > 0
    )

    improved_ece = (
        changes["ece_change"] < 0
    )

    increased_complexity = (
        changes["cpt_entry_change"] > 0
    )

    improvements = sum(
        [
            improved_logloss,
            improved_brier,
            improved_auc,
            improved_ece,
        ]
    )

    if improvements >= 3:

        recommendation = (
            "TREATMENT_EDGE_SUPPORTED_BY_TEST_PREDICTION"
        )

    elif improvements >= 2:

        recommendation = (
            "TREATMENT_EDGE_REQUIRES_FURTHER_REVIEW"
        )

    else:

        recommendation = (
            "KEEP_CURRENT_DAG"
        )

    print(
        f"\nLog Loss improved: "
        f"{improved_logloss}"
    )

    print(
        f"Brier improved: "
        f"{improved_brier}"
    )

    print(
        f"ROC-AUC improved: "
        f"{improved_auc}"
    )

    print(
        f"ECE improved: "
        f"{improved_ece}"
    )

    print(
        f"CPT complexity increased: "
        f"{increased_complexity}"
    )

    print(
        f"\nRecommendation:"
    )

    print(
        recommendation
    )

    print(
        "\nIMPORTANT:"
    )

    print(
        "This recommendation is a sensitivity-analysis "
        "result, not a causal proof."
    )

    # --------------------------------------------------------
    # Save results
    # --------------------------------------------------------

    comparison_path = (
        OUTPUT_DIR
        / "model_comparison.csv"
    )

    comparison.to_csv(
        comparison_path,
        index=False,
    )

    changes_path = (
        OUTPUT_DIR
        / "metric_changes.json"
    )

    with open(
        changes_path,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            changes,
            f,
            indent=2,
        )

    summary = {
        "phase": 13,
        "development_rows": int(
            len(development)
        ),
        "test_rows": int(
            len(test)
        ),
        "current_dag_edges": int(
            len(current_model.edges())
        ),
        "treatment_edge_dag_edges": int(
            len(sensitivity_model.edges())
        ),
        "current_dag_cpt_entries": int(
            current_metrics["cpt_entries"]
        ),
        "treatment_edge_dag_cpt_entries": int(
            sensitivity_metrics["cpt_entries"]
        ),
        "metrics": comparison.to_dict(
            orient="records"
        ),
        "changes": changes,
        "recommendation": recommendation,
        "test_set_used_only_for_evaluation": True,
        "final_dag_modified": False,
    }

    summary_path = (
        OUTPUT_DIR
        / "sensitivity_summary.json"
    )

    with open(
        summary_path,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            summary,
            f,
            indent=2,
        )

    print("\n" + "=" * 70)
    print("PHASE-13 COMPLETE")
    print("=" * 70)

    print(
        "\nSaved:"
    )

    print(
        comparison_path
    )

    print(
        changes_path
    )

    print(
        treatment_diagnostic_path
    )

    print(
        summary_path
    )

    print(
        "\nOriginal final DAG was NOT modified."
    )


if __name__ == "__main__":
    main()