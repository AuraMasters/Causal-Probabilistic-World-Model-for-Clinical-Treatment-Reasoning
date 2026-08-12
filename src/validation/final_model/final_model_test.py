from pathlib import Path
import json
import warnings

import numpy as np
import pandas as pd
import networkx as nx

from sklearn.metrics import (
    accuracy_score,
    log_loss,
    brier_score_loss,
    roc_auc_score,
)

from pgmpy.models import DiscreteBayesianNetwork
from pgmpy.estimators import BayesianEstimator
from pgmpy.inference import VariableElimination


# ============================================================
# WARNING CONFIGURATION
# ============================================================

warnings.filterwarnings(
    "ignore",
    category=FutureWarning,
)


# ============================================================
# PROJECT PATHS
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
    / "final_model"
    / "dag"
    / "final_dag_edges.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "results"
    / "validation"
    / "final_model"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# VARIABLES
# ============================================================

VARIABLES = [
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
    "label",
]

TARGET = "label"

EQUIVALENT_SAMPLE_SIZE = 10


# ============================================================
# PRINTING
# ============================================================

def header(title):
    print()
    print("=" * 70)
    print(title)
    print("=" * 70)


# Backward-compatible alias.
def print_header(title):
    header(title)


# ============================================================
# LOAD DATA
# ============================================================

def load_data(path):

    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found:\n{path}"
        )

    df = pd.read_csv(path)

    missing_columns = (
        set(VARIABLES)
        - set(df.columns)
    )

    if missing_columns:
        raise ValueError(
            "Dataset is missing required variables:\n"
            f"{sorted(missing_columns)}"
        )

    df = df[VARIABLES].copy()

    # The project uses discrete categorical states.
    for column in VARIABLES:
        df[column] = df[column].astype(str)

    return df


# ============================================================
# LOAD FINAL DAG
# ============================================================

def load_dag():

    if not FINAL_DAG_PATH.exists():
        raise FileNotFoundError(
            f"Final DAG not found:\n"
            f"{FINAL_DAG_PATH}"
        )

    edges_df = pd.read_csv(
        FINAL_DAG_PATH
    )

    required_columns = {
        "source",
        "target",
    }

    if not required_columns.issubset(
        edges_df.columns
    ):
        raise ValueError(
            "Final DAG file must contain "
            "'source' and 'target' columns."
        )

    edges = []

    for _, row in edges_df.iterrows():

        source = str(
            row["source"]
        ).strip()

        target = str(
            row["target"]
        ).strip()

        edges.append(
            (
                source,
                target,
            )
        )

    return edges


# ============================================================
# DAG VALIDATION
# ============================================================

def validate_dag(edges):

    graph = nx.DiGraph()

    graph.add_nodes_from(
        VARIABLES
    )

    graph.add_edges_from(
        edges
    )

    # --------------------------------------------------------
    # DAG
    # --------------------------------------------------------

    if not nx.is_directed_acyclic_graph(
        graph
    ):
        raise ValueError(
            "Final graph is not a DAG."
        )

    print(
        "DAG structure: PASSED"
    )

    # --------------------------------------------------------
    # Forbidden direction
    # --------------------------------------------------------

    if (
        "label",
        "trt",
    ) in edges:

        raise ValueError(
            "Forbidden edge detected: "
            "label -> trt"
        )

    print(
        "Forbidden label -> trt: PASSED"
    )

    # --------------------------------------------------------
    # Required treatment → outcome edge
    # --------------------------------------------------------

    if (
        "trt",
        "label",
    ) not in edges:

        raise ValueError(
            "Required edge missing: "
            "trt -> label"
        )

    print(
        "Treatment -> outcome: PASSED"
    )

    # --------------------------------------------------------
    # Final model must contain 23 edges
    # --------------------------------------------------------

    if len(edges) != 23:

        raise ValueError(
            f"Expected 23 final edges, "
            f"found {len(edges)}."
        )

    print(
        "Final 23-edge structure: PASSED"
    )


# ============================================================
# BUILD BAYESIAN NETWORK
# ============================================================

def build_model(
    development,
    edges,
):

    model = DiscreteBayesianNetwork()

    model.add_nodes_from(
        VARIABLES
    )

    model.add_edges_from(
        edges
    )

    estimator = BayesianEstimator(
        model,
        development,
    )

    # pgmpy 1.1.2:
    #
    # prior_type and equivalent_sample_size
    # are supplied to get_parameters(),
    # NOT model.fit().

    cpds = estimator.get_parameters(
        prior_type="BDeu",
        equivalent_sample_size=(
            EQUIVALENT_SAMPLE_SIZE
        ),
    )

    model.add_cpds(
        *cpds
    )

    if not model.check_model():

        raise ValueError(
            "Bayesian network consistency "
            "check failed."
        )

    return model


# ============================================================
# TEST STATE VALIDATION
# ============================================================

def validate_test_states(
    development,
    test,
):

    print(
        "\nChecking test states against "
        "development states..."
    )

    for column in VARIABLES:

        development_states = set(
            development[column].unique()
        )

        test_states = set(
            test[column].unique()
        )

        unseen_states = (
            test_states
            - development_states
        )

        if unseen_states:

            raise ValueError(
                f"Test contains unseen states "
                f"for '{column}': "
                f"{sorted(unseen_states)}"
            )

    print(
        "All test states represented: PASSED"
    )


# ============================================================
# GENERATE TEST PREDICTIONS
# ============================================================

def generate_predictions(
    model,
    test,
):

    inference = VariableElimination(
        model
    )

    predictions = []

    total = len(test)

    print(
        f"\nPatients to evaluate: "
        f"{total}"
    )

    for index, row in test.iterrows():

        # ----------------------------------------------------
        # Evidence
        # ----------------------------------------------------

        evidence = {}

        for variable in VARIABLES:

            if variable == TARGET:
                continue

            evidence[
                variable
            ] = row[
                variable
            ]

        # ----------------------------------------------------
        # Inference
        # ----------------------------------------------------

        result = inference.query(
            variables=[
                TARGET
            ],
            evidence=evidence,
            show_progress=False,
        )

        target_states = list(
            result.state_names[
                TARGET
            ]
        )

        probability_values = (
            np.asarray(
                result.values,
                dtype=float,
            )
            .reshape(-1)
        )

        # ----------------------------------------------------
        # Positive class
        # ----------------------------------------------------

        if "1" not in target_states:

            raise ValueError(
                "Target state '1' is missing "
                "from inference result."
            )

        positive_index = (
            target_states.index("1")
        )

        probability_positive = float(
            probability_values[
                positive_index
            ]
        )

        # ----------------------------------------------------
        # Actual label
        # ----------------------------------------------------

        actual_label = int(
            row[TARGET]
        )

        predicted_label = int(
            probability_positive >= 0.5
        )

        predictions.append(
            {
                "row_index": int(index),
                "actual_label": actual_label,
                "predicted_probability": (
                    probability_positive
                ),
                "predicted_label": (
                    predicted_label
                ),
                "treatment": row["trt"],
            }
        )

        processed = len(
            predictions
        )

        if (
            processed % 50 == 0
            or processed == total
        ):

            print(
                f"Processed "
                f"{processed}/{total}"
            )

    return pd.DataFrame(
        predictions
    )


# ============================================================
# EXPECTED CALIBRATION ERROR
# ============================================================

def calculate_ece(
    y_true,
    probabilities,
    bins=10,
):

    y_true = np.asarray(
        y_true,
        dtype=float,
    )

    probabilities = np.asarray(
        probabilities,
        dtype=float,
    )

    boundaries = np.linspace(
        0.0,
        1.0,
        bins + 1,
    )

    ece = 0.0

    for i in range(bins):

        lower = boundaries[i]
        upper = boundaries[i + 1]

        if i == bins - 1:

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

        confidence = float(
            probabilities[mask].mean()
        )

        observed_rate = float(
            y_true[mask].mean()
        )

        weight = (
            mask.sum()
            / len(probabilities)
        )

        ece += (
            weight
            * abs(
                observed_rate
                - confidence
            )
        )

    return float(ece)


# ============================================================
# METRICS
# ============================================================

def calculate_metrics(
    predictions
):

    y_true = (
        predictions[
            "actual_label"
        ]
        .astype(int)
        .to_numpy()
    )

    probabilities = (
        predictions[
            "predicted_probability"
        ]
        .astype(float)
        .to_numpy()
    )

    predicted_labels = (
        probabilities >= 0.5
    ).astype(int)

    metrics = {}

    metrics[
        "test_rows"
    ] = int(
        len(predictions)
    )

    metrics[
        "positive_rate"
    ] = float(
        y_true.mean()
    )

    metrics[
        "mean_predicted_probability"
    ] = float(
        probabilities.mean()
    )

    metrics[
        "log_loss"
    ] = float(
        log_loss(
            y_true,
            probabilities,
            labels=[0, 1],
        )
    )

    metrics[
        "brier_score"
    ] = float(
        brier_score_loss(
            y_true,
            probabilities,
        )
    )

    metrics[
        "accuracy"
    ] = float(
        accuracy_score(
            y_true,
            predicted_labels,
        )
    )

    if len(
        np.unique(y_true)
    ) == 2:

        metrics[
            "roc_auc"
        ] = float(
            roc_auc_score(
                y_true,
                probabilities,
            )
        )

    else:

        metrics[
            "roc_auc"
        ] = None

    metrics[
        "ece"
    ] = calculate_ece(
        y_true,
        probabilities,
    )

    return metrics


# ============================================================
# TREATMENT DIAGNOSTIC
# ============================================================

def treatment_diagnostic(
    predictions
):

    rows = []

    treatments = sorted(
        predictions[
            "treatment"
        ].unique()
    )

    for treatment in treatments:

        subset = predictions[
            predictions[
                "treatment"
            ] == treatment
        ]

        observed_probability = float(
            subset[
                "actual_label"
            ]
            .astype(int)
            .mean()
        )

        model_probability = float(
            subset[
                "predicted_probability"
            ]
            .astype(float)
            .mean()
        )

        rows.append(
            {
                "treatment": treatment,
                "test_rows": int(
                    len(subset)
                ),
                "observed_P_label_1": (
                    observed_probability
                ),
                "model_P_label_1": (
                    model_probability
                ),
                "absolute_difference": (
                    abs(
                        observed_probability
                        - model_probability
                    )
                ),
            }
        )

    return pd.DataFrame(
        rows
    )


# ============================================================
# SAVE JSON
# ============================================================

def save_json(
    path,
    data,
):

    with open(
        path,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            data,
            file,
            indent=4,
        )


# ============================================================
# MAIN
# ============================================================

def main():

    # ========================================================
    # HEADER
    # ========================================================

    print_header(
        "ACTG175 PHASE-16"
    )

    print(
        "FINAL 23-EDGE MODEL TEST EVALUATION"
    )

    print(
        "\nIMPORTANT:"
    )

    print(
        "Parameters are learned from "
        "DEVELOPMENT data only."
    )

    print(
        "TEST data is used only for "
        "final evaluation."
    )

    # ========================================================
    # LOAD DEVELOPMENT
    # ========================================================

    development = load_data(
        DEVELOPMENT_PATH
    )

    print(
        f"\nDevelopment dataset:"
        f"\n{DEVELOPMENT_PATH}"
    )

    print(
        f"Development shape: "
        f"{development.shape}"
    )

    # ========================================================
    # LOAD TEST
    # ========================================================

    test = load_data(
        TEST_PATH
    )

    print(
        f"\nTest dataset:"
        f"\n{TEST_PATH}"
    )

    print(
        f"Test shape: "
        f"{test.shape}"
    )

    # ========================================================
    # MISSING VALUES
    # ========================================================

    development_missing = int(
        development.isna()
        .sum()
        .sum()
    )

    test_missing = int(
        test.isna()
        .sum()
        .sum()
    )

    print(
        f"\nDevelopment missing: "
        f"{development_missing}"
    )

    print(
        f"Test missing: "
        f"{test_missing}"
    )

    if development_missing != 0:

        raise ValueError(
            "Development data contains "
            "missing values."
        )

    if test_missing != 0:

        raise ValueError(
            "Test data contains "
            "missing values."
        )

    # ========================================================
    # LOAD FINAL DAG
    # ========================================================

    edges = load_dag()

    print(
        f"\nFinal DAG:"
        f"\n{FINAL_DAG_PATH}"
    )

    print(
        f"Final DAG edges: "
        f"{len(edges)}"
    )

    print(
        "\nFinal DAG:"
    )

    for source, target in sorted(
        edges
    ):

        print(
            f"{source} -> {target}"
        )

    # ========================================================
    # VALIDATE FINAL DAG
    # ========================================================

    print_header(
        "FINAL DAG VALIDATION"
    )

    validate_dag(
        edges
    )

    print(
        "\nFinal DAG validation: PASSED"
    )

    # ========================================================
    # TEST STATE VALIDATION
    # ========================================================

    validate_test_states(
        development,
        test,
    )

    # ========================================================
    # PARAMETER LEARNING
    # ========================================================

    print_header(
        "PARAMETER LEARNING"
    )

    print(
        "Training data: DEVELOPMENT"
    )

    print(
        "Test data: NOT USED"
    )

    print(
        "Prior: BDeu"
    )

    print(
        f"Equivalent sample size: "
        f"{EQUIVALENT_SAMPLE_SIZE}"
    )

    print(
        "\nLearning final model parameters..."
    )

    model = build_model(
        development,
        edges,
    )

    print(
        f"CPDs learned: "
        f"{len(model.get_cpds())}"
    )

    print(
        "Model consistency: PASSED"
    )

    # ========================================================
    # INFERENCE
    # ========================================================

    print_header(
        "FINAL MODEL TEST INFERENCE"
    )

    predictions = generate_predictions(
        model,
        test,
    )

    print(
        "\nProbability generation: PASSED"
    )

    # ========================================================
    # METRICS
    # ========================================================

    print_header(
        "FINAL MODEL TEST METRICS"
    )

    metrics = calculate_metrics(
        predictions
    )

    print(
        f"Test rows: "
        f"{metrics['test_rows']}"
    )

    print(
        f"Positive rate: "
        f"{metrics['positive_rate']:.6f}"
    )

    print(
        f"Mean predicted probability: "
        f"{metrics['mean_predicted_probability']:.6f}"
    )

    print(
        f"Log Loss: "
        f"{metrics['log_loss']:.6f}"
    )

    print(
        f"Brier Score: "
        f"{metrics['brier_score']:.6f}"
    )

    if metrics["roc_auc"] is not None:

        print(
            f"ROC-AUC: "
            f"{metrics['roc_auc']:.6f}"
        )

    else:

        print(
            "ROC-AUC: N/A"
        )

    print(
        f"Accuracy: "
        f"{metrics['accuracy']:.6f}"
    )

    print(
        f"ECE: "
        f"{metrics['ece']:.6f}"
    )

    # ========================================================
    # TREATMENT DIAGNOSTIC
    # ========================================================

    print(
        "\nTreatment diagnostic:"
    )

    diagnostic = treatment_diagnostic(
        predictions
    )

    print(
        diagnostic.to_string(
            index=False
        )
    )

    # ========================================================
    # SAVE PREDICTIONS
    # ========================================================

    predictions_path = (
        OUTPUT_DIR
        / "predictions.csv"
    )

    predictions.to_csv(
        predictions_path,
        index=False,
    )

    # ========================================================
    # SAVE METRICS
    # ========================================================

    metrics_path = (
        OUTPUT_DIR
        / "metrics.json"
    )

    save_json(
        metrics_path,
        metrics,
    )

    # ========================================================
    # SAVE TREATMENT DIAGNOSTIC
    # ========================================================

    diagnostic_path = (
        OUTPUT_DIR
        / "treatment_diagnostic.csv"
    )

    diagnostic.to_csv(
        diagnostic_path,
        index=False,
    )

    # ========================================================
    # METADATA
    # ========================================================

    metadata = {
        "phase": 16,
        "model": (
            "ACTG175 Final 23-Edge "
            "Bayesian Network"
        ),
        "training_dataset": str(
            DEVELOPMENT_PATH
        ),
        "test_dataset": str(
            TEST_PATH
        ),
        "training_rows": int(
            len(development)
        ),
        "test_rows": int(
            len(test)
        ),
        "edge_count": int(
            len(edges)
        ),
        "treatment_edge": (
            "trt -> label"
        ),
        "parameter_learning": (
            "development_only"
        ),
        "test_used_for_parameter_learning": (
            False
        ),
        "prior": "BDeu",
        "equivalent_sample_size": (
            EQUIVALENT_SAMPLE_SIZE
        ),
        "inference_method": (
            "Variable Elimination"
        ),
        "metrics": metrics,
    }

    metadata_path = (
        OUTPUT_DIR
        / "metadata.json"
    )

    save_json(
        metadata_path,
        metadata,
    )

    # ========================================================
    # FINAL RESULT
    # ========================================================

    print_header(
        "PHASE-16 RESULT"
    )

    print(
        "Final 23-edge DAG: PASSED"
    )

    print(
        "Development-only parameter "
        "learning: PASSED"
    )

    print(
        "Test-state validation: PASSED"
    )

    print(
        "Test inference: PASSED"
    )

    print(
        f"\nTest rows evaluated: "
        f"{len(test)}"
    )

    print(
        f"Log Loss: "
        f"{metrics['log_loss']:.4f}"
    )

    print(
        f"Brier Score: "
        f"{metrics['brier_score']:.4f}"
    )

    if metrics["roc_auc"] is not None:

        print(
            f"ROC-AUC: "
            f"{metrics['roc_auc']:.4f}"
        )

    print(
        f"Accuracy: "
        f"{metrics['accuracy']:.4f}"
    )

    print(
        f"ECE: "
        f"{metrics['ece']:.4f}"
    )

    print(
        "\nSaved:"
    )

    print(
        f"{predictions_path}"
    )

    print(
        f"{metrics_path}"
    )

    print(
        f"{diagnostic_path}"
    )

    print(
        f"{metadata_path}"
    )

    print(
        "\nPHASE-16 COMPLETE"
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()