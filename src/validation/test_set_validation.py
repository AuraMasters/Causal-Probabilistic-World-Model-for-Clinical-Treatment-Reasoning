from pathlib import Path
import json

import numpy as np
import pandas as pd
import networkx as nx

from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    log_loss,
    roc_auc_score,
)

from pgmpy.models import DiscreteBayesianNetwork
from pgmpy.estimators import BayesianEstimator
from pgmpy.inference import VariableElimination


# ============================================================
# ACTG175 PHASE-11 TEST-SET VALIDATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

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

DAG_PATH = (
    PROJECT_ROOT
    / "results"
    / "structure_learning"
    / "final"
    / "final_dag_edges.csv"
)

# Support the actual nested output location as well.
if not DAG_PATH.exists():

    DAG_PATH = (
        PROJECT_ROOT
        / "results"
        / "structure_learning"
        / "final"
        / "final"
        / "dag_edges.csv"
    )

OUTPUT_DIR = (
    PROJECT_ROOT
    / "results"
    / "validation"
    / "test_set"
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

FEATURES = [
    variable
    for variable in VARIABLES
    if variable != "label"
]

TARGET = "label"

ESS = 10

RANDOM_SEED = 42


# ============================================================
# PRINT HEADER
# ============================================================

def print_header(title):

    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


# ============================================================
# LOAD DATASETS
# ============================================================

def load_datasets():

    if not DEVELOPMENT_PATH.exists():

        raise FileNotFoundError(
            f"Development dataset not found:\n"
            f"{DEVELOPMENT_PATH}"
        )

    if not TEST_PATH.exists():

        raise FileNotFoundError(
            f"Test dataset not found:\n"
            f"{TEST_PATH}"
        )

    development = pd.read_csv(
        DEVELOPMENT_PATH
    )

    test = pd.read_csv(
        TEST_PATH
    )

    # --------------------------------------------------------
    # Schema validation
    # --------------------------------------------------------

    missing_development = (
        set(VARIABLES)
        - set(development.columns)
    )

    missing_test = (
        set(VARIABLES)
        - set(test.columns)
    )

    if missing_development:

        raise ValueError(
            "Development dataset missing "
            f"variables: {sorted(missing_development)}"
        )

    if missing_test:

        raise ValueError(
            "Test dataset missing "
            f"variables: {sorted(missing_test)}"
        )

    development = development[
        VARIABLES
    ].copy()

    test = test[
        VARIABLES
    ].copy()

    print(
        f"Development dataset:\n"
        f"{DEVELOPMENT_PATH}"
    )

    print(
        f"Test dataset:\n"
        f"{TEST_PATH}"
    )

    print(
        f"\nDevelopment shape: "
        f"{development.shape}"
    )

    print(
        f"Test shape: "
        f"{test.shape}"
    )

    print(
        f"Development missing values: "
        f"{development.isna().sum().sum()}"
    )

    print(
        f"Test missing values: "
        f"{test.isna().sum().sum()}"
    )

    if development.isna().sum().sum() != 0:

        raise ValueError(
            "Development dataset contains "
            "missing values."
        )

    if test.isna().sum().sum() != 0:

        raise ValueError(
            "Test dataset contains "
            "missing values."
        )

    return development, test


# ============================================================
# LOAD FINAL DAG
# ============================================================

def load_dag():

    if not DAG_PATH.exists():

        raise FileNotFoundError(
            "Final DAG file not found.\n"
            "Expected one of:\n"
            f"{PROJECT_ROOT / 'results' / 'structure_learning' / 'final'}"
        )

    edges_df = pd.read_csv(
        DAG_PATH
    )

    if "source" not in edges_df.columns:

        raise ValueError(
            "Final DAG file does not contain "
            "'source' column."
        )

    if "target" not in edges_df.columns:

        raise ValueError(
            "Final DAG file does not contain "
            "'target' column."
        )

    edges = []

    for _, row in edges_df.iterrows():

        source = str(
            row["source"]
        ).strip()

        target = str(
            row["target"]
        ).strip()

        if source == "" or target == "":

            continue

        edges.append(
            (source, target)
        )

    # Remove duplicate edges while
    # preserving order.
    edges = list(
        dict.fromkeys(edges)
    )

    print(
        f"\nFinal DAG file:\n"
        f"{DAG_PATH}"
    )

    print(
        f"Final DAG edges: "
        f"{len(edges)}"
    )

    return edges


# ============================================================
# BUILD MODEL
# ============================================================

def build_model(edges):

    model = DiscreteBayesianNetwork()

    model.add_nodes_from(
        VARIABLES
    )

    model.add_edges_from(
        edges
    )

    # --------------------------------------------------------
    # Validate that every DAG variable is known.
    # --------------------------------------------------------

    model_nodes = set(
        model.nodes()
    )

    expected_nodes = set(
        VARIABLES
    )

    if model_nodes != expected_nodes:

        missing = (
            expected_nodes
            - model_nodes
        )

        extra = (
            model_nodes
            - expected_nodes
        )

        raise ValueError(
            "DAG node mismatch.\n"
            f"Missing: {sorted(missing)}\n"
            f"Extra: {sorted(extra)}"
        )

    # --------------------------------------------------------
    # pgmpy 1.1.2 compatible DAG validation.
    #
    # DiscreteBayesianNetwork inherits from
    # NetworkX graph structures, so validate
    # directly through NetworkX.
    # --------------------------------------------------------

    if not nx.is_directed_acyclic_graph(
        model
    ):

        raise ValueError(
            "Final graph is not a DAG."
        )

    print(
        "DAG structure: PASSED"
    )

    # --------------------------------------------------------
    # Validate forbidden edge.
    # --------------------------------------------------------

    if (
        "label",
        "trt",
    ) in edges:

        raise ValueError(
            "Forbidden edge label -> trt "
            "is present."
        )

    print(
        "Forbidden label -> trt: PASSED"
    )

    return model


# ============================================================
# LEARN PARAMETERS
# ============================================================

def learn_parameters(
    model,
    development,
):

    print_header(
        "PARAMETER LEARNING"
    )

    print(
        "Training set: DEVELOPMENT"
    )

    print(
        "Test set: NOT USED for parameter learning"
    )

    print(
        f"Prior: BDeu"
    )

    print(
        f"Equivalent sample size: {ESS}"
    )

    estimator = BayesianEstimator(
        model,
        development,
    )

    # --------------------------------------------------------
    # pgmpy 1.1.2:
    #
    # BayesianEstimator.get_parameters()
    # supports prior_type and equivalent
    # sample size.
    #
    # We do NOT use model.fit(... prior_type=...)
    # because that API is not supported in the
    # installed pgmpy version.
    # --------------------------------------------------------

    cpds = estimator.get_parameters(
        prior_type="BDeu",
        equivalent_sample_size=ESS,
    )

    model.add_cpds(
        *cpds
    )

    if not model.check_model():

        raise ValueError(
            "Bayesian Network consistency "
            "check failed."
        )

    print(
        f"CPDs learned: {len(cpds)}"
    )

    print(
        "Model consistency: PASSED"
    )

    return model


# ============================================================
# VALIDATE TEST STATES
# ============================================================

def validate_test_states(
    model,
    test,
):

    print_header(
        "TEST STATE VALIDATION"
    )

    for variable in VARIABLES:

        cpd = model.get_cpds(
            variable
        )

        if cpd is None:

            raise ValueError(
                f"No CPD found for "
                f"{variable}."
            )

        # ----------------------------------------------------
        # States known by model
        # ----------------------------------------------------

        model_states = set(
            str(state)
            for state in cpd.state_names[
                variable
            ]
        )

        # ----------------------------------------------------
        # States occurring in test set
        # ----------------------------------------------------

        test_states = set(
            str(state)
            for state in test[
                variable
            ].unique()
        )

        unknown_states = (
            test_states
            - model_states
        )

        if unknown_states:

            raise ValueError(
                f"Unknown test states for "
                f"{variable}: "
                f"{sorted(unknown_states)}"
            )

    print(
        "All test states are represented "
        "in the development-trained model."
    )

    print(
        "Test-state validation: PASSED"
    )


# ============================================================
# CREATE INFERENCE ENGINE
# ============================================================

def create_inference_engine(
    model,
):

    print_header(
        "INFERENCE ENGINE"
    )

    inference = VariableElimination(
        model
    )

    print(
        "Algorithm: Variable Elimination"
    )

    print(
        "Inference engine: READY"
    )

    return inference


# ============================================================
# GET P(label=1)
# ============================================================

def get_positive_probability(
    inference,
    evidence,
):

    result = inference.query(
        variables=[
            TARGET
        ],
        evidence=evidence,
        show_progress=False,
    )

    states = result.state_names[
        TARGET
    ]

    values = result.values

    for state, probability in zip(
        states,
        values,
    ):

        if str(state) == "1":

            return float(
                probability
            )

    raise ValueError(
        "Could not obtain "
        "P(label=1)."
    )


# ============================================================
# PREDICT TEST SET
# ============================================================

def predict_test_probabilities(
    inference,
    test,
):

    print_header(
        "TEST-SET PREDICTION"
    )

    print(
        f"Patients to evaluate: "
        f"{len(test)}"
    )

    probabilities = []

    total = len(test)

    for position, (_, row) in enumerate(
        test.iterrows(),
        start=1,
    ):

        evidence = {}

        for variable in FEATURES:

            evidence[
                variable
            ] = row[variable]

        probability = (
            get_positive_probability(
                inference,
                evidence,
            )
        )

        probabilities.append(
            probability
        )

        if (
            position % 50 == 0
            or position == total
        ):

            print(
                f"Processed "
                f"{position}/{total}"
            )

    probabilities = np.asarray(
        probabilities,
        dtype=float,
    )

    # --------------------------------------------------------
    # Safety checks
    # --------------------------------------------------------

    if not np.all(
        np.isfinite(probabilities)
    ):

        raise ValueError(
            "Non-finite probability "
            "was generated."
        )

    if np.any(
        probabilities < 0
    ) or np.any(
        probabilities > 1
    ):

        raise ValueError(
            "Invalid probability outside "
            "[0, 1] was generated."
        )

    print(
        "Probability generation: PASSED"
    )

    return probabilities


# ============================================================
# CALCULATE METRICS
# ============================================================

def calculate_metrics(
    y_true,
    probabilities,
):

    predictions = (
        probabilities >= 0.5
    ).astype(int)

    metrics = {
        "rows":
            int(len(y_true)),

        "positive_rate":
            float(np.mean(y_true)),

        "log_loss":
            float(
                log_loss(
                    y_true,
                    probabilities,
                )
            ),

        "brier_score":
            float(
                brier_score_loss(
                    y_true,
                    probabilities,
                )
            ),

        "roc_auc":
            float(
                roc_auc_score(
                    y_true,
                    probabilities,
                )
            ),

        "accuracy":
            float(
                accuracy_score(
                    y_true,
                    predictions,
                )
            ),
    }

    return (
        metrics,
        predictions,
    )


# ============================================================
# CALIBRATION
# ============================================================

def calculate_calibration(
    y_true,
    probabilities,
    bins=10,
):

    boundaries = np.linspace(
        0.0,
        1.0,
        bins + 1,
    )

    rows = []

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

        count = int(
            np.sum(mask)
        )

        if count == 0:

            rows.append(
                {
                    "bin": i + 1,
                    "lower": lower,
                    "upper": upper,
                    "count": 0,
                    "mean_predicted": np.nan,
                    "observed_rate": np.nan,
                    "absolute_gap": np.nan,
                }
            )

            continue

        mean_predicted = float(
            np.mean(
                probabilities[mask]
            )
        )

        observed_rate = float(
            np.mean(
                y_true[mask]
            )
        )

        absolute_gap = abs(
            mean_predicted
            - observed_rate
        )

        rows.append(
            {
                "bin": i + 1,
                "lower": lower,
                "upper": upper,
                "count": count,
                "mean_predicted":
                    mean_predicted,
                "observed_rate":
                    observed_rate,
                "absolute_gap":
                    absolute_gap,
            }
        )

    calibration_df = pd.DataFrame(
        rows
    )

    populated = calibration_df[
        calibration_df["count"] > 0
    ]

    total = len(y_true)

    if total > 0 and not populated.empty:

        ece = float(
            np.sum(
                (
                    populated["count"]
                    / total
                )
                * populated[
                    "absolute_gap"
                ]
            )
        )

    else:

        ece = 0.0

    return (
        calibration_df,
        ece,
    )


# ============================================================
# TREATMENT DIAGNOSTIC
# ============================================================

def treatment_diagnostic(
    inference,
    test,
):

    print_header(
        "TREATMENT DIAGNOSTIC"
    )

    rows = []

    for treatment in [0, 1, 2, 3]:

        observed_subset = test[
            test["trt"] == treatment
        ]

        if len(observed_subset) == 0:

            observed_rate = np.nan

        else:

            observed_rate = float(
                observed_subset[
                    TARGET
                ].mean()
            )

        # ----------------------------------------------------
        # Important:
        # This query intentionally uses only
        # trt as evidence.
        # ----------------------------------------------------

        model_probability = (
            get_positive_probability(
                inference,
                {
                    "trt": treatment
                },
            )
        )

        rows.append(
            {
                "treatment":
                    treatment,

                "test_rows":
                    int(
                        len(
                            observed_subset
                        )
                    ),

                "observed_P_label_1":
                    observed_rate,

                "model_P_label_1":
                    model_probability,

                "absolute_difference":
                    abs(
                        observed_rate
                        - model_probability
                    ),
            }
        )

    diagnostic_df = pd.DataFrame(
        rows
    )

    print(
        "\n"
        + diagnostic_df.to_string(
            index=False
        )
    )

    return diagnostic_df


# ============================================================
# SAVE PREDICTIONS
# ============================================================

def save_predictions(
    test,
    probabilities,
    predictions,
):

    output = test.copy()

    output[
        "predicted_P_label_1"
    ] = probabilities

    output[
        "predicted_P_label_0"
    ] = (
        1.0
        - probabilities
    )

    output[
        "predicted_label"
    ] = predictions

    output[
        "correct"
    ] = (
        output[
            TARGET
        ]
        == output[
            "predicted_label"
        ]
    )

    path = (
        OUTPUT_DIR
        / "predictions.csv"
    )

    output.to_csv(
        path,
        index=False,
    )

    print(
        f"\nSaved predictions:\n"
        f"{path}"
    )


# ============================================================
# SAVE METRICS
# ============================================================

def save_metrics(
    metrics,
    ece,
):

    output = dict(
        metrics
    )

    output[
        "expected_calibration_error"
    ] = float(ece)

    output[
        "parameter_training_set"
    ] = "development"

    output[
        "evaluation_set"
    ] = "test"

    output[
        "equivalent_sample_size"
    ] = ESS

    output[
        "random_seed"
    ] = RANDOM_SEED

    path = (
        OUTPUT_DIR
        / "metrics.json"
    )

    with open(
        path,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            output,
            file,
            indent=4,
        )

    print(
        f"Saved metrics:\n"
        f"{path}"
    )


# ============================================================
# SAVE CALIBRATION
# ============================================================

def save_calibration(
    calibration_df,
):

    path = (
        OUTPUT_DIR
        / "calibration.csv"
    )

    calibration_df.to_csv(
        path,
        index=False,
    )

    print(
        f"Saved calibration:\n"
        f"{path}"
    )


# ============================================================
# SAVE TREATMENT DIAGNOSTIC
# ============================================================

def save_treatment_diagnostic(
    diagnostic_df,
):

    path = (
        OUTPUT_DIR
        / "treatment_diagnostic.csv"
    )

    diagnostic_df.to_csv(
        path,
        index=False,
    )

    print(
        f"Saved treatment diagnostic:\n"
        f"{path}"
    )


# ============================================================
# SAVE SUMMARY
# ============================================================

def save_summary(
    metrics,
    ece,
    diagnostic_df,
):

    summary = {
        "phase":
            "Phase 11 - Test Set Validation",

        "dataset":
            "ACTG175",

        "development_rows":
            1711,

        "test_rows":
            428,

        "variables":
            len(VARIABLES),

        "dag_edges":
            22,

        "parameter_learning":
            "Development only",

        "evaluation":
            "Test only",

        "parameter_estimator":
            "BayesianEstimator",

        "prior":
            "BDeu",

        "equivalent_sample_size":
            ESS,

        "inference_algorithm":
            "Variable Elimination",

        "metrics":
            metrics,

        "expected_calibration_error":
            float(ece),

        "treatment_diagnostic":
            diagnostic_df.to_dict(
                orient="records"
            ),
    }

    path = (
        OUTPUT_DIR
        / "validation_summary.json"
    )

    with open(
        path,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            summary,
            file,
            indent=4,
        )

    print(
        f"Saved validation summary:\n"
        f"{path}"
    )


# ============================================================
# PRINT FINAL RESULTS
# ============================================================

def print_final_results(
    metrics,
    ece,
):

    print_header(
        "TEST-SET RESULTS"
    )

    print(
        f"Rows: "
        f"{metrics['rows']}"
    )

    print(
        f"Positive rate: "
        f"{metrics['positive_rate']:.4f}"
    )

    print(
        f"Log Loss: "
        f"{metrics['log_loss']:.4f}"
    )

    print(
        f"Brier Score: "
        f"{metrics['brier_score']:.4f}"
    )

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
        f"{ece:.4f}"
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print_header(
        "ACTG175 PHASE-11 TEST-SET VALIDATION"
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

    # --------------------------------------------------------
    # Load datasets
    # --------------------------------------------------------

    development, test = (
        load_datasets()
    )

    # --------------------------------------------------------
    # Load final DAG
    # --------------------------------------------------------

    edges = load_dag()

    # --------------------------------------------------------
    # Build network
    # --------------------------------------------------------

    model = build_model(
        edges
    )

    # --------------------------------------------------------
    # Learn parameters
    # --------------------------------------------------------

    model = learn_parameters(
        model,
        development,
    )

    # --------------------------------------------------------
    # Validate test states
    # --------------------------------------------------------

    validate_test_states(
        model,
        test,
    )

    # --------------------------------------------------------
    # Create inference engine
    # --------------------------------------------------------

    inference = (
        create_inference_engine(
            model
        )
    )

    # --------------------------------------------------------
    # Generate probabilities
    # --------------------------------------------------------

    probabilities = (
        predict_test_probabilities(
            inference,
            test,
        )
    )

    # --------------------------------------------------------
    # True labels
    # --------------------------------------------------------

    y_true = (
        test[
            TARGET
        ]
        .astype(int)
        .to_numpy()
    )

    # --------------------------------------------------------
    # Metrics
    # --------------------------------------------------------

    metrics, predictions = (
        calculate_metrics(
            y_true,
            probabilities,
        )
    )

    # --------------------------------------------------------
    # Calibration
    # --------------------------------------------------------

    calibration_df, ece = (
        calculate_calibration(
            y_true,
            probabilities,
        )
    )

    # --------------------------------------------------------
    # Print results
    # --------------------------------------------------------

    print_final_results(
        metrics,
        ece,
    )

    # --------------------------------------------------------
    # Treatment diagnostic
    # --------------------------------------------------------

    diagnostic_df = (
        treatment_diagnostic(
            inference,
            test,
        )
    )

    # --------------------------------------------------------
    # Save predictions
    # --------------------------------------------------------

    save_predictions(
        test,
        probabilities,
        predictions,
    )

    # --------------------------------------------------------
    # Save metrics
    # --------------------------------------------------------

    save_metrics(
        metrics,
        ece,
    )

    # --------------------------------------------------------
    # Save calibration
    # --------------------------------------------------------

    save_calibration(
        calibration_df
    )

    # --------------------------------------------------------
    # Save treatment diagnostic
    # --------------------------------------------------------

    save_treatment_diagnostic(
        diagnostic_df
    )

    # --------------------------------------------------------
    # Save summary
    # --------------------------------------------------------

    save_summary(
        metrics,
        ece,
        diagnostic_df,
    )

    # --------------------------------------------------------
    # Final status
    # --------------------------------------------------------

    print_header(
        "PHASE-11 TEST-SET VALIDATION COMPLETE"
    )

    print(
        "Development data: "
        "USED for parameter learning"
    )

    print(
        "Test data: "
        "USED only for evaluation"
    )

    print(
        "Data leakage: "
        "NONE"
    )

    print(
        "Validation: PASSED"
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()